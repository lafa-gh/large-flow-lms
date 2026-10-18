"""Train a two-time LAFA student from a frozen Hugging Face teacher."""

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

from src.dataset import load_and_tokenize_dataset
from src.lafa.callbacks import EMACallback, TimeWarpCallback
from src.lafa.data_collator import FlowDataCollator
from src.lafa.distillation import distillation_loss
from src.lafa.configuration_lafa import LafaConfig
from src.lafa.modeling_lafa import LafaForCausalLM
from src.utils.config import BaseConfig, parse_config_and_training_args
from src.utils.hub import resolve_checkpoint
from src.utils.optim import MuonWithAdamW
from train_flow import FlowTrainer, _resolve_resume_checkpoint, accelerator


@dataclass
class DistillConfig(BaseConfig):
    distill_teacher_path: str = "lafa-hf/lafa-0.5B"
    revision: str | None = None
    subfolder: str = ""
    dataset_name: str = "allenai/Dolci-Instruct-SFT"
    block_size: int = 512
    diagonal_fraction: float = 0.5
    s0_fraction: float = 0.5
    noise_scale: float = 0.5
    ema_decay: float = 0.95
    muon_lr: float = 3e-5
    muon_momentum: float = 0.95

    def validate(self):
        for name in ("diagonal_fraction", "s0_fraction", "ema_decay"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("noise_scale", "muon_lr"):
            if not 0 < getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.muon_momentum < 1:
            raise ValueError("muon_momentum must be in [0, 1)")


class DistillationModel(LafaForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return distillation_loss(self, input_ids, labels, attention_mask)


class DistillationTrainer(FlowTrainer):
    def create_optimizer(self, model=None):
        if self.optimizer is None:
            matrix, auxiliary = [], []
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    group = (
                        matrix
                        if parameter.ndim == 2
                        and not any(key in name for key in ("embed_tokens", "lm_head"))
                        else auxiliary
                    )
                    group.append(parameter)
            recipe = self.model.config.distillation_config
            groups = [
                dict(
                    params=matrix,
                    use_muon=True,
                    lr=recipe["muon_lr"],
                    momentum=recipe["muon_momentum"],
                    weight_decay=0.0,
                )
            ]
            if auxiliary:
                groups.append(
                    dict(
                        params=auxiliary,
                        use_muon=False,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        eps=self.args.adam_epsilon,
                        weight_decay=self.args.weight_decay,
                    )
                )
            self.optimizer = MuonWithAdamW(groups)
        return self.optimizer


def build_student(recipe, resume=None):
    if resume and recipe.revision is None:
        saved = LafaConfig.from_pretrained(resume).distillation_config
        recipe.revision = saved["revision"]
    source = resolve_checkpoint(
        recipe.distill_teacher_path, revision=recipe.revision, subfolder=recipe.subfolder
    )
    if not Path(recipe.distill_teacher_path).exists():
        snapshot = next(
            path
            for path in (Path(source), *Path(source).parents)
            if path.parent.name == "snapshots"
        )
        recipe.revision = snapshot.name
    config = LafaConfig.from_pretrained(source)
    if config.model_type != "lafa" or not config.backbone_config:
        raise ValueError("Use an exported LAFA teacher (local directory or HF repository)")
    if config.flow_config.get("two_time"):
        raise ValueError("PSD requires a single-time teacher")
    teacher = LafaForCausalLM.from_pretrained(source, dtype=torch.bfloat16).eval()
    teacher.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if resume:
        student = DistillationModel.from_pretrained(resume, dtype=torch.bfloat16)
        if student.config.distillation_config != asdict(recipe):
            raise ValueError("Resume recipe differs from the saved distillation_config")
    else:
        config = copy.deepcopy(teacher.config)
        config.flow_config.update(
            two_time=True,
            distill_two_time_mode="shared_embedding",
            distill_second_time_input="target",
            distill_second_time_bias=True,
            enable_self_cond=False,
            freeze_embed=True,
            freeze_lm_head=True,
            ar_loss=True,
            ar_loss_weight=1.0,
            flow_ce_enable=False,
            time_warp_start_step=0,
            time_warp_init_path=None,
            ema_decay=recipe.ema_decay,
        )
        student, info = DistillationModel.from_pretrained(
            source, config=config, dtype=torch.bfloat16, output_loading_info=True
        )
        if any(not key.startswith("time_embed_tgt.") for key in info["missing_keys"]) or any(
            not key.startswith(("sc_lin_", "sc_proj.", "sc_mlp.", "sc_scale"))
            for key in info["unexpected_keys"]
        ):
            raise ValueError(f"Unexpected teacher/student tensor mismatch: {info}")
        from src.lafa.layers import TimestepEmbedder

        student.time_embed_tgt = TimestepEmbedder(student._d_model).to(torch.bfloat16)
        torch.nn.init.zeros_(student.time_embed_tgt.mlp[2].weight)
        torch.nn.init.zeros_(student.time_embed_tgt.mlp[2].bias)
        student.config.weight_variant = "raw"
        student.config.distillation_config = asdict(recipe)
    student.embed_tokens.weight.requires_grad_(False)
    student.lm_head.weight.requires_grad_(False)
    # Keep the teacher out of student parameters, DDP, optimizer, and exports.
    object.__setattr__(student, "_teacher", teacher)
    student.ar_loss_type = "kl"
    student.ar_loss_auto_balance = False
    student._distill_diagonal_fraction = recipe.diagonal_fraction
    student._distill_s0_fraction = recipe.s0_fraction
    student._distill_s0_noise_scale = recipe.noise_scale
    student._distill_offdiag_noise_scale = recipe.noise_scale
    student._distill_diag_noise_scale = recipe.noise_scale
    return student, tokenizer


def main():
    recipe, args = parse_config_and_training_args(DistillConfig)
    recipe.validate()
    if args.deepspeed:
        raise ValueError("PSD uses replicated frozen teachers and native Muon; launch with DDP")
    if args.save_only_model:
        raise ValueError(
            "PSD checkpoints must retain optimizer state; save_only_model must be false"
        )
    if args.per_device_train_batch_size < 2:
        raise ValueError(
            "PSD requires per_device_train_batch_size >= 2 for diagonal/off-diagonal rows"
        )
    args.ddp_find_unused_parameters = False
    if args.gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    set_seed(args.seed)
    resume = _resolve_resume_checkpoint(args)
    args.resume_from_checkpoint = resume
    model, tokenizer = build_student(recipe, resume)
    if recipe.block_size % model.bd_size:
        raise ValueError("block_size must be divisible by the teacher block size")
    if accelerator.is_main_process:
        recipe.save(str(Path(args.output_dir) / "distillation_config.json"))
    dataset = load_and_tokenize_dataset(
        dataset_name=recipe.dataset_name, tokenizer=tokenizer, block_size=recipe.block_size
    )
    trainer = DistillationTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=FlowDataCollator(
            pad_token_id=tokenizer.pad_token_id, bd_size=model.bd_size, return_attention_mask=True
        ),
        callbacks=[TimeWarpCallback(plot_every=0, hist_every=0), EMACallback()],
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()


if __name__ == "__main__":
    main()
