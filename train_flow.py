#!/usr/bin/env python
# ruff: noqa: E402
"""Train Lafa flow models on instruction data."""

import os
import json

# Set precision before initializing AcceleratorState.
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from src.utils.config import (
    BaseConfig,
    parse_config_and_training_args,
)
from src.utils.checkpoints import latest_checkpoint
from src.utils.distributed import setup_accelerator, unwrap_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    set_seed,
)

accelerator = setup_accelerator()

from src.lafa.flow_model import FlowModelWrapper
from src.lafa.init_checkpoint import load_init_state_dict
from src.lafa.data_collator import FlowDataCollator
from src.lafa.callbacks import (
    EMACallback,
    EmbedNormProjectCallback,
    EpochCheckpointCallback,
    TimeWarpCallback,
)
from src.dataset import load_and_tokenize_dataset


class FlowTrainer(Trainer):
    """Log component losses and support weights-only checkpoint recovery.

    Component losses enter the Trainer log before callbacks run, so they share
    the main logging step. Recovery preserves the Trainer and data position
    while creating a fresh optimizer and scheduler if ZeRO shards are missing.
    """

    resume_weights_only = False

    def _init_loss_acc(self):
        from collections import defaultdict

        self._comp_sum = defaultdict(float)
        self._comp_n = 0

    def training_step(self, *args, **kwargs):
        if not hasattr(self, "_comp_sum"):
            self._init_loss_acc()
        result = super().training_step(*args, **kwargs)
        m = unwrap_model(self.model)
        comps = getattr(m, "_loss_components", None)
        if comps:
            for k, v in comps.items():
                self._comp_sum[k] += float(v)
            self._comp_n += 1
        return result

    def log(self, logs, *args, **kwargs):
        if not hasattr(self, "_comp_sum"):
            self._init_loss_acc()
        if self._comp_n and isinstance(logs, dict):
            # Trainer adds "train/" to these keys.
            for k, total in self._comp_sum.items():
                logs[k] = total / self._comp_n
            self._comp_sum.clear()
            self._comp_n = 0
        return super().log(logs, *args, **kwargs)

    def _prepare_for_training(
        self, max_steps, train_dataloader, resume_from_checkpoint
    ):
        if self.resume_weights_only and resume_from_checkpoint is not None:
            accelerator.print(
                "[Flow] Weights-only resume: preserving Trainer/data position from "
                f"{resume_from_checkpoint}, but creating a fresh optimizer and scheduler."
            )
            return super()._prepare_for_training(
                max_steps, train_dataloader, resume_from_checkpoint=None
            )
        return super()._prepare_for_training(
            max_steps, train_dataloader, resume_from_checkpoint
        )

    def _load_rng_state(self, checkpoint):
        if self.resume_weights_only and checkpoint is not None:
            if not getattr(self, "_weights_only_rng_notice_printed", False):
                accelerator.print(
                    "[Flow] Weights-only resume: rank RNG state is not restored "
                    "because the multi-node checkpoint is incomplete."
                )
                self._weights_only_rng_notice_printed = True
            return
        return super()._load_rng_state(checkpoint)


@dataclass
class FlowConfig(BaseConfig):
    """Flow training configuration."""

    # Model and data
    model_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})
    dataset_name: str = field(metadata={"help": "Hugging Face instruction dataset ID"})
    bd_size: int = field(
        default=8, metadata={"help": "Block size (attention mask granularity)"}
    )
    block_size: int = field(default=512, metadata={"help": "Training sequence length"})
    revision: Optional[str] = field(default=None)
    subfolder: str = field(default='')
    reset_time_warper: bool = field(default=False)


    # Attention
    block_causal: bool = field(
        default=False,
        metadata={"help": "Use block-causal (vs context-causal) attention"},
    )
    noisy_causal: bool = field(
        default=False,
        metadata={"help": "Causal (not bidirectional) attention within noisy blocks"},
    )

    # Noise and time sampling
    noise_scale_mult: float = field(
        default=4.0, metadata={"help": "Noise scaling: σ_embed × noise_scale_mult"}
    )
    time_warp_num_bins: int = field(
        default=200,
        metadata={"help": "Number of flow-time bins in the EMA loss histogram"},
    )
    time_warp_per_position: bool = field(
        default=True,
        metadata={
            "help": "Track an EMA loss histogram per block position and sample with one shared quantile per block. False uses one shared histogram."
        },
    )
    time_warp_start_step: int = field(
        default=500,
        metadata={"help": "Warper: uniform-t warm-up until this global step"},
    )
    warp_uniform_est: bool = field(
        default=False,
        metadata={
            "help": "Estimate the warp loss-vs-t curve with a separate no-grad forward every warp_est_every steps. Each block has a uniform-time anchor; per-position warpers use coupled times for other positions."
        },
    )
    warp_est_every: int = field(
        default=4,
        metadata={
            "help": "warp_uniform_est: run the uniform estimation forward every N steps (amortizes the extra forward; the EMA is slow so N=4-8 is fine)."
        },
    )
    warp_est_samples: int = field(
        default=4,
        metadata={
            "help": "warp_uniform_est: run N independent warp-fit draws and concatenate their samples for a lower-variance fit. 1 = single draw."
        },
    )

    # Time warp logging
    time_warp_plot_every: int = field(
        default=500,
        metadata={
            "help": "Log the per-position generation grid to wandb every N steps (0 = off)"
        },
    )
    time_warp_hist_every: int = field(
        default=1000,
        metadata={
            "help": "Log a histogram of sampled flow times every N steps (0 = off)"
        },
    )

    # DiT conditioning
    adaln_init_scale: float = field(
        default=0.001, metadata={"help": "DiT AdaLN zero-init scale"}
    )
    adaln_no_shift: bool = field(
        default=True, metadata={"help": "Disable shift in AdaLN (scale-only)"}
    )
    adaln_gate_init: float = field(
        default=0.0,
        metadata={
            "help": "Initial AdaLN gate bias; with adaln_init_scale=0 use 1.0 to start as the vanilla pretrained model"
        },
    )
    adaln_share_layers: bool = field(
        default=False,
        metadata={
            "help": "Use ONE DiTAdaLN for every decoder layer instead of one per layer: 78.19M -> 4.25M adaLN params (18.4x) on the 0.5B model. Stored as a 1-entry adaln_layers ModuleList so checkpoint naming and the adaln_lr_mult param group are unchanged. False = original per-layer architecture."
        },
    )

    # Self-conditioning
    enable_self_cond: bool = field(
        default=True, metadata={"help": "Enable self-conditioning"}
    )
    sc_mode: str = field(
        default="residual",
        metadata={"help": "Self-cond mode: concat | residual | simple | mlp | scale"},
    )
    sc_drop_rate: float = field(
        default=0.5, metadata={"help": "Self-cond dropout rate during training"}
    )

    # Embeddings
    freeze_embed: bool = field(
        default=True, metadata={"help": "Freeze embedding layer"}
    )
    freeze_lm_head: bool = field(default=True, metadata={"help": "Freeze LM head"})
    untie_lm_head: bool = field(
        default=False,
        metadata={
            "help": "Untie lm_head from embed (required if freeze_embed=True + freeze_lm_head=False)"
        },
    )
    embed_norm_project: bool = field(
        default=False,
        metadata={
            "help": "PGD: pin each embedding row's L2 norm to its pretrained value after every step"
        },
    )

    # Losses
    logit_shift: bool = field(
        default=False,
        metadata={"help": "Shift denoising target by +1 (NTP convention)"},
    )
    flow_ce_enable: bool = field(
        default=True,
        metadata={
            "help": "Enable noisy-half CE (flow) loss. False = skip its loss contribution regardless of flow_ce_weight."
        },
    )
    flow_ce_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight on the noisy-half CE (flow) loss. Set to 0 to disable token-CE supervision (e.g., when using denoise_mse_weight as sole signal)."
        },
    )
    denoise_mse_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight for denoise-embedding MSE: ‖softmax(noisy_logits)@E − e_1‖² (post-softmax target alignment)"
        },
    )
    ar_loss: bool = field(
        default=False,
        metadata={"help": "Add auxiliary AR loss on clean context tokens"},
    )
    ar_loss_weight: float = field(default=1.0, metadata={"help": "Weight for AR loss"})

    # Optimization and weight EMA
    adaln_lr_mult: float = field(
        default=5.0,
        metadata={
            "help": "LR multiplier for AdaLN/time_embed/self-cond params (>1 trains flow layers faster)"
        },
    )
    ema_decay: float = field(
        default=0.0,
        metadata={
            "help": "EMA decay for weights (e.g. 0.999/0.9999). <=0 disables EMA."
        },
    )
    ema_start_step: int = field(
        default=0,
        metadata={"help": "Begin EMA updates at this global step (warm-up before)."},
    )

    # Checkpoints
    init_from: Optional[str] = field(
        default=None,
        metadata={"help": "Path to safetensors file to init flow model from"},
    )
    time_warp_init_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Initialize the EMA histogram from time_warper.pt (file or directory), for example when continuing weights from init_from."
        },
    )
    resume_weights_only: bool = field(
        default=False,
        metadata={
            "help": "Disaster recovery: restore Trainer step/data position plus EMA/warper from resume_from_checkpoint, while loading model weights through init_from and creating a fresh optimizer/scheduler. Use only when native ZeRO optimizer shards are incomplete."
        },
    )
    save_every_epoch: bool = field(
        default=False,
        metadata={
            "help": "Save an epoch-{N}/ model snapshot with EMA and time warper sidecars, independent of save_steps."
        },
    )


def _load_backbone_and_tokenizer(flow_args):
    tokenizer = AutoTokenizer.from_pretrained(flow_args.model_name_or_path)
    backbone = AutoModelForCausalLM.from_pretrained(
        flow_args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if flow_args.untie_lm_head and getattr(
        backbone.config, "tie_word_embeddings", True
    ):
        backbone.config.tie_word_embeddings = False
        embedding = backbone.get_input_embeddings().weight
        # ZeRO-3 may partition the embedding here; config dimensions remain available.
        vocab, hidden = backbone.config.vocab_size, backbone.config.hidden_size
        lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        try:
            import deepspeed  # noqa: F401
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

            zero3 = is_deepspeed_zero3_enabled()
        except Exception:
            zero3 = False
        if zero3:
            import deepspeed

            with deepspeed.zero.GatheredParameters([embedding], modifier_rank=0):
                lm_head.weight = torch.nn.Parameter(embedding.data.detach().clone())
        else:
            lm_head.weight = torch.nn.Parameter(embedding.data.detach().clone())
        backbone.lm_head = lm_head
        accelerator.print(
            f"[Flow] Untied lm_head from embeddings (vocab={vocab}, hidden={hidden}, zero3={zero3})"
        )

    return backbone, tokenizer


def _build_flow_model(backbone, flow_args):
    model = FlowModelWrapper(
        backbone,
        bd_size=flow_args.bd_size,
        noise_scale_mult=flow_args.noise_scale_mult,
        adaln_init_scale=flow_args.adaln_init_scale,
        adaln_no_shift=flow_args.adaln_no_shift,
        adaln_share_layers=flow_args.adaln_share_layers,
        adaln_gate_init=flow_args.adaln_gate_init,
        enable_self_cond=flow_args.enable_self_cond,
        sc_mode=flow_args.sc_mode,
        sc_drop_rate=flow_args.sc_drop_rate,
        freeze_embed=flow_args.freeze_embed,
        freeze_lm_head=flow_args.freeze_lm_head,
        block_causal=flow_args.block_causal,
        noisy_causal=flow_args.noisy_causal,
        logit_shift=flow_args.logit_shift,
        ar_loss=flow_args.ar_loss,
        ar_loss_weight=flow_args.ar_loss_weight,
        flow_ce_weight=flow_args.flow_ce_weight,
        denoise_mse_weight=flow_args.denoise_mse_weight,
        ema_decay=flow_args.ema_decay,
        time_warp_num_bins=flow_args.time_warp_num_bins,
        time_warp_per_position=flow_args.time_warp_per_position,
        time_warp_init_path=flow_args.time_warp_init_path,
        flow_ce_enable=flow_args.flow_ce_enable,
        time_warp_start_step=flow_args.time_warp_start_step,
        warp_uniform_est=flow_args.warp_uniform_est,
        warp_est_every=flow_args.warp_est_every,
        warp_est_samples=flow_args.warp_est_samples,
        embed_norm_project=flow_args.embed_norm_project,
    )
    model.to(torch.bfloat16)
    return model


def _load_initial_weights(model, init_from):
    from safetensors.torch import load_file as safe_load

    accelerator.print(f"[Flow] Loading weights from: {init_from}")
    init_sd = safe_load(init_from)
    config_path = os.path.join(os.path.dirname(init_from), "config.json")
    tie_word_embeddings = None
    if os.path.isfile(config_path):
        with open(config_path) as config_file:
            tie_word_embeddings = json.load(config_file).get("tie_word_embeddings")
    missing, unexpected = load_init_state_dict(
        model, init_sd, strict=False, tie_word_embeddings=tie_word_embeddings
    )
    if missing:
        accelerator.print(
            f"[Flow] init_from: {len(missing)} missing keys (first 5): {missing[:5]}"
        )
    if unexpected:
        accelerator.print(
            f"[Flow] init_from: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}"
        )
    accelerator.print(
        f"[Flow] Loaded {len(init_sd) - len(unexpected)} matching keys from checkpoint"
    )


def _build_callbacks(flow_args):
    callbacks = [
        TimeWarpCallback(
            plot_every=flow_args.time_warp_plot_every,
            hist_every=flow_args.time_warp_hist_every,
        ),
    ]
    if flow_args.embed_norm_project:
        callbacks.append(EmbedNormProjectCallback())
    if flow_args.ema_decay > 0:
        callbacks.append(EMACallback(start_step=flow_args.ema_start_step))

    epoch_ckpt_cb = None
    if flow_args.save_every_epoch:
        epoch_ckpt_cb = EpochCheckpointCallback()
        callbacks.append(epoch_ckpt_cb)
    return callbacks, epoch_ckpt_cb


def _trainer_class_for_config(flow_args):
    if flow_args.adaln_lr_mult == 1.0:
        return FlowTrainer

    adaln_names = {
        "adaln_layers",
        "time_embed",
        "sc_proj",
        "sc_lin_a",
        "sc_lin_b",
        "sc_mlp",
        "sc_scale",
    }

    class AdaLNLRTrainer(FlowTrainer):
        def create_optimizer(self, model=None):
            if self.optimizer is not None:
                return self.optimizer
            target = model if model is not None else self.model
            adaln_params, other_params = [], []
            for name, param in target.named_parameters():
                if not param.requires_grad:
                    continue
                if any(part in name for part in adaln_names):
                    adaln_params.append(param)
                else:
                    other_params.append(param)

            lr = self.args.learning_rate
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": lr},
                    {"params": adaln_params, "lr": lr * flow_args.adaln_lr_mult},
                ],
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
                eps=self.args.adam_epsilon,
            )
            accelerator.print(
                f"[AdaLN LR] backbone={lr}, adaln={lr * flow_args.adaln_lr_mult} "
                f"({flow_args.adaln_lr_mult}x), adaln_params={len(adaln_params)}, "
                f"other_params={len(other_params)}"
            )
            return self.optimizer

    return AdaLNLRTrainer


def _resolve_resume_checkpoint(training_args):
    resume_from = training_args.resume_from_checkpoint
    if resume_from is None or resume_from == "auto":
        resume_from = latest_checkpoint(training_args.output_dir)
        if resume_from:
            accelerator.print(f"[Flow] Auto-resuming from: {resume_from}")
    return resume_from


def main():
    t_start = time.time()
    accelerator.print(f"[Flow] Script started at {time.strftime('%H:%M:%S')}")

    flow_args, training_args = parse_config_and_training_args(FlowConfig)
    training_args.ddp_find_unused_parameters = (
        flow_args.enable_self_cond and flow_args.sc_drop_rate > 0
    )
    # Non-reentrant checkpointing supports multiple losses from one backbone output.
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    set_seed(training_args.seed)
    if accelerator.is_main_process:
        flow_args.save(f"{training_args.output_dir}/flow_config.json")
    accelerator.print(f"[Flow] Args parsed ({time.time() - t_start:.1f}s)")

    from src.utils.hub import resolve_checkpoint
    from transformers.utils.hub import cached_file
    explicit = getattr(flow_args, '_explicit_fields', set())
    config_path = cached_file(flow_args.model_name_or_path, 'config.json',
                             revision=flow_args.revision, subfolder=flow_args.subfolder)
    with open(config_path) as handle:
        source_config = json.load(handle)
    if source_config.get('model_type') == 'lafa':
        from src.lafa.modeling_lafa import LafaForCausalLM
        if flow_args.init_from:
            raise ValueError('Use the Lafa model path/subfolder directly instead of init_from')
        source = resolve_checkpoint(flow_args.model_name_or_path,
                                    revision=flow_args.revision, subfolder=flow_args.subfolder)
        saved = source_config['flow_config']
        architecture_keys = ('adaln_no_shift', 'adaln_share_layers', 'enable_self_cond',
                             'sc_mode', 'block_causal', 'noisy_causal', 'logit_shift',
                             'time_warp_num_bins', 'time_warp_per_position')
        for key in architecture_keys:
            if key in explicit and key in saved and getattr(flow_args,key) != saved[key]:
                raise ValueError(f'{key} conflicts with the pretrained Lafa architecture')
        requested_bd = flow_args.bd_size if 'bd_size' in explicit else saved['bd_size']
        if requested_bd != saved['bd_size'] and not flow_args.reset_time_warper:
            raise ValueError('Changing bd_size requires --reset_time_warper true')
        for key, value in saved.items():
            if hasattr(flow_args, key) and key not in explicit:
                setattr(flow_args, key, value)
        # A pretrained schedule is already calibrated; new optimizer steps start at zero.
        if 'time_warp_start_step' not in explicit:
            flow_args.time_warp_start_step = 0
        model = LafaForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        for key in saved:
            if hasattr(flow_args,key):
                value = getattr(flow_args,key)
                model.config.flow_config[key] = value
                if hasattr(model,key):
                    setattr(model,key,value)
        model.bd_size = requested_bd
        flow_args.bd_size = requested_bd
        model.config.flow_config['bd_size'] = requested_bd
        model._embed_norm_project = flow_args.embed_norm_project
        model.embed_tokens.weight.requires_grad_(not flow_args.freeze_embed)
        model.lm_head.weight.requires_grad_(not flow_args.freeze_lm_head)
        if flow_args.reset_time_warper:
            model.time_warper = None
            model._ensure_time_warper('cpu')
        model.config.weight_variant = 'raw'
        accelerator.print(f'[Flow] Additional training from {source}; fresh optimizer/scheduler/EMA')
    else:
        backbone, tokenizer = _load_backbone_and_tokenizer(flow_args)
        model = _build_flow_model(backbone, flow_args)
        if flow_args.init_from:
            _load_initial_weights(model, flow_args.init_from)
    training_args.ddp_find_unused_parameters = (
        flow_args.enable_self_cond and flow_args.sc_drop_rate > 0)
    if accelerator.is_main_process:
        flow_args.save(f'{training_args.output_dir}/flow_config.json')

    accelerator.print(
        f"[Flow] Loading dataset {flow_args.dataset_name} "
        f"({time.time() - t_start:.1f}s)"
    )
    train_dataset = load_and_tokenize_dataset(
        dataset_name=flow_args.dataset_name,
        tokenizer=tokenizer,
        block_size=flow_args.block_size,
    )
    accelerator.print(
        f"[Flow] Dataset ready: {len(train_dataset)} samples "
        f"({time.time() - t_start:.1f}s)"
    )

    collator = FlowDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        bd_size=flow_args.bd_size,
    )

    callbacks, epoch_ckpt_cb = _build_callbacks(flow_args)

    accelerator.print(f"[Flow] Creating Trainer ({time.time() - t_start:.1f}s)")
    trainer_class = _trainer_class_for_config(flow_args)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.resume_weights_only = flow_args.resume_weights_only
    if epoch_ckpt_cb is not None:
        epoch_ckpt_cb.trainer = trainer

    accelerator.print(f"[Flow] Starting training ({time.time() - t_start:.1f}s)")
    resume_from = _resolve_resume_checkpoint(training_args)
    accelerator.print(f"[Flow] resume_from_checkpoint = {resume_from!r}")
    if flow_args.resume_weights_only:
        if resume_from is None:
            raise ValueError(
                "resume_weights_only=true requires a checkpoint in output_dir "
                "or an explicit resume_from_checkpoint path."
            )
        if not flow_args.init_from:
            raise ValueError(
                "resume_weights_only=true requires init_from to point at the "
                "checkpoint's consolidated model.safetensors."
            )
        accelerator.print(
            "[Flow] Recovery mode enabled: model weights came from init_from; "
            "Trainer state, EMA, and time warper will come from "
            f"{resume_from}. Adam moments and rank RNG state will restart."
        )
    # State callbacks read the resolved checkpoint path from training_args.
    training_args.resume_from_checkpoint = resume_from
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()
    accelerator.print(f"[Flow] Done! ({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
