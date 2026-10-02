#!/usr/bin/env python3
"""Lafa integration with lm-evaluation-harness.

Registers ``lafa`` and the checkpoint-compatible ``ar2flow`` alias for
generation tasks. Point ``pretrained`` at a local Lafa checkpoint and
``base_model`` at its pretrained backbone. A saved flow_config.json supplies
training settings that cannot be inferred from model weights.

Example:
    python -m src.evaluation.harness --model lafa --tasks gsm8k_cot \
        --num_fewshot 0 --apply_chat_template \
        --model_args "pretrained=output/checkpoint-1000,base_model=Qwen/Qwen2.5-0.5B-Instruct,gen_mode=confidence,num_steps=1,threshold=0.95,use_ema=true"
"""

import glob
import json
import contextlib
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Pin mixed precision before an Accelerator initializes its process-wide state.
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")

import accelerate
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from src.flow.model import FlowModelWrapper
from src.utils.checkpoints import latest_checkpoint
from src.utils.hub import resolve_checkpoint

# Offline mode avoids Hub requests from each process.
_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0").lower() in (
    "1",
    "true",
) or os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in ("1", "true")


def _set_nccl_timeout(seconds: int = None):
    """Set the NCCL process-group timeout after initialization."""
    if not dist.is_initialized():
        return
    if seconds is None:
        seconds = int(os.environ.get("TORCH_NCCL_TIMEOUT", "7200"))
    dist.distributed_c10d._set_pg_timeout(timedelta(seconds=seconds))


def _str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def _generation_settings(kwargs, max_new_tokens):
    """The model limit is an upper bound, including task-specific requests."""
    task_limit = kwargs.get("max_gen_toks")
    limit = int(max_new_tokens)
    if task_limit is not None:
        limit = min(limit, int(task_limit))
    if limit < 1:
        raise ValueError("Generation token limits must be positive")
    return limit, float(kwargs.get("temperature") or 0.0)


def _load_state_dict(checkpoint):
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    single = os.path.join(checkpoint, "model.safetensors")
    binf = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(index):
        with open(index) as f:
            shards = set(json.load(f)["weight_map"].values())
        sd = {}
        for s in sorted(shards):
            sd.update(load_file(os.path.join(checkpoint, s)))
        return sd
    if os.path.exists(single):
        return load_file(single)
    shards = sorted(glob.glob(os.path.join(checkpoint, "model-*.safetensors")))
    if shards:
        sd = {}
        for s in shards:
            sd.update(load_file(s))
        return sd
    if os.path.exists(binf):
        return torch.load(binf, map_location="cpu")
    raise FileNotFoundError(f"No weights in {checkpoint}")


def _find_last_checkpoint(path: str) -> str:
    return latest_checkpoint(path) or path


def _eval_output_dir() -> str:
    """Resolve the lm-eval --output_path so gen_stats.json is saved alongside the
    eval results. Falls back to $OUTPUT_PATH, then the current working dir."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--output_path" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--output_path="):
            return a.split("=", 1)[1]
    return os.environ.get("OUTPUT_PATH") or os.getcwd()


def _build_model(
    pretrained,
    base_model,
    device,
    *,
    block_size=8,
    noise_scale_mult=4.0,
    block_causal=False,
    noisy_causal=False,
    logit_shift=False,
    force_block_size=0,
    time_warp_per_position_override=None,
    noise_scale_override=None,
    dtype=torch.bfloat16,
    revision=None,
    subfolder='',
):
    """Load checkpoint into FlowModelWrapper with auto-detected architecture."""
    is_main = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    pretrained = resolve_checkpoint(pretrained, revision=revision, subfolder=subfolder,
                                    local_files_only=_OFFLINE)
    pretrained = _find_last_checkpoint(pretrained)
    config_file = os.path.join(pretrained, 'config.json')
    if os.path.isfile(config_file):
        with open(config_file) as handle:
            saved_config = json.load(handle)
        if saved_config.get('model_type') == 'lafa':
            from src.lafa.modeling_lafa import LafaForCausalLM
            model = LafaForCausalLM.from_pretrained(pretrained, dtype=dtype).to(device).eval()
            if force_block_size and int(force_block_size) != model.bd_size:
                raise ValueError('HF checkpoints require their saved block size for inference')
            if noise_scale_override is not None:
                model.noise_scale_mult = float(noise_scale_override)
            if time_warp_per_position_override is not None:
                state = model.time_warper.state_dict()
                model.time_warp_per_position = bool(time_warp_per_position_override)
                model.time_warper = None
                model._ensure_time_warper(device).load_state_dict(state)
            return model, pretrained, pretrained

    # Prefer the checkpoint config, then the parent run config.
    fcfg = {}
    for _cand in (
        os.path.join(pretrained, "flow_config.json"),
        os.path.join(os.path.dirname(os.path.normpath(pretrained)), "flow_config.json"),
    ):
        if os.path.exists(_cand):
            with open(_cand) as config_file:
                fcfg = json.load(config_file)
            break
    if fcfg:
        block_size = int(fcfg.get("bd_size", block_size))
        noise_scale_mult = float(fcfg.get("noise_scale_mult", noise_scale_mult))
        block_causal = bool(fcfg.get("block_causal", block_causal))
        noisy_causal = bool(fcfg.get("noisy_causal", noisy_causal))
        logit_shift = bool(fcfg.get("logit_shift", logit_shift))
    base_model = (
        base_model or fcfg.get("model_name_or_path") or "Qwen/Qwen2.5-0.5B-Instruct"
    )
    if noise_scale_override is not None:
        noise_scale_mult = float(noise_scale_override)
    # Allow evaluation at a block size different from training.
    if force_block_size and int(force_block_size) > 0:
        if is_main:
            print(
                f"[ar2flow] force_block_size={force_block_size} overrides trained "
                f"bd_size={block_size} (cross-block-size transfer eval)"
            )
        block_size = int(force_block_size)
    tw_num_bins = int(fcfg.get("time_warp_num_bins", 100))
    if fcfg.get("time_warp_dist", "ema") != "ema":
        raise ValueError("Only EMA time warper checkpoints are supported")
    # Older checkpoints used beta_theta_per_pos.
    saved_per_position = bool(
        fcfg.get("time_warp_per_position", fcfg.get("beta_theta_per_pos", True))
    )
    time_warp_per_position = (
        saved_per_position
        if time_warp_per_position_override is None
        else bool(time_warp_per_position_override)
    )
    if time_warp_per_position_override is not None and is_main:
        print(
            f"[ar2flow] time_warp_per_position OVERRIDE -> {time_warp_per_position} "
            f"(fcfg had {saved_per_position}); grid_mode=warp will use "
            f"{'per-position grids' if time_warp_per_position else 'a POOLED shared 1-D grid'}"
        )

    if is_main:
        if fcfg:
            print("[ar2flow] arch auto-loaded from flow_config.json")
        print(f"[ar2flow] Loading checkpoint: {pretrained}")
        # Populate the cache before other ranks load.
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            attn_implementation="eager",
            local_files_only=_OFFLINE,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if not is_main:
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            attn_implementation="eager",
            local_files_only=_OFFLINE,
        )
    d = backbone.config.hidden_size

    state_dict = _load_state_dict(pretrained)

    # Separate saved embedding and head weights require untied parameters.
    emb_w = state_dict.get("backbone.model.embed_tokens.weight")
    head_w = state_dict.get("backbone.lm_head.weight")
    untied_ckpt = (
        emb_w is not None and head_w is not None and not torch.equal(emb_w, head_w)
    )
    if untied_ckpt and getattr(backbone.config, "tie_word_embeddings", True):
        backbone.config.tie_word_embeddings = False
        E = backbone.get_input_embeddings().weight
        lm_head = torch.nn.Linear(E.shape[1], E.shape[0], bias=False)
        lm_head.weight = torch.nn.Parameter(E.detach().clone())
        backbone.lm_head = lm_head
        if is_main:
            print(
                "[ar2flow] untied lm_head (checkpoint has separate lm_head; matching training)"
            )

    adaln_key = next(
        (
            k
            for k in state_dict
            if "adaln_layers" in k and "adaLN_modulation.1.weight" in k
        ),
        None,
    )
    if adaln_key is None:
        raise RuntimeError("Checkpoint has no adaln_layers — not a flow checkpoint?")
    n_out = state_dict[adaln_key].shape[0] // d
    adaln_no_shift = n_out == 4
    adaln_indices = {
        int(key.split("adaln_layers.", 1)[1].split(".", 1)[0])
        for key in state_dict
        if "adaln_layers." in key
        and key.split("adaln_layers.", 1)[1].split(".", 1)[0].isdigit()
    }
    adaln_share_layers = bool(
        fcfg.get(
            "adaln_share_layers",
            len(adaln_indices) == 1 and len(backbone.model.layers) > 1,
        )
    )
    has_target_time = any(key.startswith("time_embed_tgt.") for key in state_dict)
    has_target_adaln = any(key.startswith("adaln_layers_tgt.") for key in state_dict)
    two_time = bool(fcfg.get("two_time", has_target_time))
    default_two_time_mode = "separate_adaln" if has_target_adaln else "shared_embedding"
    two_time_mode = fcfg.get(
        "distill_two_time_mode",
        default_two_time_mode,
    )
    distill_mode = bool(fcfg.get("distill_mode", False))

    has_sc_lin_a = "sc_lin_a.weight" in state_dict
    has_sc_lin_b = "sc_lin_b.weight" in state_dict
    has_sc_proj = "sc_proj.weight" in state_dict
    has_sc_mlp = "sc_mlp.0.weight" in state_dict
    has_sc_scale = "sc_scale" in state_dict

    if has_sc_lin_a and has_sc_lin_b:
        sc_mode = "residual"
        enable_self_cond = True
    elif has_sc_lin_b and not has_sc_lin_a:
        sc_mode = "simple"
        enable_self_cond = True
    elif has_sc_proj:
        sc_mode = "concat"
        enable_self_cond = True
    elif has_sc_mlp:
        sc_mode = "mlp"
        enable_self_cond = True
    elif has_sc_scale:
        sc_mode = "scale"
        enable_self_cond = True
    else:
        sc_mode = "residual"
        enable_self_cond = False
    if distill_mode:
        enable_self_cond = False

    if is_main:
        print(
            f"[ar2flow] Auto-detected: no_shift={adaln_no_shift} "
            f"sc_mode={sc_mode} "
            f"enable_self_cond={enable_self_cond} "
            f"adaln_share_layers={adaln_share_layers} "
            f"two_time={two_time} two_time_mode={two_time_mode}"
        )
        print(
            f"[ar2flow] Model args: block_size={block_size} "
            f"noise_scale_mult={noise_scale_mult} block_causal={block_causal} "
            f"noisy_causal={noisy_causal} logit_shift={logit_shift}"
        )
        print(f"[ar2flow] Paths: pretrained={pretrained} base_model={base_model}")

    model = FlowModelWrapper(
        backbone,
        bd_size=block_size,
        noise_scale_mult=noise_scale_mult,
        adaln_init_scale=0.0,
        adaln_no_shift=adaln_no_shift,
        adaln_share_layers=adaln_share_layers,
        two_time=two_time,
        distill_two_time_mode=two_time_mode,
        distill_second_time_input=fcfg.get(
            "distill_second_time_input",
            "target",
        ),
        distill_second_time_bias=bool(fcfg.get("distill_second_time_bias", True)),
        enable_self_cond=enable_self_cond,
        sc_mode=sc_mode,
        sc_drop_rate=0.0,
        freeze_embed=True,
        freeze_lm_head=True,
        block_causal=block_causal,
        noisy_causal=noisy_causal,
        logit_shift=logit_shift,
        time_warp_num_bins=tw_num_bins,
        time_warp_per_position=time_warp_per_position,
    )

    ignored_sc = []
    if distill_mode:
        sc_prefixes = (
            "sc_proj.",
            "sc_lin_a.",
            "sc_lin_b.",
            "sc_mlp.",
            "sc_scale",
        )
        ignored_sc = [key for key in state_dict if key.startswith(sc_prefixes)]
        for key in ignored_sc:
            state_dict.pop(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    inference_noise_sigma = fcfg.get("inference_noise_sigma")
    if inference_noise_sigma is not None:
        model._noise_sigma = float(inference_noise_sigma)
    elif distill_mode:
        # The loaded embedding table sets corruption scale for distill students.
        model._noise_sigma = model.embed_tokens.weight.float().std().item()
    if is_main:
        if ignored_sc:
            print(
                "[ar2flow] ignored inherited frozen SC tensors for "
                f"SC-free distill student: {ignored_sc}"
            )
        print(
            f"[ar2flow] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"  missing: {missing[:5]}")
        if unexpected:
            print(f"  unexpected: {unexpected[:5]}")

    model = model.to(dtype=dtype, device=device).eval()
    # Restore the learned grid from the checkpoint or its run directory.
    src = model.load_time_warper(pretrained, device=device)
    if is_main:
        print(
            f"[ar2flow] time warper: {src if src else 'none found -> fresh (~uniform) grid'}"
        )
    return model, pretrained, base_model


@register_model("lafa", "ar2flow")
class AR2FlowEvalHarness(LM):
    """lm-eval adapter for Lafa checkpoints."""

    def __init__(
        self,
        pretrained,
        base_model=None,
        block_size=8,
        num_steps=32,
        noise_scale_mult=4.0,
        block_causal="false",
        noisy_causal="false",
        logit_shift="false",
        max_new_tokens=256,
        max_length=2048,
        grid_mode="auto",
        t_max=0.2,
        threshold=0.0,
        ode_argmax="false",
        fixed_block_noise="false",
        cond_t=1.0,
        sde_gamma=0.0,
        sde_alpha=0.0,
        sde_vp="false",
        sde_noiseless="false",  # true: deterministic α-rewind (z=α·e_t, no injection)
        euler_alpha=1.0,  # <1: damped Euler — decay retained-state term by α (1=std)
        cfg_scale=1.0,
        use_ema="false",
        gen_mode="flow",
        spec_self_cond="false",  # gen_mode=speculative: self-condition the draft (2 fwd/round)
        gen_batch_size=1,
        show_samples="true",
        commit_per_step=1,
        force_block_size=0,
        time_warp_per_position=None,  # None=inherit config; false=pooled grid
        noise_scale_override=None,  # None=inherit fcfg noise; set=tune noise HP
        seed=42,
        revision=None,
        subfolder='',
        **kwargs,
    ):
        super().__init__()
        valid_modes = ("ar", "flow", "confidence", "greedy", "speculative")
        if gen_mode not in valid_modes:
            raise ValueError(
                f"Unsupported gen_mode={gen_mode!r}; choose one of {valid_modes}. "
            )

        block_size = int(block_size)
        force_block_size = int(force_block_size)
        num_steps = int(num_steps)
        noise_scale_mult = float(noise_scale_mult)
        max_new_tokens = int(max_new_tokens)
        max_length = int(max_length)
        t_max = float(t_max)
        self.threshold = float(threshold)
        self.gen_batch_size = int(gen_batch_size)
        self.seed = int(seed)
        self.block_causal = _str2bool(block_causal)
        self.noisy_causal = _str2bool(noisy_causal)
        self.logit_shift = _str2bool(logit_shift)

        acc = accelerate.Accelerator()
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1
        self._device = acc.device
        self.accelerator = acc

        model, ckpt_path, base_model = _build_model(
            pretrained,
            base_model,
            self._device,
            revision=revision,
            subfolder=subfolder,
            block_size=block_size,
            noise_scale_mult=noise_scale_mult,
            block_causal=self.block_causal,
            noisy_causal=self.noisy_causal,
            logit_shift=self.logit_shift,
            force_block_size=force_block_size,
            time_warp_per_position_override=(
                None
                if time_warp_per_position is None
                else _str2bool(time_warp_per_position)
            ),
            noise_scale_override=(
                None if noise_scale_override is None else float(noise_scale_override)
            ),
        )
        if model._two_time and gen_mode not in ("ar", "speculative", "flow"):
            raise ValueError(
                "Two-time student checkpoints currently support only "
                "gen_mode='ar', 'flow' or 'speculative'"
            )
        self.model = model
        # Prefer the checkpoint tokenizer, then the base model tokenizer.
        tok_src = (
            ckpt_path
            if ckpt_path
            and os.path.exists(os.path.join(ckpt_path, "tokenizer_config.json"))
            else base_model
        )
        if self._rank == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_src, local_files_only=_OFFLINE,
                trust_remote_code=getattr(model.config, "model_type", None) == "lafa"
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self._rank != 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_src, local_files_only=_OFFLINE,
                trust_remote_code=getattr(model.config, "model_type", None) == "lafa"
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.block_size = model.bd_size
        self.num_steps = num_steps
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        if grid_mode == "auto":
            grid_mode = "warp"
            if self._rank == 0:
                print(f"[ar2flow] grid_mode=auto -> {grid_mode}")
        self.grid_mode = grid_mode
        self.t_max = t_max
        self.ode_argmax = _str2bool(ode_argmax)
        self.fixed_block_noise = _str2bool(fixed_block_noise)
        self.cond_t = float(cond_t)
        self.sde_gamma = float(sde_gamma)
        self.sde_alpha = float(sde_alpha)
        self.sde_vp = _str2bool(sde_vp)
        self.sde_noiseless = _str2bool(sde_noiseless)
        self.euler_alpha = float(euler_alpha)
        self.spec_self_cond = _str2bool(spec_self_cond)
        self.cfg_scale = float(cfg_scale)
        self.use_ema = _str2bool(use_ema)
        self.gen_mode = gen_mode
        self.show_samples = _str2bool(show_samples)
        self.commit_per_step = int(commit_per_step)
        self._gen_stats = []

        if getattr(self.model.config, 'weight_variant', None) == 'ema':
            self.use_ema = False  # EMA is already materialized in the public weights.
        if self.use_ema and ckpt_path is not None:
            ep = os.path.join(ckpt_path, "ema.pt")
            if os.path.exists(ep):
                self.model.load_ema_state_dict(
                    torch.load(ep, map_location="cpu"), device=self._device
                )
                if self._rank == 0:
                    print(f"[ar2flow] loaded EMA weights from {ep} (gen under EMA)")
            elif self._rank == 0:
                print(
                    f"[ar2flow][warn] use_ema set but no ema.pt in {ckpt_path}; "
                    "using raw weights"
                )

        if self._rank == 0:
            print(f"[ar2flow] checkpoint={ckpt_path}")
            print(
                f"[ar2flow] base_model={base_model} block_size={self.block_size} "
                f"num_steps={num_steps} noise_scale_mult={model.noise_scale_mult} "
                f"logit_shift={self.logit_shift} grid_mode={grid_mode} t_max={t_max} "
                f"threshold={self.threshold}"
            )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length_(self):
        return self.max_length

    @property
    def max_gen_toks(self):
        return self.max_new_tokens

    @property
    def batch_size(self):
        return 1

    def tok_encode(self, s, add_special_tokens=False):
        return self.tokenizer.encode(s, add_special_tokens=add_special_tokens)

    def tok_decode(self, ids, skip_special_tokens=False):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def all_gather(self, tensor):
        if self._world_size <= 1:
            return tensor.unsqueeze(0)
        out = [torch.zeros_like(tensor) for _ in range(self._world_size)]
        torch.distributed.all_gather(out, tensor.contiguous())
        return torch.stack(out)

    def gather_object(self, obj, dst=0):
        if self._world_size <= 1:
            return [obj]
        out = [None] * self._world_size
        torch.distributed.all_gather_object(out, obj)
        return out if self._rank == dst else None

    def barrier(self):
        if self._world_size > 1:
            torch.distributed.barrier()

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @torch.no_grad()
    def generate_until(self, requests):
        out = [None] * len(requests)

        def request_settings(instance):
            return _generation_settings(instance.args[1], self.max_new_tokens)

        def _stops(kwargs):
            until = kwargs.get("until") or []
            return [until] if isinstance(until, str) else list(until)

        def _postprocess(text, stops):
            for stop in stops:
                if stop and stop in text:
                    text = text.split(stop)[0]
            # Code completions can need leading indentation.
            return text.rstrip()

        def _record(
            i,
            n_total,
            context,
            generated,
            final,
            num_generated,
            stats,
            gen_time,
            prompt_tokens,
        ):
            if self.show_samples and self._rank == 0:
                bar = "=" * 70
                _fwd = stats.get("total_steps", stats.get("forwards", 0)) or 0
                print(
                    f"\n{bar}\n[ar2flow sample {i + 1}/{n_total}] "
                    f"mode={self.gen_mode} gen_tokens={num_generated} "
                    f"forwards={_fwd} tok/fwd={num_generated / max(_fwd, 1):.2f}"
                )
                print(f"--- PROMPT (tail) ---\n...{context[-300:]}")
                print(f"--- RAW GENERATED ---\n{generated!r}")
                print(f"--- FINAL (after stop strings) ---\n{final}\n{bar}")
            # Decode modes report forward counts under different keys.
            fwd = stats.get("total_steps", stats.get("forwards", 0))
            # Logical steps omit CFG and self-conditioning bootstrap forwards.
            logical = stats.get("logical_steps", fwd)
            record = {
                "tokens_per_step": (
                    stats.get("tokens_per_step") or (num_generated / max(fwd, 1))
                ),
                "total_steps": fwd,
                "forwards": fwd,
                "logical_steps": logical,
                "prompt_tokens": int(prompt_tokens),
                "num_generated": num_generated,
                "gen_time_s": gen_time,
                "tokens_per_second": num_generated / gen_time if gen_time > 0 else 0,
            }
            for key in (
                "accepted_per_forward",
                "fused_forwards",
                "bootstrap_forwards",
                "correction_forwards",
                "first_token_rejects",
                "self_cond",
                "block_size",
                "num_euler_steps",
            ):
                if key in stats:
                    record[key] = stats[key]
            self._gen_stats.append(record)

        eos_str = self.tokenizer.decode([self.tokenizer.eos_token_id])
        # Stop on chat and base-model EOS tags before scoring.
        stop_strs = [s for s in [eos_str, "<|im_end|>", "<|endoftext|>"] if s]
        n_total = len(requests)
        # Self-speculative and thresholded flow decoding run one sequence at a time.
        use_batched = (
            self.gen_batch_size > 1
            and self.gen_mode != "speculative"
            and not self.model._two_time
            and (
                self.gen_mode in ("ar", "confidence", "greedy") or self.threshold <= 0.0
            )
            and (
                not requests
                or all(
                    request_settings(request) == request_settings(requests[0])
                    for request in requests
                )
            )
        )

        if use_batched:
            for s in tqdm(
                range(0, n_total, self.gen_batch_size),
                desc="Generating (ar2flow, batched)...",
            ):
                chunk = list(range(s, min(s + self.gen_batch_size, n_total)))
                contexts = [requests[i].args[0] for i in chunk]
                max_gen, temperature = request_settings(requests[chunk[0]])
                enc = [
                    self.tokenizer(c, add_special_tokens=False)["input_ids"]
                    for c in contexts
                ]
                maxL = max(len(e) for e in enc)
                pad_id = self.tokenizer.pad_token_id
                B = len(enc)
                ids = torch.full((B, maxL), pad_id, dtype=torch.long)
                am = torch.zeros(B, maxL, dtype=torch.long)
                for j, e in enumerate(enc):
                    ids[j, maxL - len(e) :] = torch.tensor(e, dtype=torch.long)
                    am[j, maxL - len(e) :] = 1
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                t0 = time.perf_counter()
                _ema_ctx = (
                    self.model.ema_swap()
                    if self.use_ema and self.model.has_ema
                    else contextlib.nullcontext()
                )
                with _ema_ctx:
                    if self.gen_mode == "ar":
                        out_list, stats = self.model.generate_ar_batched(
                            ids.to(self._device),
                            am.to(self._device),
                            max_new_tokens=max_gen,
                            temperature=temperature,
                        )
                    elif self.gen_mode in ("confidence", "greedy"):
                        out_list, stats = self.model.generate_confidence_batched(
                            ids.to(self._device),
                            am.to(self._device),
                            max_new_tokens=max_gen,
                            order=self.gen_mode,
                            num_steps=self.num_steps,
                            grid_mode=self.grid_mode,
                            t_max=self.t_max,
                            temperature=temperature,
                            commit_per_step=self.commit_per_step,
                            ode_argmax=self.ode_argmax,
                            fixed_block_noise=self.fixed_block_noise,
                            threshold=self.threshold,
                            cond_t=self.cond_t,
                            sde_alpha=self.sde_alpha,
                            sde_gamma=self.sde_gamma,
                            sde_vp=self.sde_vp,
                            sde_noiseless=self.sde_noiseless,
                            euler_alpha=self.euler_alpha,
                        )
                    else:
                        out_list, stats = self.model.generate_batched(
                            ids.to(self._device),
                            am.to(self._device),
                            max_new_tokens=max_gen,
                            num_steps=self.num_steps,
                            temperature=temperature,
                            grid_mode=self.grid_mode,
                            t_max=self.t_max,
                            ode_argmax=self.ode_argmax,
                            sde_gamma=self.sde_gamma,
                            sde_alpha=self.sde_alpha,
                            sde_vp=self.sde_vp,
                            sde_noiseless=self.sde_noiseless,
                            euler_alpha=self.euler_alpha,
                            cfg_scale=self.cfg_scale,
                            seed=self.seed,
                        )
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                gen_time = (time.perf_counter() - t0) / B
                # Use per-sequence counts when available; otherwise share batch cost.
                B_eff = len(chunk)
                per_seq_fwd = stats.get("per_seq_forwards")
                batch_fwd = stats.get("forwards", stats.get("total_steps", 0))
                for j, i in enumerate(chunk):
                    generated_ids = out_list[j][len(enc[j]) :]
                    generated = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=False
                    )
                    out[i] = _postprocess(
                        generated, _stops(requests[i].args[1]) + stop_strs
                    )
                    num_generated = len(generated_ids)
                    stats_j = dict(stats)
                    stats_j["forwards"] = (
                        per_seq_fwd[j] if per_seq_fwd is not None else batch_fwd / B_eff
                    )
                    stats_j.pop("total_steps", None)
                    _record(
                        i,
                        n_total,
                        contexts[j],
                        generated,
                        out[i],
                        num_generated,
                        stats_j,
                        gen_time,
                        len(enc[j]),
                    )
        else:
            for i, instance in enumerate(
                tqdm(requests, desc="Generating (ar2flow)...")
            ):
                context, gen_kwargs = instance.args
                until = _stops(gen_kwargs)
                max_gen, temperature = request_settings(instance)

                prompt_ids = self.tokenizer(
                    context,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(self._device)

                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                t0 = time.perf_counter()
                _ema_ctx = (
                    self.model.ema_swap()
                    if self.use_ema and self.model.has_ema
                    else contextlib.nullcontext()
                )
                with _ema_ctx:
                    if self.gen_mode == "ar":
                        gen_ids, stats = self.model.generate_ar(
                            prompt_ids,
                            max_new_tokens=max_gen,
                            temperature=temperature,
                        )
                    elif self.gen_mode in ("confidence", "greedy"):
                        gen_ids, stats = self.model.generate_confidence(
                            prompt_ids,
                            max_new_tokens=max_gen,
                            order=self.gen_mode,
                            num_steps=self.num_steps,
                            grid_mode=self.grid_mode,
                            t_max=self.t_max,
                            temperature=temperature,
                            commit_per_step=self.commit_per_step,
                            ode_argmax=self.ode_argmax,
                            fixed_block_noise=self.fixed_block_noise,
                            threshold=self.threshold,
                            cond_t=self.cond_t,
                            sde_alpha=self.sde_alpha,
                            sde_gamma=self.sde_gamma,
                            sde_vp=self.sde_vp,
                            sde_noiseless=self.sde_noiseless,
                            euler_alpha=self.euler_alpha,
                        )
                    elif self.gen_mode == "speculative":
                        gen_ids, stats = self.model.generate_self_speculative(
                            prompt_ids,
                            max_new_tokens=max_gen,
                            grid_mode=self.grid_mode,
                            t_max=self.t_max,
                            seed=self.seed,
                            self_cond=self.spec_self_cond,
                        )
                    else:
                        gen_ids, stats = getattr(self.model, 'generate_flow', self.model.generate)(
                            prompt_ids,
                            max_new_tokens=max_gen,
                            num_steps=self.num_steps,
                            threshold=self.threshold,
                            temperature=temperature,
                            grid_mode=self.grid_mode,
                            t_max=self.t_max,
                            ode_argmax=self.ode_argmax,
                            sde_gamma=self.sde_gamma,
                            sde_alpha=self.sde_alpha,
                            sde_vp=self.sde_vp,
                            sde_noiseless=self.sde_noiseless,
                            euler_alpha=self.euler_alpha,
                            cfg_scale=self.cfg_scale,
                            seed=self.seed,
                        )
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                gen_time = time.perf_counter() - t0

                generated_ids = gen_ids[0, prompt_ids.shape[1] :].tolist()
                generated = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                )
                out[i] = _postprocess(generated, until + stop_strs)
                num_generated = len(generated_ids)
                _record(
                    i,
                    n_total,
                    context,
                    generated,
                    out[i],
                    num_generated,
                    stats,
                    gen_time,
                    prompt_ids.shape[1],
                )

        if self._rank == 0 and self._gen_stats:
            import json

            tot_tok = sum(s["num_generated"] for s in self._gen_stats)
            tot_fwd = sum(s["total_steps"] for s in self._gen_stats)
            tot_logical = sum(s["logical_steps"] for s in self._gen_stats)
            tot_time = sum(s["gen_time_s"] for s in self._gen_stats)
            avg_tps = sum(s["tokens_per_step"] for s in self._gen_stats) / len(
                self._gen_stats
            )
            avg_steps = sum(s["total_steps"] for s in self._gen_stats) / len(
                self._gen_stats
            )
            avg_logical = sum(s["logical_steps"] for s in self._gen_stats) / len(
                self._gen_stats
            )
            avg_tok_sec = sum(s["tokens_per_second"] for s in self._gen_stats) / len(
                self._gen_stats
            )
            aggregate_tok_sec = tot_tok / tot_time if tot_time > 0 else 0.0
            summary = {
                "_summary": True,
                "n_samples": len(self._gen_stats),
                "total_tokens": tot_tok,
                "total_forwards": tot_fwd,
                "total_logical_forwards": tot_logical,
                "total_generation_time_s": tot_time,
                "aggregate_tokens_per_second": aggregate_tok_sec,
                "tokens_per_forward": tot_tok / max(tot_fwd, 1),
                "forwards_per_token": tot_fwd / max(tot_tok, 1),
                "logical_forwards_per_token": tot_logical / max(tot_tok, 1),
                "avg_tokens_per_step": avg_tps,
                "avg_total_steps": avg_steps,
                "avg_logical_steps": avg_logical,
                "avg_tokens_per_second": avg_tok_sec,
            }
            print(
                f"[ar2flow] {len(self._gen_stats)} samples: "
                f"total_tokens={tot_tok} total_forwards={tot_fwd} "
                f"logical_forwards={tot_logical} "
                f"tokens_per_forward={tot_tok / max(tot_fwd, 1):.3f} "
                f"logical_NFE_per_token={tot_logical / max(tot_tok, 1):.3f} "
                f"avg_tokens_per_step={avg_tps:.2f} avg_total_steps={avg_steps:.1f} "
                f"aggregate_tokens_per_second={aggregate_tok_sec:.1f} "
                f"avg_tokens_per_second={avg_tok_sec:.1f}"
            )
            for stats_dir in (_eval_output_dir(),):
                if stats_dir:
                    stats_path = os.path.join(stats_dir, "gen_stats.json")
                    try:
                        os.makedirs(stats_dir, exist_ok=True)
                        with open(stats_path, "w") as f:
                            json.dump([summary] + self._gen_stats, f, indent=2)
                        print(f"[ar2flow] Gen stats saved to {stats_path}")
                    except OSError:
                        pass

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError("ar2flow does not support loglikelihood tasks")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "ar2flow does not support loglikelihood_rolling tasks"
        )


if __name__ == "__main__":
    _set_nccl_timeout()
    cli_evaluate()
