"""Embedding-space flow model with DiT time conditioning."""

import contextlib

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.modeling_outputs import CausalLMOutputWithPast

from .base_wrapper import BaseModelWrapper
from .attention import create_packed_attention_mask
from .cache import trim_kv_cache, gather_kv_cache
from .layers import TimestepEmbedder, DiTAdaLN
from .loss import cross_entropy_loss


class FlowModelWrapper(BaseModelWrapper):
    """Wrap a causal LM for block flow training and decoding."""

    def __init__(
        self,
        backbone: nn.Module,
        bd_size: int = 8,
        noise_scale_mult: float = 4.0,
        adaln_init_scale: float = 1e-3,
        adaln_no_shift: bool = True,
        adaln_gate_init: float = 0.0,
        adaln_share_layers: bool = False,
        two_time: bool = False,
        distill_two_time_mode: str = "shared_embedding",
        distill_second_time_input: str = "target",
        distill_second_time_bias: bool = True,
        enable_self_cond: bool = True,
        sc_mode: str = "residual",
        sc_drop_rate: float = 0.5,
        freeze_embed: bool = True,
        freeze_lm_head: bool = True,
        embed_norm_project: bool = False,
        block_causal: bool = False,
        noisy_causal: bool = False,
        logit_shift: bool = False,
        ar_loss: bool = False,
        ar_loss_weight: float = 1.0,
        flow_ce_weight: float = 1.0,
        denoise_mse_weight: float = 0.0,
        ema_decay: float = 0.0,
        time_warp_num_bins: int = 100,
        time_warp_per_position: bool = True,
        time_warp_init_path: str = None,
        flow_ce_enable: bool = True,
        time_warp_start_step: int = 0,
        warp_uniform_est: bool = False,
        warp_est_every: int = 4,
        warp_est_samples: int = 4,
        noise_sigma: float | None = None,
    ):
        super().__init__(backbone)
        if bd_size < 1:
            raise ValueError("bd_size must be positive")

        self.bd_size = bd_size
        self.block_causal = block_causal
        self.noisy_causal = noisy_causal
        self.noise_scale_mult = noise_scale_mult
        # The warper is saved separately from model.state_dict().
        self.time_warp_num_bins = time_warp_num_bins
        self.time_warp_per_position = time_warp_per_position
        self.time_warp_init_path = time_warp_init_path
        self.flow_ce_enable = flow_ce_enable
        self.time_warp_start_step = time_warp_start_step
        self.warp_uniform_est = warp_uniform_est
        self.warp_est_every = warp_est_every
        self.warp_est_samples = warp_est_samples
        self.time_warper = None
        self._global_step = 0  # set by TimeWarpCallback for warm-up
        self._last_sampled_t = None  # for the sampled-time histogram
        self.enable_self_cond = enable_self_cond
        self.sc_mode = sc_mode
        self.sc_drop_rate = sc_drop_rate
        self.logit_shift = logit_shift
        self.ar_loss = ar_loss
        self.ar_loss_weight = ar_loss_weight
        self.flow_ce_weight = flow_ce_weight
        self.denoise_mse_weight = denoise_mse_weight

        # Weight EMA is saved separately from model.state_dict().
        self.ema_decay = ema_decay
        self._ema_shadow = None
        # Detached losses for FlowTrainer logging.
        self._loss_components = {}

        self.__dict__["_mask_cache"] = {}

        # ZeRO-3 may partition embedding weights during construction.
        _bcfg = self.backbone.config
        self._d_model = _bcfg.hidden_size
        # Noise scale uses the full embedding table, including under ZeRO-3.
        _E = self.embed_tokens.weight
        if noise_sigma is None:
            with self._maybe_gather([_E]):
                self._noise_sigma = _E.data.std().item()
        else:
            self._noise_sigma = float(noise_sigma)

        # The persistent buffer preserves embedding row norms across resumes.
        self._embed_norm_project = embed_norm_project
        V = _bcfg.vocab_size
        self.register_buffer("_embed_init_norm", torch.zeros(V, 1), persistent=True)

        self._adaln_init_scale = adaln_init_scale
        self._adaln_no_shift = adaln_no_shift
        self._adaln_gate_init = adaln_gate_init
        self._adaln_share_layers = bool(adaln_share_layers)
        self._two_time = bool(two_time)
        self._distill_two_time_mode = str(distill_two_time_mode).strip().lower()
        if self._distill_two_time_mode not in {
            "shared_embedding",
            "source_only",
        }:
            raise ValueError(
                "Two-time inference supports "
                "distill_two_time_mode='shared_embedding' or 'source_only'"
            )
        self._distill_second_time_input = (
            str(distill_second_time_input or "target").strip().lower()
        )
        if self._distill_second_time_input != "target":
            raise ValueError(
                "Two-time inference supports absolute target-time conditioning only"
            )
        self._distill_second_time_bias = bool(distill_second_time_bias)
        if self._two_time and not self._distill_second_time_bias:
            raise ValueError(
                "Bias-free two-time embeddings are not supported by this inference"
            )
        self._freeze_embed = freeze_embed
        self._freeze_lm_head = freeze_lm_head

        self._setup_backbone()

    def _setup_backbone(self):
        if self._freeze_embed:
            self.embed_tokens.weight.requires_grad_(False)
        if self._freeze_lm_head:
            self.lm_head.weight.requires_grad_(False)

        self._setup_dit_adaln()
        if self.enable_self_cond:
            self._setup_self_cond()

    @staticmethod
    def _maybe_gather(params):
        """Gather ZeRO-3 parameters when needed; otherwise use a null context."""
        try:
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

            if is_deepspeed_zero3_enabled():
                import deepspeed

                return deepspeed.zero.GatheredParameters(params, modifier_rank=None)
        except Exception:
            pass
        return contextlib.nullcontext()

    def _setup_dit_adaln(self):
        d = self._d_model
        self.time_embed = TimestepEmbedder(d)
        if self._two_time and self._distill_two_time_mode != "source_only":
            self.time_embed_tgt = TimestepEmbedder(d)
            if self._distill_two_time_mode == "shared_embedding":
                nn.init.zeros_(self.time_embed_tgt.mlp[2].weight)
                nn.init.zeros_(self.time_embed_tgt.mlp[2].bias)
        layers = self.backbone.model.layers

        def _mk():
            return DiTAdaLN(
                d,
                init_scale=self._adaln_init_scale,
                no_shift=self._adaln_no_shift,
                gate_init=self._adaln_gate_init,
            )

        if self._adaln_share_layers:
            # Keep the state_dict prefix and AdaLN optimizer group unchanged.
            shared = _mk()
            self.adaln_layers = nn.ModuleList([shared])
            for layer in layers:
                shared.patch_layer(layer)
        else:
            self.adaln_layers = nn.ModuleList([_mk() for _ in layers])
            for layer, adaln in zip(layers, self.adaln_layers):
                adaln.patch_layer(layer)

    def _setup_self_cond(self):
        d = self._d_model
        if self.sc_mode == "concat":
            self.sc_proj = nn.Linear(d * 2, d, bias=False)
            nn.init.zeros_(self.sc_proj.weight)
            self.sc_proj.weight.data[:d, :d] = torch.eye(d)
        elif self.sc_mode == "residual":
            self.sc_lin_a = nn.Linear(d, d, bias=False)
            self.sc_lin_b = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.sc_lin_a.weight)
            nn.init.zeros_(self.sc_lin_b.weight)
        elif self.sc_mode == "simple":
            self.sc_lin_b = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.sc_lin_b.weight)
        elif self.sc_mode == "mlp":
            self.sc_mlp = nn.Sequential(
                nn.Linear(d, d, bias=False),
                nn.SiLU(),
                nn.Linear(d, d, bias=False),
            )
            nn.init.zeros_(self.sc_mlp[2].weight)
        elif self.sc_mode == "scale":
            self.sc_scale = nn.Parameter(torch.zeros(1))

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.backbone.model.embed_tokens

    @property
    def lm_head(self) -> nn.Linear:
        return self.backbone.lm_head

    def state_dict(self, *args, **kwargs):
        return nn.Module.state_dict(self, *args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Older ZeRO-3 checkpoints may have a malformed embedding norm buffer.
        # Keep the current buffer so those checkpoints still load.
        k = "_embed_init_norm"
        cur = getattr(self, k, None)
        if k in state_dict and cur is not None and state_dict[k].shape != cur.shape:
            print(
                f"[flow] load_state_dict: checkpoint '{k}' {tuple(state_dict[k].shape)} "
                f"!= model {tuple(cur.shape)} (ZeRO-3 capture artifact) -> keeping fresh "
                f"buffer",
                flush=True,
            )
            state_dict = dict(state_dict)
            state_dict[k] = cur.detach().clone()
        return nn.Module.load_state_dict(self, state_dict, *args, **kwargs)

    def _sample_noise(self, like: Tensor) -> Tensor:
        return torch.randn_like(like) * self._noise_sigma * self.noise_scale_mult

    def _interpolate(self, e_0: Tensor, e_1: Tensor, t: Tensor) -> Tensor:
        """e_t = (1-t)*e_0 + t*e_1. t: [B,1,1]-broadcastable."""
        return (1 - t) * e_0 + t * e_1

    def _compute_t_bn(
        self,
        t: Tensor,
        tgt: bool = False,
    ) -> Tensor | None:
        """Embed a source time or the PSD checkpoint's absolute target time."""
        if tgt and self._distill_two_time_mode == "source_only":
            return None
        embedder = self.time_embed_tgt if tgt else self.time_embed
        return embedder(t)

    def _compose_flowmap_time_condition(
        self,
        source_t: Tensor,
        target_t: Tensor,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """Build shared-AdaLN conditioning for mixed clean and PSD positions.

        Clean verifier/cache positions use only ``E_source``. Noisy flow-map
        draft positions use ``E_source + E_target``. Masking the target term is
        what lets one fused forward retain the checkpoint's ordinary AR path
        while also evaluating its two-time PSD map.
        """
        source = self._compute_t_bn(source_t)
        if not self._two_time or self._distill_two_time_mode == "source_only":
            return source
        if self._distill_two_time_mode != "shared_embedding":
            raise RuntimeError(
                "Fused PSD self-speculation requires shared_embedding "
                "two-time conditioning"
            )
        target = self._compute_t_bn(
            target_t,
            tgt=True,
        ).to(source.dtype)
        if target_mask is not None:
            target = target * target_mask.to(target.dtype).unsqueeze(-1)
        return source + target

    def _precompute_shared_adaln(self, t_bn: Tensor):
        """Return reusable AdaLN modulation for a static inference layout.

        Shared-AdaLN checkpoints patch the same module into every decoder layer,
        so repeatedly evaluating ``adaln(t_bn)`` produces identical tensors.
        Non-shared checkpoints compute each layer's modulation separately.
        """
        if not getattr(self, "_adaln_share_layers", False):
            return None
        return self.adaln_layers[0](t_bn)

    def _ensure_time_warper(self, device):
        """Lazily build the TimeWarping module on the right device (t in [0,1])."""
        if self.time_warper is None:
            from .time_warping import TimeWarping

            self.time_warper = TimeWarping(
                num_bins=self.time_warp_num_bins,
                t_min=0.0,
                t_max=1.0,
                device=device,
                per_position=self.time_warp_per_position,
                bd_size=self.bd_size,
            )
            _twip = getattr(self, "time_warp_init_path", None)
            if _twip:
                import os

                p = (
                    os.path.join(_twip, "time_warper.pt")
                    if os.path.isdir(_twip)
                    else _twip
                )
                self.time_warper.load_state_dict(torch.load(p, map_location=device))
        elif self.time_warper.loss_ema.device != torch.device(device):
            self.time_warper.to(device)
        return self.time_warper

    def load_time_warper(self, ckpt_path, device=None):
        """Load a trained TimeWarping for generation: prefer
        ckpt_path/time_warper.pt, else fall back to the latest
        checkpoint-*/time_warper.pt in the parent run dir. Returns the source
        path, or None (leaving a fresh ~uniform warper) if nothing is found."""
        import glob
        import os
        import re

        if device is None:
            device = self.embed_tokens.weight.device
        src = None
        direct = os.path.join(ckpt_path, "time_warper.pt")
        if os.path.exists(direct):
            src = direct
        else:
            parent = os.path.dirname(os.path.normpath(ckpt_path))
            cks = glob.glob(os.path.join(parent, "checkpoint-*", "time_warper.pt"))
            if cks:
                src = max(
                    cks, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1))
                )
        if src is None:
            return None
        self._ensure_time_warper(device)
        self.time_warper.load_state_dict(torch.load(src, map_location=device))
        return src

    def _warp_active(self):
        """True once warm-up is over (uniform-t before time_warp_start_step)."""
        return self._global_step >= self.time_warp_start_step

    def _sample_t_units(self, n, device):
        """Sample learned times after the uniform warm-up."""
        wp = self._ensure_time_warper(device)
        if self._warp_active():
            return wp.sample(n, device=device).detach()
        return torch.rand(n, device=device)

    def _embedding_shard_slices(self, embedding):
        """Yield contiguous portions of each row in this ZeRO-3 shard."""
        import torch.distributed as dist

        shard = embedding.ds_tensor
        group = getattr(embedding, "ds_process_group", None)
        rank = dist.get_rank(group=group) if dist.is_initialized() else 0
        width = self._d_model
        start = rank * shard.numel()
        remaining = max(0, min(shard.numel(), embedding.ds_numel - start))
        offset = 0
        row = start // width

        if start % width and remaining:
            length = min(remaining, width - start % width)
            yield row, shard[:length]
            offset += length
            row += 1

        while remaining - offset >= width:
            n_rows = min((remaining - offset) // width, 4096)
            length = n_rows * width
            yield (
                slice(row, row + n_rows),
                shard[offset : offset + length].view(n_rows, width),
            )
            offset += length
            row += n_rows

        if offset < remaining:
            yield row, shard[offset:remaining]

    def _embedding_row_norms(self, embedding):
        """Compute full row norms without gathering a ZeRO-3 embedding table."""
        shard = getattr(embedding, "ds_tensor", None)
        if shard is None:
            return embedding.detach().float().norm(dim=-1)

        import torch.distributed as dist

        squared = torch.zeros(
            self._embed_init_norm.shape[0], device=shard.device, dtype=torch.float32
        )
        for rows, values in self._embedding_shard_slices(embedding):
            squared[rows] = values.float().square().sum(dim=-1)
        if dist.is_initialized():
            dist.all_reduce(squared, group=getattr(embedding, "ds_process_group", None))
        return squared.sqrt()

    def _capture_embed_norms(self):
        """Capture pretrained row norms once, including from ZeRO-3 shards."""
        if not self._embed_norm_project or self._freeze_embed:
            return
        if float(self._embed_init_norm.abs().sum()) > 0:
            return
        with torch.no_grad():
            norms = self._embedding_row_norms(self.embed_tokens.weight)
            self._embed_init_norm.copy_(
                norms.unsqueeze(-1).to(self._embed_init_norm.device)
            )
        print(
            f"[EmbedNormProject] captured per-row norms "
            f"(mean={self._embed_init_norm.mean().item():.4f}).",
            flush=True,
        )

    def _post_step_project(self):
        """Restore captured row norms after an optimizer step."""
        if not self._embed_norm_project or self._freeze_embed:
            return
        if float(self._embed_init_norm.abs().sum()) == 0:
            return
        with torch.no_grad():
            embedding = self.embed_tokens.weight
            current = self._embedding_row_norms(embedding).clamp_min(1e-8)
            target = self._embed_init_norm.flatten().to(current.device)
            scale = target.float() / current
            shard = getattr(embedding, "ds_tensor", None)
            if shard is None:
                embedding.mul_(scale[:, None].to(embedding.dtype))
            else:
                for rows, values in self._embedding_shard_slices(embedding):
                    factor = scale[rows]
                    if isinstance(rows, slice):
                        factor = factor[:, None]
                    values.mul_(factor.to(values.dtype))

    # ZeRO-3 updates local EMA shards and gathers them only for saving.
    @staticmethod
    def _ema_param_view(p):
        """The tensor the EMA shadow tracks for param p: its ZeRO-3 local shard
        (p.ds_tensor) if partitioned, else the full param. Always materialized."""
        ds = getattr(p, "ds_tensor", None)
        return ds if ds is not None else p

    def ema_initialize(self):
        """Initialize fp32 EMA shadows for trainable parameters."""
        if self.ema_decay <= 0 or self._ema_shadow is not None:
            return
        self._ema_shadow = {
            n: self._ema_param_view(p).detach().float().clone()
            for n, p in self.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def ema_update(self):
        """EMA step: shadow = decay*shadow + (1-decay)*param. Call AFTER optimizer.step().
        Shard-local under ZeRO-3 (no gather); shape-guarded so a transient mismatch skips
        rather than crashes."""
        if not self._ema_shadow:
            return
        d = self.ema_decay
        for n, p in self.named_parameters():
            s = self._ema_shadow.get(n)
            if s is None:
                continue
            v = self._ema_param_view(p)
            if v.shape != s.shape:
                # A ZeRO-3 shard may not be materialized on this step.
                continue
            s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @property
    def has_ema(self):
        return bool(self._ema_shadow)

    def ema_state_dict(self):
        """Return full EMA weights for saving.

        ZeRO-3 gathers shards collectively on all ranks. Only rank 0 receives
        the complete state.
        """
        if not self._ema_shadow:
            return None
        z3 = any(
            getattr(p, "ds_tensor", None) is not None
            for _, p in self.named_parameters()
        )
        if not z3:
            return self._ema_shadow
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        full = {}
        name_to_p = dict(self.named_parameters())
        for n, s in self._ema_shadow.items():
            p = name_to_p.get(n)
            if p is None:
                continue
            # NCCL gathers on GPU; assemble on CPU to limit peak GPU memory.
            shard = s.detach().float().contiguous()
            if dist.is_initialized():
                shards = [torch.empty_like(shard) for _ in range(dist.get_world_size())]
                dist.all_gather(shards, shard)
                if rank == 0:
                    flat = torch.cat([sh.cpu() for sh in shards])
                else:
                    del shards
                    continue
            else:
                flat = shard.cpu()
            # Remove ZeRO shard padding.
            true_numel = getattr(p, "ds_numel", None) or p.numel()
            full[n] = (
                flat[:true_numel].view(getattr(p, "ds_shape", None) or p.shape).clone()
            )
        return full if rank == 0 else None

    def load_ema_state_dict(self, sd, device=None):
        """Load a FULL-tensor EMA state (as written by ema_state_dict). Under ZeRO-3,
        re-shard each full tensor into this rank's local shard to match the live layout."""
        dev = device or self.embed_tokens.weight.device
        z3 = any(
            getattr(p, "ds_tensor", None) is not None
            for _, p in self.named_parameters()
        )
        if not z3:
            self._ema_shadow = {k: v.float().to(dev) for k, v in sd.items()}
            return
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        shadow = {}
        name_to_p = dict(self.named_parameters())
        for n, full in sd.items():
            p = name_to_p.get(n)
            if p is None:
                continue
            flat = full.detach().float().flatten()
            # Match ZeRO's contiguous shard layout.
            shard_numel = getattr(p, "ds_tensor", p).numel()
            pad = shard_numel * world - flat.numel()
            if pad > 0:
                flat = torch.cat([flat, flat.new_zeros(pad)])
            shadow[n] = (
                flat[rank * shard_numel : (rank + 1) * shard_numel].to(dev).clone()
            )
        self._ema_shadow = shadow

    @contextlib.contextmanager
    def ema_swap(self):
        """Temporarily load EMA weights into the live params; restore on exit.
        No-op if no EMA shadow exists. Under ZeRO-3 this swaps each rank's local SHARD
        (via _ema_param_view -> p.ds_tensor), so the subsequently-gathered params in the
        eval forward reflect EMA — no all-gather needed here."""
        if not self._ema_shadow:
            yield
            return
        backup = {}
        with torch.no_grad():
            for n, p in self.named_parameters():
                s = self._ema_shadow.get(n)
                if s is None:
                    continue
                v = self._ema_param_view(p)
                if v.shape != s.shape:
                    continue
                backup[n] = v.detach().clone()
                v.data.copy_(s.to(v.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if n in backup:
                        self._ema_param_view(p).data.copy_(backup[n])

    def _prepare_flow_input(self, input_ids: Tensor, cond_mask: Tensor):
        """input_ids [B, L] → e_1, e_t, t_bn, position_ids, attn_mask, t.

        `t` is the sampled time, per position for a coupled warper or per block
        during warm-up and for a shared warper. It is returned for warper fitting.

        cond_mask [B, L]: 1 = answer (corrupt), 0 =
        prompt/pad (kept CLEAN, conditioned at t=1)."""
        B, L = input_ids.shape
        device = input_ids.device

        e_1 = self.embed_tokens(input_ids)  # [B, L, d]

        # Couple position-specific times with one quantile per block.
        num_blocks = L // self.bd_size
        if self.time_warp_per_position and self._warp_active():
            wp = self._ensure_time_warper(device)
            tau = torch.rand(B * num_blocks, device=device)  # one tau per block
            tc = wp.sample_tau_coupled(  # [B*nb, bd_size]
                tau, device=device
            ).detach()
            # Positions follow absolute token columns; boundary blocks have no flow loss.
            t_pos = tc.view(B, num_blocks, self.bd_size).reshape(B, L)
            t = t_pos
        else:
            t = self._sample_t_units(B * num_blocks, device).view(B, num_blocks)
            t_pos = t.repeat_interleave(self.bd_size, dim=1)  # [B, L]

        e_0 = self._sample_noise(e_1)  # [B, L, d]
        e_t = self._interpolate(
            e_0, e_1, t_pos.unsqueeze(-1).to(e_1.dtype)
        )  # [B, L, d]

        # Keep prompt/pad positions clean and condition them at t=1.
        m_e = cond_mask.unsqueeze(-1).to(e_t.dtype)  # [B, L, 1]
        e_t = m_e * e_t + (1.0 - m_e) * e_1
        m_t = cond_mask.to(t_pos.dtype)  # [B, L]
        t_pos = m_t * t_pos + (1.0 - m_t) * 1.0

        ones_clean = torch.ones(B, L, device=device, dtype=t_pos.dtype)
        per_pos_t = torch.cat([ones_clean, t_pos], dim=1)  # [B, 2L]
        t_bn = self._compute_t_bn(per_pos_t)  # [B, 2L, d]

        # Position ids: [0..L-1, 0..L-1] — noisy half shares positions with clean
        pos = torch.arange(L, device=device)
        position_ids = torch.cat([pos, pos]).unsqueeze(0).expand(B, -1)  # [B, 2L]

        cache_key = (
            L,
            self.bd_size,
            self.block_causal,
            self.noisy_causal,
            device,
            e_1.dtype,
        )
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = create_packed_attention_mask(
                L,
                self.bd_size,
                device=device,
                dtype=e_1.dtype,
                block_causal=self.block_causal,
                noisy_causal=self.noisy_causal,
            )
        attn_mask = self._mask_cache[cache_key]

        return e_1, e_t, t_bn, position_ids, attn_mask, t

    def _full_block_loss_mask(self, valid: Tensor) -> Tensor:
        """Keep positions only when their entire block is supervised."""
        B, L = valid.shape
        block_full = valid.view(B, L // self.bd_size, self.bd_size).all(dim=-1)
        return valid & block_full.repeat_interleave(self.bd_size, dim=1)

    @staticmethod
    def _answer_corruption_mask(labels: Tensor):
        """Map supervised tokens to 1 and prompt/pad tokens to clean 0."""
        return (labels != -100).float()

    def _denoising_squared_error(self, noisy_logits, e_1):
        probs = F.softmax(noisy_logits, dim=-1)
        e_denoised = probs @ self.embed_tokens.weight
        target = F.pad(e_1, (0, 0, 0, 1))[:, 1:, :] if self.logit_shift else e_1
        return ((e_denoised - target.to(e_denoised.dtype)) ** 2).sum(dim=-1)

    def _warp_loss_per_token(self, ce_pp, sq_err):
        loss = (self.flow_ce_weight if self.flow_ce_enable else 0.0) * ce_pp.float()
        if sq_err is not None:
            loss = loss + self.denoise_mse_weight * sq_err.float()
        return loss

    @torch.no_grad()
    def _warp_fit_inputs(self, ce_pp, sq_err, valid, t_raw):
        """Reduce supervised token losses to position or block histogram samples."""
        B, L = valid.shape
        nb = L // self.bd_size
        loss_pp = self._warp_loss_per_token(ce_pp, sq_err)

        if self.time_warp_per_position:
            if t_raw.numel() == B * L:
                t_pos = t_raw.reshape(B, L)
            elif t_raw.numel() == B * nb:
                t_pos = t_raw.reshape(B, nb).repeat_interleave(self.bd_size, dim=1)
            else:
                raise ValueError("Expected per-position or per-block sampled times")

            positions = (
                torch.arange(L, device=loss_pp.device)
                .remainder(self.bd_size)
                .unsqueeze(0)
                .expand(B, L)
            )
            t_by_pos, loss_by_pos = [], []
            for pos in range(self.bd_size):
                selected = valid & (positions == pos)
                t_by_pos.append(t_pos[selected])
                loss_by_pos.append(loss_pp[selected])
            return t_by_pos, loss_by_pos

        complete_blocks = valid.view(B, nb, self.bd_size).all(dim=-1)
        return (
            t_raw.reshape(B, nb)[complete_blocks],
            loss_pp.view(B, nb, self.bd_size).mean(dim=-1)[complete_blocks],
        )

    @torch.no_grad()
    def _warp_block_fit_inputs(
        self,
        e_1: Tensor,
        labels: Tensor,
        position_ids: Tensor,
        per_position: bool = False,
    ):
        """Concatenate independent warp estimates without increasing peak batch size."""
        n_samples = int(getattr(self, "warp_est_samples", 1) or 1)
        if n_samples <= 1:
            return self._warp_block_fit_once(e_1, labels, position_ids, per_position)
        if per_position:
            t_acc = [[] for _ in range(self.bd_size)]
            l_acc = [[] for _ in range(self.bd_size)]
            for _ in range(n_samples):
                tbp, lbp = self._warp_block_fit_once(e_1, labels, position_ids, True)
                for i in range(self.bd_size):
                    t_acc[i].append(tbp[i])
                    l_acc[i].append(lbp[i])
            t_by_pos = [
                torch.cat(t_acc[i]) if t_acc[i] else torch.empty(0, device=e_1.device)
                for i in range(self.bd_size)
            ]
            loss_by_pos = [
                torch.cat(l_acc[i]) if l_acc[i] else torch.empty(0, device=e_1.device)
                for i in range(self.bd_size)
            ]
            return t_by_pos, loss_by_pos
        t_all, l_all = [], []
        for _ in range(n_samples):
            tu, lu = self._warp_block_fit_once(e_1, labels, position_ids, False)
            t_all.append(tu)
            l_all.append(lu)
        return torch.cat(t_all), torch.cat(l_all)

    def _warp_block_fit_once(
        self,
        e_1: Tensor,
        labels: Tensor,
        position_ids: Tensor,
        per_position: bool = False,
    ):
        """Estimate loss at a uniform anchor time for each block."""
        B, L = e_1.shape[0], e_1.shape[1]
        device = e_1.device
        nb = L // self.bd_size

        t = torch.rand(B * nb, device=device).view(B, nb)  # [B, nb]
        if per_position:
            # A random position anchors each block at uniform t; the others share its tau.
            anchor = torch.randint(0, self.bd_size, (B * nb,), device=device)
            tc = self._ensure_time_warper(device).coupled_fit_times(
                t.reshape(B * nb), anchor
            )  # [B*nb, bd_size]
            t_pos = tc.view(B, nb, self.bd_size).reshape(B, L)  # [B, L] per-position
        else:
            t_pos = t.repeat_interleave(self.bd_size, dim=1)  # [B, L] flat shared-t
        e_0 = self._sample_noise(e_1)
        e_t = self._interpolate(e_0, e_1, t_pos.unsqueeze(-1).to(e_1.dtype))

        response = self._answer_corruption_mask(labels).bool()
        e_t = torch.where(response.unsqueeze(-1), e_t, e_1)
        t_condition = torch.where(response, t_pos, torch.ones_like(t_pos))
        ones_clean = torch.ones(B, L, device=device, dtype=t_pos.dtype)
        t_bn = self._compute_t_bn(
            torch.cat([ones_clean, t_condition], dim=1)
        )  # [B, 2L, d]

        cache_key = (
            L,
            self.bd_size,
            self.block_causal,
            self.noisy_causal,
            device,
            e_1.dtype,
        )
        attn_mask = self._mask_cache.get(cache_key)
        if attn_mask is None:  # mirror _prepare_flow_input
            attn_mask = create_packed_attention_mask(
                L,
                self.bd_size,
                device=device,
                dtype=e_1.dtype,
                block_causal=self.block_causal,
                noisy_causal=self.noisy_causal,
            )
            self._mask_cache[cache_key] = attn_mask

        packed = torch.cat([e_1, e_t], dim=1)  # [B, 2L, d]
        logits, _, _ = self._forward_backbone(
            packed, t_bn, attn_mask, use_cache=False, position_ids=position_ids
        )
        noisy_logits = logits[:, L:, :].contiguous()  # [B, L, V]

        _ce_tgt = (
            F.pad(labels, (0, 1), value=-100)[:, 1:] if self.logit_shift else labels
        )
        valid = self._full_block_loss_mask(_ce_tgt != -100)
        # Gather supports non-contiguous shifted targets.
        ce_pp = (
            -F.log_softmax(noisy_logits.float(), dim=-1)
            .gather(-1, _ce_tgt.clamp(min=0).long().unsqueeze(-1))
            .squeeze(-1)
        )  # [B, L]
        sq_err = (
            self._denoising_squared_error(noisy_logits, e_1)
            if self.denoise_mse_weight > 0
            else None
        )
        loss_pp = self._warp_loss_per_token(ce_pp, sq_err)

        if per_position:
            col = torch.arange(L, device=device)
            pos = col % self.bd_size  # [L] block-local position
            t_by_pos, loss_by_pos = [], []
            for i in range(self.bd_size):
                m = valid & (pos == i).unsqueeze(0)  # [B, L]
                t_by_pos.append(t_pos[m])
                loss_by_pos.append(loss_pp[m])
            return t_by_pos, loss_by_pos

        complete_blocks = valid.view(B, nb, self.bd_size).all(dim=-1)
        return (
            t[complete_blocks],
            loss_pp.view(B, nb, self.bd_size).mean(dim=-1)[complete_blocks],
        )

    def _use_self_cond_this_step(self) -> bool:
        """Choose whether this step pays for the self-conditioning bootstrap.

        The decision is step-seeded so every distributed rank takes the same
        backbone-forward path. Divergent choices can deadlock sharded training
        because only some ranks would enter the bootstrap forward's collectives.
        """
        if not (self.enable_self_cond and self.training):
            return False
        step = int(getattr(self, "_global_step", 0) or 0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0x5C0FF + step)
        return torch.rand(1, generator=generator).item() > self.sc_drop_rate

    def _apply_training_self_condition(
        self,
        e_1: Tensor,
        e_t: Tensor,
        t_bn: Tensor,
        attn_mask,
        cond_mask: Tensor,
        position_ids=None,
    ) -> Tensor:
        """Apply predicted self-conditioning, bypassing it on dropped steps."""
        if not (
            self.enable_self_cond and self.training and self._use_self_cond_this_step()
        ):
            return e_t

        with torch.no_grad():
            packed = torch.cat([e_1, e_t], dim=1)
            logits_first, _, _ = self._forward_backbone(
                packed,
                t_bn,
                attn_mask,
                use_cache=False,
                position_ids=position_ids,
            )
            length = e_t.shape[1]
            noisy_logits_sc = logits_first[:, length:, :]
            probs_sc = F.softmax(noisy_logits_sc.float(), dim=-1).to(e_1.dtype)
            e_prev = probs_sc @ self.embed_tokens.weight

        e_t = self._apply_self_cond(e_t, e_prev.detach())

        # Self-cond transforms (notably residual's A(e_t) branch) perturb every
        # position. Keep prompt/padding positions exactly on their clean embeds
        # when self-conditioning is active.
        mask = cond_mask.unsqueeze(-1).to(e_t.dtype)
        e_t = mask * e_t + (1.0 - mask) * e_1
        return e_t

    def _apply_self_cond(self, e_t: Tensor, e_prev: Tensor) -> Tensor:
        if self.sc_mode == "concat":
            return self.sc_proj(torch.cat([e_t, e_prev], dim=-1))
        elif self.sc_mode == "residual":
            return e_t + self.sc_lin_a(e_t) + self.sc_lin_b(e_prev)
        elif self.sc_mode == "simple":
            return e_t + self.sc_lin_b(e_prev)
        elif self.sc_mode == "mlp":
            return e_t + self.sc_mlp(e_prev)
        elif self.sc_mode == "scale":
            return e_t + self.sc_scale * e_prev
        return e_t

    def _forward_backbone(
        self,
        packed_embeds: Tensor,
        t_bn: Tensor,
        attn_mask,
        past_key_values=None,
        use_cache: bool = False,
        position_ids=None,
        adaln_modulation=None,
        t_bn_tgt=None,
        compute_logits=True,
    ):
        """Run backbone with time conditioning.

        Returns (logits, past_key_values, hidden_states).
        past_key_values is None when use_cache=False.
        """
        if t_bn_tgt is not None:
            t_bn = t_bn + t_bn_tgt
        extra_kwargs = {}
        if adaln_modulation is not None:
            extra_kwargs["dit_adaln_modulation"] = adaln_modulation
        outputs = self.backbone.model(
            inputs_embeds=packed_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            dit_t_bn=t_bn,
            return_dict=True,
            **extra_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states) if compute_logits else None
        past_kv = outputs.past_key_values if use_cache else None
        return logits, past_kv, hidden_states

    def _get_generation_time_grid(
        self,
        num_steps: int,
        device,
        grid_mode: str = "uniform_range",
        t_max: float = 0.2,
    ) -> Tensor:
        """Build Euler step schedule for generation."""
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if grid_mode == "uniform_range":
            return torch.linspace(0, t_max, num_steps + 1).to(device)
        elif grid_mode == "uniform":
            return torch.linspace(0, 1, num_steps + 1).to(device)
        elif grid_mode == "warp":
            self._ensure_time_warper(device)
            if getattr(self, "time_warp_per_position", False):
                return self.time_warper.grid_per_position(num_steps, device=device).to(
                    device
                )
            return self.time_warper.grid(num_steps, device=device).to(device)
        elif grid_mode == "warp_tau":
            # Tau=1 maps to noisy t=0. Use the top quantiles, sort for Euler,
            # and leave endpoints unpinned to preserve the restricted span.
            self._ensure_time_warper(device)
            w = self.time_warper
            tau = torch.linspace(1.0 - float(t_max), 1.0, num_steps + 1, device=device)
            if getattr(self, "time_warp_per_position", False):
                rows = []
                for i in range(w.bd_size):
                    edges = w._pos_edges(i)
                    t, _ = torch.sort(w._invert_u_row(tau, *edges))
                    rows.append(t)
                return torch.stack(rows, dim=0).to(device)  # [bd_size, num_steps+1]
            edges = w._get_bin_widths_and_edges()
            t, _ = torch.sort(w._invert_u_row(tau, *edges))
            return t.to(device)
        else:
            raise ValueError(f"Unknown grid_mode: {grid_mode}")

    def _euler_denoise_block(
        self,
        past_key_values,
        block_pos_ids: Tensor,
        time_grid: Tensor,
        block_attn_mask=None,
        prev_logit: Tensor = None,
        ode_argmax: bool = False,
        sde_gamma: float = 0.0,
        sde_alpha: float = 0.0,
        sde_vp: bool = False,
        sde_noiseless: bool = False,
        euler_alpha: float = 1.0,
        cfg_scale: float = 1.0,
    ) -> tuple[Tensor, int]:
        """Denoise one block via Euler ODE with optional self-conditioning.

        Assumes context is already encoded in past_key_values.
        Trims stale KV entries after each step (does NOT extend cache with result).

        Args:
            prev_logit: [1, V] — last logit from prefix/previous block. Required
                when logit_shift=True to correct ODE trajectory alignment.

        Returns:
            (logits [1, bd_size, V], num_forward_passes)
        """
        device = block_pos_ids.device
        dtype = self.embed_tokens.weight.dtype
        cache_len = past_key_values.get_seq_length()
        num_fwd = 0
        E = self.embed_tokens.weight
        use_cfg = abs(cfg_scale - 1.0) > 1e-6
        per_pos = time_grid.dim() == 2
        n_steps = (time_grid.shape[-1] - 1) if per_pos else (len(time_grid) - 1)

        # The unconditional mask hides prefix keys from the block.
        uncond_mask = None
        if use_cfg:
            Kc = cache_len + self.bd_size
            uncond_mask = (
                block_attn_mask.clone()
                if block_attn_mask is not None
                else torch.zeros(1, 1, self.bd_size, Kc, device=device, dtype=dtype)
            )
            uncond_mask[..., :cache_len] = float("-inf")  # block prompt keys

        def to_denoised(probs_raw):
            # position-aligned predicted-clean embedding (logit_shift handling)
            if self.logit_shift and prev_logit is not None:
                prev_probs = F.softmax(prev_logit.float(), dim=-1).to(dtype)
                pa = torch.cat([prev_probs, probs_raw[:-1]], dim=0)
                return (
                    E[pa.argmax(-1)].unsqueeze(0)
                    if ode_argmax
                    else (pa @ E).unsqueeze(0)
                )
            return (
                E[probs_raw.argmax(-1)].unsqueeze(0)
                if ode_argmax
                else (probs_raw @ E).unsqueeze(0)
            )

        e_t = (
            torch.randn(1, self.bd_size, self._d_model, device=device, dtype=dtype)
            * self._noise_sigma
            * self.noise_scale_mult
        )

        # Conditional and unconditional passes carry separate self-conditioning.
        e_self_cond = e_self_cond_u = None
        if self.enable_self_cond:
            init_t = (
                time_grid[:, 0].to(dtype)[None, :]
                if per_pos
                else torch.full(
                    (1, self.bd_size), time_grid[0].item(), device=device, dtype=dtype
                )
            )
            init_t_bn = self._compute_t_bn(init_t)
            il_c, _, _ = self._forward_backbone(
                e_t,
                init_t_bn,
                block_attn_mask,
                past_key_values=past_key_values,
                use_cache=False,
                position_ids=block_pos_ids,
            )
            trim_kv_cache(past_key_values, past_key_values.get_seq_length() - cache_len)
            e_self_cond = (
                (F.softmax(il_c[0].float(), -1).to(dtype) @ E).unsqueeze(0).detach()
            )
            num_fwd += 1
            if use_cfg:
                il_u, _, _ = self._forward_backbone(
                    e_t,
                    init_t_bn,
                    uncond_mask,
                    past_key_values=past_key_values,
                    use_cache=False,
                    position_ids=block_pos_ids,
                )
                trim_kv_cache(
                    past_key_values, past_key_values.get_seq_length() - cache_len
                )
                e_self_cond_u = (
                    (F.softmax(il_u[0].float(), -1).to(dtype) @ E).unsqueeze(0).detach()
                )
                num_fwd += 1

        for step in range(n_steps):
            if per_pos:
                t_val_vec = time_grid[:, step]  # [bd_size]
                t_out_vec = time_grid[:, step + 1]  # [bd_size]
                if sde_alpha > 0.0 or sde_gamma > 0.0:
                    dt_vec = t_out_vec - t_val_vec
                    if sde_alpha > 0.0:
                        alpha_v = torch.full_like(t_val_vec, sde_alpha)
                    else:
                        alpha_v = (1.0 - sde_gamma * dt_vec).clamp(0.0, 1.0)
                    # Noise at t=0 cannot be backtracked.
                    alpha_v = torch.where(
                        t_val_vec > 0.0, alpha_v, torch.ones_like(alpha_v)
                    )
                    if sde_noiseless:
                        # Draw noise below to keep the seeded RNG stream aligned.
                        coef_v = torch.zeros_like(alpha_v)
                    elif sde_vp:
                        coef_v = torch.sqrt(
                            (
                                (1.0 - alpha_v)
                                * (1.0 + alpha_v - 2.0 * alpha_v * t_val_vec)
                            ).clamp(min=0.0)
                        )
                    else:
                        coef_v = 1.0 - alpha_v
                    a_b = alpha_v.to(dtype)[None, :, None]  # [1, bd_size, 1]
                    c_b = coef_v.to(dtype)[None, :, None]
                    noise = (
                        torch.randn_like(e_t)
                        * self._noise_sigma
                        * self.noise_scale_mult
                    )
                    z_fwd = a_b * e_t + c_b * noise
                    t_fwd_vec = alpha_v * t_val_vec  # conditioning time = alpha*t
                else:
                    t_fwd_vec = t_val_vec
                    z_fwd = e_t
                t_bn_block = self._compute_t_bn(t_fwd_vec.to(dtype)[None, :])
            else:
                t_val = time_grid[step].item()
                t_out = time_grid[step + 1].item()
                dt = t_out - t_val

                if (sde_alpha > 0.0 or sde_gamma > 0.0) and t_val > 0.0:
                    alpha = (
                        sde_alpha
                        if sde_alpha > 0.0
                        else max(0.0, min(1.0, 1.0 - sde_gamma * dt))
                    )
                    t_fwd = alpha * t_val
                    coef = (
                        0.0
                        if sde_noiseless
                        else (
                            math.sqrt(
                                max(
                                    0.0,
                                    (1.0 - alpha) * (1.0 + alpha - 2.0 * alpha * t_val),
                                )
                            )
                            if sde_vp
                            else (1.0 - alpha)
                        )
                    )
                    noise = (
                        torch.randn_like(e_t)
                        * self._noise_sigma
                        * self.noise_scale_mult
                    )
                    z_fwd = alpha * e_t + coef * noise
                else:
                    t_fwd, z_fwd = t_val, e_t

                t_bn_block = self._compute_t_bn(
                    torch.full((1, self.bd_size), t_fwd, device=device, dtype=dtype)
                )

            e_in = (
                self._apply_self_cond(z_fwd, e_self_cond)
                if self.enable_self_cond
                else z_fwd
            )
            logits, _, _ = self._forward_backbone(
                e_in,
                t_bn_block,
                block_attn_mask,
                past_key_values=past_key_values,
                use_cache=False,
                position_ids=block_pos_ids,
            )
            trim_kv_cache(past_key_values, past_key_values.get_seq_length() - cache_len)
            num_fwd += 1
            probs_c = F.softmax(logits[0].float(), dim=-1).to(dtype)
            e_denoised = to_denoised(probs_c)

            if use_cfg:
                e_in_u = (
                    self._apply_self_cond(z_fwd, e_self_cond_u)
                    if self.enable_self_cond
                    else z_fwd
                )
                logits_u, _, _ = self._forward_backbone(
                    e_in_u,
                    t_bn_block,
                    uncond_mask,
                    past_key_values=past_key_values,
                    use_cache=False,
                    position_ids=block_pos_ids,
                )
                trim_kv_cache(
                    past_key_values, past_key_values.get_seq_length() - cache_len
                )
                num_fwd += 1
                probs_u = F.softmax(logits_u[0].float(), dim=-1).to(dtype)
                e_uncond = to_denoised(probs_u)
                e_denoised = e_uncond + cfg_scale * (e_denoised - e_uncond)

            if self.enable_self_cond:
                e_self_cond = (probs_c @ E).unsqueeze(0).detach()
                if use_cfg:
                    e_self_cond_u = (probs_u @ E).unsqueeze(0).detach()

            if step == n_steps - 1:
                break

            # Broadcast position times over embeddings and compute in fp32.
            if per_pos:
                tf = t_fwd_vec[None, :, None].float()  # [1, bd_size, 1]
                to = t_out_vec[None, :, None].float()
                e_t = (
                    z_fwd.float()
                    + (to - tf)
                    * (e_denoised.float() - z_fwd.float())
                    / (1.0 - tf).clamp(min=1e-6)
                ).to(dtype)
                if euler_alpha != 1.0:
                    c_v = (to - tf) / (1.0 - tf).clamp(min=1e-6)
                    e_t = (
                        e_t.float() - (1.0 - c_v) * (1.0 - euler_alpha) * z_fwd.float()
                    ).to(dtype)
            else:
                e_t = z_fwd + (t_out - t_fwd) * (e_denoised - z_fwd) / max(
                    1.0 - t_fwd, 1e-6
                )
                if euler_alpha != 1.0:
                    c = (t_out - t_fwd) / max(1.0 - t_fwd, 1e-6)
                    e_t = e_t - (1.0 - c) * (1.0 - euler_alpha) * z_fwd

        return logits, num_fwd

    def forward(self, input_ids=None, labels=None):
        if input_ids is None or labels is None:
            raise ValueError(
                "Flow training requires input_ids and response-masked labels"
            )
        if input_ids.shape != labels.shape:
            raise ValueError("Flow input_ids and labels must have the same shape")
        if input_ids.shape[1] < self.bd_size or input_ids.shape[1] % self.bd_size != 0:
            raise ValueError("Flow training sequences must align to bd_size")
        B, L = input_ids.shape

        cond_mask = self._answer_corruption_mask(labels)

        e_1, e_t, t_bn, position_ids, attn_mask, t_raw = self._prepare_flow_input(
            input_ids, cond_mask=cond_mask
        )

        e_t = self._apply_training_self_condition(
            e_1=e_1,
            e_t=e_t,
            t_bn=t_bn,
            attn_mask=attn_mask,
            position_ids=position_ids,
            cond_mask=cond_mask,
        )

        packed_embeds = torch.cat([e_1, e_t], dim=1)  # [B, 2L, d]
        logits, _, _ = self._forward_backbone(
            packed_embeds, t_bn, attn_mask, use_cache=False, position_ids=position_ids
        )

        noisy_logits = logits[:, L:, :].contiguous()  # [B, L, V]
        _ce_tgt = (
            F.pad(labels, (0, 1), value=-100)[:, 1:] if self.logit_shift else labels
        )
        # Only complete response blocks contribute to flow loss.
        flow_valid = self._full_block_loss_mask(_ce_tgt != -100)
        ce_valid = flow_valid.float()  # [B, L]
        ce_pp = (
            -F.log_softmax(noisy_logits.float(), dim=-1)
            .gather(-1, _ce_tgt.clamp(min=0).long().unsqueeze(-1))
            .squeeze(-1)
        )  # [B, L]
        flow_loss = (ce_pp * ce_valid).sum() / ce_valid.sum().clamp(min=1.0)
        comps = {"flow_ce": float(flow_loss.detach())}
        sq_err = None

        flow_term = (
            self.flow_ce_weight * flow_loss
            if getattr(self, "flow_ce_enable", True)
            else 0.0 * flow_loss
        )

        if self.ar_loss:
            clean_logits = logits[:, :L, :].contiguous()
            ar_loss_val = cross_entropy_loss(clean_logits, labels, shift=True)
            comps["ar"] = float(ar_loss_val.detach())
            loss = flow_term + self.ar_loss_weight * ar_loss_val
        else:
            loss = flow_term

        if self.denoise_mse_weight > 0:
            sq_err = self._denoising_squared_error(noisy_logits, e_1)
            sq_err_masked = torch.where(flow_valid, sq_err, torch.zeros_like(sq_err))
            n_valid = flow_valid.sum().clamp(min=1).to(sq_err_masked.dtype)
            denoise_mse_val = sq_err_masked.sum().float() / n_valid.float()

            comps["denoise_mse"] = float(denoise_mse_val.detach())
            loss = loss + self.denoise_mse_weight * denoise_mse_val

        # Warm-up samples cover the full time range for the EMA fit.
        if self.training and self.time_warper is not None:
            if self.warp_uniform_est:
                # Every rank uses the same step gate before the extra forward.
                if self._global_step % max(1, self.warp_est_every) == 0:
                    with torch.no_grad():
                        if self.time_warp_per_position:
                            t_by_pos, loss_by_pos = self._warp_block_fit_inputs(
                                e_1, labels, position_ids, per_position=True
                            )
                            for i in range(self.bd_size):
                                self.time_warper.update_warping(
                                    t_by_pos[i], loss_by_pos[i], pos=i
                                )
                        else:
                            t_u, loss_u = self._warp_block_fit_inputs(
                                e_1, labels, position_ids
                            )
                            self.time_warper.update_warping(t_u, loss_u)
                t_units = None
            elif self.time_warp_per_position:
                # Every rank updates position buckets in the same order.
                t_by_pos, loss_by_pos = self._warp_fit_inputs(
                    ce_pp, sq_err, flow_valid, t_raw
                )
                for i in range(self.bd_size):
                    self.time_warper.update_warping(
                        t_by_pos[i].detach(), loss_by_pos[i].detach(), pos=i
                    )
                t_units = None
            else:
                t_units, loss_units = self._warp_fit_inputs(
                    ce_pp, sq_err, flow_valid, t_raw
                )
            # Call on every rank, including ranks with no valid samples, so the
            # per-bin count/loss all-reduce remains in the same order.
            if t_units is not None:
                self.time_warper.update_warping(t_units.detach(), loss_units.detach())

        if self.training:
            with torch.no_grad():
                self._last_sampled_t = t_raw.flatten().detach().float().cpu()

        comps["total"] = float(loss.detach())
        comps.setdefault("flow_loss", comps.get("flow_ce"))
        self._loss_components = comps
        return CausalLMOutputWithPast(loss=loss, logits=noisy_logits)

    @torch.no_grad()
    def generate_ar(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 42,
    ):
        """Decode from the clean causal stream after next-token training."""
        device = next(self.backbone.parameters()).device
        dtype = self.embed_tokens.weight.dtype
        torch.manual_seed(seed)

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(device)
        eos_id = self.backbone.config.eos_token_id

        orig_attn = getattr(self.backbone.config, "_attn_implementation", None)
        self.backbone.config._attn_implementation = "eager"

        gen_ids = prompt_ids[0].tolist()
        n = 0
        try:
            embeds = self.embed_tokens(prompt_ids)
            t_bn = self._compute_t_bn(
                torch.ones(1, prompt_ids.shape[1], device=device, dtype=dtype)
            )
            logits, past, _ = self._forward_backbone(
                embeds, t_bn, attn_mask=None, use_cache=True
            )
            prev_logit = logits[0, -1, :].float()  # P(next token | prompt)

            while n < max_new_tokens:
                if temperature > 0:
                    probs = F.softmax(prev_logit / temperature, dim=-1)
                    tok = int(torch.multinomial(probs, 1).item())
                else:
                    tok = int(prev_logit.argmax().item())
                gen_ids.append(tok)
                n += 1
                if tok == eos_id or n >= max_new_tokens:
                    break
                emb = self.embed_tokens(torch.tensor([[tok]], device=device))
                t1 = self._compute_t_bn(torch.ones(1, 1, device=device, dtype=dtype))
                pos = torch.tensor([[past.get_seq_length()]], device=device)
                logits, past, _ = self._forward_backbone(
                    emb,
                    t1,
                    attn_mask=None,
                    past_key_values=past,
                    use_cache=True,
                    position_ids=pos,
                )
                prev_logit = logits[0, -1, :].float()
        finally:
            self.backbone.config._attn_implementation = orig_attn

        out = torch.tensor(gen_ids, device=device).unsqueeze(0)
        return out, {
            "mode": "ar",
            "generated": n,
            "total_steps": n,
            "logical_steps": n,
            "tokens_per_step": 1.0,
        }

    @torch.no_grad()
    def generate_ar_batched(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 42,
    ):
        """Decode left-padded prompts in a batch with masked cache padding."""
        device = next(self.backbone.parameters()).device
        dtype = self.embed_tokens.weight.dtype
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        prompt_ids = prompt_ids.to(device)
        attention_mask = attention_mask.to(device)
        B, L = prompt_ids.shape
        eos_id = self.backbone.config.eos_token_id
        NEG = torch.finfo(dtype).min

        orig_attn = getattr(self.backbone.config, "_attn_implementation", None)
        self.backbone.config._attn_implementation = "eager"

        cache_key_bias = (1.0 - attention_mask.to(dtype)) * NEG  # [B, L]
        gen_tokens = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        try:
            embeds = self.embed_tokens(prompt_ids)
            t_bn = self._compute_t_bn(torch.ones(B, L, device=device, dtype=dtype))
            causal = torch.triu(
                torch.full((L, L), NEG, device=device, dtype=dtype), diagonal=1
            )
            prefill_mask = causal[None, None] + cache_key_bias[:, None, None, :]
            logits, past, _ = self._forward_backbone(
                embeds, t_bn, prefill_mask, use_cache=True
            )
            prev = logits[:, -1, :].float()  # [B, V]

            gen_pos = L
            n = 0
            while n < max_new_tokens and not bool(finished.all()):
                if temperature > 0:
                    probs = F.softmax(prev / temperature, dim=-1)
                    tok = torch.multinomial(probs, 1, generator=gen).squeeze(1)
                else:
                    tok = prev.argmax(dim=-1)  # [B]
                tok_l = tok.tolist()
                for b in range(B):
                    if finished[b]:
                        continue
                    gen_tokens[b].append(tok_l[b])
                    if eos_id is not None and tok_l[b] == eos_id:
                        finished[b] = True
                n += 1
                if n >= max_new_tokens or bool(finished.all()):
                    break

                emb = self.embed_tokens(tok.unsqueeze(1))  # [B, 1, d]
                t1 = self._compute_t_bn(torch.ones(B, 1, device=device, dtype=dtype))
                cache_len = past.get_seq_length()
                step_mask = torch.zeros(
                    B, 1, 1, cache_len + 1, dtype=dtype, device=device
                )
                step_mask[:, :, :, :cache_len] = cache_key_bias[:, None, None, :]
                pos = torch.arange(gen_pos, gen_pos + 1, device=device)[None].expand(
                    B, -1
                )
                logits, past, _ = self._forward_backbone(
                    emb,
                    t1,
                    step_mask,
                    past_key_values=past,
                    use_cache=True,
                    position_ids=pos,
                )
                prev = logits[:, -1, :].float()
                cache_key_bias = torch.cat(
                    [cache_key_bias, torch.zeros(B, 1, device=device, dtype=dtype)],
                    dim=1,
                )
                gen_pos += 1
        finally:
            if orig_attn is not None:
                self.backbone.config._attn_implementation = orig_attn

        out_list = []
        for b in range(B):
            real_prompt = prompt_ids[b][attention_mask[b].bool()].tolist()
            out_list.append(real_prompt + gen_tokens[b])
        return out_list, {
            "mode": "ar-batched", "generated": n, "batch": B,
            "forwards": n,
            "per_seq_forwards": [len(tokens) for tokens in gen_tokens],
        }

    @staticmethod
    def _debug_block_row(committed, blk_tok, tokenizer):
        """One line showing the current block state: committed positions show
        their token, uncommitted positions show [noise]."""
        cells = []
        for i, tok in enumerate(blk_tok):
            if committed[i]:
                cells.append(
                    tokenizer.decode([tok]) if tokenizer is not None else f"<{tok}>"
                )
            else:
                cells.append("[noise]")
        return "".join(cells)

    @staticmethod
    def _make_frame(gen_ids, prompt_len, committed, blk_tok):
        """Snapshot of the full sequence at one decoding step, for visualization.

        Returns a list of (token_id_or_None, state) where state is one of
        'prompt' (context), 'clean' (already-finalized blocks), 'active'
        (committed in the current block) or 'noise' (uncommitted in the
        current block). Future, not-yet-started blocks are omitted.
        """
        frame = [
            (tid, "prompt" if j < prompt_len else "clean")
            for j, tid in enumerate(gen_ids)
        ]
        for i, tid in enumerate(blk_tok):
            frame.append((tid, "active") if committed[i] else (None, "noise"))
        return frame

    @torch.no_grad()
    def generate_confidence(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 128,
        order: str = "confidence",
        num_steps: int = 1,
        grid_mode: str = "warp",
        t_max: float = 0.2,
        temperature: float = 0.0,
        seed: int = 42,
        commit_per_step: int = 1,
        ode_argmax: bool = False,
        fixed_block_noise: bool = False,
        threshold: float = 0.0,
        cond_t: float = 1.0,
        sde_alpha: float = 0.0,
        sde_gamma: float = 0.0,
        sde_vp: bool = False,
        sde_noiseless: bool = False,
        euler_alpha: float = 1.0,
        debug: bool = False,
        tokenizer=None,
        on_step=None,
    ):
        """Commit denoised tokens by confidence or order; assumes logit_shift=False."""
        if commit_per_step < 1:
            raise ValueError("commit_per_step must be positive")
        device = next(self.backbone.parameters()).device
        dtype = self.embed_tokens.weight.dtype
        torch.manual_seed(seed)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        eos_id = self.backbone.config.eos_token_id
        time_grid = self._get_generation_time_grid(
            num_steps, device, grid_mode=grid_mode, t_max=t_max
        )
        per_pos = time_grid.dim() == 2
        n_steps = (time_grid.shape[-1] - 1) if per_pos else (len(time_grid) - 1)
        E = self.embed_tokens.weight

        orig_attn = getattr(self.backbone.config, "_attn_implementation", None)
        self.backbone.config._attn_implementation = "eager"

        gen_ids = prompt_ids[0].tolist()
        tokens_generated = 0
        total_forward = 0
        try:
            prompt_embeds = self.embed_tokens(prompt_ids)
            t_bn_p = self._compute_t_bn(
                torch.ones(1, prompt_len, device=device, dtype=dtype)
            )
            _, past, _ = self._forward_backbone(
                prompt_embeds, t_bn_p, attn_mask=None, use_cache=True
            )

            while tokens_generated < max_new_tokens:
                block_start_pos = len(gen_ids)
                bs = min(self.bd_size, max_new_tokens - tokens_generated)
                block_pos_ids = torch.arange(
                    block_start_pos, block_start_pos + bs, device=device
                ).unsqueeze(0)
                committed = torch.zeros(1, bs, dtype=torch.bool, device=device)
                blk_tok = torch.zeros(1, bs, dtype=torch.long, device=device)
                # Optionally reuse the same noise across commit rounds.
                blk_noise_fixed = (
                    self._sample_noise(
                        self.embed_tokens(
                            torch.zeros(1, bs, dtype=torch.long, device=device)
                        )
                    )
                    if fixed_block_noise
                    else None
                )
                if debug:
                    print(f"  [block @ pos {block_start_pos}]")
                    print(
                        "    "
                        + self._debug_block_row(
                            committed[0].tolist(), blk_tok[0].tolist(), tokenizer
                        )
                    )
                if on_step is not None:
                    on_step(
                        self._make_frame(
                            gen_ids,
                            prompt_len,
                            committed[0].tolist(),
                            blk_tok[0].tolist(),
                        )
                    )

                while not bool(committed.all()):
                    cache_len = past.get_seq_length()
                    if self.noisy_causal:
                        block_attn_mask = None
                    else:
                        block_attn_mask = torch.zeros(
                            1, 1, bs, cache_len + bs, dtype=dtype, device=device
                        )
                    clean_emb = self.embed_tokens(blk_tok)
                    cmask = committed.unsqueeze(-1)
                    e_t = (
                        blk_noise_fixed.clone()
                        if fixed_block_noise
                        else self._sample_noise(
                            self.embed_tokens(
                                torch.zeros(1, bs, dtype=torch.long, device=device)
                            )
                        )
                    )

                    def time_for(t_val):
                        # Committed positions stay at cond_t.
                        if torch.is_tensor(t_val):
                            noisy_t = t_val.to(device=device, dtype=dtype).reshape(
                                1, bs
                            )
                        else:
                            noisy_t = torch.full(
                                (1, bs), t_val, device=device, dtype=dtype
                            )
                        return self._compute_t_bn(
                            torch.where(
                                committed,
                                torch.full((1, bs), cond_t, device=device, dtype=dtype),
                                noisy_t,
                            )
                        )

                    e_self_cond = None
                    if self.enable_self_cond:
                        e_in = torch.where(cmask, clean_emb, e_t)
                        init_t = time_grid[:bs, 0] if per_pos else time_grid[0].item()
                        init_logits, _, _ = self._forward_backbone(
                            e_in,
                            time_for(init_t),
                            block_attn_mask,
                            past_key_values=past,
                            use_cache=False,
                            position_ids=block_pos_ids,
                        )
                        trim_kv_cache(past, past.get_seq_length() - cache_len)
                        total_forward += 1
                        init_probs = F.softmax(init_logits[0].float(), dim=-1).to(dtype)
                        e_self_cond = (init_probs @ E).unsqueeze(0).detach()

                    logits = None
                    for step in range(n_steps):
                        if per_pos:
                            t_val_vec = time_grid[:bs, step]
                            t_out_vec = time_grid[:bs, step + 1]
                            if sde_alpha > 0.0 or sde_gamma > 0.0:
                                dt_vec = t_out_vec - t_val_vec
                                if sde_alpha > 0.0:
                                    alpha_v = torch.full_like(t_val_vec, sde_alpha)
                                else:
                                    alpha_v = (1.0 - sde_gamma * dt_vec).clamp(0.0, 1.0)
                                alpha_v = torch.where(
                                    t_val_vec > 0.0, alpha_v, torch.ones_like(alpha_v)
                                )
                                if sde_noiseless:
                                    coef_v = torch.zeros_like(
                                        alpha_v
                                    )  # deterministic α-rewind, no injection
                                elif sde_vp:
                                    coef_v = torch.sqrt(
                                        (
                                            (1.0 - alpha_v)
                                            * (
                                                1.0
                                                + alpha_v
                                                - 2.0 * alpha_v * t_val_vec
                                            )
                                        ).clamp(min=0.0)
                                    )
                                else:
                                    coef_v = 1.0 - alpha_v
                                a_b = alpha_v.to(dtype)[None, :, None]  # [1, bs, 1]
                                c_b = coef_v.to(dtype)[None, :, None]
                                noise = (
                                    torch.randn_like(e_t)
                                    * self._noise_sigma
                                    * self.noise_scale_mult
                                )
                                z_fwd = a_b * e_t + c_b * noise
                                t_fwd_vec = alpha_v * t_val_vec
                            else:
                                t_fwd_vec = t_val_vec
                                z_fwd = e_t
                            t_fwd = t_fwd_vec  # tensor → time_for
                        else:
                            t_val = time_grid[step].item()
                            t_out = time_grid[step + 1].item()
                            dt = t_out - t_val
                            if (sde_alpha > 0.0 or sde_gamma > 0.0) and t_val > 0.0:
                                alpha = (
                                    sde_alpha
                                    if sde_alpha > 0.0
                                    else max(0.0, min(1.0, 1.0 - sde_gamma * dt))
                                )
                                t_fwd = alpha * t_val
                                coef = (
                                    0.0
                                    if sde_noiseless
                                    else (
                                        math.sqrt(
                                            max(
                                                0.0,
                                                (1.0 - alpha)
                                                * (1.0 + alpha - 2.0 * alpha * t_val),
                                            )
                                        )
                                        if sde_vp
                                        else (1.0 - alpha)
                                    )
                                )
                                noise = (
                                    torch.randn_like(e_t)
                                    * self._noise_sigma
                                    * self.noise_scale_mult
                                )
                                z_fwd = alpha * e_t + coef * noise
                            else:
                                t_fwd, z_fwd = t_val, e_t
                        e_mix = torch.where(cmask, clean_emb, z_fwd)
                        if self.enable_self_cond:
                            e_mix = torch.where(
                                cmask,
                                clean_emb,
                                self._apply_self_cond(e_mix, e_self_cond),
                            )
                        logits, _, _ = self._forward_backbone(
                            e_mix,
                            time_for(t_fwd),
                            block_attn_mask,
                            past_key_values=past,
                            use_cache=False,
                            position_ids=block_pos_ids,
                        )
                        trim_kv_cache(past, past.get_seq_length() - cache_len)
                        total_forward += 1
                        if tokenizer is not None:
                            preds = logits[0].argmax(dim=-1).tolist()
                            fixed = blk_tok[0].tolist()
                            cells = []
                            for i in range(bs):
                                if committed[0][i]:
                                    s = tokenizer.decode([fixed[i]])
                                    s = (
                                        "["
                                        + s.replace("\n", "\\n").replace("\t", "\\t")
                                        + "]"
                                    )
                                else:
                                    s = (
                                        tokenizer.decode([preds[i]])
                                        .replace("\n", "\\n")
                                        .replace("\t", "\\t")
                                    )
                                cells.append(s)
                            t_disp = t_fwd.mean().item() if per_pos else t_fwd
                            print(
                                f"  step {step:2d} t={t_disp:.3f} | " + " ".join(cells)
                            )
                        if step == n_steps - 1:
                            break
                        probs_raw = F.softmax(logits[0].float(), dim=-1).to(dtype)
                        e_denoised = (
                            E[probs_raw.argmax(-1)].unsqueeze(0)
                            if ode_argmax
                            else (probs_raw @ E).unsqueeze(0)
                        )
                        if self.enable_self_cond:
                            e_self_cond = (probs_raw @ E).unsqueeze(0).detach()
                        # Broadcast position times over embeddings in fp32.
                        if per_pos:
                            tf = t_fwd_vec[None, :, None].float()
                            to = t_out_vec[None, :, None].float()
                            e_t = (
                                z_fwd.float()
                                + (to - tf)
                                * (e_denoised.float() - z_fwd.float())
                                / (1.0 - tf).clamp(min=1e-6)
                            ).to(dtype)
                            if euler_alpha != 1.0:
                                c_v = (to - tf) / (1.0 - tf).clamp(min=1e-6)
                                e_t = (
                                    e_t.float()
                                    - (1.0 - c_v) * (1.0 - euler_alpha) * z_fwd.float()
                                ).to(dtype)
                        else:
                            e_t = z_fwd + (t_out - t_fwd) * (e_denoised - z_fwd) / max(
                                1.0 - t_fwd, 1e-6
                            )
                            if euler_alpha != 1.0:
                                c = (t_out - t_fwd) / max(1.0 - t_fwd, 1e-6)
                                e_t = e_t - (1.0 - c) * (1.0 - euler_alpha) * z_fwd

                    logits = logits[0].float()  # [bs, V]
                    probs = F.softmax(logits, dim=-1)
                    max_probs, argmax_tok = probs.max(dim=-1)
                    if tokenizer is not None:
                        conf = []
                        for i in range(bs):
                            c = f"{max_probs[i].item():.2f}"
                            conf.append(("[" + c + "]") if committed[0][i] else c)
                        print("  confidence    | " + " ".join(conf))
                    max_probs = max_probs.masked_fill(committed[0], float("-inf"))
                    n_remaining = int((~committed[0]).sum().item())
                    k_eff = min(commit_per_step, n_remaining)
                    if order == "greedy":
                        sel = (~committed[0]).nonzero(as_tuple=False).flatten()[:k_eff]
                        sel = sel.tolist()
                    elif threshold > 0.0:
                        # Fall back to top-k if too few positions pass the threshold.
                        sel = (
                            (max_probs > threshold)
                            .nonzero(as_tuple=False)
                            .flatten()
                            .tolist()
                        )
                        if len(sel) < k_eff:
                            sel = torch.topk(max_probs, k_eff).indices.tolist()
                    else:
                        sel = torch.topk(max_probs, k_eff).indices.tolist()
                    for pos in sel:
                        if temperature > 0:
                            tok = int(
                                torch.multinomial(
                                    F.softmax(logits[pos] / temperature, dim=-1), 1
                                ).item()
                            )
                        else:
                            tok = int(argmax_tok[pos].item())
                        blk_tok[0, pos] = tok
                        committed[0, pos] = True
                        if tokenizer is not None:
                            s = (
                                tokenizer.decode([tok])
                                .replace("\n", "\\n")
                                .replace("\t", "\\t")
                            )
                            print(
                                f"  >>> commit pos {pos}  p={max_probs[pos]:.3f}  "
                                f"'{s}'  ({int(committed[0].sum())}/{bs} done)"
                            )
                    if tokenizer is not None:
                        print()
                    if debug:
                        print(
                            "    "
                            + self._debug_block_row(
                                committed[0].tolist(), blk_tok[0].tolist(), tokenizer
                            )
                        )
                    if on_step is not None:
                        on_step(
                            self._make_frame(
                                gen_ids,
                                prompt_len,
                                committed[0].tolist(),
                                blk_tok[0].tolist(),
                            )
                        )

                new_tokens = blk_tok[0].tolist()
                if eos_id is not None and eos_id in new_tokens:
                    new_tokens = new_tokens[: new_tokens.index(eos_id) + 1]
                gen_ids.extend(new_tokens)
                tokens_generated += len(new_tokens)

                accepted_len = len(new_tokens)
                block_embeds = self.embed_tokens(
                    torch.tensor([new_tokens], device=device)
                )
                accepted_pos_ids = torch.arange(
                    block_start_pos, block_start_pos + accepted_len, device=device
                ).unsqueeze(0)
                t_bn_final = self._compute_t_bn(
                    torch.ones(1, accepted_len, device=device, dtype=dtype)
                )
                _, past, _ = self._forward_backbone(
                    block_embeds,
                    t_bn_final,
                    attn_mask=None,
                    past_key_values=past,
                    use_cache=True,
                    position_ids=accepted_pos_ids,
                )
                if eos_id is not None and eos_id in new_tokens:
                    break
        finally:
            if orig_attn is not None:
                self.backbone.config._attn_implementation = orig_attn

        out_ids = torch.tensor([gen_ids], device=device, dtype=torch.long)
        return out_ids, {
            "mode": f"block-anyorder-{order}",
            "generated": tokens_generated,
            "forwards": total_forward,
            "ode_steps": num_steps,
        }

    @torch.no_grad()
    def generate_confidence_batched(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int = 128,
        order: str = "confidence",
        num_steps: int = 1,
        grid_mode: str = "warp",
        t_max: float = 0.2,
        temperature: float = 0.0,
        seed: int = 42,
        commit_per_step: int = 1,
        ode_argmax: bool = False,
        fixed_block_noise: bool = False,
        threshold: float = 0.0,
        cond_t: float = 1.0,
        sde_alpha: float = 0.0,
        sde_gamma: float = 0.0,
        sde_vp: bool = False,
        sde_noiseless: bool = False,
        euler_alpha: float = 1.0,
    ):
        """Decode left-padded prompts in parallel with a rectangular KV cache."""
        if commit_per_step < 1:
            raise ValueError("commit_per_step must be positive")
        device = next(self.backbone.parameters()).device
        dtype = self.embed_tokens.weight.dtype
        torch.manual_seed(seed)
        # Share one noise stream across rows so decoding is batch invariant.
        noise_gen = torch.Generator(device=device).manual_seed(seed)
        prompt_ids = prompt_ids.to(device)
        attention_mask = attention_mask.to(device)
        B, L = prompt_ids.shape
        eos_id = self.backbone.config.eos_token_id
        NEG = torch.finfo(dtype).min
        time_grid = self._get_generation_time_grid(
            num_steps, device, grid_mode=grid_mode, t_max=t_max
        )
        per_pos = time_grid.dim() == 2
        n_steps = (time_grid.shape[-1] - 1) if per_pos else (len(time_grid) - 1)
        E = self.embed_tokens.weight

        def shared_noise(width):
            """One standalone-equivalent RNG draw, broadcast across batch rows."""
            return (
                (
                    torch.randn(
                        1,
                        width,
                        self._d_model,
                        generator=noise_gen,
                        device=device,
                        dtype=dtype,
                    )
                    * self._noise_sigma
                    * self.noise_scale_mult
                )
                .expand(B, width, self._d_model)
                .contiguous()
            )

        orig_attn = getattr(self.backbone.config, "_attn_implementation", None)
        self.backbone.config._attn_implementation = "eager"

        cache_key_bias = (1.0 - attention_mask.to(dtype)) * NEG  # [B, L]

        gen_tokens = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        total_forward = 0
        # Count forwards only while each row is active.
        row_fwd = torch.zeros(B, device=device)
        tokens_generated = 0
        try:
            prompt_embeds = self.embed_tokens(prompt_ids)
            t_bn_p = self._compute_t_bn(torch.ones(B, L, device=device, dtype=dtype))
            # Mask padding before it enters the KV cache.
            causal = torch.triu(
                torch.full((L, L), NEG, device=device, dtype=dtype), diagonal=1
            )
            prefill_mask = causal[None, None] + cache_key_bias[:, None, None, :]
            _, past, _ = self._forward_backbone(
                prompt_embeds, t_bn_p, prefill_mask, use_cache=True
            )

            gen_pos = L
            while tokens_generated < max_new_tokens and not bool(finished.all()):
                bs = min(self.bd_size, max_new_tokens - tokens_generated)
                block_pos_ids = torch.arange(gen_pos, gen_pos + bs, device=device)[
                    None
                ].expand(B, -1)

                committed = torch.zeros(B, bs, dtype=torch.bool, device=device)
                blk_tok = torch.zeros(B, bs, dtype=torch.long, device=device)
                blk_noise_fixed = shared_noise(bs) if fixed_block_noise else None

                while not bool(committed.all()):
                    active = ((~committed.all(dim=1)) & (~finished)).to(row_fwd.dtype)
                    cache_len = past.get_seq_length()
                    block_mask = torch.zeros(
                        B, 1, bs, cache_len + bs, dtype=dtype, device=device
                    )
                    block_mask[:, :, :, :cache_len] = cache_key_bias[:, None, None, :]
                    if self.noisy_causal:
                        cw = torch.triu(
                            torch.full((bs, bs), NEG, device=device, dtype=dtype),
                            diagonal=1,
                        )
                        block_mask[:, :, :, cache_len:] = cw[None, None]
                    clean_emb = self.embed_tokens(blk_tok)
                    cmask = committed.unsqueeze(-1)
                    e_t = (
                        blk_noise_fixed.clone()
                        if fixed_block_noise
                        else shared_noise(bs)
                    )

                    def time_for(t_val):
                        # Committed positions stay at cond_t.
                        if torch.is_tensor(t_val):
                            noisy_t = (
                                t_val.to(device=device, dtype=dtype)
                                .reshape(1, bs)
                                .expand(B, bs)
                            )
                        else:
                            noisy_t = torch.full(
                                (B, bs), t_val, device=device, dtype=dtype
                            )
                        return self._compute_t_bn(
                            torch.where(
                                committed,
                                torch.full((B, bs), cond_t, device=device, dtype=dtype),
                                noisy_t,
                            )
                        )

                    e_self_cond = None
                    if self.enable_self_cond:
                        e_in = torch.where(cmask, clean_emb, e_t)
                        init_t = time_grid[:bs, 0] if per_pos else time_grid[0].item()
                        init_logits, _, _ = self._forward_backbone(
                            e_in,
                            time_for(init_t),
                            block_mask,
                            past_key_values=past,
                            use_cache=False,
                            position_ids=block_pos_ids,
                        )
                        trim_kv_cache(past, past.get_seq_length() - cache_len)
                        total_forward += 1
                        row_fwd += active
                        init_probs = F.softmax(init_logits.float(), dim=-1).to(dtype)
                        e_self_cond = (init_probs @ E).detach()

                    logits = None
                    for step in range(n_steps):
                        if per_pos:
                            t_val_vec = time_grid[:bs, step]
                            t_out_vec = time_grid[:bs, step + 1]
                            if sde_alpha > 0.0 or sde_gamma > 0.0:
                                dt_vec = t_out_vec - t_val_vec
                                if sde_alpha > 0.0:
                                    alpha_v = torch.full_like(t_val_vec, sde_alpha)
                                else:
                                    alpha_v = (1.0 - sde_gamma * dt_vec).clamp(0.0, 1.0)
                                alpha_v = torch.where(
                                    t_val_vec > 0.0, alpha_v, torch.ones_like(alpha_v)
                                )
                                if sde_noiseless:
                                    coef_v = torch.zeros_like(
                                        alpha_v
                                    )  # deterministic α-rewind, no injection
                                elif sde_vp:
                                    coef_v = torch.sqrt(
                                        (
                                            (1.0 - alpha_v)
                                            * (
                                                1.0
                                                + alpha_v
                                                - 2.0 * alpha_v * t_val_vec
                                            )
                                        ).clamp(min=0.0)
                                    )
                                else:
                                    coef_v = 1.0 - alpha_v
                                a_b = alpha_v.to(dtype)[
                                    None, :, None
                                ]  # [1, bs, 1] → bcast over B,d
                                c_b = coef_v.to(dtype)[None, :, None]
                                noise = (
                                    torch.randn_like(e_t)
                                    * self._noise_sigma
                                    * self.noise_scale_mult
                                )
                                z_fwd = a_b * e_t + c_b * noise
                                t_fwd_vec = alpha_v * t_val_vec
                            else:
                                t_fwd_vec = t_val_vec
                                z_fwd = e_t
                            t_fwd = t_fwd_vec
                        else:
                            t_val = time_grid[step].item()
                            t_out = time_grid[step + 1].item()
                            dt = t_out - t_val
                            if (sde_alpha > 0.0 or sde_gamma > 0.0) and t_val > 0.0:
                                alpha = (
                                    sde_alpha
                                    if sde_alpha > 0.0
                                    else max(0.0, min(1.0, 1.0 - sde_gamma * dt))
                                )
                                t_fwd = alpha * t_val
                                coef = (
                                    0.0
                                    if sde_noiseless
                                    else (
                                        math.sqrt(
                                            max(
                                                0.0,
                                                (1.0 - alpha)
                                                * (1.0 + alpha - 2.0 * alpha * t_val),
                                            )
                                        )
                                        if sde_vp
                                        else (1.0 - alpha)
                                    )
                                )
                                noise = (
                                    torch.randn_like(e_t)
                                    * self._noise_sigma
                                    * self.noise_scale_mult
                                )
                                z_fwd = alpha * e_t + coef * noise
                            else:
                                t_fwd, z_fwd = t_val, e_t
                        e_mix = torch.where(cmask, clean_emb, z_fwd)
                        if self.enable_self_cond:
                            e_mix = torch.where(
                                cmask,
                                clean_emb,
                                self._apply_self_cond(e_mix, e_self_cond),
                            )
                        logits, _, _ = self._forward_backbone(
                            e_mix,
                            time_for(t_fwd),
                            block_mask,
                            past_key_values=past,
                            use_cache=False,
                            position_ids=block_pos_ids,
                        )
                        trim_kv_cache(past, past.get_seq_length() - cache_len)
                        total_forward += 1
                        row_fwd += active
                        probs_raw = F.softmax(logits.float(), dim=-1).to(dtype)
                        e_denoised = (
                            E[probs_raw.argmax(-1)] if ode_argmax else probs_raw @ E
                        )
                        if self.enable_self_cond:
                            e_self_cond = (probs_raw @ E).detach()
                        if step == n_steps - 1:
                            break
                        # Broadcast position times over embeddings in fp32.
                        if per_pos:
                            tf = t_fwd_vec[None, :, None].float()
                            to = t_out_vec[None, :, None].float()
                            e_t = (
                                z_fwd.float()
                                + (to - tf)
                                * (e_denoised.float() - z_fwd.float())
                                / (1.0 - tf).clamp(min=1e-6)
                            ).to(dtype)
                            if euler_alpha != 1.0:
                                c_v = (to - tf) / (1.0 - tf).clamp(min=1e-6)
                                e_t = (
                                    e_t.float()
                                    - (1.0 - c_v) * (1.0 - euler_alpha) * z_fwd.float()
                                ).to(dtype)
                        else:
                            e_t = z_fwd + (t_out - t_fwd) * (e_denoised - z_fwd) / max(
                                1.0 - t_fwd, 1e-6
                            )
                            if euler_alpha != 1.0:
                                c = (t_out - t_fwd) / max(1.0 - t_fwd, 1e-6)
                                e_t = e_t - (1.0 - c) * (1.0 - euler_alpha) * z_fwd

                    logits = logits.float()  # [B, bs, V]
                    probs = F.softmax(logits, dim=-1)
                    max_probs, argmax_tok = probs.max(dim=-1)  # [B, bs]
                    max_probs = max_probs.masked_fill(committed, float("-inf"))
                    kk = min(commit_per_step, bs)
                    if order == "greedy":
                        n_rem = bs - int(committed[0].sum().item())  # greedy = lockstep
                        start = bs - n_rem
                        commit_now = torch.zeros(B, bs, dtype=torch.bool, device=device)
                        commit_now[:, start : start + min(kk, n_rem)] = True
                    elif threshold > 0.0:
                        commit_now = (max_probs > threshold) & (~committed)
                        need = commit_now.sum(1) < kk  # rows short of kk
                        if bool(need.any()):
                            fb = torch.zeros_like(commit_now)
                            fb.scatter_(
                                1, torch.topk(max_probs, kk, dim=1).indices, True
                            )
                            fb = fb & (~committed)
                            commit_now = torch.where(
                                need[:, None], commit_now | fb, commit_now
                            )
                    else:
                        commit_now = torch.zeros(B, bs, dtype=torch.bool, device=device)
                        commit_now.scatter_(
                            1, torch.topk(max_probs, kk, dim=1).indices, True
                        )
                        commit_now = commit_now & (~committed)
                    if temperature > 0:
                        toks_all = torch.multinomial(
                            F.softmax(logits / temperature, dim=-1).reshape(
                                -1, logits.shape[-1]
                            ),
                            1,
                            generator=noise_gen,
                        ).reshape(B, bs)
                    else:
                        toks_all = argmax_tok
                    blk_tok = torch.where(commit_now, toks_all, blk_tok)
                    committed = committed | commit_now

                blk_list = blk_tok.tolist()
                for b in range(B):
                    if finished[b]:
                        continue
                    toks = blk_list[b]
                    if eos_id is not None and eos_id in toks:
                        toks = toks[: toks.index(eos_id) + 1]
                        finished[b] = True
                    gen_tokens[b].extend(toks)
                tokens_generated += bs

                # Re-encode every row to keep the cache rectangular.
                block_embeds = self.embed_tokens(blk_tok)
                t_bn_final = self._compute_t_bn(
                    torch.ones(B, bs, device=device, dtype=dtype)
                )
                re_cache_len = past.get_seq_length()
                recode_mask = torch.zeros(
                    B, 1, bs, re_cache_len + bs, dtype=dtype, device=device
                )
                recode_mask[:, :, :, :re_cache_len] = cache_key_bias[:, None, None, :]
                cw = torch.triu(
                    torch.full((bs, bs), NEG, device=device, dtype=dtype), diagonal=1
                )
                recode_mask[:, :, :, re_cache_len:] = cw[None, None]
                _, past, _ = self._forward_backbone(
                    block_embeds,
                    t_bn_final,
                    recode_mask,
                    past_key_values=past,
                    use_cache=True,
                    position_ids=block_pos_ids,
                )
                cache_key_bias = torch.cat(
                    [cache_key_bias, torch.zeros(B, bs, device=device, dtype=dtype)],
                    dim=1,
                )
                gen_pos += bs
        finally:
            if orig_attn is not None:
                self.backbone.config._attn_implementation = orig_attn

        out_list = []
        for b in range(B):
            real_prompt = prompt_ids[b][attention_mask[b].bool()].tolist()
            out_list.append(real_prompt + gen_tokens[b])
        stats = {
            "mode": f"block-anyorder-{order}-batched",
            "generated": tokens_generated,
            "forwards": total_forward,
            "per_seq_forwards": [int(x) for x in row_fwd.tolist()],
            "per_seq_tokens": [len(gen_tokens[b]) for b in range(B)],
            "ode_steps": num_steps,
            "batch": B,
        }
        return out_list, stats

    def _denoise_flowmap_block(self, past, positions, grid, mask, **kwargs):
        """Two-time transitions; terminal prediction uses the vocabulary head."""
        dtype, device = self.embed_tokens.weight.dtype, positions.device
        length = positions.shape[1]
        state = torch.randn(1, length, self._d_model, device=device, dtype=dtype)
        state = state * self._noise_sigma * self.noise_scale_mult
        path = (grid.to(dtype).view(-1, 1, 1).expand(-1, 1, length)
                if grid.ndim == 1 else grid[:length].to(dtype).T.unsqueeze(1))
        cache_len = past.get_seq_length()
        for step, (source, target) in enumerate(zip(path[:-1], path[1:])):
            condition = self._compose_flowmap_time_condition(source, target)
            logits, _, _ = self._forward_backbone(
                state, condition, mask, past_key_values=past,
                use_cache=False, position_ids=positions)
            trim_kv_cache(past, past.get_seq_length() - cache_len)
            if step == len(path) - 2:
                return logits.float(), len(path) - 1
            delta = F.softmax(logits.float(), dim=-1).to(dtype) @ self.embed_tokens.weight
            if grid.ndim == 1:
                s, t = grid[step].item(), grid[step + 1].item()
                updated = state + (t - s) * (delta - state) / max(1 - s, 1e-6)
            else:
                s, t = source.float().unsqueeze(-1), target.float().unsqueeze(-1)
                updated = (state.float() + (t-s) * (delta.float()-state.float())
                           / (1-s).clamp_min(1e-6)).to(dtype)
            state = torch.where((target > source + 1e-8).unsqueeze(-1), updated, state)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 128,
        num_steps: int = 64,
        threshold: float = 0.0,
        temperature: float = 0.0,
        seed: int = 42,
        grid_mode: str = "uniform_range",
        t_max: float = 0.2,
        ode_argmax: bool = False,
        sde_gamma: float = 0.0,
        sde_alpha: float = 0.0,
        sde_vp: bool = False,
        sde_noiseless: bool = False,
        euler_alpha: float = 1.0,
        cfg_scale: float = 1.0,
    ):
        """Decode blocks with Euler steps and a KV-cached clean prefix."""
        if self._two_time and (cfg_scale != 1 or ode_argmax or sde_gamma or sde_alpha or sde_vp or sde_noiseless or euler_alpha != 1):
            raise ValueError("Student flow-map decoding requires unguided deterministic updates")
        device = next(self.backbone.parameters()).device
        dtype = self.embed_tokens.weight.dtype
        torch.manual_seed(seed)

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        eos_id = self.backbone.config.eos_token_id

        time_grid = self._get_generation_time_grid(
            num_steps, device, grid_mode=grid_mode, t_max=t_max
        )

        # Eager attention supports the dense block mask.
        orig_attn = getattr(self.backbone.config, "_attn_implementation", None)
        self.backbone.config._attn_implementation = "eager"

        gen_ids = prompt_ids[0].tolist()
        tokens_generated = 0
        total_forward = 0
        # Logical steps exclude CFG and self-conditioning bootstrap forwards.
        total_logical = 0

        try:
            prompt_embeds = self.embed_tokens(prompt_ids)  # [1, P, d]
            t_bn_prompt = self._compute_t_bn(
                torch.ones(1, prompt_len, device=device, dtype=dtype)
            )
            prompt_logits, past_key_values, _ = self._forward_backbone(
                prompt_embeds,
                t_bn_prompt,
                attn_mask=None,
                use_cache=True,
            )
            if prompt_logits.dim() == 2:  # some envs return [L, V]
                prompt_logits = prompt_logits.unsqueeze(0)
            prev_logit = prompt_logits[0, -1:, :]  # [1, V]

            # Only shifted logits make the prefix logit a next-token prediction.
            ar_mode = self.bd_size == 1 and num_steps == 1 and self.logit_shift

            while tokens_generated < max_new_tokens:
                block_start_pos = len(gen_ids)

                if ar_mode:
                    total_forward += 1
                    total_logical += 1
                    final_logits = prev_logit[0].float()  # [V]
                    if temperature > 0:
                        final_probs = F.softmax(final_logits / temperature, dim=-1)
                        tok = torch.multinomial(final_probs.unsqueeze(0), 1).item()
                    else:
                        tok = final_logits.argmax().item()
                    new_tokens = [tok]
                else:
                    block_pos_ids = torch.arange(
                        block_start_pos,
                        block_start_pos + self.bd_size,
                        device=device,
                    ).unsqueeze(0)

                    cache_len = past_key_values.get_seq_length()
                    if self.noisy_causal:
                        block_attn_mask = None
                    else:
                        block_attn_mask = torch.zeros(
                            1,
                            1,
                            self.bd_size,
                            cache_len + self.bd_size,
                            dtype=dtype,
                            device=device,
                        )

                    denoise = self._denoise_flowmap_block if self._two_time else self._euler_denoise_block
                    logits, num_fwd = denoise(
                        past_key_values,
                        block_pos_ids,
                        time_grid,
                        block_attn_mask,
                        prev_logit=prev_logit if self.logit_shift else None,
                        ode_argmax=ode_argmax,
                        sde_gamma=sde_gamma,
                        sde_alpha=sde_alpha,
                        sde_vp=sde_vp,
                        sde_noiseless=sde_noiseless,
                        euler_alpha=euler_alpha,
                        cfg_scale=cfg_scale,
                    )
                    total_forward += num_fwd
                    total_logical += num_steps

                    if self.logit_shift:
                        logits = torch.cat([prev_logit.unsqueeze(0), logits], dim=1)
                    remaining = max_new_tokens - tokens_generated
                    final_logits = logits[0, : min(self.bd_size, remaining)].float()
                    if temperature > 0:
                        final_probs = F.softmax(final_logits / temperature, dim=-1)
                        preds = torch.multinomial(final_probs, 1).squeeze(-1).tolist()
                    else:
                        final_probs = F.softmax(final_logits, dim=-1)
                        preds = final_logits.argmax(dim=-1).tolist()

                    # Keep the longest prefix above the confidence threshold.
                    if threshold > 0.0:
                        confidences = final_probs.max(dim=-1).values
                        new_tokens = [preds[0]]
                        for j in range(1, len(preds)):
                            if confidences[j].item() >= threshold:
                                new_tokens.append(preds[j])
                            else:
                                break
                    else:
                        new_tokens = preds

                if eos_id is not None and eos_id in new_tokens:
                    new_tokens = new_tokens[: new_tokens.index(eos_id) + 1]

                gen_ids.extend(new_tokens)
                tokens_generated += len(new_tokens)
                if tokens_generated >= max_new_tokens or (
                    eos_id is not None and eos_id in new_tokens
                ):
                    break

                block_ids = torch.tensor([new_tokens], device=device)
                block_embeds = self.embed_tokens(block_ids)
                accepted_len = len(new_tokens)
                accepted_pos_ids = torch.arange(
                    block_start_pos,
                    block_start_pos + accepted_len,
                    device=device,
                ).unsqueeze(0)
                t_bn_final = self._compute_t_bn(
                    torch.ones(1, accepted_len, device=device, dtype=dtype)
                )
                re_logits, past_key_values, _ = self._forward_backbone(
                    block_embeds,
                    t_bn_final,
                    attn_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=accepted_pos_ids,
                )
                prev_logit = re_logits[0, -1:, :]

        finally:
            if orig_attn is not None:
                self.backbone.config._attn_implementation = orig_attn

        generated = gen_ids[prompt_len : prompt_len + max_new_tokens]
        result = torch.tensor(
            [gen_ids[:prompt_len] + generated], device=device, dtype=torch.long
        )
        return result, {
            "threshold": threshold,
            "total_steps": total_forward,
            "logical_steps": total_logical,
            "gen_len": len(generated),
            "hit_eos": eos_id is not None and eos_id in generated,
            "tokens_per_step": tokens_generated / max(total_forward, 1),
            "logical_tokens_per_step": tokens_generated / max(total_logical, 1),
            "block_size": self.bd_size,
            "num_euler_steps": num_steps,
        }

    @torch.no_grad()
    def generate_self_speculative(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 128,
        seed: int = 42,
        grid_mode: str = "uniform_range",
        t_max: float = 0.2,
        self_cond: bool = False,
    ):
        """Self-speculative single-step decoding (batch size 1).

        Each round drafts N = bd_size tokens with a bidirectional block. The
        first proposal is checked against the prefix's next-token prediction.
        On a miss, a compact correction pass commits that token and drafts the
        next block. On a hit, a fused pass verifies the remaining proposals
        and drafts a successor block for every possible accepted prefix. Useful
        K/V is retained by index selection.

        Layout of the fused forward (after the committed prefix, which lives in
        the KV cache):

            d_0 [noise a_0]  d_1 [noise a_1]  ...  d_{N-1} [noise a_{N-1}]

        Mask rules (each row also sees the whole prefix):
          - draft token d_i: sees d_0..d_i (causal over drafts, incl. itself), no
            noise. Its clean next-token logit verifies d_{i+1}.
          - noise block a_i (N slots): sees d_0..d_i and its own slots
            (bidirectional). It is the next-block draft assuming d_0..d_i accepted.
          - recovery correction c sees only prefix + itself; its noise block
            sees prefix + c + all of its own bidirectional slots.

        Verification is greedy. Decoding uses one flow step and no SDE.
        Optional self-conditioning adds a bootstrap pass before each draft.

        Returns (output_ids [1, P+gen_len], stats_dict).
        """
        device = next(self.backbone.parameters()).device
        dtype = self.embed_tokens.weight.dtype
        torch.manual_seed(seed)

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        eos_id = self.backbone.config.eos_token_id
        N = self.bd_size
        d = self._d_model
        E = self.embed_tokens.weight
        # Draft self-conditioning adds a bootstrap forward; verification stays causal.
        sc_on = bool(self_cond and getattr(self, "enable_self_cond", False))

        tg = self._get_generation_time_grid(1, device, grid_mode=grid_mode, t_max=t_max)
        if tg.dim() == 2:  # per-position warp grid [N, 2]
            t_start = tg[:, 0].to(dtype)  # [N] block-local start times
            t_target = tg[:, 1].to(dtype)  # [N] flow-map target times
        else:
            t_start = tg[0].to(dtype).expand(N).clone()
            t_target = tg[1].to(dtype).expand(N).clone()

        orig_attn = getattr(self.backbone.config, "_attn_implementation", None)
        self.backbone.config._attn_implementation = "eager"

        gen_ids = prompt_ids[0].tolist()
        committed = 0
        fused_fwd = boot_fwd = corr_fwd = 0
        first_token_rejects = 0

        def _noise(n):
            return (
                torch.randn(1, n, d, device=device, dtype=dtype)
                * self._noise_sigma
                * self.noise_scale_mult
            )

        def _draft_from_cache(past, base_pos):
            """Bidirectional N-slot draft over the whole cache. Does NOT extend the
            cache. Returns (draft token id list, num_forwards). With self-cond a
            bootstrap forward first estimates the clean embedding (2 forwards)."""
            cache_len = past.get_seq_length()
            e = _noise(N)
            pos = torch.arange(base_pos, base_pos + N, device=device).unsqueeze(0)
            mask = torch.zeros(1, 1, N, cache_len + N, dtype=dtype, device=device)
            source_t = t_start.unsqueeze(0)
            t_bn = self._compose_flowmap_time_condition(
                source_t,
                t_target.unsqueeze(0),
            )
            adaln_modulation = self._precompute_shared_adaln(t_bn)
            nf = 0
            if sc_on:
                lb, _, _ = self._forward_backbone(
                    e,
                    t_bn,
                    mask,
                    past_key_values=past,
                    use_cache=False,
                    position_ids=pos,
                    adaln_modulation=adaln_modulation,
                )
                trim_kv_cache(past, past.get_seq_length() - cache_len)
                nf += 1
                e_sc = (F.softmax(lb[0].float(), -1).to(dtype) @ E).unsqueeze(0)
                e = self._apply_self_cond(e, e_sc)
            logits, _, _ = self._forward_backbone(
                e,
                t_bn,
                mask,
                past_key_values=past,
                use_cache=False,
                position_ids=pos,
                adaln_modulation=adaln_modulation,
            )
            trim_kv_cache(past, past.get_seq_length() - cache_len)
            nf += 1
            return logits[0].argmax(-1).tolist(), nf

        # ---- static fused-layout index maps (depend only on N) ----
        S = N * (N + 1)
        draft_idx = [i * (N + 1) for i in range(N)]  # d_i fused index
        noise_pos = [draft_idx[i] + 1 + k for i in range(N) for k in range(N)]
        blk = [0] * S
        isd = [False] * S
        for i in range(N):
            blk[draft_idx[i]] = i
            isd[draft_idx[i]] = True
            for k in range(N):
                blk[draft_idx[i] + 1 + k] = i
        blk_t = torch.tensor(blk, device=device)
        isd_t = torch.tensor(isd, device=device)
        # allowed_fused[q,k]: draft-key k visible iff its block <= q's block;
        # noise-key k visible iff q is a noise slot in the SAME block.
        kd = isd_t[None, :]
        allowed_fused = (kd & (blk_t[None, :] <= blk_t[:, None])) | (
            (~kd) & (~isd_t)[:, None] & (blk_t[None, :] == blk_t[:, None])
        )
        # per-position t: 1.0 at drafts, t_start[k] at noise slot k of each block
        ppt = torch.ones(1, S, device=device, dtype=dtype)
        target_ppt = torch.ones(1, S, device=device, dtype=dtype)
        target_mask = torch.zeros(1, S, device=device, dtype=torch.bool)
        for i in range(N):
            block_noise = [draft_idx[i] + 1 + k for k in range(N)]
            ppt[0, block_noise] = t_start
            target_ppt[0, block_noise] = t_target
            target_mask[0, block_noise] = True
        # relative position ids (absolute = rel + committed length P)
        rel_pos = torch.zeros(S, dtype=torch.long, device=device)
        for i in range(N):
            rel_pos[draft_idx[i]] = i
            for k in range(N):
                rel_pos[draft_idx[i] + 1 + k] = i + 1 + k
        neg = torch.full((), float("-inf"), device=device, dtype=dtype)
        zero = torch.zeros((), device=device, dtype=dtype)
        fused_allow = torch.where(allowed_fused, zero, neg)  # [S, S] additive

        R = N + 1
        recovery_ppt = torch.cat(
            (torch.ones(1, 1, device=device, dtype=dtype), t_start.unsqueeze(0)),
            dim=1,
        )
        recovery_target_ppt = torch.cat(
            (torch.ones(1, 1, device=device, dtype=dtype), t_target.unsqueeze(0)),
            dim=1,
        )
        recovery_target_mask = torch.zeros(1, R, device=device, dtype=torch.bool)
        recovery_target_mask[:, 1:] = True
        recovery_t_bn = self._compose_flowmap_time_condition(
            recovery_ppt,
            recovery_target_ppt,
            target_mask=recovery_target_mask,
        )
        recovery_adaln = self._precompute_shared_adaln(recovery_t_bn)
        recovery_allow = torch.zeros(R, R, device=device, dtype=dtype)
        recovery_allow[0, 1:] = float("-inf")
        recovery_rel_pos = torch.arange(R, device=device)

        def _recover_and_draft(past, base_pos, correction):
            """Commit a rejected first draft token and propose its successor."""
            cache_len = past.get_seq_length()
            assert cache_len == base_pos
            emb = torch.empty(1, R, d, device=device, dtype=dtype)
            emb[:, :1] = self.embed_tokens(
                torch.tensor([[correction]], device=device)
            ).to(dtype)
            emb[:, 1:] = _noise(N)
            mask = torch.zeros(R, cache_len + R, device=device, dtype=dtype)
            mask[:, cache_len:] = recovery_allow
            pos = (recovery_rel_pos + base_pos).unsqueeze(0)
            nf = 0

            if sc_on:
                lb, _, _ = self._forward_backbone(
                    emb,
                    recovery_t_bn,
                    mask[None, None],
                    past_key_values=past,
                    use_cache=False,
                    position_ids=pos,
                    adaln_modulation=recovery_adaln,
                )
                trim_kv_cache(past, past.get_seq_length() - cache_len)
                nf += 1
                e_sc = (F.softmax(lb[0, 1:].float(), -1).to(dtype) @ E).unsqueeze(0)
                emb[:, 1:] = self._apply_self_cond(emb[:, 1:], e_sc)

            logits, past, _ = self._forward_backbone(
                emb,
                recovery_t_bn,
                mask[None, None],
                past_key_values=past,
                use_cache=True,
                position_ids=pos,
                adaln_modulation=recovery_adaln,
            )
            nf += 1
            past = gather_kv_cache(past, torch.arange(cache_len + 1, device=device))
            return past, logits[0, 0], logits[0, 1:].argmax(-1).tolist(), nf

        try:
            # Prefill prompt (clean, causal) → cache + prev_logit predicting d_0.
            pe = self.embed_tokens(prompt_ids)
            t_bn_p = self._compute_t_bn(
                torch.ones(1, prompt_len, device=device, dtype=dtype)
            )
            prompt_adaln = self._precompute_shared_adaln(t_bn_p)
            pl, past, _ = self._forward_backbone(
                pe,
                t_bn_p,
                attn_mask=None,
                use_cache=True,
                adaln_modulation=prompt_adaln,
            )
            prev_logit = pl[0, -1, :]  # predicts next token = d_0
            # The prompt modulation can be much larger than the static decode
            # layouts and is never reused after prefill.
            del prompt_adaln, t_bn_p, pe

            D, nf = _draft_from_cache(past, past.get_seq_length())  # bootstrap draft
            boot_fwd += nf
            t_bn_fused = self._compose_flowmap_time_condition(
                ppt,
                target_ppt,
                target_mask=target_mask,
            )
            fused_adaln = self._precompute_shared_adaln(t_bn_fused)

            while committed < max_new_tokens:
                P = past.get_seq_length()

                correction = int(prev_logit.argmax().item())
                if correction != D[0]:
                    first_token_rejects += 1
                    new_tokens = [correction]
                    past, prev_logit, D, nf = _recover_and_draft(
                        past, P, correction
                    )
                    corr_fwd += nf
                else:
                    emb = torch.empty(1, S, d, device=device, dtype=dtype)
                    emb[0, draft_idx] = self.embed_tokens(
                        torch.tensor(D, device=device)
                    ).to(dtype)
                    emb[0, noise_pos] = _noise(N * N)[0]
                    m = torch.empty(S, P + S, device=device, dtype=dtype)
                    m[:, :P] = 0.0
                    m[:, P:] = fused_allow
                    pos = (rel_pos + P).unsqueeze(0)

                    if sc_on:
                        lb, _, _ = self._forward_backbone(
                            emb,
                            t_bn_fused,
                            m[None, None],
                            past_key_values=past,
                            use_cache=False,
                            position_ids=pos,
                            adaln_modulation=fused_adaln,
                        )
                        trim_kv_cache(past, past.get_seq_length() - P)
                        fused_fwd += 1
                        e_sc = F.softmax(lb[0, noise_pos].float(), -1).to(dtype) @ E
                        emb[0, noise_pos] = self._apply_self_cond(
                            emb[:, noise_pos, :], e_sc.unsqueeze(0)
                        )[0]
                    logits, past, _ = self._forward_backbone(
                        emb,
                        t_bn_fused,
                        m[None, None],
                        past_key_values=past,
                        use_cache=True,
                        position_ids=pos,
                        adaln_modulation=fused_adaln,
                    )
                    fused_fwd += 1
                    logf = logits[0]

                    j = 1
                    for i in range(1, N):
                        if logf[draft_idx[i - 1]].argmax().item() == D[i]:
                            j += 1
                        else:
                            break

                    new_tokens = D[:j]
                    keep = list(range(P)) + [P + draft_idx[i] for i in range(j)]
                    past = gather_kv_cache(past, torch.tensor(keep, device=device))
                    prev_logit = logf[draft_idx[j - 1]]
                    win = [draft_idx[j - 1] + 1 + k for k in range(N)]
                    D = logf[win].argmax(-1).tolist()

                if eos_id is not None and eos_id in new_tokens:
                    new_tokens = new_tokens[: new_tokens.index(eos_id) + 1]
                    gen_ids.extend(new_tokens)
                    committed += len(new_tokens)
                    break
                gen_ids.extend(new_tokens)
                committed += len(new_tokens)
        finally:
            if orig_attn is not None:
                self.backbone.config._attn_implementation = orig_attn

        generated = gen_ids[prompt_len : prompt_len + max_new_tokens]
        total_fwd = fused_fwd + boot_fwd + corr_fwd
        result = torch.tensor(
            [gen_ids[:prompt_len] + generated], device=device, dtype=torch.long
        )
        apf = len(generated) / max(total_fwd, 1)
        return result, {
            "total_steps": total_fwd,
            "logical_steps": total_fwd,
            "gen_len": len(generated),
            "hit_eos": eos_id is not None and eos_id in generated,
            "tokens_per_step": apf,
            "logical_tokens_per_step": apf,
            "accepted_per_forward": apf,
            "fused_forwards": fused_fwd,
            "bootstrap_forwards": boot_fwd,
            "correction_forwards": corr_fwd,
            "first_token_rejects": first_token_rejects,
            "self_cond": sc_on,
            "block_size": N,
            "num_euler_steps": 1,
        }

    @torch.no_grad()
    def generate_batched(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int = 128,
        num_steps: int = 64,
        temperature: float = 0.0,
        seed: int = 42,
        grid_mode: str = "uniform_range",
        t_max: float = 0.2,
        ode_argmax: bool = False,
        sde_gamma: float = 0.0,
        sde_alpha: float = 0.0,
        sde_vp: bool = False,
        sde_noiseless: bool = False,
        euler_alpha: float = 1.0,
        cfg_scale: float = 1.0,
    ):
        """Decode full blocks for left-padded prompts with a masked KV cache.

        Shared initial noise makes deterministic decoding nearly batch invariant.
        SDE churn uses independent noise for each batch row.
        """
        device = next(self.backbone.parameters()).device
        dtype = self.embed_tokens.weight.dtype
        torch.manual_seed(seed)
        noise_gen = torch.Generator(device=device).manual_seed(seed)
        prompt_ids = prompt_ids.to(device)
        attention_mask = attention_mask.to(device)
        B, L = prompt_ids.shape
        bd = self.bd_size
        eos_id = self.backbone.config.eos_token_id
        NEG = torch.finfo(dtype).min
        time_grid = self._get_generation_time_grid(
            num_steps, device, grid_mode=grid_mode, t_max=t_max
        )
        per_pos = time_grid.dim() == 2
        n_steps = (time_grid.shape[-1] - 1) if per_pos else (len(time_grid) - 1)

        orig_attn = getattr(self.backbone.config, "_attn_implementation", None)
        self.backbone.config._attn_implementation = "eager"

        cache_key_bias = (1.0 - attention_mask.to(dtype)) * NEG  # [B, L]
        gen_tokens = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        total_forward = 0
        tokens_generated = 0
        E = self.embed_tokens.weight

        def shared_noise():
            """One standalone-equivalent RNG draw, broadcast across batch rows."""
            return (
                (
                    torch.randn(
                        1,
                        bd,
                        self._d_model,
                        generator=noise_gen,
                        device=device,
                        dtype=dtype,
                    )
                    * self._noise_sigma
                    * self.noise_scale_mult
                )
                .expand(B, bd, self._d_model)
                .contiguous()
            )

        try:
            prompt_embeds = self.embed_tokens(prompt_ids)
            t_bn_p = self._compute_t_bn(torch.ones(B, L, device=device, dtype=dtype))
            causal = torch.triu(
                torch.full((L, L), NEG, device=device, dtype=dtype), diagonal=1
            )
            prefill_mask = causal[None, None] + cache_key_bias[:, None, None, :]
            prompt_logits, past, _ = self._forward_backbone(
                prompt_embeds, t_bn_p, prefill_mask, use_cache=True
            )
            if prompt_logits.dim() == 2:  # some envs return [B*L, V]
                prompt_logits = prompt_logits.reshape(B, -1, prompt_logits.shape[-1])
            prev_logit = prompt_logits[:, -1:, :]  # [B, 1, V]

            gen_pos = L
            while tokens_generated < max_new_tokens and not bool(finished.all()):
                block_pos_ids = torch.arange(gen_pos, gen_pos + bd, device=device)[
                    None
                ].expand(B, -1)
                cache_len = past.get_seq_length()
                block_mask = torch.zeros(
                    B, 1, bd, cache_len + bd, dtype=dtype, device=device
                )
                block_mask[:, :, :, :cache_len] = cache_key_bias[:, None, None, :]
                if self.noisy_causal:
                    cw = torch.triu(
                        torch.full((bd, bd), NEG, device=device, dtype=dtype),
                        diagonal=1,
                    )
                    block_mask[:, :, :, cache_len:] = cw[None, None]

                e_t = shared_noise()
                # Unconditional guidance hides prefix keys.
                use_cfg = abs(cfg_scale - 1.0) > 1e-6
                uncond_mask = None
                if use_cfg:
                    uncond_mask = block_mask.clone()
                    uncond_mask[:, :, :, :cache_len] = NEG

                use_shift = self.logit_shift

                def to_den(pr):  # probs [B, bd, V] -> predicted-clean embedding
                    if use_shift and prev_logit is not None:
                        pp = F.softmax(prev_logit.float(), dim=-1).to(dtype)
                        pa = torch.cat([pp, pr[:, :-1]], dim=1)
                        return E[pa.argmax(-1)] if ode_argmax else pa @ E
                    return E[pr.argmax(-1)] if ode_argmax else pr @ E

                e_self_cond = e_self_cond_u = None
                if self.enable_self_cond:
                    init_t = (
                        time_grid[:, 0].to(dtype)[None, :].expand(B, bd)
                        if per_pos
                        else torch.full(
                            (B, bd), time_grid[0].item(), device=device, dtype=dtype
                        )
                    )
                    init_t_bn = self._compute_t_bn(init_t)
                    init_logits, _, _ = self._forward_backbone(
                        e_t,
                        init_t_bn,
                        block_mask,
                        past_key_values=past,
                        use_cache=False,
                        position_ids=block_pos_ids,
                    )
                    trim_kv_cache(past, past.get_seq_length() - cache_len)
                    total_forward += 1
                    e_self_cond = (
                        F.softmax(init_logits.float(), -1).to(dtype) @ E
                    ).detach()
                    if use_cfg:
                        il_u, _, _ = self._forward_backbone(
                            e_t,
                            init_t_bn,
                            uncond_mask,
                            past_key_values=past,
                            use_cache=False,
                            position_ids=block_pos_ids,
                        )
                        trim_kv_cache(past, past.get_seq_length() - cache_len)
                        total_forward += 1
                        e_self_cond_u = (
                            F.softmax(il_u.float(), -1).to(dtype) @ E
                        ).detach()

                logits = None
                for step in range(n_steps):
                    if per_pos:
                        t_val_vec = time_grid[:, step]  # [bd]
                        t_out_vec = time_grid[:, step + 1]  # [bd]
                        if sde_alpha > 0.0 or sde_gamma > 0.0:
                            dt_vec = t_out_vec - t_val_vec
                            if sde_alpha > 0.0:
                                alpha_v = torch.full_like(t_val_vec, sde_alpha)
                            else:
                                alpha_v = (1.0 - sde_gamma * dt_vec).clamp(0.0, 1.0)
                            alpha_v = torch.where(
                                t_val_vec > 0.0, alpha_v, torch.ones_like(alpha_v)
                            )
                            if sde_noiseless:
                                coef_v = torch.zeros_like(
                                    alpha_v
                                )  # deterministic α-rewind, no injection
                            elif sde_vp:
                                coef_v = torch.sqrt(
                                    (
                                        (1.0 - alpha_v)
                                        * (1.0 + alpha_v - 2.0 * alpha_v * t_val_vec)
                                    ).clamp(min=0.0)
                                )
                            else:
                                coef_v = 1.0 - alpha_v
                            a_b = alpha_v.to(dtype)[
                                None, :, None
                            ]  # [1, bd, 1] → bcast over B,d
                            c_b = coef_v.to(dtype)[None, :, None]
                            noise = (
                                torch.randn_like(e_t)
                                * self._noise_sigma
                                * self.noise_scale_mult
                            )
                            z_fwd = a_b * e_t + c_b * noise
                            t_fwd_vec = (
                                alpha_v * t_val_vec
                            )  # conditioning time = alpha*t
                        else:
                            t_fwd_vec = t_val_vec
                            z_fwd = e_t
                        t_bn_block = self._compute_t_bn(
                            t_fwd_vec.to(dtype)[None, :].expand(B, bd)
                        )
                    else:
                        t_val = time_grid[step].item()
                        t_out = time_grid[step + 1].item()
                        dt = t_out - t_val
                        # sde_vp uses variance-preserving noise at the rewound time.
                        if (sde_alpha > 0.0 or sde_gamma > 0.0) and t_val > 0.0:
                            alpha = (
                                sde_alpha
                                if sde_alpha > 0.0
                                else max(0.0, min(1.0, 1.0 - sde_gamma * dt))
                            )
                            t_fwd = alpha * t_val
                            coef = (
                                0.0
                                if sde_noiseless
                                else (
                                    math.sqrt(
                                        max(
                                            0.0,
                                            (1.0 - alpha)
                                            * (1.0 + alpha - 2.0 * alpha * t_val),
                                        )
                                    )
                                    if sde_vp
                                    else (1.0 - alpha)
                                )
                            )
                            # Draw even in noiseless mode to keep seeded RNG aligned.
                            bk = (
                                torch.randn(
                                    B,
                                    bd,
                                    self._d_model,
                                    generator=noise_gen,
                                    device=device,
                                    dtype=dtype,
                                )
                                * self._noise_sigma
                                * self.noise_scale_mult
                            )
                            z_fwd = alpha * e_t + coef * bk
                        else:
                            t_fwd, z_fwd = t_val, e_t
                        t_bn_block = self._compute_t_bn(
                            torch.full((B, bd), t_fwd, device=device, dtype=dtype)
                        )
                    e_t_input = (
                        self._apply_self_cond(z_fwd, e_self_cond)
                        if self.enable_self_cond
                        else z_fwd
                    )
                    logits, _, _ = self._forward_backbone(
                        e_t_input,
                        t_bn_block,
                        block_mask,
                        past_key_values=past,
                        use_cache=False,
                        position_ids=block_pos_ids,
                    )
                    trim_kv_cache(past, past.get_seq_length() - cache_len)
                    total_forward += 1
                    probs_c = F.softmax(logits.float(), dim=-1).to(dtype)
                    e_denoised = to_den(probs_c)

                    if use_cfg:
                        e_in_u = (
                            self._apply_self_cond(z_fwd, e_self_cond_u)
                            if self.enable_self_cond
                            else z_fwd
                        )
                        logits_u, _, _ = self._forward_backbone(
                            e_in_u,
                            t_bn_block,
                            uncond_mask,
                            past_key_values=past,
                            use_cache=False,
                            position_ids=block_pos_ids,
                        )
                        trim_kv_cache(past, past.get_seq_length() - cache_len)
                        total_forward += 1
                        probs_u = F.softmax(logits_u.float(), dim=-1).to(dtype)
                        e_uncond = to_den(probs_u)
                        e_denoised = e_uncond + cfg_scale * (e_denoised - e_uncond)

                    if self.enable_self_cond:
                        e_self_cond = (probs_c @ E).detach()
                        if use_cfg:
                            e_self_cond_u = (probs_u @ E).detach()
                    if step == n_steps - 1:
                        break
                    # Broadcast position times over embeddings in fp32.
                    if per_pos:
                        tf = t_fwd_vec[None, :, None].float()  # [1, bd, 1]
                        to = t_out_vec[None, :, None].float()
                        e_t = (
                            z_fwd.float()
                            + (to - tf)
                            * (e_denoised.float() - z_fwd.float())
                            / (1.0 - tf).clamp(min=1e-6)
                        ).to(dtype)
                        if euler_alpha != 1.0:
                            c_v = (to - tf) / (1.0 - tf).clamp(min=1e-6)
                            e_t = (
                                e_t.float()
                                - (1.0 - c_v) * (1.0 - euler_alpha) * z_fwd.float()
                            ).to(dtype)
                    else:
                        e_t = z_fwd + (t_out - t_fwd) * (e_denoised - z_fwd) / max(
                            1.0 - t_fwd, 1e-6
                        )
                        if euler_alpha != 1.0:
                            c = (t_out - t_fwd) / max(1.0 - t_fwd, 1e-6)
                            e_t = e_t - (1.0 - c) * (1.0 - euler_alpha) * z_fwd

                if use_shift:
                    final_logits = torch.cat([prev_logit, logits], dim=1).float()
                else:
                    final_logits = logits.float()  # [B, bd, V]
                if temperature > 0:
                    fl = final_logits / temperature
                    probs = F.softmax(fl, dim=-1)
                    preds = torch.multinomial(
                        probs.reshape(-1, probs.shape[-1]), 1, generator=noise_gen
                    ).reshape(B, -1)
                else:
                    preds = final_logits.argmax(dim=-1)  # [B, bd(+1)]
                preds = preds[:, :bd]  # keep bd tokens

                pred_l = preds.tolist()
                for b in range(B):
                    if finished[b]:
                        continue
                    toks = pred_l[b]
                    if eos_id is not None and eos_id in toks:
                        toks = toks[: toks.index(eos_id) + 1]
                        finished[b] = True
                    gen_tokens[b].extend(toks)
                tokens_generated += bd

                # Re-encode every row to keep the cache rectangular.
                block_embeds = self.embed_tokens(preds)
                t_bn_final = self._compute_t_bn(
                    torch.ones(B, bd, device=device, dtype=dtype)
                )
                re_cache_len = past.get_seq_length()
                recode_mask = torch.zeros(
                    B, 1, bd, re_cache_len + bd, dtype=dtype, device=device
                )
                recode_mask[:, :, :, :re_cache_len] = cache_key_bias[:, None, None, :]
                cw = torch.triu(
                    torch.full((bd, bd), NEG, device=device, dtype=dtype), diagonal=1
                )
                recode_mask[:, :, :, re_cache_len:] = cw[None, None]
                re_logits, past, _ = self._forward_backbone(
                    block_embeds,
                    t_bn_final,
                    recode_mask,
                    past_key_values=past,
                    use_cache=True,
                    position_ids=block_pos_ids,
                )
                prev_logit = re_logits[:, -1:, :]
                cache_key_bias = torch.cat(
                    [cache_key_bias, torch.zeros(B, bd, device=device, dtype=dtype)],
                    dim=1,
                )
                gen_pos += bd
        finally:
            if orig_attn is not None:
                self.backbone.config._attn_implementation = orig_attn

        out_list = []
        for b in range(B):
            real_prompt = prompt_ids[b][attention_mask[b].bool()].tolist()
            out_list.append(real_prompt + gen_tokens[b][:max_new_tokens])
        return out_list, {
            "mode": "flow-batched",
            "total_steps": total_forward,
            "block_size": bd,
            "num_euler_steps": num_steps,
            "batch": B,
        }
