"""Timestep embedding and AdaLN conditioning for flow decoder layers."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class TimestepEmbedder(nn.Module):
    """DiT-style timestep embedder: sinusoidal(freq_dim) → MLP(freq_dim → d → d)."""

    def __init__(
        self,
        d_model: int,
        frequency_embedding_size: int = 256,
        mlp_init_std: float = 0.02,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=mlp_init_std)
        nn.init.normal_(self.mlp[2].weight, std=mlp_init_std)

    @staticmethod
    def sinusoidal_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Sinusoidal positional encoding for scalar timesteps.

        t: any shape (flattened internally), values typically in [0, 1].
        Returns: [..., dim] embedding.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        phases = t.float().unsqueeze(-1) * 1000.0 * freqs
        embedding = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """t: [B] or [B, S] → [B, d] or [B, S, d]."""
        t_freq = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class DiTAdaLN(nn.Module):
    """Produce attention and MLP scale, shift, and gate vectors.

    With no_shift, both shifts are zero. With init_scale=0 and gate_init=1,
    the patched decoder layer starts as the pretrained block.
    """

    def __init__(
        self,
        d_model: int,
        init_scale: float = 0.0,
        no_shift: bool = False,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.no_shift = no_shift
        n_outputs = 4 if no_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, n_outputs * d_model, bias=True),
        )
        if init_scale > 0:
            nn.init.normal_(self.adaLN_modulation[1].weight, std=init_scale)
            nn.init.zeros_(self.adaLN_modulation[1].bias)
        else:
            nn.init.constant_(self.adaLN_modulation[1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[1].bias, 0)

        if gate_init != 0.0:
            # Chunk order: no_shift -> (sa, ga, sm, gm); else -> (sa, sha, ga, sm, shm, gm)
            gate_chunks = (1, 3) if no_shift else (2, 5)
            bias = self.adaLN_modulation[1].bias
            with torch.no_grad():
                for gi in gate_chunks:
                    bias[gi * d_model : (gi + 1) * d_model].fill_(gate_init)

    def forward(self, c: Tensor) -> tuple:
        """c: [B, d] or [B, S, d] → tuple of 6 tensors, each same shape as c."""
        if self.no_shift:
            sa, ga, sm, gm = self.adaLN_modulation(c).chunk(4, dim=-1)
            z = torch.zeros_like(sa)
            return (sa, z, ga, sm, z, gm)
        return self.adaLN_modulation(c).chunk(6, dim=-1)

    def patch_layer(self, layer):
        """Patch a Qwen2 decoder layer to apply time-conditioned modulation."""
        input_layernorm = layer.input_layernorm
        post_attention_layernorm = layer.post_attention_layernorm
        self_attention = layer.self_attn
        mlp = layer.mlp
        adaln = self

        def patched_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            t_bn = kwargs.get("dit_t_bn", None)
            if t_bn is None:
                raise RuntimeError("dit_t_bn not passed to patched layer forward")
            cached_modulation = kwargs.get("dit_adaln_modulation", None)

            if cached_modulation is None:
                (
                    attn_scale,
                    attn_shift,
                    attn_gate,
                    mlp_scale,
                    mlp_shift,
                    mlp_gate,
                ) = adaln(t_bn)
            else:
                (
                    attn_scale,
                    attn_shift,
                    attn_gate,
                    mlp_scale,
                    mlp_shift,
                    mlp_gate,
                ) = cached_modulation
            if attn_scale.dim() == 2:
                attn_scale, attn_shift, attn_gate = (
                    attn_scale[:, None, :],
                    attn_shift[:, None, :],
                    attn_gate[:, None, :],
                )
                mlp_scale, mlp_shift, mlp_gate = (
                    mlp_scale[:, None, :],
                    mlp_shift[:, None, :],
                    mlp_gate[:, None, :],
                )

            attn_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("dit_t_bn", "dit_adaln_modulation")
            }
            residual = hidden_states
            attn_input = input_layernorm(residual) * (1 + attn_scale) + attn_shift
            attn_output, _ = self_attention(
                hidden_states=attn_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **attn_kwargs,
            )

            hidden_states = residual + attn_gate * attn_output
            residual = hidden_states
            mlp_input = (
                post_attention_layernorm(hidden_states) * (1 + mlp_scale) + mlp_shift
            )
            return residual + mlp_gate * mlp(mlp_input)

        layer.forward = patched_forward
