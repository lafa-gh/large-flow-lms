"""Attention mask for Lafa's packed clean and noisy training streams."""

import torch
from torch import Tensor


def create_packed_attention_mask(
    orig_seq_len: int,
    block_size: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
    block_causal: bool = False,
    noisy_causal: bool = False,
) -> Tensor:
    """Build a dense mask for packed [clean(L) | noisy(L)] tokens.

    Clean tokens are causal by default. Each noisy block reads earlier clean
    blocks and its own noisy block. ``block_causal`` and ``noisy_causal`` select
    alternative within-block masking rules.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    L = orig_seq_len
    S = 2 * L
    B = block_size
    num_blocks = L // B
    if dtype is None:
        dtype = torch.float32

    mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)

    # ---- Clean-Clean (MCC) ----
    if block_causal:
        for k in range(num_blocks):
            cs, ce = k * B, min((k + 1) * B, L)
            mask[cs:ce, :cs] = 0.0
            mask[cs:ce, cs:ce] = 0.0
        if L % B > 0:
            cs = num_blocks * B
            mask[cs:L, :cs] = 0.0
            mask[cs:L, cs:L] = 0.0
    else:
        mask[:L, :L] = torch.triu(
            torch.full((L, L), float("-inf"), device=device, dtype=dtype), diagonal=1
        )

    # ---- Noisy-Clean (MOBC): block k sees clean[0 .. k*B-1] ----
    for k in range(num_blocks):
        ns, ne = L + k * B, L + min((k + 1) * B, L)
        vis = k * B
        if vis > 0:
            mask[ns:ne, :vis] = 0.0
    if L % B > 0:
        ns = L + num_blocks * B
        vis = num_blocks * B
        if vis > 0:
            mask[ns:S, :vis] = 0.0

    # ---- Noisy-Noisy: block-diagonal, optionally causal within block ----
    for k in range(num_blocks):
        ns, ne = L + k * B, L + min((k + 1) * B, L)
        blen = ne - ns
        if noisy_causal:
            causal_block = torch.triu(
                torch.full((blen, blen), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            mask[ns:ne, ns:ne] = causal_block
        else:
            mask[ns:ne, ns:ne] = 0.0
    if L % B > 0:
        ns = L + num_blocks * B
        blen = S - ns
        if noisy_causal:
            causal_block = torch.triu(
                torch.full((blen, blen), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            mask[ns:S, ns:S] = causal_block
        else:
            mask[ns:S, ns:S] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)


def create_conditional_packed_attention_mask(
    answer_mask: Tensor,
    token_mask: Tensor,
    block_size: int,
    dtype: torch.dtype = None,
    block_causal: bool = False,
    noisy_causal: bool = False,
) -> Tensor:
    """Per-sample packed mask for prompt-clean instruction training.

    ``answer_mask`` and ``token_mask`` are ``[batch, length]`` booleans.  The
    clean half remains the ordinary causal (or block-causal) teacher stream.
    Each contiguous answer span is intersected with the fixed physical blocks;
    a noisy answer chunk attends to:

      * the causal clean-half history strictly before that chunk, and
      * only the noisy-half tokens in the same answer chunk.

    Prompt and padding copies in the noisy half are therefore never used as
    conditioning keys.  This prevents a clean prompt representation from
    reading the future bidirectional noisy answer, and prevents answer tokens
    from reading right-padding or a later conversation turn.  Answer-to-answer
    attention inside the active chunk stays bidirectional unless
    ``noisy_causal`` is requested.

    Physical block alignment is intentionally preserved here.  Aligning blocks
    to response starts is a separate training-policy change because it must also
    update per-block time sampling and subset/distillation bookkeeping.
    """
    if answer_mask.ndim != 2 or token_mask.ndim != 2:
        raise ValueError("answer_mask and token_mask must both have shape [B, L]")
    if answer_mask.shape != token_mask.shape:
        raise ValueError(
            f"answer_mask/token_mask shape mismatch: {answer_mask.shape} vs {token_mask.shape}"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")

    answer_mask = answer_mask.bool()
    token_mask = token_mask.bool()
    if bool((answer_mask & ~token_mask).any()):
        raise ValueError("answer_mask marks padding positions as answer tokens")

    batch, length = answer_mask.shape
    device = answer_mask.device
    if token_mask.device != device:
        token_mask = token_mask.to(device)
    if dtype is None:
        dtype = torch.float32

    total = 2 * length
    mask = torch.full(
        (batch, 1, total, total), float("-inf"), device=device, dtype=dtype
    )

    # Clean teacher stream.  Invalid right-padding is never exposed as a key.
    qpos = torch.arange(length, device=device)[:, None]
    kpos = torch.arange(length, device=device)[None, :]
    if block_causal:
        clean_visible = (qpos // block_size) >= (kpos // block_size)
    else:
        clean_visible = qpos >= kpos
    clean_visible = clean_visible.unsqueeze(0) & token_mask[:, None, :]
    mask[:, 0, :length, :length] = torch.where(
        clean_visible,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float("-inf"), device=device, dtype=dtype),
    )

    # Give invalid query rows one harmless self key so attention kernels never
    # receive an all-masked row.  These outputs are excluded from every loss and
    # are never visible to an answer query.
    for b in range(batch):
        invalid = (~token_mask[b]).nonzero(as_tuple=False).flatten()
        if invalid.numel():
            mask[b, 0, invalid, invalid] = 0.0
            mask[b, 0, length + invalid, invalid] = 0.0

    for b in range(batch):
        ans = answer_mask[b]
        valid = token_mask[b]

        # Non-answer noisy rows are not used as keys.  Give real prompt rows a
        # causal clean-half view only, producing a finite but otherwise unused
        # representation.
        prompt_rows = (valid & ~ans).nonzero(as_tuple=False).flatten()
        for q in prompt_rows.tolist():
            hist = valid & (torch.arange(length, device=device) <= q)
            mask[b, 0, length + q, :length][hist] = 0.0

        # Find contiguous assistant spans, then intersect each with physical
        # blocks.  This also prevents attention across a user turn when two
        # assistant spans happen to occupy the same physical block.
        padded = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=device),
            ans,
            torch.zeros(1, dtype=torch.bool, device=device),
        ])
        edges = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
        starts = (edges == 1).nonzero(as_tuple=False).flatten().tolist()
        ends = (edges == -1).nonzero(as_tuple=False).flatten().tolist()

        for span_start, span_end in zip(starts, ends):
            chunk_start = span_start
            while chunk_start < span_end:
                physical_end = ((chunk_start // block_size) + 1) * block_size
                chunk_end = min(span_end, physical_end)
                rows = slice(length + chunk_start, length + chunk_end)

                # Exact causal history used by generation's KV cache.
                if chunk_start > 0:
                    hist = valid[:chunk_start]
                    mask[b, 0, rows, :chunk_start][:, hist] = 0.0

                # Active answer chunk.  Keep the diffusion block bidirectional.
                chunk_len = chunk_end - chunk_start
                if noisy_causal:
                    local = torch.tril(torch.ones(
                        chunk_len, chunk_len, dtype=torch.bool, device=device
                    ))
                    target = mask[
                        b, 0, rows, length + chunk_start:length + chunk_end
                    ]
                    target[local] = 0.0
                else:
                    mask[
                        b, 0, rows, length + chunk_start:length + chunk_end
                    ] = 0.0
                chunk_start = chunk_end

    return mask
