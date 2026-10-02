"""KV cache utilities for inference."""


def trim_kv_cache(past_kv, n_trim: int):
    """Remove the last n_trim entries from a KV cache.

    Supports DynamicCache (.crop), older DynamicCache (.key_cache),
    and legacy tuple-of-tuples format.
    """
    if n_trim <= 0:
        return past_kv
    if hasattr(past_kv, "crop"):
        past_kv.crop(past_kv.get_seq_length() - n_trim)
        return past_kv
    elif hasattr(past_kv, "key_cache"):
        for i in range(len(past_kv.key_cache)):
            past_kv.key_cache[i] = past_kv.key_cache[i][:, :, :-n_trim, :]
            past_kv.value_cache[i] = past_kv.value_cache[i][:, :, :-n_trim, :]
        if hasattr(past_kv, "_seen_tokens"):
            past_kv._seen_tokens -= n_trim
        return past_kv
    else:
        return tuple(
            (kc[:, :, :-n_trim, :], vc[:, :, :-n_trim, :]) for kc, vc in past_kv
        )


def gather_kv_cache(past_kv, keep_index):
    """Keep only the sequence positions in `keep_index`, dropping all others.

    `keep_index` is a 1-D LongTensor of cache positions to retain, in the order
    they should appear afterwards. Unlike trim_kv_cache (which only drops a
    contiguous tail), this index-selects along the sequence dim, so it can drop
    *interior* entries. Self-speculative decoding uses it to compact a fused
    [draft | noise-block] cache down to prefix+accepted tokens without paying for
    a re-encode forward: the accepted draft tokens attended only to the prefix
    and earlier accepted tokens, so their cached K/V are byte-identical to a
    plain causal decode and can be kept verbatim.

    Supports DynamicCache (.key_cache/.value_cache lists or newer .layers) and
    the legacy tuple-of-tuples format.
    """
    n_keep = int(keep_index.numel())

    # DynamicCache with key_cache / value_cache lists.
    if hasattr(past_kv, "key_cache") and getattr(past_kv, "key_cache"):
        for i in range(len(past_kv.key_cache)):
            k = past_kv.key_cache[i]
            if k is None or k.numel() == 0:
                continue
            idx = keep_index.to(k.device)
            past_kv.key_cache[i] = k.index_select(2, idx).contiguous()
            past_kv.value_cache[i] = (
                past_kv.value_cache[i].index_select(2, idx).contiguous()
            )
        if hasattr(past_kv, "_seen_tokens"):
            past_kv._seen_tokens = n_keep
        return past_kv

    # Newer DynamicCache: per-layer objects exposing .keys / .values.
    if hasattr(past_kv, "layers") and getattr(past_kv, "layers"):
        for layer in past_kv.layers:
            k = getattr(layer, "keys", None)
            if k is None or k.numel() == 0:
                continue
            idx = keep_index.to(k.device)
            layer.keys = k.index_select(2, idx).contiguous()
            layer.values = layer.values.index_select(2, idx).contiguous()
        if hasattr(past_kv, "_seen_tokens"):
            past_kv._seen_tokens = n_keep
        return past_kv

    # Legacy tuple-of-tuples.
    return tuple(
        (
            kc.index_select(2, keep_index.to(kc.device)).contiguous(),
            vc.index_select(2, keep_index.to(vc.device)).contiguous(),
        )
        for kc, vc in past_kv
    )
