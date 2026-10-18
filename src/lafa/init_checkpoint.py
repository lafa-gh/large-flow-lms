"""Checkpoint loading helpers for Flow models."""

from collections.abc import Mapping

from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys


def _named_parameters_with_duplicates(model: nn.Module):
    """Return every state-dict parameter name, including aliases."""
    try:
        return model.named_parameters(remove_duplicate=False)
    except TypeError:
        # Compatibility with older PyTorch versions that predate remove_duplicate.
        return model.named_parameters()


def _raise_strict_key_errors(model, missing_keys, unexpected_keys):
    if not missing_keys and not unexpected_keys:
        return

    messages = []
    if unexpected_keys:
        messages.append(
            "Unexpected key(s) in state_dict: "
            + ", ".join(f'"{key}"' for key in unexpected_keys)
            + "."
        )
    if missing_keys:
        messages.append(
            "Missing key(s) in state_dict: "
            + ", ".join(f'"{key}"' for key in missing_keys)
            + "."
        )
    raise RuntimeError(
        f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t"
        + "\n\t".join(messages)
    )


def normalize_init_state_dict(
    model: nn.Module, state_dict: Mapping[str, Tensor], *, tie_word_embeddings=None
):
    """Accept native flow weights and standard CausalLM (AR) tensor names.

    Resolve names before either replicated or ZeRO-3 loading. Reject ambiguous
    aliases and checkpoints that would leave the entire backbone unchanged.
    """
    model_keys = set(model.state_dict())
    normalized = {}
    for key, value in state_dict.items():
        target = key
        if key not in model_keys and f"backbone.{key}" in model_keys:
            target = f"backbone.{key}"
        if target in normalized:
            raise ValueError(f"Multiple checkpoint keys resolve to {target!r}")
        normalized[target] = value
    # Safetensors omits the vocabulary head when an AR checkpoint ties it to
    # embeddings. The flow model can deliberately untie these parameters.
    if ("model.embed_tokens.weight" in state_dict
            and "backbone.lm_head.weight" in model_keys
            and "backbone.lm_head.weight" not in normalized):
        if tie_word_embeddings is not True:
            raise ValueError(
                "AR checkpoint omits lm_head.weight; loading into an untied "
                "flow model requires config.json with tie_word_embeddings=true"
            )
        normalized["backbone.lm_head.weight"] = normalized[
            "backbone.model.embed_tokens.weight"]
    if not any(key.startswith("backbone.") and key in model_keys
               for key in normalized):
        raise ValueError("Checkpoint contains no matching backbone weights")
    return normalized


def load_init_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    strict: bool = False,
    tie_word_embeddings: bool | None = None,
) -> _IncompatibleKeys:
    """Load both replicated wrapper state and ZeRO-3 backbone parameters.

    ZeRO-3 backbone parameters are partitioned placeholders at construction;
    wrapper parameters and buffers are initially replicated.
    """
    state_dict = normalize_init_state_dict(
        model, state_dict, tie_word_embeddings=tie_word_embeddings)
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        zero3_enabled = is_deepspeed_zero3_enabled()
    except (ImportError, RuntimeError):
        zero3_enabled = False

    sharded_names = {
        name
        for name, param in _named_parameters_with_duplicates(model)
        if hasattr(param, "ds_id")
    }
    if not zero3_enabled or not sharded_names:
        return model.load_state_dict(state_dict, strict=strict)

    model_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)

    # This path also preserves FlowModelWrapper.load_state_dict's compatibility
    # handling for legacy _embed_init_norm buffers.
    replicated_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_keys and key not in sharded_names
    }
    if replicated_state:
        model.load_state_dict(replicated_state, strict=False)

    sharded_state = {
        key: value for key, value in state_dict.items() if key in sharded_names
    }
    if sharded_state:
        from transformers.integrations.deepspeed import (
            _load_state_dict_into_zero3_model,
        )

        error_messages, _ = _load_state_dict_into_zero3_model(model, sharded_state)
        if error_messages:
            joined = "\n\t".join(error_messages)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t"
                f"{joined}"
            )

    if strict:
        _raise_strict_key_errors(model, missing_keys, unexpected_keys)

    return _IncompatibleKeys(missing_keys, unexpected_keys)
