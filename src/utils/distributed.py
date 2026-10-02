"""Small distributed runtime helpers for training jobs."""

import os
from datetime import timedelta


def set_cuda_device_from_local_rank():
    """Bind this process to its local CUDA device before NCCL collectives."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return None

    import torch

    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None

    device_id = int(local_rank) % device_count
    torch.cuda.set_device(device_id)
    return device_id


def _set_nccl_timeout(seconds: int = None):
    """Reset the NCCL ProcessGroup timeout after init."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    if seconds is None:
        seconds = int(os.environ.get("TORCH_NCCL_TIMEOUT", "7200"))
    dist.distributed_c10d._set_pg_timeout(timedelta(seconds=seconds))


def unwrap_model(
    model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = True
):
    """Unwrap DDP/accelerate/torch.compile wrappers to the underlying nn.Module.

    Pure function (no Accelerator instance needed): delegates to the same
    accelerate helper that Accelerator.unwrap_model wraps. Returns None for None.
    """
    if model is None:
        return None
    from accelerate.utils import extract_model_from_parallel

    return extract_model_from_parallel(model, keep_fp32_wrapper, keep_torch_compile)


def setup_accelerator(timeout_seconds: int = None):
    """Create an Accelerator after pinning CUDA, with a long PG timeout.

    Distributed model loading may wait on collectives before Trainer starts.
    Set the process-group timeout at Accelerator creation so it covers those
    waits; a post-hoc `_set_pg_timeout` cannot cover them.
    """
    set_cuda_device_from_local_rank()

    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    if timeout_seconds is None:
        timeout_seconds = int(os.environ.get("TORCH_NCCL_TIMEOUT", "7200"))

    ipg = InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds))
    # AcceleratorState must use the same precision as Trainer's Accelerator.
    os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")
    accelerator = Accelerator(kwargs_handlers=[ipg])
    _set_nccl_timeout(timeout_seconds)
    return accelerator
