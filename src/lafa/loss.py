import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    shift: bool = True,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss with optional NTP shift.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] — use -100 to ignore positions
        shift: If True, logits[i] predicts labels[i+1] (standard NTP).
               If False, logits[i] predicts labels[i] (direct prediction).

    Returns:
        Scalar mean cross-entropy loss (ignoring -100 positions).
    """
    logits = logits.float()

    if shift:
        labels = F.pad(labels, (0, 1), value=ignore_index)  # [batch_size, seq_len+1]
        labels = labels[..., 1:].contiguous()  # [batch_size, seq_len]

    vocab_size = logits.size(-1)
    logits = logits.view(-1, vocab_size)  # [batch_size*seq_len, vocab_size]
    labels = labels.view(-1)  # [batch_size*seq_len]

    return F.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction=reduction
    )
