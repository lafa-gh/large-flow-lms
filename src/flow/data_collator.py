"""
Flow Data Collator — minimal batching for embedding-space flow matching.

All flow-specific transforms (noise sampling, interpolation, packing, mask
construction) happen inside FlowModelWrapper.forward(). The collator just
batches raw token sequences, handling variable-length inputs via right-padding.

Conversation samples are variable-length. The collator pads to a block
boundary with pad_token_id in input_ids and -100 in labels. Partial final
blocks have no flow loss, while the clean AR path can train on their responses.
"""

import torch
from torch import Tensor
from typing import Dict, List


class FlowDataCollator:
    """Batch tokenized sequences for flow training.

    Produces:
      - input_ids: [B, L]
      - labels:    [B, L] (-100 on prompt and padding tokens)

    The model's forward() derives the response mask from labels internally.

    Variable-length samples are padded to a block boundary.
    """

    def __init__(self, pad_token_id: int = 0, bd_size: int = 8, return_attention_mask: bool = False):
        if bd_size < 1:
            raise ValueError("bd_size must be positive")
        self.return_attention_mask = return_attention_mask
        self.pad_token_id = pad_token_id
        self.bd_size = bd_size

    def __call__(self, features: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if not features or any("labels" not in f for f in features):
            raise ValueError("Flow instruction batches require response-masked labels")
        raw_ids = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        raw_labels = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]
        for ids, lbl in zip(raw_ids, raw_labels):
            if ids.ndim != 1 or ids.numel() == 0 or lbl.shape != ids.shape:
                raise ValueError(
                    "Flow instruction input_ids and labels must be equal-length sequences"
                )
            if lbl[0] != -100 or not torch.any(lbl[1:] != -100):
                raise ValueError(
                    "Flow instruction labels must mask the prompt and supervise a response"
                )

        if raw_ids[0].shape[0] % self.bd_size == 0 and all(
            t.shape[0] == raw_ids[0].shape[0] for t in raw_ids
        ):
            input_ids = torch.stack(raw_ids)
            labels = torch.stack(raw_labels)
            batch = {"input_ids": input_ids, "labels": labels}
            if self.return_attention_mask:
                batch["attention_mask"] = torch.ones_like(input_ids)
            return batch

        # Keep partial response blocks for the clean AR loss.
        max_len = max(t.shape[0] for t in raw_ids)
        L = ((max_len + self.bd_size - 1) // self.bd_size) * self.bd_size

        input_ids = torch.full((len(features), L), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), L), -100, dtype=torch.long)

        for i, ids in enumerate(raw_ids):
            tlen = ids.shape[0]
            input_ids[i, :tlen] = ids[:tlen]
            labels[i, :tlen] = raw_labels[i][:tlen]

        batch = {"input_ids": input_ids, "labels": labels}
        if self.return_attention_mask:
            lengths = torch.tensor([ids.numel() for ids in raw_ids])
            batch["attention_mask"] = (torch.arange(L)[None, :] < lengths[:, None]).long()
        return batch
