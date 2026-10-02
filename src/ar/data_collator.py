"""
AR Data Collator — batching for standard autoregressive training.

Conversation samples are right-padded to batch-max. The dataset masks prompt
tokens with -100 in labels.
"""

import torch
from torch import Tensor
from typing import Dict, List


class ARDataCollator:
    """Batch tokenized sequences for AR training.

    Produces:
      - input_ids: [B, L]
      - labels:    [B, L] (-100 on prompt and padding tokens)
      - attention_mask: [B, L] (1 for real tokens, 0 for padding)
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if not features or any("labels" not in f for f in features):
            raise ValueError("AR instruction batches require response-masked labels")
        raw_ids = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        raw_labels = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]
        for ids, lbl in zip(raw_ids, raw_labels):
            if ids.ndim != 1 or ids.numel() == 0 or lbl.shape != ids.shape:
                raise ValueError(
                    "AR instruction input_ids and labels must be equal-length sequences"
                )
            if lbl[0] != -100 or not torch.any(lbl[1:] != -100):
                raise ValueError(
                    "AR instruction labels must mask the prompt and supervise a response"
                )

        if all(t.shape[0] == raw_ids[0].shape[0] for t in raw_ids):
            input_ids = torch.stack(raw_ids)
            labels = torch.stack(raw_labels)
            attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        L = max(t.shape[0] for t in raw_ids)

        input_ids = torch.full((len(features), L), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), L), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(features), L, dtype=torch.long)

        for i, ids in enumerate(raw_ids):
            lbl = raw_labels[i]
            seq_len = ids.shape[0]
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = lbl

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
