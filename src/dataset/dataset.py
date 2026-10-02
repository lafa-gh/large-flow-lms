"""Load a Hugging Face conversation dataset for instruction training."""

import hashlib
import json
import logging
import os
import shutil
import time
from typing import Mapping, Optional

from datasets import config as datasets_config, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _is_node_cache_builder(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether this process should build the node-local token cache.

    Torchrun assigns LOCAL_RANK=0 once per node, while RANK=0 exists only once
    for the whole job. Token caches default to node-local storage, so each node
    needs its own builder.
    """
    env = os.environ if environ is None else environ
    local_rank = env.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return int(env.get("RANK", "0")) == 0


def _tokenize_conversations(
    examples, tokenizer: PreTrainedTokenizer, max_length: int = None
):
    """Tokenize conversations with chat template, masking prompt tokens in labels.

    Two-call approach (works with ALL chat templates, unlike return_assistant_tokens_mask
    which requires {% generation %} tag):
      1. Tokenize full conversation → get all token ids
      2. For each assistant turn boundary, tokenize messages[:boundary] with
         add_generation_prompt=True to find where the assistant response starts
      3. Mark labels=-100 on non-assistant tokens

    Returns variable-length sequences (no truncation/padding). Each conversation
    stays as a separate sample — collators handle padding to batch max.
    """
    all_input_ids = []
    all_labels = []
    n_dropped = 0

    for messages in examples["messages"]:
        if not messages or any(m.get("content") is None for m in messages):
            continue

        full_encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        input_ids = full_encoded["input_ids"]

        labels = [-100] * len(input_ids)

        # Find each assistant turn boundary via incremental tokenization.
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                prompt_encoded = tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )
                prompt_len = len(prompt_encoded["input_ids"])

                through_encoded = tokenizer.apply_chat_template(
                    messages[: i + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                )
                response_end = len(through_encoded["input_ids"])

                for j in range(prompt_len, min(response_end, len(input_ids))):
                    labels[j] = input_ids[j]

        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        # Drop conversations whose answer was truncated away. They have no
        # next-token supervision and can make AR cross-entropy NaN.
        if not any(label != -100 for label in labels[1:]):
            n_dropped += 1
            continue

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    if n_dropped:
        print(
            f"[dataset] dropped {n_dropped} all-prompt sequences (0 answer tokens after "
            f"truncate+shift; max_length={max_length}) — no supervision, would NaN the AR CE"
        )
    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
    }


def _require_instruction_dataset(dataset):
    """Reject empty or unmasked data before training starts."""
    if not {"input_ids", "labels"}.issubset(dataset.column_names):
        raise ValueError(
            "Instruction training requires input_ids and response-masked labels"
        )
    if len(dataset) == 0:
        raise ValueError("Instruction dataset has no supervised conversations")
    labels = dataset[0]["labels"]
    if (
        len(labels) == 0
        or labels[0] != -100
        or not any(label != -100 for label in labels[1:])
    ):
        raise ValueError(
            "Instruction labels must mask the prompt and supervise an assistant response"
        )
    return dataset


def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    block_size: int = 2048,
    num_workers: Optional[int] = None,
):
    """Tokenize a Hugging Face train split with assistant-only supervision.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset ID with a train split and a messages column.
    tokenizer : PreTrainedTokenizer
        Tokenizer to use.
    block_size : int
        Maximum conversation length; longer samples are truncated.
    num_workers : int, optional
        Number of workers for map operations.

    Returns
    -------
    datasets.Dataset
        Variable-length rows with input_ids and labels (collator handles padding).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 128)
        logger.info(f"Auto-detected num_workers={num_workers}")

    # Cache tokenized rows for reuse by local ranks and later runs.
    _key = json.dumps(
        {
            "dataset": dataset_name,
            "local_mtime_ns": os.stat(dataset_name).st_mtime_ns if os.path.isfile(dataset_name) else None,
            "split": "train",
            "block": block_size,
            "tok": getattr(tokenizer, "name_or_path", ""),
            "vocab": len(tokenizer),
            "chat_template": getattr(tokenizer, "chat_template", None),
            # Cache keys do not hash the tokenization code.
            "tokfn_ver": 5,
        },
        sort_keys=True,
    )
    _h = hashlib.md5(_key.encode()).hexdigest()[:16]
    # All local ranks must use the same cache directory.
    cache_root = os.environ.get("AR2FLOW_TOK_CACHE")
    if not cache_root:
        cache_root = os.path.join(
            str(datasets_config.HF_DATASETS_CACHE), "ar2flow_tok_cache"
        )
    cache_dir = os.path.join(cache_root, _h)

    def _build():
        logger.info(f"Loading train split from {dataset_name}")
        if os.path.isfile(dataset_name):
            raw_dataset = load_dataset('json', data_files={'train': dataset_name}, split='train')
        else:
            raw_dataset = load_dataset(dataset_name, split="train")
        if "messages" not in raw_dataset.column_names:
            raise ValueError(
                "Instruction training requires a messages field in every instance"
            )
        if raw_dataset.data.column("messages").null_count:
            raise ValueError("Instruction training requires messages in every instance")
        tokenized = raw_dataset.map(
            _tokenize_conversations,
            batched=True,
            remove_columns=raw_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": block_size},
            num_proc=num_workers,
            desc="Tokenizing",
        )
        tokenized = tokenized.filter(
            lambda x: (
                len(x["input_ids"]) > 0 and any(label != -100 for label in x["labels"])
            )
        )
        logger.info(
            f"Dataset ready: {len(tokenized)} conversation samples "
            f"(variable length, max {block_size})"
        )
        return _require_instruction_dataset(tokenized)

    # Local rank 0 builds the cache; peers poll a file marker without an NCCL barrier.
    _done_marker = os.path.join(cache_dir, ".tok_cache_done")
    _fail_marker = (
        cache_dir.rstrip("/") + ".tok_cache_failed"
    )  # sibling, survives rmtree
    _global_rank = int(os.environ.get("RANK", "0"))

    if _is_node_cache_builder(os.environ):
        if not os.path.exists(_done_marker):
            # Remove incomplete data from a prior cache build.
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            if os.path.exists(_fail_marker):
                os.remove(_fail_marker)
            try:
                _ds = _build()
            except Exception:
                # Signal waiting ranks if Hub loading fails.
                try:
                    os.makedirs(cache_root, exist_ok=True)
                    with open(_fail_marker, "w") as _f:
                        _f.write("build_failed")
                except OSError:
                    pass
                raise
            # Cache writes are optional; peers fall back to tokenizing in memory.
            try:
                _ds.save_to_disk(cache_dir)
                with open(_done_marker, "w") as _f:
                    _f.write("ok")
                logger.info(f"Saved tokenized cache -> {cache_dir}")
            except OSError as e:
                logger.warning(
                    f"save_to_disk failed ({e}); using in-memory dataset "
                    f"(no disk cache this run)"
                )
                try:
                    with open(_fail_marker, "w") as _f:
                        _f.write(str(e))
                except OSError:
                    pass  # marker dir also unwritable; never let cleanup crash rank0
                return _ds
        logger.info(f"Loading tokenized cache <- {cache_dir}")
        return _require_instruction_dataset(load_from_disk(cache_dir))
    else:
        # Peers load the completed cache or fall back after a failure marker.
        _waited = 0
        while not os.path.exists(_done_marker) and not os.path.exists(_fail_marker):
            time.sleep(5)
            _waited += 5
            if _waited % 60 == 0:
                logger.info(
                    f"[rank {_global_rank}] waiting for node-local tokenized cache "
                    f"({_waited}s)..."
                )
        if os.path.exists(_fail_marker):
            with open(_fail_marker) as _f:
                if _f.read().strip() == "build_failed":
                    raise RuntimeError(
                        "Dataset loading failed on the local cache builder; check its logs"
                    )
            logger.warning(
                f"[rank {_global_rank}] node-local cache build failed; "
                f"building in-memory"
            )
            return _build()
        logger.info(f"Loading tokenized cache <- {cache_dir}")
        return _require_instruction_dataset(load_from_disk(cache_dir))
