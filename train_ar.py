#!/usr/bin/env python
# ruff: noqa: E402
"""Train an autoregressive language model on instruction data."""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from src.utils.config import BaseConfig, parse_config_and_training_args
from src.utils.checkpoints import latest_checkpoint
from src.utils.distributed import setup_accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    set_seed,
)

accelerator = setup_accelerator()

from src.ar.model import ARModelWrapper
from src.ar.data_collator import ARDataCollator
from src.dataset import load_and_tokenize_dataset


@dataclass
class ARConfig(BaseConfig):
    """AR training configuration."""

    model_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})
    dataset_name: str = field(metadata={"help": "Hugging Face instruction dataset ID"})
    block_size: int = field(default=2048, metadata={"help": "Training sequence length"})

    no_system_prompt: bool = field(
        default=False,
        metadata={"help": "Remove default system prompt from chat template"},
    )

    init_from: Optional[str] = field(
        default=None, metadata={"help": "Path to safetensors file to init model from"}
    )


def main():
    t_start = time.time()
    accelerator.print(f"[AR] Script started at {time.strftime('%H:%M:%S')}")

    ar_args, training_args = parse_config_and_training_args(ARConfig)
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    set_seed(training_args.seed)
    if accelerator.is_main_process:
        ar_args.save(f"{training_args.output_dir}/ar_config.json")
    accelerator.print(f"[AR] Args parsed ({time.time() - t_start:.1f}s)")

    tokenizer = AutoTokenizer.from_pretrained(
        ar_args.model_name_or_path,
        trust_remote_code=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        ar_args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if ar_args.no_system_prompt and tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "    {%- else %}\n"
            "        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud."
            " You are a helpful assistant.<|im_end|>\\n' }}\n"
            "    {%- endif %}",
            "    {%- else %}\n    {%- endif %}",
        ).replace(
            "        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}",
            "        {{- '' }}",
        )
        accelerator.print("[AR] Removed default system prompt from chat template")

    accelerator.print(
        f"[AR] Model loaded: {ar_args.model_name_or_path}, "
        f"vocab={len(tokenizer)}, block_size={ar_args.block_size} "
        f"({time.time() - t_start:.1f}s)",
        flush=True,
    )

    model = ARModelWrapper(backbone)

    if ar_args.init_from:
        from safetensors.torch import load_file as safe_load

        accelerator.print(f"[AR] Loading weights from: {ar_args.init_from}")
        init_sd = safe_load(ar_args.init_from)
        missing, unexpected = model.load_state_dict(init_sd, strict=False)
        if missing:
            accelerator.print(
                f"[AR] init_from: {len(missing)} missing keys (first 5): {missing[:5]}"
            )
        if unexpected:
            accelerator.print(
                f"[AR] init_from: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}"
            )
        accelerator.print(f"[AR] Loaded {len(init_sd)} keys from checkpoint")

    accelerator.print(
        f"[AR] Loading dataset {ar_args.dataset_name} ({time.time() - t_start:.1f}s)"
    )
    train_dataset = load_and_tokenize_dataset(
        dataset_name=ar_args.dataset_name,
        tokenizer=tokenizer,
        block_size=ar_args.block_size,
    )
    accelerator.print(
        f"[AR] Dataset ready: {len(train_dataset)} samples ({time.time() - t_start:.1f}s)"
    )

    collator = ARDataCollator(pad_token_id=tokenizer.pad_token_id)

    accelerator.print(f"[AR] Creating Trainer ({time.time() - t_start:.1f}s)")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    accelerator.print(f"[AR] Starting training ({time.time() - t_start:.1f}s)")
    resume_from = training_args.resume_from_checkpoint
    if resume_from is None or resume_from == "auto":
        resume_from = latest_checkpoint(training_args.output_dir)
        if resume_from:
            accelerator.print(f"[AR] Auto-resuming from: {resume_from}")
    accelerator.print(f"[AR] resume_from_checkpoint = {resume_from!r}")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()
    accelerator.print(f"[AR] Done! ({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
