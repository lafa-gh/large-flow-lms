# Training

[Back to README](../README.md)

Run commands from the repository root.

The model runtime and flow training utilities live together in `src/lafa/`.
Training uses `flow_model.py`, `data_collator.py`, `callbacks.py`, and
`init_checkpoint.py`; student distillation uses `distillation.py`.
`modeling_lafa.py` provides the Hugging Face interface and exports the runtime
files needed to load checkpoints independently of this repository.

## Additional teacher training

Install the training dependencies, then select a fine-tuning preset:

```bash
pip install -r requirements.txt -c constraints-tested.txt
accelerate launch train_flow.py config/finetune_0.5b.json
# Multi-GPU ZeRO-3 examples:
accelerate launch --num_processes 4 train_flow.py config/finetune_7b.json
accelerate launch --num_processes 4 train_flow.py config/finetune_8b.json
```

`model_name_or_path` accepts a local exported model or an HF repo ID. `revision`
and `subfolder` select the snapshot and weight variant. The 0.5B preset selects
`raw`; the other presets initialize from the default model. `dataset_name`
accepts an HF dataset or a local JSON/JSONL file with a `messages` field.

Additional training creates a fresh optimizer, scheduler, and model EMA, while
retaining the learned time schedule. To change block size, pass both
`--bd_size N` and `--reset_time_warper true`. Model architecture comes from the
checkpoint; incompatible architecture overrides are rejected. Newly saved
checkpoints include HF model code, configuration, and the time schedule.

## Training data

The presets load the `train` split of `allenai/Dolci-Instruct-SFT` from the
Hugging Face Hub. Another dataset can be selected with `dataset_name`; each
row must contain a `messages` list with `role` and `content` fields:

```json
{
  "messages": [
    {"role": "user", "content": "Write a greeting."},
    {"role": "assistant", "content": "Hello!"}
  ]
}
```

The loader applies the backbone tokenizer's chat template, masks prompt tokens,
and trains on assistant responses. Tokenized data is cached under the Hugging
Face datasets cache for the other local ranks to use.

## Train a teacher from an AR backbone

```bash
accelerate launch --num_processes 8 train_flow.py config/lafa_0.5b.json
```

The presets cover Qwen2.5 0.5B Instruct, Qwen2.5 7B Instruct, and Qwen3 8B
Base. They set the paper's flow objective, per position EMA time schedule,
residual self conditioning, and model weight EMA. For 7B or 8B, use
`config/lafa_7b.json` or `config/lafa_8b.json` with enough GPUs for ZeRO 3.
Training draws one quantile per block and maps it through the learned
per-position time schedules. During schedule warm-up, each block draws one
uniform time.
The presets fit per-position schedules with a uniform-time anchor and coupled
times at the other positions. The time schedules use EMA loss histograms with
sampling density proportional to the bin loss, and have no trainable parameters
or optimizer. The config's `ema_decay` applies separately to model weights.
Flow CE, auxiliary AR, and embedding MSE use their explicit config weights.
The effective batch size is
`per_device_train_batch_size × gradient_accumulation_steps × world_size`;
adjust accumulation when changing GPU count.
To log the time warp grid and sampled-time histogram to W&B, enable
`report_to=wandb` and set `time_warp_plot_every` and `time_warp_hist_every`
to nonzero step intervals.

Configs can be overridden on the command line. For example, to continue a
trained block size 4 model at block size 8:

```bash
accelerate launch --num_processes 8 train_flow.py config/lafa_0.5b.json \
  --bd_size 8 --block_size 1024 \
  --init_from /path/to/checkpoint/model.safetensors \
  --output_dir output/lafa-0.5b-b8
```

Each output directory contains `flow_config.json`. Checkpoints contain the
model weights and, when enabled, `ema.pt` and `time_warper.pt`. Keep these
files together when moving a checkpoint for inference.
The `epoch-{N}` directories are model snapshots with the same sidecars; use
`checkpoint-{N}` for full training resume.
When initializing from a flow checkpoint, pass its `time_warper.pt` with
`--time_warp_init_path` alongside `--init_from` to retain the learned schedule.

## Train an AR initialization

The autoregressive trainer uses the same Hugging Face dataset and tokenizer as the flow
trainer. To train a Qwen2.5 0.5B next token model:

```bash
accelerate launch --num_processes 8 train_ar.py config/ar_0.5b.json
```

Use `config/ar_7b.json` for Qwen2.5 7B with DeepSpeed ZeRO 3.

AR checkpoints save backbone weights with standard CausalLM tensor names. To
initialize a matching Lafa backbone from an AR checkpoint:

```bash
accelerate launch --num_processes 8 train_flow.py config/lafa_0.5b.json \
  --init_from output/ar-0.5b/checkpoint-500/model.safetensors
```
