# Training

[Back to README](../README.md)

Lafa adapts a pretrained AR model into a block flow model. Training combines AR
cross entropy on the clean sequence with denoising cross entropy and embedding
squared error on noisy blocks, all in one forward pass.

Run commands from the repository root:

```bash
pip install -r requirements.txt -c constraints-tested.txt
```

| Goal | Command |
|---|---|
| [Adapt an AR model](#adapt-an-ar-model) | `train_flow.py config/lafa_{0.5b,7b,8b}.json` |
| [Continue training a Lafa model](#continue-training-a-lafa-model) | `train_flow.py config/finetune_{0.5b,7b,8b}.json` |
| [Train a matched AR baseline](#train-a-matched-ar-baseline) | `train_ar.py config/ar_{0.5b,7b}.json` |
| Distill a flow map | See [distillation](distillation.md) |

## Adapt an AR model

```bash
accelerate launch --num_processes 8 train_flow.py config/lafa_0.5b.json
```

| Preset | Backbone | Parallelism |
|---|---|---|
| `config/lafa_0.5b.json` | Qwen2.5 0.5B Instruct | Data parallel |
| `config/lafa_7b.json` | Qwen2.5 7B Instruct | DeepSpeed ZeRO-3 |
| `config/lafa_8b.json` | Qwen3 8B Base | DeepSpeed ZeRO-3 |

The presets follow the paper's main setting:

| Setting | Config key | Value |
|---|---|---|
| Block size `B` | `bd_size` | 4 |
| Training noise scale `σ` | `noise_scale_mult` | 4.0 |
| Learned tokenwise time | `time_warp_per_position` | per position |
| Self-conditioning | `enable_self_cond`, `sc_drop_rate` | on, probability 0.5 |
| Norm-constrained embeddings | `embed_norm_project` | on |
| Loss weights (AR / denoising CE / embedding MSE) | `ar_loss_weight`, `flow_ce_weight`, `denoise_mse_weight` | 1 / 1 / 1 |
| Weight EMA | `ema_decay` | 0.999 |

Any config key can be overridden on the command line:

```bash
accelerate launch --num_processes 8 train_flow.py config/lafa_0.5b.json \
  --bd_size 8 --output_dir output/lafa-0.5b-b8
```

The global batch size is
`per_device_train_batch_size × gradient_accumulation_steps × GPUs`. Adjust
gradient accumulation when changing the number of GPUs.

## Continue training a Lafa model

```bash
accelerate launch train_flow.py config/finetune_0.5b.json
accelerate launch --num_processes 4 train_flow.py config/finetune_7b.json
accelerate launch --num_processes 4 train_flow.py config/finetune_8b.json
```

| Key | Meaning |
|---|---|
| `model_name_or_path` | HF repo ID or local Lafa directory |
| `revision`, `subfolder` | Snapshot and weight variant (`raw` for non-EMA) |
| `dataset_name` | HF dataset or local JSON/JSONL file |

The learned tokenwise time is kept; the optimizer and EMA start fresh. To change
the block size, also pass `--reset_time_warper true`.

## Train a matched AR baseline

```bash
accelerate launch --num_processes 8 train_ar.py config/ar_0.5b.json
```

This trains the AR SFT baseline on the same data. To adapt the result into
Lafa:

```bash
accelerate launch --num_processes 8 train_flow.py config/lafa_0.5b.json \
  --init_from output/ar-0.5b/checkpoint-500/model.safetensors
```

## Training data

The presets use the `train` split of
[`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT).
To use your own data, set `dataset_name` to a JSON/JSONL file whose rows hold
chat messages:

```json
{"messages": [{"role": "user", "content": "Write a greeting."},
              {"role": "assistant", "content": "Hello!"}]}
```

Training uses assistant turns only.

## Checkpoints

| Path | Contents |
|---|---|
| `checkpoint-{N}/` | Full training state for resuming |
| `epoch-{N}/` | Model snapshot at the end of each epoch |
| `model.safetensors` | Weights |
| `ema.pt` | Weight EMA |
| `time_warper.pt` | Learned tokenwise time |
| `flow_config.json` | Training configuration |

Keep these files together when moving a checkpoint.
