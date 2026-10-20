# Distillation

[Back to README](../README.md)

Distillation turns a trained Lafa model (the teacher) into a flow map student
that generates a block in one or a few steps. It follows semigroup-based flow
map distillation:

- **Diagonal (`s = t`):** the student matches the frozen teacher denoiser.
- **Off-diagonal (`s ≠ t`):** the student matches a stop-gradient target formed
  by composing two shorter student jumps.
- **Prefix:** the student also matches the teacher's AR predictions, so prefix
  caching and self-speculative decoding still work.

All terms use KL losses. The student starts from the teacher's weights, plus an
identity-initialized adaLN for the destination time.

## Run

```bash
pip install -r requirements.txt -c constraints-tested.txt
torchrun --standalone --nproc_per_node=8 train_distill.py config/distill_0.5b.json
```

The preset distills `lafa-hf/lafa-0.5B` with a global batch of 128
(4 per GPU × 4 accumulation steps × 8 GPUs).

To use another teacher, your own data, or fewer GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train_distill.py config/distill_0.5b.json \
  --distill_teacher_path lafa-hf/lafa-0.5B \
  --dataset_name /path/to/messages.jsonl \
  --gradient_accumulation_steps 16 \
  --output_dir output/student
```

| Key | Meaning |
|---|---|
| `distill_teacher_path` | Teacher: HF repo ID or local Lafa directory |
| `revision`, `subfolder` | Teacher snapshot and weight variant |
| `dataset_name` | Same format as [training data](training.md#training-data) |
| `output_dir` | Where checkpoints are written |

To resume, rerun the same command. Training continues from the latest
`checkpoint-*` in `output_dir`.

## Use the student

Load the student like any Lafa model ([quick start](../README.md#quick-start)),
with `repo` set to the output directory. No teacher is needed.

| Decoding | Arguments |
|---|---|
| Flow map, `K` steps | `gen_mode="flow"`, `num_steps=K`, `cfg_scale=1.0`, `noise_scale_override=0.5` |
| One-step flow map | same, with `num_steps=1` |
| Confidence thresholding | same, with `num_steps=1`, `threshold=λ` |
| Self-speculative | `gen_mode="speculative"` |
| Autoregressive | `gen_mode="ar"` |

The output directory holds the raw student weights; each checkpoint's `ema.pt`
holds the weight EMA.
