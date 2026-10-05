# Large Flow Language Models

**Arnav Chavan**<sup>1,2,&#42;</sup>, **Chanhyuk Lee**<sup>3,&#42;</sup>,
**Udbhav Bamba**<sup>1,&#42;</sup>, **Jaehoon Yoo**<sup>3</sup>,<br>
**Seunghoon Hong**<sup>3</sup>, **Aditi Raghunathan**<sup>4,†</sup>,
**Jinwoo Kim**<sup>3,†</sup>

<sup>1</sup> Amazon · <sup>2</sup> UC Berkeley · <sup>3</sup> KAIST ·
<sup>4</sup> Carnegie Mellon University

<sup>&#42;</sup> Equal contribution · <sup>†</sup> Equal advising

![arXiv](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/lafa-hf/models)
![Blog](https://img.shields.io/badge/Blog-2563EB)

Official implementation of **Large Flow Language Models**.
**Lafa** is a block-autoregressive flow language model that preserves causal
order across blocks while learning asynchronous progression within each block.
It converts pretrained autoregressive models into flow language models at
0.5B, 7B, and 8B scales, using fewer than 10B training tokens.

By jointly learning a block denoiser and a next-token predictor, Lafa supports
next-token, parallel flow, self-speculative, and confidence decoding in one
model. Progressive self-distillation learns flow maps for few-step generation.
The paper evaluates instruction following, reasoning, and coding on IFEval,
BBH, MMLU-Pro, GPQA, GSM8K, and HumanEval.

## Getting started

[Inference](#hugging-face-models) · [Training](#additional-teacher-training) ·
[Distillation](#distill-a-flow-map-student) · [Evaluation](#evaluate-an-hf-model)

## Hugging Face models

| Backbone | Model repository |
| --- | --- |
| Qwen2.5 0.5B Instruct | [lafa-hf/lafa-0.5B](https://huggingface.co/lafa-hf/lafa-0.5B) |
| Qwen2.5 7B Instruct | [lafa-hf/lafa-7B](https://huggingface.co/lafa-hf/lafa-7B) |
| Qwen3 8B Base | [lafa-hf/lafa-8B](https://huggingface.co/lafa-hf/lafa-8B) |

For private repositories, authenticate with `hf auth login` using an account
with access. The model code is packaged with the weights; inference does not
require installing this source repository or downloading the original Qwen weights.

```bash
pip install -r requirements-inference.txt
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "lafa-hf/lafa-0.5B"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16
).to("cuda").eval()
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 17 + 25?"}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
output = model.generate(
    **inputs, gen_mode="flow", num_steps=4, cfg_scale=1.0,
    noise_scale_override=0.1, max_new_tokens=128,
)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Choose `gen_mode="ar"`, `"flow"`, `"confidence"`, `"greedy"`, or `"speculative"`.
The default output is a token tensor; `return_dict_in_generate=True` returns
`sequences` and per-sample `stats`. The HF interface implements greedy generation.
Batched prompts are decoded sequentially and keep their input padding in the
returned prefix. Beam search, sampling controls, and automatic multi-device
inference dispatch are not supported. Use the repository evaluation harness
for its optimized batched teacher decoding.

The root model has EMA applied to its weights. Use `subfolder="raw"` to select
original weights when using an exported raw variant. Model settings, including
noise calibration and the legacy rotary-buffer precision, are stored in the
integrated `config.json`. The learned schedule is loaded from `time_warper.pt`.
Use a fixed `revision` for both model and tokenizer to pin an experiment.

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

## Distill a flow-map student

```bash
pip install -r requirements.txt -c constraints-tested.txt
# Eight GPUs: 4 examples/GPU × 4 accumulation steps × 8 GPUs = global batch 128.
torchrun --standalone --nproc_per_node=8 train_distill.py config/distill_0.5b.json
```

`distill_teacher_path` accepts an HF model ID or a local exported Lafa directory.
`revision` and `subfolder` select the teacher snapshot. The default teacher is
`lafa-hf/lafa-0.5B`, whose root weights already contain EMA. Student and teacher
start from those same weights; the student's target-time embedding is initialized
with a zero output. The teacher, vocabulary embeddings, vocabulary head, and
learned time schedule stay frozen.

The preset follows conditional PSD: equal diagonal/off-diagonal batch rows,
independent intervals per block, gap sampling with a midpoint split, noise 0.5,
and reduced-noise source-time-zero inputs on half the off-diagonal rows.
Diagonal targets come from the frozen teacher; off-diagonal targets compose two
stopped student transitions. The optimized cross entropy has the same gradient
as target-to-student KL. Clean-prefix AR KL preserves next-token predictions.
Component KL values are logged separately.

Hidden matrices use native Muon at `muon_lr=3e-5`; other parameters use AdamW
at `learning_rate=1e-5`. Training uses DDP with one frozen teacher per GPU.
Change gradient accumulation to preserve the desired global batch size.
DeepSpeed is supported by teacher training; this PSD entrypoint uses DDP.

For a local dataset or another teacher:

```bash
torchrun --standalone --nproc_per_node=2 train_distill.py config/distill_0.5b.json \
  --distill_teacher_path lafa-hf/lafa-0.5B --dataset_name /path/to/messages.jsonl \
  --gradient_accumulation_steps 16 --output_dir output/student
```

Rerunning the same command resumes the latest `checkpoint-*` in `output_dir`.
An explicit `--resume_from_checkpoint /path/to/checkpoint-N` also works.
Checkpoints retain optimizer, scheduler, RNG, EMA, learned schedule, and step
state. Keep the same distillation settings and process count when resuming.
Use a new output directory for a new training run.

Student checkpoints load with the same `AutoModelForCausalLM` example above;
set `repo` to the saved directory. They need no teacher for inference.
For a student, `gen_mode="flow"` evaluates learned two-time transitions; use
`cfg_scale=1.0`, `grid_mode="warp"`, and an explicit `num_steps` and noise scale.
`ar` and `speculative` also work. Confidence/greedy teacher decoding is rejected
for students. The final output directory holds running weights; checkpoint
`ema.pt` contains the separately maintained weight EMA.

## Evaluate an HF model

```bash
python -m src.evaluation.harness \
  --model lafa --tasks ifeval --num_fewshot 0 --apply_chat_template \
  --output_path output/ifeval \
  --gen_kwargs max_gen_toks=1024,do_sample=false,temperature=0.0 \
  --model_args "pretrained=lafa-hf/lafa-0.5B,gen_mode=flow,num_steps=4,cfg_scale=8,noise_scale_override=0.1,max_new_tokens=1024,max_length=4096"
```

HF root weights already contain EMA, so `use_ema=true` is unnecessary. For
reproduction, specify task-specific prompt formatting, few-shot count, token
limit, noise, CFG, and step count explicitly. This command is an execution
example; matching a paper score requires that row's evaluation settings.

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

## Evaluate

Run evaluation separately after training.

The `lafa` lm-evaluation-harness model loads a checkpoint and its saved
config. For a quick generation task:

```bash
python -m src.evaluation.harness \
  --model lafa --tasks gsm8k_cot --num_fewshot 0 \
  --apply_chat_template --output_path output/eval-gsm8k \
  --model_args "pretrained=output/lafa-0.5b/checkpoint-2000,base_model=Qwen/Qwen2.5-0.5B-Instruct,gen_mode=confidence,num_steps=1,threshold=0.95,noise_scale_override=0.1,use_ema=true"
```

Use `gen_mode=ar` for next token decoding, `gen_mode=flow` for whole block
Euler decoding, `gen_mode=confidence` for threshold decoding, and
`gen_mode=speculative` for self speculative decoding. `num_steps`,
`grid_mode`, `t_max`, `threshold`, `cfg_scale`, and
`noise_scale_override` control inference. Exact benchmark settings differ
by task; see the paper tables when reproducing a result.

## Validation

```bash
pip install -r requirements.txt -c constraints-tested.txt
pip install pytest
python -m pytest tests -q
```

Tests cover checkpoint loading, masked PSD loss and gradients, frozen teacher
weights, student save/load, token limits, and decoding regressions. Checkpoint
smoke tests cover short 0.5B training and distributed resume; these are not full
benchmark reproductions.

```bash
python scripts/check_checkpoint.py --checkpoint /path/to/teacher --output output/before.json
python scripts/check_checkpoint.py --checkpoint /path/to/teacher --output output/after.json \
  --reference output/before.json
```

The comparison uses checkpoint EMA, the learned time grid, noise multiplier 0.1,
seed 42, and 32 generated tokens. It compares two prompts in AR, flow (1 and 4
steps, CFG 8), and self-speculative modes.
