# Evaluation

[Back to README](../README.md)

Run commands from the repository root.

Install the evaluation dependencies:

```bash
pip install -r requirements.txt -c constraints-tested.txt
```

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

## Evaluate a local checkpoint

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
