# Evaluation

[Back to README](../README.md)

Benchmarks run through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
with the `lafa` model in `src.evaluation.harness`. Run commands from the
repository root:

```bash
pip install -r requirements.txt -c constraints-tested.txt
```

## Run a benchmark

```bash
python -m src.evaluation.harness \
  --model lafa --tasks ifeval --num_fewshot 0 --apply_chat_template \
  --gen_kwargs max_gen_toks=1024,do_sample=false,temperature=0.0 \
  --output_path output/ifeval \
  --model_args "pretrained=lafa-hf/lafa-0.5B,gen_mode=flow,num_steps=4,cfg_scale=8,noise_scale_override=0.1,max_new_tokens=1024,max_length=4096"
```

`pretrained` takes an HF repo ID or a local checkpoint. For a local training
checkpoint, also pass `base_model=<backbone>` and `use_ema=true`; HF models
already hold EMA weights.

## Decoding settings

Set the scheme in `--model_args`:

| Scheme | `--model_args` |
|---|---|
| Autoregressive | `gen_mode=ar` |
| Flow, `K` steps | `gen_mode=flow,num_steps=K,cfg_scale=8,noise_scale_override=0.1` |
| Flow map (student), `K` steps | `gen_mode=flow,num_steps=K,cfg_scale=1,noise_scale_override=0.5` |
| Self-speculative | `gen_mode=speculative` |
| Confidence thresholding | `gen_mode=flow,num_steps=1,threshold=λ,self_cond=false` |

The harness reports tokens per forward (TPF) alongside accuracy, excluding
prompt prefill.

## Benchmarks

| Benchmark | `--tasks` | Few-shot | Chat template |
|---|---|---|---|
| IFEval | `ifeval` | 0 | yes |
| BBH | `bbh` | 3 | no |
| MMLU-Pro | `mmlu_pro` | 0 | no |
| GPQA | `gpqa_main_cot_zeroshot` | 0 | yes |
| GSM8K | `gsm8k_cot` | 0 | yes |
| HumanEval | `humaneval_instruct` (add `--confirm_run_unsafe_code`) | 0 | yes |

Add `--apply_chat_template` where the table says yes. To reproduce a paper
number, also match that row's generation budget and decoding settings.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

To check that a code change leaves generations unchanged:

```bash
python scripts/check_checkpoint.py --checkpoint /path/to/model --output output/before.json
# ...apply the change...
python scripts/check_checkpoint.py --checkpoint /path/to/model --output output/after.json \
  --reference output/before.json
```
