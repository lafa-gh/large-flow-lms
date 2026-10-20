# Inference

[Back to README](../README.md)

Lafa models load with `AutoModelForCausalLM` and `trust_remote_code=True`; see
the [quick start](../README.md#quick-start). The model code ships with each
checkpoint, so inference needs neither this repository nor the original Qwen
weights.

## Decoding schemes

All schemes reuse the prefix KV cache and decode greedily.

| Scheme | Arguments | What it does |
|---|---|---|
| Autoregressive | `gen_mode="ar"` | One token per forward from the retained AR predictor. |
| Flow | `gen_mode="flow"`, `num_steps=K` | Generates each block with `K` Euler steps of the learned flow. |
| Flow map | `gen_mode="flow"`, `num_steps=K` on a distilled student | Replaces each Euler step with a learned jump; `K=1` generates a block in one step. |
| Self-speculative | `gen_mode="speculative"` | Drafts a block in one step and verifies it with the AR predictor. Output matches greedy AR decoding. |
| Confidence thresholding | `gen_mode="flow"`, `num_steps=1`, `threshold=λ` | Accepts the longest left-to-right prefix of a one-step draft whose confidence is at least `λ` (at least one token). |

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `num_steps` | 4 | Euler steps per block (`flow`) |
| `cfg_scale` | 1.0 | Prefix guidance scale `g` (`flow`). The paper uses 8. Students use 1. |
| `noise_scale_override` | 0.1 | Inference noise scale `σ` |
| `threshold` | 0 (off) | Confidence threshold `λ` |
| `max_new_tokens` | from the generation config | Generation budget |
| `seed` | 42 | Noise seed |

Pass `return_dict_in_generate=True` to get `sequences` together with
per-sample `stats`, including forward counts.

**Tokens per forward (TPF)** is generated tokens divided by decoding forwards,
excluding prompt prefill.

## Checkpoint variants

| Option | Weights |
|---|---|
| Default | EMA weights |
| `subfolder="raw"` | Non-EMA weights, where provided |
| `revision="<commit>"` | A pinned snapshot. Use the same value for the tokenizer. |

## Limitations

- Greedy decoding only. Sampling and beam search are not supported.
- Batched prompts are decoded one row at a time.
