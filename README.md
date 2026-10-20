<div align="center">

# Large Flow Language Models

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-B31B1B?logo=arxiv&logoColor=white)](#citation)
[![Models](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/lafa-hf/models)
[![Inference](https://img.shields.io/badge/Docs-Inference-2563EB)](docs/inference.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-16A34A)](LICENSE)

**Arnav Chavan**<sup>1,2,&#42;</sup> ·
**Chanhyuk Lee**<sup>3,&#42;</sup> ·
**Udbhav Bamba**<sup>2,&#42;</sup> ·
**Jaehoon Yoo**<sup>3</sup> ·
**Seunghoon Hong**<sup>3</sup> ·
**Aditi Raghunathan**<sup>4,†</sup> ·
**Jinwoo Kim**<sup>3,†</sup>

<sub>
<sup>1</sup> UC Berkeley · <sup>2</sup> Amazon · <sup>3</sup> KAIST ·
<sup>4</sup> Carnegie Mellon University<br>
<sup>&#42;</sup> Equal contribution · <sup>†</sup> Equal advising
</sub>

</div>

---

**Lafa** is a continuous flow language model that generates text block by
block through a flow over token embeddings, which can be distilled into a flow
map for one- or few-step generation. Lafa can also verify its own flow drafts
autoregressively in self-speculative decoding, reaching 2.54× the throughput
of an autoregressive model at comparable quality.

<p align="center">
  <img src="docs/assets/overview.png"
       alt="Lafa overview with AR predictions, flow and flow map decoding, and self-speculative decoding"
       width="100%">
</p>

## Results

<p align="center">
  <img src="docs/assets/main_results.png"
       alt="Lafa quality and throughput frontier"
       width="560">
</p>

Accuracy vs. throughput of Lafa 7B against autoregressive, multi-token
prediction, and diffusion baselines, averaged over IFEval, GSM8K, and HumanEval.
Lafa 7B reaches 2.54× the throughput of an autoregressive model at matched or
better quality.

## Models

All released models use block size 4. Each checkpoint supports flow,
autoregressive, and self-speculative decoding.

| Backbone | Size | Block | Model |
|---|---:|---:|---|
| Qwen2.5 Instruct | 0.5B | 4 | [🤗 `lafa-hf/lafa-0.5B`](https://huggingface.co/lafa-hf/lafa-0.5B) |
| Qwen2.5 Instruct | 7B | 4 | [🤗 `lafa-hf/lafa-7B`](https://huggingface.co/lafa-hf/lafa-7B) |
| Qwen3 Base | 8B | 4 | [🤗 `lafa-hf/lafa-8B`](https://huggingface.co/lafa-hf/lafa-8B) |

## Quick start

Install the inference dependencies, then load a model from Hugging Face. The
model code ships with the checkpoint.

```bash
git clone https://github.com/lafa-gh/large-flow-lms.git
cd large-flow-lms
pip install -r requirements-inference.txt
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "lafa-hf/lafa-0.5B"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda").eval()

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 17 + 25?"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(
    text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

output = model.generate(
    **inputs,
    gen_mode="flow",
    num_steps=4,
    noise_scale_override=0.1,
    cfg_scale=1.0,
    max_new_tokens=128,
)

print(tokenizer.decode(
    output[0, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
))
```

Change the arguments to switch decoding:

| Decoding | Arguments |
|---|---|
| Flow | `gen_mode="flow"`, `num_steps=K` |
| Autoregressive | `gen_mode="ar"` |
| Self-speculative | `gen_mode="speculative"` |
| Confidence thresholding | `gen_mode="flow"`, `num_steps=1`, `threshold=0.9` |

## Guides

Detailed instructions for each part of the pipeline:

| | |
|---|---|
| [**Inference**](docs/inference.md) | Modes, batching, and checkpoint options |
| [**Training**](docs/training.md) | Convert or continue training a Lafa model |
| [**Distillation**](docs/distillation.md) | Train a few-step flow map |
| [**Evaluation**](docs/evaluation.md) | Run benchmarks and checkpoint checks |

## Citation

The arXiv preprint is coming soon. If you use Lafa, please cite:

```bibtex
@article{chavan2026largeflow,
  title   = {Large Flow Language Models},
  author  = {Chavan, Arnav and Lee, Chanhyuk and Bamba, Udbhav and Yoo, Jaehoon
             and Hong, Seunghoon and Raghunathan, Aditi and Kim, Jinwoo},
  journal = {arXiv preprint},
  year    = {2026}
}
```
