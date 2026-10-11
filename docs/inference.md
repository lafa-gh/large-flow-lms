# Inference

[Back to README](../README.md)

Run commands from the repository root.

For private repositories, authenticate with `hf auth login` using an account
with access. The model code is packaged with the weights; inference does not
require installing this source repository or downloading the original Qwen weights.

See the [quick start](../README.md#quick-start) for installation and a complete generation example.

## Decoding modes and batching

Choose `gen_mode="ar"`, `"flow"`, `"confidence"`, `"greedy"`, or `"speculative"`.
The default output is a token tensor; `return_dict_in_generate=True` returns
`sequences` and per-sample `stats`. The HF interface implements greedy generation.
Self-speculative decoding supports batched prompts with independent acceptance,
EOS handling, and per-sample forward counts. Other HF generation modes process
batch rows sequentially. Returned prefixes preserve input padding. Beam search,
sampling controls, and automatic multi-device inference dispatch are not supported.
The evaluation harness also supports batched teacher decoding; set
`gen_mode=speculative,gen_batch_size=8` for self-speculation. TPF is total generated
tokens divided by the sum of per-sample decoding forwards, excluding prefill;
it does not multiply by batch size. BF16 rounding can change greedy token choices and
acceptance when batch shapes change.

## Checkpoint variants

The root model has EMA applied to its weights. Use `subfolder="raw"` to select
original weights when using an exported raw variant. Model settings, including
noise calibration and the legacy rotary-buffer precision, are stored in the
integrated `config.json`. The learned schedule is loaded from `time_warper.pt`.
Use a fixed `revision` for both model and tokenizer to pin an experiment.
