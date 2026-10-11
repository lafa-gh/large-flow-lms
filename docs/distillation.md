# Distillation

[Back to README](../README.md)

Run commands from the repository root.

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

Student checkpoints load with the `AutoModelForCausalLM` example in the
[quick start](../README.md#quick-start);
set `repo` to the saved directory. They need no teacher for inference.
For a student, `gen_mode="flow"` evaluates learned two-time transitions; use
`cfg_scale=1.0`, `grid_mode="warp"`, and an explicit `num_steps` and noise scale.
`ar` and `speculative` also work. Confidence/greedy teacher decoding is rejected
for students. The final output directory holds running weights; checkpoint
`ema.pt` contains the separately maintained weight EMA.

See [training data](training.md#training-data) for the dataset format.
