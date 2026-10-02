"""Check teacher checkpoint generation, optionally against a reference JSON.

Requires a local checkpoint with EMA and a time warper. This checks generation
regressions on two prompts; it does not reproduce a benchmark score.
"""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', help='JSON from an earlier run of this script')
    args = parser.parse_args()
    sys.path.insert(0, args.code_root)

    import torch
    from transformers import AutoTokenizer
    from src.evaluation.harness import _build_model

    model, checkpoint, _ = _build_model(
        args.checkpoint, args.base_model, 'cuda', noise_scale_override=0.1)
    model.load_ema_state_dict(torch.load(
        Path(checkpoint) / 'ema.pt', map_location='cpu', weights_only=True), device='cuda')
    assert model.has_ema, 'Checkpoint has no usable EMA weights'
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True,
        trust_remote_code=getattr(model.config,'model_type',None)=='lafa')
    result = {
        'checkpoint': checkpoint,
        'code_root': args.code_root,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'accelerate', 'datasets', 'lm_eval')},
        'noise_sigma': model._noise_sigma,
        'settings': {'max_new_tokens': 32, 'seed': 42, 'use_ema': True,
                     'noise_scale_override': 0.1, 'flow_cfg_scale': 8.0,
                     'grid_mode': 'warp'},
        'cases': [],
    }
    prompts = [
        'What is 17 + 25? Answer briefly.',
        'Write a Python function that returns the larger of two integers.',
    ]
    modes = [
        ('ar', model.generate_ar, {}),
        ('flow1', getattr(model,'generate_flow',model.generate), {'num_steps': 1, 'cfg_scale': 8.0, 'grid_mode': 'warp'}),
        ('flow4', getattr(model,'generate_flow',model.generate), {'num_steps': 4, 'cfg_scale': 8.0, 'grid_mode': 'warp'}),
        ('speculative', model.generate_self_speculative,
         {'grid_mode': 'warp', 'self_cond': False}),
    ]
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], add_generation_prompt=True,
            return_tensors='pt', return_dict=False).to('cuda')
        for mode, generate, kwargs in modes:
            torch.manual_seed(42)
            torch.cuda.synchronize()
            start = time.monotonic()
            with model.ema_swap():
                output, stats = generate(ids, max_new_tokens=32, **kwargs)
            torch.cuda.synchronize()
            result['cases'].append({
                'prompt': prompt, 'mode': mode,
                'tokens': output[0, ids.shape[1]:].tolist(),
                'stats': stats, 'seconds': time.monotonic() - start,
            })
            print(f'{mode}: {len(result["cases"][-1]["tokens"])} tokens', flush=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
        assert reference['checkpoint'] == result['checkpoint'], 'Checkpoint mismatch'
        assert len(reference['cases']) == len(result['cases']), 'Case count mismatch'
        for expected, actual in zip(reference['cases'], result['cases']):
            assert (expected['prompt'], expected['mode']) == (actual['prompt'], actual['mode'])
            assert expected['tokens'] == actual['tokens'], f'Tokens changed: {actual["mode"]}'
            assert expected['stats'] == actual['stats'], f'Stats changed: {actual["mode"]}'
        print(f'PASS: {len(result["cases"])} cases match reference tokens and decoding stats')


if __name__ == '__main__':
    main()
