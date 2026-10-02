"""Export a native teacher checkpoint to a self-contained EMA/raw HF package."""
import argparse
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import AutoTokenizer
from src.lafa.configuration_lafa import LafaConfig
from src.lafa.modeling_lafa import LafaForCausalLM
from src.lafa.flow_model import FlowModelWrapper


def checksum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export(source, output, preset, base_model, noise_sigma, repo_id, ema_only=False):
    source, output = Path(source), Path(output)
    state = None if ema_only else load_file(str(source / 'model.safetensors'))
    ema = torch.load(source / 'ema.pt', map_location='cpu', weights_only=True)
    if ema_only:
        state = dict(ema)
    warper = torch.load(source / 'time_warper.pt', map_location='cpu', weights_only=True)
    backbone = json.loads((source / 'config.json').read_text())
    backbone['architectures'] = ['Qwen3ForCausalLM' if backbone['model_type'] == 'qwen3'
                                 else 'Qwen2ForCausalLM']
    settings = json.loads(Path(preset).read_text())
    allowed = set(inspect.signature(FlowModelWrapper.__init__).parameters) - {'self','backbone'}
    flow = {key: value for key,value in settings.items() if key in allowed}
    hist = warper['loss_ema']
    if hist.ndim != 2:
        raise ValueError('These release checkpoints must have per-position schedules')
    flow.update(bd_size=hist.shape[0], time_warp_num_bins=hist.shape[1],
                time_warp_per_position=True, time_warp_init_path=None,
                freeze_embed=False, freeze_lm_head=False)
    adaln = [key for key in state if key.startswith('adaln_layers.') and
             key.endswith('adaLN_modulation.1.weight')]
    if not adaln:
        raise ValueError('Checkpoint has no AdaLN tensors')
    flow['adaln_share_layers'] = len(adaln) == 1
    flow['adaln_no_shift'] = state[adaln[0]].shape[0] == 4 * backbone['hidden_size']
    if {'sc_lin_a.weight','sc_lin_b.weight'} <= state.keys():
        flow.update(enable_self_cond=True,sc_mode='residual')
    elif not any(key.startswith('sc_') for key in state):
        flow['enable_self_cond'] = False
    else:
        raise ValueError('Unexpected self-conditioning tensor layout; inspect before export')
    provenance = {
        'base_model': base_model, 'source_checkpoint': source.name,
        'metadata_source': 'architecture inferred from weights/warper; training defaults from release preset',
        'preset': Path(preset).name, 'noise_sigma_source': 'explicit export argument, matching legacy loader calibration',
        'source_weights_sha256': None if ema_only else checksum(source/'model.safetensors'),
        'source_ema_sha256': checksum(source/'ema.pt'),
    }
    cfg = LafaConfig(backbone_config=backbone, flow_config=flow,
                     noise_sigma=noise_sigma, weight_variant='raw', provenance=provenance,
                     dtype='bfloat16')
    with init_empty_weights():
        model = LafaForCausalLM(cfg)
    # Match the existing inference path's BF16 model and buffer conversion.
    state = {key: value.to(torch.bfloat16) if value.is_floating_point() else value
             for key,value in state.items()}
    if ema_only:
        state['_embed_init_norm'] = torch.zeros(backbone['vocab_size'], 1, dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    model._ensure_time_warper('cpu').load_state_dict(warper)
    expected = dict(model.named_parameters())
    if set(ema) != set(expected):
        raise ValueError(f'EMA keys differ: missing={set(expected)-set(ema)}, extra={set(ema)-set(expected)}')
    for name,value in ema.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid EMA tensor: {name}')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    if not ema_only:
        model.save_pretrained(output/'raw', max_shard_size='5GB')
        tokenizer.save_pretrained(output/'raw')
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            parameter.copy_(ema[name].to(parameter.dtype))
    model.config.weight_variant = 'ema'
    model.save_pretrained(output, max_shard_size='5GB')
    tokenizer.save_pretrained(output)
    root = Path(__file__).resolve().parents[1]
    for folder in ([output] if ema_only else [output,output/'raw']):
        for filename in ('LICENSE','NOTICE','requirements-inference.txt'):
            if (root/filename).exists():
                import shutil
                shutil.copyfile(root/filename,folder/filename)
    card = f'''---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: {base_model}
tags:
- lafa
- flow-matching
- custom_code
---

# {repo_id}

LAFA block-flow language model based on {base_model}.
Block size: {flow['bd_size']}. EMA weights in BF16.

Code and usage: [GitHub](https://github.com/lafa-gh).
'''
    (output/'README.md').write_text(card)
    report={'repo_id':repo_id,'source_checkpoint':source.name,'flow_config':flow,
            'noise_sigma':noise_sigma,'provenance':provenance,'files':{}}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'export_manifest.json':report['files'][str(path.relative_to(output))]={'bytes':path.stat().st_size,'sha256':checksum(path)}
    (output/'export_manifest.json').write_text(json.dumps(report,indent=2))
    print('EXPORTED',repo_id,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','output','preset','base-model','repo-id'):p.add_argument('--'+arg,required=True)
    p.add_argument('--noise-sigma',type=float,required=True)
    p.add_argument('--ema-only',action='store_true')
    args=p.parse_args()
    export(args.source,args.output,args.preset,args.base_model,args.noise_sigma,args.repo_id,args.ema_only)
