"""Check an HF export against the source EMA tensors and legacy decoder."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from src.lafa.flow_model import FlowModelWrapper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--package',required=True)
    p.add_argument('--output',required=True);p.add_argument('--max-new-tokens',type=int,default=32)
    a=p.parse_args();source=Path(a.source);package=Path(a.package)
    cfg=json.loads((package/'config.json').read_text())
    bc=dict(cfg['backbone_config']);kind=bc.pop('model_type')
    with init_empty_weights():
        backbone=AutoModelForCausalLM.from_config(AutoConfig.for_model(kind,**bc),attn_implementation='eager')
        reference=FlowModelWrapper(backbone,noise_sigma=cfg['noise_sigma'],**cfg['flow_config'])
    ema=torch.load(source/'ema.pt',map_location='cpu',weights_only=True)
    state={k:v.to(torch.bfloat16) for k,v in ema.items()}
    state['_embed_init_norm']=torch.zeros(bc['vocab_size'],1,dtype=torch.bfloat16)
    reference.load_state_dict(state,strict=True,assign=True)
    reference=reference.to(device='cuda',dtype=torch.bfloat16).eval()
    reference._ensure_time_warper('cuda').load_state_dict(
        torch.load(source/'time_warper.pt',map_location='cpu',weights_only=True))
    reference._global_step=reference.time_warp_start_step
    reference.noise_scale_mult=.1
    actual=AutoModelForCausalLM.from_pretrained(package,trust_remote_code=True,dtype=torch.bfloat16).to('cuda').eval()
    for key,parameter in actual.named_parameters():
        assert torch.equal(parameter,dict(reference.named_parameters())[key]),key
    tokenizer=AutoTokenizer.from_pretrained(package,trust_remote_code=True)
    cases=[]
    for prompt in ['What is 17 + 25? Answer briefly.',
                   'Write a Python function that returns the larger of two integers.']:
        ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
            add_generation_prompt=True,return_tensors='pt',return_dict=False).to('cuda')
        for name,mode,method,kwargs in [
            ('ar','ar',reference.generate_ar,{}),
            ('flow1','flow',reference.generate,{'num_steps':1,'cfg_scale':8.0,'grid_mode':'warp'}),
            ('flow4','flow',reference.generate,{'num_steps':4,'cfg_scale':8.0,'grid_mode':'warp'}),
            ('speculative','speculative',reference.generate_self_speculative,{'grid_mode':'warp'})]:
            expected,stats=method(ids,max_new_tokens=a.max_new_tokens,seed=42,**kwargs)
            result=actual.generate(ids,gen_mode=mode,max_new_tokens=a.max_new_tokens,
                                  seed=42,return_dict_in_generate=True,**kwargs)
            assert torch.equal(result.sequences,expected),(prompt,name)
            assert result.stats[0]==stats,(name,result.stats[0],stats)
            cases.append({'prompt':prompt,'mode':name,'tokens':expected[0,ids.shape[1]:].tolist(),
                          'stats':stats,'match':True})
            print('PASS',name,flush=True)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'weight_parameters_checked':len(ema),'cases':cases},indent=2))


if __name__=='__main__':main()
