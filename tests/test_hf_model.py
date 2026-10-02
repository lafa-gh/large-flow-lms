import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config
from src.lafa.configuration_lafa import LafaConfig
from src.lafa.modeling_lafa import LafaForCausalLM


def tiny_model():
    bc = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=128, tie_word_embeddings=False,
                    eos_token_id=None, pad_token_id=0)
    torch.manual_seed(42)
    return LafaForCausalLM(LafaConfig(
        backbone_config=bc.to_dict(), noise_sigma=0.02,
        flow_config={'bd_size':4, 'enable_self_cond':False, 'freeze_embed':False,
                     'freeze_lm_head':False, 'ar_loss':True,
                     'time_warp_num_bins':8, 'time_warp_per_position':True})).eval()


def test_autoclass_roundtrip_without_base_weights(tmp_path):
    model = tiny_model()
    model._ensure_time_warper('cpu').loss_ema.copy_(torch.rand(4,8)+0.5)
    model.save_pretrained(tmp_path, max_shard_size='10KB')
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    assert config.model_type == 'lafa'
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True)
    assert restored._noise_sigma == model._noise_sigma
    assert restored.bd_size == 4
    assert not restored.embed_tokens.weight.is_meta
    torch.testing.assert_close(restored.time_warper.loss_ema, model.time_warper.loss_ema)
    assert model.state_dict().keys() == restored.state_dict().keys()
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,restored.state_dict()[k])
    prompt = torch.tensor([[2,3,4]])
    for mode in ['ar','flow','confidence','speculative']:
        expected = model.generate(prompt,gen_mode=mode,max_new_tokens=5)
        actual = restored.generate(prompt,gen_mode=mode,max_new_tokens=5)
        assert torch.equal(actual,expected)
    restored.save_pretrained(tmp_path/'again')
    again = AutoModelForCausalLM.from_pretrained(tmp_path/'again',trust_remote_code=True)
    torch.testing.assert_close(again.time_warper.loss_ema,model.time_warper.loss_ema)


def test_hf_forward_and_training():
    model = tiny_model()
    ids = torch.tensor([[2,3,4,5,6,7,8,9]])
    assert model(ids,attention_mask=torch.ones_like(ids)).logits.shape == (1,8,32)
    labels=ids.clone();labels[:,:3]=-100
    model.train()
    out=model(ids,labels=labels,attention_mask=torch.ones_like(ids))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.lm_head.weight.grad is not None


def test_padding_return_type_and_zero_budget():
    model=tiny_model()
    ids=torch.tensor([[0,2,3],[4,5,6]])
    mask=torch.tensor([[0,1,1],[1,1,1]])
    result=model.generate(input_ids=ids,attention_mask=mask,gen_mode='ar',
                          max_new_tokens=3,return_dict_in_generate=True)
    assert result.sequences.shape == (2,6)
    assert torch.equal(result.sequences[:,:3],ids)
    assert len(result.stats)==2
    assert torch.equal(model.generate(ids,max_new_tokens=0),ids)


def test_raw_subfolder(tmp_path):
    tiny_model().save_pretrained(tmp_path)
    tiny_model().save_pretrained(tmp_path/'raw')
    model=AutoModelForCausalLM.from_pretrained(tmp_path,subfolder='raw',trust_remote_code=True)
    assert model.config.weight_variant=='raw'


@pytest.mark.parametrize('kwargs',[{'num_beams':2},{'do_sample':True},{'top_p':0.9}])
def test_unsupported_generation_rejected(kwargs):
    with pytest.raises(ValueError):tiny_model().generate(torch.tensor([[2,3]]),**kwargs)


def test_bf16_rotary_buffers_preserve_native_precision(tmp_path):
    model=tiny_model().to(torch.bfloat16)
    model.save_pretrained(tmp_path)
    loaded=AutoModelForCausalLM.from_pretrained(
        tmp_path,trust_remote_code=True,dtype=torch.bfloat16)
    for name,value in model.named_buffers():
        other=dict(loaded.named_buffers())[name]
        assert value.dtype == other.dtype
        torch.testing.assert_close(value,other)


def test_qwen3_shared_adaln_roundtrip(tmp_path):
    from transformers import Qwen3Config
    bc=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                  num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
                  head_dim=8,tie_word_embeddings=False,eos_token_id=None)
    model=LafaForCausalLM(LafaConfig(backbone_config=bc.to_dict(),
        flow_config={'bd_size':4,'adaln_share_layers':True,'enable_self_cond':True,
                     'freeze_embed':False,'freeze_lm_head':False})).eval()
    ids=torch.tensor([[2,3,4,5]])
    expected=model.generate(ids,gen_mode='speculative',max_new_tokens=7)
    model.save_pretrained(tmp_path)
    restored=AutoModelForCausalLM.from_pretrained(tmp_path,trust_remote_code=True)
    assert restored._adaln_share_layers
    assert torch.equal(restored.generate(ids,gen_mode='speculative',max_new_tokens=7),expected)
