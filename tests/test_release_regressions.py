"""Regression checks for public checkpoint loading and generation controls."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from src.lafa.flow_model import FlowModelWrapper
from src.lafa.init_checkpoint import load_init_state_dict, normalize_init_state_dict
from src.evaluation.harness import AR2FlowEvalHarness, _generation_settings


def tiny_backbone():
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False,
        eos_token_id=None, attention_dropout=0.0))


def tiny_flow(dtype=torch.float32):
    torch.manual_seed(42)
    return FlowModelWrapper(tiny_backbone(), bd_size=4,
                            enable_self_cond=False).to(dtype).eval()


def test_standard_ar_checkpoint_loads_every_backbone_tensor():
    torch.manual_seed(1)
    ar = tiny_backbone().state_dict()
    model = tiny_flow()
    assert not torch.equal(ar['model.embed_tokens.weight'], model.embed_tokens.weight)
    result = load_init_state_dict(model, ar)
    assert not result.unexpected_keys
    for name, value in ar.items():
        torch.testing.assert_close(model.state_dict()['backbone.' + name], value)


def test_native_flow_checkpoint_roundtrip():
    model = tiny_flow()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.embed_tokens.weight.zero_()
    result = load_init_state_dict(model, state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.testing.assert_close(model.embed_tokens.weight, state['backbone.model.embed_tokens.weight'])


def test_unrelated_and_ambiguous_checkpoints_are_rejected():
    model = tiny_flow()
    with pytest.raises(ValueError, match='no matching backbone'):
        load_init_state_dict(model, {'unrelated': torch.ones(1)})
    w = model.embed_tokens.weight.detach()
    with pytest.raises(ValueError, match='Multiple checkpoint keys'):
        normalize_init_state_dict(model, {
            'model.embed_tokens.weight': w,
            'backbone.model.embed_tokens.weight': w})


@pytest.mark.parametrize('requested,expected', [(1024,640),(128,128),(None,640)])
def test_task_generation_limit_cannot_exceed_model_limit(requested, expected):
    assert _generation_settings({'max_gen_toks': requested}, 640) == (expected,0.0)
    assert _generation_settings({},640) == (640,0.0)


def test_invalid_mode_fails_before_loading_checkpoint():
    with pytest.raises(ValueError, match='Unsupported gen_mode'):
        AR2FlowEvalHarness(pretrained='/nonexistent', gen_mode='flowmap')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_flow_generation_respects_model_dtype(dtype):
    model = tiny_flow(dtype)
    prompt = torch.tensor([[3,4,5]])
    output, stats = model.generate(prompt, max_new_tokens=5, num_steps=2)
    assert output.shape == (1,8)
    assert stats['gen_len'] == 5
    assert torch.equal(output[:,:3],prompt)


def test_fused_self_speculation_matches_fp32_greedy_ar():
    model = tiny_flow()
    prompt = torch.tensor([[3,4,5]])
    ar, _ = model.generate_ar(prompt,max_new_tokens=13)
    speculative, _ = model.generate_self_speculative(prompt,max_new_tokens=13)
    assert torch.equal(ar,speculative)


def test_tied_ar_checkpoint_initializes_both_untied_flow_weights():
    model = tiny_flow()
    ar = tiny_backbone().state_dict()
    ar.pop('lm_head.weight')
    with pytest.raises(ValueError, match='tie_word_embeddings'):
        load_init_state_dict(model, ar)
    result = load_init_state_dict(model, ar, tie_word_embeddings=True)
    assert not result.unexpected_keys
    torch.testing.assert_close(model.embed_tokens.weight, ar['model.embed_tokens.weight'])
    torch.testing.assert_close(model.lm_head.weight, ar['model.embed_tokens.weight'])
    assert model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_batched_flow_generation_dtype(dtype):
    model = tiny_flow(dtype)
    prompt = torch.tensor([[3,4,5],[6,7,8]])
    outputs, stats = model.generate_batched(
        prompt, torch.ones_like(prompt), max_new_tokens=5, num_steps=2)
    assert len(outputs) == 2
    for output in outputs:
        assert len(output) == 8


def test_batched_ar_reports_one_forward_per_generated_token():
    model = tiny_flow()
    prompt = torch.tensor([[3,4,5],[6,7,8]])
    outputs, stats = model.generate_ar_batched(
        prompt, torch.ones_like(prompt), max_new_tokens=5)
    assert stats['forwards'] == 5
    assert stats['per_seq_forwards'] == [5,5]
    assert sum(len(output)-3 for output in outputs) / sum(stats['per_seq_forwards']) == 1.0


def test_training_loss_and_backward_are_finite():
    model = tiny_flow().train()
    tokens = torch.tensor([[3,4,5,6,7,8,9,10]])
    labels = tokens.clone()
    labels[:,:3] = -100
    loss = model(input_ids=tokens, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
