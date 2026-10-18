"""PSD training, padding, and portable student checkpoint regressions."""

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from src.lafa.data_collator import FlowDataCollator
from src.lafa.configuration_lafa import LafaConfig
from src.lafa.modeling_lafa import LafaForCausalLM
from train_distill import DistillConfig, build_student


def test_psd_frozen_teacher_and_student_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(7)
    backbone = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        eos_token_id=None,
        pad_token_id=0,
    )
    teacher = LafaForCausalLM(
        LafaConfig(
            backbone_config=backbone.to_dict(),
            noise_sigma=0.02,
            flow_config=dict(
                bd_size=4,
                enable_self_cond=True,
                time_warp_num_bins=8,
                adaln_gate_init=1.0,
                adaln_init_scale=0.02,
            ),
        )
    )
    teacher.save_pretrained(tmp_path / "teacher")
    monkeypatch.setattr("train_distill.AutoTokenizer.from_pretrained", lambda *a, **k: None)
    recipe = DistillConfig(distill_teacher_path=str(tmp_path / "teacher"))
    student, _ = build_student(recipe)
    assert not student.embed_tokens.weight.requires_grad
    assert not student.lm_head.weight.requires_grad
    assert all(not p.requires_grad for p in student._teacher.parameters())
    assert not any(k.startswith("_teacher") for k in student.state_dict())
    frozen = {k: v.clone() for k, v in student._teacher.state_dict().items()}
    schedule = student.time_warper.loss_ema.clone()
    ids = torch.arange(48).reshape(4, 12) % 31
    labels = ids.clone()
    labels[:, :3] = -100
    labels[0, -2:] = -100
    mask = torch.ones_like(ids)
    mask[0, -2:] = 0
    student.train()
    torch.manual_seed(42)
    loss = student(ids, attention_mask=mask, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert student.time_embed_tgt.mlp[2].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in student._teacher.parameters())
    torch.testing.assert_close(student.time_warper.loss_ema, schedule)
    for name, value in student._teacher.state_dict().items():
        torch.testing.assert_close(value, frozen[name])
    # Padding IDs cannot change a supervised loss.
    ids[0, -2:] = 30
    torch.manual_seed(42)
    torch.testing.assert_close(student(ids, attention_mask=mask, labels=labels).loss, loss)
    student.eval()
    student.save_pretrained(tmp_path / "student")
    restored = AutoModelForCausalLM.from_pretrained(tmp_path / "student", trust_remote_code=True)
    assert not hasattr(restored, "_teacher")
    prompt = torch.tensor([[2, 3, 4]])
    for mode, steps in [("ar", 1), ("flow", 1), ("flow", 4), ("speculative", 1)]:
        expected = student.generate(prompt, gen_mode=mode, num_steps=steps, max_new_tokens=5)
        actual = restored.generate(prompt, gen_mode=mode, num_steps=steps, max_new_tokens=5)
        assert torch.equal(actual, expected)
        assert actual.shape[1] == 8
    with pytest.raises(ValueError, match="train_distill"):
        restored(ids, labels=labels)
    with pytest.raises(ValueError, match="Student"):
        restored.generate(prompt, gen_mode="confidence")
    resumed, _ = build_student(recipe, str(tmp_path / "student"))
    for name, value in student.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value)
    recipe.noise_scale = 0.1
    with pytest.raises(ValueError, match="Resume recipe"):
        build_student(recipe, str(tmp_path / "student"))


def test_distillation_collator_preserves_real_eos_tokens():
    collate = FlowDataCollator(pad_token_id=0, bd_size=4, return_attention_mask=True)
    batch = collate(
        [
            {"input_ids": [1, 2, 0], "labels": [-100, 2, 0]},
            {"input_ids": [1, 0], "labels": [-100, 0]},
        ]
    )
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
