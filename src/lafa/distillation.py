"""Conditional PSD: frozen teacher anchors and stopped student composition."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from .attention import create_conditional_packed_attention_mask


@dataclass
class _DistillBatch:
    input_ids: Tensor
    token_mask: Tensor
    answer_mask: Tensor
    cond_mask: Tensor
    clean: Tensor
    noise: Tensor
    mask_cond: Tensor
    position_ids: Tensor

    @property
    def batch_size(self):
        return self.input_ids.shape[0]

    @property
    def length(self):
        return self.input_ids.shape[1]


def _prepare_batch(model, input_ids, labels, attention_mask):
    if labels is None:
        raise ValueError("Distillation requires instruction labels")

    B, L = input_ids.shape
    device = input_ids.device
    token_mask = (
        attention_mask.to(device=device, dtype=torch.bool)
        if attention_mask is not None
        else torch.ones(B, L, dtype=torch.bool, device=device)
    )
    answer_mask = (labels != -100) & token_mask
    reference = getattr(model, "_teacher", model)
    if next(reference.parameters()).device != device:
        reference.to(device)
    cond_mask = answer_mask.to(reference.embed_tokens.weight.dtype)
    # Keep the distillation path in the frozen teacher's embedding coordinates.
    with torch.no_grad():
        clean = reference.embed_tokens(input_ids).detach()
        noise = reference._sample_noise(clean)

    mask_cond = create_conditional_packed_attention_mask(
        answer_mask=answer_mask,
        token_mask=token_mask,
        block_size=model.bd_size,
        dtype=clean.dtype,
        block_causal=model.block_causal,
        noisy_causal=model.noisy_causal,
    )
    pos = torch.arange(L, device=device)
    position_ids = torch.cat([pos, pos]).unsqueeze(0).expand(B, -1)
    return _DistillBatch(
        input_ids=input_ids,
        token_mask=token_mask,
        answer_mask=answer_mask,
        cond_mask=cond_mask,
        clean=clean,
        noise=noise,
        mask_cond=mask_cond,
        position_ids=position_ids,
    )


def _path_times(model, clock, length):
    if length % model.bd_size:
        raise ValueError("PSD sequence length must be divisible by bd_size")
    per_block = clock.ndim == 2
    flat = clock.reshape(-1)
    warper = model._ensure_time_warper(clock.device)
    if model.time_warp_per_position:
        mapped = warper.sample_tau_coupled(1.0 - flat)
    else:
        mapped = warper._invert_u_row(1.0 - flat, *warper._get_bin_widths_and_edges())[:, None]
        mapped = mapped.expand(-1, model.bd_size)
    mapped = torch.where(flat[:, None] <= 0, 0.0, mapped)
    mapped = torch.where(flat[:, None] >= 1, 1.0, mapped)
    if per_block:
        return mapped.reshape(clock.shape[0], length)
    return mapped.repeat(1, length // model.bd_size)


def _interpolate(batch, tau):
    mix = tau.unsqueeze(-1).to(batch.clean.dtype)
    answer = batch.cond_mask.unsqueeze(-1)
    noisy = (1.0 - mix) * batch.noise + mix * batch.clean
    return answer * noisy + (1.0 - answer) * batch.clean


def _packed_time(batch, tau):
    clean_time = torch.ones_like(tau)
    noisy_time = batch.cond_mask * tau + (1.0 - batch.cond_mask)
    return torch.cat([clean_time, noisy_time], dim=1)


@torch.no_grad()
def _select_batch(batch, indices):
    return _DistillBatch(
        **{
            field: getattr(batch, field).index_select(0, indices)
            for field in _DistillBatch.__dataclass_fields__
        }
    )


def _sample_psd_times(model, batch_size, device, length):
    """Balance diagonal rows across ranks; independently sample each block."""
    if length % model.bd_size:
        raise ValueError("PSD sequence length must be divisible by bd_size")
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    n_diag = int(batch_size * world * model._distill_diagonal_fraction)
    base, extra = divmod(n_diag, world)
    assignments = torch.arange(batch_size, device=device) < base + int(rank < extra)
    generator = torch.Generator(device=device).manual_seed(model._global_step * world + rank)
    assignments = assignments[torch.randperm(batch_size, device=device, generator=generator)]
    idx_diag = assignments.nonzero(as_tuple=True)[0]
    idx_off = (~assignments).nonzero(as_tuple=True)[0]
    shape = (batch_size, length // model.bd_size)
    diagonal = torch.rand(shape, device=device)
    gap = torch.rand(shape, device=device)
    source = torch.rand(shape, device=device) * (1 - gap)
    target = source + gap
    # Preserve the reference recipe's RNG stream before replacing the random split.
    torch.rand(shape, device=device)
    middle = 0.5 * (source + target)
    for clock in (source, middle, target):
        clock[idx_diag] = diagonal[idx_diag]
    return source, middle, target, idx_diag, idx_off


def _student_forward(
    model, batch, noisy, s, t=None, mask=None, compute_answer_logits=True, compute_clean_logits=None
):
    """Predict the noisy block and, when needed, the clean causal prefix."""
    length = batch.length
    answer = batch.cond_mask.unsqueeze(-1)
    noisy = answer * noisy + (1.0 - answer) * batch.clean
    packed = torch.cat([batch.clean, noisy], dim=1)
    source_times = _packed_time(batch, s)
    source = model._compute_t_bn(source_times)
    target = None
    if t is not None:
        if not model._two_time:
            raise ValueError("PSD distillation requires two_time=True")
        target = model._compute_t_bn(_packed_time(batch, t), tgt=True)
        flow_tokens = torch.cat([torch.zeros_like(batch.cond_mask), batch.cond_mask], dim=1)
        target = target * flow_tokens.unsqueeze(-1).to(target.dtype)
    need_clean = bool(model.ar_loss) if compute_clean_logits is None else compute_clean_logits
    logits, _, hidden = model._forward_backbone(
        packed,
        source,
        batch.mask_cond if mask is None else mask,
        position_ids=batch.position_ids,
        t_bn_tgt=target,
        compute_logits=need_clean,
    )
    answer_hidden = hidden[:, length:]
    if need_clean:
        return logits[:, length:], answer_hidden, logits[:, :length]
    answer_logits = (
        model.lm_head(answer_hidden.to(model.lm_head.weight.dtype))
        if compute_answer_logits
        else None
    )
    return answer_logits, answer_hidden, None


@torch.no_grad()
def _teacher_endpoint(model, batch, noisy, time, mask, return_clean_logits=False):
    """One frozen conditional teacher pass, including clean logits for AR KL."""
    teacher = getattr(model, "_teacher", None)
    if teacher is None:
        raise RuntimeError("Distillation requires a frozen teacher")
    if next(teacher.parameters()).device != noisy.device:
        teacher.to(noisy.device)
    teacher.eval()
    expected_versions = getattr(model, "_teacher_param_versions", None)
    current_params = dict(teacher.named_parameters())
    if expected_versions is None:
        model._teacher_param_versions = tuple(
            (name, param._version) for name, param in current_params.items()
        )
    else:
        for name, version in expected_versions:
            if current_params[name]._version != version:
                raise RuntimeError(f"Frozen teacher parameter was modified: {name}")
    answer = batch.cond_mask.unsqueeze(-1)
    model_input = answer * noisy + (1.0 - answer) * batch.clean
    packed = torch.cat([batch.clean, model_input], dim=1)
    logits, _, hidden = teacher._forward_backbone(
        packed,
        teacher._compute_t_bn(_packed_time(batch, time)),
        mask,
        position_ids=batch.position_ids,
    )
    result = (logits[:, batch.length :].float(), hidden[:, batch.length :])
    if return_clean_logits:
        result += (logits[:, : batch.length].float(),)
    return result


def _masked_mean(values, mask):
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _target_kl_per_token(target, log_probability, chunk_size=16384, direction="forward"):
    """Categorical KL without materializing a full fp32 probability copy."""
    if target.shape != log_probability.shape:
        raise ValueError("KL target and prediction shapes must match")
    if direction not in ("forward", "reverse"):
        raise ValueError("KL direction must be forward or reverse")
    result = log_probability.new_zeros(log_probability.shape[:-1])
    vocab = target.shape[-1]
    for start in range(0, vocab, chunk_size):
        stop = min(start + chunk_size, vocab)
        target_chunk = target[..., start:stop].float()
        log_prediction_chunk = log_probability[..., start:stop]
        if direction == "forward":
            contribution = target_chunk * (
                target_chunk.clamp_min(1e-20).log() - log_prediction_chunk
            )
        else:
            prediction_chunk = log_prediction_chunk.exp()
            contribution = prediction_chunk * (
                log_prediction_chunk - target_chunk.clamp_min(1e-20).log()
            )
        result = result + contribution.sum(dim=-1)
    return result


def _probability_psd_loss(model, input_ids, labels, attention_mask):
    """Stopped-target cross entropy (KL gradient) plus clean-prefix AR KL."""
    batch = _prepare_batch(model, input_ids, labels, attention_mask)
    B, L = batch.batch_size, batch.length
    device = input_ids.device
    clock_s, clock_u, clock_t, idx_diag, idx_off = _sample_psd_times(model, B, device, length=L)
    s0_rows = torch.zeros(B, dtype=torch.bool, device=device)
    s0_fraction = float(getattr(model, "_distill_s0_fraction", 0.0))
    if s0_fraction > 0 and idx_off.numel():
        candidates = idx_off
        count = int(idx_off.numel() * s0_fraction)
        selected = candidates[torch.randperm(candidates.numel(), device=device)[:count]]
        s0_rows[selected] = True
        # Keep sampled clock gap and split; change only the origin, before warping.
        clock_u[selected] -= clock_s[selected]
        clock_t[selected] -= clock_s[selected]
        clock_s[selected] = 0.0
    s = _path_times(model, clock_s, L)
    u = _path_times(model, clock_u, L)
    t = _path_times(model, clock_t, L)
    noisy_s = _interpolate(batch, s)

    diag_noise_scale = float(getattr(model, "_distill_diag_noise_scale", 0.0))
    if diag_noise_scale > 0:
        diag_s = s[idx_diag].unsqueeze(-1).to(batch.noise.dtype)
        noisy_s[idx_diag] = diag_s * batch.clean[idx_diag] + (1.0 - diag_s) * batch.noise[
            idx_diag
        ] * (diag_noise_scale / model._teacher.noise_scale_mult)

    off_noise_scale = float(getattr(model, "_distill_offdiag_noise_scale", 0.0))
    if off_noise_scale > 0:
        ordinary_off = idx_off[~s0_rows[idx_off]]
        source_time = s[ordinary_off].unsqueeze(-1).to(batch.noise.dtype)
        ratio = off_noise_scale / model._teacher.noise_scale_mult
        noisy_s[ordinary_off] = (1.0 - source_time) * (
            batch.noise[ordinary_off] * ratio
        ) + source_time * batch.clean[ordinary_off]

    if s0_rows.any():
        # batch.noise and context stay at the teacher scale.
        ratio = model._distill_s0_noise_scale / model._teacher.noise_scale_mult
        noisy_s[s0_rows] = batch.noise[s0_rows] * ratio

    logits_student, _, clean_logits = _student_forward(
        model,
        batch,
        noisy_s,
        s,
        t=t,
    )
    log_p_student = F.log_softmax(logits_student.float(), dim=-1)

    zero = logits_student.new_zeros(())
    offdiag_ce = offdiag_kl = zero
    loss_weight_components = {}
    if off_noise_scale > 0:
        loss_weight_components["offdiag_noise_scale"] = off_noise_scale
    if s0_fraction > 0:
        loss_weight_components["s0_row_fraction"] = float(s0_rows.float().mean())
        loss_weight_components["s0_noise_scale"] = model._distill_s0_noise_scale
    if idx_off.numel() > 0:
        off_batch = _select_batch(batch, idx_off)
        off_s = s.index_select(0, idx_off)
        off_u = u.index_select(0, idx_off)
        off_t = t.index_select(0, idx_off)
        off_noisy = noisy_s.index_select(0, idx_off)
        dtype = off_noisy.dtype
        with torch.no_grad():
            p_su = F.softmax(
                _student_forward(
                    model,
                    off_batch,
                    off_noisy,
                    off_s,
                    t=off_u,
                )[0].float(),
                dim=-1,
            )
            embed = model.embed_tokens.weight
            s_coeff = off_s.float().unsqueeze(-1)
            u_coeff = off_u.float().unsqueeze(-1)
            t_coeff = off_t.float().unsqueeze(-1)
            denoised_su = p_su.to(dtype) @ embed
            state_u = ((1.0 - u_coeff) / (1.0 - s_coeff).clamp_min(1e-8)) * off_noisy.float() + (
                (u_coeff - s_coeff) / (1.0 - s_coeff).clamp_min(1e-8)
            ) * denoised_su.float()
            answer = off_batch.cond_mask.unsqueeze(-1)
            state_u = answer * state_u + (1.0 - answer) * off_batch.clean
            p_ut = F.softmax(
                _student_forward(
                    model,
                    off_batch,
                    state_u.to(dtype),
                    off_u,
                    t=off_t,
                )[0].float(),
                dim=-1,
            )
            gamma = (
                ((1.0 - t_coeff) * (u_coeff - s_coeff))
                / ((1.0 - u_coeff) * (t_coeff - s_coeff)).clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            target_off = gamma * p_su + (1.0 - gamma) * p_ut

        log_p_off = log_p_student.index_select(0, idx_off)
        ce_pp = -(target_off * log_p_off).sum(dim=-1)
        kl_pp = _target_kl_per_token(target_off, log_p_off, direction="forward")
        off_weight = off_batch.answer_mask & (off_t > off_s + 1e-8)
        if s0_fraction > 0:
            selected_off = s0_rows.index_select(0, idx_off).unsqueeze(-1)
            for name, rows in (("s0", selected_off), ("interpolant", ~selected_off)):
                selected_mask = off_weight & rows
                if selected_mask.any():
                    loss_weight_components[f"offdiag_{name}_kl"] = float(
                        _masked_mean(kl_pp, selected_mask).detach()
                    )
                    loss_weight_components[f"offdiag_{name}_ce"] = float(
                        _masked_mean(ce_pp, selected_mask).detach()
                    )
        offdiag_ce = _masked_mean(ce_pp, off_weight)
        offdiag_kl = _masked_mean(kl_pp, off_weight)

    diag_ce = diag_kl = zero
    clean_teacher_kl = zero
    use_clean_teacher_kl = model.ar_loss and getattr(model, "ar_loss_type", "ce") in ("kl", "both")
    if idx_diag.numel() > 0:
        diag_batch = _select_batch(batch, idx_diag)
        diag_noisy = noisy_s.index_select(0, idx_diag)
        diag_t = s.index_select(0, idx_diag)
        log_p_diag = log_p_student.index_select(0, idx_diag)
        with torch.no_grad():
            teacher_result = _teacher_endpoint(
                model,
                diag_batch,
                diag_noisy,
                diag_t,
                diag_batch.mask_cond,
                return_clean_logits=use_clean_teacher_kl,
            )
            teacher_logits, _ = teacher_result[:2]
            target_diag = F.softmax(teacher_logits, dim=-1)
            if use_clean_teacher_kl:
                teacher_clean_logits = teacher_result[2]
                teacher_clean = F.softmax(teacher_clean_logits[:, :-1].float(), dim=-1)
                teacher_clean_log = teacher_clean.clamp_min(1e-20).log()
        if use_clean_teacher_kl:
            student_clean_logits = clean_logits.index_select(0, idx_diag)
            student_clean_log = F.log_softmax(student_clean_logits[:, :-1].float(), dim=-1)
            clean_kl_pp = (teacher_clean * (teacher_clean_log - student_clean_log)).sum(dim=-1)
            clean_teacher_kl = _masked_mean(clean_kl_pp, diag_batch.answer_mask[:, 1:])
        ce_pp = -(target_diag * log_p_diag).sum(dim=-1)
        target_log = target_diag.clamp_min(1e-20).log()
        kl_pp = (target_diag * (target_log - log_p_diag)).sum(dim=-1)
        diag_ce = _masked_mean(ce_pp, diag_batch.answer_mask)
        diag_kl = _masked_mean(kl_pp, diag_batch.answer_mask)

    offdiag_term = (idx_off.numel() / max(B, 1)) * offdiag_ce
    diag_term = (idx_diag.numel() / max(B, 1)) * diag_ce
    loss = offdiag_term + diag_term
    components = {
        "psd": float(loss.detach()),
        "offdiag_weighted": float(offdiag_term.detach()),
        "diag_weighted": float(diag_term.detach()),
        "offdiag_cond_kl": float(offdiag_kl.detach()),
        "diag_cond_kl": float(diag_kl.detach()),
        "offdiag_cond_ce": float(offdiag_ce.detach()),
        "diag_cond_ce": float(diag_ce.detach()),
    }
    components.update(loss_weight_components)
    if use_clean_teacher_kl:
        clean_kl_term = model.ar_loss_weight * clean_teacher_kl
        loss = loss + clean_kl_term
        components["clean_teacher_kl"] = float(clean_teacher_kl.detach())
        components["clean_teacher_kl_weighted"] = float(clean_kl_term.detach())
    components["total"] = float(loss.detach())
    model._loss_components = components
    return CausalLMOutputWithPast(loss=loss, logits=logits_student)


def distillation_loss(model, input_ids, labels, attention_mask):
    return _probability_psd_loss(model, input_ids, labels, attention_mask)
