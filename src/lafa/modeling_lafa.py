"""Transformers interface backed by the same Lafa runtime as the trainer."""
import copy
import inspect
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from dataclasses import dataclass

from .configuration_lafa import LafaConfig
from .flow_model import FlowModelWrapper
from .time_warping import TimeWarping
# Direct imports also make every transitive module available in HF's local code cache.
from .attention import create_packed_attention_mask
from .base_wrapper import BaseModelWrapper
from .cache import trim_kv_cache, gather_kv_cache
from .layers import TimestepEmbedder, DiTAdaLN
from .loss import cross_entropy_loss


@dataclass
class LafaGenerateOutput(ModelOutput):
    sequences: torch.LongTensor = None
    stats: list = None


class LafaForCausalLM(FlowModelWrapper):
    config_class = LafaConfig
    base_model_prefix = 'backbone'
    accepts_loss_kwargs = False  # Flow loss is already normalized, not summed over tokens.
    _supports_sdpa = False
    _no_split_modules = ['Qwen2DecoderLayer', 'Qwen3DecoderLayer', 'DiTAdaLN']

    def __init__(self, config):
        if not config.backbone_config:
            raise ValueError('Lafa config must contain backbone_config')
        bc = dict(config.backbone_config)
        model_type = bc.pop('model_type')
        backbone_config = AutoConfig.for_model(model_type, **bc)
        backbone_config._attn_implementation = 'eager'
        backbone = AutoModelForCausalLM.from_config(backbone_config, attn_implementation='eager')
        allowed = set(inspect.signature(FlowModelWrapper.__init__).parameters) - {'self', 'backbone'}
        unknown = set(config.flow_config) - allowed
        if unknown:
            raise ValueError(f'Unknown Lafa architecture/training settings: {sorted(unknown)}')
        args = dict(config.flow_config)
        args['noise_sigma'] = config.noise_sigma
        super().__init__(backbone, **args)
        self.config = config
        self.generation_config = GenerationConfig(
            max_new_tokens=128, do_sample=False,
            eos_token_id=config.eos_token_id, pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
        )
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Resolve the auxiliary schedule at the same revision/subfolder as weights.
        from transformers.utils.hub import cached_file
        resolution = {key: kwargs[key] for key in
                      ('revision', 'subfolder', 'cache_dir', 'token', 'local_files_only',
                       'force_download') if key in kwargs}
        output_loading_info = kwargs.get('output_loading_info', False)
        result = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model = result[0] if output_loading_info else result
        # Native Lafa inference casts rotary caches together with all model buffers.
        # HF otherwise recreates these non-persistent buffers in FP32, changing drafts.
        buffer_dtype = (model.embed_tokens.weight.dtype
                        if model.config.rotary_buffer_dtype == 'model' else torch.float32)
        for module in model.backbone.modules():
            for name in module._non_persistent_buffers_set:
                value = module._buffers.get(name)
                if value is not None and value.is_floating_point():
                    module._buffers[name] = value.to(dtype=buffer_dtype)
        path = cached_file(pretrained_model_name_or_path, 'time_warper.pt', **resolution)
        state = torch.load(path, map_location='cpu', weights_only=True)
        model._ensure_time_warper('cpu').load_state_dict(state)
        # Loaded schedules are immediately active for inference and additional training.
        model._global_step = model.time_warp_start_step
        return result

    def save_pretrained(self, save_directory, *args, **kwargs):
        # Keep runtime changes made by fine-tuning in the exported configuration.
        for key in tuple(self.config.flow_config):
            if hasattr(self, key) and key not in ('noise_sigma', 'time_warp_init_path'):
                value = getattr(self, key)
                if isinstance(value, (bool, str, int, float)):
                    self.config.flow_config[key] = value
        self.config.noise_sigma = self._noise_sigma
        super().save_pretrained(save_directory, *args, **kwargs)
        if not kwargs.get('is_main_process', True):
            return
        output = Path(save_directory)
        warper = self._ensure_time_warper('cpu' if self.embed_tokens.weight.is_meta
                                          else self.embed_tokens.weight.device)
        torch.save(warper.state_dict(), output / 'time_warper.pt')
        # Flat, relative imports let AutoClass work without the training repository.
        source = Path(__file__).resolve().parent
        for name in ('configuration_lafa.py', 'modeling_lafa.py', 'flow_model.py',
                     'base_wrapper.py', 'layers.py', 'time_warping.py', 'attention.py',
                     'cache.py', 'loss.py'):
            if (source / name).resolve() != (output / name).resolve():
                shutil.copyfile(source / name, output / name)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def set_input_embeddings(self, value):
        return self.backbone.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.backbone.set_output_embeddings(value)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=None, position_ids=None,
                return_dict=True, **kwargs):
        if input_ids is None:
            raise ValueError('input_ids is required')
        if kwargs:
            # Trainer supplies this only for loss normalization; the flow loss is already normalized.
            unknown = set(kwargs) - {'num_items_in_batch'}
            if unknown:
                raise ValueError(f'Unsupported forward arguments: {sorted(unknown)}')
        if labels is not None:
            if self._two_time:
                raise ValueError('Use train_distill.py to train a student with its frozen teacher')
            if attention_mask is not None:
                labels = labels.masked_fill(~attention_mask.bool(), -100)
            output = super().forward(input_ids=input_ids, labels=labels)
        else:
            embeds = self.embed_tokens(input_ids)
            times = torch.ones(input_ids.shape, device=embeds.device, dtype=embeds.dtype)
            t_bn = self._compute_t_bn(times)
            if position_ids is None and attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
                position_ids = position_ids[:, -input_ids.shape[1]:]
            logits, past, _ = self._forward_backbone(
                embeds, t_bn, attention_mask, past_key_values=past_key_values,
                use_cache=bool(use_cache), position_ids=position_ids)
            output = CausalLMOutputWithPast(logits=logits, past_key_values=past)
        return output if return_dict else output.to_tuple()

    # Legacy trainer/harness API continues to return (tokens, statistics).
    def generate_flow(self, *args, **kwargs):
        return FlowModelWrapper.generate(self, *args, **kwargs)

    @torch.no_grad()
    def generate(self, inputs=None, generation_config=None, *, input_ids=None,
                 attention_mask=None, gen_mode='flow', max_new_tokens=None,
                 num_steps=4, grid_mode='warp', cfg_scale=1.0,
                 noise_scale_override=0.1, seed=42, return_dict_in_generate=False,
                 do_sample=None, temperature=None, self_cond=None, **kwargs):
        ids = input_ids if input_ids is not None else inputs
        if ids is None or ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('Expected nonempty input_ids with shape [batch, sequence]')
        if gen_mode not in ('ar', 'flow', 'speculative'):
            raise ValueError(f'Unsupported gen_mode={gen_mode!r}')
        gc = generation_config or self.generation_config
        limit = int(max_new_tokens if max_new_tokens is not None else gc.max_new_tokens or 128)
        if limit < 0:
            raise ValueError('max_new_tokens must be non-negative')
        if do_sample or (do_sample is None and gc.do_sample) or (temperature not in (None, 0, 0.0)):
            raise ValueError('The HF generation interface currently supports greedy decoding only')
        if kwargs.pop('num_beams', 1) != 1:
            raise ValueError('Beam search is not supported')
        if attention_mask is None:
            attention_mask = torch.ones_like(ids)
        if attention_mask.shape != ids.shape or not attention_mask.bool().any(dim=1).all():
            raise ValueError('attention_mask must match input_ids and retain a nonempty prompt')
        eos = kwargs.pop('eos_token_id', gc.eos_token_id)
        pad = kwargs.pop('pad_token_id', gc.pad_token_id)
        pad = pad if pad is not None else (eos[0] if isinstance(eos, list) else eos)
        if pad is None:
            pad = 0
        params = {'max_new_tokens': limit, 'seed': seed}
        if gen_mode == 'ar':
            method = self.generate_ar
        elif gen_mode == 'speculative':
            method = self.generate_self_speculative
            params.update(grid_mode=grid_mode)
        else:
            method = self.generate_flow
            params.update(num_steps=num_steps, grid_mode=grid_mode, cfg_scale=cfg_scale)
        unknown = set(kwargs) - set(inspect.signature(method).parameters)
        # generate_flow delegates to the legacy implementation's signature.
        if gen_mode == 'flow':
            unknown = set(kwargs) - set(inspect.signature(FlowModelWrapper.generate).parameters)
        if unknown:
            raise ValueError(f'Unsupported generation arguments: {sorted(unknown)}')
        params.update(kwargs)
        saved_eos, saved_noise = self.backbone.config.eos_token_id, self.noise_scale_mult
        saved_sc = self.enable_self_cond
        rows, stats = [], []
        try:
            self.backbone.config.eos_token_id = eos
            if noise_scale_override is not None:
                self.noise_scale_mult = float(noise_scale_override)
            # None keeps the checkpoint's flow self-conditioning; False drops its extra forward per block.
            if self_cond is not None:
                self.enable_self_cond = bool(self_cond) and saved_sc
            for row, mask in zip(ids, attention_mask):
                prompt = row[mask.bool()].unsqueeze(0).to(self.embed_tokens.weight.device)
                if limit:
                    tokens, record = method(prompt, **params)
                    continuation = tokens[0, prompt.shape[1]:]
                else:
                    continuation = prompt[0, :0]
                    record = {'total_steps': 0, 'gen_len': 0}
                rows.append(torch.cat([row.to(prompt.device), continuation]))
                stats.append(record)
        finally:
            self.backbone.config.eos_token_id, self.noise_scale_mult = saved_eos, saved_noise
            self.enable_self_cond = saved_sc
        size = max(row.numel() for row in rows)
        sequences = torch.full((len(rows), size), pad, dtype=ids.dtype, device=rows[0].device)
        for i, row in enumerate(rows):
            sequences[i, :row.numel()] = row
        if return_dict_in_generate:
            return LafaGenerateOutput(sequences=sequences, stats=stats)
        return sequences
