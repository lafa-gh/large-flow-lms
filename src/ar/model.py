"""AR Model Wrapper for assistant-response next-token training.

Checkpoints strip the 'backbone.' prefix via BaseModelWrapper.state_dict(),
so they're loadable directly by AutoModelForCausalLM.from_pretrained().
"""

import torch
from torch import Tensor

from transformers.modeling_outputs import CausalLMOutputWithPast

from src.utils.base_wrapper import BaseModelWrapper
from src.utils.loss import cross_entropy_loss


class ARModelWrapper(BaseModelWrapper):
    """Wraps a pretrained CausalLM for standard autoregressive training.

    Training forward:
      1. Receives input_ids [B, L] and labels [B, L] from collator
      2. Embeds input_ids, runs backbone with causal attention
      3. NTP loss: logits[i] predicts labels[i+1]

    The collator sets labels=-100 on prompt tokens, so only assistant
    response positions contribute to the loss.
    """

    def forward(self, input_ids=None, labels=None, attention_mask=None):
        if input_ids is None or labels is None:
            raise ValueError(
                "AR training requires input_ids and response-masked labels"
            )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits

        loss = cross_entropy_loss(logits, labels, shift=True)

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ):
        """Standard AR generation using HF's generate() API.

        Returns:
            (output_ids [1, P+gen_len], stats_dict)
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        device = next(self.backbone.parameters()).device
        prompt_ids = prompt_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        output_ids = self.backbone.generate(prompt_ids, **gen_kwargs)

        gen_len = output_ids.shape[1] - prompt_len
        eos_id = self.backbone.config.eos_token_id
        generated = output_ids[0, prompt_len:].tolist()

        return output_ids, {
            "gen_len": gen_len,
            "hit_eos": eos_id is not None and eos_id in generated,
            "tokens_per_step": 1.0,
            "total_steps": gen_len,
        }
