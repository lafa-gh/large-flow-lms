"""
Base wrapper for HF Trainer compatibility.

Inherits PreTrainedModel so Trainer gets save_pretrained, from_pretrained,
tied weight handling, gradient checkpointing — all for free.

Subclasses store the backbone as `self.backbone` and must implement
forward() and generate().
"""

from abc import abstractmethod
from transformers import PreTrainedModel


class BaseModelWrapper(PreTrainedModel):
    """Abstract base that wraps a CausalLM backbone for HF Trainer.

    Uses the backbone's config. The default state_dict strips the backbone
    prefix for AR checkpoints; FlowModelWrapper overrides it to retain flow
    layers in its checkpoints.

    Subclasses must implement forward() and generate().
    """

    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def __init__(self, backbone: PreTrainedModel):
        import copy

        config = copy.deepcopy(backbone.config)
        config._attn_implementation_internal = "eager"
        super().__init__(config)
        self.backbone = backbone
        if backbone._tied_weights_keys is not None:
            self._tied_weights_keys = backbone._tied_weights_keys

    def _init_weights(self, module):
        pass

    @abstractmethod
    def forward(self, **kwargs): ...

    @abstractmethod
    def generate(self, **kwargs): ...

    def state_dict(self, *args, **kwargs):
        """Strip 'backbone.' prefix so checkpoints are loadable by AutoModel."""
        sd = super().state_dict(*args, **kwargs)
        return {k.removeprefix("backbone."): v for k, v in sd.items()}

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load into backbone directly (handles keys with or without prefix)."""
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = {k.removeprefix("backbone."): v for k, v in state_dict.items()}
        return self.backbone.load_state_dict(state_dict, *args, **kwargs)
