"""Self-contained Lafa configuration for Transformers AutoClass loading."""
from transformers import PretrainedConfig


class LafaConfig(PretrainedConfig):
    model_type = 'lafa'
    is_composition = True

    def __init__(self, backbone_config=None, flow_config=None, noise_sigma=0.02,
                 weight_variant='raw', provenance=None, rotary_buffer_dtype='model', **kwargs):
        self.backbone_config = dict(backbone_config or {})
        self.flow_config = dict(flow_config or {})
        self.noise_sigma = float(noise_sigma)
        if not self.noise_sigma > 0:
            raise ValueError('noise_sigma must be positive')
        if weight_variant not in ('raw', 'ema'):
            raise ValueError('weight_variant must be raw or ema')
        self.weight_variant = weight_variant
        self.provenance = dict(provenance or {})
        if rotary_buffer_dtype not in ('model','float32'):
            raise ValueError('rotary_buffer_dtype must be model or float32')
        self.rotary_buffer_dtype = rotary_buffer_dtype
        for key in ('vocab_size', 'hidden_size', 'num_hidden_layers',
                    'num_attention_heads', 'num_key_value_heads',
                    'max_position_embeddings', 'bos_token_id', 'eos_token_id',
                    'pad_token_id', 'tie_word_embeddings'):
            if key in self.backbone_config:
                kwargs.setdefault(key, self.backbone_config[key])
        kwargs.setdefault('tie_word_embeddings', False)
        kwargs.setdefault('architectures', ['LafaForCausalLM'])
        kwargs.setdefault('auto_map', {
            'AutoConfig': 'configuration_lafa.LafaConfig',
            'AutoModel': 'modeling_lafa.LafaForCausalLM',
            'AutoModelForCausalLM': 'modeling_lafa.LafaForCausalLM',
        })
        super().__init__(**kwargs)
