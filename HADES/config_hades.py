from dataclasses import dataclass, field
from transformers.configuration_utils import PretrainedConfig

@dataclass
class HADESConfig(PretrainedConfig):

    d_model: int = 1024
    d_intermediate: int = 0
    n_layer: int = 48
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=lambda: {'layer': 'HADES'})
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 16
    tie_embeddings: bool = True
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 2048
    expert_nheads: int = 16
    shared_nheads: int = 8
    load_balance_coef: float = 0.001
    diversity_coef: float = 0.001
    gamma: float = 0.25