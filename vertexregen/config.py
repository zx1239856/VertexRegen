from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    num_pos_tokens: int
    path: str
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    sep_token_id: int = 3
    nil_token_id: int = 4
    pos_token_offset: int = 5
    max_seq_length: int = 8192
    attn_implementation: str = "flash_attention_2"


@dataclass
class DataConfig:
    path: str
    num_pos_tokens: int
    pad_to_multiple_of: int = 32
    padding_side: str = "right"
    random_scale: bool = True
    random_scale_max: float = 1.1
    random_shift: bool = True
    random_shift_min: float = -0.1
    random_shift_max: float = 0.1
    random_rotate: bool = True
    overfit: bool = False
