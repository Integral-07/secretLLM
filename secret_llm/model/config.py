from dataclasses import dataclass

@dataclass
class TransformerConfig:

	# Vocabulary
	vocab_size: int = 256
	max_seq_len:int = 128

	# Model dimensions
	d_model: int = 128
	n_heads: int = 4
	d_head: int = 32
	d_ff: int = 512
	n_layers: int = 4

	# Secret adapter
	adapter_rank: int = 16
	adapter_nonlinearity: str = "gelu"
	adapter_scale: float = 0.5

	# Secret projection
	secret_proj_rank: int = 32

	# Regularization
	dropout: float = 0.1

	# Special tokens
	pad_token_id: int = 0
	bos_token_id: int = 1
	eos_token_id: int = 2
