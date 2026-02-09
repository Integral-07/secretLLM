import torch

from .key_manager import SessionKeySet


class WeightGenerator:
	"""HKDF派生バイト列をPyTorchテンソルに変換する。

	バイト列を直接float32に解釈すると NaN/Inf が発生しうるため、
	先頭8バイトをPRNGのシードとして使い、torch.Generatorで
	正規分布テンソルを生成する。
	"""

	@staticmethod
	def bytes_to_tensor(raw_bytes: bytes, shape: tuple, dtype=torch.float32) -> torch.Tensor:
		"""バイト列 → 正規分布テンソル。Adapterの W_down, W_up 用。"""
		seed = int.from_bytes(raw_bytes[:8], "big")
		gen = torch.Generator()
		gen.manual_seed(seed)

		tensor = torch.randn(shape, generator=gen, dtype=dtype)

		fan_in = shape[0] if len(shape) >= 2 else shape[0]
		tensor = tensor / (fan_in ** 0.5)

		return tensor

	@staticmethod
	def bytes_to_orthogonal(raw_bytes: bytes, shape: tuple) -> torch.Tensor:
		"""バイト列 → 直交行列。秘密射影 S_q, S_k 用。

		QR分解でランダム行列から直交行列を生成する。
		直交行列はノルムを保存し (||Qx|| = ||x||)、
		意味空間を「回転」するが「潰さない」。

		Args:
			shape: (n_heads, d_head, d_head)
		"""
		seed = int.from_bytes(raw_bytes[:8], "big")
		gen = torch.Generator()
		gen.manual_seed(seed)

		n_heads, d1, d2 = shape
		matrices = []
		for _ in range(n_heads):
			random_matrix = torch.randn(d1, d2, generator=gen)
			q, r = torch.linalg.qr(random_matrix)
			# QR分解の符号不定性を解消
			d = torch.diag(r)
			ph = torch.sign(d)
			q = q * ph.unsqueeze(0)
			matrices.append(q)

		return torch.stack(matrices, dim=0)

	def generate_adapter_weights(
		self, session_keys: SessionKeySet, layer_idx: int,
		position: str, d_model: int, rank: int,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""1つのSecretGatingAdapter用の重みとゲートバイアスを生成する。

		Args:
			position: "attn" または "ffn"
		Returns:
			(W_down, W_up, gate_bias)
			  W_down: (d_model, rank)
			  W_up: (rank, d_model)
			  gate_bias: (d_model,) — 正の値 (ゲート開放用)
		"""
		raw_down = session_keys.derive_component_bytes(
			layer_idx, f"adapter_{position}_down", 8,
		)
		raw_up = session_keys.derive_component_bytes(
			layer_idx, f"adapter_{position}_up", 8,
		)
		raw_bias = session_keys.derive_component_bytes(
			layer_idx, f"adapter_{position}_gate_bias", 8,
		)

		w_down = self.bytes_to_tensor(raw_down, (d_model, rank))
		w_up = self.bytes_to_tensor(raw_up, (rank, d_model))

		# gate_bias: 正の値に生成 (sigmoid(gate_bias) ≈ 1 になるように)
		# PRNG生成 → abs() + 2.5 で [2.5, ~5.0] の範囲にシフト
		seed = int.from_bytes(raw_bias[:8], "big")
		gen = torch.Generator()
		gen.manual_seed(seed)
		gate_bias = torch.randn(d_model, generator=gen).abs() + 2.5

		return w_down, w_up, gate_bias

	def generate_secret_projections(
		self, session_keys: SessionKeySet, layer_idx: int,
		n_heads: int, d_head: int,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""1層分の S_q, S_k 秘密射影行列を生成する。

		Returns:
			(S_q: (n_heads, d_head, d_head), S_k: (n_heads, d_head, d_head))
		"""
		raw_q = session_keys.derive_component_bytes(layer_idx, "proj_q", 8)
		raw_k = session_keys.derive_component_bytes(layer_idx, "proj_k", 8)

		s_q = self.bytes_to_orthogonal(raw_q, (n_heads, d_head, d_head))
		s_k = self.bytes_to_orthogonal(raw_k, (n_heads, d_head, d_head))

		return s_q, s_k
