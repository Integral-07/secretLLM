import torch
import torch.nn as nn


class SecretGatingAdapter(nn.Module):
	"""秘密ゲーティングAdapter。

	秘密依存のゲートでAttention/FFN出力を変調する。
	秘密の有無で出力が構造的に変わることを保証する。

	ゲート機構:
	  h = nonlinearity(x @ W_down) @ W_up      # 秘密変換
	  gate = sigmoid(gate_bias + scale * h)      # ゲート値
	  output = x * gate                          # ゲーティング

	状態ごとの挙動:
	  Phase 1 (初期化時):
	    W_down, W_up = 0, gate_bias = +5
	    → h = 0, gate = sigmoid(5) ≈ 0.993
	    → output ≈ x (ほぼ恒等写像、通常学習可能)

	  秘密注入後 (adapt_to_secret):
	    W_down, W_up, gate_bias = セッション固有値
	    → gate は秘密依存パターン
	    → 公開重みが秘密パターンに適応

	  秘密クリア後 (clear_secrets):
	    W_down, W_up = 0, gate_bias = -5
	    → gate = sigmoid(-5) ≈ 0.007
	    → output ≈ 0 (信号遮断 → 出力崩壊)

	  間違った秘密:
	    gate パターンが異なる → 公開重みと不整合 → 出力崩壊
	"""

	GATE_OPEN = 5.0     # 初期値: ゲート開放 (Phase 1用)
	GATE_CLOSED = -5.0  # クリア値: ゲート閉鎖

	def __init__(self, d_model: int, rank: int, nonlinearity: str = "gelu", scale: float = 0.5):
		super().__init__()

		self.d_model = d_model
		self.rank = rank
		self.scale = scale

		# 秘密重み (HKDF派生、requires_grad=False)
		self.W_down = nn.Parameter(torch.zeros(d_model, rank), requires_grad=False)
		self.W_up = nn.Parameter(torch.zeros(rank, d_model), requires_grad=False)

		# ゲートバイアス (秘密パラメータ、初期値=GATE_OPEN でPhase 1学習を妨げない)
		self.gate_bias = nn.Parameter(
			torch.full((d_model,), self.GATE_OPEN), requires_grad=False,
		)

		if nonlinearity == "gelu":
			self.nonlinearity = nn.GELU()
		elif nonlinearity == "silu":
			self.nonlinearity = nn.SiLU()
		elif nonlinearity == "tanh":
			self.nonlinearity = nn.Tanh()
		else:
			raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

	def _secret_transform(self, x: torch.Tensor) -> torch.Tensor:
		"""秘密変換: nonlinearity(x @ W_down) @ W_up"""
		h = x @ self.W_down                # (B, T, d_model) → (B, T, rank)
		h = self.nonlinearity(h)            # 非線形変換
		h = h @ self.W_up                   # (B, T, rank) → (B, T, d_model)
		return h

	def set_secret_weights(
		self,
		w_down: torch.Tensor,
		w_up: torch.Tensor,
		gate_bias: torch.Tensor,
	):
		"""HKDFから派生した秘密重みを注入する。"""
		self.W_down.data.copy_(w_down)
		self.W_up.data.copy_(w_up)
		self.gate_bias.data.copy_(gate_bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""ゲーティング順変換。

		入力: (B, T, d_model)
		出力: (B, T, d_model)
		"""
		h = self._secret_transform(x)
		gate = torch.sigmoid(self.gate_bias + self.scale * h)
		return x * gate


# 後方互換エイリアス
SecretAdapter = SecretGatingAdapter
