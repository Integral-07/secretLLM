import torch
import torch.nn as nn


class SecretAdapter(nn.Module):
	"""低ランク非線形Adapter (秘密重み)。

	アーキテクチャ:
	  x → W_down (d_model → rank) → GELU → W_up (rank → d_model) → × scale
	  出力 = x + scale * adapter(x)   (残差接続)

	W_down, W_up はHKDFから派生した秘密値であり、学習しない。
	非線形性(GELU)が重要:
	  - なければ W_down @ W_up は1つの線形変換に退化
	  - 退化するとベース重みに吸収され秘匿効果がなくなる

	scale=0.1 で秘密歪みの強さを制御:
	  - 大きすぎ(>0.5): ベース表現を圧倒し学習が不安定
	  - 小さすぎ(<0.01): 秘匿効果が弱い
	"""

	def __init__(self, d_model: int, rank: int, nonlinearity: str = "gelu", scale: float = 0.1):
		super().__init__()

		self.W_down = nn.Parameter(torch.zeros(d_model, rank), requires_grad=False)
		self.W_up = nn.Parameter(torch.zeros(rank, d_model), requires_grad=False)
		self.scale = scale

		if nonlinearity == "gelu":
			self.nonlinearity = nn.GELU()
		elif nonlinearity == "silu":
			self.nonlinearity = nn.SiLU()
		elif nonlinearity == "tanh":
			self.nonlinearity = nn.Tanh()
		else:
			raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

	def set_secret_weights(self, w_down: torch.Tensor, w_up: torch.Tensor):
		"""HKDFから派生した秘密重みを注入する。"""
		self.W_down.data.copy_(w_down)
		self.W_up.data.copy_(w_up)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		入力: (B, T, d_model)
		出力: (B, T, d_model) — 秘密歪みが残差的に加算される
		"""
		h = x @ self.W_down           # (B, T, d_model) → (B, T, rank) 次元削減
		h = self.nonlinearity(h)       # 非線形変換 (退化防止)
		h = h @ self.W_up             # (B, T, rank) → (B, T, d_model) 次元復元
		return x + self.scale * h      # 残差接続: 元の表現 + 秘密歪み
