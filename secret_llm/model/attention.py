from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


class MultiHeadAttention(nn.Module):
	"""標準Multi-Head Attention (秘密機構なし)。

	データフロー:
	  入力 x: (B, T, d_model)
	    → W_q, W_k, W_v で Q, K, V に線形射影
	    → n_heads個のヘッドに分割 (各ヘッド d_head次元)
	    → 各ヘッドで: スコア = Q @ K^T / sqrt(d_head)
	    → softmaxで正規化 → 注意重み
	    → 注意重み @ V で文脈ベクトルを計算
	    → 全ヘッドを結合 → W_o で出力射影
	"""

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.n_heads = config.n_heads
		self.d_head = config.d_head
		self.d_model = config.d_model

		# Q, K, V, O の線形射影 (全て公開・学習可能)
		self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
		self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
		self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
		self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

		self.dropout = nn.Dropout(config.dropout)
		self.scale = config.d_head ** -0.5  # 1/sqrt(d_head)

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Args:
			x: (B, T, d_model) 入力テンソル
			mask: (1, 1, T, T) causal mask。0の位置は注意を遮断する
		Returns:
			(B, T, d_model)
		"""
		B, T, _ = x.shape

		# 線形射影してヘッドに分割: (B, T, d_model) → (B, n_heads, T, d_head)
		Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
		K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
		V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

		# Attention score: (B, H, T, T)
		attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

		# Causal mask適用: 未来のトークンへの注意を-infで遮断
		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

		# Softmaxで確率分布に正規化
		attn_weights = F.softmax(attn_scores, dim=-1)
		attn_weights = self.dropout(attn_weights)

		# 注意重み × V で文脈ベクトルを集約: (B, H, T, d_head)
		out = torch.matmul(attn_weights, V)

		# ヘッドを結合して出力射影: (B, T, d_model)
		out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
		return self.W_o(out)
