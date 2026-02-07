from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


class SecretProjectionAttention(nn.Module):
	"""秘密射影付きMulti-Head Attention。

	標準Attention:
	  A = softmax(Q @ K^T / sqrt(d))

	秘密射影Attention:
	  Q' = Q @ S_q    (秘密行列で回転)
	  K' = K @ S_k    (秘密行列で回転)
	  A  = softmax(Q' @ K'^T / sqrt(d))

	S_q, S_k は直交行列で、ヘッドごと・層ごと・セッションごとに異なる。
	V(値ベクトル)は変更しない:
	  - 「何に注目するか」(routing) は秘匿される
	  - 「何を集約するか」(content) は変更されない
	  - これにより学習の安定性と秘匿性を両立
	"""

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.n_heads = config.n_heads
		self.d_head = config.d_head
		self.d_model = config.d_model

		# 公開 (学習可能) な射影
		self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
		self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
		self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
		self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

		# 秘密射影行列: (n_heads, d_head, d_head)
		# デフォルトは単位行列 → 秘密なしで標準Attentionと同一動作
		self.S_q = nn.Parameter(
			torch.eye(config.d_head).unsqueeze(0).expand(config.n_heads, -1, -1).clone(),
			requires_grad=False,
		)
		self.S_k = nn.Parameter(
			torch.eye(config.d_head).unsqueeze(0).expand(config.n_heads, -1, -1).clone(),
			requires_grad=False,
		)

		self.dropout = nn.Dropout(config.dropout)
		self.scale = config.d_head ** -0.5

	def set_secret_projections(self, s_q: torch.Tensor, s_k: torch.Tensor):
		"""秘密射影行列を注入する。

		Args:
			s_q: (n_heads, d_head, d_head) Q用秘密射影
			s_k: (n_heads, d_head, d_head) K用秘密射影
		"""
		self.S_q.data.copy_(s_q)
		self.S_k.data.copy_(s_k)

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		B, T, _ = x.shape

		# 公開射影: (B, T, d_model) → (B, H, T, d_head)
		Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
		K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
		V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

		# 秘密射影の適用: 各ヘッドのQ, Kを秘密行列で回転
		# Q: (B, H, T, d_head), S_q: (H, d_head, d_head) → Q': (B, H, T, d_head)
		Q_prime = torch.einsum("bhtd,hde->bhte", Q, self.S_q)
		K_prime = torch.einsum("bhtd,hde->bhte", K, self.S_k)

		# 秘密Q', K'でAttention score計算
		attn_scores = torch.einsum("bhte,bhse->bhts", Q_prime, K_prime) * self.scale

		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

		attn_weights = F.softmax(attn_scores, dim=-1)
		attn_weights = self.dropout(attn_weights)

		# V は秘密射影しない (content は変更なし)
		out = torch.einsum("bhts,bhsd->bhtd", attn_weights, V)
		out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

		return self.W_o(out)
