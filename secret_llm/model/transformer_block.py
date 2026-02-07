from typing import Optional

import torch
import torch.nn as nn

from .config import TransformerConfig
from .secret_attention import SecretProjectionAttention
from .feedforward import PositionwiseFeedForward
from .secret_adapter import SecretAdapter


class SecretTransformerBlock(nn.Module):
	"""Transformerの1層。秘密射影Attention + 秘密Adapter + FFN を組み合わせる。

	データフロー:
	  x
	  ├→ LayerNorm → SecretProjectionAttention → SecretAdapter(attn) → +残差
	  │                                                                  ↓
	  └──────────────────────────────────────────────────────────────→ x_new
	  ├→ LayerNorm → FeedForward → SecretAdapter(ffn) → +残差
	  │                                                    ↓
	  └──────────────────────────────────────────────→ output

	Pre-LayerNorm方式 (GPT-2スタイル):
	  LayerNormを変換の「前」に適用。Post-Normより勾配が安定し、
	  learning rate warmupなしでも学習可能。
	"""

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.norm1 = nn.LayerNorm(config.d_model)
		self.norm2 = nn.LayerNorm(config.d_model)
		self.attention = SecretProjectionAttention(config)
		self.ffn = PositionwiseFeedForward(config)
		self.adapter_attn = SecretAdapter(
			config.d_model, config.adapter_rank, config.adapter_nonlinearity,
		)
		self.adapter_ffn = SecretAdapter(
			config.d_model, config.adapter_rank, config.adapter_nonlinearity,
		)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Args:
			x: (B, T, d_model)
			mask: (1, 1, T, T) causal mask
		Returns:
			(B, T, d_model)
		"""
		# Attention + 秘密Adapter (残差接続)
		h = self.norm1(x)
		attn_out = self.attention(h, mask=mask)
		attn_out = self.adapter_attn(attn_out)
		x = x + self.dropout(attn_out)

		# FFN + 秘密Adapter (残差接続)
		h = self.norm2(x)
		ffn_out = self.ffn(h)
		ffn_out = self.adapter_ffn(ffn_out)
		x = x + self.dropout(ffn_out)

		return x
