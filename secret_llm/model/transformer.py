from typing import Optional

import torch
import torch.nn as nn

from .config import TransformerConfig
from .embedding import TokenEmbedding, PositionalEncoding
from .transformer_block import SecretTransformerBlock
from .secret_adapter import SecretGatingAdapter
from ..crypto.key_manager import KeyManager
from ..crypto.weight_generator import WeightGenerator


class SecretTransformer(nn.Module):
	"""秘匿推論Transformer。

	公開パラメータ (学習可能):
	  - TokenEmbedding, PositionalEncoding
	  - 各層の W_q, W_k, W_v, W_o (Attention射影)
	  - 各層の FFN (fc1, fc2)
	  - 各層の LayerNorm
	  - 出力射影 (Embeddingと重み共有)

	秘密パラメータ (HKDF派生・固定):
	  - 各層の S_q, S_k (秘密射影行列)
	  - 各層の SecretGatingAdapter W_down, W_up, gate_bias (Attention後 + FFN後)

	inject_secrets() で秘密を注入、clear_secrets() でリセット。
	秘密未注入時 (Phase 1) は gate_bias=+5 で標準Transformerとほぼ同一動作。
	clear_secrets() 後は gate_bias=-5 で信号遮断。
	"""

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.config = config
		self.embedding = TokenEmbedding(config)
		self.pos_encoding = PositionalEncoding(config)
		self.layers = nn.ModuleList([
			SecretTransformerBlock(config) for _ in range(config.n_layers)
		])
		self.final_norm = nn.LayerNorm(config.d_model)
		self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

		# 出力射影の重みをEmbeddingと共有 (weight tying)
		# パラメータ数を削減し、入力と出力の意味空間を一致させる
		self.output_proj.weight = self.embedding.embedding.weight

	def forward(
		self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Args:
			token_ids: (B, T) トークンIDのLongTensor
			mask: (1, 1, T, T) causal mask。Noneなら自動生成
		Returns:
			logits: (B, T, vocab_size) 各位置での次トークン予測スコア
		"""
		x = self.embedding(token_ids)
		x = self.pos_encoding(x)

		for layer in self.layers:
			x = layer(x, mask=mask)

		x = self.final_norm(x)
		logits = self.output_proj(x)
		return logits

	def inject_secrets(self, key_manager: KeyManager, session_id: str):
		"""HKDFからセッション固有の秘密重みを全層に注入する。

		推論前に1回呼ぶ。同じ key_manager + session_id なら
		常に同じ秘密重みが注入される (決定論性)。
		"""
		session_keys = key_manager.derive_session(session_id)
		weight_gen = WeightGenerator()

		for i, layer in enumerate(self.layers):
			# 秘密射影行列 S_q, S_k を注入
			s_q, s_k = weight_gen.generate_secret_projections(
				session_keys, i, self.config.n_heads, self.config.d_head,
			)
			layer.attention.set_secret_projections(s_q, s_k)

			# Attention後Adapterの秘密重みを注入 (W_down, W_up, gate_bias)
			wd_attn, wu_attn, gb_attn = weight_gen.generate_adapter_weights(
				session_keys, i, "attn", self.config.d_model, self.config.adapter_rank,
			)
			layer.adapter_attn.set_secret_weights(wd_attn, wu_attn, gb_attn)

			# FFN後Adapterの秘密重みを注入 (W_down, W_up, gate_bias)
			wd_ffn, wu_ffn, gb_ffn = weight_gen.generate_adapter_weights(
				session_keys, i, "ffn", self.config.d_model, self.config.adapter_rank,
			)
			layer.adapter_ffn.set_secret_weights(wd_ffn, wu_ffn, gb_ffn)

	def clear_secrets(self):
		"""全秘密重みをリセット。

		S_q, S_k → 単位行列 (秘密射影なし)
		Adapter W_down, W_up → ゼロ, gate_bias → GATE_CLOSED (-5)
		→ gate = sigmoid(-5) ≈ 0.007 → 信号遮断
		"""
		d_model = self.config.d_model
		rank = self.config.adapter_rank

		for layer in self.layers:
			n_heads, d_head = self.config.n_heads, self.config.d_head
			identity = torch.eye(d_head).unsqueeze(0).expand(n_heads, -1, -1).clone()
			layer.attention.set_secret_projections(identity, identity.clone())

			layer.adapter_attn.set_secret_weights(
				torch.zeros(d_model, rank),
				torch.zeros(rank, d_model),
				torch.full((d_model,), SecretGatingAdapter.GATE_CLOSED),
			)
			layer.adapter_ffn.set_secret_weights(
				torch.zeros(d_model, rank),
				torch.zeros(rank, d_model),
				torch.full((d_model,), SecretGatingAdapter.GATE_CLOSED),
			)

	def count_parameters(self) -> tuple[int, int]:
		"""公開パラメータ数と秘密パラメータ数を返す。"""
		public = sum(p.numel() for p in self.parameters() if p.requires_grad)
		secret = sum(p.numel() for p in self.parameters() if not p.requires_grad)
		return public, secret
