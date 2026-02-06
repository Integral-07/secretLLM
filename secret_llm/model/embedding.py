import math, torch
import torch.nn as nn

from .config import TransformerConfig

class TokenEmbedding(nn.Module):

	"""
	tokenId -> d_model次元ベクトルへの変換

	内部にvocab_size * d_modelの学習可能な重み行列を持ち
	各トークンIDに対応するぎょうベクトルをルックアップする。

	出力はsqrt(de_model)でスケールして位置エンコーディングとバランスを取る
	"""

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.embedding = nn.Embedding(
			num_embeddings=config.vocab_size,
			embedding_dim=config.d_model,
			  # pad token is always zore vector
			padding=config.pad_token_id,
		)
	
		# scaling coefficient
		self.scale = math.sqrt(config.d_model)

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:	
		"""
		入力：(batch_size, seq_len)のLongTensor
		出力：(batch_size, seq_len)のfloatTensor
		"""

		return self.embedding(token_ids) * self.scale

	
class PositionalEncoding(nn.Module):
	"""
	サインカーブによる位置エンコーディング
	"""

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.dropout = nn.Dropout(config.dropout)

		pe = torch.zeros(config.max_seq_len, config.d_model)
		possition = torch.arange(0, config.max_seq_len).unsqueeze(1).float()
		div_term = torch.exp(
			torch.arange(0, config.d_model, 2).float() * -(math.log(10000.0) / config.d_model)
		)
		
		# 偶数次元
		pe[:, 0::2] = torch.sin(position * div_term)

		# 奇数次元
		pe[:, 1::2] = torch.cos(position * div_term)

		self.register_buffer("pe", pe.unsqueeze(0))

	def forward(self, x: torch.Tensor) -> torch.Tensor:

		"""
		入力：(batch_size, seq_len, d_model) - TokenEmbeddingの出力
		出力：(batch_size, seq_len, d_model) - 位置情報が加算されたベクトル
		"""

		# 系列長分だけスライスして加算
		x = x + self.pe[:, : x.size(1)]

		return self.dropout(x)
