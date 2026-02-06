import torch.nn as nn

from .config import TransformerConfig

class PositionwiseFeedForward(nn.Module):

	"""
	位置ごとのフィードフォワードネットワーク
	"""

	def __init__(self, config: TransformerConfig):

		super().__init__()
		self.fc1 = nn.Linear(config.d_model, config.d_ff)
		self.fc2 = nn.Linear(config.d_ff, config.d_model)
		self.activation = nn.GELU()

		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		
		x = self.fc1(x)

		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc2(x)

		x = self.dropout(x)

		return x
		
