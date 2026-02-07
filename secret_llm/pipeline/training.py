from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..model.config import TransformerConfig
from ..model.transformer import SecretTransformer
from ..model.tokenizer import CharTokenizer
from ..crypto.key_manager import KeyManager


class TextDataset(Dataset):
	"""文字レベル言語モデリング用データセット。

	テキストをtokenizerでエンコードし、固定長のチャンクに分割する。
	各サンプルは (input_ids, target_ids) のペアで、
	target_ids は input_ids を1文字ずらしたもの (次文字予測)。
	"""

	def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int):
		self.seq_len = seq_len
		# テキスト全体を1つのトークン列にエンコード (BOS/EOSなし)
		self.token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)

	def __len__(self):
		# seq_len + 1 文字分 (入力 + ターゲット1文字) が取れるチャンク数
		return max(0, len(self.token_ids) - self.seq_len)

	def __getitem__(self, idx):
		chunk = self.token_ids[idx : idx + self.seq_len + 1]
		x = torch.tensor(chunk[:-1], dtype=torch.long)   # 入力: 先頭 seq_len 文字
		y = torch.tensor(chunk[1:], dtype=torch.long)     # ターゲット: 1文字ずらし
		return x, y


class SecretTrainingPipeline:
	"""2段階学習パイプライン。

	Phase 1 (train_base):
	  秘密なし (S_q=S_k=I, Adapter=0) で次文字予測を学習。
	  ベースモデルの言語能力を獲得する。

	Phase 2 (train_with_secret):
	  秘密重みを注入・固定し、公開重みのみを低学習率で再学習。
	  公開重みが秘密空間に共適応し、正しい秘密でのみ
	  正しい出力が得られるモデルになる。
	"""

	def __init__(self, config: TransformerConfig, master_secret: bytes, session_id: str):
		self.config = config
		self.model = SecretTransformer(config)
		self.key_manager = KeyManager(master_secret)
		self.session_id = session_id
		self.tokenizer = CharTokenizer(config.vocab_size)

	def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
		"""Causal mask生成。未来のトークンを見れないよう下三角行列を作る。"""
		mask = torch.tril(torch.ones(seq_len, seq_len))
		return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

	def _train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
		"""1エポック分の学習を実行し、平均損失を返す。"""
		self.model.train()
		total_loss = 0.0
		num_batches = 0

		for x, y in dataloader:
			mask = self._create_causal_mask(x.size(1))
			logits = self.model(x, mask=mask)  # (B, T, vocab_size)

			# Cross-entropy: 各位置で次文字を予測
			loss = F.cross_entropy(
				logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
				y.view(-1),                         # (B*T,)
				ignore_index=self.tokenizer.PAD_ID,
			)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			num_batches += 1

		return total_loss / max(num_batches, 1)

	def train_base(
		self, text: str, epochs: int = 10, lr: float = 3e-4,
		batch_size: int = 32, seq_len: Optional[int] = None,
	) -> list[float]:
		"""Phase 1: 秘密なしでベースモデルを事前学習する。

		Args:
			text: 学習テキスト
			epochs: エポック数
			lr: 学習率
			batch_size: バッチサイズ
			seq_len: 系列長 (Noneならconfig.max_seq_len)
		Returns:
			各エポックの平均損失リスト
		"""
		if seq_len is None:
			seq_len = self.config.max_seq_len

		self.model.clear_secrets()  # 秘密なし状態を保証

		dataset = TextDataset(text, self.tokenizer, seq_len)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		# 学習可能パラメータのみ最適化
		optimizer = torch.optim.AdamW(
			[p for p in self.model.parameters() if p.requires_grad],
			lr=lr,
		)

		losses = []
		for epoch in range(epochs):
			avg_loss = self._train_epoch(dataloader, optimizer)
			losses.append(avg_loss)
			print(f"[Base] Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}")

		return losses

	def train_with_secret(
		self, text: str, epochs: int = 5, lr: float = 1e-4,
		batch_size: int = 32, seq_len: Optional[int] = None,
	) -> list[float]:
		"""Phase 2: 秘密注入済みで公開重みをfine-tuningする。

		秘密重みは固定 (requires_grad=False) のまま、
		公開重みだけが低学習率で更新される。
		"""
		if seq_len is None:
			seq_len = self.config.max_seq_len

		# 秘密重みを注入 (固定)
		self.model.inject_secrets(self.key_manager, self.session_id)

		dataset = TextDataset(text, self.tokenizer, seq_len)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		optimizer = torch.optim.AdamW(
			[p for p in self.model.parameters() if p.requires_grad],
			lr=lr,
		)

		losses = []
		for epoch in range(epochs):
			avg_loss = self._train_epoch(dataloader, optimizer)
			losses.append(avg_loss)
			print(f"[Secret] Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}")

		return losses

	def save_public_weights(self, path: str):
		"""公開パラメータのみを保存する。秘密パラメータは含まない。"""
		public_state = {
			k: v for k, v in self.model.state_dict().items()
			if any(p.data_ptr() == v.data_ptr() for p in self.model.parameters() if p.requires_grad)
		}
		torch.save(public_state, path)

	def load_public_weights(self, path: str):
		"""保存された公開パラメータを読み込む。"""
		state = torch.load(path, weights_only=True)
		self.model.load_state_dict(state, strict=False)
