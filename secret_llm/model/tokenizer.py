import torch
from typing import List, Optional

class CharTokenizer:

	"""
	特殊トークン:
          0 = PAD (パディング: バッチ内の長さ揃え用)
          1 = BOS (Beginning of Sequence: 系列開始マーカー)
          2 = EOS (End of Sequence: 系列終了マーカー)

        一般トークン:
          文字のUnicodeコードポイント + 3 (特殊トークン分オフセット)
	"""

	PAD_ID = 0
	BOS_ID = 1
	EOS_ID = 2
	SPECIAL_OFFSET = 3

	def __init__(self, vocab_size: int = 256):
		self.vocab_size = vocab_size

	def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
	
		ids = []
		if add_bos:
			ids.append(self.BOS_ID)
		for ch in text:
			token_id = (ord(ch) + self.SPECIAL_OFFSET) % self.vocab_size

			# 特殊トークンIDの衝突を回避
			if token_id < self.SPECIAL_OFFSET:
				token_id = self.SPECIAL_OFFSET

			ids.append(token_id)

		if add_eos:
			ids.append(self.EOS_ID)
		
		return ids

	def decode(self, ids: List[int]) -> str:

		chars = []
		for token_id in ids:
			if token_id < self.SPECIAL_OFFSET:
				continue # PAD, BOS, EOSはスキップ

			ch = chr((token_id - self.SPECIAL_OFFSET) % (self.vocab_size - self.SPECIAL_OFFSET))

			chars.append(ch)

		return "".join(chars)


	def encode_batch(self, texts: List[str], max_len: Optional[int] = None, add_bos: bool = True, add_eos: bool = True) -> torch.Tensor:

		encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
		
		if max_len is None:
			max_len = max(len(e) for e in encoded)
		batch = torch.full((len(encoded), max_len), self.PAD_ID, dtype=torch.long)

		for i, ids in enumerate(encoded):
			length = min(len(ids), max_len)
			batch[i, :length] = torch.tensor(ids[:length], dtype=torch.long)

		return batch
