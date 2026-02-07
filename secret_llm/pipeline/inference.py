from typing import Optional

import torch
import torch.nn.functional as F

from ..model.transformer import SecretTransformer
from ..model.tokenizer import CharTokenizer
from ..crypto.key_manager import KeyManager


class SecretInferencePipeline:
	"""秘匿推論パイプライン。

	使い方:
	  1. start_session(session_id) でセッション開始 (秘密注入)
	  2. generate(prompt) でテキスト生成
	  3. end_session() でセッション終了 (秘密破棄)

	セッションごとに異なる秘密が注入されるため、
	同じ入力でもセッションが異なれば内部表現は完全に異なる。
	"""

	def __init__(self, model: SecretTransformer, key_manager: KeyManager, tokenizer: CharTokenizer):
		self.model = model
		self.key_manager = key_manager
		self.tokenizer = tokenizer
		self._current_session: Optional[str] = None

	def start_session(self, session_id: str):
		"""セッション開始。HKDFから秘密重みを導出しモデルに注入する。"""
		self.model.inject_secrets(self.key_manager, session_id)
		self._current_session = session_id

	def end_session(self):
		"""セッション終了。秘密重みをモデルから消去する。"""
		self.model.clear_secrets()
		self._current_session = None

	def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
		mask = torch.tril(torch.ones(seq_len, seq_len))
		return mask.unsqueeze(0).unsqueeze(0)

	@torch.no_grad()
	def generate(
		self, prompt: str, max_new_tokens: int = 50,
		temperature: float = 1.0, top_k: int = 0,
	) -> str:
		"""自己回帰テキスト生成。

		promptの続きをmax_new_tokensだけ生成する。
		各ステップで:
		  1. 現在の系列をモデルに入力
		  2. 最後の位置のlogitsから次トークンをサンプリング
		  3. 系列に追加して繰り返す

		Args:
			prompt: 入力テキスト
			max_new_tokens: 最大生成トークン数
			temperature: サンプリング温度。低い=確定的、高い=多様
			top_k: 上位k個のトークンからのみサンプリング (0=全体)
		Returns:
			生成されたテキスト (prompt部分を除く)
		"""
		assert self._current_session is not None, "Must call start_session() first"

		self.model.eval()
		token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
		input_tensor = torch.tensor([token_ids], dtype=torch.long)

		for _ in range(max_new_tokens):
			# max_seq_lenを超えたら末尾を切り出す
			input_trimmed = input_tensor[:, -self.model.config.max_seq_len:]
			T = input_trimmed.size(1)
			mask = self._create_causal_mask(T)

			logits = self.model(input_trimmed, mask=mask)
			next_logits = logits[:, -1, :] / temperature  # 最後の位置のlogits

			# Top-kフィルタリング: 上位k個以外を-infにして確率0にする
			if top_k > 0:
				top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
				next_logits = torch.full_like(next_logits, float("-inf"))
				next_logits.scatter_(1, top_k_indices, top_k_logits)

			probs = F.softmax(next_logits, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)

			# EOSが出たら生成を終了
			if next_token.item() == self.tokenizer.EOS_ID:
				break

			input_tensor = torch.cat([input_tensor, next_token], dim=1)

		# prompt部分を除いた生成結果をデコード
		generated_ids = input_tensor[0, len(token_ids):].tolist()
		return self.tokenizer.decode(generated_ids)

	@torch.no_grad()
	def get_internal_representations(self, text: str) -> dict[str, torch.Tensor]:
		"""各層の出力テンソルを取得する。

		セッション間で同じ入力に対する内部表現を比較することで、
		秘密が実際に表現を変化させていることを検証できる。
		cosine similarityが≈0なら、秘匿が機能している。

		Returns:
			{"layer_0": Tensor, "layer_1": Tensor, ...}
		"""
		assert self._current_session is not None, "Must call start_session() first"

		self.model.eval()
		token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)
		input_tensor = torch.tensor([token_ids], dtype=torch.long)

		representations = {}
		hooks = []

		# 各層にforward hookを登録して出力を記録する
		for i, layer in enumerate(self.model.layers):
			def make_hook(layer_idx):
				def hook_fn(module, input, output):
					representations[f"layer_{layer_idx}"] = output.detach().clone()
				return hook_fn
			h = layer.register_forward_hook(make_hook(i))
			hooks.append(h)

		mask = self._create_causal_mask(input_tensor.size(1))
		self.model(input_tensor, mask=mask)

		# hookを除去 (登録したままだとメモリリーク)
		for h in hooks:
			h.remove()

		return representations
