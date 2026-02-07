"""推論デモ。正しい秘密/誤った秘密/秘密なしの出力を比較する。

実行: python3 examples/demo_inference.py
  (先に demo_training.py を実行して public_weights.pt を生成すること)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from secret_llm.model.config import TransformerConfig
from secret_llm.model.transformer import SecretTransformer
from secret_llm.model.tokenizer import CharTokenizer
from secret_llm.crypto.key_manager import KeyManager
from secret_llm.pipeline.inference import SecretInferencePipeline


# demo_training.py の出力から取得した値
MASTER_SECRET_HEX = "4ebf1418c391fba06b224c8bac17d32be8400d393934c89b38a08a360428d1b3"
CORRECT_SESSION = "training-session-001"
WRONG_SESSION = "wrong-session-999"


def main():
    config = TransformerConfig()
    model = SecretTransformer(config)
    tokenizer = CharTokenizer(config.vocab_size)

    # 公開重み読み込み
    weights_path = os.path.join(os.path.dirname(__file__), "public_weights.pt")
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found. Run demo_training.py first.")
        return
    state = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state, strict=False)
    print(f"Loaded public weights from: {weights_path}\n")

    master_secret = bytes.fromhex(MASTER_SECRET_HEX)
    key_manager = KeyManager(master_secret)
    pipeline = SecretInferencePipeline(model, key_manager, tokenizer)

    prompts = ["the cat ", "the dog ", "the bird "]

    print("=== Secret Reasoning LLM Inference Demo ===\n")

    for prompt in prompts:
        print(f'Prompt: "{prompt}"')

        # 1. 正しい秘密
        pipeline.start_session(CORRECT_SESSION)
        out = pipeline.generate(prompt, max_new_tokens=40, temperature=0.5, top_k=5)
        pipeline.end_session()
        print(f"  Correct secret: \"{out}\"")

        # 2. 誤った秘密
        pipeline.start_session(WRONG_SESSION)
        out = pipeline.generate(prompt, max_new_tokens=40, temperature=0.5, top_k=5)
        pipeline.end_session()
        print(f"  Wrong secret:   \"{out}\"")

        # 3. 秘密なし
        model.clear_secrets()
        pipeline._current_session = "none"
        out = pipeline.generate(prompt, max_new_tokens=40, temperature=0.5, top_k=5)
        pipeline._current_session = None
        print(f"  No secret:      \"{out}\"")
        print()

    print("If training succeeded, only 'Correct secret' should produce coherent continuations.")


if __name__ == "__main__":
    main()
