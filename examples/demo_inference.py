"""推論デモ。秘匿化前後の応答を比較する。

セッション固有の重み (public_weights_{session}.pt) と
ベース重み (public_weights_base.pt) の両方で推論し、
秘匿化が推論結果にどう影響するかを確認する。

前提: demo_training.py → demo_finetune.py を実行済みであること。

実行: python3 -m examples.demo_inference --session sess-001
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from secret_llm.model.config import TransformerConfig
from secret_llm.model.transformer import SecretTransformer
from secret_llm.model.tokenizer import CharTokenizer
from secret_llm.crypto.key_manager import KeyManager
from secret_llm.pipeline.inference import SecretInferencePipeline


def load_model(weights_path, config):
    model = SecretTransformer(config)
    state = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state, strict=False)
    return model


def generate_with_session(pipeline, model, session_id, prompt, max_tokens, temperature):
    pipeline.start_session(session_id)
    out = pipeline.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, top_k=5)
    pipeline.end_session()
    return out


def generate_without_secret(pipeline, model, prompt, max_tokens, temperature):
    model.clear_secrets()
    pipeline._current_session = "none"
    out = pipeline.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, top_k=5)
    pipeline._current_session = None
    return out


def main():
    parser = argparse.ArgumentParser(description="秘匿化前後の応答比較")
    parser.add_argument("--session", default="sess-001", help="正しいセッションID")
    parser.add_argument("--wrong-session", default="wrong-session-999", help="誤ったセッションID")
    parser.add_argument("--prompts", nargs="+", default=["the cat ", "the dog ", "the bird "],
                        help="試すプロンプト")
    parser.add_argument("--max-tokens", type=int, default=40, help="最大生成トークン数")
    parser.add_argument("--temperature", type=float, default=0.5, help="サンプリング温度")
    parser.add_argument("--secret", type=str, default=None, help="Path to master_secret.hex")
    args = parser.parse_args()

    examples_dir = os.path.dirname(__file__)
    secret_path = args.secret or os.path.join(examples_dir, "master_secret.hex")
    base_weights_path = os.path.join(examples_dir, "public_weights_base.pt")
    session_weights_path = os.path.join(examples_dir, f"public_weights_{args.session}.pt")

    # --- バリデーション ---
    if not os.path.exists(secret_path):
        print(f"Error: {secret_path} not found. Run demo_training.py first.")
        sys.exit(1)
    if not os.path.exists(session_weights_path):
        print(f"Error: {session_weights_path} not found.")
        print(f"Run: python3 -m examples.demo_finetune --session {args.session}")
        sys.exit(1)

    has_base = os.path.exists(base_weights_path)

    with open(secret_path) as f:
        master_secret = bytes.fromhex(f.read().strip())

    config = TransformerConfig()
    tokenizer = CharTokenizer(config.vocab_size)
    key_manager = KeyManager(master_secret)

    print("=== Inference: 秘匿化前後の応答比較 ===")
    print(f"Session weights: {session_weights_path}")
    if has_base:
        print(f"Base weights:    {base_weights_path}")
    print(f"Correct session: {args.session}")
    print(f"Wrong session:   {args.wrong_session}")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}\n")

    # --- セッション重みでの推論 ---
    model = load_model(session_weights_path, config)
    pipeline = SecretInferencePipeline(model, key_manager, tokenizer)

    for prompt in args.prompts:
        print(f'Prompt: "{prompt}"')

        # 1. 正しい秘密 (Phase 2 済み重み + 正しいセッション)
        out = generate_with_session(pipeline, model, args.session, prompt,
                                    args.max_tokens, args.temperature)
        print(f'  [Correct secret] "{out}"')

        # 2. 誤った秘密 (Phase 2 済み重み + 誤ったセッション)
        out = generate_with_session(pipeline, model, args.wrong_session, prompt,
                                    args.max_tokens, args.temperature)
        print(f'  [Wrong secret]   "{out}"')

        # 3. 秘密なし (Phase 2 済み重み + 秘密クリア)
        out = generate_without_secret(pipeline, model, prompt,
                                      args.max_tokens, args.temperature)
        print(f'  [No secret]      "{out}"')

        # 4. ベースモデル (Phase 1 のみの重み、秘密なし)
        if has_base:
            base_model = load_model(base_weights_path, config)
            base_pipeline = SecretInferencePipeline(base_model, key_manager, tokenizer)
            out = generate_without_secret(base_pipeline, base_model, prompt,
                                          args.max_tokens, args.temperature)
            print(f'  [Base (Phase1)]  "{out}"')

        print()

    print("--- 解釈 ---")
    print("Correct secret: Phase 2 で学習した秘密空間での推論 → coherent な出力")
    print("Wrong secret:   誤った秘密空間 → 崩れた出力")
    print("No secret:      秘密なしで Phase 2 済み重みを使用 → 崩れた出力")
    if has_base:
        print("Base (Phase1):   Phase 1 のみの重み (秘匿化前) → ある程度 coherent")


if __name__ == "__main__":
    main()
