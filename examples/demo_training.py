"""2段階学習 + 高速適応デモ。

Phase 1: 秘密なしでベースモデルを事前学習
Phase 2: 秘密注入済みで公開重みをfine-tuning (Session A)
Phase 3: adapt_to_secret() で別セッション (Session B) に高速切り替え

実行: python3 -m examples.demo_training
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secret_llm.model.config import TransformerConfig
from secret_llm.crypto.key_manager import KeyManager
from secret_llm.pipeline.training import SecretTrainingPipeline


# === 学習テキスト ===
# 小さなテキストで文字レベルの次文字予測を学習する
TRAIN_TEXT = (
    "the cat sat on the mat. "
    "the dog sat on the log. "
    "the bird flew over the tree. "
    "the fish swam in the sea. "
    "the cat chased the dog around the yard. "
    "the bird sang a song in the morning. "
    "the fish jumped out of the water. "
    "the dog barked at the cat on the mat. "
) * 50  # 繰り返してデータ量を確保


def main():
    config = TransformerConfig()
    master_secret = KeyManager.generate_master_secret()
    session_id = "training-session-001"

    print("=== Secret Reasoning LLM Training Demo ===")
    print(f"Model: {config.n_layers} layers, d_model={config.d_model}, {config.n_heads} heads")

    pipeline = SecretTrainingPipeline(config, master_secret, session_id)
    pub, sec = pipeline.model.count_parameters()
    print(f"Public params: {pub:,} / Secret params: {sec:,}\n")

    # --- Phase 1: ベース学習 (秘密なし) ---
    print("--- Phase 1: Base Training (no secrets) ---")
    base_losses = pipeline.train_base(
        TRAIN_TEXT, epochs=40, lr=3e-4, batch_size=32, seq_len=64,
    )

    # --- 公開重み保存 (Phase 1 完了時点) ---
    examples_dir = os.path.dirname(__file__)
    base_weights_path = os.path.join(examples_dir, "public_weights_base.pt")
    secret_path = os.path.join(examples_dir, "master_secret.hex")
    pipeline.save_public_weights(base_weights_path)
    with open(secret_path, "w") as f:
        f.write(master_secret.hex())

    print(f"\n=== Phase 1 Complete ===")
    print(f"Base loss: {base_losses[0]:.4f} -> {base_losses[-1]:.4f}")
    print(f"Weights saved to: {base_weights_path}")
    print(f"Secret saved to:  {secret_path}")
    print(f"\nNext: python3 -m examples.demo_finetune --session sess-001")


if __name__ == "__main__":
    main()
