"""秘密適応 (adapt_to_secret) によるセッション固有モデルの生成。

Phase 1 で保存した public_weights_base.pt と master_secret.hex を読み込み、
指定セッションの秘密を注入して公開重みを適応させる。
重みはセッションごとに別ファイルに保存される。

使い方:
  # 基本: ベースからセッション適応 (180ステップ)
  python3 -m examples.demo_finetune --session sess-001

  # ステップ数・学習率を調整
  python3 -m examples.demo_finetune --session sess-001 --steps 300 --lr 2e-4

  # 既存セッション重みから別セッションへ切り替え
  python3 -m examples.demo_finetune --session sess-002 --from sess-001

  # ベースからやり直す場合
  python3 -m examples.demo_finetune --session sess-001 --from base
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secret_llm.model.config import TransformerConfig
from secret_llm.crypto.key_manager import KeyManager
from secret_llm.pipeline.training import SecretTrainingPipeline


TRAIN_TEXT = (
    "the cat sat on the mat. "
    "the dog sat on the log. "
    "the bird flew over the tree. "
    "the fish swam in the sea. "
    "the cat chased the dog around the yard. "
    "the bird sang a song in the morning. "
    "the fish jumped out of the water. "
    "the dog barked at the cat on the mat. "
) * 50


def main():
    parser = argparse.ArgumentParser(description="秘密適応: adapt_to_secret")
    parser.add_argument("--session", required=True, help="適応先のセッションID")
    parser.add_argument("--steps", type=int, default=180, help="適応ステップ数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学習率")
    parser.add_argument("--from", dest="from_source", type=str, default=None,
                        help="ロード元: 'base' またはセッションID (省略時は自動判定)")
    parser.add_argument("--secret", type=str, default=None, help="master_secret.hex のパス")
    args = parser.parse_args()

    examples_dir = os.path.dirname(__file__)
    base_weights_path = os.path.join(examples_dir, "public_weights_base.pt")
    session_weights_path = os.path.join(examples_dir, f"public_weights_{args.session}.pt")
    secret_path = args.secret or os.path.join(examples_dir, "master_secret.hex")

    # --- ロード元の決定 ---
    if args.from_source == "base":
        load_path = base_weights_path
    elif args.from_source:
        load_path = os.path.join(examples_dir, f"public_weights_{args.from_source}.pt")
    elif os.path.exists(session_weights_path):
        load_path = session_weights_path
    else:
        load_path = base_weights_path

    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found.")
        if load_path == base_weights_path:
            print("Run: python3 -m examples.demo_training")
        sys.exit(1)
    if not os.path.exists(secret_path):
        print(f"Error: {secret_path} not found. Run demo_training.py first.")
        sys.exit(1)

    with open(secret_path) as f:
        master_secret = bytes.fromhex(f.read().strip())

    config = TransformerConfig()
    pipeline = SecretTrainingPipeline(config, master_secret, args.session)
    pipeline.load_public_weights(load_path)

    pub, sec = pipeline.model.count_parameters()
    print(f"=== adapt_to_secret ===")
    print(f"Loaded from: {load_path}")
    print(f"Session:     {args.session}")
    print(f"Steps:       {args.steps}")
    print(f"LR:          {args.lr}")
    print(f"Params:      {pub:,} public / {sec:,} secret\n")

    # --- 適応 ---
    losses = pipeline.adapt_to_secret(
        TRAIN_TEXT, session_id=args.session, steps=args.steps,
        lr=args.lr, batch_size=32, seq_len=64,
    )

    # --- 保存 ---
    pipeline.save_public_weights(session_weights_path)
    print(f"Saved to: {session_weights_path}")
    print(f"\nNext: python3 -m examples.demo_inference --session {args.session}")


if __name__ == "__main__":
    main()
