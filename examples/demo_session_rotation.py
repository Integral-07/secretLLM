"""セッションローテーション + 高速適応デモ。

同じ入力に対し異なるセッションの内部表現を比較し、
秘匿が機能していること (cosine similarity ≈ 0) を確認する。
さらに adapt_to_secret() による適応速度を可視化する。

実行: python3 -m examples.demo_session_rotation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from secret_llm.model.config import TransformerConfig
from secret_llm.model.transformer import SecretTransformer
from secret_llm.model.tokenizer import CharTokenizer
from secret_llm.crypto.key_manager import KeyManager
from secret_llm.pipeline.inference import SecretInferencePipeline
from secret_llm.pipeline.training import SecretTrainingPipeline


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """2つのテンソルをflattenしてcosine similarityを計算する。"""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def get_logits(model, tokenizer, text):
    """入力テキストに対するlogitsを取得する。"""
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    T = input_tensor.size(1)
    mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        return model(input_tensor, mask=mask)


def main():
    config = TransformerConfig()
    master_secret = KeyManager.generate_master_secret()
    model = SecretTransformer(config)
    tokenizer = CharTokenizer(config.vocab_size)
    key_manager = KeyManager(master_secret)
    pipeline = SecretInferencePipeline(model, key_manager, tokenizer)

    test_text = "hello world"
    sessions = ["session-A", "session-B", "session-C"]

    print("=== Session Rotation Demo ===")
    print(f"Input text: '{test_text}'")
    print(f"Sessions: {sessions}\n")

    # --- 各セッションの内部表現とlogitsを取得 ---
    all_reps = {}
    all_logits = {}
    for sid in sessions:
        pipeline.start_session(sid)
        reps = pipeline.get_internal_representations(test_text)
        logits = get_logits(model, tokenizer, test_text)
        all_reps[sid] = reps
        all_logits[sid] = logits
        pipeline.end_session()

    # --- 1. 層出力のcosine similarity (残差接続込み) ---
    print("--- 1. Hidden State Similarity (includes residual) ---")
    print("   (残差接続によりembeddingが支配的 → 高いsimilarityは想定通り)")
    layer_names = sorted(all_reps[sessions[0]].keys())
    print(f"{'':>22}", end="")
    for ln in layer_names:
        print(f"  {ln:>10}", end="")
    print()
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            sa, sb = sessions[i], sessions[j]
            print(f"  {sa} vs {sb}:", end="")
            for ln in layer_names:
                sim = cosine_sim(all_reps[sa][ln], all_reps[sb][ln])
                print(f"  {sim:>10.6f}", end="")
            print()

    # --- 2. 層出力の差分ベクトル (残差を除去して秘密成分のみ比較) ---
    print("\n--- 2. Secret Component Similarity (residual removed) ---")
    print("   (秘密なしの出力を基準とし、各セッションの差分ベクトルを比較)")

    # 秘密なしの内部表現を基準として取得
    model.clear_secrets()
    pipeline._current_session = "baseline"
    baseline_reps = pipeline.get_internal_representations(test_text)
    pipeline._current_session = None

    # 各セッションの差分 = (秘密あり出力) - (秘密なし出力)
    deltas = {}
    for sid in sessions:
        deltas[sid] = {}
        for ln in layer_names:
            deltas[sid][ln] = all_reps[sid][ln] - baseline_reps[ln]

    print(f"{'':>22}", end="")
    for ln in layer_names:
        print(f"  {ln:>10}", end="")
    print()
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            sa, sb = sessions[i], sessions[j]
            print(f"  {sa} vs {sb}:", end="")
            for ln in layer_names:
                sim = cosine_sim(deltas[sa][ln], deltas[sb][ln])
                print(f"  {sim:>10.4f}", end="")
            print()

    # --- 3. Logits (最終出力) の比較 ---
    print("\n--- 3. Logits Similarity (final output) ---")
    print("   (モデルの最終出力 → 次トークン予測の分布がどれだけ異なるか)")
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            sa, sb = sessions[i], sessions[j]
            sim = cosine_sim(all_logits[sa], all_logits[sb])
            print(f"  {sa} vs {sb}: {sim:.6f}")

    # --- 4. 決定論性確認 (同一セッション2回) ---
    print("\n--- 4. Determinism Check (same session twice) ---")
    pipeline.start_session(sessions[0])
    reps_1 = pipeline.get_internal_representations(test_text)
    logits_1 = get_logits(model, tokenizer, test_text)
    pipeline.end_session()

    pipeline.start_session(sessions[0])
    reps_2 = pipeline.get_internal_representations(test_text)
    logits_2 = get_logits(model, tokenizer, test_text)
    pipeline.end_session()

    for ln in layer_names:
        sim = cosine_sim(reps_1[ln], reps_2[ln])
        print(f"  {ln}: {sim:.6f}")
    sim_logits = cosine_sim(logits_1, logits_2)
    print(f"  logits:  {sim_logits:.6f}")
    print("  (should be exactly 1.000000)")

    # --- 5. 高速適応デモ ---
    print("\n--- 5. Fast Adaptation Speed ---")
    print("   (adapt_to_secret() で新セッションに切り替えた際の損失推移)")

    train_text = (
        "the cat sat on the mat. "
        "the dog sat on the log. "
        "the bird flew over the tree. "
    ) * 50

    train_pipeline = SecretTrainingPipeline(config, master_secret, sessions[0])

    # ベース学習 (軽量)
    print("  Base training (5 epochs)...")
    train_pipeline.train_base(train_text, epochs=5, lr=3e-4, batch_size=32, seq_len=64)

    # Session A で秘密学習
    print("  Secret fine-tuning on Session A (5 epochs)...")
    train_pipeline.train_with_secret(train_text, epochs=5, lr=1e-4, batch_size=32, seq_len=64)

    # Session B に高速適応
    print(f"  Adapting to Session B ({sessions[1]})...")
    adapt_losses = train_pipeline.adapt_to_secret(
        train_text, session_id=sessions[1], steps=50, lr=1e-4, batch_size=32, seq_len=64,
    )

    # 適応カーブ表示 (10ステップごと)
    print("\n  Adaptation curve (every 10 steps):")
    for i in range(0, len(adapt_losses), 10):
        bar_len = int(adapt_losses[i] * 10)
        bar = "#" * min(bar_len, 50)
        print(f"    step {i:>3}: {adapt_losses[i]:.4f} {bar}")
    print(f"    step {len(adapt_losses) - 1:>3}: {adapt_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
