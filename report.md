# 秘匿推論LLMアーキテクチャ 技術レポート

## 1. 概要

本プロジェクト **secretLLM** は、LLMが入力の平文や意味内容を直接知ることなく推論を成立させる、暗号的発想に基づく秘匿推論アーキテクチャの概念実証 (PoC) である。

Transformerの意味空間とAttention構造を秘密Adapterで歪めることで、モデル内部の中間表現（Hidden States, Attention Map）を秘密鍵なしでは解釈不能にする。公開鍵暗号の「公開鍵で暗号化、秘密鍵で復号」という構造を、「公開重みで推論構造を定義、秘密重みで意味空間を制御」に対応させている。

### 公開鍵暗号との対応関係

| 暗号理論 | 本提案 |
|---|---|
| 公開鍵 | 公開モデル重み (W_q, W_k, W_v, FFN等) |
| 秘密鍵 | ゲーティングAdapter重み (W_down, W_up, gate_bias) + 秘密射影行列 (S_q, S_k) |
| 暗号文 | 秘匿されたベクトル表現 (Hidden States) |
| 復号 | 正しい秘密重み下での推論 |

### なぜ通常の暗号（AES等）ではダメか

AES等の対称鍵暗号では「計算前に復号」が必要であり、LLMは復号後の平文を必ず見る。本提案では入力をベクトルのまま秘匿した状態で推論する。

---

## 2. 脅威モデル

### 想定する攻撃者の能力

- モデル重み（公開パラメータ）を完全に入手できる
- 推論時の中間表現（Attention重み、Hidden States）を観測できる
- 長時間の統計的観測が可能

### 守りたいもの

- 入力文の意味内容
- トークン間の意味的関係（Attention構造）
- 推論経路

---

## 3. アーキテクチャ全体像

### 3.1 モデル仕様

| パラメータ | 値 | 説明 |
|---|---|---|
| vocab_size | 256 | 文字レベル (ASCII) |
| max_seq_len | 128 | 最大系列長 |
| d_model | 128 | 埋め込み次元 |
| n_heads | 4 | Attentionヘッド数 |
| d_head | 32 | ヘッドあたり次元 |
| d_ff | 512 | FFN中間次元 (4x d_model) |
| n_layers | 4 | Transformer層数 |
| adapter_rank | 16 | Adapter低ランク次元 |

公開パラメータ: **824,064** / 秘密パラメータ: **65,536** (比率 7.9%)

### 3.2 データフロー（1層あたり）

```
入力 x (B, T, 128)
│
├─ LayerNorm
│  ├─ SecretProjectionAttention     ← Q'=Q@S_q, K'=K@S_k で注意パターンを秘匿
│  │     Q = x @ W_q (公開)
│  │     K = x @ W_k (公開)
│  │     V = x @ W_v (公開)
│  │     Q' = Q @ S_q (秘密)        ← 意味空間を回転
│  │     K' = K @ S_k (秘密)        ← 意味空間を回転
│  │     A = softmax(Q'K'^T / √d)   ← 秘匿されたAttention Map
│  │     out = A @ V                 ← Vは未変更
│  │
│  └─ SecretGatingAdapter (attn)      ← 秘密ゲーティングAdapter
│        h = GELU(out @ W_down) @ W_up   秘密変換
│        gate = sigmoid(gate_bias + 0.5*h) ゲート計算
│        out = out * gate                 信号制御
│        ※ 秘密なし: gate≈0 → 信号遮断
│        ※ 正しい秘密: gate≈1 → 信号通過
│
├─ + 残差 → x_new
│
├─ LayerNorm
│  ├─ FeedForward                    ← 公開 (128→512→128)
│  └─ SecretGatingAdapter (ffn)      ← 秘密ゲーティングAdapter
│
└─ + 残差 → 出力
```

この構造を4層積み重ねた後、最終LayerNorm → 出力射影 (weight tying) でlogitsを生成する。

---

## 4. 秘匿メカニズム

### 4.1 秘密射影Attention（コアアイデア①）

Transformerで最も情報が漏れるのはAttention Map `A = softmax(QK^T)` である。この行列は文構造、推論経路、重要トークンをほぼそのまま表す。

本提案ではQ, Kに秘密の直交行列 S_q, S_k を右から掛ける：

```
Q' = Q @ S_q
K' = K @ S_k
A  = softmax(Q' @ K'^T / √d_head)
```

S_q, S_k はヘッドごと・層ごと・セッションごとに異なる直交行列である。

**直交行列を使う理由**:
- ベクトルのノルムを保存する (`||Qx|| = ||x||`)
- Attention scoreの数値安定性を維持する
- 意味空間を「回転」するが「潰さない」

**Vを変更しない理由**:
- 「何に注目するか」(routing) は秘匿される
- 「何を集約するか」(content) は変更されない
- 学習時にV経由の勾配がクリーンに保たれ、学習安定性が向上する

### 4.2 秘密ゲーティングAdapter / SecretGatingAdapter（コアアイデア②）

Attention後とFFN後に配置される秘密依存のゲーティングモジュール。sigmoidゲートで信号の通過/遮断を制御し、秘密の有無による出力差を**構造的に保証**する。

```
ゲーティング変換:
  h = GELU(x @ W_down) @ W_up        秘密変換 (低ランクボトルネック)
  gate = sigmoid(gate_bias + scale * h)  ゲート値 [0, 1]
  output = x * gate                   要素ごとのゲーティング
```

**3つの状態と挙動**:

| 状態 | gate_bias | W_down, W_up | gate値 | 出力 |
|---|---|---|---|---|
| Phase 1 (初期化) | +5 (GATE_OPEN) | ゼロ | sigmoid(5) ≈ 0.993 | ≈ x（ほぼ恒等写像） |
| 正しい秘密注入 | セッション固有値 (+2.5〜5.0) | セッション固有 | 秘密依存パターン | 適応後は正常出力 |
| 秘密クリア | -5 (GATE_CLOSED) | ゼロ | sigmoid(-5) ≈ 0.007 | ≈ 0（**信号99.3%遮断**） |
| 間違った秘密 | 異なる値 | 異なる重み | 不正なパターン | 出力崩壊 |

**旧方式（ReversibleAdapter）からの変更理由**:

旧方式は Additive Coupling (NICE方式) で可逆変換を実装していたが、ゼロ重み時に恒等写像となるため「秘密なしでも正常に動作する」問題があった。ゲーティング方式では `clear_secrets()` 時に gate_bias=-5 を設定することで信号を構造的に遮断し、秘密の有無による出力差を保証する。可逆性は犠牲になるが、推論パイプラインで逆変換は使用されていなかったため実用上の影響はない。

**パラメータ数 (4,224/adapter)**:
- W_down (128, 16) + W_up (16, 128) = 4,096
- gate_bias (128) = 128

**非線形性 (GELU) が必須である理由**:
非線形性がなければ秘密変換が線形に退化し、公開重みに吸収されて秘匿効果がなくなる。GELUを挟むことで入力依存の非線形歪みが生じ、事前計算で除去できなくなる。

**スケール係数 0.5 の設計**:
- 大きすぎ (>1.0): 学習が不安定
- 小さすぎ (<0.1): 秘匿効果が弱い（旧方式で実験的に確認）
- 0.5は「ゲートが秘密に十分感度を持ちつつ学習を安定させる」バランス点

### 4.3 鍵階層とセッションローテーション（コアアイデア③）

永続的な秘密は長期観測やメモリダンプで漏洩するリスクがある。これを防ぐためにHKDF (HMAC-based Key Derivation Function) による鍵階層を構築する。

```
Master Secret (ユーザ保持、32バイト)
  │
  │ HKDF-Extract: HMAC-SHA256(salt="secret-llm-v1", master_secret)
  ↓
PRK (擬似乱数鍵、32バイト)
  │
  │ HKDF-Expand: info="session:<session_id>"
  ↓
Session Secret (32バイト、セッション固有)
  │
  │ HKDF-Expand: info="layer:<i>:<component>"
  ↓
Component Seed (8バイト) → torch.Generator → テンソル
```

これにより：
- 同じ入力でもセッションが変われば内部表現が完全に変わる
- 1つのセッションの秘密が漏洩しても他のセッションは影響を受けない
- マスターシークレットが1つあれば無数のセッションを生成できる

---

## 5. 学習戦略

### Phase 1: ベース学習（秘密なし）

秘密を注入しない状態 (S_q=S_k=I, Adapter=0) で次文字予測 (character-level language modeling) を学習する。標準的なTransformerとして公開重みの言語能力を獲得する。

### Phase 2: 秘密付きfine-tuning

秘密重みを注入・固定 (requires_grad=False) し、公開重みのみを低学習率で再学習する。公開重みが秘密空間に共適応し、「正しい秘密がないと正しい出力が出ない」状態を作る。

Phase 2の初期に損失が跳ね上がる（実験値: 0.10→0.66）のが重要な指標であり、秘密注入により意味空間が歪んだことを示す。fine-tuningで秘密空間に適応し回復する（0.66→0.08）。

### Phase 3: 高速適応 (adapt_to_secret)

Phase 2完了後に別のセッションへ切り替える際の高速再適応。Phase 2と同じ処理（秘密注入→公開重みの再学習）だが、ステップ数ベース（デフォルト50ステップ）で実行することで高速に収束させる。

```
adapt_to_secret(text, session_id="new-session", steps=50, lr=1e-4)
```

- エポック数ではなくステップ数で制御（`itertools.cycle(dataloader)` でデータを循環利用）
- Phase 2でフル学習済みのモデルからの「切り替え」を想定しており、既に秘密空間への適応力を持つ公開重みが少ないステップで新しい秘密に再適応する
- SecretGatingAdapterのゲート構造により、適応は秘密パターンへの公開重みの共適応として進行する

---

## 6. 実験結果

### 6.1 学習曲線

```
Phase 1 (Base):   15.88 → 0.10  (20 epochs, lr=3e-4)
Phase 2 (Secret):  0.66 → 0.08  (10 epochs, lr=1e-4)
```

### 6.2 推論結果の比較（ゲーティングAdapter導入後）

入力に対する4条件での生成結果（temperature=0.1, adapt_to_secret 300ステップ）：

**Prompt: `"the cat "`**

| 条件 | 出力 | 判定 |
|---|---|---|
| 正しい秘密 | `on the mat. the cat sat on the mat. the ` | coherent ✓ |
| 間違った秘密 | ` sw on the the t t on thee t t the t t t` | garbled ✓ |
| 秘密なし | `                                        ` | **信号遮断** ✓ |
| Base (Phase1) | `d t t t on the the le bired bired t t t ` | garbled ✓ |

**Prompt: `"the dog "`**

| 条件 | 出力 | 判定 |
|---|---|---|
| 正しい秘密 | `around the yard. the bird sang a song in` | coherent ✓ |
| 間違った秘密 | `t og. the t t og og the og t og t the t ` | garbled ✓ |
| 秘密なし | `                                        ` | **信号遮断** ✓ |
| Base (Phase1) | `t t t t the bire d d se w the se se se s` | garbled ✓ |

**Prompt: `"the bird "`**

| 条件 | 出力 | 判定 |
|---|---|---|
| 正しい秘密 | `flew over the tree. the fish swam in the` | coherent ✓ |
| 間違った秘密 | `thee she she t sw og. thee t t theee t t` | garbled ✓ |
| 秘密なし | `                                        ` | **信号遮断** ✓ |
| Base (Phase1) | `se se se se se se se se se w se se se se` | garbled ✓ |

**3つのセキュリティ特性がすべて成立**:
1. **正しい鍵** → 学習テキストと一致する正常な出力
2. **間違った鍵** → ゲートパターンが不正 → 出力崩壊
3. **鍵なし** → gate=sigmoid(-5)≈0.007 → 信号99.3%遮断 → 完全に空白

旧方式（ReversibleAdapter, scale=0.5）では「秘密なし」でも正常出力が得られていた問題が、ゲーティング方式への移行で完全に解決された。

### 6.3 旧方式（ReversibleAdapter）との比較

| 方式 | 正しい秘密 | 間違った秘密 | 秘密なし |
|---|---|---|---|
| ReversibleAdapter (scale=0.5) | coherent ✓ | garbled ✓ | **coherent ✗** (恒等写像のため) |
| SecretGatingAdapter (scale=0.5) | coherent ✓ | garbled ✓ | **信号遮断 ✓** (ゲート閉鎖) |

### 6.4 テスト結果

47テスト全パス。暗号層、秘密Attention、ゲーティングAdapter（信号遮断テスト含む）、Transformer統合、決定論性の全項目を検証済み。

---

## 7. コード構成と各モジュールの詳細

### 7.1 プロジェクト構造

```
secretLLM/
├── secret_llm/
│   ├── crypto/              # 暗号層
│   │   ├── key_manager.py   # HKDF鍵階層
│   │   └── weight_generator.py  # バイト→テンソル変換
│   ├── model/               # モデル層
│   │   ├── config.py        # ハイパーパラメータ
│   │   ├── tokenizer.py     # 文字レベルトークナイザ
│   │   ├── embedding.py     # 埋め込み + 位置エンコーディング
│   │   ├── feedforward.py   # FFN
│   │   ├── attention.py     # 標準Attention (比較用)
│   │   ├── secret_attention.py  # 秘密射影Attention
│   │   ├── secret_adapter.py    # 秘密Adapter
│   │   ├── transformer_block.py # 1層の組み立て
│   │   └── transformer.py       # 全体モデル
│   └── pipeline/            # パイプライン層
│       ├── training.py      # 2段階学習 + 高速適応
│       └── inference.py     # 推論 + セッション管理
├── tests/                   # テスト (47件)
└── examples/                # デモスクリプト
    ├── demo_training.py     # Phase 1 (ベース学習)
    ├── demo_finetune.py     # Phase 2/3 (fine-tuning / 高速適応)
    ├── demo_inference.py    # 推論デモ
    └── demo_session_rotation.py  # セッションローテーション + 適応速度デモ
```

### 7.2 暗号層

#### `key_manager.py`

| クラス/関数 | 役割 |
|---|---|
| `KeyManager.__init__(master_secret)` | HKDF-Extractをhmac.newで実行し、PRK (擬似乱数鍵) を生成・保持する |
| `KeyManager.derive_session(session_id)` | HKDF-ExpandでPRKからセッション固有の32バイト鍵を導出し、SessionKeySetとして返す |
| `KeyManager.generate_master_secret()` | `os.urandom(32)` で暗号学的に安全な32バイトのマスターシークレットを生成する |
| `SessionKeySet.__init__(session_secret)` | セッション鍵を保持する |
| `SessionKeySet.derive_component_bytes(layer_idx, component, num_bytes)` | infoラベル `"layer:{i}:{component}"` でHKDF-Expandを実行し、層・コンポーネントに固有のバイト列を導出する |
| `SessionKeySet._derive_large(base_info, num_bytes)` | 8160バイト超の導出をチャンク分割で処理する（実際には8バイトシードのみ使うため通常は不要） |

#### `weight_generator.py`

| クラス/関数 | 役割 |
|---|---|
| `WeightGenerator.bytes_to_tensor(raw_bytes, shape)` | 先頭8バイトをtorch.Generatorのシードに使い、正規分布テンソルを生成する。fan_inスケーリング (1/√fan_in) を適用して初期値を適切な大きさに調整する |
| `WeightGenerator.bytes_to_orthogonal(raw_bytes, shape)` | シードからランダム行列を生成し、QR分解で直交行列を抽出する。対角符号補正 (`sign(diag(R))`) でQR分解の非一意性を解消し決定論性を保証する |
| `WeightGenerator.generate_adapter_weights(session_keys, layer_idx, position, d_model, rank)` | SessionKeySetからSecretGatingAdapter用の重みとゲートバイアスを生成する。W_down (d_model, rank), W_up (rank, d_model) は `bytes_to_tensor()` で正規分布から生成。gate_bias (d_model,) は `abs(randn) + 2.5` で正の値 [2.5, ~5.0] に生成し、sigmoidゲートが開放状態になるようにする。positionは "attn" か "ffn" |
| `WeightGenerator.generate_secret_projections(session_keys, layer_idx, n_heads, d_head)` | SessionKeySetからS_q, S_k (n_heads, d_head, d_head) の直交行列ペアを生成する |

### 7.3 モデル層

#### `config.py`

| クラス | 役割 |
|---|---|
| `TransformerConfig` | 全ハイパーパラメータを保持するdataclass。モデル次元、秘密Adapter設定、特殊トークンIDを一元管理する |

#### `tokenizer.py`

| クラス/関数 | 役割 |
|---|---|
| `CharTokenizer.__init__(vocab_size)` | 語彙サイズを設定する。特殊トークン (PAD=0, BOS=1, EOS=2) を予約し、一般文字はオフセット+3で配置する |
| `CharTokenizer.encode(text, add_bos, add_eos)` | 文字列を `ord(ch) + 3` でトークンIDリストに変換する。特殊トークンID (0-2) との衝突を回避する |
| `CharTokenizer.decode(ids)` | トークンIDリストを文字列に逆変換する。特殊トークンはスキップする |
| `CharTokenizer.encode_batch(texts, max_len)` | 複数文字列をPAD付きのバッチテンソル (B, max_len) に変換する |

#### `embedding.py`

| クラス/関数 | 役割 |
|---|---|
| `TokenEmbedding.__init__(config)` | nn.Embedding (vocab_size × d_model) を作成する。padding_idx=0でPADトークンを常にゼロベクトルにする |
| `TokenEmbedding.forward(token_ids)` | 埋め込みルックアップ後、√d_modelでスケーリングする。これは位置エンコーディングとの大きさのバランスを取るための標準手法 |
| `PositionalEncoding.__init__(config)` | 正弦波テーブル (max_seq_len × d_model) を事前計算し、register_bufferで登録する。偶数次元=sin、奇数次元=cos |
| `PositionalEncoding.forward(x)` | 入力テンソルに位置エンコーディングを加算し、dropoutを適用する |

#### `feedforward.py`

| クラス/関数 | 役割 |
|---|---|
| `PositionwiseFeedForward.__init__(config)` | 2層MLP (d_model→d_ff→d_model) をGELU活性化とdropoutで構成する |
| `PositionwiseFeedForward.forward(x)` | 各トークン位置に独立に同じMLPを適用する。128→512に拡張して非線形変換し、128に戻す |

#### `attention.py`

| クラス/関数 | 役割 |
|---|---|
| `MultiHeadAttention.__init__(config)` | 標準的なMulti-Head Attentionを構成する。W_q, W_k, W_v, W_o (全てbias=False) とdropout、スケーリング係数 (1/√d_head) を持つ。秘密機構のない比較ベースライン |
| `MultiHeadAttention.forward(x, mask)` | Q,K,Vに線形射影→ヘッド分割→スケールドドット積→softmax→V集約→ヘッド結合→出力射影の標準フローを実行する |

#### `secret_attention.py`

| クラス/関数 | 役割 |
|---|---|
| `SecretProjectionAttention.__init__(config)` | 標準Attentionの構成に加え、S_q, S_k (n_heads, d_head, d_head) を単位行列で初期化する。requires_grad=Falseで学習対象外とする |
| `SecretProjectionAttention.set_secret_projections(s_q, s_k)` | HKDFから派生した直交行列をS_q, S_kにコピーする。`.data.copy_()` で計算グラフを経由せず直接書き換える |
| `SecretProjectionAttention.forward(x, mask)` | Q,K,Vへの公開射影後、`Q' = einsum("bhtd,hde->bhte", Q, S_q)` で秘密射影を適用する。Q'K'^Tでスコアを計算し、Vには射影を適用しない |

#### `secret_adapter.py`

| クラス/関数 | 役割 |
|---|---|
| `SecretGatingAdapter.__init__(d_model, rank, nonlinearity, scale)` | W_down (d_model, rank), W_up (rank, d_model) をゼロ初期化、gate_bias (d_model,) を GATE_OPEN (+5) で初期化する (全て requires_grad=False)。初期状態では gate=sigmoid(5)≈0.993 でほぼ恒等写像となり Phase 1 学習を妨げない |
| `SecretGatingAdapter._secret_transform(x)` | 秘密変換 `GELU(x @ W_down) @ W_up` を計算する |
| `SecretGatingAdapter.set_secret_weights(w_down, w_up, gate_bias)` | HKDFから派生した秘密重みとゲートバイアスを注入する |
| `SecretGatingAdapter.forward(x)` | ゲーティング変換。`h = secret_transform(x)`, `gate = sigmoid(gate_bias + scale * h)`, `output = x * gate` で信号をゲート制御する |
| `SecretAdapter` | `SecretGatingAdapter` の後方互換エイリアス |

#### `transformer_block.py`

| クラス/関数 | 役割 |
|---|---|
| `SecretTransformerBlock.__init__(config)` | 1層分のコンポーネントを組み立てる: LayerNorm×2、SecretProjectionAttention、PositionwiseFeedForward、SecretGatingAdapter×2 (Attention後・FFN後) |
| `SecretTransformerBlock.forward(x, mask)` | Pre-LayerNorm方式で実行する。(1) norm→attention→adapter_attn→残差加算、(2) norm→ffn→adapter_ffn→残差加算 |

#### `transformer.py`

| クラス/関数 | 役割 |
|---|---|
| `SecretTransformer.__init__(config)` | TokenEmbedding、PositionalEncoding、SecretTransformerBlock×n_layers、最終LayerNorm、出力射影を構成する。出力射影とEmbeddingでweight tyingを行い、パラメータ数を削減する |
| `SecretTransformer.forward(token_ids, mask)` | 埋め込み→位置エンコーディング→N層のTransformerBlock→最終LayerNorm→出力射影(logits)の完全なforward passを実行する |
| `SecretTransformer.inject_secrets(key_manager, session_id)` | KeyManagerからSessionKeySetを導出し、WeightGeneratorで全層のS_q, S_k, Adapter重み (W_down, W_up, gate_bias) を生成して注入する。これが暗号層とモデル層の接続点である |
| `SecretTransformer.clear_secrets()` | 全秘密パラメータをリセットする: S_q,S_k→単位行列、Adapter W_down/W_up→ゼロ、gate_bias→GATE_CLOSED (-5)。ゲートが閉鎖され信号が遮断される |
| `SecretTransformer.count_parameters()` | requires_gradの有無で公開/秘密パラメータ数をカウントして返す |

### 7.4 パイプライン層

#### `training.py`

| クラス/関数 | 役割 |
|---|---|
| `TextDataset.__init__(text, tokenizer, seq_len)` | テキスト全体をエンコードし、固定長チャンクに分割可能な形で保持する |
| `TextDataset.__getitem__(idx)` | 位置idxから始まるseq_len+1文字のチャンクを取り出し、1文字ずらしの(入力, ターゲット)ペアを返す |
| `SecretTrainingPipeline.__init__(config, master_secret, session_id)` | モデル、KeyManager、トークナイザを初期化する |
| `SecretTrainingPipeline._create_causal_mask(seq_len)` | 下三角行列のcausal mask (1, 1, T, T) を生成する。未来のトークンへの注意を遮断する |
| `SecretTrainingPipeline._train_epoch(dataloader, optimizer)` | 1エポック分のミニバッチ学習を実行し、平均cross-entropy損失を返す |
| `SecretTrainingPipeline.train_base(text, epochs, lr, ...)` | **Phase 1**: `clear_secrets()` で秘密なし状態を保証し、AdamW (lr=3e-4) で公開重みを学習する |
| `SecretTrainingPipeline.train_with_secret(text, epochs, lr, ...)` | **Phase 2**: `inject_secrets()` で秘密を注入・固定し、AdamW (lr=1e-4, Phase 1の1/3) で公開重みのみを再学習する |
| `SecretTrainingPipeline.adapt_to_secret(text, session_id, steps, lr, ...)` | **Phase 3**: 新しいセッション秘密への高速適応。`inject_secrets()` で新秘密を注入し、ステップ数ベース（デフォルト50ステップ）で公開重みを再学習する。`itertools.cycle(dataloader)` でデータを循環利用する |
| `SecretTrainingPipeline.save_public_weights(path)` | requires_grad=Trueのパラメータのみをstate_dictから抽出して保存する。秘密パラメータは含まれない |
| `SecretTrainingPipeline.load_public_weights(path)` | 保存された公開パラメータを `strict=False` で読み込む。秘密パラメータは変更されない |

#### `inference.py`

| クラス/関数 | 役割 |
|---|---|
| `SecretInferencePipeline.__init__(model, key_manager, tokenizer)` | モデル、鍵管理、トークナイザを保持する。現在のセッションIDをNoneで初期化する |
| `SecretInferencePipeline.start_session(session_id)` | `inject_secrets()` を呼び出してセッション固有の秘密重みを注入する |
| `SecretInferencePipeline.end_session()` | `clear_secrets()` で秘密をモデルから消去し、セッションを終了する |
| `SecretInferencePipeline.generate(prompt, max_new_tokens, temperature, top_k)` | 自己回帰テキスト生成を実行する。各ステップで最終位置のlogitsからtemperatureスケーリング→top-kフィルタリング→multinomialサンプリングで次トークンを選択する。EOSで停止する |
| `SecretInferencePipeline.get_internal_representations(text)` | 各Transformer層にforward hookを登録し、forward pass中の中間出力を `{"layer_0": Tensor, ...}` として取得する。セッション間の表現比較に使用する |

---

## 8. 秘密の注入から推論までの完全なフロー

```
1. ユーザがmaster_secretとsession_idを指定

2. KeyManager.__init__(master_secret)
   → HMAC-SHA256(salt, master_secret) でPRKを生成

3. KeyManager.derive_session(session_id)
   → HKDF-Expand(PRK, "session:xxx") でsession_secretを導出
   → SessionKeySetを返す

4. SecretTransformer.inject_secrets() が各層について:
   a. WeightGenerator.generate_secret_projections()
      → SessionKeySet.derive_component_bytes(i, "proj_q", 8)
      → bytes_to_orthogonal() でQR分解を経て直交行列S_qを生成
      → 同様にS_kを生成
      → SecretProjectionAttention.set_secret_projections(S_q, S_k)

   b. WeightGenerator.generate_adapter_weights()
      → SessionKeySet.derive_component_bytes(i, "adapter_attn_down", 8) 等
      → bytes_to_tensor() で W_down, W_up を生成、gate_bias は abs(randn)+2.5 で正の値に生成
      → SecretGatingAdapter.set_secret_weights(W_down, W_up, gate_bias)

   c. FFN後のAdapterも同様

5. 推論 (forward pass):
   token_ids → TokenEmbedding (×√d_model) → PositionalEncoding (+sin/cos)
   → 各SecretTransformerBlock:
       LayerNorm → SecretProjectionAttention(Q@S_q, K@S_k) → SecretGatingAdapter → +残差
       LayerNorm → FFN → SecretGatingAdapter → +残差
   → final LayerNorm → output_proj → logits

6. end_session() で秘密を消去
```

---

## 9. 設計上の重要な判断

### 9.1 なぜeinsumを使うか

`torch.einsum("bhtd,hde->bhte", Q, S_q)` はバッチ(B)とヘッド(H)ごとに、各トークン位置(T)のd_head次元ベクトルにS_qを右から掛ける。ループなしで効率的に計算でき、ヘッドごとに異なる秘密行列を適用する必要があるため `torch.matmul` では記述が煩雑になる。

### 9.2 なぜweight tyingを行うか

出力射影の重みを埋め込みテーブルと共有することで、入力空間と出力空間を同一の意味空間に揃える。パラメータ数も256×128 = 32,768だけ削減される。

### 9.3 なぜPre-LayerNormか

LayerNormを変換の前に適用するPre-Norm方式 (GPT-2スタイル) は、Post-Normより勾配が安定し、learning rate warmupなしでも学習可能である。秘密Adapterが追加の変換を挟む本アーキテクチャでは、勾配の安定性が特に重要である。

### 9.4 バイト→テンソル変換でPRNGシードを使う理由

HKDFで導出したバイト列を直接float32にキャストすると、任意のビットパターンがNaN/Infになりうる。代わりに先頭8バイトをPRNGシードとして使い、`torch.randn` で正規分布テンソルを生成する。8バイト (=64ビット) のシードで任意サイズのテンソルを決定論的に生成でき、HKDFの出力量制限 (8160バイト) も問題にならない。

---

## 10. 既存手法との比較

| 手法 | 秘匿性 | 実用性 | 本提案との差異 |
|---|---|---|---|
| 完全準同型暗号 (FHE) | 数学的に証明可能 | 計算コスト数万倍 | 本提案は実用的な速度で動作するが安全性は発見的 |
| Trusted Execution Environment (TEE) | ハードウェア信頼前提 | 実用的 | 本提案はハードウェア信頼を不要とする |
| Adapter Fine-tuning (LoRA等) | 秘匿目的ではない | 高い | 本提案はAdapterを秘匿手段として転用し、非学習・HKDF派生とする |
| Differential Privacy | 統計的プライバシー | 実用的 | 出力にノイズを加える手法であり、内部表現の秘匿とは異なる |

Attentionと意味空間そのものを秘匿する本提案の設計は、既存研究においてほぼ未踏の領域である。

### 10.1 FHEとの計算量比較: Phase 2 再学習の実用性

本提案はセッション切り替え時に Phase 2 の再学習（fine-tuning）を要する。これを FHE の常時オーバーヘッドと比較し、再学習戦略の実用性を示す。

#### 記法

| 記号 | 意味 |
|---|---|
| $N$ | 学習対象パラメータ数（公開重みのみ） |
| $N_{\text{total}}$ | 全パラメータ数（公開 + 秘密） |
| $D$ | 学習に使う総トークン数 = epochs × dataset_tokens |
| $T$ | 推論時の系列長 |
| $\alpha$ | FHE の計算オーバーヘッド係数（典型的に $10^4$ ~ $10^6$） |

#### Phase 2 の計算量

Transformer 学習の標準的な FLOPs 近似（forward + backward）:

$$C_{\text{phase2}} \approx 6 \cdot N \cdot D$$

現在のモデル（$d_{\text{model}}=128$, $L=4$）:
- $N = 824{,}064$
- データセット ≈ 12,400 文字, seq_len=64 → ~188 サンプル/エポック
- 30 エポック → $D \approx 30 \times 188 \times 64 = 361{,}000$ トークン

$$C_{\text{phase2}} \approx 6 \times 824{,}064 \times 361{,}000 \approx 1.79 \times 10^{12} \text{ FLOPs} \approx 1.8 \text{ TFLOPs}$$

#### FHE 推論の計算量

FHE では推論 1 回あたり:

$$C_{\text{FHE}} = \alpha \cdot C_{\text{inference}}, \quad C_{\text{inference}} \approx 2 \cdot N_{\text{total}} \cdot T$$

- $N_{\text{total}} \approx 890{,}000$, $T = 128$

$$C_{\text{inference}} \approx 2 \times 890{,}000 \times 128 \approx 2.28 \times 10^{8} \text{ FLOPs}$$

#### 損益分岐点

本提案では Phase 2 は 1 回のみ実行し、以後の推論は通常速度。FHE は毎回 $\alpha$ 倍のコストがかかる。$n$ 回推論後の総コスト:

$$\text{本提案: } C_{\text{phase2}} + n \cdot C_{\text{inference}}$$
$$\text{FHE: } n \cdot \alpha \cdot C_{\text{inference}}$$

本提案が有利になる条件:

$$n > \frac{C_{\text{phase2}}}{(\alpha - 1) \cdot C_{\text{inference}}} \approx \frac{C_{\text{phase2}}}{\alpha \cdot C_{\text{inference}}} \quad (\alpha \gg 1)$$

$$n > \frac{1.79 \times 10^{12}}{\alpha \times 2.28 \times 10^{8}} = \frac{7{,}850}{\alpha}$$

| FHE オーバーヘッド $\alpha$ | 損益分岐 $n$（推論回数） | 解釈 |
|---|---|---|
| $10^4$ | $< 1$ | Phase 2 自体が FHE 推論 1 回より安い |
| $10^5$ | $< 1$ | 同上 |
| $10^6$ | $\ll 1$ | 同上 |

$\alpha = 10^4$ でも $C_{\text{FHE}} \approx 2.28 \times 10^{12} \approx 2.3$ TFLOPs であり、$C_{\text{phase2}} \approx 1.8$ TFLOPs と同等。すなわち **FHE で 1 回推論する計算量で Phase 2 の再学習が完了する**。

#### GPT-2 クラスへのスケーリング

モデル規模が大きくなると Phase 2 の絶対コストも増加する。しかし FHE のオーバーヘッドも $N_{\text{total}}$ に比例して増大するため、相対的な優位性は維持される。

| モデル | $N$ (公開) | $C_{\text{phase2}}$ (30ep) | $C_{\text{FHE}}$ 1回 ($\alpha=10^4$) | Phase 2 / FHE 1回 |
|---|---|---|---|---|
| 現行 (128d, 4L) | 824K | 1.8 TFLOPs | 2.3 TFLOPs | 0.78× |
| GPT-2 Small (768d, 12L) | 124M | 269 TFLOPs | 41 TFLOPs | 6.6× |
| GPT-2 Medium (1024d, 24L) | 355M | 769 TFLOPs | 117 TFLOPs | 6.6× |

GPT-2 クラスでは Phase 2 が FHE 推論約 7 回分に相当する。つまり **同一セッションで 7 回以上推論すれば本提案が FHE より計算効率的** であり、これは実用上ほぼ常に満たされる条件である。$\alpha = 10^5$ なら 1 回目から有利になる。

#### adapt_to_secret() による高速適応

セッション切り替え時にフル Phase 2 ではなく `adapt_to_secret()` を使えば、計算量はさらに削減される。$S$ ステップの適応のコスト:

$$C_{\text{adapt}} \approx 6 \cdot N \cdot S \cdot B \cdot T_{\text{seq}}$$

- $S = 50$, $B = 32$, $T_{\text{seq}} = 64$

$$C_{\text{adapt}} \approx 6 \times 824{,}064 \times 50 \times 32 \times 64 \approx 5.07 \times 10^{11} \text{ FLOPs} \approx 0.51 \text{ TFLOPs}$$

これは FHE 推論 1 回（$\alpha = 10^4$）の約 22% であり、セッション切り替えのコストは極めて小さい。

---

## 11. 数学的安全性定義

本節では、secretLLMの3つの秘匿メカニズム（秘密射影Attention、秘密Adapter、セッションローテーション）に対して、計算量的暗号理論の枠組みで安全性を形式的に定義する。

### 記法

| 記号 | 意味 |
|---|---|
| $\lambda$ | セキュリティパラメータ（鍵長に相当、本実装では $\lambda = 256$ ビット） |
| $\mathcal{A}$ | 確率的多項式時間 (PPT) 攻撃者 |
| $\text{negl}(\lambda)$ | 任意の多項式 $p(\lambda)$ に対して十分大きい $\lambda$ で $< 1/p(\lambda)$ となる関数 |
| $\mathcal{F}_\theta$ | 公開パラメータ $\theta$ を持つTransformerモデル |
| $s = (S_q, S_k, W_\downarrow, W_\uparrow, \text{bias})$ | 秘密パラメータの集合（全層分） |
| $\text{KDF}(k, \text{info})$ | HKDF-Expandによる鍵導出 |
| $x \overset{R}{\leftarrow} X$ | 集合 $X$ から一様ランダムに $x$ を選択 |
| $\text{Adv}[\mathcal{A}, \text{Game}]$ | ゲーム Game における攻撃者 $\mathcal{A}$ の優位性 (advantage) |

### 11.1 隠れ状態の意味的秘匿性 (Semantic Security of Hidden States)

**定義 (SS-Hidden).** 秘匿推論スキーム $\Pi$ が隠れ状態について意味的に安全であるとは、任意のPPT攻撃者 $\mathcal{A}$ に対して以下が成り立つことである：

$$\text{Adv}^{\text{ss-hid}}[\mathcal{A}, \Pi] = \left| \Pr[\text{Game}^{\text{ss-hid}}_0(\mathcal{A}) = 1] - \Pr[\text{Game}^{\text{ss-hid}}_1(\mathcal{A}) = 1] \right| \leq \text{negl}(\lambda)$$

**ゲーム $\text{Game}^{\text{ss-hid}}_b$:**

1. チャレンジャーが $k \overset{R}{\leftarrow} \{0,1\}^\lambda$ を生成する
2. $\mathcal{A}$ がモデルの公開パラメータ $\theta$ を受け取る
3. $\mathcal{A}$ が2つの入力 $x_0, x_1$ を選ぶ（$|x_0| = |x_1|$）
4. チャレンジャーが $s \leftarrow \text{KDF}(k, \text{sid})$ で秘密パラメータを生成する
5. チャレンジャーが $h_b \leftarrow \mathcal{F}_{\theta,s}^{(\ell)}(x_b)$（第 $\ell$ 層の隠れ状態）を計算し $\mathcal{A}$ に渡す
6. $\mathcal{A}$ がビット $b'$ を出力する

**直観**: 攻撃者は公開重み $\theta$ と隠れ状態 $h_b$ を観測しても、入力が $x_0$ か $x_1$ かを確率 $1/2$ より有意に高く当てられない。

### 11.2 Attention Map不識別性 (Attention Map Indistinguishability)

Attention Map は最も情報量の多い中間表現である。標準Attentionでは $A = \text{softmax}(QK^T / \sqrt{d})$ が文構造をほぼそのまま表す。

**定義 (IND-Attn).** 秘匿推論スキーム $\Pi$ がAttention Map不識別性を持つとは、任意のPPT攻撃者 $\mathcal{A}$ に対して：

$$\text{Adv}^{\text{ind-attn}}[\mathcal{A}, \Pi] = \left| \Pr[\mathcal{A}(\theta, A_s) = 1] - \Pr[\mathcal{A}(\theta, A_R) = 1] \right| \leq \text{negl}(\lambda)$$

ここで：
- $A_s = \text{softmax}(Q S_q (K S_k)^T / \sqrt{d})$ : 秘密射影によるAttention Map
- $A_R = \text{softmax}(Q R_q (K R_k)^T / \sqrt{d})$ : $R_q, R_k \overset{R}{\leftarrow} O(d)$（直交群からの一様サンプル）

**直観**: 秘密射影によるAttention Mapが、ランダム直交行列によるAttention Mapと識別不能であれば、攻撃者はAttention構造から元の意味的関係を復元できない。

**安全性の帰着**: HKDF出力が擬似乱数的であると仮定すると（PRF仮定）：

$$\text{Adv}^{\text{ind-attn}}[\mathcal{A}, \Pi] \leq \text{Adv}^{\text{prf}}[\mathcal{B}, \text{HKDF}] + \epsilon_{\text{QR}}$$

ここで $\epsilon_{\text{QR}}$ はQR分解による直交行列生成の一様性からの乖離（有限次元上の近似誤差）。

### 11.3 Adapter秘密復元困難性 (Adapter Secret Recovery Hardness)

SecretGatingAdapterの変換 $g_s(x) = x \odot \sigma(\text{bias} + \alpha \cdot \sigma(x W_\downarrow) W_\uparrow)$ において、観測から秘密重み $(W_\downarrow, W_\uparrow, \text{bias})$ を推定する困難性を定義する。

**定義 (SR-Adapter).** 秘密Adapter $g_s$ が $(t, \epsilon)$-秘密復元困難であるとは、計算時間 $t$ 以下の任意の攻撃者 $\mathcal{A}$ に対して：

$$\Pr\left[ \| \hat{s} - s \|_F < \delta \;\middle|\; \hat{s} \leftarrow \mathcal{A}(\theta, \{(x_i, g_s(x_i))\}_{i=1}^{n}) \right] \leq \epsilon$$

ここで $s = (W_\downarrow, W_\uparrow, \text{bias})$（1つのSecretGatingAdapterの秘密パラメータ）、$\delta$ は許容誤差である。

**ゲーティング構造による追加困難性**: 旧方式（加算型）では $y = x + \alpha \cdot f(x)$ から $f(x) = (y - x) / \alpha$ で秘密変換の出力が直接観測可能だった。ゲーティング方式では $y = x \odot \sigma(\text{bias} + \alpha \cdot h(x))$ であり、$h(x)$ を分離するには：

$$h(x) = \frac{\sigma^{-1}(y / x) - \text{bias}}{\alpha}$$

$x$ のゼロ要素で未定義となり、$\sigma^{-1}$（logit関数）を経由する非線形逆問題に帰着される。

**非線形性の重要性**: 非線形活性化 $\sigma$ がなければ $h(x) = x W_\downarrow W_\uparrow$ は線形であり、十分な $(x, h(x))$ ペアから $O(d^3)$ で復元可能。GELU等の非線形性があると非線形連立方程式の求解に帰着され、一般には計算量的に困難。ただし低ランク ($r = 16$, $d = 128$) は探索空間を制限する要因でもある。

### 11.4 セッション非連結性 (Session Unlinkability)

異なるセッション間で同一ユーザの推論パターンを紐付けることの困難性を定義する。

**定義 (UNL-Session).** 秘匿推論スキーム $\Pi$ がセッション非連結性を持つとは、任意のPPT攻撃者 $\mathcal{A}$ に対して：

$$\text{Adv}^{\text{unl}}[\mathcal{A}, \Pi] = \left| \Pr[\text{Game}^{\text{unl}}_0(\mathcal{A}) = 1] - \Pr[\text{Game}^{\text{unl}}_1(\mathcal{A}) = 1] \right| \leq \text{negl}(\lambda)$$

**ゲーム $\text{Game}^{\text{unl}}_b$:**

1. チャレンジャーが $k_0, k_1 \overset{R}{\leftarrow} \{0,1\}^\lambda$ を生成する
2. $\mathcal{A}$ が公開パラメータ $\theta$ を受け取る
3. $\mathcal{A}$ が入力 $x$ を選ぶ
4. チャレンジャーが3つのセッション秘密を生成する：
   - $s_1 \leftarrow \text{KDF}(k_0, \text{sid}_1)$
   - $b=0$ のとき: $s_2 \leftarrow \text{KDF}(k_0, \text{sid}_2)$（同一マスター鍵）
   - $b=1$ のとき: $s_2 \leftarrow \text{KDF}(k_1, \text{sid}_2)$（異なるマスター鍵）
5. チャレンジャーが $(h_1, h_2) = (\mathcal{F}_{\theta,s_1}^{(\ell)}(x), \mathcal{F}_{\theta,s_2}^{(\ell)}(x))$ を計算して渡す
6. $\mathcal{A}$ がビット $b'$ を出力する

**直観**: 攻撃者は2つのセッションの隠れ状態を観測しても、それらが同一マスター鍵から派生したものか別のマスター鍵から派生したものかを区別できない。これはHKDFの擬似乱数性から直接導かれる。

**帰着**:

$$\text{Adv}^{\text{unl}}[\mathcal{A}, \Pi] \leq 2 \cdot \text{Adv}^{\text{prf}}[\mathcal{B}, \text{HKDF}]$$

### 11.5 鍵復元困難性 (Key Recovery Hardness)

**定義 (KR-Hard).** 秘匿推論スキーム $\Pi$ が鍵復元困難性を持つとは、任意のPPT攻撃者 $\mathcal{A}$ に対して：

$$\Pr\left[ k' = k \;\middle|\; k' \leftarrow \mathcal{A}\left(\theta, \{(x_i, \mathcal{F}_{\theta,s}(x_i))\}_{i=1}^{n}\right) \right] \leq \text{negl}(\lambda)$$

ここで $s \leftarrow \text{KDF}(k, \text{sid})$、$n = \text{poly}(\lambda)$。

HMAC-SHA256のPRF安全性を仮定すると、鍵復元の困難性は以下のチェーンで帰着される：

```
マスター鍵 k
  ↓  HMAC-SHA256 (PRF仮定)
PRK (擬似乱数的)
  ↓  HKDF-Expand (PRF仮定)
セッション鍵 (擬似乱数的)
  ↓  HKDF-Expand (PRF仮定)
コンポーネントシード (擬似乱数的)
  ↓  torch.Generator (PRNG)
秘密テンソル (擬似乱数的)
```

各段の安全性は前段の出力の擬似乱数性にのみ依存し、鍵導出の一方向性が保証される。

### 11.6 安全性の限界と未証明領域

上記の定義は理想的な条件下での形式的フレームワークであり、現時点で **完全な安全性証明は存在しない**。以下の点が未解決である：

1. **残差接続のリーク**: 残差接続により埋め込みベクトル（公開情報）が秘密変換をバイパスして伝搬する。SS-Hidden定義における $h_b$ が残差成分を含む場合、意味的秘匿性は大幅に弱まる。本実装では秘密成分が全体の約0.5%にとどまる。

2. **低ランク制約による攻撃面**: SecretGatingAdapterの秘密空間は $r \times d + d \times r + d = 2rd + d$ 次元（$r=16, d=128$ で4,224パラメータ/Adapter）であり、十分な入出力ペアから統計的に推定可能な可能性がある。SR-Adapterの $\epsilon$ の具体的上界は未導出。

3. **softmaxの非線形性を経由する情報漏洩**: IND-Attn定義では直交行列の作用がsoftmaxを経由するため、一様ランダムからの乖離（$\epsilon_{\text{QR}}$）の精密な評価が必要である。特に低次元（$d_{\text{head}}=32$）ではHaar測度からの乖離が無視できない可能性がある。

4. **適応的攻撃**: 攻撃者が入力を自由に選べる場合（CPA/CCA相当）、特殊な入力パターンを用いた線形プロービング攻撃等が有効となりうる。この攻撃モデルでの安全性は未分析。

---

## 12. 現在の制約と今後の課題

### 制約

1. **安全性が未証明**: 第11節で形式的定義を与えたが、完全な安全性証明は未達成（特に残差リーク・低ランク制約・適応的攻撃に関して）
2. **残差接続による希釈**: 秘密成分の寄与がembeddingの残差に対して小さく、層出力レベルでの秘匿効果は残差路の存在に制約される（ただしゲーティング方式により、秘密なし時のサブレイヤー出力は遮断される）
3. **再学習の必要性**: セッション切り替えには adapt_to_secret() による再適応が必要（300ステップ程度）。完全なゼロショット切り替えは未達成
4. **文字レベルTokenizer**: PoC用の簡易実装であり、実用的な言語モデリング性能は限定的

### 今後の課題

1. **安全性評価**: 線形プロービング攻撃、Attention Map逆解析、統計的復元攻撃のベンチマーク
2. **複数秘密での学習 (Strategy B)**: 複数の秘密を同時に使って学習し、特定の秘密への過適合を防ぐ
3. **Adapterスケールの動的制御**: 層ごと・学習段階ごとにscale係数を調整する
4. **より大規模なモデルでの検証**: GPT-2クラスのモデルでの実証
5. **安全性証明の完成**: 第11節の形式的定義に基づく証明の完成、特にAdv上界の具体的導出
6. **適応ステップ数の最適化**: adapt_to_secret() の最適ステップ数・学習率の体系的な探索
7. **ゲートバイアスの適応的生成**: セッションごとのgate_bias値の最適分布の探索
