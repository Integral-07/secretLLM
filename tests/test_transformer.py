"""SecretTransformer のテスト。

検証項目:
  - forward pass の形状
  - inject_secrets / clear_secrets の動作
  - 異なるセッション → 異なる出力
  - 同一セッション → 同一出力 (決定論性)
"""
import torch
import pytest
from secret_llm.model.config import TransformerConfig
from secret_llm.model.transformer import SecretTransformer
from secret_llm.crypto.key_manager import KeyManager


@pytest.fixture
def config():
    return TransformerConfig()


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return SecretTransformer(config)


@pytest.fixture
def key_manager():
    return KeyManager(b"\x01" * 32)


@pytest.fixture
def token_ids():
    return torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)


@pytest.fixture
def mask():
    m = torch.tril(torch.ones(8, 8))
    return m.unsqueeze(0).unsqueeze(0)


class TestSecretTransformer:
    def test_forward_shape(self, model, config, token_ids, mask):
        """logits shape = (B, T, vocab_size)。"""
        logits = model(token_ids, mask=mask)
        assert logits.shape == (1, 8, config.vocab_size)

    def test_inject_changes_secret_params(self, model, key_manager):
        """inject_secrets で秘密パラメータが変化する。"""
        s_q_before = model.layers[0].attention.S_q.clone()
        model.inject_secrets(key_manager, "sess-1")
        s_q_after = model.layers[0].attention.S_q.clone()
        assert not torch.equal(s_q_before, s_q_after)

    def test_clear_resets_to_identity(self, model, key_manager):
        """clear_secrets で単位行列/ゼロに戻る。"""
        model.inject_secrets(key_manager, "sess-1")
        model.clear_secrets()

        eye = torch.eye(32).unsqueeze(0).expand(4, -1, -1)
        assert torch.allclose(model.layers[0].attention.S_q, eye)
        assert torch.allclose(model.layers[0].attention.S_k, eye)

        # SecretGatingAdapter: W_down, W_up がゼロ、gate_bias が GATE_CLOSED (-5)
        adapter = model.layers[0].adapter_attn
        assert torch.allclose(adapter.W_down, torch.zeros(128, 16))
        assert torch.allclose(adapter.W_up, torch.zeros(16, 128))
        assert torch.allclose(adapter.gate_bias, torch.full((128,), -5.0))

    def test_different_sessions_different_logits(self, model, key_manager, token_ids, mask):
        """異なるセッション → 異なるlogits。"""
        model.inject_secrets(key_manager, "sess-1")
        logits_1 = model(token_ids, mask=mask).detach()

        model.inject_secrets(key_manager, "sess-2")
        logits_2 = model(token_ids, mask=mask).detach()

        assert not torch.allclose(logits_1, logits_2, atol=1e-3)

    def test_same_session_deterministic(self, model, key_manager, token_ids, mask):
        """同一セッション → 同一logits。"""
        model.eval()  # dropoutを無効化して決定論的にする
        model.inject_secrets(key_manager, "sess-1")
        logits_1 = model(token_ids, mask=mask).detach()

        model.inject_secrets(key_manager, "sess-1")
        logits_2 = model(token_ids, mask=mask).detach()

        assert torch.equal(logits_1, logits_2)

    def test_count_parameters(self, model):
        """パラメータ数が公開/秘密に分かれている。"""
        pub, sec = model.count_parameters()
        assert pub > 0
        assert sec > 0
        assert sec < pub  # 秘密は公開より少ない
