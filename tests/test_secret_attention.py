"""SecretProjectionAttention のテスト。

検証項目:
  - S_q=S_k=I のとき標準Attentionと同一出力
  - 秘密射影で出力が変化する
  - 出力形状
  - Causal maskの動作
  - 勾配がpublic weightsに流れ、secret weightsには流れない
"""
import torch
import pytest
from secret_llm.model.config import TransformerConfig
from secret_llm.model.attention import MultiHeadAttention
from secret_llm.model.secret_attention import SecretProjectionAttention


@pytest.fixture
def config():
    return TransformerConfig()


@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(2, 10, 128)  # (B=2, T=10, d_model=128)


class TestSecretProjectionAttention:
    def test_identity_equals_standard(self, config, input_tensor):
        """S_q=S_k=I → 標準Attentionと同一出力。"""
        torch.manual_seed(0)
        std_attn = MultiHeadAttention(config)
        torch.manual_seed(0)
        sec_attn = SecretProjectionAttention(config)
        # 同じ初期重みを保証
        sec_attn.load_state_dict(std_attn.state_dict(), strict=False)

        # eval()でdropoutを無効化 (学習モードではdropoutが確率的で結果が変わる)
        std_attn.eval()
        sec_attn.eval()

        out_std = std_attn(input_tensor)
        out_sec = sec_attn(input_tensor)
        assert torch.allclose(out_std, out_sec, atol=1e-5)

    def test_secret_changes_output(self, config, input_tensor):
        """秘密射影注入で出力が変化する。"""
        sec_attn = SecretProjectionAttention(config)
        out_identity = sec_attn(input_tensor).detach()

        # ランダム直交行列を注入
        from secret_llm.crypto.weight_generator import WeightGenerator
        s_q = WeightGenerator.bytes_to_orthogonal(b"\x01" * 8, (4, 32, 32))
        s_k = WeightGenerator.bytes_to_orthogonal(b"\x02" * 8, (4, 32, 32))
        sec_attn.set_secret_projections(s_q, s_k)
        out_secret = sec_attn(input_tensor).detach()

        assert not torch.allclose(out_identity, out_secret, atol=1e-3)

    def test_output_shape(self, config, input_tensor):
        """出力形状が (B, T, d_model)。"""
        attn = SecretProjectionAttention(config)
        out = attn(input_tensor)
        assert out.shape == input_tensor.shape

    def test_causal_mask(self, config):
        """Causal mask: 未来のトークンに注意が向かない。"""
        attn = SecretProjectionAttention(config)
        x = torch.randn(1, 5, 128)
        mask = torch.tril(torch.ones(5, 5)).unsqueeze(0).unsqueeze(0)

        out_masked = attn(x, mask=mask)
        # マスク付きで先頭トークンの出力は、2番目以降のトークンの影響を受けない
        # → 入力の2番目以降を変えても先頭の出力は同じ
        x2 = x.clone()
        x2[0, 1:, :] = torch.randn(4, 128)
        out_masked2 = attn(x2, mask=mask)
        assert torch.allclose(out_masked[0, 0], out_masked2[0, 0], atol=1e-5)

    def test_gradient_flow(self, config, input_tensor):
        """勾配: public weights → grad あり、secret weights → grad なし。"""
        attn = SecretProjectionAttention(config)
        out = attn(input_tensor)
        loss = out.sum()
        loss.backward()

        # Public weights (W_q等) は勾配あり
        assert attn.W_q.weight.grad is not None

        # Secret weights (S_q, S_k) は勾配なし
        assert attn.S_q.grad is None
        assert attn.S_k.grad is None
