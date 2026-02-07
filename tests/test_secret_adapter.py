"""SecretAdapter のテスト。

検証項目:
  - ゼロ重み → 恒等写像 (パススルー)
  - 秘密重み注入で出力が変化
  - 出力形状
  - 非線形性の効果
"""
import torch
import pytest
from secret_llm.model.secret_adapter import SecretAdapter


@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(2, 10, 128)


class TestSecretAdapter:
    def test_zero_is_identity(self, input_tensor):
        """W_down=W_up=0 → 出力=入力 (パススルー)。"""
        adapter = SecretAdapter(128, 16)
        out = adapter(input_tensor)
        assert torch.allclose(out, input_tensor, atol=1e-7)

    def test_secret_changes_output(self, input_tensor):
        """秘密重み注入で出力が変化する。"""
        adapter = SecretAdapter(128, 16)
        out_zero = adapter(input_tensor).detach()

        w_down = torch.randn(128, 16) * 0.1
        w_up = torch.randn(16, 128) * 0.1
        adapter.set_secret_weights(w_down, w_up)
        out_secret = adapter(input_tensor).detach()

        assert not torch.allclose(out_zero, out_secret, atol=1e-3)

    def test_output_shape(self, input_tensor):
        """出力形状 = 入力形状。"""
        adapter = SecretAdapter(128, 16)
        out = adapter(input_tensor)
        assert out.shape == input_tensor.shape

    def test_residual_connection(self, input_tensor):
        """出力 = 入力 + scale * adapter(入力)。差分がscaleに比例する。"""
        adapter = SecretAdapter(128, 16, scale=0.1)
        torch.manual_seed(99)
        w_down = torch.randn(128, 16) * 0.1
        w_up = torch.randn(16, 128) * 0.1
        adapter.set_secret_weights(w_down, w_up)

        out = adapter(input_tensor)
        diff = (out - input_tensor).abs().mean().item()
        # scale=0.1 かつ重みも小さいので差分は小さい
        assert diff < 1.0
        assert diff > 0.0

    def test_nonlinearity_effect(self, input_tensor):
        """非線形性なし(仮に線形)と異なる結果になる。"""
        adapter = SecretAdapter(128, 16, nonlinearity="gelu")
        w_down = torch.randn(128, 16)
        w_up = torch.randn(16, 128)
        adapter.set_secret_weights(w_down, w_up)
        out_gelu = adapter(input_tensor).detach()

        # 手動で線形版を計算
        h = input_tensor @ w_down  # GELU なし
        h = h @ w_up
        out_linear = input_tensor + 0.1 * h

        assert not torch.allclose(out_gelu, out_linear, atol=1e-3)

    def test_different_nonlinearities(self, input_tensor):
        """gelu / silu / tanh で異なる出力。"""
        results = {}
        for nl in ["gelu", "silu", "tanh"]:
            adapter = SecretAdapter(128, 16, nonlinearity=nl)
            w_down = torch.randn(128, 16)
            w_up = torch.randn(16, 128)
            adapter.set_secret_weights(w_down, w_up)
            results[nl] = adapter(input_tensor).detach()

        assert not torch.allclose(results["gelu"], results["silu"], atol=1e-3)
        assert not torch.allclose(results["gelu"], results["tanh"], atol=1e-3)

    def test_unknown_nonlinearity_raises(self):
        """未知の非線形性でValueError。"""
        with pytest.raises(ValueError):
            SecretAdapter(128, 16, nonlinearity="unknown")

    def test_no_gradient_on_secrets(self, input_tensor):
        """秘密重みはrequires_grad=Falseである。"""
        adapter = SecretAdapter(128, 16)
        w_down = torch.randn(128, 16)
        w_up = torch.randn(16, 128)
        adapter.set_secret_weights(w_down, w_up)

        assert not adapter.W_down.requires_grad
        assert not adapter.W_up.requires_grad
