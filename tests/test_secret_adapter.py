"""SecretGatingAdapter のテスト。

検証項目:
  - 初期状態 (gate_bias=+5) → ほぼ恒等写像
  - 秘密重み注入で出力が変化
  - 出力形状
  - クリア後 (gate_bias=-5) → 信号遮断
  - 非線形性の効果
"""
import torch
import pytest
from secret_llm.model.secret_adapter import SecretAdapter, SecretGatingAdapter


@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(2, 10, 128)


def _make_weights(d_model=128, rank=16):
    """秘密重み (W_down, W_up, gate_bias) を生成する。"""
    return (
        torch.randn(d_model, rank) * 0.1,
        torch.randn(rank, d_model) * 0.1,
        torch.randn(d_model).abs() + 2.5,  # 正のゲートバイアス
    )


class TestSecretGatingAdapter:
    def test_init_is_near_identity(self, input_tensor):
        """初期状態 (gate_bias=+5) → gate ≈ 0.993 → ほぼ恒等写像。"""
        adapter = SecretAdapter(128, 16)
        out = adapter(input_tensor)
        # sigmoid(5) ≈ 0.9933, so output ≈ 0.993 * input
        expected_gate = torch.sigmoid(torch.tensor(5.0))
        assert torch.allclose(out, input_tensor * expected_gate, atol=1e-6)

    def test_secret_changes_output(self, input_tensor):
        """秘密重み注入で出力が変化する。"""
        adapter = SecretAdapter(128, 16)
        out_init = adapter(input_tensor).detach()

        torch.manual_seed(99)
        adapter.set_secret_weights(*_make_weights())
        out_secret = adapter(input_tensor).detach()

        assert not torch.allclose(out_init, out_secret, atol=1e-3)

    def test_output_shape(self, input_tensor):
        """出力形状 = 入力形状。"""
        adapter = SecretAdapter(128, 16)
        out = adapter(input_tensor)
        assert out.shape == input_tensor.shape

    def test_clear_kills_signal(self, input_tensor):
        """gate_bias=-5 → gate ≈ 0.007 → 信号がほぼ遮断される。"""
        adapter = SecretAdapter(128, 16)
        # clear状態をシミュレート
        adapter.set_secret_weights(
            torch.zeros(128, 16),
            torch.zeros(16, 128),
            torch.full((128,), SecretGatingAdapter.GATE_CLOSED),
        )
        out = adapter(input_tensor)
        # sigmoid(-5) ≈ 0.0067, 出力は入力の約0.7%
        ratio = out.abs().mean() / input_tensor.abs().mean()
        assert ratio < 0.01  # 1%未満に減衰

    def test_gating_modulation(self, input_tensor):
        """秘密注入後、ゲートが次元ごとに異なるパターンを生成する。"""
        adapter = SecretAdapter(128, 16)
        torch.manual_seed(99)
        adapter.set_secret_weights(*_make_weights())

        out = adapter(input_tensor)
        # ゲートは次元ごとに異なるはず
        diff = (out - input_tensor).abs()
        # 一部の次元は大きく変化し、一部は小さい
        assert diff.max() > diff.min()

    def test_different_nonlinearities(self, input_tensor):
        """gelu / silu / tanh で異なる出力。"""
        results = {}
        for nl in ["gelu", "silu", "tanh"]:
            adapter = SecretAdapter(128, 16, nonlinearity=nl)
            torch.manual_seed(55)
            adapter.set_secret_weights(*_make_weights())
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
        adapter.set_secret_weights(*_make_weights())

        assert not adapter.W_down.requires_grad
        assert not adapter.W_up.requires_grad
        assert not adapter.gate_bias.requires_grad

    def test_gate_values_bounded(self, input_tensor):
        """ゲート値は [0, 1] の範囲。"""
        adapter = SecretAdapter(128, 16)
        torch.manual_seed(42)
        adapter.set_secret_weights(*_make_weights())

        out = adapter(input_tensor)
        # output = input * gate, gate ∈ [0, 1]
        # output の絶対値は input の絶対値以下
        assert (out.abs() <= input_tensor.abs() + 1e-6).all()

    def test_alias(self):
        """SecretAdapter は SecretGatingAdapter のエイリアス。"""
        assert SecretAdapter is SecretGatingAdapter
