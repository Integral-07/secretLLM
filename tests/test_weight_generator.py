"""WeightGenerator のテスト。

検証項目:
  - テンソル生成の決定論性と形状
  - 直交行列の直交性 (Q^T Q ≈ I)
  - Adapter/射影の生成
"""
import torch
import pytest
from secret_llm.crypto.key_manager import KeyManager
from secret_llm.crypto.weight_generator import WeightGenerator


class TestBytesToTensor:
    def test_deterministic(self):
        """同じバイト列 → 同じテンソル。"""
        raw = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        t1 = WeightGenerator.bytes_to_tensor(raw, (16, 8))
        t2 = WeightGenerator.bytes_to_tensor(raw, (16, 8))
        assert torch.equal(t1, t2)

    def test_shape(self):
        """出力形状が指定通り。"""
        raw = b"\x00" * 8
        for shape in [(128, 16), (16, 128), (4, 32, 32)]:
            t = WeightGenerator.bytes_to_tensor(raw, shape)
            assert t.shape == shape

    def test_different_bytes_different_tensor(self):
        """異なるバイト列 → 異なるテンソル。"""
        t1 = WeightGenerator.bytes_to_tensor(b"\x01" * 8, (16, 8))
        t2 = WeightGenerator.bytes_to_tensor(b"\x02" * 8, (16, 8))
        assert not torch.equal(t1, t2)

    def test_no_nan_inf(self):
        """NaN/Infが含まれない。"""
        raw = b"\xff" * 8
        t = WeightGenerator.bytes_to_tensor(raw, (128, 128))
        assert not torch.isnan(t).any()
        assert not torch.isinf(t).any()


class TestBytesToOrthogonal:
    def test_orthogonality(self):
        """Q^T Q ≈ I (直交性)。"""
        raw = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        matrices = WeightGenerator.bytes_to_orthogonal(raw, (4, 32, 32))
        for i in range(4):
            q = matrices[i]
            eye = torch.eye(32)
            product = q.T @ q
            assert torch.allclose(product, eye, atol=1e-5)

    def test_deterministic(self):
        """同じバイト列 → 同じ直交行列。"""
        raw = b"\xab\xcd" * 4
        m1 = WeightGenerator.bytes_to_orthogonal(raw, (4, 32, 32))
        m2 = WeightGenerator.bytes_to_orthogonal(raw, (4, 32, 32))
        assert torch.equal(m1, m2)

    def test_shape(self):
        """出力形状が (n_heads, d_head, d_head)。"""
        raw = b"\x00" * 8
        m = WeightGenerator.bytes_to_orthogonal(raw, (4, 32, 32))
        assert m.shape == (4, 32, 32)

    def test_norm_preservation(self):
        """直交行列はベクトルのノルムを保存する: ||Qx|| = ||x||。"""
        raw = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        matrices = WeightGenerator.bytes_to_orthogonal(raw, (1, 32, 32))
        q = matrices[0]
        x = torch.randn(32)
        qx = q @ x
        assert torch.allclose(x.norm(), qx.norm(), atol=1e-5)


class TestGenerateAdapterWeights:
    def test_shapes(self):
        """3つの重み: W_down (d_model, rank), W_up (rank, d_model), gate_bias (d_model,)。"""
        km = KeyManager(b"\x01" * 32)
        s = km.derive_session("test")
        wg = WeightGenerator()
        w_down, w_up, gate_bias = wg.generate_adapter_weights(s, 0, "attn", 128, 16)
        assert w_down.shape == (128, 16)
        assert w_up.shape == (16, 128)
        assert gate_bias.shape == (128,)

    def test_gate_bias_positive(self):
        """gate_bias は正の値 (≥ 2.5)。"""
        km = KeyManager(b"\x01" * 32)
        s = km.derive_session("test")
        wg = WeightGenerator()
        _, _, gate_bias = wg.generate_adapter_weights(s, 0, "attn", 128, 16)
        assert (gate_bias >= 2.5).all()

    def test_deterministic(self):
        """同じ入力 → 同じ重み。"""
        km = KeyManager(b"\x01" * 32)
        wg = WeightGenerator()
        s1 = km.derive_session("test")
        s2 = km.derive_session("test")
        w1 = wg.generate_adapter_weights(s1, 0, "attn", 128, 16)
        w2 = wg.generate_adapter_weights(s2, 0, "attn", 128, 16)
        for i in range(3):
            assert torch.equal(w1[i], w2[i])


class TestGenerateSecretProjections:
    def test_shapes(self):
        """S_q, S_k: (n_heads, d_head, d_head)。"""
        km = KeyManager(b"\x01" * 32)
        s = km.derive_session("test")
        wg = WeightGenerator()
        s_q, s_k = wg.generate_secret_projections(s, 0, 4, 32)
        assert s_q.shape == (4, 32, 32)
        assert s_k.shape == (4, 32, 32)

    def test_projections_are_orthogonal(self):
        """生成されたS_q, S_kが直交行列である。"""
        km = KeyManager(b"\x01" * 32)
        s = km.derive_session("test")
        wg = WeightGenerator()
        s_q, s_k = wg.generate_secret_projections(s, 0, 4, 32)
        eye = torch.eye(32)
        for i in range(4):
            assert torch.allclose(s_q[i].T @ s_q[i], eye, atol=1e-5)
            assert torch.allclose(s_k[i].T @ s_k[i], eye, atol=1e-5)
