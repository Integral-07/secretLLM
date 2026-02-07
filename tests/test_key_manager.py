"""KeyManager / SessionKeySet のテスト。

検証項目:
  - 決定論性: 同じ入力 → 同じ鍵
  - 分離性: 異なるセッション/マスター → 異なる鍵
  - バイト長の正確性
  - 大容量導出のチャンク分割
"""
import pytest
from secret_llm.crypto.key_manager import KeyManager, SessionKeySet


class TestKeyManager:
    def setup_method(self):
        self.master = b"\x01" * 32
        self.km = KeyManager(self.master)

    def test_deterministic_derivation(self):
        """同じmaster + 同じsession_id → 同一のSessionKeySet。"""
        km1 = KeyManager(self.master)
        km2 = KeyManager(self.master)
        s1 = km1.derive_session("sess-1")
        s2 = km2.derive_session("sess-1")
        b1 = s1.derive_component_bytes(0, "proj_q", 32)
        b2 = s2.derive_component_bytes(0, "proj_q", 32)
        assert b1 == b2

    def test_different_sessions_different_keys(self):
        """異なるsession_id → 異なる鍵。"""
        s1 = self.km.derive_session("sess-1")
        s2 = self.km.derive_session("sess-2")
        b1 = s1.derive_component_bytes(0, "proj_q", 32)
        b2 = s2.derive_component_bytes(0, "proj_q", 32)
        assert b1 != b2

    def test_different_masters_different_keys(self):
        """異なるmaster_secret → 異なる鍵。"""
        km1 = KeyManager(b"\x01" * 32)
        km2 = KeyManager(b"\x02" * 32)
        b1 = km1.derive_session("sess-1").derive_component_bytes(0, "proj_q", 32)
        b2 = km2.derive_session("sess-1").derive_component_bytes(0, "proj_q", 32)
        assert b1 != b2

    def test_different_components_different_keys(self):
        """同じ層・異なるコンポーネント → 異なる鍵。"""
        s = self.km.derive_session("sess-1")
        b1 = s.derive_component_bytes(0, "proj_q", 32)
        b2 = s.derive_component_bytes(0, "proj_k", 32)
        assert b1 != b2

    def test_different_layers_different_keys(self):
        """異なる層 → 異なる鍵。"""
        s = self.km.derive_session("sess-1")
        b1 = s.derive_component_bytes(0, "proj_q", 32)
        b2 = s.derive_component_bytes(1, "proj_q", 32)
        assert b1 != b2

    def test_key_length(self):
        """導出バイト列が指定長と一致する。"""
        s = self.km.derive_session("sess-1")
        for length in [8, 32, 64, 256, 1024]:
            b = s.derive_component_bytes(0, "proj_q", length)
            assert len(b) == length

    def test_generate_master_secret_length(self):
        """generate_master_secret は32バイトを返す。"""
        secret = KeyManager.generate_master_secret()
        assert len(secret) == 32

    def test_generate_master_secret_randomness(self):
        """2回のgenerate_master_secretは異なる値を返す。"""
        s1 = KeyManager.generate_master_secret()
        s2 = KeyManager.generate_master_secret()
        assert s1 != s2


class TestSessionKeySetLargeDerivation:
    def test_large_derivation(self):
        """8160バイト超の導出がチャンク分割で正常に動作する。"""
        km = KeyManager(b"\xab" * 32)
        s = km.derive_session("sess-large")
        b = s.derive_component_bytes(0, "test", 10000)
        assert len(b) == 10000

    def test_large_derivation_deterministic(self):
        """大容量導出も決定論的。"""
        km = KeyManager(b"\xab" * 32)
        s1 = km.derive_session("sess-large")
        s2 = km.derive_session("sess-large")
        b1 = s1.derive_component_bytes(0, "test", 10000)
        b2 = s2.derive_component_bytes(0, "test", 10000)
        assert b1 == b2
