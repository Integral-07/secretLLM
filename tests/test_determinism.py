"""決定論性・セッションローテーションのテスト。

検証項目:
  - 完全パイプラインの決定論性
  - セッション間の内部表現の乖離
"""
import torch
import torch.nn.functional as F
import pytest
from secret_llm.model.config import TransformerConfig
from secret_llm.model.transformer import SecretTransformer
from secret_llm.model.tokenizer import CharTokenizer
from secret_llm.crypto.key_manager import KeyManager
from secret_llm.pipeline.inference import SecretInferencePipeline


def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


@pytest.fixture
def setup():
    config = TransformerConfig()
    torch.manual_seed(42)
    model = SecretTransformer(config)
    tokenizer = CharTokenizer(config.vocab_size)
    km = KeyManager(b"\x01" * 32)
    pipeline = SecretInferencePipeline(model, km, tokenizer)
    return model, km, tokenizer, pipeline


class TestDeterminism:
    def test_full_pipeline_deterministic(self, setup):
        """鍵導出→重み生成→forward が完全に決定論的。"""
        model, km, tokenizer, pipeline = setup

        pipeline.start_session("det-test")
        reps1 = pipeline.get_internal_representations("hello")
        pipeline.end_session()

        pipeline.start_session("det-test")
        reps2 = pipeline.get_internal_representations("hello")
        pipeline.end_session()

        for key in reps1:
            assert torch.equal(reps1[key], reps2[key])

    def test_session_rotation_changes_representations(self, setup):
        """異なるセッション → 内部表現が異なる。"""
        model, km, tokenizer, pipeline = setup
        text = "test input"

        # セッションA
        pipeline.start_session("sess-A")
        reps_a = pipeline.get_internal_representations(text)
        pipeline.end_session()

        # セッションB
        pipeline.start_session("sess-B")
        reps_b = pipeline.get_internal_representations(text)
        pipeline.end_session()

        # 異なるセッション → 異なる内部表現
        for key in reps_a:
            assert not torch.equal(reps_a[key], reps_b[key]), \
                f"{key}: identical representations for different sessions"

    def test_wrong_secret_different_logits(self, setup):
        """正しい秘密と誤った秘密でlogitsが異なる。"""
        model, km, tokenizer, pipeline = setup
        ids = torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.long)
        mask = torch.tril(torch.ones(5, 5)).unsqueeze(0).unsqueeze(0)

        model.inject_secrets(km, "correct")
        with torch.no_grad():
            logits_correct = model(ids, mask=mask).detach()

        model.inject_secrets(km, "wrong")
        with torch.no_grad():
            logits_wrong = model(ids, mask=mask).detach()

        assert not torch.allclose(logits_correct, logits_wrong, atol=1e-3)
