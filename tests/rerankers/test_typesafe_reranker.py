"""Tests for the TypeSafe (Jev) reranker against a mocked ``typesafe_sdk`` client."""

from types import SimpleNamespace

import pytest

from mem0.configs.rerankers.typesafe import TypeSafeRerankerConfig
from mem0.utils.factory import RerankerFactory

SCORES = {"alpha": 0.2, "beta": 0.9, "gamma": 0.5, "delta": 0.7, "epsilon": 0.1}


def _answer_by_memory_text(state, questions):
    """Answer each question with the probability assigned to the memory it asks about."""
    nouls = {name: SimpleNamespace(noul=SCORES[q["instructions"]["memory"]]) for name, q in questions.items()}
    return SimpleNamespace(nouls=nouls)


def _docs(*texts):
    return [{"id": text, "memory": text} for text in texts]


class TestTypeSafeReranker:
    def test_orders_by_relevance_probability(self, mock_typesafe):
        module, fake_client = mock_typesafe
        fake_client.system_one.side_effect = _answer_by_memory_text

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key"))
        result = reranker.rerank("query", _docs("alpha", "beta", "gamma"))

        assert [doc["id"] for doc in result] == ["beta", "gamma", "alpha"]
        assert [doc["rerank_score"] for doc in result] == [0.9, 0.5, 0.2]

    def test_sends_query_as_state_and_one_question_per_memory(self, mock_typesafe):
        module, fake_client = mock_typesafe
        fake_client.system_one.side_effect = _answer_by_memory_text

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key"))
        reranker.rerank("where does the user live?", _docs("alpha", "beta"))

        fake_client.system_one.assert_called_once()
        kwargs = fake_client.system_one.call_args.kwargs
        assert kwargs["state"] == {"query": "where does the user live?"}
        assert list(kwargs["questions"]) == ["memory_0", "memory_1"]
        assert kwargs["questions"]["memory_1"]["instructions"]["memory"] == "beta"
        assert set(kwargs["questions"]["memory_1"]["criteria"]) == {"true", "false"}

    def test_batches_requests_and_keeps_scores_aligned(self, mock_typesafe):
        module, fake_client = mock_typesafe
        fake_client.system_one.side_effect = _answer_by_memory_text

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key", batch_size=2))
        result = reranker.rerank("query", _docs("alpha", "beta", "gamma", "delta", "epsilon"))

        assert fake_client.system_one.call_count == 3
        assert [doc["id"] for doc in result] == ["beta", "delta", "gamma", "alpha", "epsilon"]

    def test_truncates_long_memories(self, mock_typesafe):
        module, fake_client = mock_typesafe
        fake_client.system_one.return_value = SimpleNamespace(nouls={"memory_0": SimpleNamespace(noul=0.5)})

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key", max_memory_chars=4))
        reranker.rerank("query", [{"memory": "abcdefgh"}])

        questions = fake_client.system_one.call_args.kwargs["questions"]
        assert questions["memory_0"]["instructions"]["memory"] == "abcd"

    def test_reads_text_and_content_fields(self, mock_typesafe):
        module, fake_client = mock_typesafe
        fake_client.system_one.side_effect = _answer_by_memory_text

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key"))
        result = reranker.rerank("query", [{"text": "alpha"}, {"content": "beta"}])

        assert [doc["rerank_score"] for doc in result] == [0.9, 0.2]

    def test_top_k_per_call_overrides_config(self, mock_typesafe):
        module, fake_client = mock_typesafe
        fake_client.system_one.side_effect = _answer_by_memory_text

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key", top_k=2))

        assert len(reranker.rerank("query", _docs("alpha", "beta", "gamma"))) == 2
        assert len(reranker.rerank("query", _docs("alpha", "beta", "gamma"), top_k=1)) == 1

    def test_empty_documents_skip_the_api(self, mock_typesafe):
        module, fake_client = mock_typesafe

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key"))

        assert reranker.rerank("query", []) == []
        fake_client.system_one.assert_not_called()

    def test_client_receives_model_base_url_and_timeout(self, mock_typesafe):
        module, _ = mock_typesafe

        module.TypeSafeReranker(
            TypeSafeRerankerConfig(api_key="test-key", model="jev-1.13", base_url="https://example.test", timeout=5.0)
        )

        module.TypeSafeClient.assert_called_once_with(
            api_key="test-key", model="jev-1.13", base_url="https://example.test", timeout=5.0
        )

    def test_api_key_from_environment(self, mock_typesafe, monkeypatch):
        module, _ = mock_typesafe
        monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")

        reranker = module.TypeSafeReranker(TypeSafeRerankerConfig())

        assert reranker.api_key == "env-key"

    def test_missing_api_key_is_rejected(self, mock_typesafe, monkeypatch):
        module, _ = mock_typesafe
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            module.TypeSafeReranker(TypeSafeRerankerConfig())

    def test_missing_sdk_raises_import_error(self, mock_typesafe, monkeypatch):
        module, _ = mock_typesafe
        monkeypatch.setattr(module, "TYPESAFE_AVAILABLE", False)

        with pytest.raises(ImportError, match="typesafe-sdk"):
            module.TypeSafeReranker(TypeSafeRerankerConfig(api_key="test-key"))

    def test_factory_creates_typesafe_reranker(self, mock_typesafe):
        module, _ = mock_typesafe

        reranker = RerankerFactory.create("typesafe", {"api_key": "test-key", "top_k": 3})

        assert isinstance(reranker, module.TypeSafeReranker)
        assert reranker.config.top_k == 3
