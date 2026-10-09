import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_llm():
    with patch("mem0.reranker.llm_reranker.LlmFactory") as mock_factory:
        mock_llm_instance = MagicMock()
        mock_factory.create.return_value = mock_llm_instance
        yield mock_factory, mock_llm_instance


@pytest.fixture
def mock_cohere(monkeypatch):
    """Provide a fake ``cohere`` module so CohereReranker imports/constructs."""
    fake_cohere = ModuleType("cohere")
    fake_client = MagicMock()
    fake_cohere.Client = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "cohere", fake_cohere)

    import mem0.reranker.cohere_reranker as cohere_reranker

    monkeypatch.setattr(cohere_reranker, "cohere", fake_cohere, raising=False)
    monkeypatch.setattr(cohere_reranker, "COHERE_AVAILABLE", True, raising=False)
    return cohere_reranker, fake_client


@pytest.fixture
def mock_zero_entropy(monkeypatch):
    """Provide a fake ``zeroentropy`` module so ZeroEntropyReranker imports."""
    fake_module = ModuleType("zeroentropy")
    fake_client = MagicMock()
    fake_module.ZeroEntropy = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "zeroentropy", fake_module)

    import mem0.reranker.zero_entropy_reranker as zero_entropy_reranker

    monkeypatch.setattr(zero_entropy_reranker, "ZeroEntropy", fake_module.ZeroEntropy, raising=False)
    monkeypatch.setattr(zero_entropy_reranker, "ZERO_ENTROPY_AVAILABLE", True, raising=False)
    return zero_entropy_reranker, fake_client


@pytest.fixture
def mock_typesafe(monkeypatch):
    """Provide a fake ``typesafe_sdk`` module so TypeSafeReranker imports/constructs."""
    fake_module = ModuleType("typesafe_sdk")
    fake_client = MagicMock()
    fake_module.TypeSafeClient = MagicMock(return_value=fake_client)
    fake_module.Noul = MagicMock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", fake_module)

    import mem0.reranker.typesafe_reranker as typesafe_reranker

    monkeypatch.setattr(typesafe_reranker, "TypeSafeClient", fake_module.TypeSafeClient, raising=False)
    monkeypatch.setattr(typesafe_reranker, "Noul", fake_module.Noul, raising=False)
    monkeypatch.setattr(typesafe_reranker, "TYPESAFE_AVAILABLE", True, raising=False)
    return typesafe_reranker, fake_client
