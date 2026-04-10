"""
Shared fixtures for the Mnemon unit test suite.

Provides lightweight fakes for all external-dependency abstractions so that
every unit test runs without network calls, GPU, or real LLM credentials.
"""

from __future__ import annotations

from typing import Any

import pytest

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import EmbeddingProvider, LLMProvider

# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> MnemonConfig:
    """Default MnemonConfig built from built-in defaults (no env vars needed)."""
    return MnemonConfig()


# ---------------------------------------------------------------------------
# Backend store fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_store(config: MnemonConfig) -> InMemoryVectorStore:
    return InMemoryVectorStore(config)


@pytest.fixture
def document_store(config: MnemonConfig) -> InMemoryDocumentStore:
    return InMemoryDocumentStore(config)


@pytest.fixture
def graph_store(config: MnemonConfig) -> InMemoryGraphStore:
    return InMemoryGraphStore(config)


# ---------------------------------------------------------------------------
# Fake EmbeddingProvider
# ---------------------------------------------------------------------------

_EMBED_DIM = 8


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings of fixed dimension 8.

    The i-th component of the embedding for text *t* is:
        v[i] = sin(hash(t) * (i + 1) / 1000)

    This ensures:
    - Different texts produce different (but deterministic) vectors.
    - The same text always produces the same vector.
    - Vectors are not random so search-order assertions are reproducible.
    """

    @property
    def dimensions(self) -> int:
        return _EMBED_DIM

    @property
    def model_name(self) -> str:
        return "fake-embedding-v1"

    async def embed(self, text: str) -> list[float]:
        import math

        h = hash(text)
        return [math.sin(h * (i + 1) / 1_000) for i in range(_EMBED_DIM)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


# ---------------------------------------------------------------------------
# Fake LLMProvider
# ---------------------------------------------------------------------------


class FakeLLMProvider(LLMProvider):
    """Canned-response LLM provider.

    Returns a predictable string so summarisation logic can be tested
    without live API calls.  The canned summary is intentionally shorter
    than a typical input to trigger the "summary is smaller" branch in
    WorkingMemoryManager.
    """

    CANNED_SUMMARY = "Summary: key facts preserved."

    async def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: ARG002
        return self.CANNED_SUMMARY

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def token_count(self, text: str) -> int:
        return max(1, len(text) // 4)


@pytest.fixture
def fake_llm_provider() -> FakeLLMProvider:
    return FakeLLMProvider()
