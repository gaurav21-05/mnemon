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
from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.valence import ValenceMemoryStore
from mnemon.services import MemoryService


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        base = sum(ord(ch) for ch in text) % 97
        return [((base + i) % 17) / 17.0 for i in range(self._dimensions)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "fake-embedding"


class FakeLLMProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return "ok"

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "triples": [
                {
                    "subject": "Rohit",
                    "predicate": "works_on",
                    "object": "mnemon",
                    "confidence": 0.9,
                }
            ]
        }

    async def token_count(self, text: str) -> int:
        return len(text.split())


def build_service(with_consolidation: bool) -> MemoryService:
    config = MnemonConfig()
    embedder = FakeEmbeddingProvider()
    llm = FakeLLMProvider() if with_consolidation else None

    episodic_vs = InMemoryVectorStore(config)
    episodic_ds = InMemoryDocumentStore(config)
    semantic_vs = InMemoryVectorStore(config)
    semantic_ds = InMemoryDocumentStore(config)
    semantic_gs = InMemoryGraphStore(config)

    episodic = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=episodic_vs,
        document_store=episodic_ds,
        embedding_provider=embedder,
    )
    semantic = SemanticMemoryStore(
        config=config.semantic,
        graph_store=semantic_gs,
        vector_store=semantic_vs,
        document_store=semantic_ds,
        embedding_provider=embedder,
        llm_provider=llm,
    )
    valence = ValenceMemoryStore(config=config.valence, embedding_provider=embedder)
    replay = PrioritizedReplayBuffer(capacity=128)

    consolidation = None
    if llm is not None:
        consolidation = ConsolidationEngine(
            config=config.consolidation,
            episodic_memory=episodic,
            semantic_memory=semantic,
            llm=llm,
            embedding_provider=embedder,
            replay_buffer=replay,
        )

    return MemoryService(
        episodic=episodic,
        semantic=semantic,
        valence=valence,
        replay=replay,
        embedder=embedder,
        consolidation=consolidation,
    )


@pytest.mark.asyncio
async def test_write_and_retrieve_memory() -> None:
    service = build_service(with_consolidation=False)

    write_result = await service.write_memory(content="My name is Rohit and I work on mnemon")
    assert "episode_id" in write_result

    retrieval = await service.retrieve_memory(query="What does Rohit work on?", top_k=3)
    assert retrieval["counts"]["episodic"] >= 1
    assert any("Rohit" in item["content"] for item in retrieval["episodic"])


@pytest.mark.asyncio
async def test_state_reflects_written_memory() -> None:
    service = build_service(with_consolidation=False)
    await service.write_memory(content="Remember that project deadline is Friday")

    state = await service.state()
    assert state["episodic_memories"] == 1
    assert state["replay_buffer_size"] == 1


@pytest.mark.asyncio
async def test_consolidation_skipped_without_llm() -> None:
    service = build_service(with_consolidation=False)
    await service.write_memory(content="Rohit works on mnemon")

    result = await service.consolidate()
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_consolidation_extracts_semantic_facts_with_llm() -> None:
    service = build_service(with_consolidation=True)
    await service.write_memory(content="Rohit works on mnemon")

    result = await service.consolidate()
    assert result["status"] == "ok"
    assert result["episodes_processed"] >= 1

    state = await service.state()
    assert state["semantic_facts"] >= 1
