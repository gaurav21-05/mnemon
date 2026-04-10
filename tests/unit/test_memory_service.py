from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import EmbeddingProvider, LLMProvider
from mnemon.core.models import EntityRef, SemanticTriple
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


@pytest.mark.asyncio
async def test_retrieve_memory_surfaces_semantic_evidence_metadata() -> None:
    service = build_service(with_consolidation=False)
    write_result = await service.write_memory(content="Rohit works on mnemon")
    episode_id = UUID(write_result["episode_id"])

    triple = SemanticTriple(
        subject=EntityRef(entity_id=uuid4(), name="Rohit"),
        predicate="works_on",
        object="mnemon",
        confidence=0.9,
        source_episodes=[episode_id],
        embedding=await service.embedder.embed("Rohit works_on mnemon"),
    )
    await service.semantic.upsert_triples([triple])

    retrieval = await service.retrieve_memory(query="What does Rohit work on?", top_k=3)

    assert retrieval["semantic"][0]["fact"] == "Rohit works_on mnemon"
    assert retrieval["semantic"][0]["source_episode_ids"] == [str(episode_id)]
    assert retrieval["semantic"][0]["evidence_count"] == 1
    assert retrieval["semantic"][0]["current"] is True


@pytest.mark.asyncio
async def test_explain_fact_returns_evidence_chain() -> None:
    service = build_service(with_consolidation=False)
    write_result = await service.write_memory(content="Rohit works on mnemon")
    episode_id = UUID(write_result["episode_id"])

    triple = SemanticTriple(
        subject=EntityRef(entity_id=uuid4(), name="Rohit"),
        predicate="works_on",
        object="mnemon",
        confidence=0.9,
        source_episodes=[episode_id],
        embedding=await service.embedder.embed("Rohit works_on mnemon"),
    )
    await service.semantic.upsert_triples([triple])

    explanation = await service.explain_fact(triple.id)

    assert explanation["fact"] == "Rohit works_on mnemon"
    assert explanation["confidence"] == 0.9
    assert explanation["source_episode_ids"] == [str(episode_id)]
    assert explanation["evidence_chain"][0]["episode_id"] == str(episode_id)
    assert explanation["evidence_chain"][0]["context"] == "Rohit works on mnemon"


@pytest.mark.asyncio
async def test_causal_trace_returns_linked_episode_chain() -> None:
    service = build_service(with_consolidation=False)
    first = await service.write_memory(content="Started deployment work", tags=["deploy"])
    second = await service.write_memory(content="Deployment failed on first try", tags=["deploy"])

    await service.episodic.update(UUID(second["episode_id"]), caused_by=UUID(first["episode_id"]))
    await service.episodic.update(UUID(first["episode_id"]), led_to=[UUID(second["episode_id"])])

    trace = await service.causal_trace(episode_id=second["episode_id"])

    assert trace["target_episode_id"] == second["episode_id"]
    assert [item["episode_id"] for item in trace["chain"]] == [
        first["episode_id"],
        second["episode_id"],
    ]


@pytest.mark.asyncio
async def test_recent_facts_and_profile_snapshot_are_resource_ready() -> None:
    service = build_service(with_consolidation=False)
    write_result = await service.write_memory(
        content="I now use Anthropic", tags=["profile_static"]
    )
    episode_id = UUID(write_result["episode_id"])

    triple = SemanticTriple(
        subject=EntityRef(entity_id=uuid4(), name="Rohit"),
        predicate="uses_provider",
        object="Anthropic",
        confidence=0.9,
        source_episodes=[episode_id],
        embedding=await service.embedder.embed("Rohit uses_provider Anthropic"),
    )
    await service.semantic.upsert_triples([triple])

    facts = await service.recent_facts(limit=5)
    profile = await service.profile_snapshot()

    assert facts["count"] == 1
    assert facts["facts"][0]["fact"] == "Rohit uses_provider Anthropic"
    assert profile["profile"]["static"][0]["text"] == "I now use Anthropic"
