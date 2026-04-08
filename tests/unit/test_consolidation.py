"""Regression tests for consolidation failure handling."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.models import ConsolidationState, Episode
from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from tests.unit.conftest import FakeEmbeddingProvider, FakeLLMProvider

pytestmark = pytest.mark.asyncio


class FailingExtractionLLM(FakeLLMProvider):
    async def generate_structured(self, prompt: str, response_schema: dict, **kwargs: object) -> dict:
        raise RuntimeError("Cannot connect to localhost:11434")


class EmptyTriplesLLM(FakeLLMProvider):
    async def generate_structured(self, prompt: str, response_schema: dict, **kwargs: object) -> dict:
        return {"triples": []}


def _make_engine(config, llm):
    embedder = FakeEmbeddingProvider()
    episodic = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
    )
    semantic = SemanticMemoryStore(
        config=config.semantic,
        graph_store=InMemoryGraphStore(config),
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
        llm_provider=llm,
    )
    replay = PrioritizedReplayBuffer(
        capacity=32,
        alpha=config.consolidation.replay.alpha,
        beta_start=config.consolidation.replay.beta_start,
    )
    engine = ConsolidationEngine(
        config=config.consolidation,
        episodic_memory=episodic,
        semantic_memory=semantic,
        llm=llm,
        embedding_provider=embedder,
        replay_buffer=replay,
    )
    return engine, episodic, replay


async def test_failed_extractions_mark_episode_failed_after_retry_limit(config) -> None:
    config.consolidation.max_extraction_retries = 2
    engine, episodic, replay = _make_engine(config, FailingExtractionLLM())

    episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="user asked about databases",
        action="explained how sqlite works",
        outcome="",
        importance=0.8,
    )
    await episodic.encode(episode)
    replay.add(episode.id, priority=episode.importance)

    first = await engine.run_cycle()
    assert first.episodes_processed == 0
    stored = await episodic.get(episode.id)
    assert stored is not None
    assert stored.consolidation_state == ConsolidationState.RAW
    assert stored.consolidation_attempts == 1

    replay.add(episode.id, priority=episode.importance)
    second = await engine.run_cycle()
    assert second.episodes_processed == 0
    stored = await episodic.get(episode.id)
    assert stored is not None
    assert stored.consolidation_state == ConsolidationState.FAILED
    assert stored.consolidation_attempts == 2


async def test_empty_extraction_marks_episode_consolidated(config) -> None:
    engine, episodic, replay = _make_engine(config, EmptyTriplesLLM())

    episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="brief pleasantry with no durable facts",
        action="said hello back",
        outcome="conversation ended",
        importance=0.4,
    )
    await episodic.encode(episode)
    replay.add(episode.id, priority=episode.importance)

    result = await engine.run_cycle()
    assert result.episodes_processed == 1
    assert result.triples_extracted == 0

    stored = await episodic.get(episode.id)
    assert stored is not None
    assert stored.consolidation_state == ConsolidationState.CONSOLIDATED
    assert stored.consolidation_attempts == 0
