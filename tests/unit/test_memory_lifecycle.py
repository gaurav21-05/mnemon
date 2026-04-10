from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from mnemon.backends.memory_store import InMemoryDocumentStore, InMemoryVectorStore
from mnemon.core.config import MnemonConfig
from mnemon.core.models import (
    ConsolidationState,
    Episode,
    MemoryLifecycleState,
    RetrievalQuery,
)
from mnemon.memory.episodic import EpisodicMemoryStore
from tests.unit.conftest import FakeEmbeddingProvider


@pytest.mark.asyncio
async def test_new_episode_defaults_to_durable_lifecycle() -> None:
    config = MnemonConfig()
    episodic = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )
    episode = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="hello",
        action="said hello",
        outcome="stored",
    )
    await episodic.encode(episode)

    stored = await episodic.get(episode.id)
    assert stored is not None
    assert stored.lifecycle_state == MemoryLifecycleState.DURABLE


@pytest.mark.asyncio
async def test_mark_consolidated_updates_lifecycle_state() -> None:
    config = MnemonConfig()
    episodic = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )
    episode = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="hello",
        action="said hello",
        outcome="stored",
    )
    await episodic.encode(episode)
    await episodic.mark_consolidated([episode.id])

    stored = await episodic.get(episode.id)
    assert stored is not None
    assert stored.lifecycle_state == MemoryLifecycleState.CONSOLIDATED


@pytest.mark.asyncio
async def test_decay_sweep_archives_then_forgets_consolidated_episode() -> None:
    config = MnemonConfig()
    config.episodic.decay.forget_threshold = 0.99
    episodic = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )
    episode = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="old memory",
        action="old action",
        outcome="old outcome",
        consolidation_state=ConsolidationState.CONSOLIDATED,
        lifecycle_state=MemoryLifecycleState.CONSOLIDATED,
        last_accessed=datetime.now(UTC) - timedelta(days=30),
        decay_lambda=10.0,
    )
    await episodic.encode(episode)

    first = await episodic.run_decay_sweep()
    archived = await episodic.get(episode.id)
    assert first == 1
    assert archived is not None
    assert archived.lifecycle_state == MemoryLifecycleState.ARCHIVED

    second = await episodic.run_decay_sweep()
    forgotten = await episodic.get(episode.id)
    assert second == 1
    assert forgotten is not None
    assert forgotten.lifecycle_state == MemoryLifecycleState.FORGOTTEN


@pytest.mark.asyncio
async def test_retrieval_reinforces_episode_strength() -> None:
    config = MnemonConfig()
    episodic = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )
    episode = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="retrieve me",
        action="stored",
        outcome="ready",
        base_strength=1.0,
    )
    await episodic.encode(episode)

    await episodic.retrieve(
        RetrievalQuery(query_text="retrieve me stored ready", top_k=1, min_score=0.0)
    )

    stored = await episodic.get(episode.id)
    assert stored is not None
    assert stored.base_strength > 1.0
    assert stored.retrieval_uses == 1
