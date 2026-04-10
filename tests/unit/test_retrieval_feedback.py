from __future__ import annotations

from uuid import uuid4

import pytest

from mnemon.backends.memory_store import InMemoryDocumentStore, InMemoryVectorStore
from mnemon.core.config import MnemonConfig
from mnemon.core.models import Episode, RetrievalQuery
from mnemon.memory.episodic import EpisodicMemoryStore
from tests.unit.conftest import FakeEmbeddingProvider


@pytest.mark.asyncio
async def test_retrieve_prefers_helpful_memory_when_similarity_is_equal() -> None:
    config = MnemonConfig()
    store = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )

    first = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="deploy workflow",
        action="deploy workflow",
        outcome="deploy workflow",
        importance=0.5,
    )
    second = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="deploy workflow",
        action="deploy workflow",
        outcome="deploy workflow",
        importance=0.5,
        retrieval_uses=4,
        retrieval_help_count=4,
    )
    await store.encode(first)
    await store.encode(second)

    result = await store.retrieve(RetrievalQuery(query_text="deploy workflow", top_k=2))

    assert result.items[0].metadata["episode_id"] == str(second.id)


@pytest.mark.asyncio
async def test_retrieve_keeps_recent_memory_when_vector_similarity_is_negative() -> None:
    config = MnemonConfig()
    store = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )
    episode = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="fresh memory",
        action="fresh memory",
        outcome="fresh memory",
        importance=0.5,
        embedding=[1.0, 0.0],
    )
    await store.encode(episode)

    result = await store.retrieve(
        RetrievalQuery(
            query_text="opposite vector",
            query_embedding=[-1.0, 0.0],
            top_k=1,
            min_score=0.0,
        )
    )

    assert len(result.items) == 1


@pytest.mark.asyncio
async def test_helpful_feedback_strengthens_episode_importance() -> None:
    config = MnemonConfig()
    store = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )
    episode = Episode(
        agent_id="test",
        session_id=uuid4(),
        context="memory",
        action="action",
        outcome="outcome",
        importance=0.4,
    )
    await store.encode(episode)

    stored = await store.get(episode.id)
    assert stored is not None
    await store.update(
        episode.id,
        retrieval_uses=stored.retrieval_uses + 1,
        retrieval_help_count=stored.retrieval_help_count + 1,
        importance=min(1.0, stored.importance + 0.02),
    )

    updated = await store.get(episode.id)
    assert updated is not None
    assert updated.retrieval_uses == 1
    assert updated.retrieval_help_count == 1
    assert updated.importance == pytest.approx(0.42)
