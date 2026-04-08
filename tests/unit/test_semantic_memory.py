"""Regression tests for semantic vector/document consistency repair."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.models import EntityRef, SemanticTriple
from mnemon.memory.semantic import SemanticMemoryStore
from tests.unit.conftest import FakeEmbeddingProvider, FakeLLMProvider

pytestmark = pytest.mark.asyncio


def _make_store(config):
    vector_store = InMemoryVectorStore(config)
    document_store = InMemoryDocumentStore(config)
    store = SemanticMemoryStore(
        config=config.semantic,
        graph_store=InMemoryGraphStore(config),
        vector_store=vector_store,
        document_store=document_store,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLLMProvider(),
    )
    return store, vector_store, document_store


async def test_similarity_search_clears_orphaned_vectors(config) -> None:
    store, vector_store, _document_store = _make_store(config)
    orphan_id = uuid4()
    embedding = await store._embedder.embed("orphan triple")
    await vector_store.insert(
        orphan_id,
        embedding,
        {"_type": "triple", "triple_id": str(orphan_id)},
    )

    results = await store.retrieve_by_similarity(embedding, top_k=5)

    assert results == []
    assert await vector_store.count() == 0


async def test_similarity_search_rebuilds_missing_vectors_from_documents(config) -> None:
    store, vector_store, document_store = _make_store(config)
    triple = SemanticTriple(
        subject=EntityRef(entity_id=uuid4(), name="Python"),
        predicate="is_used_for",
        object="AI",
        confidence=0.9,
        embedding=await store._embedder.embed("Python is_used_for AI"),
    )
    doc = triple.model_dump(mode="json")
    doc["_type"] = "triple"
    await document_store.put(triple.id, doc)

    results = await store.retrieve_by_similarity(triple.embedding or [], top_k=5)

    assert len(results) == 1
    assert results[0].id == triple.id
    assert await vector_store.count() == 1
