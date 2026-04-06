"""Integration tests for memory encode/retrieve roundtrips across all stores."""

from uuid import uuid4

import pytest

from mnemon.core.models import (
    EntityRef,
    Episode,
    Modality,
    PerceptUnit,
    RetrievalQuery,
    SemanticTriple,
    Skill,
    SkillType,
)
from mnemon.factory import Mnemon

from .conftest import FakeEmbeddingProvider

pytestmark = pytest.mark.asyncio


async def test_episodic_encode_retrieve(brain: Mnemon) -> None:
    """Encoding an episode then retrieving it should return results."""
    embedder = FakeEmbeddingProvider()

    episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="User asked about machine learning",
        action="Explained gradient descent",
        outcome="User understood the concept",
        importance=0.8,
    )
    ep_id = await brain.memory.episodic.encode(episode)
    assert ep_id is not None

    # Retrieve with a similar query
    query = RetrievalQuery(
        query_text="machine learning gradient descent",
        query_embedding=await embedder.embed("machine learning gradient descent"),
        top_k=5,
        min_score=0.0,
    )
    result = await brain.memory.episodic.retrieve(query)
    assert len(result.items) >= 1
    assert any("machine learning" in item.content.lower() or "gradient" in item.content.lower()
               for item in result.items)


async def test_episodic_get_by_id(brain: Mnemon) -> None:
    """Should be able to fetch an episode by its UUID."""
    episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="Test episode for direct fetch",
        action="No action",
        outcome="Stored successfully",
        importance=0.5,
    )
    ep_id = await brain.memory.episodic.encode(episode)
    fetched = await brain.memory.episodic.get(ep_id)

    assert fetched is not None
    assert fetched.context == "Test episode for direct fetch"


async def test_semantic_upsert_retrieve(brain: Mnemon) -> None:
    """Upserting semantic triples then retrieving should return results."""
    embedder = FakeEmbeddingProvider()

    # Pre-compute embeddings so triples are indexed in the vector store
    emb1 = await embedder.embed("Python is_a programming language")
    emb2 = await embedder.embed("Python used_for machine learning")

    triples = [
        SemanticTriple(
            subject=EntityRef(entity_id=uuid4(), name="Python"),
            predicate="is_a",
            object=EntityRef(entity_id=uuid4(), name="programming language"),
            confidence=0.95,
            source_episodes=[uuid4()],
            embedding=emb1,
        ),
        SemanticTriple(
            subject=EntityRef(entity_id=uuid4(), name="Python"),
            predicate="used_for",
            object=EntityRef(entity_id=uuid4(), name="machine learning"),
            confidence=0.90,
            source_episodes=[uuid4()],
            embedding=emb2,
        ),
    ]

    count = await brain.memory.semantic.upsert_triples(triples)
    assert count >= 1

    # Retrieve by similarity
    query_emb = await embedder.embed("Python programming")
    results = await brain.memory.semantic.retrieve_by_similarity(query_emb, top_k=5)
    assert len(results) >= 1


async def test_procedural_register_retrieve(brain: Mnemon) -> None:
    """Registering a skill then retrieving it should work."""
    embedder = FakeEmbeddingProvider()

    skill = Skill(
        name="summarize_text",
        description="Summarize a given text passage into key points",
        type=SkillType.PROMPT_TEMPLATE,
        definition="Given the text, extract the main ideas.",
        utility=0.7,
    )
    skill_id = await brain.memory.procedural.register(skill)
    assert skill_id is not None

    # Retrieve by situation embedding
    situation_emb = await embedder.embed("I need to summarize this document")
    results = await brain.memory.procedural.retrieve(situation_emb, top_k=5)
    assert len(results) >= 1
    assert any(s.name == "summarize_text" for s in results)


async def test_valence_update_appraise(brain: Mnemon) -> None:
    """Updating valence associations should affect subsequent appraisals."""
    # Create a positive association
    await brain.memory.valence.update(["good news"], 0.9)

    # Appraise a percept containing the trigger
    percept = PerceptUnit(
        modality=Modality.TEXT,
        raw_content="I have good news to share",
        normalized="i have good news to share",
        tokens=7,
        ttl_ms=30000,
    )
    salience = await brain.memory.valence.appraise(percept)
    assert salience.raw_salience > 0.0


async def test_working_memory_inject_and_state(brain: Mnemon) -> None:
    """Working memory should track injected context blocks."""
    from mnemon.core.models import ContextBlock, ContextSource

    block = ContextBlock(
        content="Important context about the user's preferences",
        token_count=8,
        source=ContextSource.USER_INPUT,
        importance=0.8,
    )
    await brain.memory.working.inject(block)

    state = brain.memory.working.get_state()
    assert state.token_used > 0
    assert len(state.active_context) >= 1
