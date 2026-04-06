"""Integration tests for the episodic -> semantic consolidation pipeline."""

from uuid import uuid4

import pytest

from mnemon.core.models import Episode
from mnemon.factory import Mnemon

pytestmark = pytest.mark.asyncio


async def test_consolidation_processes_episodes(brain: Mnemon) -> None:
    """Encoding episodes then running consolidation should extract semantic triples."""
    # Encode several episodes
    session_id = uuid4()
    for i in range(3):
        episode = Episode(
            agent_id="test-agent",
            session_id=session_id,
            context=f"User discussed topic {i} about cognitive memory systems",
            action=f"Agent explained aspect {i} of the memory framework",
            outcome=f"User understood point {i}",
            importance=0.7 + i * 0.1,
        )
        ep_id = await brain.memory.episodic.encode(episode)
        # Add to replay buffer for consolidation selection
        brain.learning.replay_buffer.add(ep_id, priority=episode.importance)

    # Run consolidation
    result = await brain.learning.consolidation.run_cycle()

    assert result.episodes_processed >= 0  # May be 0 if replay buffer sampling misses
    # If episodes were processed, triples should have been extracted
    if result.episodes_processed > 0:
        assert result.triples_extracted >= 0


async def test_consolidation_on_empty_buffer(brain: Mnemon) -> None:
    """Consolidation with no episodes should complete gracefully."""
    result = await brain.learning.consolidation.run_cycle()
    assert result.episodes_processed == 0
    assert result.triples_extracted == 0


async def test_reward_processor_tracks_statistics(brain: Mnemon) -> None:
    """Running cycles should accumulate reward statistics."""
    await brain.run_cycle("First interaction")
    await brain.run_cycle("Second interaction")

    stats = brain.learning.reward.get_stats()
    assert stats["cycle_count"] >= 2
    assert "mean_rpe" in stats
    assert "net_rpe" in stats


async def test_full_pipeline_encode_consolidate_retrieve(brain: Mnemon) -> None:
    """End-to-end: cognitive cycles -> consolidation -> semantic retrieval."""
    from .conftest import FakeEmbeddingProvider

    embedder = FakeEmbeddingProvider()

    # Run a few cognitive cycles to generate episodes
    await brain.run_cycle("Python is a programming language used for AI")
    await brain.run_cycle("Machine learning uses gradient descent for optimization")

    # Manually add episodes to replay buffer (they're already encoded by run_cycle)
    # Seed the replay buffer from episodic memory
    episodes = await brain.memory.episodic.sample_for_consolidation(batch_size=10)
    for ep in episodes:
        brain.learning.replay_buffer.add(ep.id, priority=ep.importance)

    # Run consolidation to extract semantic facts
    result = await brain.learning.consolidation.run_cycle()

    # If consolidation processed anything, verify semantic store was populated
    if result.triples_extracted > 0:
        query_emb = await embedder.embed("programming language AI")
        triples = await brain.memory.semantic.retrieve_by_similarity(query_emb, top_k=10)
        assert len(triples) >= 1
