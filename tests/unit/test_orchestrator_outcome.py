"""Test that update_last_episode_outcome patches the episode."""
from __future__ import annotations

import math
from typing import Any

import pytest

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.control.attention import AttentionController
from mnemon.control.goals import GoalManager
from mnemon.control.metacognition import MetaCognitionController
from mnemon.control.orchestrator import Orchestrator
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import EmbeddingProvider, LLMProvider
from mnemon.learning.reward import RewardProcessor
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.procedural import ProceduralMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.sensory import SensoryBuffer
from mnemon.memory.valence import ValenceMemoryStore
from mnemon.memory.working import WorkingMemoryManager

pytestmark = pytest.mark.asyncio

_EMBED_DIM = 8


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return _EMBED_DIM

    @property
    def model_name(self) -> str:
        return "fake-embedding-v1"

    async def embed(self, text: str) -> list[float]:
        h = hash(text)
        return [math.sin(h * (i + 1) / 1_000) for i in range(_EMBED_DIM)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class FakeLLMProvider(LLMProvider):
    CANNED_SUMMARY = "Summary: key facts preserved."

    async def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: ARG002
        return self.CANNED_SUMMARY

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:  # noqa: ARG002
        required = response_schema.get("required", [])
        if "assessment" in required or "lessons" in required:
            return {
                "assessment": "Cycle proceeded normally.",
                "lessons": ["Retrieval was effective."],
                "strategy_recommended": None,
            }
        result: dict[str, Any] = {}
        props = response_schema.get("properties", {})
        for key in required:
            prop_type = props.get(key, {}).get("type", "string")
            if prop_type == "array":
                result[key] = []
            elif prop_type == "string":
                result[key] = ""
            elif prop_type == "number":
                result[key] = 0.0
            else:
                result[key] = None
        return result

    async def token_count(self, text: str) -> int:
        return max(1, len(text) // 4)


def _make_orchestrator(config: MnemonConfig) -> Orchestrator:
    llm = FakeLLMProvider()
    embedder = FakeEmbeddingProvider()

    vector_store = InMemoryVectorStore(config)
    document_store = InMemoryDocumentStore(config)
    graph_store = InMemoryGraphStore(config)

    ep_store = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=vector_store,
        document_store=document_store,
        embedding_provider=embedder,
    )
    semantic = SemanticMemoryStore(
        config=config.semantic,
        graph_store=graph_store,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
        llm_provider=llm,
    )
    procedural = ProceduralMemoryStore(
        config=config.procedural,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
    )
    valence = ValenceMemoryStore(config=config.valence, embedding_provider=embedder)
    attention = AttentionController(config=config.attention, valence=valence)
    goal_manager = GoalManager(llm=llm)
    meta_cognition = MetaCognitionController(config=config.meta_cognition, llm=llm)
    reward_processor = RewardProcessor(config=config.reward)

    return Orchestrator(
        config=config,
        sensory=SensoryBuffer(config=config.sensory),
        working_memory=WorkingMemoryManager(config=config.working_memory, llm=llm),
        episodic=ep_store,
        semantic=semantic,
        procedural=procedural,
        valence=valence,
        attention=attention,
        goal_manager=goal_manager,
        meta_cognition=meta_cognition,
        reward_processor=reward_processor,
        embedding_provider=embedder,
    )


async def test_update_last_episode_outcome_patches_stored_episode() -> None:
    config = MnemonConfig()
    orch = _make_orchestrator(config)

    await orch.run_cycle(raw_input="hello world")
    assert orch._last_episode_id is not None

    await orch.update_last_episode_outcome("I said hello back")

    ep = await orch._episodic.get(orch._last_episode_id)
    assert ep is not None
    assert ep.outcome == "I said hello back"


async def test_update_last_episode_outcome_noop_when_no_prior_cycle() -> None:
    config = MnemonConfig()
    orch = _make_orchestrator(config)
    # Should not raise — no cycle has run yet
    await orch.update_last_episode_outcome("some reply")


async def test_last_episode_id_updated_after_each_cycle() -> None:
    config = MnemonConfig()
    orch = _make_orchestrator(config)

    await orch.run_cycle(raw_input="first message")
    id1 = orch._last_episode_id

    await orch.run_cycle(raw_input="second message")
    id2 = orch._last_episode_id

    assert id1 is not None
    assert id2 is not None
    assert id1 != id2
