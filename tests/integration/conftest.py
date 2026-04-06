"""
Integration test fixtures for the Mnemon cognitive framework.

Provides a fully wired ``Mnemon`` instance using in-memory backends and
fake LLM/embedding providers — no API keys or network access required.
"""

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
from mnemon.core.bus import CognitiveBus
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import EmbeddingProvider, LLMProvider
from mnemon.factory import (
    Mnemon,
    _ControlNamespace,
    _LearningNamespace,
    _MemoryNamespace,
)
from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.learning.reward import RewardProcessor
from mnemon.learning.skill_acquirer import SkillAcquirer
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.procedural import ProceduralMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.sensory import SensoryBuffer
from mnemon.memory.valence import ValenceMemoryStore
from mnemon.memory.working import WorkingMemoryManager

# ---------------------------------------------------------------------------
# Fake providers (enhanced for integration testing)
# ---------------------------------------------------------------------------

_EMBED_DIM = 8


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings for integration tests."""

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


class IntegrationFakeLLMProvider(LLMProvider):
    """Fake LLM that returns schema-aware canned responses.

    Unlike the unit-test fake (which returns ``{}`` for structured output),
    this implementation returns valid data matching expected schemas so
    consolidation, goal decomposition, and metacognition pipelines work.
    """

    CANNED_SUMMARY = "Summary: key facts preserved."

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.CANNED_SUMMARY

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        required = response_schema.get("required", [])

        # Consolidation: extract triples from episodes
        if "triples" in required:
            return {
                "triples": [
                    {
                        "subject": "user",
                        "predicate": "discussed",
                        "object": "cognitive memory",
                        "confidence": 0.85,
                    },
                ]
            }

        # Goal decomposition
        if "subgoals" in required:
            return {
                "subgoals": [
                    {
                        "description": "Gather information",
                        "priority": 0.8,
                        "success_criteria": "Information collected",
                    },
                    {
                        "description": "Synthesize findings",
                        "priority": 0.7,
                        "success_criteria": "Summary produced",
                    },
                ]
            }

        # Metacognition reflexion
        if "assessment" in required or "lessons" in required:
            return {
                "assessment": "Cycle proceeded normally.",
                "lessons": ["Retrieval was effective."],
                "strategy_recommended": None,
            }

        # Skill detection
        if "skills" in required or "needs" in required:
            return {"needs": [], "skills": []}

        # Default: return empty object with required keys set to defaults
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


# ---------------------------------------------------------------------------
# Brain assembly
# ---------------------------------------------------------------------------


async def build_test_brain(config: MnemonConfig | None = None) -> Mnemon:
    """Assemble a fully wired Mnemon with in-memory backends and fake providers.

    Mirrors the 9-stage pipeline in MnemonFactory.build() but injects fakes
    so no API keys or network access are needed.
    """
    cfg = config or MnemonConfig()
    llm = IntegrationFakeLLMProvider()
    embedder = FakeEmbeddingProvider()

    # Backends
    vector_store = InMemoryVectorStore(cfg)
    document_store = InMemoryDocumentStore(cfg)
    graph_store = InMemoryGraphStore(cfg)

    # Memory stores
    sensory = SensoryBuffer(config=cfg.sensory)
    working = WorkingMemoryManager(config=cfg.working_memory, llm=llm)
    episodic = EpisodicMemoryStore(
        config=cfg.episodic,
        vector_store=vector_store,
        document_store=document_store,
        embedding_provider=embedder,
    )
    semantic = SemanticMemoryStore(
        config=cfg.semantic,
        graph_store=graph_store,
        vector_store=vector_store,
        document_store=document_store,
        embedding_provider=embedder,
        llm_provider=llm,
    )
    procedural = ProceduralMemoryStore(
        config=cfg.procedural,
        vector_store=vector_store,
        document_store=document_store,
        embedding_provider=embedder,
    )
    valence = ValenceMemoryStore(config=cfg.valence, embedding_provider=embedder)

    # Learning
    replay_buffer = PrioritizedReplayBuffer(
        capacity=cfg.episodic.capacity.max_episodes,
        alpha=cfg.consolidation.replay.alpha,
        beta_start=cfg.consolidation.replay.beta_start,
    )
    reward_processor = RewardProcessor(config=cfg.reward)
    consolidation = ConsolidationEngine(
        config=cfg.consolidation,
        episodic_memory=episodic,
        semantic_memory=semantic,
        llm=llm,
        embedding_provider=embedder,
        replay_buffer=replay_buffer,
    )
    skill_acquirer = SkillAcquirer(
        config=cfg.procedural,
        procedural_memory=procedural,
        episodic_memory=episodic,
        llm=llm,
        embedding_provider=embedder,
    )

    # Control
    attention = AttentionController(config=cfg.attention, valence=valence)
    goal_manager = GoalManager(llm=llm)
    meta_cognition = MetaCognitionController(config=cfg.meta_cognition, llm=llm)

    # Bus
    bus = CognitiveBus()

    # Orchestrator
    orchestrator = Orchestrator(
        config=cfg,
        sensory=sensory,
        working_memory=working,
        episodic=episodic,
        semantic=semantic,
        procedural=procedural,
        valence=valence,
        attention=attention,
        goal_manager=goal_manager,
        meta_cognition=meta_cognition,
        reward_processor=reward_processor,
        embedding_provider=embedder,
        bus=bus,
    )

    # Assemble facade
    memory_ns = _MemoryNamespace(
        sensory=sensory,
        working=working,
        episodic=episodic,
        semantic=semantic,
        procedural=procedural,
        valence=valence,
    )
    learning_ns = _LearningNamespace(
        consolidation=consolidation,
        reward=reward_processor,
        skill_acquirer=skill_acquirer,
        replay_buffer=replay_buffer,
    )
    control_ns = _ControlNamespace(
        attention=attention,
        goals=goal_manager,
        meta_cognition=meta_cognition,
    )

    return Mnemon(
        orchestrator=orchestrator,
        memory=memory_ns,
        learning=learning_ns,
        control=control_ns,
        bus=bus,
        config=cfg,
        _backends=[vector_store, document_store, graph_store],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def brain():
    """Yield a fully wired Mnemon instance inside its async context."""
    b = await build_test_brain()
    async with b:
        yield b
    await b.close()


@pytest.fixture
def fake_llm() -> IntegrationFakeLLMProvider:
    return IntegrationFakeLLMProvider()


@pytest.fixture
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()
