from __future__ import annotations

import json

import pytest

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import EmbeddingProvider
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.valence import ValenceMemoryStore
from mnemon.services import (
    MemoryService,
    episodes_resource_uri,
    facts_resource_uri,
    known_resource_uris,
    profile_resource_uri,
    qualify_tool_name,
    read_resource,
    state_resource_uri,
)


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


def build_service() -> MemoryService:
    config = MnemonConfig()
    embedder = FakeEmbeddingProvider()

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
        llm_provider=None,
    )
    valence = ValenceMemoryStore(config=config.valence, embedding_provider=embedder)
    replay = PrioritizedReplayBuffer(capacity=128)

    return MemoryService(
        episodic=episodic,
        semantic=semantic,
        valence=valence,
        replay=replay,
        embedder=embedder,
        consolidation=None,
    )


def test_qualify_tool_name_uses_namespace_prefix() -> None:
    assert qualify_tool_name("mnemon", "memory_write") == "mnemon.memory_write"
    assert qualify_tool_name("team memory", "memory_state") == "team_memory.memory_state"


def test_known_resources_are_stable() -> None:
    uris = known_resource_uris("mnemon")
    assert uris == [
        state_resource_uri("mnemon"),
        episodes_resource_uri("mnemon"),
        facts_resource_uri("mnemon"),
        profile_resource_uri("mnemon"),
    ]


@pytest.mark.asyncio
async def test_read_state_resource_returns_json_text() -> None:
    service = build_service()
    await service.write_memory(content="Remember my favorite language is Python")

    uri = state_resource_uri("mnemon")
    payload = await read_resource(service, "mnemon", uri)
    assert payload["uri"] == uri
    parsed = json.loads(payload["text"])
    assert parsed["episodic_memories"] == 1


@pytest.mark.asyncio
async def test_read_recent_episodes_resource_returns_items() -> None:
    service = build_service()
    await service.write_memory(content="Remember my favorite language is Python")

    uri = episodes_resource_uri("mnemon")
    payload = await read_resource(service, "mnemon", uri)
    parsed = json.loads(payload["text"])
    assert parsed["count"] == 1
    assert len(parsed["episodes"]) == 1


@pytest.mark.asyncio
async def test_read_recent_facts_and_profile_resources_return_json() -> None:
    service = build_service()
    await service.write_memory(content="I now use Anthropic", tags=["profile_static"])

    facts_payload = await read_resource(service, "mnemon", facts_resource_uri("mnemon"))
    profile_payload = await read_resource(service, "mnemon", profile_resource_uri("mnemon"))

    facts = json.loads(facts_payload["text"])
    profile = json.loads(profile_payload["text"])
    assert "facts" in facts
    assert profile["profile"]["static"][0]["text"] == "I now use Anthropic"


@pytest.mark.asyncio
async def test_read_resource_rejects_unknown_uri() -> None:
    service = build_service()
    with pytest.raises(ValueError, match="Unknown resource URI"):
        await read_resource(service, "mnemon", "memory://mnemon/unknown")
