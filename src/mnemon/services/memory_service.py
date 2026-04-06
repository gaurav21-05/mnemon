"""High-level service API for exposing Mnemon memory to external agents."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import EmbeddingProvider
from mnemon.core.models import Episode, RetrievalQuery
from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.valence import ValenceMemoryStore
from mnemon.providers.litellm_provider import LiteLLMEmbeddingProvider, LiteLLMProvider


class MemoryService:
    """External-facing API over Mnemon memory modules.

    This layer keeps MCP/REST adapters thin while centralizing core behavior:
    write episodic memory, retrieve memory/facts, inspect state, and run
    consolidation cycles.
    """

    def __init__(
        self,
        *,
        episodic: EpisodicMemoryStore,
        semantic: SemanticMemoryStore,
        valence: ValenceMemoryStore,
        replay: PrioritizedReplayBuffer,
        embedder: EmbeddingProvider,
        consolidation: ConsolidationEngine | None,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.valence = valence
        self.replay = replay
        self.embedder = embedder
        self.consolidation = consolidation

    @classmethod
    async def create_default(
        cls,
        *,
        model: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        temperature: float = 0.2,
    ) -> "MemoryService":
        """Build an in-memory service with optional consolidation support.

        Consolidation requires an LLM model. If model is None, episodic +
        semantic retrieval works but consolidation is disabled.
        """
        config = MnemonConfig()
        llm = (
            LiteLLMProvider(model=model, temperature=temperature, max_tokens=1024)
            if model is not None
            else None
        )
        embedder = LiteLLMEmbeddingProvider(model=embedding_model, dimensions=embedding_dim)

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
        replay = PrioritizedReplayBuffer(capacity=10_000)

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

        return cls(
            episodic=episodic,
            semantic=semantic,
            valence=valence,
            replay=replay,
            embedder=embedder,
            consolidation=consolidation,
        )

    async def write_memory(
        self,
        *,
        content: str,
        agent_id: str = "mcp-agent",
        session_id: str | UUID | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        """Encode one episodic memory item and queue it for consolidation."""
        sid = UUID(str(session_id)) if session_id is not None else uuid4()
        imp = self._estimate_importance(content) if importance is None else max(0.0, min(1.0, importance))
        episode = Episode(
            agent_id=agent_id,
            session_id=sid,
            context=content,
            action="Stored via memory service",
            outcome="Available for future recall",
            tags=tags or [],
            importance=imp,
        )
        episode_id = await self.episodic.encode(episode)
        self.replay.add(episode_id, priority=imp)

        return {
            "episode_id": str(episode_id),
            "session_id": str(sid),
            "importance": imp,
        }

    async def retrieve_memory(
        self,
        *,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
    ) -> dict[str, Any]:
        """Retrieve episodic memories and semantic facts for a query."""
        retrieval = await self.episodic.retrieve(
            RetrievalQuery(query_text=query, top_k=top_k, min_score=min_score)
        )
        episodic_items = [
            {
                "content": item.content,
                "score": item.score,
                "metadata": item.metadata,
            }
            for item in retrieval.items
        ]

        query_embedding = await self.embedder.embed(query)
        triples = await self.semantic.retrieve_by_similarity(query_embedding, top_k=top_k)
        semantic_items = []
        for triple in triples:
            obj_name = triple.object.name if hasattr(triple.object, "name") else str(triple.object)
            semantic_items.append(
                {
                    "fact": f"{triple.subject.name} {triple.predicate} {obj_name}",
                    "confidence": triple.confidence,
                    "triple_id": str(triple.id),
                }
            )

        return {
            "query": query,
            "episodic": episodic_items,
            "semantic": semantic_items,
            "counts": {
                "episodic": len(episodic_items),
                "semantic": len(semantic_items),
            },
        }

    async def consolidate(self) -> dict[str, Any]:
        """Run one consolidation cycle if an LLM-backed engine is configured."""
        if self.consolidation is None:
            return {
                "status": "skipped",
                "reason": "Consolidation requires an LLM model",
            }
        result = await self.consolidation.run_cycle()
        return {
            "status": "ok",
            "episodes_processed": result.episodes_processed,
            "triples_extracted": result.triples_extracted,
            "entities_resolved": result.entities_resolved,
            "conflicts_detected": result.conflicts_detected,
            "duration_ms": result.duration_ms,
        }

    async def state(self) -> dict[str, Any]:
        """Return aggregate memory stats for monitoring and diagnostics."""
        ep_docs = await self.episodic._document_store.query(filters={}, limit=100_000)
        triple_docs = await self.semantic._docs.query(filters={"_type": "triple"}, limit=100_000)
        return {
            "episodic_memories": len(ep_docs),
            "semantic_facts": len(triple_docs),
            "replay_buffer_size": self.replay.size,
        }

    @staticmethod
    def _estimate_importance(text: str) -> float:
        importance = 0.3
        text_lower = text.lower()
        markers = [
            "my name",
            "i am",
            "i'm",
            "i work",
            "i live",
            "i like",
            "remember",
            "important",
        ]
        if any(marker in text_lower for marker in markers):
            importance += 0.2
        if "?" in text:
            importance += 0.1
        if len(text) > 120:
            importance += 0.1
        return min(1.0, importance)
