"""High-level service API for exposing Mnemon memory to external agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.config import MnemonConfig
from mnemon.core.models import EntityRef, Episode, RetrievalQuery, SemanticTriple
from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.valence import ValenceMemoryStore
from mnemon.providers.litellm_provider import LiteLLMEmbeddingProvider, LiteLLMProvider

if TYPE_CHECKING:
    from mnemon.core.interfaces import EmbeddingProvider


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
    ) -> MemoryService:
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

    @staticmethod
    def _fact_text(triple: SemanticTriple) -> str:
        object_text = (
            triple.object.name if isinstance(triple.object, EntityRef) else str(triple.object)
        )
        return f"{triple.subject.name} {triple.predicate} {object_text}"

    async def _evidence_chain(self, source_episode_ids: list[UUID]) -> list[dict[str, Any]]:
        evidence_chain: list[dict[str, Any]] = []
        for episode_id in source_episode_ids:
            episode = await self.episodic.get(episode_id)
            if episode is None:
                continue
            evidence_chain.append(
                {
                    "episode_id": str(episode.id),
                    "timestamp": episode.timestamp.isoformat(),
                    "context": episode.context,
                    "action": episode.action,
                    "outcome": episode.outcome,
                    "goal_id": str(episode.goal_id) if episode.goal_id else None,
                    "workspace_path": episode.workspace_path,
                    "repo_name": episode.repo_name,
                    "citation": f"[memory:{episode.id}]",
                }
            )
        evidence_chain.sort(key=lambda item: str(item["timestamp"]))
        return evidence_chain

    async def write_memory(
        self,
        *,
        content: str,
        agent_id: str = "mcp-agent",
        session_id: str | UUID | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        scope_type: str = "personal",
        scope_id: str = "personal",
        workspace_path: str | None = None,
        repo_name: str | None = None,
    ) -> dict[str, Any]:
        """Encode one episodic memory item and queue it for consolidation."""
        sid = UUID(str(session_id)) if session_id is not None else uuid4()
        imp = (
            self._estimate_importance(content)
            if importance is None
            else max(0.0, min(1.0, importance))
        )
        episode = Episode(
            agent_id=agent_id,
            session_id=sid,
            context=content,
            action="Stored via memory service",
            outcome="Available for future recall",
            tags=tags or [],
            importance=imp,
            scope_type=scope_type,
            scope_id=scope_id,
            workspace_path=workspace_path,
            repo_name=repo_name,
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
            source_episode_ids = [str(item) for item in triple.source_episodes]
            semantic_items.append(
                {
                    "fact": self._fact_text(triple),
                    "confidence": triple.confidence,
                    "triple_id": str(triple.id),
                    "source_episode_ids": source_episode_ids,
                    "evidence_count": len(source_episode_ids),
                    "last_confirmed": triple.last_confirmed.isoformat(),
                    "current": triple.current,
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
    def _profile_slot(text: str) -> str:
        lowered = " ".join(text.lower().split())
        slot_prefixes = (
            ("i now use ", "uses_provider"),
            ("i use ", "uses_provider"),
            ("i prefer ", "preferences"),
            ("i like ", "preferences"),
            ("i love ", "preferences"),
            ("i work on ", "current_work"),
        )
        for prefix, slot in slot_prefixes:
            if lowered.startswith(prefix):
                return slot
        return lowered

    @staticmethod
    def _profile_fact(
        episode: Episode,
        *,
        current: bool,
        supersedes: list[str] | None = None,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        return {
            "text": episode.context,
            "memory_id": str(episode.id),
            "current": current,
            "supersedes": supersedes or [],
            "superseded_by": superseded_by,
            "timestamp": episode.timestamp.isoformat(),
        }

    async def _collect_profile_static(
        self,
        *,
        scope_type: str = "all",
        scope_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        filters: dict[str, Any] = {}
        if scope_type != "all":
            filters["scope_type"] = scope_type
            filters["scope_id"] = scope_id or scope_type

        all_docs = await self.episodic._document_store.query(filters={}, limit=10_000)
        profile_static: dict[str, list[Episode]] = {}
        for doc in all_docs:
            episode = Episode.model_validate(doc)
            if filters and (
                episode.scope_type != filters["scope_type"]
                or episode.scope_id != filters["scope_id"]
            ):
                continue
            if "profile_static" not in episode.tags:
                continue
            profile_static.setdefault(self._profile_slot(episode.context), []).append(episode)

        current_static: list[dict[str, Any]] = []
        historical_static: list[dict[str, Any]] = []
        for episodes in profile_static.values():
            ordered = sorted(episodes, key=lambda episode: episode.timestamp, reverse=True)
            current_episode = ordered[0]
            historical_episodes = ordered[1:]
            current_static.append(
                self._profile_fact(
                    current_episode,
                    current=True,
                    supersedes=[str(episode.id) for episode in historical_episodes],
                )
            )
            for historical_episode in historical_episodes:
                historical_static.append(
                    self._profile_fact(
                        historical_episode,
                        current=False,
                        superseded_by=str(current_episode.id),
                    )
                )

        current_static.sort(key=lambda item: str(item["timestamp"]), reverse=True)
        historical_static.sort(key=lambda item: str(item["timestamp"]), reverse=True)
        return current_static, historical_static

    async def profile_recall(
        self,
        *,
        query: str,
        top_k: int = 5,
        scope_type: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Return scoped recall results plus current and historical profile facts."""
        filters: dict[str, Any] = {}
        if scope_type != "all":
            filters["scope_type"] = scope_type
            filters["scope_id"] = scope_id or scope_type

        retrieval = await self.episodic.retrieve(
            RetrievalQuery(
                query_text=query,
                top_k=top_k,
                min_score=0.01,
                filters=filters,
            )
        )
        results = [
            {
                "content": item.content,
                "score": item.score,
                "metadata": item.metadata,
            }
            for item in retrieval.items
        ]

        current_static, historical_static = await self._collect_profile_static(
            scope_type=scope_type,
            scope_id=scope_id,
        )

        return {
            "query": query,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "results": results,
            "profile": {"static": current_static},
            "history": {"static": historical_static},
        }

    async def profile_snapshot(
        self,
        *,
        scope_type: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        current_static, historical_static = await self._collect_profile_static(
            scope_type=scope_type,
            scope_id=scope_id,
        )
        return {
            "scope_type": scope_type,
            "scope_id": scope_id,
            "profile": {"static": current_static},
            "history": {"static": historical_static},
        }

    async def recent_facts(self, *, limit: int = 10) -> dict[str, Any]:
        triple_docs = await self.semantic._docs.query(filters={"_type": "triple"}, limit=10_000)
        triples = [SemanticTriple.model_validate(doc) for doc in triple_docs]
        triples.sort(
            key=lambda triple: (triple.last_confirmed, triple.confidence, int(triple.current)),
            reverse=True,
        )
        items = [
            {
                "triple_id": str(triple.id),
                "fact": self._fact_text(triple),
                "confidence": triple.confidence,
                "current": triple.current,
                "last_confirmed": triple.last_confirmed.isoformat(),
                "source_episode_ids": [str(item) for item in triple.source_episodes],
            }
            for triple in triples[:limit]
        ]
        return {"count": len(items), "facts": items}

    async def causal_trace(
        self,
        *,
        episode_id: str | UUID | None = None,
        outcome_query: str | None = None,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        if episode_id is None and not (outcome_query and outcome_query.strip()):
            return {"error": "episode_id or outcome_query is required"}

        target_episode: Episode | None = None
        if episode_id is not None:
            target_episode = await self.episodic.get(UUID(str(episode_id)))
        elif outcome_query is not None:
            retrieval = await self.episodic.retrieve(
                RetrievalQuery(query_text=outcome_query, top_k=1, min_score=0.01)
            )
            if retrieval.items:
                retrieved_episode_id = retrieval.items[0].metadata.get("episode_id")
                if retrieved_episode_id:
                    target_episode = await self.episodic.get(UUID(str(retrieved_episode_id)))

        if target_episode is None:
            return {"error": "causal target episode not found"}

        chain: list[dict[str, Any]] = []
        visited: set[str] = set()
        current: Episode | None = target_episode
        depth = 0
        while current is not None and depth < max_depth and str(current.id) not in visited:
            visited.add(str(current.id))
            chain.append(
                {
                    "episode_id": str(current.id),
                    "timestamp": current.timestamp.isoformat(),
                    "context": current.context,
                    "action": current.action,
                    "outcome": current.outcome,
                    "caused_by": str(current.caused_by) if current.caused_by else None,
                    "led_to": [str(item) for item in current.led_to],
                    "citation": f"[memory:{current.id}]",
                }
            )
            if current.caused_by is None:
                break
            current = await self.episodic.get(current.caused_by)
            depth += 1

        chain.reverse()
        return {
            "target_episode_id": str(target_episode.id),
            "outcome_query": outcome_query,
            "chain_length": len(chain),
            "chain": chain,
        }

    async def explain_fact(self, triple_id: str | UUID) -> dict[str, Any]:
        """Return an evidence chain for one semantic fact."""
        triple_uuid = UUID(str(triple_id))
        doc = await self.semantic._docs.get(triple_uuid)
        if doc is None or doc.get("_type") != "triple":
            return {"error": f"Semantic fact {triple_uuid} not found"}

        triple = SemanticTriple.model_validate(doc)
        evidence_chain = await self._evidence_chain(triple.source_episodes)
        related_facts: list[dict[str, Any]] = []

        if triple.contradiction_group:
            all_docs = await self.semantic._docs.query(filters={"_type": "triple"}, limit=10_000)
            related_docs = [
                candidate
                for candidate in all_docs
                if candidate.get("contradiction_group") == triple.contradiction_group
                and str(candidate.get("id", "")) != str(triple.id)
            ]
            for related_doc in related_docs:
                related_triple = SemanticTriple.model_validate(related_doc)
                related_facts.append(
                    {
                        "triple_id": str(related_triple.id),
                        "fact": self._fact_text(related_triple),
                        "confidence": related_triple.confidence,
                        "current": related_triple.current,
                        "last_confirmed": related_triple.last_confirmed.isoformat(),
                    }
                )

        return {
            "triple_id": str(triple.id),
            "fact": self._fact_text(triple),
            "confidence": triple.confidence,
            "current": triple.current,
            "last_confirmed": triple.last_confirmed.isoformat(),
            "source_episode_ids": [str(item) for item in triple.source_episodes],
            "evidence_count": len(evidence_chain),
            "evidence_chain": evidence_chain,
            "related_facts": related_facts,
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
