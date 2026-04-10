"""
Episodic memory module — hippocampal formation analog.

Brain analog: The hippocampus performs rapid, one-shot encoding of
autobiographical episodes (context → action → outcome) and supports
pattern-completion-based retrieval.  This module mirrors the three core
hippocampal functions:

  1. **Encoding** (CA3/DG): embeds the episode text into a dense vector and
     persists both the vector (for similarity search) and the full document
     (for faithful reconstruction).

  2. **Retrieval** (CA1): uses cue-based search combining semantic similarity,
     recency bias, and importance weighting — the same multi-factor scoring
     thought to govern hippocampal cue-driven recall.

  3. **Consolidation readiness** (CA1 → neocortex): surfaces raw episodes
     prioritised by importance × recency, ready for offline replay into
     semantic memory, mirroring slow-wave-sleep hippocampal replay.

Decay follows the Ebbinghaus forgetting curve: strength = base_strength ×
exp(-λ × Δt).  Episodes whose strength drops below the forget threshold
and have already been consolidated are eligible for deletion.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mnemon.core.exceptions import MemoryError, RetrievalError
from mnemon.core.interfaces import (
    DocumentStore,
    EmbeddingProvider,
    EpisodicMemoryInterface,
    VectorStore,
)
from mnemon.core.models import (
    ConsolidationState,
    Episode,
    MemoryLifecycleState,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
)

if TYPE_CHECKING:
    from uuid import UUID

    from mnemon.core.config import EpisodicConfig

logger = logging.getLogger(__name__)

_STORE_NAME = "episodic"


class EpisodicMemoryStore(EpisodicMemoryInterface):
    """Concrete episodic memory implementation backed by a VectorStore and DocumentStore.

    The store is intentionally backend-agnostic: any conforming VectorStore and
    DocumentStore implementation can be injected (in-memory for tests, Qdrant +
    PostgreSQL for production).

    Parameters
    ----------
    config:
        EpisodicConfig section from the root MnemonConfig.
    vector_store:
        Backend used for embedding-based similarity search.
    document_store:
        Backend used for full episode document persistence.
    embedding_provider:
        Provider used to compute embeddings on-demand when an episode or
        query arrives without a pre-computed embedding.
    """

    def __init__(
        self,
        config: EpisodicConfig,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._document_store = document_store
        self._embedding_provider = embedding_provider
        logger.info("EpisodicMemoryStore initialised backend=%s", config.backend)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hours_since(dt: datetime) -> float:
        """Return fractional hours elapsed since *dt* (UTC-aware)."""
        now = datetime.now(UTC)
        if dt.tzinfo is None:
            # Treat naive datetimes as UTC
            dt = dt.replace(tzinfo=UTC)
        delta = now - dt
        return delta.total_seconds() / 3600.0

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors.

        Brain analog: measures the angular distance between two activation
        patterns in the CA3 attractor space — vectors pointing in the same
        direction represent semantically similar episodes.

        Returns a value in [-1, 1]; returns 0.0 for zero-norm vectors to
        avoid division-by-zero on unembedded episodes.
        """
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _mmr_rerank(
        scored_items: list[tuple[float, Episode]],
        query_embedding: list[float],
        top_k: int,
        lambda_param: float = 0.7,
    ) -> list[tuple[float, Episode]]:
        """Re-rank candidates by Maximal Marginal Relevance (MMR).

        Brain analog: hippocampal pattern separation — the dentate gyrus
        actively decorrelates similar engrams so that retrieved memories cover
        distinct aspects of experience rather than repeating the same trace.

        MMR iteratively selects the candidate that maximises:

            MMR(d) = λ · sim(d, query) - (1-λ) · max_{d' ∈ S} sim(d, d')

        where S is the set of already-selected episodes.  λ=1 reduces to
        pure relevance ranking; λ=0 reduces to maximum diversity.

        Parameters
        ----------
        scored_items:
            Candidates as (hybrid_score, Episode) pairs, pre-sorted by score.
        query_embedding:
            Dense vector representing the retrieval cue.
        top_k:
            Number of episodes to select.
        lambda_param:
            Trade-off between relevance (1.0) and diversity (0.0).
            Defaults to 0.7 — favours relevance while still injecting diversity.

        Returns
        -------
        list[tuple[float, Episode]]
            Up to *top_k* episodes in MMR-selected order, retaining the
            original hybrid score for downstream consumers.
        """
        if not scored_items:
            return []

        # Normalise hybrid scores to [0, 1] so they are commensurable with
        # cosine similarities when computing the MMR objective.
        max_score = max(s for s, _ in scored_items)
        min_score = min(s for s, _ in scored_items)
        score_range = max_score - min_score or 1.0  # guard against constant scores

        def _norm_score(s: float) -> float:
            return (s - min_score) / score_range

        remaining = list(scored_items)
        selected: list[tuple[float, Episode]] = []

        while remaining and len(selected) < top_k:
            best_mmr: float | None = None
            best_idx: int = 0

            for idx, (score, episode) in enumerate(remaining):
                emb = episode.embedding

                # Relevance term: normalised hybrid score blended with
                # query-embedding cosine similarity when an embedding exists.
                if emb is not None:
                    rel = lambda_param * EpisodicMemoryStore._cosine_similarity(
                        emb, query_embedding
                    )
                else:
                    # No embedding available — fall back to normalised score.
                    rel = lambda_param * _norm_score(score)

                # Diversity penalty: maximum similarity to any already-selected
                # episode.  On the first iteration S is empty so penalty is 0.
                if selected and emb is not None:
                    max_sim = (
                        max(
                            EpisodicMemoryStore._cosine_similarity(emb, sel_ep.embedding)
                            for _, sel_ep in selected
                            if sel_ep.embedding is not None
                        )
                        if any(sel_ep.embedding is not None for _, sel_ep in selected)
                        else 0.0
                    )
                else:
                    max_sim = 0.0

                mmr_score = rel - (1.0 - lambda_param) * max_sim

                if best_mmr is None or mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def _episode_text(episode: Episode) -> str:
        """Concatenate the three narrative fields for embedding."""
        return f"{episode.context} {episode.action} {episode.outcome}"

    def _build_vector_metadata(self, episode: Episode) -> dict[str, Any]:
        """Extract the subset of episode fields stored alongside the vector."""
        return {
            "agent_id": episode.agent_id,
            "session_id": str(episode.session_id),
            "timestamp_iso": episode.timestamp.isoformat(),
            "importance": episode.importance,
            "consolidation_state": episode.consolidation_state,
            "tags": episode.tags,
            "scope_type": episode.scope_type,
            "scope_id": episode.scope_id,
            "workspace_path": episode.workspace_path,
            "repo_name": episode.repo_name,
            "caused_by": str(episode.caused_by) if episode.caused_by else None,
            "led_to": [str(item) for item in episode.led_to],
            "source_episode_ids": [str(item) for item in episode.source_episode_ids],
            "summary_kind": episode.summary_kind,
            "summary_of_count": episode.summary_of_count,
            "lifecycle_state": episode.lifecycle_state,
            "retrieval_uses": episode.retrieval_uses,
            "retrieval_help_count": episode.retrieval_help_count,
        }

    def _compute_hybrid_score(
        self,
        vector_score: float,
        episode: Episode,
    ) -> float:
        """Combine semantic similarity, recency, and importance into a single score.

        score = w_semantic * vector_score
              + w_recency  * exp(-λ * hours_since_last_access)
              + w_importance * episode.importance
        """
        weights = self._config.retrieval_weights
        decay_lambda = max(episode.decay_lambda, self._config.decay.base_lambda)
        hours = self._hours_since(episode.last_accessed)
        recency_score = math.exp(-decay_lambda * hours)
        help_ratio = (
            episode.retrieval_help_count / episode.retrieval_uses
            if episode.retrieval_uses > 0
            else 0.0
        )
        usefulness_bonus = min(
            0.12,
            0.04 * episode.retrieval_help_count + 0.08 * help_ratio,
        )

        semantic_score = max(0.0, vector_score)
        return (
            weights.semantic * semantic_score
            + weights.recency * recency_score
            + weights.importance * episode.importance
            + usefulness_bonus
        )

    # ------------------------------------------------------------------
    # EpisodicMemoryInterface implementation
    # ------------------------------------------------------------------

    async def encode(self, episode: Episode) -> UUID:
        """Embed and persist *episode* to both the vector store and document store.

        If *episode.embedding* is None the embedding is computed on the fly
        from the concatenated context/action/outcome text.

        Returns
        -------
        UUID
            The episode's stable identifier.

        Raises
        ------
        MemoryError
            If either the embedding step or a store write fails.
        """
        try:
            if episode.embedding is None:
                text = self._episode_text(episode)
                embedding = await self._embedding_provider.embed(text)
                # Rebuild episode with the computed embedding (Episode is a
                # Pydantic model — use model_copy for immutable-style update)
                episode = episode.model_copy(update={"embedding": embedding})
            else:
                embedding = episode.embedding

            metadata = self._build_vector_metadata(episode)
            await self._vector_store.insert(episode.id, embedding, metadata)
            await self._document_store.put(episode.id, episode.model_dump(mode="json"))

            logger.debug(
                "EpisodicMemory.encode id=%s agent=%s session=%s",
                episode.id,
                episode.agent_id,
                episode.session_id,
            )
            return episode.id

        except Exception as exc:
            raise MemoryError(f"Failed to encode episode {episode.id}: {exc}") from exc

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Retrieve episodes matching *query* via hybrid scoring.

        Steps:
          1. Compute query embedding if not provided.
          2. Over-fetch ``top_k * 2`` candidates from the vector store.
          3. Load full episode documents for each candidate.
          4. Apply time_range filter if present.
          5. Compute hybrid score (semantic + recency + importance).
          6. Apply min_score threshold.
          7. Sort by hybrid score, take top_k.
          8. Update last_accessed and access_count for returned episodes.

        Raises
        ------
        RetrievalError
            If the embedding step or vector store query fails.
        """
        t_start = time.monotonic()
        try:
            if query.query_embedding is None:
                query_embedding = await self._embedding_provider.embed(query.query_text)
            else:
                query_embedding = query.query_embedding

            # Over-fetch to allow re-ranking to pick the best top_k
            candidates = await self._vector_store.search(
                query_embedding,
                top_k=query.top_k * 2,
                filters=query.filters if query.filters else None,
            )
        except Exception as exc:
            raise RetrievalError(
                f"EpisodicMemory.retrieve failed during vector search: {exc}"
            ) from exc

        scored_items: list[tuple[float, Episode]] = []

        for candidate in candidates:
            raw_doc = await self._document_store.get(candidate.id)
            if raw_doc is None:
                logger.warning(
                    "EpisodicMemory: vector entry %s has no document, skipping",
                    candidate.id,
                )
                continue

            try:
                episode = Episode.model_validate(raw_doc)
            except Exception as exc:
                logger.warning("EpisodicMemory: failed to parse episode %s: %s", candidate.id, exc)
                continue

            if episode.lifecycle_state == MemoryLifecycleState.FORGOTTEN:
                continue

            # Apply time_range filter
            if query.time_range is not None:
                start, end = query.time_range
                ts = episode.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if not (start <= ts <= end):
                    continue

            hybrid_score = self._compute_hybrid_score(candidate.score, episode)

            if hybrid_score < query.min_score:
                continue

            scored_items.append((hybrid_score, episode))

        # Sort descending by hybrid score.
        scored_items.sort(key=lambda t: t[0], reverse=True)

        # Apply MMR reranking when there are more candidates than needed and
        # at least one episode carries an embedding (otherwise diversity is
        # meaningless and we fall back to pure score ordering).
        has_embeddings = any(ep.embedding is not None for _, ep in scored_items)
        if len(scored_items) > query.top_k and has_embeddings:
            top_items = self._mmr_rerank(
                scored_items,
                query_embedding,
                top_k=query.top_k,
            )
        else:
            top_items = scored_items[: query.top_k]

        # Update access metadata for retrieved episodes
        now_iso = datetime.now(UTC).isoformat()
        retrieved_items: list[RetrievedItem] = []
        for hybrid_score, episode in top_items:
            updated_doc = episode.model_dump(mode="json")
            updated_doc["last_accessed"] = now_iso
            updated_doc["access_count"] = episode.access_count + 1
            updated_doc["retrieval_uses"] = episode.retrieval_uses + 1
            updated_doc["retrieval_last_used_at"] = now_iso
            updated_doc["base_strength"] = min(
                episode.base_strength + 0.05 + min(0.15, episode.access_count * 0.01),
                3.0,
            )
            await self._document_store.put(episode.id, updated_doc)

            retrieved_items.append(
                RetrievedItem(
                    source_store=_STORE_NAME,
                    content=self._episode_text(episode),
                    score=min(max(hybrid_score, 0.0), 1.0),
                    metadata={
                        "episode_id": str(episode.id),
                        "agent_id": episode.agent_id,
                        "session_id": str(episode.session_id),
                        "timestamp": episode.timestamp.isoformat(),
                        "importance": episode.importance,
                        "tags": episode.tags,
                        "scope_type": episode.scope_type,
                        "scope_id": episode.scope_id,
                        "workspace_path": episode.workspace_path,
                        "repo_name": episode.repo_name,
                        "caused_by": str(episode.caused_by) if episode.caused_by else None,
                        "led_to": [str(item) for item in episode.led_to],
                        "source_episode_ids": [str(item) for item in episode.source_episode_ids],
                        "summary_kind": episode.summary_kind,
                        "summary_of_count": episode.summary_of_count,
                    },
                )
            )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        logger.debug(
            "EpisodicMemory.retrieve query=%r candidates=%d returned=%d elapsed_ms=%.1f",
            query.query_text[:60],
            len(candidates),
            len(retrieved_items),
            elapsed_ms,
        )
        return RetrievalResult(
            items=retrieved_items,
            query_time_ms=elapsed_ms,
            store_name=_STORE_NAME,
        )

    async def get(self, episode_id: UUID) -> Episode | None:
        """Fetch a single episode by its UUID.

        Returns None if no episode with *episode_id* exists.
        """
        raw_doc = await self._document_store.get(episode_id)
        if raw_doc is None:
            return None
        try:
            return Episode.model_validate(raw_doc)
        except Exception as exc:
            logger.error("EpisodicMemory.get: failed to parse episode %s: %s", episode_id, exc)
            return None

    async def update(self, episode_id: UUID, **updates: Any) -> None:
        """Apply partial updates to an existing episode.

        If embedding-related fields (context, action, outcome) change, the
        vector store entry is also refreshed with a newly computed embedding.

        Raises
        ------
        MemoryError
            If the episode does not exist or a store write fails.
        """
        raw_doc = await self._document_store.get(episode_id)
        if raw_doc is None:
            raise MemoryError(f"Episode {episode_id} not found")

        try:
            episode = Episode.model_validate(raw_doc)
        except Exception as exc:
            raise MemoryError(f"Failed to parse episode {episode_id} for update: {exc}") from exc

        embedding_fields = {"context", "action", "outcome"}
        needs_reembed = any(k in updates for k in embedding_fields)

        updated_episode = episode.model_copy(update=updates)

        if needs_reembed:
            new_text = self._episode_text(updated_episode)
            new_embedding = await self._embedding_provider.embed(new_text)
            updated_episode = updated_episode.model_copy(update={"embedding": new_embedding})
            metadata = self._build_vector_metadata(updated_episode)
            await self._vector_store.update(episode_id, new_embedding, metadata)
            logger.debug("EpisodicMemory.update id=%s re-embedded due to field change", episode_id)

        await self._document_store.put(episode_id, updated_episode.model_dump(mode="json"))
        logger.debug("EpisodicMemory.update id=%s fields=%s", episode_id, list(updates.keys()))

    async def sample_for_consolidation(self, batch_size: int = 32) -> list[Episode]:
        """Return up to *batch_size* raw episodes prioritised for consolidation.

        Prioritisation: importance × recency_factor, where recency_factor
        decays exponentially with hours since last access.  Raw episodes
        that are both important and recent are consolidated first.

        Returns
        -------
        list[Episode]
            Prioritised episodes with consolidation_state == RAW.
        """
        raw_docs = await self._document_store.query(
            filters={"consolidation_state": ConsolidationState.RAW},
            limit=10_000,  # load a large batch for client-side prioritisation
        )

        episodes: list[Episode] = []
        for doc in raw_docs:
            try:
                episodes.append(Episode.model_validate(doc))
            except Exception as exc:
                logger.warning("sample_for_consolidation: skipping malformed doc: %s", exc)

        def _priority(ep: Episode) -> float:
            hours = self._hours_since(ep.last_accessed)
            recency = math.exp(-max(ep.decay_lambda, self._config.decay.base_lambda) * hours)
            help_ratio = (
                ep.retrieval_help_count / ep.retrieval_uses if ep.retrieval_uses > 0 else 0.0
            )
            usefulness_factor = 1.0 + min(0.25, 0.05 * ep.retrieval_help_count + 0.1 * help_ratio)
            return ep.importance * recency * usefulness_factor

        episodes.sort(key=_priority, reverse=True)
        sampled = episodes[:batch_size]
        logger.debug(
            "EpisodicMemory.sample_for_consolidation raw_total=%d sampled=%d",
            len(episodes),
            len(sampled),
        )
        return sampled

    async def mark_consolidated(self, episode_ids: list[UUID]) -> None:
        """Set consolidation_state to CONSOLIDATED for all *episode_ids*.

        Episodes that cannot be found are silently skipped with a warning.
        """
        for episode_id in episode_ids:
            raw_doc = await self._document_store.get(episode_id)
            if raw_doc is None:
                logger.warning(
                    "EpisodicMemory.mark_consolidated: episode %s not found, skipping",
                    episode_id,
                )
                continue
            raw_doc["consolidation_state"] = ConsolidationState.CONSOLIDATED
            raw_doc["lifecycle_state"] = MemoryLifecycleState.CONSOLIDATED
            await self._document_store.put(episode_id, raw_doc)

        logger.debug("EpisodicMemory.mark_consolidated count=%d", len(episode_ids))

    async def run_decay_sweep(self) -> int:
        """Delete stale, already-consolidated episodes whose strength has fallen below threshold.

        Strength is computed as:
            strength = base_strength × exp(-decay_lambda × hours_since_last_access)

        An episode is deleted when:
          - strength < config.decay.forget_threshold
          - consolidation_state == CONSOLIDATED

        Returns
        -------
        int
            Number of episodes transitioned toward archival/forgetting.
        """
        all_docs = await self._document_store.query(filters={}, limit=1_000_000)
        forget_threshold = self._config.decay.forget_threshold
        changed_count = 0

        for doc in all_docs:
            try:
                episode = Episode.model_validate(doc)
            except Exception as exc:
                logger.warning("run_decay_sweep: skipping malformed episode doc: %s", exc)
                continue

            if episode.lifecycle_state == MemoryLifecycleState.FORGOTTEN:
                continue
            if episode.consolidation_state not in {
                ConsolidationState.CONSOLIDATED,
                ConsolidationState.ARCHIVED,
            }:
                continue

            hours = self._hours_since(episode.last_accessed)
            strength = episode.base_strength * math.exp(-episode.decay_lambda * hours)

            if strength < forget_threshold:
                if episode.lifecycle_state == MemoryLifecycleState.ARCHIVED:
                    doc["lifecycle_state"] = MemoryLifecycleState.FORGOTTEN
                    doc["consolidation_state"] = ConsolidationState.ARCHIVED
                    await self._document_store.put(episode.id, doc)
                    changed_count += 1
                    logger.debug(
                        "run_decay_sweep: forgot archived episode %s strength=%.4f threshold=%.4f",
                        episode.id,
                        strength,
                        forget_threshold,
                    )
                else:
                    doc["lifecycle_state"] = MemoryLifecycleState.ARCHIVED
                    doc["consolidation_state"] = ConsolidationState.ARCHIVED
                    await self._document_store.put(episode.id, doc)
                    await self._vector_store.delete(episode.id)
                    changed_count += 1
                    logger.debug(
                        "run_decay_sweep: archived episode %s strength=%.4f threshold=%.4f",
                        episode.id,
                        strength,
                        forget_threshold,
                    )

        logger.info(
            "EpisodicMemory.run_decay_sweep changed=%d total_scanned=%d",
            changed_count,
            len(all_docs),
        )
        return changed_count
