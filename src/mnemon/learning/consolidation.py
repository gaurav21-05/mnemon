"""
Offline memory consolidation pipeline — hippocampal replay into neocortex.

Brain analog: Slow-wave sleep (SWS) memory consolidation. During SWS the
hippocampus replays episodic traces via sharp-wave ripples (SWRs), driving
Hebbian potentiation in neocortical circuits to build stable, compressed,
context-independent semantic representations.

This module implements that pipeline:
  1. High-priority episodes are sampled from the replay buffer (SWR selection).
  2. An LLM extracts structured subject–predicate–object facts (cortical
     decoding of hippocampal patterns).
  3. Extracted triples are upserted into semantic memory (neocortical
     long-term potentiation).
  4. Processed episodes are marked consolidated, and replay priorities are
     updated to reflect how much information was still extractable (TD-like
     information gain signal).

The result mirrors the brain's nightly knowledge distillation: episodic
detail is gradually replaced by generalised semantic structure.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any
from uuid import UUID

from mnemon.core.config import ConsolidationConfig
from mnemon.core.exceptions import ConsolidationError
from mnemon.core.interfaces import (
    ConsolidationEngineInterface,
    EmbeddingProvider,
    EpisodicMemoryInterface,
    LLMProvider,
    SemanticMemoryInterface,
)
from mnemon.core.models import (
    ConsolidationResult,
    ConsolidationState,
    EntityRef,
    Episode,
    SemanticTriple,
)
from mnemon.learning.replay import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)

# JSON Schema used for structured LLM extraction of semantic triples.
_TRIPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["subject", "predicate", "object", "confidence"],
            },
        }
    },
    "required": ["triples"],
}

# Heuristic: objects shorter than this character threshold are treated as
# literal string values rather than named entity references.
_ENTITY_NAME_MIN_LEN: int = 2


def _make_entity_ref(name: str) -> EntityRef:
    """Build a deterministic EntityRef from a string name.

    Uses UUID5 over NAMESPACE_DNS so the same canonical name always
    produces the same UUID, enabling natural deduplication across runs.
    """
    canonical = name.lower().strip()
    entity_id = uuid.uuid5(uuid.NAMESPACE_DNS, canonical)
    return EntityRef(entity_id=entity_id, name=name.strip())


def _looks_like_entity(text: str) -> bool:
    """Return True when *text* should be treated as a named entity reference.

    Simple heuristic: non-empty, more than a single word token of minimum
    length, and not a pure numeric / boolean literal.
    """
    stripped = text.strip()
    if len(stripped) < _ENTITY_NAME_MIN_LEN:
        return False
    # Pure numeric values and simple boolean literals are kept as strings.
    try:
        float(stripped)
        return False
    except ValueError:
        pass
    if stripped.lower() in {"true", "false", "null", "none", "yes", "no"}:
        return False
    return True


class ConsolidationEngine(ConsolidationEngineInterface):
    """Sleep-like consolidation pipeline from episodic to semantic memory.

    Implements :class:`~mnemon.core.interfaces.ConsolidationEngineInterface`.

    Each call to :meth:`run_cycle` processes one batch of high-priority
    episodes through four sequential stages:

    1. **Selection** — sample from the prioritised replay buffer.
    2. **Extraction** — invoke LLM to extract semantic triples per episode.
    3. **Integration** — upsert triples into the semantic knowledge graph.
    4. **Cleanup** — mark episodes consolidated; update replay priorities.
    """

    def __init__(
        self,
        config: ConsolidationConfig,
        episodic_memory: EpisodicMemoryInterface,
        semantic_memory: SemanticMemoryInterface,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
        replay_buffer: PrioritizedReplayBuffer,
    ) -> None:
        self._config = config
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._llm = llm
        self._embedding = embedding_provider
        self._replay = replay_buffer
        # Scheduled trigger configurations keyed by trigger name.
        self._scheduled_triggers: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # ConsolidationEngineInterface implementation
    # ------------------------------------------------------------------

    async def run_cycle(self) -> ConsolidationResult:
        """Execute one consolidation pass over the pending episode queue.

        Runs four stages — selection, extraction, integration, cleanup —
        and returns a :class:`~mnemon.core.models.ConsolidationResult`
        summarising the outcome.

        Returns
        -------
        ConsolidationResult
            Summary of triples extracted, entities resolved, and episodes
            processed during this cycle.

        Raises
        ------
        ConsolidationError
            If a fatal, non-recoverable error occurs outside of per-episode
            LLM extraction (which is handled gracefully).
        """
        start = time.monotonic()
        logger.info(
            "ConsolidationEngine: starting cycle (batch_size=%d, buffer_size=%d)",
            self._config.batch_size,
            self._replay.size,
        )

        # ----------------------------------------------------------------
        # Stage 1: SELECTION
        # ----------------------------------------------------------------
        if self._replay.size == 0:
            logger.info("ConsolidationEngine: replay buffer empty — returning zero result")
            return ConsolidationResult(
                episodes_processed=0,
                triples_extracted=0,
                entities_resolved=0,
                conflicts_detected=0,
                duration_ms=(time.monotonic() - start) * 1000.0,
            )

        sampled_experiences = self._replay.sample(self._config.batch_size)
        logger.debug("Stage 1 (SELECTION): sampled %d replay experiences", len(sampled_experiences))

        # Load full Episode objects; skip unavailable or already-consolidated ones.
        valid_episodes: list[tuple[Episode, int, float]] = []  # (episode, tree_index, priority)
        for experience in sampled_experiences:
            episode: Episode | None = await self._episodic.get(experience.episode_id)
            if episode is None:
                logger.debug(
                    "Episode %s not found in episodic store — skipping",
                    experience.episode_id,
                )
                continue
            if episode.consolidation_state != ConsolidationState.RAW:
                logger.debug(
                    "Episode %s not raw (state=%s) — skipping",
                    experience.episode_id,
                    episode.consolidation_state,
                )
                continue
            valid_episodes.append((episode, experience.tree_index, experience.priority))

        logger.debug(
            "Stage 1 (SELECTION): %d valid episodes after filtering", len(valid_episodes)
        )

        if not valid_episodes:
            logger.info(
                "ConsolidationEngine: no valid episodes after filtering — returning zero result"
            )
            return ConsolidationResult(
                episodes_processed=0,
                triples_extracted=0,
                entities_resolved=0,
                conflicts_detected=0,
                duration_ms=(time.monotonic() - start) * 1000.0,
            )

        # ----------------------------------------------------------------
        # Stage 2: EXTRACTION (LLM-powered)
        # ----------------------------------------------------------------
        logger.info(
            "Stage 2 (EXTRACTION): extracting triples from %d episodes", len(valid_episodes)
        )

        # Maps episode_id → extracted triples for that episode.
        episode_triples: dict[UUID, list[SemanticTriple]] = {}
        extracted_episode_ids: set[UUID] = set()
        failed_extractions: dict[UUID, int] = {}

        for episode, tree_index, priority in valid_episodes:
            prompt = (
                "Given this agent experience:\n"
                f"Context: {episode.context}\n"
                f"Action taken: {episode.action}\n"
                f"Outcome: {episode.outcome}\n\n"
                "Extract all factual knowledge as structured triples.\n"
                'Return a JSON object with key "triples" containing a list of objects,\n'
                'each with: "subject", "predicate", "object", "confidence" (0.0-1.0).\n'
                "Only extract concrete, generalizable facts — not ephemeral observations."
            )

            try:
                result = await self._llm.generate_structured(
                    prompt=prompt,
                    response_schema=_TRIPLE_SCHEMA,
                )
            except Exception as exc:  # noqa: BLE001
                attempts = episode.consolidation_attempts + 1
                logger.warning(
                    "LLM extraction failed for episode %s (attempt %d/%d) — skipping. Error: %s",
                    episode.id,
                    attempts,
                    self._config.max_extraction_retries,
                    exc,
                )
                episode_triples[episode.id] = []
                failed_extractions[episode.id] = attempts
                continue

            raw_triples: list[dict[str, Any]] = result.get("triples", [])
            extracted_episode_ids.add(episode.id)
            logger.debug(
                "Episode %s: LLM returned %d raw triples", episode.id, len(raw_triples)
            )

            triples: list[SemanticTriple] = []
            # Collect triple texts for batch embedding.
            triple_texts: list[str] = [
                f"{t['subject']} {t['predicate']} {t['object']}"
                for t in raw_triples
            ]

            embeddings: list[list[float]] = []
            if triple_texts:
                try:
                    embeddings = await self._embedding.embed_batch(triple_texts)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Embedding failed for episode %s triples — proceeding without embeddings. Error: %s",
                        episode.id,
                        exc,
                    )
                    embeddings = [None] * len(triple_texts)  # type: ignore[list-item]

            for idx, raw in enumerate(raw_triples):
                subject_str: str = raw.get("subject", "").strip()
                predicate_str: str = raw.get("predicate", "").strip()
                object_str: str = raw.get("object", "").strip()
                confidence: float = float(raw.get("confidence", 0.5))

                if not subject_str or not predicate_str or not object_str:
                    logger.debug(
                        "Skipping malformed triple from episode %s: %s", episode.id, raw
                    )
                    continue

                subject_ref = _make_entity_ref(subject_str)

                if _looks_like_entity(object_str):
                    object_val: EntityRef | str = _make_entity_ref(object_str)
                else:
                    object_val = object_str

                embedding_vec: list[float] | None = embeddings[idx] if idx < len(embeddings) else None

                triple = SemanticTriple(
                    subject=subject_ref,
                    predicate=predicate_str,
                    object=object_val,
                    confidence=max(0.0, min(1.0, confidence)),
                    source_episodes=[episode.id],
                    embedding=embedding_vec,
                )
                triples.append(triple)

            episode_triples[episode.id] = triples
            logger.debug(
                "Episode %s: built %d SemanticTriple objects", episode.id, len(triples)
            )

        # ----------------------------------------------------------------
        # Stage 3: SEMANTIC INTEGRATION
        # ----------------------------------------------------------------
        logger.info(
            "Stage 3 (INTEGRATION): upserting triples for %d episodes",
            len(episode_triples),
        )

        total_triples_written = 0
        total_entities_resolved = 0
        upserted_episode_ids: set[UUID] = set()

        for episode, _tree_index, _priority in valid_episodes:
            triples = episode_triples.get(episode.id, [])
            if episode.id not in extracted_episode_ids:
                continue
            if not triples:
                upserted_episode_ids.add(episode.id)
                logger.debug("Episode %s: no triples to upsert", episode.id)
                continue
            try:
                written = await self._semantic.upsert_triples(triples)
                total_triples_written += written
                upserted_episode_ids.add(episode.id)
                # Count distinct entity refs across subject and object fields.
                entity_ids: set[UUID] = set()
                for t in triples:
                    entity_ids.add(t.subject.entity_id)
                    if isinstance(t.object, EntityRef):
                        entity_ids.add(t.object.entity_id)
                total_entities_resolved += len(entity_ids)
                logger.debug(
                    "Episode %s: upserted %d triples, resolved %d entities",
                    episode.id,
                    written,
                    len(entity_ids),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Semantic upsert failed for episode %s — continuing. Error: %s",
                    episode.id,
                    exc,
                )

        # ----------------------------------------------------------------
        # Stage 4: CLEANUP
        # ----------------------------------------------------------------
        processed_ids: list[UUID] = [
            ep.id for ep, _, _ in valid_episodes
            if ep.id in upserted_episode_ids
        ]
        logger.info(
            "Stage 4 (CLEANUP): marking %d/%d episodes consolidated",
            len(processed_ids),
            len(valid_episodes),
        )

        try:
            await self._episodic.mark_consolidated(processed_ids)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "mark_consolidated failed for %d episodes: %s", len(processed_ids), exc
            )
            raise ConsolidationError(
                f"Failed to mark episodes consolidated: {exc}"
            ) from exc

        for episode_id in processed_ids:
            await self._episodic.update(episode_id, consolidation_attempts=0)

        for episode_id, attempts in failed_extractions.items():
            next_state = (
                ConsolidationState.FAILED
                if attempts >= self._config.max_extraction_retries
                else ConsolidationState.RAW
            )
            await self._episodic.update(
                episode_id,
                consolidation_attempts=attempts,
                consolidation_state=next_state,
            )

        # Update replay priorities.
        # Episodes that yielded many triples have had their information extracted;
        # lower priority so they are not replayed again soon.
        # Episodes that yielded few triples are still information-rich; raise
        # priority slightly to encourage re-sampling.
        max_triples_in_batch = max(
            (len(episode_triples.get(ep.id, [])) for ep, _, _ in valid_episodes),
            default=1,
        ) or 1

        for episode, tree_index, original_priority in valid_episodes:
            n_triples = len(episode_triples.get(episode.id, []))
            if episode.id in failed_extractions:
                continue
            # Normalised information-extracted fraction in [0, 1].
            extracted_fraction = n_triples / max_triples_in_batch
            # High extraction → low new priority; low extraction → keep high priority.
            new_priority = original_priority * (1.0 - 0.9 * extracted_fraction)
            new_priority = max(new_priority, 1e-6)
            self._replay.update_priorities([tree_index], [new_priority])
            logger.debug(
                "Episode %s: updated replay priority %.6f → %.6f (%d triples extracted)",
                episode.id,
                original_priority,
                new_priority,
                n_triples,
            )

        duration_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "ConsolidationEngine: cycle complete — episodes_processed=%d, "
            "triples_extracted=%d, entities_resolved=%d, duration_ms=%.1f",
            len(processed_ids),
            total_triples_written,
            total_entities_resolved,
            duration_ms,
        )

        return ConsolidationResult(
            episodes_processed=len(processed_ids),
            triples_extracted=total_triples_written,
            entities_resolved=total_entities_resolved,
            conflicts_detected=0,
            duration_ms=duration_ms,
        )

    def queue_status(self) -> dict[str, Any]:
        """Return the current state of the consolidation queue.

        Returns
        -------
        dict[str, Any]
            ``{"pending": int, "batch_size": int}``
        """
        return {
            "pending": self._replay.size,
            "batch_size": self._config.batch_size,
        }

    def schedule(self, trigger: str, **kwargs: Any) -> None:
        """Register a consolidation trigger for future orchestrator use.

        The actual scheduling is performed by the orchestrator; this method
        only records the trigger configuration so it can be inspected or
        replayed.

        Parameters
        ----------
        trigger:
            Named event (e.g. ``"idle"``, ``"episode_count_threshold"``).
        **kwargs:
            Trigger-specific parameters (thresholds, cron expressions, etc.).
        """
        self._scheduled_triggers[trigger] = dict(kwargs)
        logger.info(
            "ConsolidationEngine: registered trigger '%s' with config %s",
            trigger,
            kwargs,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    async def seed_replay_buffer(self) -> None:
        """Populate the replay buffer from unconsolidated episodic memory.

        Samples ``config.batch_size * 4`` candidate episodes from episodic
        memory and computes a priority for each based on its reward signal and
        importance score.  Should be called once during system initialisation
        (or after a cold-start) to prime the buffer before the first
        :meth:`run_cycle` call.

        Priority formula::

            priority = (|reward_signal| + 0.01) ** alpha * importance

        where ``alpha`` is ``config.replay.alpha``.
        """
        seed_size = self._config.batch_size * 4
        logger.info(
            "ConsolidationEngine: seeding replay buffer (requesting %d episodes)", seed_size
        )

        try:
            episodes: list[Episode] = await self._episodic.sample_for_consolidation(
                batch_size=seed_size
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("seed_replay_buffer: episodic sample failed: %s", exc)
            raise ConsolidationError(f"Failed to seed replay buffer: {exc}") from exc

        alpha = self._config.replay.alpha
        added = 0
        for episode in episodes:
            priority = (abs(episode.reward_signal) + 0.01) ** alpha * episode.importance
            self._replay.add(episode.id, priority)
            added += 1

        logger.info(
            "ConsolidationEngine: seeded replay buffer with %d episodes", added
        )
