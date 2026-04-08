"""
SensoryBuffer — pre-attentive input processing module.

Brain analog: Primary sensory cortices (V1 for vision, A1 for audition) — the
earliest cortical processing stages that transduce raw stimuli into normalised,
modality-tagged representations before the thalamic attention gate routes them
for higher processing. The circular buffer models the brief persistence of
pre-attentive icons in iconic/echoic memory (~30 s decay).
"""

from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime, timezone

from mnemon.core.config import SensoryConfig
from mnemon.core.interfaces import SensoryBufferInterface
from mnemon.core.models import Entity, Modality, PerceptUnit

logger = logging.getLogger(__name__)

_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b")


class SensoryBuffer(SensoryBufferInterface):
    """Circular buffer with TTL-based eviction for incoming percepts.

    Items are stored as :class:`~mnemon.core.models.PerceptUnit` objects.
    Expired items are pruned lazily on every :meth:`process` and :meth:`peek`
    call, mirroring the passive decay of sensory memory.
    """

    def __init__(self, config: SensoryConfig) -> None:
        self._config = config
        self._buffer: deque[PerceptUnit] = deque(maxlen=config.capacity)
        logger.debug(
            "SensoryBuffer initialised — capacity=%d ttl_ms=%d",
            config.capacity,
            config.ttl_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _now_ms(self) -> float:
        """Current time as milliseconds since epoch."""
        return datetime.now(timezone.utc).timestamp() * 1_000

    def _is_expired(self, percept: PerceptUnit) -> bool:
        """Return True if *percept* has exceeded its TTL."""
        age_ms = self._now_ms() - percept.timestamp.timestamp() * 1_000
        return age_ms > percept.ttl_ms

    def _prune_expired(self) -> None:
        """Remove expired percepts from the left of the deque in-place."""
        before = len(self._buffer)
        # Rebuild deque keeping only live items; deque has no in-place filter.
        live = [p for p in self._buffer if not self._is_expired(p)]
        if len(live) != before:
            self._buffer.clear()
            self._buffer.extend(live)
            logger.debug(
                "SensoryBuffer pruned %d expired percept(s); %d remaining",
                before - len(live),
                len(live),
            )

    def _extract_entities(self, raw_input: str) -> list[Entity]:
        """Cheap heuristic NER fallback using title-case and acronym spans."""
        seen: set[str] = set()
        entities: list[Entity] = []
        for match in _ENTITY_PATTERN.findall(raw_input):
            canonical = match.strip()
            normalized = canonical.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            entities.append(Entity(canonical_name=canonical, type="unknown"))
        return entities

    # ------------------------------------------------------------------
    # SensoryBufferInterface implementation
    # ------------------------------------------------------------------

    async def process(
        self,
        raw_input: str,
        modality: Modality = Modality.TEXT,
    ) -> PerceptUnit:
        """Normalise *raw_input* and store a new PerceptUnit in the buffer.

        Normalisation applies strip + lowercase.  Token count is estimated as
        ``len(text) // 4`` (a rough approximation of sub-word tokenisation).
        Embedding, entities, and sentiment are left at their defaults; the
        attention controller enriches them later in the cognitive cycle.
        """
        self._prune_expired()

        normalized = raw_input.strip().lower()
        token_count = max(1, len(normalized) // 4)

        percept = PerceptUnit(
            modality=modality,
            raw_content=raw_input,
            normalized=normalized,
            tokens=token_count,
            embedding=None,
            entities=self._extract_entities(raw_input) if modality == Modality.TEXT else [],
            sentiment=0.0,
            ttl_ms=self._config.ttl_ms,
        )

        self._buffer.append(percept)
        logger.debug(
            "SensoryBuffer stored percept id=%s modality=%s tokens=%d",
            percept.id,
            modality,
            token_count,
        )
        return percept

    def peek(self) -> list[PerceptUnit]:
        """Return all live percepts without consuming them.

        Expired items are pruned before the snapshot is taken.
        """
        self._prune_expired()
        return list(self._buffer)

    def clear(self) -> None:
        """Discard all buffered percepts (models sensory decay / reset)."""
        count = len(self._buffer)
        self._buffer.clear()
        logger.debug("SensoryBuffer cleared %d percept(s)", count)
