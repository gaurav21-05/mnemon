"""
ValenceMemoryStore — emotional salience tagging and rapid appraisal.

Brain analog: Amygdala — rapidly evaluates incoming stimuli for emotional
significance by matching them against learned trigger-valence associations.
Arousal modulates encoding strength (high-arousal memories persist longer),
and extinction learning gradually weakens unreinforced associations, mirroring
the basolateral–central amygdala circuit dynamics underlying fear extinction.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mnemon.core.interfaces import EmbeddingProvider, ValenceMemoryInterface
from mnemon.core.models import PerceptUnit, SalienceScore, ValenceAssociation

if TYPE_CHECKING:
    from mnemon.core.config import ValenceConfig

logger = logging.getLogger(__name__)

# Minimum values below which an association is considered extinguished and removed.
_EXTINCTION_FLOOR = 0.01


class ValenceMemoryStore(ValenceMemoryInterface):
    """In-memory valence association store with Pavlovian update and extinction.

    Associations are keyed by their trigger string.  Appraisal uses substring
    matching against :attr:`~mnemon.core.models.PerceptUnit.normalized` to
    locate relevant associations — an approximation of how the amygdala
    pattern-matches incoming stimuli against stored threat/reward templates.

    An optional :class:`~mnemon.core.interfaces.EmbeddingProvider` is accepted
    for future similarity-based matching but is not required for the in-process
    implementation (Phase 1).
    """

    def __init__(
        self,
        config: ValenceConfig,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._config = config
        self._embedding_provider = embedding_provider
        # Primary store: trigger string → ValenceAssociation
        self._store: dict[str, ValenceAssociation] = {}
        logger.debug(
            "ValenceMemoryStore initialised — lr=%.4f extinction_rate=%.4f",
            config.learning_rate,
            config.extinction_rate,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    def _matching_associations(self, normalized_text: str) -> list[ValenceAssociation]:
        """Return all associations whose trigger appears as a substring of *normalized_text*."""
        return [
            assoc
            for trigger, assoc in self._store.items()
            if trigger in normalized_text
        ]

    # ------------------------------------------------------------------
    # ValenceMemoryInterface implementation
    # ------------------------------------------------------------------

    async def appraise(self, percept: PerceptUnit) -> SalienceScore:
        """Compute a :class:`~mnemon.core.models.SalienceScore` for *percept*.

        Sub-score derivation
        --------------------
        * ``raw_salience``: maximum of ``abs(valence) * arousal`` over all
          matching associations, or 0.0 if none match.
        * ``goal_relevance``: 0.5 (default; enriched by the attention controller).
        * ``novelty``: 1.0 if no associations match (completely novel stimulus),
          otherwise ``1.0 - max(confidence)`` of matching associations.
        * ``combined``: weighted sum with a small sentiment contribution.
        """
        matches = self._matching_associations(percept.normalized)

        if matches:
            raw_salience = max(abs(a.valence) * a.arousal for a in matches)
            novelty = 1.0 - max(a.confidence for a in matches)
        else:
            raw_salience = 0.0
            novelty = 1.0

        goal_relevance = 0.5
        sentiment_contrib = abs(percept.sentiment)

        combined = (
            0.3 * raw_salience
            + 0.3 * goal_relevance
            + 0.3 * novelty
            + 0.1 * sentiment_contrib
        )

        score = SalienceScore(
            percept_id=percept.id,
            raw_salience=self._clamp(raw_salience),
            goal_relevance=self._clamp(goal_relevance),
            novelty=self._clamp(novelty),
            combined=self._clamp(combined),
        )
        logger.debug(
            "Appraised percept %s — raw=%.3f novelty=%.3f combined=%.3f matches=%d",
            percept.id,
            score.raw_salience,
            score.novelty,
            score.combined,
            len(matches),
        )
        return score

    async def update(self, triggers: list[str], reward_signal: float) -> None:
        """Update valence associations for each trigger via a Pavlovian rule.

        For existing associations:

        .. code-block:: text

            valence  = (1 - lr) * valence  + lr * reward_signal
            arousal  = (1 - lr) * arousal  + lr * abs(reward_signal)

        New triggers are seeded with a low initial confidence (0.3) to signal
        that the association is tentative until further reinforcement.
        """
        lr = self._config.learning_rate
        now = datetime.now(UTC)

        for trigger in triggers:
            if trigger in self._store:
                assoc = self._store[trigger]
                new_valence = (1 - lr) * assoc.valence + lr * reward_signal
                new_arousal = (1 - lr) * assoc.arousal + lr * abs(reward_signal)
                # Pydantic models are mutable by default (no frozen=True here).
                assoc.valence = self._clamp(new_valence, -1.0, 1.0)
                assoc.arousal = self._clamp(new_arousal)
                assoc.exposure_count += 1
                assoc.last_encountered = now
                logger.debug(
                    "Updated association '%s' valence=%.3f arousal=%.3f",
                    trigger,
                    assoc.valence,
                    assoc.arousal,
                )
            else:
                assoc = ValenceAssociation(
                    trigger=trigger,
                    valence=self._clamp(reward_signal, -1.0, 1.0),
                    arousal=self._clamp(abs(reward_signal)),
                    confidence=0.3,
                    exposure_count=1,
                    last_encountered=now,
                )
                self._store[trigger] = assoc
                logger.debug(
                    "Created association '%s' valence=%.3f arousal=%.3f",
                    trigger,
                    assoc.valence,
                    assoc.arousal,
                )

    async def get_associations(self, trigger: str) -> list[ValenceAssociation]:
        """Return the association for *trigger* (exact match), or an empty list."""
        assoc = self._store.get(trigger)
        return [assoc] if assoc is not None else []

    async def run_extinction_sweep(self) -> int:
        """Decay or remove unreinforced associations.

        Associations not encountered in the last hour are weakened:

        .. code-block:: text

            arousal = arousal * (1 - extinction_rate)
            valence = valence * (1 - extinction_rate)

        Associations whose arousal *and* abs(valence) both drop below
        ``_EXTINCTION_FLOOR`` are removed from the store entirely.

        Returns the total count of associations weakened or removed.
        """
        rate = self._config.extinction_rate
        now = datetime.now(UTC)
        one_hour_s = 3_600.0
        affected = 0

        to_delete: list[str] = []
        for trigger, assoc in self._store.items():
            if assoc.last_encountered is None:
                last_s = 0.0
            else:
                last_s = (now - assoc.last_encountered).total_seconds()

            if last_s < one_hour_s:
                continue  # Recently reinforced — leave untouched.

            assoc.arousal = assoc.arousal * (1 - rate)
            assoc.valence = assoc.valence * (1 - rate)
            affected += 1

            if assoc.arousal < _EXTINCTION_FLOOR and abs(assoc.valence) < _EXTINCTION_FLOOR:
                to_delete.append(trigger)

        for trigger in to_delete:
            del self._store[trigger]
            logger.debug("Extinguished association '%s'", trigger)

        logger.debug(
            "Extinction sweep complete — %d affected, %d removed",
            affected,
            len(to_delete),
        )
        return affected
