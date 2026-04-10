"""
AttentionController — selective attention and Global Workspace broadcasting.

Brain analog: Basal forebrain cholinergic system — modulates cortical
signal-to-noise ratio, determining which stimuli are amplified to global
workspace visibility (GWT broadcast) and which are suppressed. Acetylcholine
release sharpens cortical representations and raises the effective threshold
for non-salient inputs, mirroring the adaptive threshold logic implemented here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from mnemon.core.interfaces import AttentionControllerInterface, ValenceMemoryInterface
from mnemon.core.models import GateDecision, Goal, PerceptUnit, SalienceScore

if TYPE_CHECKING:
    from mnemon.core.config import AttentionConfig

logger: Final = logging.getLogger(__name__)

_DEFAULT_URGENCY: Final[float] = 0.5
_DEFAULT_GOAL_RELEVANCE: Final[float] = 0.3
_DEFAULT_NOVELTY: Final[float] = 0.5


class AttentionController(AttentionControllerInterface):
    """Selective attention gate implementing the basal forebrain cholinergic analog.

    Scores incoming percepts along four dimensions — valence, goal relevance,
    novelty, and urgency — then routes them to BROADCAST, QUEUE, or DISCARD
    based on configurable thresholds that adapt under cognitive load.
    """

    def __init__(self, config: AttentionConfig, valence: ValenceMemoryInterface) -> None:
        self._config = config
        self._valence = valence
        self._broadcast_threshold: float = config.broadcast_threshold
        self._attention_threshold: float = config.attention_threshold
        logger.info(
            "AttentionController initialised — broadcast_threshold=%.3f "
            "attention_threshold=%.3f adaptive=%s",
            self._broadcast_threshold,
            self._attention_threshold,
            config.adaptive_thresholds,
        )

    async def score(
        self,
        percept: PerceptUnit,
        active_goals: list[Goal],
    ) -> SalienceScore:
        """Compute multi-dimensional attentional salience for *percept*.

        Parameters
        ----------
        percept:
            Normalised percept unit from the sensory buffer.
        active_goals:
            Goals currently held in working memory that bias top-down attention.

        Returns
        -------
        SalienceScore
            Decomposed score combining valence, goal relevance, novelty, and
            urgency into a single gating-ready combined value.
        """
        # 1. Valence component — delegate to amygdala analog.
        preliminary = await self._valence.appraise(percept)
        raw_salience: float = preliminary.raw_salience

        # 2. Goal relevance — top-down attentional bias.
        goal_relevance: float = _DEFAULT_GOAL_RELEVANCE
        if active_goals:
            percept_words: set[str] = set(percept.normalized.lower().split())
            max_relevance: float = 0.0
            for goal in active_goals:
                goal_words: set[str] = set(goal.description.lower().split())
                shared: int = len(percept_words & goal_words)
                denominator: int = max(len(percept_words), len(goal_words), 1)
                relevance: float = shared / denominator
                if relevance > max_relevance:
                    max_relevance = relevance
            goal_relevance = max_relevance

        # 3. Novelty — use preliminary score if available, else default.
        novelty: float = preliminary.novelty if preliminary.novelty > 0.0 else _DEFAULT_NOVELTY

        # 4. Combined weighted score with default urgency (no urgency system yet).
        urgency: float = _DEFAULT_URGENCY
        w = self._config.weights
        combined: float = (
            w.valence * raw_salience
            + w.goal_relevance * goal_relevance
            + w.novelty * novelty
            + w.urgency * urgency
        )
        combined = max(0.0, min(1.0, combined))

        logger.debug(
            "Scored percept %s — raw_salience=%.3f goal_relevance=%.3f "
            "novelty=%.3f urgency=%.3f combined=%.3f",
            percept.id,
            raw_salience,
            goal_relevance,
            novelty,
            urgency,
            combined,
        )

        return SalienceScore(
            percept_id=percept.id,
            raw_salience=raw_salience,
            goal_relevance=goal_relevance,
            novelty=novelty,
            combined=combined,
        )

    def gate(self, salience: SalienceScore) -> GateDecision:
        """Route *salience* to BROADCAST, QUEUE, or DISCARD.

        Decision boundaries mirror thalamic gating: only sufficiently salient
        signals reach global workspace awareness; sub-threshold signals are
        queued for deferred processing or discarded entirely.
        """
        if salience.combined >= self._broadcast_threshold:
            logger.debug(
                "Gate BROADCAST percept %s (combined=%.3f >= broadcast_threshold=%.3f)",
                salience.percept_id,
                salience.combined,
                self._broadcast_threshold,
            )
            return GateDecision.BROADCAST

        if salience.combined >= self._attention_threshold:
            logger.debug(
                "Gate QUEUE percept %s (combined=%.3f >= attention_threshold=%.3f)",
                salience.percept_id,
                salience.combined,
                self._attention_threshold,
            )
            return GateDecision.QUEUE

        logger.debug(
            "Gate DISCARD percept %s (combined=%.3f < attention_threshold=%.3f)",
            salience.percept_id,
            salience.combined,
            self._attention_threshold,
        )
        return GateDecision.DISCARD

    def adjust_thresholds(self, cognitive_load: float) -> None:
        """Adapt attention thresholds based on current cognitive load.

        Under high load (cognitive_load → 1.0) thresholds rise, increasing
        selectivity and filtering out marginal stimuli. This mirrors the
        cholinergic suppression of cortical noise under working-memory pressure.

        Parameters
        ----------
        cognitive_load:
            Current occupancy of working memory in [0, 1] where 1 = full.
        """
        if not self._config.adaptive_thresholds:
            logger.debug(
                "Adaptive thresholds disabled — ignoring cognitive_load=%.3f",
                cognitive_load,
            )
            return

        load = max(0.0, min(1.0, cognitive_load))
        self._broadcast_threshold = min(
            1.0, self._config.broadcast_threshold + 0.15 * load
        )
        self._attention_threshold = min(
            1.0, self._config.attention_threshold + 0.1 * load
        )
        logger.info(
            "Thresholds adjusted for cognitive_load=%.3f — "
            "broadcast_threshold=%.3f attention_threshold=%.3f",
            load,
            self._broadcast_threshold,
            self._attention_threshold,
        )
