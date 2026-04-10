"""
MetaCognitionController — self-monitoring, error detection, and strategy adjustment.

Brain analog: Anterior cingulate cortex (ACC) — monitors for conflicts and
prediction errors in ongoing cognitive processes, signalling the need for
increased cognitive control and triggering strategy switches when current
approaches are failing.  The ACC works in concert with dorsolateral PFC
(executive control) and the ventromedial PFC (value-based evaluation) to
implement a Planning-Monitoring-Evaluating (PME) cycle that continuously
calibrates the agent's confidence and selects appropriate recovery strategies.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mnemon.core.interfaces import LLMProvider, MetaCognitionInterface
from mnemon.core.models import Episode, MetaEvaluation, Strategy

if TYPE_CHECKING:
    from mnemon.core.config import MetaCognitionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in strategy library
# ---------------------------------------------------------------------------

DEFAULT_STRATEGIES: list[Strategy] = [
    Strategy(name="decompose", trigger="complex goal", action="break into subgoals", weight=1.0),
    Strategy(
        name="retrieve_more",
        trigger="insufficient info",
        action="expand retrieval scope",
        weight=0.9,
    ),
    Strategy(
        name="try_different",
        trigger="repeated failure",
        action="switch approach or skill",
        weight=0.85,
    ),
    Strategy(name="ask_user", trigger="low confidence", action="request clarification", weight=0.7),
    Strategy(name="simplify", trigger="resource pressure", action="reduce goal scope", weight=0.8),
    Strategy(
        name="reflect",
        trigger="unexpected outcome",
        action="analyze and learn from surprise",
        weight=0.95,
    ),
]

_STRATEGY_INDEX: dict[str, Strategy] = {s.name: s for s in DEFAULT_STRATEGIES}

# JSON Schema for reflexion LLM call
_REFLEXION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "lessons": {
            "type": "array",
            "items": {"type": "string"},
        },
        "strategy_recommended": {"type": ["string", "null"]},
    },
    "required": ["assessment", "lessons"],
}


# ---------------------------------------------------------------------------
# MetaCognitionController
# ---------------------------------------------------------------------------


class MetaCognitionController(MetaCognitionInterface):
    """ACC-inspired metacognitive controller.

    Monitors prediction errors (RPE) and confidence signals across cognitive
    cycles, triggering reflexion via an LLM when surprises are large enough
    and recommending strategy switches to keep the agent on-track.

    Parameters
    ----------
    config:
        Meta-cognition configuration (reflexion triggers, max switches, etc.).
    llm:
        LLM provider used for reflexion (structured introspection).
    """

    _HISTORY_MAXLEN: int = 50

    def __init__(self, config: MetaCognitionConfig, llm: LLMProvider) -> None:
        self._config = config
        self._llm = llm

        self._confidence_history: deque[float] = deque(maxlen=self._HISTORY_MAXLEN)
        self._prediction_errors: deque[float] = deque(maxlen=self._HISTORY_MAXLEN)
        self._strategy_switches: int = 0
        self._lessons: list[dict[str, Any]] = []

        logger.info(
            "MetaCognitionController initialised (reflexion=%s, max_switches=%d)",
            config.reflexion.enabled,
            config.max_strategy_switches,
        )

    # ------------------------------------------------------------------
    # MetaCognitionInterface implementation
    # ------------------------------------------------------------------

    async def evaluate_cycle(self, episode: Episode, rpe: float) -> MetaEvaluation:
        """Evaluate a completed cognitive cycle and optionally trigger reflexion.

        Parameters
        ----------
        episode:
            The episode produced by flushing working memory at cycle end.
        rpe:
            Reward prediction error from the RewardProcessor for this cycle.

        Returns
        -------
        MetaEvaluation
            Assessment including confidence, detected errors, lessons, and
            optional strategy recommendation.
        """
        confidence = self._compute_confidence(episode, rpe)
        lessons: list[str] = []
        strategy_name: str | None = None

        # Trigger reflexion when RPE is large and reflexion is enabled
        reflexion_triggered = self._config.reflexion.enabled and abs(rpe) > 0.5

        if reflexion_triggered:
            strategy_name, lessons = await self._run_reflexion(episode, rpe)

        # Fall back to heuristic strategy if reflexion didn't produce one
        if not reflexion_triggered or strategy_name is None:
            strategy_name = self._heuristic_strategy(rpe, confidence)

        # Update rolling history
        self._confidence_history.append(confidence)
        self._prediction_errors.append(rpe)

        logger.debug(
            "Cycle %s evaluated — confidence=%.3f rpe=%.3f strategy=%s lessons=%d",
            episode.id,
            confidence,
            rpe,
            strategy_name,
            len(lessons),
        )

        return MetaEvaluation(
            cycle_id=episode.id,
            confidence=confidence,
            prediction_error=rpe,
            strategy_recommended=strategy_name,
            lessons=lessons,
        )

    def recommend_strategy(self, state: dict[str, Any]) -> Strategy | None:
        """Recommend a strategy based on the current cognitive state.

        Checks state signals against heuristic thresholds and returns the
        highest-priority matching strategy, respecting the per-goal switch
        limit from config.

        Returns ``None`` if no switch is warranted or the limit is reached.
        """
        if self._strategy_switches >= self._config.max_strategy_switches:
            logger.debug(
                "Strategy switch limit (%d) reached; suppressing recommendation.",
                self._config.max_strategy_switches,
            )
            return None

        strategy_name: str | None = None

        if state.get("consecutive_failures", 0) >= 3:
            strategy_name = "try_different"
        elif state.get("cognitive_load", 0.0) > 0.9:
            strategy_name = "simplify"
        elif state.get("confidence", 1.0) < 0.3:
            strategy_name = "ask_user"
        elif state.get("goal_complexity", 0.0) > 0.7:
            strategy_name = "decompose"

        if strategy_name is None:
            return None

        strategy = _STRATEGY_INDEX.get(strategy_name)
        if strategy is None:
            logger.warning("Unknown strategy name resolved: %s", strategy_name)
            return None

        # Note: counter is incremented on recommendation. Call reset_strategy_counter()
        # on goal change. If callers need dry-run queries without consuming the limit,
        # they should check get_calibration() or inspect state directly.
        self._strategy_switches += 1
        logger.info(
            "Strategy recommended: '%s' (switch #%d of %d)",
            strategy_name,
            self._strategy_switches,
            self._config.max_strategy_switches,
        )
        return strategy

    async def record_lesson(self, lesson: str, context: str) -> None:
        """Persist a metacognitive lesson with timestamp and context."""
        entry: dict[str, Any] = {
            "lesson": lesson,
            "context": context,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        self._lessons.append(entry)
        logger.info("Lesson recorded: %s (context=%s)", lesson, context)

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def get_calibration(self) -> dict[str, float]:
        """Return calibration metrics computed from rolling history.

        Returns
        -------
        dict[str, float]
            Keys: ``mean_confidence``, ``mean_prediction_error``,
            ``overconfidence_rate``, ``cycle_count``.
        """
        conf_list = list(self._confidence_history)
        rpe_list = list(self._prediction_errors)

        mean_confidence = sum(conf_list) / len(conf_list) if conf_list else 0.0
        mean_prediction_error = (
            sum(abs(e) for e in rpe_list) / len(rpe_list) if rpe_list else 0.0
        )

        overconfident_cycles = sum(
            1
            for c, e in zip(conf_list, rpe_list, strict=False)
            if c > 0.7 and abs(e) > 0.5
        )
        paired_count = min(len(conf_list), len(rpe_list))
        overconfidence_rate = (
            overconfident_cycles / paired_count if paired_count > 0 else 0.0
        )

        return {
            "mean_confidence": mean_confidence,
            "mean_prediction_error": mean_prediction_error,
            "overconfidence_rate": overconfidence_rate,
            "cycle_count": float(len(conf_list)),
        }

    def reset_strategy_counter(self) -> None:
        """Reset the per-goal strategy switch counter (call on goal change)."""
        self._strategy_switches = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(episode: Episode, rpe: float) -> float:
        """Compute a confidence score for the episode in [0, 1]."""
        raw = 0.5 + 0.3 * episode.importance - 0.2 * abs(rpe)
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _heuristic_strategy(rpe: float, confidence: float) -> str | None:
        """Select a strategy name from simple RPE/confidence heuristics."""
        if rpe < -0.5:
            return "try_different"
        if confidence < 0.3:
            return "ask_user"
        if rpe < 0.0:
            return "reflect"
        return None

    async def _run_reflexion(
        self, episode: Episode, rpe: float
    ) -> tuple[str | None, list[str]]:
        """Ask the LLM to reflect on the episode; fall back gracefully on error."""
        prompt = (
            "Evaluate this cognitive cycle:\n"
            f"Context: {episode.context}\n"
            f"Action: {episode.action}\n"
            f"Outcome: {episode.outcome}\n"
            f"Prediction error: {rpe}\n\n"
            "What went well? What went wrong? What lesson should be learned?\n"
            'Return JSON with: "assessment", "lessons" (list of strings), '
            '"strategy_recommended" (string or null)'
        )

        try:
            data = await self._llm.generate_structured(
                prompt=prompt,
                response_schema=_REFLEXION_SCHEMA,
            )

            lessons: list[str] = [
                str(item) for item in data.get("lessons", []) if item
            ]
            strategy_name: str | None = data.get("strategy_recommended") or None

            # Record each lesson
            for lesson in lessons:
                await self.record_lesson(lesson, episode.context)

            logger.info(
                "Reflexion complete for cycle %s — strategy=%s lessons=%d",
                episode.id,
                strategy_name,
                len(lessons),
            )
            return strategy_name, lessons

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reflexion LLM call failed for cycle %s (%s); falling back to heuristics.",
                episode.id,
                exc,
            )
            return None, []
