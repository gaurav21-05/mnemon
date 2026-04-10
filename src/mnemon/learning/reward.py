"""
Reward prediction error computation for the Mnemon learning subsystem.

Brain analog: The VTA/Substantia Nigra dopaminergic system that computes
reward prediction errors (RPE) driving reinforcement learning updates
across procedural memory (basal ganglia) and valence associations (amygdala).
A positive RPE encodes "better than expected" and triggers potentiation;
a negative RPE encodes "worse than expected" and triggers depression of
recently active synaptic pathways.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mnemon.core.interfaces import RewardProcessorInterface
from mnemon.core.models import RewardSignal

if TYPE_CHECKING:
    from uuid import UUID

    from mnemon.core.config import RewardConfig

logger = logging.getLogger(__name__)


class RewardProcessor(RewardProcessorInterface):
    """Temporal-difference reward processor implementing VTA/SN dopaminergic RPE.

    Computes TD(0) reward prediction errors and composite reward signals
    from weighted source contributions. Maintains running statistics on
    positive and negative RPE accumulation for diagnostic monitoring.

    Brain analog: VTA/Substantia Nigra dopaminergic system — computes
    reward prediction errors (RPE) that drive reinforcement learning
    updates across procedural memory and valence associations.

    Parameters
    ----------
    config:
        RewardConfig governing the discount factor and per-source weights.
    """

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._total_positive_rpe: float = 0.0
        self._total_negative_rpe: float = 0.0
        self._cycle_count: int = 0
        logger.debug(
            "RewardProcessor initialised (gamma=%.3f)",
            config.discount_factor,
        )

    # ------------------------------------------------------------------
    # RewardProcessorInterface implementation
    # ------------------------------------------------------------------

    async def compute_rpe(
        self,
        episode_id: UUID,
        predicted_value: float,
        actual_reward: float,
        next_value: float = 0.0,
    ) -> RewardSignal:
        """Compute the temporal-difference reward prediction error.

        Uses the TD(0) update rule:
            RPE = actual_reward + γ * next_value - predicted_value

        Parameters
        ----------
        episode_id:
            UUID of the episode this signal is associated with.
        predicted_value:
            The value estimate made before the outcome was observed.
        actual_reward:
            The reward actually received at this timestep.
        next_value:
            Bootstrap value of the successor state (0.0 for terminal states).

        Returns
        -------
        RewardSignal
            Full signal object carrying the raw RPE and metadata.
        """
        gamma = self._config.discount_factor
        rpe = actual_reward + gamma * next_value - predicted_value

        # Update running statistics
        self._cycle_count += 1
        if rpe >= 0.0:
            self._total_positive_rpe += rpe
        else:
            self._total_negative_rpe += rpe

        logger.debug(
            "RPE computed for episode %s: predicted=%.4f actual=%.4f "
            "next=%.4f gamma=%.3f rpe=%.4f",
            episode_id,
            predicted_value,
            actual_reward,
            next_value,
            gamma,
            rpe,
        )

        return RewardSignal(
            episode_id=episode_id,
            predicted_value=predicted_value,
            actual_reward=actual_reward,
            rpe=rpe,
        )

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def compute_composite_reward(self, sources: dict[str, float]) -> float:
        """Compute a weighted composite reward from named source signals.

        Each source key is matched against the configured weights
        (``task_success``, ``efficiency``, ``user_feedback``, ``goal_progress``).
        Unrecognised keys are silently skipped.  The result is clamped to
        ``[-1, 1]`` to prevent runaway value estimates.

        Parameters
        ----------
        sources:
            A dict mapping source name to raw signal value, e.g.
            ``{"task_success": 0.8, "efficiency": 0.5, "user_feedback": 1.0}``.

        Returns
        -------
        float
            Weighted sum of recognised sources, normalised to ``[-1, 1]``.
        """
        weights = self._config.source_weights
        weight_map: dict[str, float] = {
            "task_success": weights.task_success,
            "efficiency": weights.efficiency,
            "user_feedback": weights.user_feedback,
            "goal_progress": weights.goal_progress,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for source_key, signal_value in sources.items():
            w = weight_map.get(source_key)
            if w is None:
                logger.debug("Skipping unknown reward source key: %r", source_key)
                continue
            weighted_sum += w * signal_value
            total_weight += w

        if total_weight == 0.0:
            logger.debug("compute_composite_reward: no recognised source keys, returning 0.0")
            return 0.0

        # Normalise by total applied weight so partial source sets remain in range
        normalised = weighted_sum / total_weight

        # Clamp to [-1, 1]
        result = max(-1.0, min(1.0, normalised))
        logger.debug(
            "Composite reward: weighted_sum=%.4f total_weight=%.4f result=%.4f",
            weighted_sum,
            total_weight,
            result,
        )
        return result

    def get_stats(self) -> dict[str, float]:
        """Return running statistics over all computed RPE signals.

        Returns
        -------
        dict[str, float]
            Keys: ``total_positive_rpe``, ``total_negative_rpe``,
            ``cycle_count``, ``mean_rpe``, ``net_rpe``.
        """
        net_rpe = self._total_positive_rpe + self._total_negative_rpe
        mean_rpe = net_rpe / self._cycle_count if self._cycle_count > 0 else 0.0
        return {
            "total_positive_rpe": self._total_positive_rpe,
            "total_negative_rpe": self._total_negative_rpe,
            "cycle_count": float(self._cycle_count),
            "mean_rpe": mean_rpe,
            "net_rpe": net_rpe,
        }
