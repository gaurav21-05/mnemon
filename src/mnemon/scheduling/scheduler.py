"""
ConsolidationScheduler — automatic consolidation triggering.

Brain analog: The circadian rhythm controller (suprachiasmatic nucleus) that
triggers slow-wave sleep consolidation at regular intervals, plus reactive
triggers when the hippocampus signals memory buffer saturation.

Uses APScheduler 4.x for async-native scheduling with cron, interval, and
threshold-based triggers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mnemon.core.config import ConsolidationScheduleConfig

if TYPE_CHECKING:
    from mnemon.core.interfaces import ConsolidationEngineInterface

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional APScheduler 4.x import
# ---------------------------------------------------------------------------

try:
    from apscheduler import AsyncScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    AsyncScheduler = None  # type: ignore[assignment,misc]
    CronTrigger = None  # type: ignore[assignment,misc]
    IntervalTrigger = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# ConsolidationScheduler
# ---------------------------------------------------------------------------


class ConsolidationScheduler:
    """Wraps APScheduler 4.x to drive automatic consolidation cycles.

    Brain analog: The suprachiasmatic nucleus — the master circadian clock
    that coordinates when the brain transitions into consolidation-heavy
    slow-wave sleep states.

    Parameters
    ----------
    config:
        Scheduling configuration slice from :class:`~mnemon.core.config.ConsolidationScheduleConfig`.
    consolidation_engine:
        The engine whose :meth:`run_cycle` will be invoked on each tick.
    """

    def __init__(
        self,
        config: ConsolidationScheduleConfig,
        consolidation_engine: ConsolidationEngineInterface,
    ) -> None:
        self._config = config
        self._engine = consolidation_engine
        self._scheduler: Any = None
        self._job_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Add the consolidation job and start the scheduler.

        Trigger selection is driven by ``config.mode``:

        * ``"periodic"`` — fires every ``config.periodic_interval_s`` seconds.
        * ``"cron"`` — fires once per day at midnight (00:00 UTC).
        * ``"idle"``, ``"threshold"``, ``"manual"`` — application-driven;
          a periodic fallback is still registered as a safety net.

        Raises
        ------
        RuntimeError
            If APScheduler is not installed.
        """
        if not _APSCHEDULER_AVAILABLE:
            raise RuntimeError(
                "APScheduler 4.x is required for automatic consolidation scheduling. "
                "Install it with: pip install 'mnemon[scheduler]'  "
                "or: pip install 'apscheduler>=4.0'"
            )

        self._scheduler = AsyncScheduler()

        mode = self._config.mode
        if mode == "periodic":
            trigger = IntervalTrigger(seconds=self._config.periodic_interval_s)
            logger.info(
                "ConsolidationScheduler: periodic trigger every %d s.",
                self._config.periodic_interval_s,
            )
        elif mode == "cron":
            trigger = CronTrigger(hour=0, minute=0)
            logger.info("ConsolidationScheduler: cron trigger at midnight UTC.")
        elif mode in ("idle", "threshold", "manual"):
            # These modes are handled outside the scheduler (application-level
            # logic calls trigger_now()). Register a long-interval fallback so
            # the scheduler is still usable as a safety net.
            trigger = IntervalTrigger(seconds=self._config.periodic_interval_s)
            logger.info(
                "ConsolidationScheduler: mode=%r — using periodic fallback "
                "interval %d s (primary trigger is application-driven).",
                mode,
                self._config.periodic_interval_s,
            )
        else:
            logger.warning(
                "ConsolidationScheduler: unknown mode %r — falling back to "
                "periodic interval (%d s).",
                mode,
                self._config.periodic_interval_s,
            )
            trigger = IntervalTrigger(seconds=self._config.periodic_interval_s)

        await self._scheduler.__aenter__()
        self._job_id = await self._scheduler.add_schedule(
            self._run_consolidation,
            trigger,
            id="mnemon_consolidation",
        )
        logger.info(
            "ConsolidationScheduler started (job_id=%s, mode=%s).",
            self._job_id,
            mode,
        )

    async def stop(self) -> None:
        """Shut down the scheduler gracefully.

        Safe to call even if :meth:`start` was never called or if
        APScheduler is not installed — both are no-ops.
        """
        if self._scheduler is None:
            return
        try:
            await self._scheduler.__aexit__(None, None, None)
            logger.info("ConsolidationScheduler stopped.")
        except Exception as exc:
            logger.warning("ConsolidationScheduler stop error (suppressed): %s", exc)

    # ------------------------------------------------------------------
    # Job callback
    # ------------------------------------------------------------------

    async def _run_consolidation(self) -> None:
        """APScheduler job callback — execute one consolidation cycle.

        Exceptions from the engine are caught and logged so a single
        failing cycle cannot crash the scheduler loop.
        """
        logger.debug("ConsolidationScheduler: firing consolidation cycle.")
        try:
            result = await self._engine.run_cycle()
            logger.info(
                "ConsolidationScheduler: cycle complete — %s",
                result,
            )
        except Exception as exc:
            logger.error(
                "ConsolidationScheduler: cycle raised an exception: %s",
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Manual trigger
    # ------------------------------------------------------------------

    async def trigger_now(self) -> None:
        """Manually fire a consolidation cycle outside the regular schedule.

        Runs the consolidation job callback directly as a one-shot
        coroutine, bypassing the APScheduler job queue.  Useful for
        threshold-based triggering from application code.
        """
        logger.info("ConsolidationScheduler: manual trigger_now() called.")
        await self._run_consolidation()
