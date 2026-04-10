"""
CronObserver — generates time-awareness percepts on a schedule.

Brain analog: Suprachiasmatic nucleus (SCN) — the brain's master clock that
generates circadian timing signals. Without time awareness, the brain cannot
plan, anticipate, or track how long it has been since events occurred. The
CronObserver injects temporal context so the daemon knows what time it is,
how long since the last user interaction, and can trigger time-based behaviours.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import anyio

from mnemon.core.models import Modality
from mnemon.daemon.observers import ObserverPlugin

logger = logging.getLogger(__name__)


class CronObserver(ObserverPlugin):
    """Periodically injects time-awareness percepts into the sensory buffer."""

    def __init__(self, interval_s: int = 300) -> None:
        self._interval_s = interval_s
        self._brain: Any = None
        self._running = False

    @property
    def name(self) -> str:
        return "cron"

    async def start(self, brain: Any) -> None:
        self._brain = brain
        self._running = True
        logger.info("CronObserver started — interval=%ds", self._interval_s)

    async def stop(self) -> None:
        self._running = False
        logger.info("CronObserver stopped.")

    def is_running(self) -> bool:
        return self._running

    async def run(self) -> None:
        """Emit a time percept every interval_s seconds."""
        while self._running:
            await anyio.sleep(self._interval_s)
            if not self._running:
                break

            try:
                await self._emit_time_percept()
            except Exception:
                logger.exception("CronObserver tick error.")

    async def _emit_time_percept(self) -> None:
        """Generate and inject a temporal awareness percept."""
        if self._brain is None:
            return

        now = datetime.now(UTC)
        local_time = now.astimezone()

        description = (
            f"[time] Current time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"({local_time.strftime('%A')})"
        )

        logger.debug("CronObserver percept: %s", description)
        try:
            await self._brain.memory.sensory.process(
                description, modality=Modality.STRUCTURED_DATA
            )
        except Exception:
            logger.exception("Failed to emit time percept.")
