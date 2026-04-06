"""
Observer plugin system — the daemon's sensory organs.

Brain analog: Peripheral sensory receptors (retinal ganglion cells, cochlear
hair cells, mechanoreceptors) that continuously transduce environmental
stimuli into neural signals without requiring conscious attention. Observers
feed raw percepts into the SensoryBuffer where the attention gate decides
what warrants higher processing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ObserverPlugin(ABC):
    """Base class for environment observation plugins.

    Observers watch external state and produce percepts that are fed into
    the Mnemon SensoryBuffer, exactly as if a user had typed input.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this observer."""

    @abstractmethod
    async def start(self, brain: Any) -> None:
        """Begin observing. Called once when daemon starts."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop observing. Called on daemon shutdown."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return True if the observer is actively watching."""


class ObserverRegistry:
    """Discovers and manages observer plugin instances."""

    def __init__(self) -> None:
        self._observers: dict[str, ObserverPlugin] = {}

    def register(self, observer: ObserverPlugin) -> None:
        """Register an observer plugin by its name."""
        self._observers[observer.name] = observer
        logger.debug("Observer registered: %s", observer.name)

    def get(self, name: str) -> ObserverPlugin | None:
        return self._observers.get(name)

    def all(self) -> list[ObserverPlugin]:
        return list(self._observers.values())

    def running(self) -> list[ObserverPlugin]:
        return [o for o in self._observers.values() if o.is_running()]
