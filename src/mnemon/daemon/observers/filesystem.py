"""
FileSystemObserver — watches directories for changes and feeds them as percepts.

Brain analog: Visual cortex peripheral detection — notices movement and
change in the visual field without requiring focused attention. A file
appearing or changing is like peripheral motion: the sensory buffer
registers it, and the attention gate decides whether it warrants focus.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anyio

from mnemon.core.models import Modality
from mnemon.daemon.observers import ObserverPlugin

logger = logging.getLogger(__name__)


class FileSystemObserver(ObserverPlugin):
    """Watches file system paths for changes and injects percepts.

    Uses polling via anyio (no external dependency). For production use,
    install ``watchfiles`` for efficient OS-level notifications.
    """

    def __init__(self, paths: list[str] | None = None) -> None:
        self._paths = [Path(p).expanduser().resolve() for p in (paths or ["."])]
        self._brain: Any = None
        self._running = False
        self._snapshot: dict[Path, float] = {}

    @property
    def name(self) -> str:
        return "filesystem"

    async def start(self, brain: Any) -> None:
        self._brain = brain
        self._running = True
        # Take initial snapshot
        self._snapshot = self._scan()
        logger.info(
            "FileSystemObserver started — watching %d path(s), %d files tracked",
            len(self._paths),
            len(self._snapshot),
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("FileSystemObserver stopped.")

    def is_running(self) -> bool:
        return self._running

    async def run(self) -> None:
        """Polling loop: scan for changes every 5 seconds."""
        while self._running:
            await anyio.sleep(5.0)
            if not self._running:
                break

            try:
                current = self._scan()
                changes = self._diff(self._snapshot, current)
                self._snapshot = current

                for change_type, path in changes:
                    await self._emit_percept(change_type, path)
            except Exception:
                logger.exception("FileSystemObserver scan error.")

    def _scan(self) -> dict[Path, float]:
        """Return {path: mtime} for all files under watched directories."""
        result: dict[Path, float] = {}
        for root in self._paths:
            if not root.exists():
                continue
            if root.is_file():
                try:
                    result[root] = root.stat().st_mtime
                except OSError:
                    pass
                continue
            try:
                for p in root.rglob("*"):
                    if p.is_file() and not any(
                        part.startswith(".") for part in p.parts
                    ):
                        try:
                            result[p] = p.stat().st_mtime
                        except OSError:
                            pass
            except PermissionError:
                pass
        return result

    @staticmethod
    def _diff(
        old: dict[Path, float], new: dict[Path, float]
    ) -> list[tuple[str, Path]]:
        """Compare snapshots and return list of (change_type, path)."""
        changes: list[tuple[str, Path]] = []
        for path, mtime in new.items():
            if path not in old:
                changes.append(("created", path))
            elif old[path] != mtime:
                changes.append(("modified", path))
        for path in old:
            if path not in new:
                changes.append(("deleted", path))
        return changes

    async def _emit_percept(self, change_type: str, path: Path) -> None:
        """Feed a file change event into the brain's sensory buffer."""
        if self._brain is None:
            return

        description = f"[filesystem] {change_type}: {path}"
        logger.debug("FileSystemObserver percept: %s", description)

        try:
            await self._brain.memory.sensory.process(
                description, modality=Modality.STRUCTURED_DATA
            )
        except Exception:
            logger.exception("Failed to emit filesystem percept for %s", path)
