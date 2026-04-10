"""
PersistentGoalStore — SQLite-backed persistence for the GoalManager.

Brain analog: The prefrontal cortex's sustained neural firing patterns that
maintain goal representations across delays and interruptions. In biology,
these patterns are actively maintained by recurrent neural circuits. Here we
use a simpler mechanism — SQLite — to ensure goals survive daemon restarts,
just as the PFC maintains intentions across sleep/wake transitions.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from mnemon.core.models import Goal

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

logger = logging.getLogger(__name__)

_GOALS_FILENAME = "goals.json"


class PersistentGoalStore:
    """JSON-file-backed persistence for goals.

    On daemon startup: load goals from disk and inject into GoalManager.
    On daemon shutdown (and periodically): flush GoalManager goals to disk.

    Uses a simple JSON file rather than SQLite to avoid the aiosqlite
    dependency for the daemon layer. Can be upgraded to SQLite later.
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._path = state_dir / _GOALS_FILENAME

    def load(self) -> dict[UUID, Goal]:
        """Load goals from disk. Returns an empty dict if no file exists."""
        if not self._path.exists():
            logger.info("No persisted goals at %s — starting with empty goal store.", self._path)
            return {}

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            goals: dict[UUID, Goal] = {}
            for item in data:
                goal = Goal.model_validate(item)
                goals[goal.id] = goal
            logger.info("Loaded %d persisted goals from %s", len(goals), self._path)
            return goals
        except Exception:
            logger.warning(
                "Failed to load goals from %s — starting fresh.",
                self._path,
                exc_info=True,
            )
            return {}

    def save(self, goals: dict[UUID, Goal]) -> int:
        """Persist all goals to disk. Returns count saved."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = [goal.model_dump(mode="json") for goal in goals.values()]
        self._path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Persisted %d goals to %s", len(data), self._path)
        return len(data)

    def inject_into_manager(self, goal_manager: Any, goals: dict[UUID, Goal]) -> int:
        """Inject loaded goals into a GoalManager's internal store.

        Accesses ``_goals`` directly — this is an internal integration
        point, not a public API.
        """
        if not goals:
            return 0
        goal_manager._goals.update(goals)
        logger.info("Injected %d goals into GoalManager", len(goals))
        return len(goals)

    def sync_from_manager(self, goal_manager: Any) -> int:
        """Read current goals from GoalManager and persist them."""
        goals: dict[UUID, Goal] = dict(goal_manager._goals)
        return self.save(goals)
