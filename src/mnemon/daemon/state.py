"""
DaemonState — persistent runtime state for the Jarvis daemon.

Brain analog: The reticular activating system's tonic state — the background
arousal level, accumulated fatigue, and homeostatic variables that persist
across individual cognitive cycles. When the brain "reboots" from sleep, it
resumes from this baseline rather than starting from scratch.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ThoughtEntry(BaseModel):
    """A single recorded idle thinking result."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activity: str = Field(description="Type of idle activity: consolidation, reflection, planning, exploration.")
    summary: str = Field(description="Brief summary of what was thought/discovered.")
    details: dict[str, Any] = Field(default_factory=dict)


class ProactiveMessage(BaseModel):
    """A message Jarvis wants to share with the user unprompted.

    Generated during idle thinking when the brain produces a thought it
    considers worth surfacing — a connection, question, or insight that
    arose spontaneously. This is the agent initiating conversation rather
    than waiting to be asked.
    """

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_activity: str = Field(description="Which idle activity generated this (reflection/wandering/planning).")
    content: str = Field(description="The message Jarvis wants to share.")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    read: bool = Field(default=False)


class ObserverStats(BaseModel):
    """Cumulative statistics for a single observer plugin."""

    events_observed: int = 0
    last_event_at: datetime | None = None


class DaemonState(BaseModel):
    """Serialisable snapshot of daemon runtime state.

    Persisted to disk periodically and on shutdown. Loaded on restart
    to restore continuity — the daemon picks up where it left off.
    """

    daemon_started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_cycles: int = 0
    total_idle_ticks: int = 0
    last_user_interaction: datetime | None = None
    last_consolidation: datetime | None = None
    recent_thoughts: list[ThoughtEntry] = Field(default_factory=list)
    observer_stats: dict[str, ObserverStats] = Field(default_factory=dict)
    pending_approvals: list[dict[str, Any]] = Field(default_factory=list)
    proactive_inbox: list[ProactiveMessage] = Field(default_factory=list)
    pending_curiosity_question: str | None = None

    # Ring buffer limits
    _MAX_THOUGHTS: int = 100
    _MAX_INBOX: int = 50

    def add_thought(self, entry: ThoughtEntry) -> None:
        """Append a thought, evicting the oldest if at capacity."""
        self.recent_thoughts.append(entry)
        if len(self.recent_thoughts) > self._MAX_THOUGHTS:
            self.recent_thoughts = self.recent_thoughts[-self._MAX_THOUGHTS:]

    def add_proactive_message(self, msg: ProactiveMessage) -> None:
        """Push a proactive message to inbox, evicting oldest if at capacity."""
        self.proactive_inbox.append(msg)
        if len(self.proactive_inbox) > self._MAX_INBOX:
            self.proactive_inbox = self.proactive_inbox[-self._MAX_INBOX:]

    def get_unread_inbox(self) -> list[ProactiveMessage]:
        """Return unread proactive messages, oldest first."""
        return [m for m in self.proactive_inbox if not m.read]

    def mark_inbox_read(self, message_id: str | None = None) -> int:
        """Mark message(s) as read. If message_id is None, marks all. Returns count marked."""
        count = 0
        for m in self.proactive_inbox:
            if not m.read and (message_id is None or m.id == message_id):
                m.read = True
                count += 1
        return count

    def record_observer_event(self, observer_name: str) -> None:
        """Increment event counter for an observer."""
        if observer_name not in self.observer_stats:
            self.observer_stats[observer_name] = ObserverStats()
        stats = self.observer_stats[observer_name]
        stats.events_observed += 1
        stats.last_event_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_STATE_FILENAME = "daemon_state.json"


def save_state(state: DaemonState, state_dir: Path) -> None:
    """Persist daemon state to JSON file."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _STATE_FILENAME
    data = state.model_dump_json(indent=2)
    path.write_text(data, encoding="utf-8")
    logger.debug("Daemon state saved to %s", path)


def load_state(state_dir: Path) -> DaemonState:
    """Load daemon state from JSON file, or return a fresh state."""
    path = state_dir / _STATE_FILENAME
    if not path.exists():
        logger.info("No existing daemon state at %s — starting fresh.", path)
        return DaemonState()
    try:
        raw = path.read_text(encoding="utf-8")
        state = DaemonState.model_validate_json(raw)
        logger.info(
            "Loaded daemon state: %d cycles, %d thoughts, started %s",
            state.total_cycles,
            len(state.recent_thoughts),
            state.daemon_started_at,
        )
        return state
    except Exception:
        logger.warning(
            "Failed to load daemon state from %s — starting fresh.",
            path,
            exc_info=True,
        )
        return DaemonState()
