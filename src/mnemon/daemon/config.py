"""
Daemon configuration — settings for the always-on Jarvis layer.

Brain analog: The hypothalamus — sets the operating parameters (circadian
rhythm, arousal levels, homeostatic set-points) that govern the brain's
background processes. Just as hypothalamic settings determine when the brain
consolidates memories (sleep cycles), explores (curiosity drive), or conserves
energy (idle states), DaemonConfig controls when and how aggressively the
daemon thinks, observes, and acts.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AutonomyLevel(StrEnum):
    """Permission tier governing what the daemon may do without user approval.

    Brain analog: Dorsal ACC arousal gating — higher arousal states permit
    more automatic action; lower states require deliberate executive approval.
    """

    PASSIVE = "passive"
    SUGGEST = "suggest"
    SEMI_AUTO = "semi_auto"
    AUTONOMOUS = "autonomous"


class RiskLevel(StrEnum):
    """Risk classification for proposed daemon actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class IdleLoopConfig(BaseSettings):
    """Controls the background thinking loop (resting-state network analog).

    Weights are relative probabilities for selecting each idle activity on a
    given tick — normalised internally, so only the ratios matter.

    Priority philosophy (mirrors a living person's thinking hierarchy):
      1. help_master  — Where is the user stuck? What do they need? (highest)
      2. know_master  — Who are they as a person? What drives them?
      3. grow         — Who am I? What have I learned? What should I learn?
      4. consolidate  — Memory maintenance (background)
      5. explore      — Knowledge graph maintenance (background)
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__DAEMON__IDLE__", env_nested_delimiter="__"
    )

    tick_interval_s: float = Field(
        default=30.0,
        ge=5.0,
        description="Seconds between idle thinking ticks.",
    )
    # Priority 1: Goal/task thinking — where is master stuck, what to do next
    help_master_weight: float = Field(
        default=0.4, ge=0.0, description="Relative probability of goal-directed thinking toward master's needs."
    )
    # Priority 2: Understanding the master as a person
    know_master_weight: float = Field(
        default=0.25, ge=0.0, description="Relative probability of deepening understanding of the master."
    )
    # Priority 3: Self-development — who am I, what have I learned
    grow_weight: float = Field(
        default=0.2, ge=0.0, description="Relative probability of self-reflection and growth."
    )
    # Background: memory maintenance
    consolidation_weight: float = Field(
        default=0.1, ge=0.0, description="Relative probability of episodic consolidation."
    )
    exploration_weight: float = Field(
        default=0.05, ge=0.0, description="Relative probability of knowledge graph exploration."
    )
    max_idle_cycles_per_hour: int = Field(
        default=60,
        ge=1,
        description="Hard cap to limit LLM costs during idle thinking.",
    )


class WebSourceConfig(BaseSettings):
    """Config for a single web learning source."""

    model_config = SettingsConfigDict(env_prefix="", env_nested_delimiter="__")

    url: str = Field(description="URL of the RSS feed or web page.")
    name: str = Field(default="", description="Human-readable label for this source.")
    kind: str = Field(default="rss", description="Feed type: 'rss', 'atom', or 'url'.")
    interval_s: int = Field(default=3600, ge=60, description="Fetch interval in seconds.")


class ObserverConfig(BaseSettings):
    """Controls which environment observers are active."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__DAEMON__OBSERVERS__", env_nested_delimiter="__"
    )

    filesystem_enabled: bool = Field(default=True, description="Watch filesystem for changes.")
    filesystem_paths: list[str] = Field(
        default_factory=list,
        description="Paths to watch. Empty = current working directory.",
    )
    clipboard_enabled: bool = Field(default=False, description="Monitor clipboard changes.")
    cron_enabled: bool = Field(
        default=True, description="Generate time-awareness percepts."
    )
    cron_interval_s: int = Field(
        default=300, ge=30, description="Seconds between time-awareness percepts."
    )
    web_learning_enabled: bool = Field(
        default=True,
        description="Enable automatic web content ingestion.",
    )
    web_learning_sources: list[WebSourceConfig] = Field(
        default_factory=list,
        description="Additional web sources beyond the built-in defaults.",
    )
    web_learning_use_defaults: bool = Field(
        default=False,
        description="Include built-in starter sources (HN, arXiv, O'Reilly).",
    )


class IPCConfig(BaseSettings):
    """IPC server settings for CLI ↔ daemon communication."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__DAEMON__IPC__", env_nested_delimiter="__"
    )

    socket_path: str = Field(
        default="~/.mnemon/daemon.sock",
        description="Unix domain socket path for JSON-RPC IPC.",
    )
    max_message_size: int = Field(
        default=1_048_576, ge=4096, description="Maximum IPC message size in bytes."
    )


# ---------------------------------------------------------------------------
# Top-level daemon config
# ---------------------------------------------------------------------------


class DaemonConfig(BaseSettings):
    """Root configuration for the Mnemon daemon (Jarvis layer).

    Composes idle loop, observer, and IPC settings. Loaded from environment
    variables with the ``MNEMON__DAEMON__`` prefix or programmatically.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__DAEMON__", env_nested_delimiter="__"
    )

    pidfile: str = Field(
        default="~/.mnemon/daemon.pid",
        description="Path to PID file for daemon process management.",
    )
    state_dir: str = Field(
        default="~/.mnemon/state/",
        description="Directory for persisted daemon state (goals, thoughts, etc.).",
    )
    log_file: str = Field(
        default="~/.mnemon/daemon.log",
        description="Daemon log file path.",
    )
    autonomy_level: AutonomyLevel = Field(
        default=AutonomyLevel.SUGGEST,
        description="Default permission level for autonomous actions.",
    )
    auto_restart: bool = Field(
        default=True,
        description="Automatically restart the daemon on crash.",
    )
    max_restart_attempts: int = Field(
        default=5, ge=1, description="Maximum consecutive restart attempts."
    )
    git_journal_enabled: bool = Field(
        default=True,
        description="Commit identity files (soul.md, master.md, learnings.md) and state to a local git repo.",
    )
    git_journal_interval_s: int = Field(
        default=600, ge=60,
        description="Seconds between automatic git commits of the state directory.",
    )
    webui_enabled: bool = Field(
        default=True,
        description="Serve the web dashboard (chat + stats) on the local network.",
    )
    webui_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the web UI — 0.0.0.0 makes it reachable on LAN (e.g. from phone).",
    )
    webui_port: int = Field(
        default=7777, ge=1024, le=65535,
        description="Port for the web dashboard.",
    )

    idle_loop: IdleLoopConfig = Field(default_factory=IdleLoopConfig)
    observers: ObserverConfig = Field(default_factory=ObserverConfig)
    ipc: IPCConfig = Field(default_factory=IPCConfig)

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir).expanduser()

    @property
    def pid_path(self) -> Path:
        return Path(self.pidfile).expanduser()

    @property
    def log_path(self) -> Path:
        return Path(self.log_file).expanduser()

    @property
    def socket_path(self) -> Path:
        return Path(self.ipc.socket_path).expanduser()
