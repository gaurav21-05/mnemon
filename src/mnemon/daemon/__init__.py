"""
Mnemon Daemon — the always-on Jarvis layer.

Brain analog: The complete organism — not just the brain (Mnemon) but the
body, senses, and autonomic nervous system that keep it alive and connected
to the world. The daemon wraps the Mnemon cognitive framework with:

  - An idle thinking loop (Default Mode Network)
  - Environment observers (peripheral sensory systems)
  - Persistent goal pursuit (prefrontal sustained firing)
  - An IPC interface (thalamocortical gateway)
  - Autonomy controls (orbitofrontal risk gating)
  - State persistence (brainstem homeostatic memory)

Usage::

    from mnemon.daemon import DaemonFactory
    from mnemon.daemon.config import DaemonConfig

    daemon = await DaemonFactory(DaemonConfig()).build()
    await daemon.run()     # Runs until shutdown signal
    await daemon.shutdown()
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import subprocess

import anyio

from mnemon.core.config import MnemonConfig
from mnemon.daemon.autonomy import AutonomyController
from mnemon.daemon.config import DaemonConfig
from mnemon.daemon.goals.persistent_store import PersistentGoalStore
from mnemon.daemon.ipc import DaemonIPCServer
from mnemon.daemon.loop import IdleThinkingLoop
from mnemon.daemon.observers import ObserverPlugin, ObserverRegistry
from mnemon.daemon.observers.cron import CronObserver
from mnemon.daemon.observers.filesystem import FileSystemObserver
from mnemon.daemon.observers.web import WebLearningObserver, WebSource
from mnemon.daemon.state import DaemonState, load_state, save_state

logger = logging.getLogger(__name__)


class JarvisDaemon:
    """The assembled daemon: Mnemon brain + idle loop + observers + IPC.

    This is the daemon-layer equivalent of the Mnemon facade. It holds
    references to all daemon components and manages their lifecycle.

    Brain analog: The whole organism — brain (Mnemon) embedded in a body
    (daemon) with senses (observers), drives (idle loop), and communication
    (IPC).
    """

    def __init__(
        self,
        brain: Any,  # Mnemon instance
        config: DaemonConfig,
        idle_loop: IdleThinkingLoop,
        observers: list[ObserverPlugin],
        autonomy: AutonomyController,
        goal_store: PersistentGoalStore,
        ipc_server: DaemonIPCServer,
        state: DaemonState,
    ) -> None:
        self.brain = brain
        self.config = config
        self.idle_loop = idle_loop
        self.observers = observers
        self.autonomy = autonomy
        self.goal_store = goal_store
        self.ipc_server = ipc_server
        self.state = state
        self._shutdown_event: anyio.Event | None = None

    async def run(self) -> None:
        """Main daemon loop. Runs until shutdown signal.

        Enters the Mnemon async context (starts CognitiveBus), then launches
        all subsystems as concurrent tasks within a single TaskGroup.
        """
        self._shutdown_event = anyio.Event()

        async with self.brain:
            logger.info("Mnemon brain online — starting daemon subsystems.")

            async with anyio.create_task_group() as tg:
                # Start IPC server
                await self.ipc_server.start(tg)

                # Start observers
                for observer in self.observers:
                    try:
                        await observer.start(self.brain)
                        tg.start_soon(observer.run)
                        logger.info("Observer started: %s", observer.name)
                    except Exception:
                        logger.exception("Failed to start observer: %s", observer.name)

                # Start idle thinking loop
                tg.start_soon(self.idle_loop.run)

                # Start periodic state persistence
                tg.start_soon(self._periodic_save)

                # Start git journal if enabled
                if self.config.git_journal_enabled:
                    self._init_git_journal()
                    tg.start_soon(self._periodic_git_commit)

                logger.info(
                    "Daemon fully operational — %d observers, idle loop active, "
                    "autonomy=%s",
                    len(self.observers),
                    self.autonomy.level,
                )

                # Wait for shutdown signal
                await self._shutdown_event.wait()

                # Cancel all tasks
                tg.cancel_scope.cancel()

    async def shutdown(self) -> None:
        """Gracefully shut down all daemon components."""
        logger.info("Daemon shutdown initiated.")

        # Stop idle loop
        self.idle_loop.stop()

        # Stop observers
        for observer in self.observers:
            try:
                await observer.stop()
            except Exception:
                logger.exception("Error stopping observer: %s", observer.name)

        # Stop IPC
        await self.ipc_server.stop()

        # Persist final state
        self._save_state()

        # Persist goals
        self.goal_store.sync_from_manager(self.brain.control.goals)

        # Final git commit
        if self.config.git_journal_enabled:
            try:
                self._git_commit_state()
            except Exception:
                logger.warning("Final git commit failed.", exc_info=True)

        # Close brain
        await self.brain.close()

        logger.info("Daemon shutdown complete.")

    def request_shutdown(self) -> None:
        """Signal the daemon to shut down (called from IPC or signal handler)."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    async def _periodic_save(self) -> None:
        """Periodically persist state and goals to disk."""
        while True:
            await anyio.sleep(60.0)  # Save every 60 seconds
            try:
                self._save_state()
                self.goal_store.sync_from_manager(self.brain.control.goals)
            except Exception:
                logger.exception("Periodic state save failed.")

    def _save_state(self) -> None:
        """Save daemon state to disk."""
        save_state(self.state, self.config.state_path)

    def _init_git_journal(self) -> None:
        """Initialize a git repo in the state directory if one doesn't exist."""
        state_dir = self.config.state_path
        state_dir.mkdir(parents=True, exist_ok=True)
        git_dir = state_dir / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=state_dir,
                    capture_output=True,
                    check=True,
                )
                # Write a .gitignore so only identity + state files are tracked
                gitignore = state_dir / ".gitignore"
                gitignore.write_text("*.sock\n*.pid\n*.log\n*.tmp\n", encoding="utf-8")
                logger.info("Initialized git journal at %s", state_dir)
            except Exception:
                logger.warning("Failed to initialize git journal.", exc_info=True)

    async def _periodic_git_commit(self) -> None:
        """Periodically commit changes to the state directory git repo.

        Tracks soul.md, master.md, learnings.md, daemon_state.json, and
        goals.json — the living record of who Jarvis is and what it knows.
        Commit message summarises the most recent idle thought so the log
        is readable as a journal.
        """
        while True:
            await anyio.sleep(self.config.git_journal_interval_s)
            try:
                self._git_commit_state()
            except Exception:
                logger.exception("Periodic git commit failed.")

    def _git_commit_state(self) -> None:
        """Stage and commit any changes in the state directory."""
        state_dir = self.config.state_path
        git_dir = state_dir / ".git"
        if not git_dir.exists():
            return

        # Stage all tracked file types
        files_to_track = [
            "soul.md", "master.md", "learnings.md",
            "daemon_state.json", "goals.json", ".gitignore",
        ]
        existing = [f for f in files_to_track if (state_dir / f).exists()]
        if not existing:
            return

        try:
            subprocess.run(
                ["git", "add"] + existing,
                cwd=state_dir,
                capture_output=True,
                check=True,
            )
        except Exception:
            logger.warning("git add failed in journal.", exc_info=True)
            return

        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=state_dir,
            capture_output=True,
        )
        if status.returncode == 0:
            return  # Nothing staged

        # Build a commit message from the most recent thought
        msg = "Jarvis: journal update"
        if self.state.recent_thoughts:
            last = self.state.recent_thoughts[-1]
            activity_label = {
                "help_master": "thought about helping master",
                "know_master": "learned about master",
                "grow": "reflected on self",
                "consolidation": "consolidated memories",
                "exploration": "explored knowledge graph",
            }.get(last.activity, last.activity)
            summary = last.summary[:80].rstrip(".")
            msg = f"Jarvis: {activity_label} — {summary}"

        try:
            subprocess.run(
                ["git", "commit", "--no-gpg-sign", "-m", msg],
                cwd=state_dir,
                capture_output=True,
                check=True,
                env={**__import__("os").environ, "GIT_AUTHOR_NAME": "Jarvis", "GIT_AUTHOR_EMAIL": "jarvis@local", "GIT_COMMITTER_NAME": "Jarvis", "GIT_COMMITTER_EMAIL": "jarvis@local"},
            )
            logger.info("Git journal committed: %s", msg[:80])
        except subprocess.CalledProcessError as exc:
            logger.warning("git commit failed: %s", exc.stderr.decode()[:200] if exc.stderr else str(exc))


class DaemonFactory:
    """Builds a JarvisDaemon from configuration.

    Mirrors the MnemonFactory pattern: reads config, instantiates all
    components with their dependencies, and returns a fully wired daemon.

    Brain analog: Embryonic development of the whole organism — not just
    growing the brain (MnemonFactory's job) but also the sensory organs,
    autonomic nervous system, and body plan.
    """

    def __init__(
        self,
        daemon_config: DaemonConfig | None = None,
        mnemon_config: MnemonConfig | None = None,
    ) -> None:
        self._daemon_config = daemon_config or DaemonConfig()
        self._mnemon_config = mnemon_config or MnemonConfig()

    async def build(self) -> JarvisDaemon:
        """Construct and return a fully-wired JarvisDaemon.

        Stages:
        1. Build Mnemon brain via MnemonFactory
        2. Load persisted state from disk
        3. Create autonomy controller
        4. Load and inject persisted goals
        5. Create idle thinking loop
        6. Create and register observers
        7. Create IPC server
        8. Assemble JarvisDaemon
        """
        from mnemon.factory import MnemonFactory

        daemon_config = self._daemon_config

        # 1. Build the brain
        # Default to in-memory backends so the daemon works out of the box
        # without requiring Qdrant, FalkorDB, etc.
        mnemon_config = self._mnemon_config
        self._apply_daemon_defaults(mnemon_config)

        logger.info("DaemonFactory: building Mnemon brain...")
        brain = await MnemonFactory(mnemon_config).build()

        # 2. Load persisted state
        state = load_state(daemon_config.state_path)

        # 3. Autonomy controller
        autonomy = AutonomyController(daemon_config)

        # 4. Goal persistence
        goal_store = PersistentGoalStore(daemon_config.state_path)
        saved_goals = goal_store.load()
        if saved_goals:
            goal_store.inject_into_manager(brain.control.goals, saved_goals)

        # 5. Idle thinking loop
        idle_loop = IdleThinkingLoop(
            brain=brain,
            config=daemon_config.idle_loop,
            state=state,
            state_dir=daemon_config.state_path,
        )

        # 6. Observers
        observers: list[ObserverPlugin] = []
        obs_config = daemon_config.observers

        if obs_config.filesystem_enabled:
            paths = obs_config.filesystem_paths or None
            observers.append(FileSystemObserver(paths=paths))

        if obs_config.cron_enabled:
            observers.append(CronObserver(interval_s=obs_config.cron_interval_s))

        if obs_config.web_learning_enabled:
            sources: list[WebSource] = []
            if obs_config.web_learning_use_defaults:
                from mnemon.daemon.observers.web import _default_sources
                sources.extend(_default_sources())
            for s in obs_config.web_learning_sources:
                sources.append(WebSource(url=s.url, name=s.name, kind=s.kind, interval_s=s.interval_s))
            observers.append(WebLearningObserver(sources=sources))

        # 7. IPC server
        ipc_server = DaemonIPCServer(
            socket_path=daemon_config.socket_path,
            brain=brain,
            state=state,
            autonomy=autonomy,
            idle_loop=idle_loop,
        )

        # 8. Assemble
        daemon = JarvisDaemon(
            brain=brain,
            config=daemon_config,
            idle_loop=idle_loop,
            observers=observers,
            autonomy=autonomy,
            goal_store=goal_store,
            ipc_server=ipc_server,
            state=state,
        )

        logger.info(
            "DaemonFactory.build() complete — "
            "%d observers, autonomy=%s, %d persisted goals",
            len(observers),
            autonomy.level,
            len(saved_goals),
        )
        return daemon

    @staticmethod
    def _apply_daemon_defaults(config: MnemonConfig) -> None:
        """Override config to use in-memory backends and local LLM if available.

        The daemon should work out of the box with zero external dependencies.
        If the user hasn't explicitly configured a backend that is available,
        fall back to the in-memory implementations. Similarly, if no API key
        is set for the default LLM provider, try to use local Ollama.
        """
        import importlib
        import os
        import shutil

        # LLM/Embedding: fall back to local Ollama if no API key is configured
        provider_name = config.llm.default_provider
        provider_cfg = config.llm.providers.get(provider_name, {})
        has_api_key = bool(
            provider_cfg.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not has_api_key and shutil.which("ollama"):
            # Prefer better models if already pulled; fall back to qwen2.5:7b
            preferred_models = ["llama3.1:8b", "llama3:8b", "mistral-nemo", "qwen2.5:7b"]
            chosen_model = "qwen2.5:7b"
            try:
                import subprocess
                result = subprocess.run(
                    ["ollama", "list"], capture_output=True, text=True, timeout=5
                )
                available = result.stdout.lower()
                for m in preferred_models:
                    if m.lower() in available:
                        chosen_model = m
                        break
            except Exception:
                pass

            config.llm.default_provider = "ollama"
            config.llm.providers["ollama"] = {
                "model": f"ollama/{chosen_model}",
                "embedding_model": "ollama/nomic-embed-text",
                "embedding_dimensions": 768,
                "api_base": "http://localhost:11434",
            }
            logger.info(
                "No API key found — using local Ollama (model=%s, embedding=nomic-embed-text).",
                chosen_model,
            )

        # Vector store: check if configured backend is importable
        backend = config.episodic.backend
        if backend == "qdrant" and not _is_importable("qdrant_client"):
            if _is_importable("hnswlib"):
                config.episodic.backend = "hnswlib"
                logger.info("Qdrant not installed — using hnswlib vector store.")
            else:
                config.episodic.backend = "memory"
                logger.info("Neither Qdrant nor hnswlib installed — using in-memory vector store.")
        elif backend == "hnswlib" and not _is_importable("hnswlib"):
            config.episodic.backend = "memory"
            logger.info("HNSWlib not installed — using in-memory vector store.")

        # Graph store: check if configured backend is importable
        graph_backend = config.semantic.graph_backend
        if graph_backend == "falkordb" and not _is_importable("falkordb"):
            if _is_importable("igraph"):
                config.semantic.graph_backend = "igraph"
                logger.info("FalkorDB not installed — using igraph graph store.")
            else:
                config.semantic.graph_backend = "memory"
                logger.info("Neither FalkorDB nor igraph installed — using in-memory graph store.")
        elif graph_backend == "neo4j" and not _is_importable("neo4j"):
            config.semantic.graph_backend = "memory"
            logger.info("Neo4j not installed — using in-memory graph store.")


def _is_importable(module_name: str) -> bool:
    """Return True if *module_name* can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
