"""
DaemonProcess — OS-level daemon lifecycle management.

Brain analog: The brainstem — keeps the organism alive (heartbeat, respiration)
regardless of what higher cognitive functions are doing. It handles startup,
monitors vital signs, triggers auto-restart on failure, and manages the
graceful shutdown sequence. Without the brainstem, the brain dies; without
the DaemonProcess, the cognitive loop stops.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from mnemon.core.config import MnemonConfig
from mnemon.core.exceptions import ConfigError

if TYPE_CHECKING:
    from mnemon.daemon.config import DaemonConfig

logger = logging.getLogger(__name__)


class DaemonProcess:
    """OS-level daemon lifecycle: PID files, signals, daemonization, auto-restart."""

    def __init__(
        self,
        daemon_config: DaemonConfig,
        mnemon_config: MnemonConfig | None = None,
    ) -> None:
        self._daemon_config = daemon_config
        self._mnemon_config = mnemon_config or MnemonConfig()
        self._shutdown_requested = False
        self._active_daemon: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, foreground: bool = False) -> None:
        """Entry point: start the daemon.

        Parameters
        ----------
        foreground:
            If True, run in the current terminal (for development/systemd).
            If False, fork into background as a true daemon.
        """
        # Load ~/.mnemon/.env before anything else so tokens are available
        self._load_env_file()

        # Ensure only one daemon instance runs at a time
        existing = self.status()
        if existing.get("running"):
            pid = existing.get("pid", "?")
            print(
                "Daemon already running "
                f"(pid={pid}). Stop it first with: mnemon daemon stop"
            )
            return

        if not foreground:
            self._daemonize()

        self._write_pid()
        self._setup_signals()
        self._setup_logging(foreground=foreground)

        logger.info("Daemon process starting (pid=%d, foreground=%s)", os.getpid(), foreground)

        # Run with auto-restart loop
        attempts = 0
        max_attempts = self._daemon_config.max_restart_attempts

        while attempts < max_attempts and not self._shutdown_requested:
            try:
                anyio.run(self._run)
                break  # Clean exit
            except KeyboardInterrupt:
                logger.info("Daemon interrupted by keyboard.")
                break
            except ConfigError as exc:
                # Config errors are permanent — don't retry
                logger.error("Daemon cannot start due to configuration error: %s", exc)
                print(f"\nConfiguration error: {exc}")
                print("Fix the config and try again. Auto-restart skipped for config errors.")
                break
            except Exception:
                attempts += 1
                logger.exception(
                    "Daemon crashed (attempt %d/%d). %s",
                    attempts,
                    max_attempts,
                    (
                        "Restarting..."
                        if self._daemon_config.auto_restart and attempts < max_attempts
                        else "Giving up."
                    ),
                )
                if not self._daemon_config.auto_restart:
                    break

        self._remove_pid()
        logger.info("Daemon process exited.")

    def stop(self) -> None:
        """Send SIGTERM to a running daemon identified by its PID file."""
        pid_path = self._daemon_config.pid_path
        if not pid_path.exists():
            print("No daemon PID file found. Is the daemon running?")
            return

        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to daemon (pid={pid}).")
            import time as _time
            end = _time.time() + 5.0
            while _time.time() < end:
                if not self._pid_is_live(pid):
                    pid_path.unlink(missing_ok=True)
                    print("Daemon stopped cleanly.")
                    return
                _time.sleep(0.2)

            print("Daemon did not exit after SIGTERM; sending SIGKILL...")
            os.kill(pid, signal.SIGKILL)

            hard_end = _time.time() + 2.0
            while _time.time() < hard_end:
                if not self._pid_is_live(pid):
                    pid_path.unlink(missing_ok=True)
                    print("Daemon force-stopped.")
                    return
                _time.sleep(0.1)

            print("Daemon is still present after SIGKILL. Check process state manually.")
        except ProcessLookupError:
            print("Daemon process not found. Removing stale PID file.")
            pid_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"Failed to stop daemon: {exc}")

    def status(self) -> dict[str, Any]:
        """Check if the daemon is running."""
        pid_path = self._daemon_config.pid_path
        if not pid_path.exists():
            return {"running": False, "reason": "no pid file"}

        try:
            pid = int(pid_path.read_text().strip())
            if not self._pid_is_live(pid):
                pid_path.unlink(missing_ok=True)
                return {"running": False, "reason": "stale pid file"}
            return {"running": True, "pid": pid}
        except (ProcessLookupError, ValueError):
            return {"running": False, "reason": "stale pid file"}
        except PermissionError:
            return {"running": True, "pid": pid, "note": "permission denied on check"}

    # ------------------------------------------------------------------
    # Main async loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main async loop: build brain, start all subsystems, await shutdown."""
        from mnemon.daemon import DaemonFactory
        from mnemon.daemon.config import DaemonConfig

        # Re-read config now that .env is loaded into os.environ
        # (pydantic-settings reads env vars at construction time, so we
        # must construct *after* _load_env_file has populated os.environ)
        daemon_config = DaemonConfig()

        factory = DaemonFactory(daemon_config, self._mnemon_config)
        daemon = await factory.build()
        self._active_daemon = daemon

        try:
            await daemon.run()
        finally:
            try:
                await daemon.shutdown()
            finally:
                self._active_daemon = None

    # ------------------------------------------------------------------
    # OS-level helpers
    # ------------------------------------------------------------------

    def _daemonize(self) -> None:
        """Fork into background using double-fork technique."""
        # First fork
        pid = os.fork()
        if pid > 0:
            sys.exit(0)  # Parent exits

        os.setsid()

        # Second fork
        pid = os.fork()
        if pid > 0:
            sys.exit(0)  # First child exits

        # Redirect stdio to /dev/null
        sys.stdin = os.fdopen(os.open(os.devnull, os.O_RDONLY))
        sys.stdout = os.fdopen(os.open(os.devnull, os.O_WRONLY), "w")
        sys.stderr = os.fdopen(os.open(os.devnull, os.O_WRONLY), "w")

    def _write_pid(self) -> None:
        """Write current PID to the configured PID file."""
        pid_path = self._daemon_config.pid_path
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))
        logger.debug("PID %d written to %s", os.getpid(), pid_path)

    def _remove_pid(self) -> None:
        """Remove the PID file on exit."""
        pid_path = self._daemon_config.pid_path
        pid_path.unlink(missing_ok=True)
        logger.debug("PID file removed: %s", pid_path)

    def _load_env_file(self) -> None:
        """Load ~/.mnemon/.env into os.environ if it exists."""
        env_path = Path("~/.mnemon/.env").expanduser()
        if not env_path.exists():
            return
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            logger.debug("Loaded env from ~/.mnemon/.env")
        except Exception as exc:
            logger.warning("Failed to load ~/.mnemon/.env: %s", exc)

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def _handle_term(signum: int, frame: Any) -> None:
            logger.info("Received signal %d — initiating shutdown.", signum)
            self._shutdown_requested = True
            if self._active_daemon is not None:
                try:
                    self._active_daemon.request_shutdown()
                except Exception:
                    logger.exception("Failed to request daemon shutdown from signal handler.")

        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)

    def _pid_is_live(self, pid: int) -> bool:
        """Return True when *pid* exists and is not a zombie."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False

        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists():
            return True

        try:
            raw = stat_path.read_text(encoding="utf-8")
        except Exception:
            return True

        parts = raw.split()
        return not (len(parts) >= 3 and parts[2] == "Z")

    def _setup_logging(self, foreground: bool = False) -> None:
        """Configure logging for the daemon."""
        log_path = self._daemon_config.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

        if foreground:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
