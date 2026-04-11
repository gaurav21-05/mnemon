"""
DaemonIPCServer — JSON-RPC over Unix domain socket for CLI ↔ daemon communication.

Brain analog: The thalamocortical interface — the gateway through which
external commands (user intent) enter the cognitive system. Just as the
thalamus relays sensory input to the cortex, the IPC server relays user
commands to the appropriate daemon subsystem and returns results.

Protocol: Newline-delimited JSON-RPC 2.0 over a Unix domain socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import Counter, deque
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import anyio
import anyio.abc
from anyio import create_unix_listener

from mnemon.core.models import Episode, GoalStatus
from mnemon.daemon.autonomy import AutonomyController, ProposedAction
from mnemon.daemon.capture_policy import classify_interaction
from mnemon.daemon.config import DaemonConfig, RiskLevel
from mnemon.daemon.identity import JarvisIdentity, MasterProfile, ProfileFact
from mnemon.daemon.improve import Phase, SelfImprovementOrchestrator
from mnemon.daemon.privacy import apply_redactions, load_privacy_rules, should_exclude_text
from mnemon.daemon.reports import ReportEngine
from mnemon.daemon.scenario import ScenarioEngine

if TYPE_CHECKING:
    from mnemon.daemon.state import DaemonState

logger = logging.getLogger(__name__)

_READ_PREFIXES = (
    "read ",
    "read file ",
    "show file ",
    "show me file ",
    "open file ",
    "open ",
    "cat ",
)
_LIST_PREFIXES = (
    "ls",
    "ls ",
    "list files",
    "list folders",
    "list directory",
    "list dir",
    "show files",
    "show folders",
    "show directory",
    "what's in ",
    "what is in ",
)
_EXEC_PREFIXES = (
    "run command ",
    "execute command ",
    "run shell command ",
    "execute shell command ",
)

_TOOL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ["browse", "list", "read", "write", "exec"],
                    },
                    "task": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "command": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["tool"],
            },
        }
    },
    "required": ["steps"],
}

_TOOL_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "respond",
                "browse",
                "list",
                "read",
                "write",
                "patch",
                "verify",
                "diff",
                "git_status",
                "worktree_create",
                "worktree_remove",
                "exec",
            ],
        },
        "reply": {"type": "string"},
        "task": {"type": "string"},
        "path": {"type": "string"},
        "content": {"type": "string"},
        "search": {"type": "string"},
        "replace": {"type": "string"},
        "replace_all": {"type": "boolean"},
        "command": {"type": "string"},
        "commands": {"type": "array", "items": {"type": "string"}},
        "append": {"type": "boolean"},
        "cwd": {"type": "string"},
        "branch": {"type": "string"},
        "base_ref": {"type": "string"},
        "force": {"type": "boolean"},
    },
    "required": ["action"],
}

_AGENTIC_HINTS = (
    "build",
    "website",
    "portfolio",
    "resume",
    "design",
    "html",
    "css",
    "file",
    "folder",
    "directory",
    "repo",
    "repository",
    "workspace",
    "command",
    "shell",
    "script",
    "code",
    "python",
    "test",
    "fix",
    "implement",
    "edit",
    "patch",
    "diff",
    "verify",
    "lint",
    "typecheck",
    "worktree",
    "branch",
    "browse",
    "search",
    "look up",
    "latest",
)
_MAX_AGENT_TOOL_STEPS = 6
_GOAL_LEAD_VERBS = ("build", "make", "create", "design", "develop", "write", "ship", "start")
_EXECUTION_FOLLOWUP_PATTERNS = (
    "start now",
    "start building",
    "start building now",
    "build it",
    "build now",
    "go ahead",
    "do it",
    "continue",
    "proceed",
    "ship it",
    "start working",
)
_PROGRESS_STATUS_PATTERNS = (
    "where are you building",
    "where is this",
    "where is it",
    "where are the files",
    "where did you put",
    "what path",
    "which folder",
    "where on my local machine",
    "are you doing",
    "did you build",
    "have you built",
    "have you started",
    "did you start",
)


def _strip_wrapping_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"', "`"}:
        return cleaned[1:-1].strip()
    return cleaned


def _extract_backticked_chunks(text: str) -> list[str]:
    return [_strip_wrapping_quotes(match) for match in re.findall(r"`([^`]+)`", text)]


def _infer_workspace_intent(message: str) -> dict[str, Any] | None:
    stripped = message.strip()
    lower = stripped.lower()
    backticked = _extract_backticked_chunks(stripped)

    if lower.startswith(_READ_PREFIXES):
        path = backticked[0] if backticked else stripped.split(maxsplit=2)[-1]
        return {"tool": "read", "path": _strip_wrapping_quotes(path)}

    if lower.startswith(_LIST_PREFIXES):
        if backticked:
            path = backticked[0]
        elif " in " in stripped:
            path = stripped.split(" in ", 1)[1]
        elif lower.startswith("ls "):
            path = stripped[3:]
        else:
            path = "."
        return {"tool": "list", "path": _strip_wrapping_quotes(path or ".")}

    if lower.startswith("append to "):
        match = re.match(r"append to\s+(`[^`]+`|\S+)\s+(.*)", stripped, flags=re.IGNORECASE)
        if match:
            return {
                "tool": "write",
                "path": _strip_wrapping_quotes(match.group(1)),
                "content": match.group(2),
                "append": True,
            }

    if lower.startswith(("write ", "create file ", "create ", "save to ")):
        if lower.startswith("save to "):
            match = re.match(r"save to\s+(`[^`]+`|\S+)\s+(.*)", stripped, flags=re.IGNORECASE)
        else:
            match = re.match(
                r"(?:write|create file|create)\s+(`[^`]+`|\S+)\s+(.*)",
                stripped,
                flags=re.IGNORECASE,
            )
        if match:
            return {
                "tool": "write",
                "path": _strip_wrapping_quotes(match.group(1)),
                "content": match.group(2),
                "append": False,
            }

    if lower.startswith(_EXEC_PREFIXES):
        for prefix in _EXEC_PREFIXES:
            if lower.startswith(prefix):
                command = stripped[len(prefix) :]
                return {"tool": "exec", "command": _strip_wrapping_quotes(command)}

    if lower.startswith(("run `", "execute `")) and backticked:
        return {"tool": "exec", "command": backticked[0]}

    return None


def _sanitize_tool_steps(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for raw in raw_steps[:2]:
        tool = str(raw.get("tool", "")).strip().lower()
        if tool not in {"browse", "list", "read", "write", "exec"}:
            continue

        step: dict[str, Any] = {"tool": tool}
        if tool == "browse":
            task = _strip_wrapping_quotes(str(raw.get("task", "")).strip())
            if not task:
                continue
            step["task"] = task
        elif tool in {"list", "read"}:
            path = _strip_wrapping_quotes(str(raw.get("path", "")).strip())
            if not path:
                continue
            step["path"] = path
        elif tool == "write":
            path = _strip_wrapping_quotes(str(raw.get("path", "")).strip())
            content = str(raw.get("content", ""))
            if not path or not content:
                continue
            step["path"] = path
            step["content"] = content
            step["append"] = bool(raw.get("append", False))
        elif tool == "exec":
            command = _strip_wrapping_quotes(str(raw.get("command", "")).strip())
            if not command:
                continue
            step["command"] = command

        steps.append(step)
    return steps


def _sanitize_tool_action(raw: dict[str, Any]) -> dict[str, Any] | None:
    action = str(raw.get("action", "")).strip().lower()
    if action not in {
        "respond",
        "browse",
        "list",
        "read",
        "write",
        "patch",
        "verify",
        "diff",
        "git_status",
        "worktree_create",
        "worktree_remove",
        "exec",
    }:
        return None

    if action == "respond":
        reply = str(raw.get("reply", "")).strip()
        if not reply:
            return None
        return {"action": "respond", "reply": reply}

    if action == "browse":
        task = _strip_wrapping_quotes(str(raw.get("task", "")).strip())
        return {"action": "browse", "task": task} if task else None

    if action in {"list", "read"}:
        path = _strip_wrapping_quotes(str(raw.get("path", "")).strip())
        return {"action": action, "path": path} if path else None

    if action == "write":
        path = _strip_wrapping_quotes(str(raw.get("path", "")).strip())
        content = str(raw.get("content", ""))
        if not path or not content:
            return None
        return {
            "action": "write",
            "path": path,
            "content": content,
            "append": bool(raw.get("append", False)),
        }

    if action == "patch":
        path = _strip_wrapping_quotes(str(raw.get("path", "")).strip())
        search = str(raw.get("search", ""))
        replace = str(raw.get("replace", ""))
        if not path or not search:
            return None
        payload = {
            "action": "patch",
            "path": path,
            "search": search,
            "replace": replace,
            "replace_all": bool(raw.get("replace_all", False)),
        }
        cwd = _strip_wrapping_quotes(str(raw.get("cwd", "")).strip())
        if cwd:
            payload["cwd"] = cwd
        return payload

    if action == "verify":
        commands = raw.get("commands", [])
        if not isinstance(commands, list):
            return None
        cleaned = [str(item).strip() for item in commands if str(item).strip()]
        if not cleaned:
            return None
        payload = {"action": "verify", "commands": cleaned}
        cwd = _strip_wrapping_quotes(str(raw.get("cwd", "")).strip())
        if cwd:
            payload["cwd"] = cwd
        return payload

    if action == "diff":
        payload = {"action": "diff"}
        cwd = _strip_wrapping_quotes(str(raw.get("cwd", "")).strip())
        if cwd:
            payload["cwd"] = cwd
        return payload

    if action == "git_status":
        payload = {"action": "git_status"}
        cwd = _strip_wrapping_quotes(str(raw.get("cwd", "")).strip())
        if cwd:
            payload["cwd"] = cwd
        return payload

    if action == "worktree_create":
        branch = _strip_wrapping_quotes(str(raw.get("branch", "")).strip())
        if not branch:
            return None
        payload = {
            "action": "worktree_create",
            "branch": branch,
            "base_ref": _strip_wrapping_quotes(str(raw.get("base_ref", "HEAD")).strip()) or "HEAD",
        }
        path = _strip_wrapping_quotes(str(raw.get("path", "")).strip())
        if path:
            payload["path"] = path
        return payload

    if action == "worktree_remove":
        path = _strip_wrapping_quotes(str(raw.get("path", "")).strip())
        if not path:
            return None
        return {
            "action": "worktree_remove",
            "path": path,
            "force": bool(raw.get("force", False)),
        }

    command = _strip_wrapping_quotes(str(raw.get("command", "")).strip())
    return {"action": "exec", "command": command} if command else None


def _looks_like_agentic_tool_request(message: str) -> bool:
    lowered = message.lower()
    if _infer_workspace_intent(message) is not None:
        return True
    if any(hint in lowered for hint in _AGENTIC_HINTS):
        return True
    return bool(_extract_backticked_chunks(message))


# Jarvis persona — base system prompt, memory context injected per-call
_JARVIS_SYSTEM_BASE = """\
You are Jarvis, a personal AI companion. You are direct, honest, and genuinely useful.

You are meeting this person FOR THE FIRST TIME. You have NO memory of past conversations.
Do NOT say things like "you mentioned before", "in our previous conversations", or "as you said" — \
there is no prior history. If they ask if you know them, say honestly that you don't yet.

HARD RULES — never break these:
- You can browse the web and use local workspace tools when the user's request
  clearly calls for them.
  Available tool-equivalent commands are: browse web, list files, read files,
  write files, and run local commands.
  If you use a tool, only claim what the tool actually returned.
- NEVER invent facts about the user. Do not assume routines, habits, or
  feelings they haven't stated.
- NEVER pretend you've done something you haven't.
- NEVER claim browsing, file creation, local paths, project setup, or build progress
  unless the live state below explicitly says it happened.
- Ask ONE follow-up question at most. Never fire a list of questions.
- Keep replies concise. No filler phrases like "Great question!" or "Certainly!".\
"""

_JARVIS_SYSTEM_WITH_MEMORY = """\
You are Jarvis, a personal AI companion with persistent memory. You are direct and genuinely useful.

You know the following about this person (from past conversations):
{memories}

HARD RULES — never break these:
- You can browse the web and use local workspace tools when the user's request
  clearly calls for them.
  Available tool-equivalent commands are: browse web, list files, read files,
  write files, and run local commands.
  If you use a tool, only claim what the tool actually returned.
- NEVER invent observations about the user beyond what's explicitly in the memories above.
  Do not fabricate routines, habits, moods, or behaviors they haven't stated.
- Use memories naturally — don't announce "I remember you said...". Just use what you know.
- NEVER claim browsing, file creation, local paths, project setup, or build progress
  unless the live state below explicitly says it happened.
- Ask ONE follow-up question at most.
- Keep replies concise. No padding, no filler.\
"""


class DaemonIPCServer:
    """Unix socket JSON-RPC server running inside the daemon process."""

    def __init__(
        self,
        socket_path: Path,
        brain: Any,  # Mnemon
        state: DaemonState,
        autonomy: AutonomyController,
        idle_loop: Any,  # IdleThinkingLoop
    ) -> None:
        self._socket_path = socket_path
        self._brain = brain
        self._state = state
        self._autonomy = autonomy
        self._idle_loop = idle_loop
        self._running = False
        self._task_group: anyio.abc.TaskGroup | None = None
        # Rolling conversation history — last 20 turns (user+assistant pairs)
        self._chat_history: deque[dict[str, str]] = deque(maxlen=40)
        # Lazy-initialised browser tool
        self._browser: Any = None
        self._workspace: Any = None
        self._pending_tool_actions: dict[UUID, dict[str, Any]] = {}
        self._improver: SelfImprovementOrchestrator | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._scenario_engine: ScenarioEngine | None = None
        self._report_engine: ReportEngine | None = None
        self._conversation_activity: deque[dict[str, str]] = deque(maxlen=20)
        self._handlers: dict[str, Any] = {
            "chat": self._rpc_chat,
            "status": self._rpc_status,
            "thoughts": self._rpc_thoughts,
            "goals.list": self._rpc_goals_list,
            "goals.add": self._rpc_goals_add,
            "goals.update": self._rpc_goals_update,
            "goals.update_status": self._rpc_goals_update_status,
            "memory.recent": self._rpc_memory_recent,
            "memory.profile": self._rpc_memory_profile,
            "memory.get": self._rpc_memory_get,
            "memory.timeline": self._rpc_memory_timeline,
            "memory.recall": self._rpc_memory_recall,
            "memory.hybrid": self._rpc_memory_hybrid,
            "memory.graph": self._rpc_memory_graph,
            "memory.clear": self._rpc_memory_clear,
            "memory.explain_fact": self._rpc_memory_explain_fact,
            "memory.causal_trace": self._rpc_memory_causal_trace,
            "debug.db_snapshot": self._rpc_debug_db_snapshot,
            "debug.clear_all": self._rpc_debug_clear_all,
            "scenario.run": self._rpc_scenario_run,
            "report.run": self._rpc_report_run,
            "timeline.recent": self._rpc_timeline_recent,
            "autonomy.set_level": self._rpc_autonomy_set_level,
            "improve.analyze": self._rpc_improve_analyze,
            "improve.start": self._rpc_improve_start,
            "improve.status": self._rpc_improve_status,
            "improve.approve": self._rpc_improve_approve,
            "improve.abort": self._rpc_improve_abort,
            "approve": self._rpc_approve,
            "deny": self._rpc_deny,
            "pending": self._rpc_pending,
            "pending.clear": self._rpc_pending_clear,
            "inbox.mark_read": self._rpc_inbox_mark_read,
            "browse": self._rpc_browse,
            "memory.search": self._rpc_memory_search,
            "workspace.list": self._rpc_workspace_list,
            "workspace.read": self._rpc_workspace_read,
            "workspace.write": self._rpc_workspace_write,
            "workspace.patch": self._rpc_workspace_patch,
            "workspace.exec": self._rpc_workspace_exec,
            "workspace.verify": self._rpc_workspace_verify,
            "workspace.git_diff": self._rpc_workspace_git_diff,
            "workspace.git_status": self._rpc_workspace_git_status,
            "workspace.worktree_create": self._rpc_workspace_worktree_create,
            "workspace.worktree_remove": self._rpc_workspace_worktree_remove,
            "chat.clear": self._rpc_chat_clear,
            "shutdown": self._rpc_shutdown,
        }

    async def start(self, task_group: anyio.abc.TaskGroup) -> None:
        """Start listening on the Unix socket."""
        # Ensure parent directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove stale socket
        if self._socket_path.exists():
            self._socket_path.unlink()

        self._running = True
        self._task_group = task_group
        task_group.start_soon(self._serve)
        logger.info("IPC server starting on %s", self._socket_path)

    async def stop(self) -> None:
        """Stop accepting connections."""
        self._running = False
        if self._socket_path.exists():
            with suppress(OSError):
                self._socket_path.unlink()
        logger.info("IPC server stopped.")

    async def _serve(self) -> None:
        """Accept and handle connections."""
        try:
            listener = await create_unix_listener(str(self._socket_path))
            async with listener:
                while self._running:
                    try:
                        with anyio.fail_after(1.0):
                            conn = await listener.accept()
                    except TimeoutError:
                        continue
                    except anyio.ClosedResourceError:
                        break

                    if self._task_group is not None:
                        self._task_group.start_soon(self._handle_connection_and_close, conn)
                    else:
                        await self._handle_connection_and_close(conn)
        except Exception:
            if self._running:
                logger.exception("IPC server error.")

    async def _handle_connection_and_close(self, conn: anyio.abc.ByteStream) -> None:
        """Handle one client connection without blocking new accepts."""
        try:
            await self._handle_connection(conn)
        finally:
            await conn.aclose()

    async def _handle_connection(self, stream: anyio.abc.ByteStream) -> None:
        """Read one JSON-RPC request, dispatch, and send response."""
        try:
            data = await stream.receive(65536)
            if not data:
                return

            request = json.loads(data.decode("utf-8"))
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id", None)

            handler = self._handlers.get(method)
            if handler is None:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }
            else:
                try:
                    result = await handler(**params)
                    response = {"jsonrpc": "2.0", "id": req_id, "result": result}
                except Exception as exc:
                    logger.exception("RPC handler %s failed.", method)
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": str(exc)},
                    }

            with suppress(Exception):
                await stream.send(json.dumps(response).encode("utf-8"))

        except (anyio.BrokenResourceError, anyio.EndOfStream):
            pass  # Client disconnected mid-request
        except Exception:
            logger.exception("IPC connection handler error.")

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    def _get_improver(self) -> SelfImprovementOrchestrator:
        if self._improver is None:
            try:
                llm = self._brain.control.goals._llm
            except Exception as exc:
                raise RuntimeError("LLM not available") from exc
            self._improver = SelfImprovementOrchestrator(
                workspace=self._get_workspace(),
                llm=llm,
            )
        return self._improver

    async def _rpc_improve_analyze(self) -> dict[str, Any]:
        improver = self._get_improver()
        return await improver.analyze()

    async def _rpc_improve_start(
        self,
        goal: str = "improve code quality",
    ) -> dict[str, Any]:
        improver = self._get_improver()
        session = improver.session
        if session is not None and session.phase not in {
            Phase.DONE,
            Phase.ABORTED,
            Phase.FAILED,
        }:
            return {
                "started": False,
                "error": "A self-improvement session is already running",
                "phase": str(session.phase),
            }

        async def _run() -> None:
            try:
                await improver.run(goal)
            except Exception:
                logger.exception("Self-improvement session failed")

        task = asyncio.create_task(_run())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return {"started": True, "goal": goal}

    async def _rpc_improve_status(self) -> dict[str, Any]:
        if self._improver is None:
            return {"phase": "idle", "session_id": None}
        return self._improver.status()

    async def _rpc_improve_approve(self) -> dict[str, Any]:
        if self._improver is None:
            return {"ok": False, "error": "no session awaiting approval"}
        return await self._improver.approve()

    async def _rpc_improve_abort(self) -> dict[str, Any]:
        if self._improver is None:
            return {"ok": False, "error": "no active session to abort"}
        return await self._improver.abort()

    @staticmethod
    def _memory_preview(doc: dict[str, Any], limit: int = 220) -> str:
        context = str(doc.get("context", "")).strip()
        action = str(doc.get("action", "")).strip()
        outcome = str(doc.get("outcome", "")).strip()
        joined = " — ".join(part for part in (context, action, outcome) if part)
        if len(joined) <= limit:
            return joined
        return joined[: max(0, limit - 1)].rstrip() + "…"

    @classmethod
    def _serialize_memory_index(
        cls,
        doc: dict[str, Any],
        *,
        score: float | None = None,
        source: str = "episodic",
    ) -> dict[str, Any]:
        tags = [str(tag) for tag in doc.get("tags", [])]
        memory_id = str(doc.get("id", ""))
        return {
            "id": memory_id,
            "preview": cls._memory_preview(doc),
            "content": cls._memory_preview(doc),
            "score": score,
            "source": source,
            "timestamp": str(doc.get("timestamp", "")),
            "importance": float(doc.get("importance", 0.0) or 0.0),
            "tags": tags,
            "session_id": str(doc.get("session_id", "")),
            "scope_type": str(doc.get("scope_type", "personal")),
            "scope_id": str(doc.get("scope_id", "")),
            "workspace_path": str(doc.get("workspace_path", "")),
            "repo_name": str(doc.get("repo_name", "")),
            "citation": f"[memory:{memory_id}]" if memory_id else "",
            "caused_by": str(doc.get("caused_by", "")) or None,
            "led_to": [str(item) for item in doc.get("led_to", [])],
            "source_episode_ids": [str(item) for item in doc.get("source_episode_ids", [])],
            "summary_kind": str(doc.get("summary_kind", "")),
            "summary_of_count": int(doc.get("summary_of_count", 0) or 0),
        }

    @classmethod
    def _serialize_memory_detail(
        cls,
        doc: dict[str, Any],
        *,
        score: float | None = None,
        source: str = "episodic",
    ) -> dict[str, Any]:
        item = cls._serialize_memory_index(doc, score=score, source=source)
        item.update(
            {
                "context": str(doc.get("context", "")),
                "action": str(doc.get("action", "")),
                "outcome": str(doc.get("outcome", "")),
                "access_count": int(doc.get("access_count", 0) or 0),
                "last_accessed": str(doc.get("last_accessed", "")),
            }
        )
        return item

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    async def _load_memory_docs(self, limit: int = 200) -> list[dict[str, Any]]:
        docs = await self._brain.memory.episodic._document_store.query(
            filters={},
            limit=max(1, min(limit, 1_000)),
        )
        docs.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
        return docs

    def _workspace_scope(self) -> dict[str, str]:
        """Return the daemon's current workspace scope metadata."""
        workspace = self._get_workspace()
        root = workspace.root
        return {
            "scope_type": "workspace",
            "scope_id": root.name,
            "workspace_path": str(root),
            "repo_name": root.name,
        }

    def _normalize_scope(
        self,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, str | None]:
        """Normalize public scope selectors into explicit metadata."""
        normalized = scope.strip().lower() or "all"
        if normalized == "workspace":
            metadata = self._workspace_scope()
            if scope_id:
                metadata["scope_id"] = scope_id
            return metadata
        if normalized == "personal":
            return {
                "scope_type": "personal",
                "scope_id": scope_id or "personal",
                "workspace_path": None,
                "repo_name": None,
            }
        return {
            "scope_type": "all",
            "scope_id": scope_id,
            "workspace_path": None,
            "repo_name": None,
        }

    @staticmethod
    def _matches_scope(doc: dict[str, Any], scope_type: str, scope_id: str | None) -> bool:
        """Return True if a serialized memory doc belongs to the requested scope."""
        if scope_type == "all":
            return True
        if str(doc.get("scope_type", "personal")) != scope_type:
            return False
        if scope_id is None:
            return True
        return str(doc.get("scope_id", "")) == scope_id

    @staticmethod
    def _fact_payloads_with_citations(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Decorate fact payloads with citation strings."""
        enriched: list[dict[str, Any]] = []
        for fact in facts:
            source_ids = [str(item) for item in fact.get("source_ids", [])]
            enriched.append(
                {
                    **fact,
                    "citations": [f"[memory:{source_id}]" for source_id in source_ids if source_id],
                }
            )
        return enriched

    @staticmethod
    def _semantic_fact_text(doc: dict[str, Any]) -> str:
        subject = doc.get("subject", {})
        subject_name = subject.get("name", "") if isinstance(subject, dict) else ""
        object_value = doc.get("object", "")
        if isinstance(object_value, dict):
            object_name = object_value.get("name", "")
        else:
            object_name = str(object_value)
        return f"{subject_name} {doc.get('predicate', '')} {object_name}".strip()

    async def _semantic_evidence_chain(
        self,
        source_episode_ids: list[str],
    ) -> list[dict[str, Any]]:
        evidence_chain: list[dict[str, Any]] = []
        for source_episode_id in source_episode_ids:
            try:
                raw_doc = await self._brain.memory.episodic._document_store.get(
                    UUID(source_episode_id)
                )
            except Exception:
                raw_doc = None
            if raw_doc is None:
                continue
            item = self._serialize_memory_detail(raw_doc, source="episodic")
            item["episode_id"] = item["id"]
            evidence_chain.append(item)
        evidence_chain.sort(key=lambda item: str(item.get("timestamp", "")))
        return evidence_chain

    async def _rpc_memory_profile(self) -> dict[str, Any]:
        """Return a lightweight profile + recent context for the daemon user."""
        try:
            identity = JarvisIdentity(DaemonConfig().state_path)
            recent_docs = await self._load_memory_docs(limit=120)
            profile_model = self._build_master_profile(identity, recent_docs)
            identity.write_master_profile(profile_model)
            profile = identity.read_master_profile()
            tag_counts = Counter(
                tag
                for doc in recent_docs
                for tag in doc.get("tags", [])
                if isinstance(tag, str) and tag.strip()
            )
            active_goals = [
                goal.description for goal in self._brain.control.goals.get_active_goals()[:5]
            ]
            recent_memories = [self._serialize_memory_index(doc) for doc in recent_docs[:5]]
            dynamic = list(
                dict.fromkeys(
                    [
                        *[str(item) for item in profile.get("dynamic", [])],
                        *active_goals,
                    ]
                )
            )
            return {
                "static": profile.get("static", []),
                "dynamic": dynamic,
                "questions": profile.get("questions", []),
                "static_facts": self._fact_payloads_with_citations(profile.get("static_facts", [])),
                "dynamic_facts": self._fact_payloads_with_citations(
                    profile.get("dynamic_facts", [])
                ),
                "question_facts": self._fact_payloads_with_citations(
                    profile.get("question_facts", [])
                ),
                "top_tags": [
                    {"tag": tag, "count": count} for tag, count in tag_counts.most_common(5)
                ],
                "recent_memories": recent_memories,
                "recent_changes": self._fact_payloads_with_citations(
                    self._recent_profile_changes(profile_model)
                ),
                "active_scope": self._workspace_scope(),
                "raw_markdown": profile.get("raw_markdown", ""),
            }
        except Exception as exc:
            logger.warning("memory.profile failed: %s", exc)
            return {
                "static": [],
                "dynamic": [],
                "questions": [],
                "static_facts": [],
                "dynamic_facts": [],
                "question_facts": [],
                "top_tags": [],
                "recent_memories": [],
                "recent_changes": [],
                "active_scope": self._workspace_scope(),
                "error": str(exc),
            }

    def _build_master_profile(
        self,
        identity: JarvisIdentity,
        docs: list[dict[str, Any]],
    ) -> MasterProfile:
        """Build a structured profile from captured episodic docs + existing questions."""
        existing = identity.read_master_profile_model()
        facts: list[ProfileFact] = []

        for doc in docs:
            tags = {str(tag) for tag in doc.get("tags", [])}
            source_id = str(doc.get("id", ""))
            timestamp = str(doc.get("timestamp", ""))
            context = str(doc.get("context", "")).strip()
            if not context:
                continue

            if "profile_static" in tags:
                section = self._static_section_for_text(context)
                facts.append(
                    ProfileFact(
                        text=context,
                        section=section,
                        source_ids=[source_id] if source_id else [],
                        updated_at=timestamp,
                    )
                )

            if "profile_dynamic" in tags or "project_context" in tags:
                facts.append(
                    ProfileFact(
                        text=context,
                        section="What They're Working On",
                        source_ids=[source_id] if source_id else [],
                        updated_at=timestamp,
                    )
                )

        for goal in self._active_goal_descriptions():
            facts.append(
                ProfileFact(
                    text=goal,
                    section="What They're Working On",
                    source_ids=[],
                    updated_at="",
                )
            )

        preserved_facts = [
            fact
            for fact in existing.facts
            if fact.section == "Questions I Want to Ask Them" or not fact.source_ids
        ]

        profile = MasterProfile(facts=[])
        for fact in [*preserved_facts, *facts]:
            profile = JarvisIdentity._merge_fact(profile, fact)
        return profile

    @staticmethod
    def _static_section_for_text(text: str) -> str:
        """Route a stable profile fact to a human-readable section."""
        lowered = text.lower()
        if any(marker in lowered for marker in ("prefer", "like", "love", "favorite")):
            return "What Drives Them"
        return "Who Is My Master"

    @staticmethod
    def _recent_profile_changes(profile: MasterProfile) -> list[dict[str, Any]]:
        """Return a small latest-first list of recently updated profile facts."""
        with_timestamps = [fact for fact in profile.facts if fact.updated_at]
        with_timestamps.sort(key=lambda fact: fact.updated_at, reverse=True)
        return [fact.model_dump(mode="json") for fact in with_timestamps[:5]]

    async def _rpc_memory_recall(
        self,
        query: str = "",
        top_k: int = 10,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Return profile + relevant memories together in one call."""
        return await self._rpc_memory_search(
            query=query,
            top_k=top_k,
            scope=scope,
            scope_id=scope_id,
        )

    async def _rpc_memory_hybrid(
        self,
        query: str = "",
        top_k: int = 10,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Return profile + memories + goals + workspace snippets together."""
        if not query.strip():
            return {"query": "", "hybrid_results": []}

        normalized_scope = self._normalize_scope(scope, scope_id)
        memory_result = await self._rpc_memory_search(
            query=query,
            top_k=top_k,
            scope=scope,
            scope_id=scope_id,
        )
        if memory_result.get("error"):
            return memory_result

        goals = [
            {
                "id": str(goal.id),
                "description": goal.description,
                "priority": goal.priority,
                "status": str(goal.status),
            }
            for goal in self._brain.control.goals.get_active_goals()[:5]
        ]

        workspace_items: list[dict[str, Any]] = []
        try:
            workspace_listing = await self._rpc_workspace_list(".")
            workspace_items = workspace_listing.get("entries", [])[:10]
        except Exception:
            logger.debug("memory.hybrid workspace listing failed", exc_info=True)

        hybrid_items: list[dict[str, Any]] = []
        for item in memory_result.get("results", []):
            hybrid_items.append(
                {
                    "kind": "memory",
                    "score": float(item.get("score", 0.0) or 0.0),
                    "title": str(item.get("preview", "")),
                    "payload": item,
                }
            )
        for goal in goals:
            score = 0.75 if query.lower() in goal["description"].lower() else 0.45
            hybrid_items.append(
                {
                    "kind": "goal",
                    "score": score,
                    "title": goal["description"],
                    "payload": goal,
                }
            )
        for item in workspace_items:
            label = str(item.get("path") or item.get("name") or "")
            score = 0.7 if query.lower() in label.lower() else 0.4
            hybrid_items.append(
                {
                    "kind": "workspace",
                    "score": score,
                    "title": label,
                    "payload": item,
                }
            )

        hybrid_items.sort(key=lambda item: float(item["score"]), reverse=True)

        return {
            "query": query,
            "scope": normalized_scope["scope_type"],
            "scope_id": normalized_scope["scope_id"],
            "profile": memory_result.get("profile", {}),
            "memory_results": memory_result.get("results", []),
            "goal_results": goals,
            "workspace_results": workspace_items,
            "hybrid_results": hybrid_items[: max(top_k * 3, 10)],
        }

    async def _rpc_memory_graph(
        self,
        limit: int = 40,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a lightweight graph of memory storage and relationships."""
        normalized_scope = self._normalize_scope(scope, scope_id)
        docs = await self._load_memory_docs(limit=max(1, min(limit, 120)))
        docs = [
            doc
            for doc in docs
            if self._matches_scope(
                doc,
                str(normalized_scope["scope_type"]),
                normalized_scope["scope_id"],
            )
        ]

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        def add_node(node: dict[str, Any]) -> None:
            node_id = str(node["id"])
            if node_id in seen_nodes:
                return
            seen_nodes.add(node_id)
            nodes.append(node)

        for doc in docs:
            memory_id = str(doc.get("id", "")).strip()
            if not memory_id:
                continue
            scope_label = (
                str(doc.get("repo_name", "")).strip()
                or str(doc.get("scope_id", "")).strip()
                or "personal"
            )
            scope_node_id = f"scope:{doc.get('scope_type', 'personal')}:{scope_label}"
            add_node(
                {
                    "id": scope_node_id,
                    "label": scope_label,
                    "kind": "scope",
                    "memory_type": "scope",
                    "scope_type": str(doc.get("scope_type", "personal")),
                }
            )

            node_kind = "summary" if int(doc.get("summary_of_count", 0) or 0) > 0 else "memory"
            add_node(
                {
                    "id": f"memory:{memory_id}",
                    "memory_id": memory_id,
                    "label": self._memory_preview(doc, limit=44),
                    "kind": node_kind,
                    "memory_type": "episodic",
                    "importance": float(doc.get("importance", 0.0) or 0.0),
                    "summary_of_count": int(doc.get("summary_of_count", 0) or 0),
                    "lifecycle_state": str(doc.get("lifecycle_state", "durable")),
                    "emotional_valence": float(doc.get("emotional_valence", 0.0) or 0.0),
                    "scope_type": str(doc.get("scope_type", "personal")),
                    "scope_id": str(doc.get("scope_id", "")),
                    "citation": f"[memory:{memory_id}]",
                }
            )
            edges.append(
                {
                    "id": f"edge:scope:{memory_id}",
                    "source": f"memory:{memory_id}",
                    "target": scope_node_id,
                    "kind": "stored_in",
                }
            )

            for source_episode_id in doc.get("source_episode_ids", []):
                source_id = str(source_episode_id).strip()
                if not source_id:
                    continue
                source_doc = next(
                    (candidate for candidate in docs if str(candidate.get("id", "")) == source_id),
                    None,
                )
                if source_doc is not None:
                    add_node(
                        {
                            "id": f"memory:{source_id}",
                            "memory_id": source_id,
                            "label": self._memory_preview(source_doc, limit=44),
                            "kind": "memory",
                            "memory_type": "episodic",
                            "importance": float(source_doc.get("importance", 0.0) or 0.0),
                            "summary_of_count": int(source_doc.get("summary_of_count", 0) or 0),
                            "lifecycle_state": str(source_doc.get("lifecycle_state", "durable")),
                            "scope_type": str(source_doc.get("scope_type", "personal")),
                            "scope_id": str(source_doc.get("scope_id", "")),
                            "citation": f"[memory:{source_id}]",
                        }
                    )
                edges.append(
                    {
                        "id": f"edge:summary:{memory_id}:{source_id}",
                        "source": f"memory:{memory_id}",
                        "target": f"memory:{source_id}",
                        "kind": "summarizes",
                    }
                )

            caused_by_id = str(doc.get("caused_by", "")).strip()
            if caused_by_id and caused_by_id.lower() != "none":
                edges.append(
                    {
                        "id": f"edge:caused_by:{memory_id}:{caused_by_id}",
                        "source": f"memory:{memory_id}",
                        "target": f"memory:{caused_by_id}",
                        "kind": "caused_by",
                    }
                )

            for led_to_id in doc.get("led_to", []):
                target_id = str(led_to_id).strip()
                if not target_id or target_id.lower() == "none":
                    continue
                edges.append(
                    {
                        "id": f"edge:led_to:{memory_id}:{target_id}",
                        "source": f"memory:{memory_id}",
                        "target": f"memory:{target_id}",
                        "kind": "led_to",
                    }
                )

        # --- Semantic triples (knowledge graph facts) ---
        try:
            triple_docs = await self._brain.memory.semantic._docs.query(
                filters={"_type": "triple"}, limit=min(limit, 60)
            )
            for tdoc in triple_docs:
                tid = str(tdoc.get("id", "")).strip()
                if not tid:
                    continue
                subj_raw = tdoc.get("subject", {})
                subj = tdoc.get("subject_name", "") or (
                    subj_raw.get("name", "") if isinstance(subj_raw, dict) else ""
                )
                pred = str(tdoc.get("predicate", ""))
                obj_raw_val = tdoc.get("object", {})
                obj_raw = tdoc.get("object_name", "") or (
                    obj_raw_val.get("name", "")
                    if isinstance(obj_raw_val, dict)
                    else str(obj_raw_val)
                )
                label = f"{subj} {pred} {obj_raw}".strip()[:44] or "semantic fact"
                add_node(
                    {
                        "id": f"semantic:{tid}",
                        "label": label,
                        "kind": "semantic",
                        "memory_type": "semantic",
                        "confidence": float(tdoc.get("confidence", 0.5) or 0.5),
                        "predicate": pred,
                    }
                )
                for ep_id in (tdoc.get("source_episodes") or []):
                    src = str(ep_id).strip()
                    if src:
                        edges.append(
                            {
                                "id": f"edge:sem_src:{tid}:{src}",
                                "source": f"semantic:{tid}",
                                "target": f"memory:{src}",
                                "kind": "extracted_from",
                            }
                        )
        except Exception:
            pass

        return {
            "scope": normalized_scope["scope_type"],
            "scope_id": normalized_scope["scope_id"],
            "nodes": nodes,
            "edges": edges,
        }

    async def _rpc_memory_clear(self, confirm: bool = False) -> dict[str, Any]:
        """Clear all episodic memories and reset vector index. Requires confirm=True."""
        if not confirm:
            return {"error": "Pass confirm=true to clear all memories. This is irreversible."}
        try:
            await self._clear_document_store(self._brain.memory.episodic._document_store)
            await self._brain.memory.episodic._vector_store.clear()
            return {"ok": True, "message": "Episodic memory cleared."}
        except Exception as exc:
            return {"error": str(exc)}

    async def _clear_document_store(self, store: Any) -> int:
        before = await store.count()
        clear = getattr(store, "clear", None)
        if callable(clear):
            result = clear()
            if hasattr(result, "__await__"):
                await result
            return int(before)
        docs = await store.query(filters={}, limit=1_000_000)
        for doc in docs:
            raw_id = str(doc.get("id", "")).strip()
            if raw_id:
                with suppress(ValueError):
                    await store.delete(UUID(raw_id))
        return int(before)

    async def _clear_vector_store(self, store: Any) -> int:
        before = await store.count()
        clear = getattr(store, "clear", None)
        if callable(clear):
            result = clear()
            if hasattr(result, "__await__"):
                await result
        return int(before)

    async def _rpc_debug_db_snapshot(self, limit: int = 5) -> dict[str, Any]:
        """Return transparent counts and sample records across daemon stores."""
        stores = {
            "episodic": self._brain.memory.episodic._document_store,
            "semantic": self._brain.memory.semantic._docs,
            "procedural": self._brain.memory.procedural._docs,
        }
        snapshot: dict[str, Any] = {}
        for name, store in stores.items():
            docs = await store.query(filters={}, limit=max(1, min(limit, 25)))
            snapshot[name] = {"count": await store.count(), "sample": docs}

        snapshot["runtime"] = {
            "thoughts": len(self._state.recent_thoughts),
            "proactive_inbox": len(self._state.proactive_inbox),
            "pending_approvals": len(self._autonomy.get_pending()),
            "chat_history": len(self._chat_history),
            "active_goals": len(self._brain.control.goals.get_active_goals()),
        }
        return snapshot

    async def _rpc_debug_clear_all(self, confirm: bool = False) -> dict[str, Any]:
        """Clear all DBs, thoughts, pending queues, goals, and Jarvis files."""
        if not confirm:
            return {"error": "Pass confirm=true to clear all daemon memory and state."}

        cleared = {
            "episodic_documents": await self._clear_document_store(
                self._brain.memory.episodic._document_store
            ),
            "semantic_documents": await self._clear_document_store(
                self._brain.memory.semantic._docs
            ),
            "procedural_documents": await self._clear_document_store(
                self._brain.memory.procedural._docs
            ),
            "episodic_vectors": await self._clear_vector_store(
                self._brain.memory.episodic._vector_store
            ),
            "semantic_vectors": await self._clear_vector_store(
                self._brain.memory.semantic._vectors
            ),
            "procedural_vectors": await self._clear_vector_store(
                self._brain.memory.procedural._vectors
            ),
        }

        graph = getattr(self._brain.memory.semantic, "_graph", None)
        if graph is not None and hasattr(graph, "_graph"):
            with suppress(Exception):
                import igraph

                graph._graph = igraph.Graph(directed=True)
                graph._uuid_to_vtx = {}
                graph_path = getattr(graph, "_graph_path", "")
                if graph_path:
                    Path(graph_path).unlink(missing_ok=True)

        self._state.recent_thoughts = []
        self._state.proactive_inbox = []
        self._state.pending_approvals = []
        self._state.pending_curiosity_question = None
        self._state.daemon_started_at = datetime.now(UTC)
        self._state.total_cycles = 0
        self._state.total_idle_ticks = 0
        self._state.last_user_interaction = None
        self._state.last_consolidation = None
        self._state.observer_stats = {}
        self._chat_history.clear()
        self._pending_tool_actions.clear()
        self._autonomy.clear_pending()

        from mnemon.daemon.identity import _LEARNINGS_INIT, _MASTER_INIT, _SOUL_INIT

        state_path = DaemonConfig().state_path
        state_path.mkdir(parents=True, exist_ok=True)
        (state_path / "soul.md").write_text(_SOUL_INIT, encoding="utf-8")
        (state_path / "master.md").write_text(_MASTER_INIT, encoding="utf-8")
        (state_path / "learnings.md").write_text(_LEARNINGS_INIT, encoding="utf-8")
        (state_path / "master_profile.json").unlink(missing_ok=True)
        (state_path / "goals.json").write_text("[]\n", encoding="utf-8")

        goals = getattr(self._brain.control.goals, "_goals", None)
        if isinstance(goals, dict):
            cleared["goals"] = len(goals)
            goals.clear()

        with suppress(Exception):
            from mnemon.daemon.state import save_state

            save_state(self._state, state_path)

        logger.warning("Cleared all daemon DBs and runtime state for testing.")
        return {"ok": True, "cleared": cleared}

    async def _rpc_memory_explain_fact(self, triple_id: str = "") -> dict[str, Any]:
        """Return a semantic fact plus the episodic evidence chain behind it."""
        if not triple_id:
            return {"error": "triple_id is required"}

        try:
            triple_uuid = UUID(triple_id)
        except ValueError as exc:
            return {"error": f"invalid triple_id: {exc}"}

        raw_doc = await self._brain.memory.semantic._docs.get(triple_uuid)
        if raw_doc is None or raw_doc.get("_type") != "triple":
            return {"error": f"semantic fact not found: {triple_id}"}

        source_episode_ids = [str(item) for item in raw_doc.get("source_episodes", [])]
        evidence_chain = await self._semantic_evidence_chain(source_episode_ids)
        contradiction_group = str(raw_doc.get("contradiction_group", "")).strip()
        related_facts: list[dict[str, Any]] = []

        if contradiction_group:
            triple_docs = await self._brain.memory.semantic._docs.query(
                filters={"_type": "triple"},
                limit=10_000,
            )
            related_facts = [
                {
                    "triple_id": str(doc.get("id", "")),
                    "fact": self._semantic_fact_text(doc),
                    "confidence": float(doc.get("confidence", 0.0) or 0.0),
                    "current": bool(doc.get("current", False)),
                    "last_confirmed": str(doc.get("last_confirmed", "")),
                }
                for doc in triple_docs
                if doc.get("contradiction_group") == contradiction_group
                and str(doc.get("id", "")) != triple_id
            ]

        return {
            "triple_id": triple_id,
            "fact": self._semantic_fact_text(raw_doc),
            "confidence": float(raw_doc.get("confidence", 0.0) or 0.0),
            "current": bool(raw_doc.get("current", False)),
            "last_confirmed": str(raw_doc.get("last_confirmed", "")),
            "source_episode_ids": source_episode_ids,
            "evidence_count": len(evidence_chain),
            "evidence_chain": evidence_chain,
            "related_facts": related_facts,
        }

    async def _rpc_memory_causal_trace(
        self,
        episode_id: str | None = None,
        outcome_query: str | None = None,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        """Trace the causal chain behind an episode or outcome query."""

        async def _get_episode_doc(raw_id: str) -> dict[str, Any] | None:
            try:
                return await self._brain.memory.episodic._document_store.get(UUID(raw_id))
            except ValueError:
                docs = await self._brain.memory.episodic._document_store.query(
                    filters={}, limit=10_000
                )
                return next((doc for doc in docs if str(doc.get("id", "")) == raw_id), None)

        target_doc: dict[str, Any] | None = None

        if episode_id:
            target_doc = await _get_episode_doc(episode_id)
        elif outcome_query and outcome_query.strip():
            search_result = await self._rpc_memory_search(query=outcome_query, top_k=1)
            if search_result.get("results"):
                result_id = str(search_result["results"][0].get("id", ""))
                if result_id:
                    target_doc = await _get_episode_doc(result_id)
        else:
            return {"error": "episode_id or outcome_query is required"}

        if target_doc is None:
            return {"error": "causal target episode not found"}

        chain: list[dict[str, Any]] = []
        visited: set[str] = set()
        current_doc = target_doc
        depth = 0
        while current_doc is not None and depth < max_depth:
            current_id = str(current_doc.get("id", ""))
            if current_id in visited:
                break
            visited.add(current_id)
            chain.append(self._serialize_memory_detail(current_doc))
            caused_by = str(current_doc.get("caused_by", "")).strip()
            if not caused_by:
                break
            current_doc = await _get_episode_doc(caused_by)
            depth += 1

        chain.reverse()
        target_id = str(target_doc.get("id", ""))
        return {
            "target_episode_id": target_id,
            "outcome_query": outcome_query,
            "chain_length": len(chain),
            "chain": chain,
            "citations": [f"[memory:{item['id']}]" for item in chain if item.get("id")],
        }

    async def _rpc_scenario_run(
        self,
        scenario: str = "",
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a bounded scenario report grounded in current memory/goals/workspace."""
        if not scenario.strip():
            return {"error": "scenario is required"}

        hybrid = await self._rpc_memory_hybrid(
            query=scenario,
            top_k=6,
            scope=scope,
            scope_id=scope_id,
        )
        if hybrid.get("error"):
            return hybrid

        engine = self._get_scenario_engine()
        return await engine.run(
            scenario=scenario,
            profile=hybrid.get("profile", {}),
            goals=hybrid.get("goal_results", []),
            memories=hybrid.get("memory_results", []),
            workspace_items=hybrid.get("workspace_results", []),
        )

    async def _rpc_report_run(
        self,
        report_type: str = "weekly",
        focus: str = "",
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a bounded weekly/project report grounded in current context."""
        normalized_type = report_type.strip().lower() or "weekly"
        if normalized_type not in {"weekly", "project"}:
            return {"error": "report_type must be 'weekly' or 'project'"}

        query = focus or normalized_type
        hybrid = await self._rpc_memory_hybrid(
            query=query,
            top_k=6,
            scope=scope,
            scope_id=scope_id,
        )
        if hybrid.get("error"):
            return hybrid

        engine = self._get_report_engine()
        return await engine.run(
            report_type=normalized_type,
            focus=focus,
            profile=hybrid.get("profile", {}),
            goals=hybrid.get("goal_results", []),
            memories=hybrid.get("memory_results", []),
            workspace_items=hybrid.get("workspace_results", []),
        )

    async def _rpc_memory_search(
        self,
        query: str = "",
        top_k: int = 10,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            return {"query": "", "results": []}

        try:
            from mnemon.core.models import RetrievalQuery

            normalized_scope = self._normalize_scope(scope, scope_id)
            filters: dict[str, Any] = {}
            if normalized_scope["scope_type"] != "all":
                filters = {
                    "scope_type": normalized_scope["scope_type"],
                    "scope_id": normalized_scope["scope_id"],
                }
            retrieval = await self._brain.memory.episodic.retrieve(
                RetrievalQuery(
                    query_text=query,
                    top_k=max(1, top_k),
                    filters=filters,
                )
            )
            results: list[dict[str, Any]] = []
            for item in retrieval.items:
                episode_id = str(item.metadata.get("episode_id", "")).strip()
                raw_doc: dict[str, Any] | None = None
                if episode_id:
                    with suppress(Exception):
                        raw_doc = await self._brain.memory.episodic._document_store.get(
                            UUID(episode_id)
                        )
                doc = raw_doc or {
                    "id": episode_id,
                    "context": item.content,
                    "timestamp": item.metadata.get("timestamp", ""),
                    "importance": item.metadata.get("importance", 0.0),
                    "tags": item.metadata.get("tags", []),
                    "session_id": item.metadata.get("session_id", ""),
                }
                results.append(
                    self._serialize_memory_index(
                        doc,
                        score=float(item.score),
                        source=str(item.source_store),
                    )
                )
            return {
                "query": query,
                "scope": normalized_scope["scope_type"],
                "scope_id": normalized_scope["scope_id"],
                "profile": await self._rpc_memory_profile(),
                "results": sorted(
                    results,
                    key=lambda item: (
                        int((item.get("summary_of_count", 0) or 0) > 0),
                        float(item.get("score", 0.0) or 0.0),
                    ),
                    reverse=True,
                ),
            }
        except Exception as exc:
            logger.warning("memory.search failed: %s", exc)
            return {"query": query, "results": [], "error": str(exc)}

    async def _rpc_memory_get(self, ids: list[str] | None = None) -> dict[str, Any]:
        """Fetch full episodic memory records by id."""
        if not ids:
            return {"items": [], "missing": []}

        items: list[dict[str, Any]] = []
        missing: list[str] = []

        for raw_id in ids[:50]:
            try:
                episode_id = UUID(str(raw_id))
            except ValueError:
                missing.append(str(raw_id))
                continue

            doc = await self._brain.memory.episodic._document_store.get(episode_id)
            if doc is None:
                missing.append(str(raw_id))
                continue
            items.append(self._serialize_memory_detail(doc))

        return {"items": items, "missing": missing}

    async def _rpc_memory_recent(self, limit: int = 20) -> dict[str, Any]:
        """Return recent episodic memory entries for browsing."""
        try:
            docs = await self._load_memory_docs(limit=max(1, min(limit * 5, 500)))
            items = [self._serialize_memory_detail(doc) for doc in docs[: max(1, min(limit, 100))]]
            return {"items": items}
        except Exception as exc:
            logger.warning("memory.recent failed: %s", exc)
            return {"items": [], "error": str(exc)}

    async def _rpc_memory_timeline(
        self,
        anchor_id: str = "",
        limit: int = 6,
    ) -> dict[str, Any]:
        """Return memories surrounding an anchor event."""
        if not anchor_id.strip():
            return {"anchor_id": "", "items": []}

        try:
            anchor_uuid = UUID(anchor_id)
        except ValueError:
            return {"anchor_id": anchor_id, "items": [], "error": "invalid anchor id"}

        anchor_doc = await self._brain.memory.episodic._document_store.get(anchor_uuid)
        if anchor_doc is None:
            return {"anchor_id": anchor_id, "items": [], "error": "anchor not found"}

        docs = await self._load_memory_docs(limit=300)
        anchor_session = str(anchor_doc.get("session_id", ""))
        if anchor_session:
            scoped_docs = [doc for doc in docs if str(doc.get("session_id", "")) == anchor_session]
            if len(scoped_docs) >= 2:
                docs = scoped_docs

        docs.sort(key=lambda doc: self._parse_timestamp(str(doc.get("timestamp", ""))))
        anchor_index = next(
            (idx for idx, doc in enumerate(docs) if str(doc.get("id", "")) == anchor_id),
            None,
        )
        if anchor_index is None:
            docs.append(anchor_doc)
            docs.sort(key=lambda doc: self._parse_timestamp(str(doc.get("timestamp", ""))))
            anchor_index = next(
                idx for idx, doc in enumerate(docs) if str(doc.get("id", "")) == anchor_id
            )

        total = max(1, min(limit, 20))
        before = max(0, anchor_index - total // 2)
        after = min(len(docs), before + total)
        window = docs[before:after]

        items: list[dict[str, Any]] = []
        for doc in window:
            item = self._serialize_memory_index(doc)
            item["anchor"] = str(doc.get("id", "")) == anchor_id
            items.append(item)

        return {
            "anchor_id": anchor_id,
            "session_id": str(anchor_doc.get("session_id", "")),
            "items": items,
        }

    async def _rpc_timeline_recent(self, limit: int = 40) -> dict[str, Any]:
        """Return a merged timeline of recent daemon activity."""
        items: list[dict[str, Any]] = []

        for thought in self._state.recent_thoughts[-limit:]:
            items.append(
                {
                    "kind": "thought",
                    "timestamp": str(thought.timestamp),
                    "title": thought.activity,
                    "summary": thought.summary,
                }
            )

        for message in self._state.proactive_inbox[-limit:]:
            items.append(
                {
                    "kind": "inbox",
                    "timestamp": str(message.timestamp),
                    "title": message.source_activity,
                    "summary": message.content,
                    "read": message.read,
                }
            )

        for approval in self._autonomy.get_pending():
            items.append(
                {
                    "kind": "approval",
                    "timestamp": str(approval.proposed_at),
                    "title": approval.source,
                    "summary": approval.description,
                    "risk": str(approval.risk_level),
                }
            )

        try:
            recent_memories = await self._rpc_memory_recent(limit=min(limit, 10))
            for memory in recent_memories.get("items", []):
                items.append(
                    {
                        "kind": "memory",
                        "timestamp": memory["timestamp"],
                        "title": "episodic memory",
                        "summary": memory["context"][:220],
                        "importance": memory["importance"],
                        "memory_id": memory["id"],
                        "tags": memory.get("tags", []),
                    }
                )
        except Exception:
            logger.debug("timeline memory merge failed", exc_info=True)

        items.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return {"items": items[: max(1, min(limit, 100))]}

    async def _rpc_autonomy_set_level(self, level: str = "") -> dict[str, Any]:
        """Set daemon autonomy level."""
        if not level:
            return {"error": "level is required"}
        from mnemon.daemon.config import AutonomyLevel

        try:
            self._autonomy.level = AutonomyLevel(level)
        except Exception as exc:
            return {"error": f"invalid autonomy level: {exc}"}
        return {"ok": True, "level": str(self._autonomy.level)}

    def _active_goal_descriptions(self) -> list[str]:
        """Return active goal descriptions for capture/ranking heuristics."""
        try:
            goals = self._brain.control.goals.get_active_goals()
        except Exception:
            return []
        return [str(goal.description) for goal in goals]

    @staticmethod
    def _normalize_goal_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @classmethod
    def _infer_goal_from_message(cls, message: str) -> str | None:
        cleaned = " ".join(message.strip().split()).strip(" .!?")
        if not cleaned:
            return None
        pattern = re.compile(
            r"^(?:please\s+|can you\s+|could you\s+|would you\s+|help me\s+|let's\s+)?"
            r"(?P<verb>build|make|create|design|develop|write|ship|start)\s+"
            r"(?P<body>.+)$",
            flags=re.IGNORECASE,
        )
        match = pattern.match(cleaned)
        if match is None:
            return None
        body = re.sub(r"\bfor me\b$", "", match.group("body"), flags=re.IGNORECASE).strip(" .!?")
        if len(body) < 5:
            return None
        verb = match.group("verb").lower()
        lead = "Ship" if verb == "ship" else "Build"
        return f"{lead} {body}"

    async def _maybe_create_goal_from_message(self, message: str) -> None:
        if _infer_workspace_intent(message) is not None:
            return
        goal_text = self._infer_goal_from_message(message)
        if goal_text is None:
            return
        try:
            goal_manager = self._brain.control.goals
            existing = self._active_goal_descriptions()
            normalized_goal = self._normalize_goal_text(goal_text)
            if any(
                normalized_goal == self._normalize_goal_text(item)
                or normalized_goal in self._normalize_goal_text(item)
                or self._normalize_goal_text(item) in normalized_goal
                for item in existing
            ):
                return
            await goal_manager.create_goal(goal_text, priority=0.75)
        except Exception:
            logger.debug("goal auto-capture failed", exc_info=True)

    def _record_conversation_activity(self, kind: str, detail: str) -> None:
        self._conversation_activity.append({"kind": kind, "detail": detail.strip()})

    def _grounded_progress_reply(self, message: str) -> str | None:
        lowered = message.lower()
        if not any(pattern in lowered for pattern in _PROGRESS_STATUS_PATTERNS):
            return None

        writes = [
            item
            for item in self._conversation_activity
            if item["kind"] in {"write", "patch", "worktree_create", "exec"}
        ]
        browses = [item for item in self._conversation_activity if item["kind"] == "browse"]
        if writes:
            latest = writes[-1]["detail"]
            return f"I only have verified local progress where I actually touched `{latest}`."
        if browses:
            latest = browses[-1]["detail"]
            return (
                "I haven't created anything locally yet. "
                f"So far I only researched `{latest}`."
            )
        return (
            "I haven't created files or started building this locally yet. "
            "So far we've only discussed the direction, and I don't have a verified local path."
        )

    def _conversation_state_block(self) -> str:
        goals = self._active_goal_descriptions()
        workspace_root = str(self._get_workspace().root)
        lines = [f"- Workspace root: {workspace_root}"]
        if goals:
            lines.append(f"- Active goals: {'; '.join(goals[:3])}")
        else:
            lines.append("- Active goals: none")

        writes = [
            item["detail"]
            for item in self._conversation_activity
            if item["kind"] in {"write", "patch", "worktree_create", "exec"}
        ]
        browses = [
            item["detail"]
            for item in self._conversation_activity
            if item["kind"] == "browse"
        ]
        if writes:
            lines.append(f"- Verified local work happened in: {writes[-1]}")
        else:
            lines.append("- Verified local work: none yet")
        if browses:
            lines.append(f"- Verified browsing happened for: {browses[-1]}")
        else:
            lines.append("- Verified browsing: none yet")
        lines.append(
            "- Treat the bullets above as authoritative. "
            "If local work is 'none yet', say so plainly."
        )
        return "\n".join(lines)

    def _recent_chat_context(self, limit: int = 6) -> str:
        history = list(self._chat_history)[-max(0, limit) :]
        if not history:
            return "(none)"
        return "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '').strip()}"
            for item in history
            if str(item.get("content", "")).strip()
        ) or "(none)"

    def _should_treat_as_execution_followup(self, message: str) -> bool:
        lowered = " ".join(message.lower().split())
        if not lowered:
            return False
        if any(pattern in lowered for pattern in _EXECUTION_FOLLOWUP_PATTERNS):
            return True
        return lowered in {"start", "build", "continue", "proceed"}

    async def _remember_verified_tool_action(
        self,
        *,
        message: str,
        step: dict[str, Any],
        output: str,
    ) -> None:
        tool = str(step.get("tool", "")).strip().lower()
        if tool in {"", "browse"}:
            return

        privacy_rules = load_privacy_rules(DaemonConfig().state_path)
        if (
            should_exclude_text(message, privacy_rules)
            or should_exclude_text(output, privacy_rules)
        ):
            return

        workspace = self._get_workspace()
        scope = self._workspace_scope()
        path_hint = (
            str(step.get("path", "")).strip()
            or str(step.get("cwd", "")).strip()
            or str(workspace.root)
        )
        redacted_message = apply_redactions(message, privacy_rules)
        redacted_output = apply_redactions(output, privacy_rules)
        action = self._describe_tool_step(step)
        tags = [
            "auto_capture",
            "source:tool",
            f"tool:{tool}",
            "verified_workspace_action",
            "project_context",
        ]
        importance = 0.55 if tool in {"list", "read", "git_status", "diff"} else 0.75
        episode = Episode(
            agent_id="jarvis",
            session_id=uuid4(),
            context=f"[tool request] {redacted_message}",
            action=f"{action} (verified at {path_hint})",
            outcome=redacted_output[:2000],
            tags=tags,
            importance=importance,
            scope_type=str(scope["scope_type"]),
            scope_id=str(scope["scope_id"]),
            workspace_path=str(scope["workspace_path"]) if scope["workspace_path"] else None,
            repo_name=str(scope["repo_name"]) if scope["repo_name"] else None,
        )
        await self._brain.memory.episodic.encode(episode)

    def _get_scenario_engine(self) -> ScenarioEngine:
        """Lazy-create the bounded scenario engine."""
        if self._scenario_engine is None:
            llm = self._brain.control.goals._llm
            self._scenario_engine = ScenarioEngine(llm)
        return self._scenario_engine

    def _get_report_engine(self) -> ReportEngine:
        """Lazy-create the bounded report engine."""
        if self._report_engine is None:
            llm = self._brain.control.goals._llm
            self._report_engine = ReportEngine(llm)
        return self._report_engine

    async def _rpc_chat(self, message: str = "") -> dict[str, Any]:
        """User sends a message -> run cognitive cycle -> generate reply -> return."""
        if not message:
            return {"error": "message is required"}

        await self._maybe_create_goal_from_message(message)

        tool_result = await self._handle_tool_request(message)
        if tool_result is not None:
            return tool_result

        # Pause idle thinking and wait for any in-flight tick to finish
        # before running the LLM — Ollama is single-threaded and concurrent
        # calls just queue up, making chat feel unresponsive.
        self._idle_loop.pause()
        if self._idle_loop.is_busy:
            logger.info("Waiting for in-flight idle tick to finish before chat…")
            waited = 0.0
            while self._idle_loop.is_busy and waited < 95:
                await anyio.sleep(0.5)
                waited += 0.5

        self._state.last_user_interaction = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
        self._state.total_cycles += 1
        privacy_rules = load_privacy_rules(DaemonConfig().state_path)

        active_goals = self._active_goal_descriptions()
        capture_pre = classify_interaction(
            user_message=message,
            active_goals=active_goals,
            source="chat",
            excluded_phrases=privacy_rules.excluded_phrases,
        )
        if not capture_pre.store_memory:
            with suppress(Exception):
                self._brain.orchestrator.suppress_next_episode_storage()
        elif privacy_rules.redaction_phrases:
            with suppress(Exception):
                self._brain.orchestrator.configure_next_episode_redactions(
                    privacy_rules.redaction_phrases
                )

        try:
            result = await self._brain.run_cycle(raw_input=message)
            deliberation = result.get("deliberation", {})

            # Detect if the user is asking Jarvis to research/browse something
            browse_result = ""
            if await self._is_browse_request(message):
                browse_result = await self._handle_browse_request(message)

            # Pass pending curiosity question as context — let the LLM weave it
            # in naturally only if the conversation flow allows it
            pending_q = self._pop_curiosity_question()

            # Generate reply with full conversation history + retrieved memories
            reply = await self._generate_reply(
                message,
                deliberation,
                pending_curiosity=pending_q,
                browse_result=browse_result,
            )
            citation_ids = [
                str(item) for item in deliberation.get("citation_ids", []) if str(item).strip()
            ]
            if citation_ids and self._wants_citations(message):
                citation_text = " ".join(f"[memory:{item}]" for item in citation_ids[:5])
                reply = f"{reply}\n\nSources: {citation_text}"

            # Patch the episode outcome with Jarvis's actual reply
            if capture_pre.store_memory:
                try:
                    reply = apply_redactions(reply, privacy_rules)
                    await self._brain.orchestrator.update_last_episode_outcome(reply)
                    capture_post = classify_interaction(
                        user_message=message,
                        assistant_reply=reply,
                        active_goals=self._active_goal_descriptions(),
                        source="chat",
                        excluded_phrases=privacy_rules.excluded_phrases,
                    )
                    await self._brain.orchestrator.update_last_episode_metadata(
                        tags=capture_post.tags,
                        importance=capture_post.importance,
                        scope_type="personal",
                        scope_id="personal",
                        workspace_path=None,
                        repo_name=None,
                    )
                    if citation_ids:
                        await self._brain.orchestrator.record_retrieval_feedback(
                            citation_ids,
                            helpful=True,
                        )
                except Exception:
                    logger.warning("Could not update episode capture metadata", exc_info=True)

            # Update conversation history
            self._chat_history.append({"role": "user", "content": message})
            self._chat_history.append({"role": "assistant", "content": reply})

            return {
                "cycle": result.get("cycle_number"),
                "phases": result.get("phases_completed", []),
                "retrieved": result.get("retrieved_count", 0),
                "meta": result.get("meta_evaluation"),
                "deliberation": deliberation,
                "reply": reply,
                "citations": [f"[memory:{item}]" for item in citation_ids[:5]],
            }
        finally:
            self._idle_loop.resume()

    def _pop_curiosity_question(self) -> str:
        """Return and clear the pending curiosity question Jarvis thought of while idle."""
        q = getattr(self._state, "pending_curiosity_question", None)
        if q:
            self._state.pending_curiosity_question = None
            return q
        return ""

    async def _is_browse_request(self, message: str) -> bool:
        """Detect if the user is asking Jarvis to look something up online."""
        msg = message.lower()
        browse_keywords = (
            "research",
            "look up",
            "find",
            "search",
            "browse",
            "check online",
            "what is",
            "who is",
            "how does",
            "latest",
            "news about",
            "strategy",
            "script that",
            "write a script",
            "can you find",
            "find me",
            "get me",
        )
        return any(kw in msg for kw in browse_keywords)

    @staticmethod
    def _wants_citations(message: str) -> bool:
        """Return True when the user explicitly asks for sourced output."""
        lowered = message.lower()
        triggers = (
            "with citations",
            "with sources",
            "cite",
            "citation",
            "source ids",
            "show sources",
        )
        return any(trigger in lowered for trigger in triggers)

    async def _handle_browse_request(self, message: str) -> str:
        """Run a browse task derived from the user's message."""
        logger.info("Browse request detected: %s", message[:80])
        try:
            browser = self._get_browser()
            result = await browser.browse(message, store_in_memory=True)
            self._record_conversation_activity("browse", message)
            return result
        except Exception as exc:
            logger.warning("Browse request failed: %s", exc)
            return ""

    async def _generate_reply(
        self,
        message: str,
        deliberation: dict[str, Any],
        pending_curiosity: str = "",
        browse_result: str = "",
    ) -> str:
        """Generate a conversational reply using full history and retrieved memories."""
        direct_reply = self._grounded_progress_reply(message)
        if direct_reply is not None:
            return direct_reply

        try:
            llm = self._brain.control.goals._llm
            retrieved_context = deliberation.get("context", "").strip()
            goal_context = str(deliberation.get("goal", "")).strip()
            citation_ids = [
                str(item) for item in deliberation.get("citation_ids", []) if str(item).strip()
            ]

            memory_lines: list[str] = []
            for citation_id in citation_ids[:10]:
                with suppress(Exception):
                    raw_doc = await self._brain.memory.episodic._document_store.get(
                        UUID(citation_id)
                    )
                    if raw_doc is not None:
                        context = str(raw_doc.get("context", "")).strip()
                        if context and context.lower() not in {"(empty)", "empty"}:
                            memory_lines.append(context)
            if not memory_lines and retrieved_context:
                for ln in retrieved_context.splitlines():
                    ln = ln.strip()
                    for prefix in ("[episodic]", "[semantic]", "[procedural]"):
                        if ln.startswith(prefix):
                            ln = ln[len(prefix) :].strip()
                    if ln and len(ln) > 15 and ln.lower() not in ("(empty)", "empty"):
                        memory_lines.append(ln)

            # Deduplicate preserving order
            seen: set[str] = set()
            unique_memories: list[str] = []
            for ln in memory_lines:
                if ln not in seen:
                    seen.add(ln)
                    unique_memories.append(ln)

            # Pick system prompt based on whether we have memories
            if unique_memories:
                mem_text = "\n".join(f"- {m}" for m in unique_memories[:10])
                system = _JARVIS_SYSTEM_WITH_MEMORY.format(memories=mem_text)
            else:
                system = _JARVIS_SYSTEM_BASE

            system += f"\n\nLive state:\n{self._conversation_state_block()}"
            if goal_context and goal_context != "No specific goal":
                system += f"\n\nCurrent goal context:\n- {goal_context}"

            # Inject live browsing results if we did a browse
            if browse_result:
                system += (
                    "\n\nYou just browsed the web for the user's request. "
                    "Here is what you found:\n"
                    f"{browse_result[:3000]}\n\n"
                    "Use this to answer the user's question. "
                    "Be specific and reference actual findings."
                )

            if pending_curiosity:
                system += (
                    f'\n\nYou\'ve been thinking about asking: "{pending_curiosity}"'
                    "\nOnly ask it if it fits naturally at the end of your reply — "
                    "skip it if the topic is unrelated."
                )

            history = list(self._chat_history)
            reply = await llm.generate_chat(
                system=system,
                history=history,
                message=message,
            )
            return reply.strip()
        except Exception as exc:
            logger.warning("Reply generation failed: %s", exc)
            return ""

    async def _handle_tool_request(self, message: str) -> dict[str, Any] | None:
        command_result = await self._handle_tool_command(message)
        if command_result is not None:
            return command_result

        browse_hint = None
        if await self._is_browse_request(message):
            browse_hint = {"action": "browse", "task": message}

        inferred = _infer_workspace_intent(message)
        initial_action = (
            browse_hint or self._step_to_action(inferred) if inferred or browse_hint else None
        )
        if initial_action is None:
            has_agentic_signal = _looks_like_agentic_tool_request(message)
            if not has_agentic_signal and self._should_treat_as_execution_followup(message):
                has_agentic_signal = bool(self._active_goal_descriptions()) or bool(
                    self._chat_history
                )
            if not has_agentic_signal:
                return None
        return await self._run_agentic_tool_loop(
            message=message,
            initial_action=initial_action,
        )

    def _step_to_action(self, step: dict[str, Any] | None) -> dict[str, Any] | None:
        if step is None:
            return None
        action = {"action": step["tool"]}
        for key in ("task", "path", "content", "command", "append"):
            if key in step:
                action[key] = step[key]
        return action

    async def _run_agentic_tool_loop(
        self,
        message: str,
        initial_action: dict[str, Any] | None = None,
        tool_results: list[dict[str, str]] | None = None,
        steps_used: int = 0,
    ) -> dict[str, Any] | None:
        tool_results = list(tool_results or [])
        next_action = initial_action

        while steps_used < _MAX_AGENT_TOOL_STEPS:
            action = next_action or await self._decide_next_tool_action(
                message=message,
                tool_results=tool_results,
                steps_used=steps_used,
            )
            next_action = None

            if action is None:
                if not tool_results:
                    return None
                return self._tool_chat_result(
                    message,
                    "\n\n".join(item["output"] for item in tool_results),
                )

            if action["action"] == "respond":
                return self._tool_chat_result(message, action["reply"])

            step = self._action_to_step(action)
            approval_reply = self._maybe_queue_tool_approval(
                message=message,
                step=step,
                tool_results=tool_results,
                steps_used=steps_used,
            )
            if approval_reply is not None:
                return self._tool_chat_result(message, approval_reply["reply"])

            output = await self._execute_tool_step(step)
            with suppress(Exception):
                await self._remember_verified_tool_action(
                    message=message,
                    step=step,
                    output=output,
                )
            tool_results.append({"tool": step["tool"], "output": output})
            steps_used += 1

        return self._tool_chat_result(
            message,
            "\n\n".join(item["output"] for item in tool_results) or "Tool limit reached.",
        )

    async def _decide_next_tool_action(
        self,
        message: str,
        tool_results: list[dict[str, str]],
        steps_used: int,
    ) -> dict[str, Any] | None:
        try:
            llm = self._brain.control.goals._llm
        except Exception:
            return None

        results_text = (
            "\n\n".join(
                f"Tool: {item['tool']}\nOutput:\n{item['output'][:3000]}" for item in tool_results
            )
            or "(no tool results yet)"
        )

        prompt = (
            "You are an autonomous local coding assistant deciding the next tool action.\n"
            "Available actions:\n"
            "- respond: answer the user directly when the task is complete or no tool is needed\n"
            "- browse: browse the web for current info\n"
            "- list: list files/folders in the workspace\n"
            "- read: read a workspace file\n"
            "- write: create or overwrite a workspace file with provided content\n"
            "- patch: apply a targeted search/replace patch to an existing file\n"
            "- verify: run one or more verification commands such as tests, lint, or typecheck\n"
            "- diff: show the current git diff\n"
            "- git_status: show git status\n"
            "- worktree_create: create a managed git worktree for isolated edits\n"
            "- worktree_remove: remove a managed worktree when done\n"
            "- exec: run a local workspace command\n\n"
            f"Workspace root: {Path.cwd()}\n"
            f"Steps already used: {steps_used}/{_MAX_AGENT_TOOL_STEPS}\n\n"
            "Rules:\n"
            "- Use tools when they materially help answer or complete the task.\n"
            "- For coding requests, inspect files before editing when feasible.\n"
            "- Prefer worktree_create before risky multi-file edits.\n"
            "- Prefer patch over write when updating an existing file.\n"
            "- Prefer verify after code changes.\n"
            "- For write, provide the full intended file content.\n"
            "- If you already have enough information, choose respond.\n"
            "- If the latest message is a continuation like 'start now', 'build it', or "
            "'continue', infer the task from the active goals and recent conversation.\n"
            "- Do not invent tool output.\n\n"
            f"Active goals:\n{'; '.join(self._active_goal_descriptions()) or '(none)'}\n\n"
            f"Recent conversation:\n{self._recent_chat_context()}\n\n"
            f"User request:\n{message}\n\n"
            f"Previous tool results:\n{results_text}"
        )

        try:
            result = await llm.generate_structured(
                prompt=prompt,
                response_schema=_TOOL_ACTION_SCHEMA,
            )
        except Exception:
            return None

        if not isinstance(result, dict):
            return None
        return _sanitize_tool_action(result)

    def _action_to_step(self, action: dict[str, Any]) -> dict[str, Any]:
        step = {"tool": action["action"]}
        for key in (
            "task",
            "path",
            "content",
            "search",
            "replace",
            "replace_all",
            "command",
            "commands",
            "append",
            "cwd",
            "branch",
            "base_ref",
            "force",
        ):
            if key in action:
                step[key] = action[key]
        return step

    def _maybe_queue_tool_approval(
        self,
        message: str,
        step: dict[str, Any],
        tool_results: list[dict[str, str]],
        steps_used: int,
    ) -> dict[str, str] | None:
        risk = self._step_risk(step)
        if risk == RiskLevel.LOW:
            return None

        description = self._describe_tool_step(step)
        action = ProposedAction(
            description=description,
            risk_level=risk,
            source="ipc.tool_router",
            context=step,
        )
        permission = self._autonomy.check(action)
        if permission.allowed:
            return None

        self._pending_tool_actions[action.id] = {
            "message": message,
            "step": dict(step),
            "tool_results": [dict(item) for item in tool_results],
            "steps_used": steps_used,
        }
        return {
            "reply": (
                f"Pending approval ({risk}) for: {description}. "
                f"Use `mnemon-daemon pending` then `mnemon-daemon approve {action.id}`."
            ),
            "action_id": str(action.id),
        }

    def _step_risk(self, step: dict[str, Any]) -> RiskLevel:
        tool = step["tool"]
        if tool in {"list", "read", "browse", "verify", "diff", "git_status"}:
            return RiskLevel.LOW
        if tool in {"write", "patch", "worktree_create"}:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def _describe_tool_step(self, step: dict[str, Any]) -> str:
        tool = step["tool"]
        if tool == "browse":
            return f"browse the web for '{step['task'][:120]}'"
        if tool == "list":
            return f"list workspace directory '{step['path']}'"
        if tool == "read":
            return f"read workspace file '{step['path']}'"
        if tool == "write":
            mode = "append to" if step.get("append") else "write"
            return f"{mode} workspace file '{step['path']}'"
        if tool == "patch":
            return f"patch workspace file '{step['path']}'"
        if tool == "verify":
            return f"run verification commands ({len(step.get('commands', []))} step(s))"
        if tool == "diff":
            return "show git diff"
        if tool == "git_status":
            return "show git status"
        if tool == "worktree_create":
            return f"create managed worktree branch '{step['branch']}'"
        if tool == "worktree_remove":
            return f"remove managed worktree '{step['path']}'"
        return f"run workspace command '{step['command'][:120]}'"

    async def _execute_tool_step(self, step: dict[str, Any]) -> str:
        tool = step["tool"]
        if tool == "browse":
            result = await self._rpc_browse(step["task"])
            self._record_conversation_activity("browse", step["task"])
            return result.get("result", "")
        if tool == "list":
            result = await self._rpc_workspace_list(step["path"])
            lines = [f"{entry['type']:>4}  {entry['path']}" for entry in result["entries"]]
            return "\n".join(lines) if lines else "(empty directory)"
        if tool == "read":
            result = await self._rpc_workspace_read(step["path"])
            reply = result.get("content", "")
            if result.get("truncated"):
                reply += "\n...<truncated>..."
            return reply
        if tool == "write":
            result = await self._rpc_workspace_write(
                step["path"],
                step["content"],
                append=step.get("append", False),
            )
            self._record_conversation_activity("write", step["path"])
            mode = "Appended" if step.get("append") else "Wrote"
            return f"{mode} {result['bytes_written']} bytes to {result['path']}"
        if tool == "patch":
            result = await self._rpc_workspace_patch(
                path=step["path"],
                search=step["search"],
                replace=step.get("replace", ""),
                cwd=step.get("cwd"),
                replace_all=step.get("replace_all", False),
            )
            self._record_conversation_activity("patch", step["path"])
            return result.get("diff", "")
        if tool == "verify":
            result = await self._rpc_workspace_verify(
                commands=step.get("commands", []),
                cwd=step.get("cwd"),
                timeout_s=120.0,
            )
            sections = [f"passed={result.get('passed')}"]
            for item in result.get("results", []):
                sections.append(f"$ {item.get('command')}")
                sections.append(
                    f"exit_code={item.get('exit_code')} timed_out={item.get('timed_out')}"
                )
                if item.get("stdout"):
                    sections.append(f"stdout:\n{item['stdout']}")
                if item.get("stderr"):
                    sections.append(f"stderr:\n{item['stderr']}")
            return "\n\n".join(sections)
        if tool == "diff":
            result = await self._rpc_workspace_git_diff(step.get("cwd"))
            return result.get("stdout", "") or result.get("stderr", "")
        if tool == "git_status":
            result = await self._rpc_workspace_git_status(step.get("cwd"))
            return result.get("stdout", "") or result.get("stderr", "")
        if tool == "worktree_create":
            result = await self._rpc_workspace_worktree_create(
                branch=step["branch"],
                base_ref=step.get("base_ref", "HEAD"),
                path=step.get("path"),
            )
            self._record_conversation_activity(
                "worktree_create",
                result.get("path") or step["branch"],
            )
            sections = [f"worktree_path={result.get('path')}"]
            if result.get("stdout"):
                sections.append(result["stdout"])
            if result.get("stderr"):
                sections.append(result["stderr"])
            return "\n".join(sections)
        if tool == "worktree_remove":
            result = await self._rpc_workspace_worktree_remove(
                path=step["path"],
                force=step.get("force", False),
            )
            sections = [f"removed_worktree={result.get('path')}"]
            if result.get("stdout"):
                sections.append(result["stdout"])
            if result.get("stderr"):
                sections.append(result["stderr"])
            return "\n".join(sections)

        result = await self._rpc_workspace_exec(step["command"])
        self._record_conversation_activity("exec", step["command"])
        sections = [
            f"exit_code={result['exit_code']}",
            f"cwd={result['cwd']}",
        ]
        if result.get("stdout"):
            sections.append(f"stdout:\n{result['stdout']}")
        if result.get("stderr"):
            sections.append(f"stderr:\n{result['stderr']}")
        if result.get("timed_out"):
            sections.append("timed_out=true")
        return "\n\n".join(sections)

    async def _handle_tool_command(self, message: str) -> dict[str, Any] | None:
        stripped = message.strip()
        if not stripped.startswith("/"):
            return None

        command, _, remainder = stripped.partition(" ")
        command = command.lower()
        remainder = remainder.strip()

        if command == "/browse":
            if not remainder:
                reply = "Usage: /browse <task>"
                return self._tool_chat_result(message, reply)
            return await self._run_one_shot_tool_action(
                message,
                {"action": "browse", "task": remainder},
            )

        if command == "/ls":
            return await self._run_one_shot_tool_action(
                message,
                {"action": "list", "path": remainder or "."},
            )

        if command == "/read":
            if not remainder:
                return self._tool_chat_result(message, "Usage: /read <path>")
            return await self._run_one_shot_tool_action(
                message,
                {"action": "read", "path": remainder},
            )

        if command == "/write":
            parts = stripped.split(maxsplit=2)
            if len(parts) < 3:
                return self._tool_chat_result(message, "Usage: /write <path> <content>")
            return await self._run_one_shot_tool_action(
                message,
                {"action": "write", "path": parts[1], "content": parts[2], "append": False},
            )

        if command == "/exec":
            if not remainder:
                return self._tool_chat_result(message, "Usage: /exec <command>")
            return await self._run_one_shot_tool_action(
                message,
                {"action": "exec", "command": remainder},
            )

        return None

    async def _run_one_shot_tool_action(
        self,
        message: str,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        step = self._action_to_step(action)
        approval_reply = self._maybe_queue_tool_approval(
            message=message,
            step=step,
            tool_results=[],
            steps_used=0,
        )
        if approval_reply is not None:
            return self._tool_chat_result(message, approval_reply["reply"])

        output = await self._execute_tool_step(step)
        with suppress(Exception):
            await self._remember_verified_tool_action(
                message=message,
                step=step,
                output=output,
            )
        return self._tool_chat_result(message, output)

    def _tool_chat_result(self, message: str, reply: str) -> dict[str, Any]:
        self._chat_history.append({"role": "user", "content": message})
        self._chat_history.append({"role": "assistant", "content": reply})
        return {
            "cycle": None,
            "phases": ["tool"],
            "retrieved": 0,
            "meta": None,
            "deliberation": {},
            "reply": reply,
        }

    async def _rpc_status(self) -> dict[str, Any]:
        """Return daemon state and brain state."""
        brain_state = self._brain.get_state()
        daemon_config = DaemonConfig()
        telegram_pair_file = daemon_config.state_path / "telegram_chat_id.txt"
        paired_chat_id = (
            telegram_pair_file.read_text().strip() if telegram_pair_file.exists() else ""
        )
        return {
            "daemon": {
                "started_at": str(self._state.daemon_started_at),
                "total_cycles": self._state.total_cycles,
                "total_idle_ticks": self._state.total_idle_ticks,
                "last_user_interaction": str(self._state.last_user_interaction),
                "last_consolidation": str(self._state.last_consolidation),
                "autonomy_level": str(self._autonomy.level),
            },
            "brain": brain_state,
            "observers": {
                name: {"events": stats.events_observed, "last": str(stats.last_event_at)}
                for name, stats in self._state.observer_stats.items()
            },
            "pending_approvals": [
                {
                    "id": str(a.id),
                    "description": a.description,
                    "risk": str(a.risk_level),
                    "source": a.source,
                    "proposed_at": str(a.proposed_at),
                    "context": a.context,
                }
                for a in self._autonomy.get_pending()
            ],
            "channels": {
                "telegram": {
                    "configured": bool(
                        daemon_config.telegram_token or os.environ.get("JARVIS_TELEGRAM_TOKEN", "")
                    ),
                    "paired": bool(paired_chat_id),
                    "chat_id": paired_chat_id,
                    "poll_interval_s": daemon_config.telegram_poll_interval_s,
                }
            },
            "proactive_inbox": [
                {
                    "id": m.id,
                    "source_activity": m.source_activity,
                    "content": m.content,
                    "priority": m.priority,
                    "read": m.read,
                    "timestamp": str(m.timestamp),
                }
                for m in self._state.proactive_inbox
            ],
            "chat_history": list(self._chat_history)[-20:],
            "config": {
                "socket_path": str(getattr(daemon_config, "socket_path", "")),
                "log_path": str(getattr(daemon_config, "log_path", "")),
                "state_path": str(daemon_config.state_path),
                "webui_enabled": bool(getattr(daemon_config, "webui_enabled", True)),
                "webui_host": str(getattr(daemon_config, "webui_host", "")),
                "webui_port": int(getattr(daemon_config, "webui_port", 7777)),
                "git_journal_enabled": bool(getattr(daemon_config, "git_journal_enabled", False)),
            },
        }

    async def _rpc_chat_clear(self) -> dict[str, Any]:
        """Clear the in-memory conversation history for a fresh start."""
        self._chat_history.clear()
        logger.info("Chat history cleared.")
        return {"cleared": True}

    async def _rpc_inbox_mark_read(self, message_id: str | None = None) -> dict[str, Any]:
        """Mark proactive inbox messages as read."""
        count = self._state.mark_inbox_read(message_id)
        return {"marked": count}

    async def _rpc_browse(self, task: str = "") -> dict[str, Any]:
        """Browse the web for a task and return a text summary."""
        if not task:
            return {"error": "task is required"}
        browser = self._get_browser()
        result = await browser.browse(task, store_in_memory=True)
        return {"result": result, "task": task}

    def _get_browser(self) -> Any:
        """Lazily initialise the JarvisBrowser."""
        if self._browser is None:
            from mnemon.daemon.tools.browser import JarvisBrowser

            self._browser = JarvisBrowser(brain=self._brain)
        return self._browser

    async def _rpc_workspace_list(self, path: str = ".") -> dict[str, Any]:
        workspace = self._get_workspace()
        return await workspace.list_dir(path)

    async def _rpc_workspace_read(self, path: str = "") -> dict[str, Any]:
        if not path:
            return {"error": "path is required"}
        workspace = self._get_workspace()
        return await workspace.read_file(path)

    async def _rpc_workspace_write(
        self,
        path: str = "",
        content: str = "",
        append: bool = False,
    ) -> dict[str, Any]:
        if not path:
            return {"error": "path is required"}
        workspace = self._get_workspace()
        return await workspace.write_file(path, content, append=append)

    async def _rpc_workspace_patch(
        self,
        path: str = "",
        search: str = "",
        replace: str = "",
        cwd: str | None = None,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        if not path or not search:
            return {"error": "path and search are required"}
        workspace = self._get_workspace()
        return await workspace.patch_file(
            path=path,
            search=search,
            replace=replace,
            cwd=cwd,
            replace_all=replace_all,
        )

    async def _rpc_workspace_exec(
        self,
        command: str = "",
        cwd: str | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        if not command:
            return {"error": "command is required"}
        workspace = self._get_workspace()
        return await workspace.exec_command(command, cwd=cwd, timeout_s=timeout_s)

    async def _rpc_workspace_verify(
        self,
        commands: list[str] | None = None,
        cwd: str | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        if not commands:
            return {"error": "commands are required"}
        workspace = self._get_workspace()
        return await workspace.verify(commands=commands, cwd=cwd, timeout_s=timeout_s)

    async def _rpc_workspace_git_diff(self, cwd: str | None = None) -> dict[str, Any]:
        workspace = self._get_workspace()
        return await workspace.git_diff(cwd=cwd)

    async def _rpc_workspace_git_status(self, cwd: str | None = None) -> dict[str, Any]:
        workspace = self._get_workspace()
        return await workspace.git_status(cwd=cwd)

    async def _rpc_workspace_worktree_create(
        self,
        branch: str = "",
        base_ref: str = "HEAD",
        path: str | None = None,
    ) -> dict[str, Any]:
        if not branch:
            return {"error": "branch is required"}
        workspace = self._get_workspace()
        return await workspace.create_worktree(branch=branch, base_ref=base_ref, path=path)

    async def _rpc_workspace_worktree_remove(
        self,
        path: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        if not path:
            return {"error": "path is required"}
        workspace = self._get_workspace()
        return await workspace.remove_worktree(path=path, force=force)

    def _get_workspace(self) -> Any:
        if self._workspace is None:
            from mnemon.daemon.tools.workspace import JarvisWorkspace

            self._workspace = JarvisWorkspace(root=Path.cwd())
        return self._workspace

    async def _rpc_thoughts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent idle thinking results."""
        thoughts = self._state.recent_thoughts[-limit:]
        return [
            {
                "timestamp": str(t.timestamp),
                "activity": t.activity,
                "summary": t.summary,
            }
            for t in thoughts
        ]

    async def _rpc_goals_list(self) -> list[dict[str, Any]]:
        """Return all active goals."""
        goals = list(self._brain.control.goals.get_active_goals())
        return [
            {
                "id": str(g.id),
                "description": g.description,
                "priority": g.priority,
                "status": str(g.status),
                "progress": g.progress,
                "parent_id": str(g.parent_goal_id) if g.parent_goal_id else None,
                "subgoals": [str(s) for s in g.subgoals],
                "success_criteria": g.success_criteria,
            }
            for g in goals
        ]

    async def _rpc_goals_add(self, description: str = "", priority: float = 0.5) -> dict[str, Any]:
        """Create a new goal."""
        if not description:
            return {"error": "description is required"}
        goal = await self._brain.control.goals.create_goal(description, priority)
        return {
            "id": str(goal.id),
            "description": goal.description,
            "priority": goal.priority,
        }

    async def _rpc_goals_update(
        self,
        goal_id: str = "",
        description: str | None = None,
        priority: float | None = None,
        success_criteria: str | None = None,
    ) -> dict[str, Any]:
        """Update editable goal fields."""
        if not goal_id:
            return {"error": "goal_id is required"}

        try:
            goal_uuid = UUID(goal_id)
        except Exception as exc:
            return {"error": f"invalid goal_id: {exc}"}

        updated = await self._brain.control.goals.update_goal(
            goal_uuid,
            description=description,
            priority=priority,
            success_criteria=success_criteria,
        )
        return {
            "ok": True,
            "goal": {
                "id": str(updated.id),
                "description": updated.description,
                "priority": updated.priority,
                "status": str(updated.status),
                "progress": updated.progress,
                "parent_id": str(updated.parent_goal_id) if updated.parent_goal_id else None,
                "subgoals": [str(s) for s in updated.subgoals],
                "success_criteria": updated.success_criteria,
            },
        }

    async def _rpc_goals_update_status(
        self,
        goal_id: str = "",
        status: str = "",
    ) -> dict[str, Any]:
        """Update a goal lifecycle status."""
        if not goal_id or not status:
            return {"error": "goal_id and status are required"}

        try:
            goal_uuid = UUID(goal_id)
            goal_status = GoalStatus(status)
        except Exception as exc:
            return {"error": f"invalid goal update: {exc}"}

        await self._brain.control.goals.update_status(goal_uuid, goal_status)
        goals = await self._rpc_goals_list()
        updated = next((goal for goal in goals if goal["id"] == goal_id), None)
        return {
            "ok": True,
            "goal_id": goal_id,
            "status": status,
            "goal": updated,
        }

    async def _rpc_approve(self, action_id: str = "") -> dict[str, Any]:
        """Approve a pending action."""
        if not action_id:
            return {"error": "action_id is required"}
        action_uuid = UUID(action_id)
        ok = self._autonomy.approve(action_uuid)
        if not ok:
            return {"approved": False}

        pending_state = self._pending_tool_actions.pop(action_uuid, None)
        if not pending_state:
            return {"approved": True}

        step = dict(pending_state["step"])
        tool_results = [dict(item) for item in pending_state["tool_results"]]
        output = await self._execute_tool_step(step)
        tool_results.append({"tool": step["tool"], "output": output})

        continued = await self._run_agentic_tool_loop(
            message=str(pending_state["message"]),
            tool_results=tool_results,
            steps_used=int(pending_state["steps_used"]) + 1,
        )
        if continued is None:
            return {"approved": True, "reply": output, "result": output}
        return {
            "approved": True,
            "reply": continued.get("reply", output),
            "result": continued.get("reply", output),
        }

    async def _rpc_deny(self, action_id: str = "") -> dict[str, Any]:
        """Deny a pending action."""
        if not action_id:
            return {"error": "action_id is required"}
        action_uuid = UUID(action_id)
        self._pending_tool_actions.pop(action_uuid, None)
        ok = self._autonomy.deny(action_uuid)
        return {"denied": ok}

    async def _rpc_pending(self) -> list[dict[str, Any]]:
        """List pending approval requests."""
        return [
            {
                "id": str(a.id),
                "description": a.description,
                "risk": str(a.risk_level),
                "source": a.source,
                "proposed_at": str(a.proposed_at),
                "context": a.context,
            }
            for a in self._autonomy.get_pending()
        ]

    async def _rpc_pending_clear(self) -> dict[str, Any]:
        """Clear all pending approval requests."""
        cleared = self._autonomy.clear_pending()
        self._pending_tool_actions.clear()
        return {"cleared": cleared}

    async def _rpc_shutdown(self) -> dict[str, Any]:
        """Request daemon shutdown."""
        self._running = False
        return {"status": "shutting_down"}
