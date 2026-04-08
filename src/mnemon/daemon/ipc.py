"""
DaemonIPCServer — JSON-RPC over Unix domain socket for CLI ↔ daemon communication.

Brain analog: The thalamocortical interface — the gateway through which
external commands (user intent) enter the cognitive system. Just as the
thalamus relays sensory input to the cortex, the IPC server relays user
commands to the appropriate daemon subsystem and returns results.

Protocol: Newline-delimited JSON-RPC 2.0 over a Unix domain socket.
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import Any
from uuid import UUID

import anyio
import anyio.abc
from anyio import create_unix_listener

from mnemon.daemon.autonomy import AutonomyController, ProposedAction
from mnemon.daemon.config import RiskLevel
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
                command = stripped[len(prefix):]
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
- You can browse the web and use local workspace tools when the user's request clearly calls for them.
  Available tool-equivalent commands are: browse web, list files, read files, write files, and run local commands.
  If you use a tool, only claim what the tool actually returned.
- NEVER invent facts about the user. Do not assume routines, habits, or feelings they haven't stated.
- NEVER pretend you've done something you haven't.
- Ask ONE follow-up question at most. Never fire a list of questions.
- Keep replies concise. No filler phrases like "Great question!" or "Certainly!".\
"""

_JARVIS_SYSTEM_WITH_MEMORY = """\
You are Jarvis, a personal AI companion with persistent memory. You are direct and genuinely useful.

You know the following about this person (from past conversations):
{memories}

HARD RULES — never break these:
- You can browse the web and use local workspace tools when the user's request clearly calls for them.
  Available tool-equivalent commands are: browse web, list files, read files, write files, and run local commands.
  If you use a tool, only claim what the tool actually returned.
- NEVER invent observations about the user beyond what's explicitly in the memories above.
  Do not fabricate routines, habits, moods, or behaviors they haven't stated.
- Use memories naturally — don't announce "I remember you said...". Just use what you know.
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
        # Rolling conversation history — last 20 turns (user+assistant pairs)
        self._chat_history: deque[dict[str, str]] = deque(maxlen=40)
        # Lazy-initialised browser tool
        self._browser: Any = None
        self._workspace: Any = None
        self._pending_tool_actions: dict[UUID, dict[str, Any]] = {}
        self._handlers: dict[str, Any] = {
            "chat": self._rpc_chat,
            "status": self._rpc_status,
            "thoughts": self._rpc_thoughts,
            "goals.list": self._rpc_goals_list,
            "goals.add": self._rpc_goals_add,
            "approve": self._rpc_approve,
            "deny": self._rpc_deny,
            "pending": self._rpc_pending,
            "inbox.mark_read": self._rpc_inbox_mark_read,
            "browse": self._rpc_browse,
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
        task_group.start_soon(self._serve)
        logger.info("IPC server starting on %s", self._socket_path)

    async def stop(self) -> None:
        """Stop accepting connections."""
        self._running = False
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except OSError:
                pass
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

                    # Handle connection and close it
                    try:
                        await self._handle_connection(conn)
                    finally:
                        await conn.aclose()
        except Exception:
            if self._running:
                logger.exception("IPC server error.")

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

            try:
                await stream.send(json.dumps(response).encode("utf-8"))
            except Exception:
                pass  # Client disconnected before we could reply — not an error

        except (anyio.BrokenResourceError, anyio.EndOfStream):
            pass  # Client disconnected mid-request
        except Exception:
            logger.exception("IPC connection handler error.")

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    async def _rpc_chat(self, message: str = "") -> dict[str, Any]:
        """User sends a message -> run cognitive cycle -> generate reply -> return."""
        if not message:
            return {"error": "message is required"}

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
                message, deliberation,
                pending_curiosity=pending_q,
                browse_result=browse_result,
            )

            # Patch the episode outcome with Jarvis's actual reply
            try:
                await self._brain.orchestrator.update_last_episode_outcome(reply)
            except Exception:
                logger.warning("Could not update episode outcome", exc_info=True)

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
            "research", "look up", "find", "search", "browse",
            "check online", "what is", "who is", "how does", "latest",
            "news about", "strategy", "script that", "write a script",
            "can you find", "find me", "get me",
        )
        return any(kw in msg for kw in browse_keywords)

    async def _handle_browse_request(self, message: str) -> str:
        """Run a browse task derived from the user's message."""
        logger.info("Browse request detected: %s", message[:80])
        try:
            browser = self._get_browser()
            result = await browser.browse(message, store_in_memory=True)
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
        try:
            llm = self._brain.control.goals._llm
            retrieved_context = deliberation.get("context", "").strip()

            # Extract clean memory lines — only real user input, strip system noise
            memory_lines: list[str] = []
            if retrieved_context:
                for ln in retrieved_context.splitlines():
                    ln = ln.strip()
                    for prefix in ("[episodic]", "[semantic]", "[procedural]"):
                        if ln.startswith(prefix):
                            ln = ln[len(prefix):].strip()
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

            # Inject live browsing results if we did a browse
            if browse_result:
                system += (
                    f"\n\nYou just browsed the web for the user's request. Here is what you found:\n"
                    f"{browse_result[:3000]}\n\n"
                    "Use this to answer the user's question. Be specific and reference actual findings."
                )

            if pending_curiosity:
                system += (
                    f"\n\nYou've been thinking about asking: \"{pending_curiosity}\""
                    "\nOnly ask it if it fits naturally at the end of your reply — skip it if the topic is unrelated."
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
        initial_action = browse_hint or self._step_to_action(inferred) if inferred or browse_hint else None
        if initial_action is None and not _looks_like_agentic_tool_request(message):
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

        results_text = "\n\n".join(
            f"Tool: {item['tool']}\nOutput:\n{item['output'][:3000]}"
            for item in tool_results
        ) or "(no tool results yet)"

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
            "- Do not invent tool output.\n\n"
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
        goals = self._brain.control.goals.get_active_goals()
        return [
            {
                "id": str(g.id),
                "description": g.description,
                "priority": g.priority,
                "status": str(g.status),
                "progress": g.progress,
                "subgoals": [str(s) for s in g.subgoals],
            }
            for g in goals
        ]

    async def _rpc_goals_add(
        self, description: str = "", priority: float = 0.5
    ) -> dict[str, Any]:
        """Create a new goal."""
        if not description:
            return {"error": "description is required"}
        goal = await self._brain.control.goals.create_goal(description, priority)
        return {
            "id": str(goal.id),
            "description": goal.description,
            "priority": goal.priority,
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
            }
            for a in self._autonomy.get_pending()
        ]

    async def _rpc_shutdown(self) -> dict[str, Any]:
        """Request daemon shutdown."""
        self._running = False
        return {"status": "shutting_down"}
