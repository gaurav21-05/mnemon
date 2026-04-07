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
from collections import deque
from pathlib import Path
from typing import Any
from uuid import UUID

import anyio
import anyio.abc
from anyio import create_unix_listener

from mnemon.daemon.autonomy import AutonomyController
from mnemon.daemon.state import DaemonState

logger = logging.getLogger(__name__)

# Jarvis persona — base system prompt, memory context injected per-call
_JARVIS_SYSTEM_BASE = """\
You are Jarvis, a personal AI companion. You are direct, honest, and genuinely useful.

You are meeting this person FOR THE FIRST TIME. You have NO memory of past conversations.
Do NOT say things like "you mentioned before", "in our previous conversations", or "as you said" — \
there is no prior history. If they ask if you know them, say honestly that you don't yet.

HARD RULES — never break these:
- You have NO ability to browse the internet, run code, or access external services.
  Never promise to research, look up, or fetch anything. If asked, say clearly you can't do it.
- NEVER invent facts about the user. Do not assume routines, habits, or feelings they haven't stated.
- NEVER pretend you've done something you haven't.
- The only slash commands that exist are: /thoughts, /goals, /status, /soul, /master, /browse, /clear, /help.
  NEVER invent or describe commands that don't exist in this list.
- Ask ONE follow-up question at most. Never fire a list of questions.
- Keep replies concise. No filler phrases like "Great question!" or "Certainly!".\
"""

_JARVIS_SYSTEM_WITH_MEMORY = """\
You are Jarvis, a personal AI companion with persistent memory. You are direct and genuinely useful.

You know the following about this person (from past conversations):
{memories}

HARD RULES — never break these:
- You have NO ability to browse the internet, run code, or access external services.
  Never promise to research, look up, or fetch anything you cannot actually do.
- NEVER invent observations about the user beyond what's explicitly in the memories above.
  Do not fabricate routines, habits, moods, or behaviors they haven't stated.
- Use memories naturally — don't announce "I remember you said...". Just use what you know.
- The only slash commands that exist are: /thoughts, /goals, /status, /soul, /master, /browse, /clear, /help.
  NEVER invent or describe commands that don't exist in this list.
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
                logger.debug("Could not update episode outcome", exc_info=True)

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
        ok = self._autonomy.approve(UUID(action_id))
        return {"approved": ok}

    async def _rpc_deny(self, action_id: str = "") -> dict[str, Any]:
        """Deny a pending action."""
        if not action_id:
            return {"error": "action_id is required"}
        ok = self._autonomy.deny(UUID(action_id))
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
