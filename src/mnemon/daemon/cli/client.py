"""
DaemonClient — IPC client for communicating with a running daemon.

Connects to the daemon's Unix domain socket and sends JSON-RPC requests.
Used by the CLI and can also be used programmatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import anyio


class DaemonClient:
    """Connects to a running Mnemon daemon via Unix socket."""

    def __init__(self, socket_path: str | Path) -> None:
        self._socket_path = Path(socket_path).expanduser()

    async def call(self, method: str, **params: Any) -> Any:
        """Send a JSON-RPC request and return the result.

        Raises RuntimeError on connection failure or RPC error.
        """
        if not self._socket_path.exists():
            raise RuntimeError(
                f"Daemon socket not found at {self._socket_path}. "
                "Is the daemon running? Start it with: mnemon daemon start"
            )

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        try:
            stream = await anyio.connect_unix(str(self._socket_path))
            try:
                await stream.send(json.dumps(request).encode("utf-8"))
                chunks: list[bytes] = []
                while True:
                    try:
                        chunk = await stream.receive(262_144)
                    except anyio.EndOfStream:
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
                response = json.loads(b"".join(chunks).decode("utf-8"))
            finally:
                await stream.aclose()
        except ConnectionRefusedError as exc:
            raise RuntimeError(
                "Connection refused. The daemon may have crashed. "
                "Check logs with: mnemon daemon logs"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"IPC communication failed: {exc}") from exc

        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"RPC error ({err.get('code', '?')}): {err.get('message', '?')}")

        return response.get("result")

    async def _call_dict(self, method: str, **params: Any) -> dict[str, Any]:
        return cast("dict[str, Any]", await self.call(method, **params))

    async def _call_list(self, method: str, **params: Any) -> list[dict[str, Any]]:
        return cast("list[dict[str, Any]]", await self.call(method, **params))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def chat(self, message: str) -> dict[str, Any]:
        return await self._call_dict("chat", message=message)

    async def status(self) -> dict[str, Any]:
        return await self._call_dict("status")

    async def thoughts(self, limit: int = 10) -> list[dict[str, Any]]:
        return await self._call_list("thoughts", limit=limit)

    async def list_goals(self) -> list[dict[str, Any]]:
        return await self._call_list("goals.list")

    async def add_goal(self, description: str, priority: float = 0.5) -> dict[str, Any]:
        return await self._call_dict("goals.add", description=description, priority=priority)

    async def update_goal(
        self,
        goal_id: str,
        description: str | None = None,
        priority: float | None = None,
        success_criteria: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"goal_id": goal_id}
        if description is not None:
            params["description"] = description
        if priority is not None:
            params["priority"] = priority
        if success_criteria is not None:
            params["success_criteria"] = success_criteria
        return await self._call_dict("goals.update", **params)

    async def update_goal_status(self, goal_id: str, status: str) -> dict[str, Any]:
        return await self._call_dict("goals.update_status", goal_id=goal_id, status=status)

    async def improve_analyze(self) -> dict[str, Any]:
        return await self._call_dict("improve.analyze")

    async def improve_start(self, goal: str = "improve code quality") -> dict[str, Any]:
        return await self._call_dict("improve.start", goal=goal)

    async def improve_status(self) -> dict[str, Any]:
        return await self._call_dict("improve.status")

    async def improve_approve(self) -> dict[str, Any]:
        return await self._call_dict("improve.approve")

    async def improve_abort(self) -> dict[str, Any]:
        return await self._call_dict("improve.abort")

    async def memory_search(
        self,
        query: str,
        top_k: int = 10,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "memory.search",
            query=query,
            top_k=top_k,
            scope=scope,
            scope_id=scope_id,
        )

    async def memory_recall(
        self,
        query: str,
        top_k: int = 10,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "memory.recall",
            query=query,
            top_k=top_k,
            scope=scope,
            scope_id=scope_id,
        )

    async def memory_hybrid(
        self,
        query: str,
        top_k: int = 10,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "memory.hybrid",
            query=query,
            top_k=top_k,
            scope=scope,
            scope_id=scope_id,
        )

    async def memory_graph(
        self,
        limit: int = 40,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "memory.graph",
            limit=limit,
            scope=scope,
            scope_id=scope_id,
        )

    async def memory_clear(self, confirm: bool = False) -> dict[str, Any]:
        return await self._call_dict("memory.clear", confirm=confirm)

    async def debug_db_snapshot(self, limit: int = 5) -> dict[str, Any]:
        return await self._call_dict("debug.db_snapshot", limit=limit)

    async def debug_clear_all(self, confirm: bool = False) -> dict[str, Any]:
        return await self._call_dict("debug.clear_all", confirm=confirm)

    async def memory_explain_fact(self, triple_id: str) -> dict[str, Any]:
        return await self._call_dict("memory.explain_fact", triple_id=triple_id)

    async def memory_causal_trace(
        self,
        episode_id: str | None = None,
        outcome_query: str | None = None,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "memory.causal_trace",
            episode_id=episode_id,
            outcome_query=outcome_query,
            max_depth=max_depth,
        )

    async def run_scenario(
        self,
        scenario: str,
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "scenario.run",
            scenario=scenario,
            scope=scope,
            scope_id=scope_id,
        )

    async def run_report(
        self,
        report_type: str = "weekly",
        focus: str = "",
        scope: str = "all",
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._call_dict(
            "report.run",
            report_type=report_type,
            focus=focus,
            scope=scope,
            scope_id=scope_id,
        )

    async def memory_profile(self) -> dict[str, Any]:
        return await self._call_dict("memory.profile")

    async def memory_get(self, ids: list[str]) -> dict[str, Any]:
        return await self._call_dict("memory.get", ids=ids)

    async def memory_timeline(self, anchor_id: str, limit: int = 6) -> dict[str, Any]:
        return await self._call_dict("memory.timeline", anchor_id=anchor_id, limit=limit)

    async def memory_recent(self, limit: int = 20) -> dict[str, Any]:
        return await self._call_dict("memory.recent", limit=limit)

    async def timeline_recent(self, limit: int = 40) -> dict[str, Any]:
        return await self._call_dict("timeline.recent", limit=limit)

    async def set_autonomy_level(self, level: str) -> dict[str, Any]:
        return await self._call_dict("autonomy.set_level", level=level)

    async def approve(self, action_id: str) -> dict[str, Any]:
        return await self._call_dict("approve", action_id=action_id)

    async def deny(self, action_id: str) -> dict[str, Any]:
        return await self._call_dict("deny", action_id=action_id)

    async def pending(self) -> list[dict[str, Any]]:
        return await self._call_list("pending")

    async def clear_pending(self) -> dict[str, Any]:
        return await self._call_dict("pending.clear")

    async def browse(self, task: str) -> dict[str, Any]:
        return await self._call_dict("browse", task=task)

    async def list_dir(self, path: str = ".") -> dict[str, Any]:
        return await self._call_dict("workspace.list", path=path)

    async def read_file(self, path: str) -> dict[str, Any]:
        return await self._call_dict("workspace.read", path=path)

    async def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        return await self._call_dict("workspace.write", path=path, content=content, append=append)

    async def patch_file(
        self,
        path: str,
        search: str,
        replace: str,
        cwd: str | None = None,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "path": path,
            "search": search,
            "replace": replace,
            "replace_all": replace_all,
        }
        if cwd is not None:
            params["cwd"] = cwd
        return await self._call_dict("workspace.patch", **params)

    async def exec_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"command": command, "timeout_s": timeout_s}
        if cwd is not None:
            params["cwd"] = cwd
        return await self._call_dict("workspace.exec", **params)

    async def verify(
        self,
        commands: list[str],
        cwd: str | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "commands": commands,
            "timeout_s": timeout_s,
        }
        if cwd is not None:
            params["cwd"] = cwd
        return await self._call_dict("workspace.verify", **params)

    async def git_diff(self, cwd: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if cwd is not None:
            params["cwd"] = cwd
        return await self._call_dict("workspace.git_diff", **params)

    async def git_status(self, cwd: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if cwd is not None:
            params["cwd"] = cwd
        return await self._call_dict("workspace.git_status", **params)

    async def create_worktree(
        self,
        branch: str,
        base_ref: str = "HEAD",
        path: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"branch": branch, "base_ref": base_ref}
        if path is not None:
            params["path"] = path
        return await self._call_dict("workspace.worktree_create", **params)

    async def remove_worktree(self, path: str, force: bool = False) -> dict[str, Any]:
        return await self._call_dict("workspace.worktree_remove", path=path, force=force)

    async def mark_inbox_read(self, message_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"message_id": message_id} if message_id else {}
        return await self._call_dict("inbox.mark_read", **params)

    async def shutdown(self) -> dict[str, Any]:
        return await self._call_dict("shutdown")

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        """Raw RPC call — for internal use."""
        return await self.call(method, **params)
