"""
DaemonClient — IPC client for communicating with a running daemon.

Connects to the daemon's Unix domain socket and sends JSON-RPC requests.
Used by the CLI and can also be used programmatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
                data = await stream.receive(65536)
                response = json.loads(data.decode("utf-8"))
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

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def chat(self, message: str) -> dict[str, Any]:
        return await self.call("chat", message=message)

    async def status(self) -> dict[str, Any]:
        return await self.call("status")

    async def thoughts(self, limit: int = 10) -> list[dict[str, Any]]:
        return await self.call("thoughts", limit=limit)

    async def list_goals(self) -> list[dict[str, Any]]:
        return await self.call("goals.list")

    async def add_goal(self, description: str, priority: float = 0.5) -> dict[str, Any]:
        return await self.call("goals.add", description=description, priority=priority)

    async def improve_analyze(self) -> dict[str, Any]:
        return await self.call("improve.analyze")

    async def improve_start(self, goal: str = "improve code quality") -> dict[str, Any]:
        return await self.call("improve.start", goal=goal)

    async def improve_status(self) -> dict[str, Any]:
        return await self.call("improve.status")

    async def improve_approve(self) -> dict[str, Any]:
        return await self.call("improve.approve")

    async def improve_abort(self) -> dict[str, Any]:
        return await self.call("improve.abort")

    async def memory_search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return await self.call("memory.search", query=query, top_k=top_k)

    async def approve(self, action_id: str) -> dict[str, Any]:
        return await self.call("approve", action_id=action_id)

    async def deny(self, action_id: str) -> dict[str, Any]:
        return await self.call("deny", action_id=action_id)

    async def pending(self) -> list[dict[str, Any]]:
        return await self.call("pending")

    async def browse(self, task: str) -> dict[str, Any]:
        return await self.call("browse", task=task)

    async def list_dir(self, path: str = ".") -> dict[str, Any]:
        return await self.call("workspace.list", path=path)

    async def read_file(self, path: str) -> dict[str, Any]:
        return await self.call("workspace.read", path=path)

    async def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        return await self.call("workspace.write", path=path, content=content, append=append)

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
        return await self.call("workspace.patch", **params)

    async def exec_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"command": command, "timeout_s": timeout_s}
        if cwd is not None:
            params["cwd"] = cwd
        return await self.call("workspace.exec", **params)

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
        return await self.call("workspace.verify", **params)

    async def git_diff(self, cwd: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if cwd is not None:
            params["cwd"] = cwd
        return await self.call("workspace.git_diff", **params)

    async def git_status(self, cwd: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if cwd is not None:
            params["cwd"] = cwd
        return await self.call("workspace.git_status", **params)

    async def create_worktree(
        self,
        branch: str,
        base_ref: str = "HEAD",
        path: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"branch": branch, "base_ref": base_ref}
        if path is not None:
            params["path"] = path
        return await self.call("workspace.worktree_create", **params)

    async def remove_worktree(self, path: str, force: bool = False) -> dict[str, Any]:
        return await self.call("workspace.worktree_remove", path=path, force=force)

    async def mark_inbox_read(self, message_id: str | None = None) -> dict[str, Any]:
        params = {"message_id": message_id} if message_id else {}
        return await self.call("inbox.mark_read", **params)

    async def shutdown(self) -> dict[str, Any]:
        return await self.call("shutdown")

    async def _rpc(self, method: str, params: dict) -> Any:
        """Raw RPC call — for internal use."""
        return await self.call(method, **params)
