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
        except ConnectionRefusedError:
            raise RuntimeError(
                "Connection refused. The daemon may have crashed. "
                "Check logs with: mnemon daemon logs"
            )
        except Exception as exc:
            raise RuntimeError(f"IPC communication failed: {exc}")

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

    async def approve(self, action_id: str) -> dict[str, Any]:
        return await self.call("approve", action_id=action_id)

    async def deny(self, action_id: str) -> dict[str, Any]:
        return await self.call("deny", action_id=action_id)

    async def pending(self) -> list[dict[str, Any]]:
        return await self.call("pending")

    async def shutdown(self) -> dict[str, Any]:
        return await self.call("shutdown")
