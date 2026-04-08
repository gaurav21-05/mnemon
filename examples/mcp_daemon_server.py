#!/usr/bin/env python3
"""MCP daemon bridge for a running Mnemon daemon.

Requires:
    pip install "mnemon[mcp]"
    mnemon-daemon start  # in another terminal

Exposes daemon IPC operations as MCP tools so external agents can chat with
Jarvis, inspect goals, create goals, and search episodic memory through the
already-running daemon process.
"""

from __future__ import annotations

from typing import Any

from mnemon.daemon.cli.client import DaemonClient
from mnemon.daemon.config import DaemonConfig

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing MCP dependency. Install with: pip install 'mnemon[mcp]'"
    ) from exc


mcp = FastMCP("mnemon-daemon")
_client = DaemonClient(DaemonConfig().socket_path)


@mcp.tool(name="daemon_chat")
async def daemon_chat(message: str) -> dict[str, Any]:
    """Send a message to the running Jarvis daemon and return the full response."""
    return await _client.chat(message)


@mcp.tool(name="daemon_goals_list")
async def daemon_goals_list() -> dict[str, Any]:
    """List the daemon's active goals."""
    goals = await _client.list_goals()
    return {"count": len(goals), "goals": goals}


@mcp.tool(name="daemon_goals_add")
async def daemon_goals_add(
    description: str,
    priority: float = 0.5,
) -> dict[str, Any]:
    """Create a new goal in the running daemon."""
    return await _client.add_goal(description=description, priority=priority)


@mcp.tool(name="daemon_memory_search")
async def daemon_memory_search(
    query: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search the daemon's episodic memory for relevant past context."""
    return await _client.memory_search(query=query, top_k=top_k)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
