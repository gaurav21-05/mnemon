"""Unit tests for improvement- and memory-search IPC helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemon.daemon.autonomy import AutonomyController
from mnemon.daemon.config import DaemonConfig
from mnemon.daemon.ipc import DaemonIPCServer
from mnemon.daemon.state import DaemonState

pytestmark = pytest.mark.asyncio


class _DummyIdleLoop:
    is_busy = False

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


class _FakeEpisodicMemory:
    async def retrieve(self, query: object) -> object:
        top_k = getattr(query, "top_k", None)
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    content=f"match for top_k={top_k}",
                    score=0.8,
                    source_store="episodic",
                    metadata={"tag": "demo"},
                )
            ]
        )


def _make_server(brain: object) -> DaemonIPCServer:
    return DaemonIPCServer(
        socket_path=Path("/tmp/mnemon-test.sock"),
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(DaemonConfig()),
        idle_loop=_DummyIdleLoop(),
    )


async def test_improve_status_is_idle_before_session_exists() -> None:
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
    )
    server = _make_server(brain)

    assert await server._rpc_improve_status() == {"phase": "idle", "session_id": None}


async def test_improve_approve_and_abort_fail_without_session() -> None:
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
    )
    server = _make_server(brain)

    assert await server._rpc_improve_approve() == {
        "ok": False,
        "error": "no session awaiting approval",
    }
    assert await server._rpc_improve_abort() == {
        "ok": False,
        "error": "no active session to abort",
    }


async def test_memory_search_returns_serialized_retrieval_results() -> None:
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory()),
    )
    server = _make_server(brain)

    result = await server._rpc_memory_search(query="search term", top_k=0)

    assert result["query"] == "search term"
    assert result["results"] == [
        {
            "content": "match for top_k=1",
            "score": 0.8,
            "source": "episodic",
            "metadata": {"tag": "demo"},
        }
    ]
