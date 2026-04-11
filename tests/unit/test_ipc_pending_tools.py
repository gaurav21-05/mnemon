"""Unit tests for pending approval execution of tool steps."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemon.daemon.autonomy import AutonomyController
from mnemon.daemon.config import AutonomyLevel, DaemonConfig
from mnemon.daemon.ipc import DaemonIPCServer
from mnemon.daemon.state import DaemonState

pytestmark = pytest.mark.asyncio


class _DummyIdleLoop:
    is_busy = False

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


class _DummyLLM:
    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict,
        **kwargs: object,
    ) -> dict:
        return {"steps": []}


class _FakeWorkspace:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str, bool]] = []

    async def write_file(self, path: str, content: str, append: bool = False) -> dict[str, object]:
        self.writes.append((path, content, append))
        return {"path": path, "bytes_written": len(content.encode("utf-8")), "append": append}


def _make_server() -> tuple[DaemonIPCServer, _FakeWorkspace]:
    llm = _DummyLLM()
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(_llm=llm)),
    )
    config = DaemonConfig(autonomy_level=AutonomyLevel.SUGGEST)
    server = DaemonIPCServer(
        socket_path=Path("/tmp/mnemon-test.sock"),
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(config),
        idle_loop=_DummyIdleLoop(),
    )
    workspace = _FakeWorkspace()
    server._workspace = workspace
    return server, workspace


async def test_write_intent_requires_approval_and_executes_after_approve() -> None:
    server, workspace = _make_server()

    result = await server._handle_tool_request("write notes.txt hello world")

    assert result is not None
    assert "Pending approval" in result["reply"]
    pending = server._autonomy.get_pending()
    assert len(pending) == 1
    action_id = str(pending[0].id)

    approval = await server._rpc_approve(action_id)
    assert approval["approved"] is True
    assert workspace.writes == [("notes.txt", "hello world", False)]


async def test_list_intent_executes_without_approval() -> None:
    server, _workspace = _make_server()

    async def fake_list(path: str = ".") -> dict[str, object]:
        return {"entries": [{"type": "file", "path": "notes.txt"}]}

    server._rpc_workspace_list = fake_list  # type: ignore[method-assign]

    result = await server._handle_tool_request("list files in .")

    assert result is not None
    assert result["reply"].strip() == "file  notes.txt"
    assert server._autonomy.get_pending() == []


async def test_slash_write_requires_approval() -> None:
    server, _workspace = _make_server()

    result = await server._handle_tool_command("/write notes.txt hello")

    assert result is not None
    assert "Pending approval" in result["reply"]
    pending = server._autonomy.get_pending()
    assert len(pending) == 1


async def test_write_intent_auto_executes_under_default_semi_auto() -> None:
    llm = _DummyLLM()
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(_llm=llm)),
    )
    server = DaemonIPCServer(
        socket_path=Path("/tmp/mnemon-test.sock"),
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(DaemonConfig()),
        idle_loop=_DummyIdleLoop(),
    )
    workspace = _FakeWorkspace()
    server._workspace = workspace

    result = await server._handle_tool_request("write notes.txt hello world")

    assert result is not None
    assert "Pending approval" not in result["reply"]
    assert workspace.writes == [("notes.txt", "hello world", False)]


async def test_execution_followup_uses_recent_chat_context() -> None:
    class _HistoryAwareLLM:
        async def generate_structured(
            self,
            prompt: str,
            response_schema: dict,
            **kwargs: object,
        ) -> dict:
            del response_schema, kwargs
            assert "designs" in prompt
            assert "resume" in prompt
            assert "single page" in prompt
            if "Previous tool results:\n(no tool results yet)" not in prompt:
                return {"action": "respond", "reply": "Built the page."}
            return {
                "action": "write",
                "path": "portfolio/index.html",
                "content": "<html>built from resume and designs</html>",
                "append": False,
            }

    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(_llm=_HistoryAwareLLM())),
    )
    server = DaemonIPCServer(
        socket_path=Path("/tmp/mnemon-test.sock"),
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(DaemonConfig()),
        idle_loop=_DummyIdleLoop(),
    )
    workspace = _FakeWorkspace()
    server._workspace = workspace
    server._chat_history.extend(
        [
            {"role": "user", "content": "take design from /home/rohit/Downloads/designs"},
            {"role": "assistant", "content": "I can use that design source."},
            {"role": "user", "content": "build all sections from my resume"},
            {"role": "assistant", "content": "I will build all sections from your resume."},
            {"role": "user", "content": "single page"},
            {"role": "assistant", "content": "I'll make it a single page site."},
        ]
    )

    result = await server._handle_tool_request("start building now")

    assert result is not None
    assert "Pending approval" not in result["reply"]
    assert workspace.writes == [
        ("portfolio/index.html", "<html>built from resume and designs</html>", False)
    ]


async def test_tool_planner_falls_back_to_plain_json_when_structured_fails() -> None:
    class _FallbackLLM:
        async def generate_structured(
            self,
            prompt: str,
            response_schema: dict,
            **kwargs: object,
        ) -> dict:
            del prompt, response_schema, kwargs
            raise ValueError("schema mode unsupported")

        async def generate(self, prompt: str, **kwargs: object) -> str:
            del kwargs
            assert "write notes.txt hello world" in prompt
            if "Previous tool results:\n(no tool results yet)" not in prompt:
                return '{"action":"respond","reply":"Wrote the file."}'
            return '{"action":"write","path":"notes.txt","content":"hello world","append":false}'

    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(_llm=_FallbackLLM())),
    )
    server = DaemonIPCServer(
        socket_path=Path("/tmp/mnemon-test.sock"),
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(DaemonConfig()),
        idle_loop=_DummyIdleLoop(),
    )
    workspace = _FakeWorkspace()
    server._workspace = workspace

    result = await server._handle_tool_request("write notes.txt hello world")

    assert result is not None
    assert "Pending approval" not in result["reply"]
    assert workspace.writes == [("notes.txt", "hello world", False)]
