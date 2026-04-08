"""Unit tests for IPC natural-language tool intent inference."""

from __future__ import annotations

from mnemon.daemon.ipc import _infer_workspace_intent


def test_infer_list_intent_from_natural_language() -> None:
    intent = _infer_workspace_intent("list files in src/mnemon/daemon")
    assert intent == {"tool": "list", "path": "src/mnemon/daemon"}


def test_infer_read_intent_from_natural_language() -> None:
    intent = _infer_workspace_intent("read `src/mnemon/daemon/ipc.py`")
    assert intent == {"tool": "read", "path": "src/mnemon/daemon/ipc.py"}


def test_infer_write_intent_from_natural_language() -> None:
    intent = _infer_workspace_intent("write notes.txt hello world")
    assert intent == {
        "tool": "write",
        "path": "notes.txt",
        "content": "hello world",
        "append": False,
    }


def test_infer_append_intent_from_natural_language() -> None:
    intent = _infer_workspace_intent("append to `notes.txt` more text")
    assert intent == {
        "tool": "write",
        "path": "notes.txt",
        "content": "more text",
        "append": True,
    }


def test_infer_exec_intent_from_natural_language() -> None:
    intent = _infer_workspace_intent("run command `pytest -q`")
    assert intent == {"tool": "exec", "command": "pytest -q"}


def test_non_tool_message_returns_none() -> None:
    assert _infer_workspace_intent("how are you today?") is None
