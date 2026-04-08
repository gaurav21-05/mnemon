"""Unit tests for SelfImprovementOrchestrator."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from mnemon.daemon.improve import ImprovementSession, Phase, SelfImprovementOrchestrator
from mnemon.daemon.tools.workspace import JarvisWorkspace

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


def _init_git(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True)


def _make_ws(tmp_path: Path) -> JarvisWorkspace:
    return JarvisWorkspace(root=tmp_path)


def _make_llm(
    generate_return: str = "Analysis summary.",
    plan_return: dict[str, Any] | None = None,
) -> Any:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=generate_return)
    llm.generate_structured = AsyncMock(
        return_value=plan_return
        or {
            "summary": "fix a typo",
            "steps": [
                {
                    "description": "Fix typo in dummy file",
                    "file": "dummy.txt",
                    "search": "helo",
                    "replace": "hello",
                }
            ],
        }
    )
    return llm


async def test_status_idle_when_no_session(tmp_path: Path) -> None:
    orch = SelfImprovementOrchestrator(workspace=_make_ws(tmp_path), llm=_make_llm())
    s = orch.status()
    assert s["phase"] == "idle"
    assert s["session_id"] is None


async def test_analyze_returns_summary(tmp_path: Path) -> None:
    _init_git(tmp_path)
    llm = _make_llm(generate_return="The project looks clean.")
    orch = SelfImprovementOrchestrator(workspace=_make_ws(tmp_path), llm=llm)
    result = await orch.analyze()
    assert result["summary"] == "The project looks clean."
    assert "git_status" in result
    assert "test_output" in result


async def test_run_fails_when_no_patch_steps(tmp_path: Path) -> None:
    _init_git(tmp_path)
    llm = _make_llm(plan_return={"summary": "nothing to do", "steps": []})
    orch = SelfImprovementOrchestrator(workspace=_make_ws(tmp_path), llm=llm)
    await orch.run("improve something")
    assert orch.session is not None
    assert orch.session.phase == Phase.FAILED
    assert "no patch steps" in orch.session.error


async def test_abort_when_awaiting_approval(tmp_path: Path) -> None:
    _init_git(tmp_path)
    orch = SelfImprovementOrchestrator(workspace=_make_ws(tmp_path), llm=_make_llm())
    # Manually put session in awaiting_approval state with no worktree
    session = ImprovementSession()
    session.phase = Phase.AWAITING_APPROVAL
    session.worktree_path = ""
    orch._session = session
    result = await orch.abort()
    assert result["ok"] is True
    assert orch.session.phase == Phase.ABORTED


async def test_second_run_raises_when_session_active(tmp_path: Path) -> None:
    _init_git(tmp_path)
    orch = SelfImprovementOrchestrator(workspace=_make_ws(tmp_path), llm=_make_llm())
    session = ImprovementSession()
    session.phase = Phase.AWAITING_APPROVAL
    orch._session = session
    with pytest.raises(RuntimeError, match="already running"):
        await orch.run("another goal")
