"""Unit tests for JarvisWorkspace."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mnemon.daemon.tools.workspace import JarvisWorkspace

pytestmark = pytest.mark.asyncio


async def test_list_dir_returns_entries(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    workspace = JarvisWorkspace(root=tmp_path)

    result = await workspace.list_dir(".")

    assert [entry["path"] for entry in result["entries"]] == ["src", "README.md"]


async def test_read_file_returns_content(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("mnemon", encoding="utf-8")
    workspace = JarvisWorkspace(root=tmp_path)

    result = await workspace.read_file("notes.txt")

    assert result["content"] == "mnemon"
    assert result["truncated"] is False


async def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    workspace = JarvisWorkspace(root=tmp_path)

    result = await workspace.write_file("src/app.py", "print('hi')\n")

    assert result["path"] == "src/app.py"
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "print('hi')\n"


async def test_exec_command_captures_stdout(tmp_path: Path) -> None:
    workspace = JarvisWorkspace(root=tmp_path)

    result = await workspace.exec_command("python3 -c \"print('hi')\"")

    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "hi"
    assert result["timed_out"] is False


async def test_resolve_blocks_path_escape(tmp_path: Path) -> None:
    workspace = JarvisWorkspace(root=tmp_path)

    with pytest.raises(ValueError):
        await workspace.read_file("../outside.txt")


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "mnemon@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Mnemon Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


async def test_patch_file_returns_diff(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    workspace = JarvisWorkspace(root=tmp_path)

    result = await workspace.patch_file("app.py", "hello", "goodbye")

    assert "--- app.py" in result["diff"]
    assert "goodbye" in target.read_text(encoding="utf-8")


async def test_verify_runs_commands_sequentially(tmp_path: Path) -> None:
    workspace = JarvisWorkspace(root=tmp_path)

    result = await workspace.verify(
        ["python3 -c \"print('ok')\"", "python3 -c \"print('done')\""]
    )

    assert result["passed"] is True
    assert len(result["results"]) == 2
    assert result["results"][0]["stdout"].strip() == "ok"


async def test_git_status_and_diff_report_changes(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    target = tmp_path / "app.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "app.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    target.write_text("print('goodbye')\n", encoding="utf-8")
    workspace = JarvisWorkspace(root=tmp_path)

    status = await workspace.git_status()
    diff = await workspace.git_diff()

    assert "M app.py" in status["stdout"]
    assert "goodbye" in diff["stdout"]


async def test_create_and_remove_worktree(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("root\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    workspace = JarvisWorkspace(root=tmp_path)

    created = await workspace.create_worktree("feature/test")
    worktree_path = Path(created["path"])

    assert created["exit_code"] == 0
    assert worktree_path.exists()

    removed = await workspace.remove_worktree(created["path"], force=True)
    assert removed["exit_code"] == 0
