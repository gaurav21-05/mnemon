"""
JarvisWorkspace — bounded local workspace access for the daemon.

Provides explicit file/folder inspection, file writing, and simple command
execution rooted at the daemon's current working directory. This gives Jarvis
practical local tool access without exposing arbitrary host-wide operations.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import shlex
from pathlib import Path
from typing import Any


class JarvisWorkspace:
    """Filesystem and bounded process access rooted at a workspace path."""

    _MAX_READ_BYTES = 48_000
    _MAX_OUTPUT_CHARS = 12_000

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or Path.cwd()).expanduser().resolve()
        self._worktree_base = self._root.parent / f".{self._root.name}-worktrees"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def worktree_base(self) -> Path:
        return self._worktree_base

    def _allowed_roots(self) -> list[Path]:
        roots = [self._root]
        if self._worktree_base.exists():
            roots.append(self._worktree_base.resolve())
        return roots

    def _resolve(self, path: str | None = None) -> Path:
        raw = (path or ".").strip() or "."
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        for base in self._allowed_roots():
            try:
                resolved.relative_to(base)
                return resolved
            except ValueError:
                continue
        raise ValueError(f"Path escapes allowed roots: {resolved}")
        return resolved

    def _trim(self, text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...<truncated>..."

    def _relative_to_known_root(self, target: Path) -> str:
        for base in self._allowed_roots():
            try:
                return str(target.relative_to(base))
            except ValueError:
                continue
        return str(target)

    def _sanitize_branch_name(self, branch: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", branch.strip()).strip("-")
        if not cleaned:
            raise ValueError("branch is required")
        return cleaned

    async def _exec_argv(
        self,
        argv: list[str],
        cwd: Path,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()

        return {
            "argv": argv,
            "cwd": self._relative_to_known_root(cwd),
            "exit_code": process.returncode,
            "stdout": self._trim(stdout.decode("utf-8", errors="replace")),
            "stderr": self._trim(stderr.decode("utf-8", errors="replace")),
            "timed_out": timed_out,
        }

    async def list_dir(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target}")

        entries: list[dict[str, Any]] = []
        for child in sorted(
            target.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        ):
            try:
                stat = child.stat()
                entry_type = "dir" if child.is_dir() else "file"
                entries.append(
                    {
                        "name": child.name,
                        "path": self._relative_to_known_root(child),
                        "type": entry_type,
                        "size": stat.st_size,
                    }
                )
            except OSError:
                continue
        return {
            "root": str(self._root),
            "path": self._relative_to_known_root(target),
            "entries": entries,
        }

    async def read_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {target}")

        data = target.read_bytes()
        text = data[: self._MAX_READ_BYTES].decode("utf-8", errors="replace")
        truncated = len(data) > self._MAX_READ_BYTES
        return {
            "path": self._relative_to_known_root(target),
            "content": text,
            "bytes": len(data),
            "truncated": truncated,
        }

    async def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return {
            "path": self._relative_to_known_root(target),
            "bytes_written": len(content.encode("utf-8")),
            "append": append,
        }

    async def patch_file(
        self,
        path: str,
        search: str,
        replace: str,
        cwd: str | None = None,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        base = self._resolve(cwd)
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {base}")

        target = self._resolve(str(base / path) if not Path(path).is_absolute() else path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {target}")
        if not search:
            raise ValueError("search is required")

        original = target.read_text(encoding="utf-8")
        occurrences = original.count(search)
        if occurrences == 0:
            raise ValueError("search text not found")
        if occurrences > 1 and not replace_all:
            raise ValueError("search text appears multiple times; set replace_all=True")

        updated = (
            original.replace(search, replace)
            if replace_all
            else original.replace(search, replace, 1)
        )
        target.write_text(updated, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=self._relative_to_known_root(target),
                tofile=self._relative_to_known_root(target),
            )
        )
        return {
            "path": self._relative_to_known_root(target),
            "occurrences": occurrences,
            "replace_all": replace_all,
            "diff": self._trim(diff, limit=self._MAX_READ_BYTES),
        }

    async def git_diff(self, cwd: str | None = None) -> dict[str, Any]:
        workdir = self._resolve(cwd)
        result = await self._exec_argv(["git", "diff", "--", "."], workdir, timeout_s=30.0)
        result["command"] = "git diff -- ."
        return result

    async def git_status(self, cwd: str | None = None) -> dict[str, Any]:
        workdir = self._resolve(cwd)
        result = await self._exec_argv(["git", "status", "--short"], workdir, timeout_s=30.0)
        result["command"] = "git status --short"
        return result

    async def verify(
        self,
        commands: list[str],
        cwd: str | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        if not commands:
            raise ValueError("commands are required")
        workdir = self._resolve(cwd)
        if not workdir.is_dir():
            raise NotADirectoryError(f"Not a directory: {workdir}")

        results: list[dict[str, Any]] = []
        passed = True
        for command in commands:
            result = await self.exec_command(command, cwd=str(workdir), timeout_s=timeout_s)
            results.append(result)
            if result["exit_code"] != 0 or result["timed_out"]:
                passed = False
                break
        return {
            "cwd": self._relative_to_known_root(workdir),
            "passed": passed,
            "results": results,
        }

    async def create_worktree(
        self,
        branch: str,
        base_ref: str = "HEAD",
        path: str | None = None,
    ) -> dict[str, Any]:
        safe_branch = self._sanitize_branch_name(branch)
        self._worktree_base.mkdir(parents=True, exist_ok=True)

        if path:
            target = Path(path).expanduser()
            if not target.is_absolute():
                target = self._worktree_base / target
        else:
            target = self._worktree_base / safe_branch.replace("/", "__")
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        result = await self._exec_argv(
            ["git", "worktree", "add", "-b", safe_branch, str(target), base_ref],
            self._root,
            timeout_s=120.0,
        )
        result["branch"] = safe_branch
        result["path"] = str(target)
        result["managed_path"] = self._relative_to_known_root(target)
        return result

    async def remove_worktree(self, path: str, force: bool = False) -> dict[str, Any]:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self._worktree_base / target
        target = target.resolve()
        target.relative_to(self._worktree_base.resolve())

        argv = ["git", "worktree", "remove"]
        if force:
            argv.append("--force")
        argv.append(str(target))
        result = await self._exec_argv(argv, self._root, timeout_s=120.0)
        result["path"] = str(target)
        result["managed_path"] = self._relative_to_known_root(target)
        return result

    async def exec_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        argv = shlex.split(command)
        if not argv:
            raise ValueError("command is required")

        workdir = self._resolve(cwd)
        if not workdir.is_dir():
            raise NotADirectoryError(f"Not a directory: {workdir}")

        result = await self._exec_argv(argv, workdir, timeout_s=timeout_s)
        return {
            "command": command,
            "cwd": result["cwd"],
            "exit_code": result["exit_code"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "timed_out": result["timed_out"],
        }
