# Complete Remaining Changes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three remaining features: self-improvement orchestration, web UI enhancements (goal creation + memory search), and an MCP daemon bridge example.

**Architecture:** Bug fixes #3, #4, and #7 are already implemented in the codebase. The genuine remaining work is three independent additions: (1) a supervised self-improvement workflow in `daemon/improve.py` wired into the IPC server, (2) two new HTTP endpoints and HTML for goal creation and memory search in the web UI, and (3) an example MCP server that bridges the running daemon's IPC to MCP tools.

**Tech Stack:** Python 3.12, anyio, aiohttp (webui), MCP SDK (mcp_daemon), pytest-asyncio for tests.

---

## Files Modified / Created

| File | Role |
|------|------|
| `src/mnemon/daemon/improve.py` | New — `SelfImprovementOrchestrator` with 6-phase workflow |
| `src/mnemon/daemon/ipc.py` | Add 3 handlers: `improve.analyze`, `improve.start`, `improve.status`; add `memory.search` handler |
| `src/mnemon/daemon/webui.py` | Add POST `/api/goals` + GET `/api/memory/search` endpoints; add HTML form + search widget |
| `src/mnemon/daemon/cli/app.py` | Add `mnemon-daemon improve` command |
| `src/mnemon/daemon/cli/client.py` | Add `improve_analyze`, `improve_start`, `improve_status`, `memory_search` convenience methods |
| `examples/mcp_daemon_server.py` | New — MCP server bridging running daemon via IPC |
| `tests/unit/test_improve.py` | New — unit tests for self-improvement session phases |

---

## Task 1 — Self-Improvement Orchestrator (`daemon/improve.py`)

**Files:**
- Create: `src/mnemon/daemon/improve.py`
- Test: `tests/unit/test_improve.py`

The orchestrator runs a 6-phase supervised workflow. Each phase is async and returns a structured result dict. The IPC server holds one active session at a time.

**Phases:**
1. `analyze` — run `git status --short` + `pytest tests/unit -q --tb=no` to assess current state; ask LLM to summarise issues and propose improvements
2. `plan` — LLM produces an ordered list of `{description, file, search, replace}` patch steps
3. `worktree` — `git worktree add -b jarvis/improve-<ts> ~/.mnemon-worktrees/improve-<ts> HEAD`
4. `patch` — apply each patch step sequentially
5. `verify` — run `pytest tests/unit -q --tb=short` in worktree; stop on first failure
6. `approve_or_abort` — request human approval; on approve run `git merge` + remove worktree; on deny remove worktree and return

```python
# Complete file: src/mnemon/daemon/improve.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from mnemon.daemon.tools.workspace import JarvisWorkspace

logger = logging.getLogger(__name__)

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "file": {"type": "string"},
                    "search": {"type": "string"},
                    "replace": {"type": "string"},
                },
                "required": ["description", "file", "search", "replace"],
            },
        },
    },
    "required": ["summary", "steps"],
}


class Phase(StrEnum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    WORKTREE = "worktree"
    PATCHING = "patching"
    VERIFYING = "verifying"
    AWAITING_APPROVAL = "awaiting_approval"
    MERGING = "merging"
    DONE = "done"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class PatchStep:
    description: str
    file: str
    search: str
    replace: str


@dataclass
class ImprovementSession:
    id: UUID = field(default_factory=uuid4)
    phase: Phase = Phase.IDLE
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_summary: str = ""
    plan_summary: str = ""
    patch_steps: list[PatchStep] = field(default_factory=list)
    steps_applied: int = 0
    worktree_path: str = ""
    branch: str = ""
    verify_output: str = ""
    verify_passed: bool = False
    error: str = ""
    approval_id: UUID = field(default_factory=uuid4)


class SelfImprovementOrchestrator:
    """Supervised 6-phase self-improvement workflow.

    Analyzes the current repo state, plans targeted patches, applies them in
    an isolated worktree, verifies, and gates on human approval before merging.
    """

    def __init__(self, workspace: JarvisWorkspace, llm: Any) -> None:
        self._workspace = workspace
        self._llm = llm
        self._session: ImprovementSession | None = None

    @property
    def session(self) -> ImprovementSession | None:
        return self._session

    def status(self) -> dict[str, Any]:
        if self._session is None:
            return {"phase": "idle", "session_id": None}
        s = self._session
        return {
            "session_id": str(s.id),
            "phase": s.phase,
            "started_at": s.started_at.isoformat(),
            "analysis_summary": s.analysis_summary,
            "plan_summary": s.plan_summary,
            "patch_steps_total": len(s.patch_steps),
            "steps_applied": s.steps_applied,
            "worktree_path": s.worktree_path,
            "branch": s.branch,
            "verify_passed": s.verify_passed,
            "verify_output": s.verify_output[:2000],
            "error": s.error,
            "approval_id": str(s.approval_id),
        }

    async def analyze(self) -> dict[str, Any]:
        """Run analysis phase only — returns summary without starting full workflow."""
        ws = self._workspace
        git_result = await ws.git_status()
        git_out = git_result.get("stdout", "")

        verify_result = await ws.verify(
            commands=["python -m pytest tests/unit -q --tb=no --no-header"],
            timeout_s=120.0,
        )
        verify_out = "\n".join(
            r.get("stdout", "") + r.get("stderr", "")
            for r in verify_result.get("results", [])
        )

        prompt = (
            "You are reviewing a Python project for self-improvement opportunities.\n\n"
            f"Git status:\n{git_out or '(clean)'}\n\n"
            f"Test output:\n{verify_out[:3000] or '(no output)'}\n\n"
            "Summarise: what issues exist and what would make the most impactful improvement? "
            "Be specific and concise (3-6 sentences)."
        )
        try:
            summary = await self._llm.generate(prompt)
        except Exception as exc:
            summary = f"LLM unavailable: {exc}"

        return {
            "git_status": git_out,
            "test_output": verify_out[:3000],
            "summary": summary,
        }

    async def run(self, goal: str) -> None:
        """Run the complete 6-phase workflow for the given improvement goal.

        Intended to be launched as a background task. Poll status() for progress.
        """
        if self._session is not None and self._session.phase not in (
            Phase.DONE, Phase.ABORTED, Phase.FAILED
        ):
            raise RuntimeError("A session is already running")

        session = ImprovementSession()
        self._session = session
        ws = self._workspace

        try:
            # --- Phase 1: ANALYZE ---
            session.phase = Phase.ANALYZING
            analysis = await self.analyze()
            session.analysis_summary = analysis["summary"]

            # --- Phase 2: PLAN ---
            session.phase = Phase.PLANNING
            plan_prompt = (
                f"You are an AI that improves its own codebase. Goal: {goal}\n\n"
                f"Current state: {session.analysis_summary}\n\n"
                "Produce a JSON plan with key 'summary' (one sentence) and 'steps' (list of patches).\n"
                "Each step: {description, file (relative path), search (exact text to find), replace (replacement text)}.\n"
                "Only include steps where search text will be found exactly in the file.\n"
                "Max 5 steps. Return JSON only."
            )
            try:
                plan = await self._llm.generate_structured(plan_prompt, _PLAN_SCHEMA)
                session.plan_summary = plan.get("summary", "")
                session.patch_steps = [
                    PatchStep(**s)
                    for s in plan.get("steps", [])
                    if s.get("file") and s.get("search")
                ]
            except Exception as exc:
                session.error = f"Planning failed: {exc}"
                session.phase = Phase.FAILED
                return

            if not session.patch_steps:
                session.error = "LLM produced no patch steps"
                session.phase = Phase.FAILED
                return

            # --- Phase 3: WORKTREE ---
            session.phase = Phase.WORKTREE
            ts = int(time.time())
            branch = f"jarvis/improve-{ts}"
            session.branch = branch
            wt_result = await ws.create_worktree(branch=branch)
            if wt_result.get("exit_code", 1) != 0:
                session.error = wt_result.get("stderr", "worktree creation failed")
                session.phase = Phase.FAILED
                return
            session.worktree_path = wt_result.get("path", "")

            # --- Phase 4: PATCH ---
            session.phase = Phase.PATCHING
            for step in session.patch_steps:
                try:
                    await ws.patch_file(
                        path=step.file,
                        search=step.search,
                        replace=step.replace,
                        cwd=session.worktree_path,
                    )
                    session.steps_applied += 1
                    logger.info("Applied patch step: %s", step.description)
                except Exception as exc:
                    logger.warning("Patch step failed (%s): %s", step.description, exc)

            # --- Phase 5: VERIFY ---
            session.phase = Phase.VERIFYING
            verify = await ws.verify(
                commands=["python -m pytest tests/unit -q --tb=short --no-header"],
                cwd=session.worktree_path,
                timeout_s=180.0,
            )
            session.verify_passed = verify.get("passed", False)
            session.verify_output = "\n".join(
                r.get("stdout", "") + r.get("stderr", "")
                for r in verify.get("results", [])
            )

            # --- Phase 6: AWAIT APPROVAL ---
            session.phase = Phase.AWAITING_APPROVAL
            logger.info(
                "Improvement session awaiting approval (id=%s, verify_passed=%s)",
                session.approval_id,
                session.verify_passed,
            )

        except Exception as exc:
            session.error = str(exc)
            session.phase = Phase.FAILED
            logger.exception("Self-improvement session failed")

    async def approve(self) -> dict[str, Any]:
        """Merge the worktree branch into HEAD and remove it."""
        s = self._session
        if s is None or s.phase != Phase.AWAITING_APPROVAL:
            return {"ok": False, "error": "no session awaiting approval"}

        s.phase = Phase.MERGING
        ws = self._workspace
        merge_result = await ws._exec_argv(
            ["git", "merge", "--no-ff", s.branch, "-m", f"Jarvis: {s.plan_summary or s.branch}"],
            ws.root,
        )
        if merge_result.get("exit_code", 1) != 0:
            s.error = merge_result.get("stderr", "merge failed")
            s.phase = Phase.FAILED
            return {"ok": False, "error": s.error}

        await ws.remove_worktree(s.worktree_path, force=True)
        s.phase = Phase.DONE
        return {"ok": True, "branch": s.branch, "summary": s.plan_summary}

    async def abort(self) -> dict[str, Any]:
        """Remove worktree without merging."""
        s = self._session
        if s is None or s.phase not in (Phase.AWAITING_APPROVAL, Phase.VERIFYING, Phase.PATCHING):
            return {"ok": False, "error": "no active session to abort"}

        if s.worktree_path:
            try:
                await self._workspace.remove_worktree(s.worktree_path, force=True)
            except Exception:
                pass
        s.phase = Phase.ABORTED
        return {"ok": True}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_improve.py`:

```python
"""Unit tests for SelfImprovementOrchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemon.daemon.improve import ImprovementSession, Phase, SelfImprovementOrchestrator
from mnemon.daemon.tools.workspace import JarvisWorkspace


def _make_workspace(tmp_path: Path) -> JarvisWorkspace:
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
                    "description": "Fix typo",
                    "file": "dummy.txt",
                    "search": "helo",
                    "replace": "hello",
                }
            ],
        }
    )
    return llm


pytestmark = pytest.mark.asyncio


async def test_status_returns_idle_when_no_session(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    orch = SelfImprovementOrchestrator(workspace=ws, llm=_make_llm())
    status = orch.status()
    assert status["phase"] == "idle"
    assert status["session_id"] is None


async def test_analyze_returns_summary(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    # Make it a git repo so git_status doesn't fail
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

    llm = _make_llm(generate_return="The project looks good.")
    orch = SelfImprovementOrchestrator(workspace=ws, llm=llm)
    result = await orch.analyze()
    assert "summary" in result
    assert result["summary"] == "The project looks good."


async def test_run_fails_when_no_patch_steps(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

    ws = _make_workspace(tmp_path)
    llm = _make_llm(plan_return={"summary": "nothing to do", "steps": []})
    orch = SelfImprovementOrchestrator(workspace=ws, llm=llm)
    await orch.run("improve something")
    assert orch.session is not None
    assert orch.session.phase == Phase.FAILED
    assert "no patch steps" in orch.session.error


async def test_abort_clears_session(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

    ws = _make_workspace(tmp_path)
    llm = _make_llm(plan_return={"summary": "nothing to do", "steps": []})
    orch = SelfImprovementOrchestrator(workspace=ws, llm=llm)
    # Manually put session in awaiting_approval
    session = ImprovementSession()
    session.phase = Phase.AWAITING_APPROVAL
    session.worktree_path = ""
    orch._session = session

    result = await orch.abort()
    assert result["ok"] is True
    assert orch.session.phase == Phase.ABORTED


async def test_second_run_raises_when_session_active(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

    ws = _make_workspace(tmp_path)
    llm = _make_llm()
    orch = SelfImprovementOrchestrator(workspace=ws, llm=llm)
    session = ImprovementSession()
    session.phase = Phase.AWAITING_APPROVAL
    orch._session = session

    with pytest.raises(RuntimeError, match="already running"):
        await orch.run("another goal")
```

- [ ] **Step 2: Run tests to see them fail**

```bash
cd /home/rohit/mnemon
.venv/bin/python -m pytest tests/unit/test_improve.py -v
```

Expected: `ModuleNotFoundError: No module named 'mnemon.daemon.improve'`

- [ ] **Step 3: Write `src/mnemon/daemon/improve.py`**

Create the file with the exact content from the code block at the top of this task.

- [ ] **Step 4: Run tests again — must pass**

```bash
cd /home/rohit/mnemon
.venv/bin/python -m pytest tests/unit/test_improve.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/rohit/mnemon
git add src/mnemon/daemon/improve.py tests/unit/test_improve.py
git commit -m "feat(daemon): add SelfImprovementOrchestrator with 6-phase supervised workflow"
```

---

## Task 2 — Wire Improvement Orchestrator into IPC + CLI

**Files:**
- Modify: `src/mnemon/daemon/ipc.py` — add improve handlers + memory.search handler
- Modify: `src/mnemon/daemon/cli/client.py` — add convenience methods
- Modify: `src/mnemon/daemon/cli/app.py` — add `improve` subcommand

- [ ] **Step 1: Add `SelfImprovementOrchestrator` instance to `DaemonIPCServer.__init__`**

In `src/mnemon/daemon/ipc.py`, import at the top (after existing imports):

```python
from mnemon.daemon.improve import SelfImprovementOrchestrator
```

In `DaemonIPCServer.__init__` (after `self._pending_tool_actions`):

```python
        self._improver: SelfImprovementOrchestrator | None = None
```

At the end of `__init__`, add three new handler entries to `self._handlers`:

```python
            "improve.analyze": self._rpc_improve_analyze,
            "improve.start": self._rpc_improve_start,
            "improve.status": self._rpc_improve_status,
            "improve.approve": self._rpc_improve_approve,
            "improve.abort": self._rpc_improve_abort,
            "memory.search": self._rpc_memory_search,
```

- [ ] **Step 2: Add `_get_improver` helper + 5 RPC handlers in `ipc.py`**

Add these methods to `DaemonIPCServer`, before `_rpc_chat`:

```python
    def _get_improver(self) -> SelfImprovementOrchestrator:
        if self._improver is None:
            from mnemon.daemon.tools.workspace import JarvisWorkspace
            ws = JarvisWorkspace()
            try:
                llm = self._brain.control.goals._llm
            except Exception:
                raise RuntimeError("LLM not available")
            self._improver = SelfImprovementOrchestrator(workspace=ws, llm=llm)
        return self._improver

    async def _rpc_improve_analyze(self) -> dict[str, Any]:
        improver = self._get_improver()
        return await improver.analyze()

    async def _rpc_improve_start(self, goal: str = "improve code quality") -> dict[str, Any]:
        improver = self._get_improver()
        import anyio
        # Run in background — caller polls status
        async def _run() -> None:
            try:
                await improver.run(goal)
            except Exception:
                logger.exception("Self-improvement session failed")
        self._background_tasks = getattr(self, "_background_tasks", [])
        # Fire and forget — we can't use task groups from here easily
        asyncio.create_task(_run())
        return {"started": True, "goal": goal}

    async def _rpc_improve_status(self) -> dict[str, Any]:
        improver = self._get_improver()
        return improver.status()

    async def _rpc_improve_approve(self) -> dict[str, Any]:
        improver = self._get_improver()
        return await improver.approve()

    async def _rpc_improve_abort(self) -> dict[str, Any]:
        improver = self._get_improver()
        return await improver.abort()

    async def _rpc_memory_search(self, query: str = "", top_k: int = 10) -> dict[str, Any]:
        if not query:
            return {"results": []}
        try:
            from mnemon.core.models import RetrievalQuery
            q = RetrievalQuery(query_text=query, top_k=top_k)
            try:
                embedding = await self._brain.providers.embedding.embed(query)
                q = RetrievalQuery(query_text=query, query_embedding=embedding, top_k=top_k)
            except Exception:
                pass
            results = await self._brain.memory.episodic.retrieve(q)
            items = [
                {"content": item.content, "score": item.score, "source": item.source_store}
                for item in results.items
            ]
            return {"results": items, "query": query}
        except Exception as exc:
            return {"results": [], "error": str(exc)}
```

Note: The `asyncio.create_task` import is already available since `ipc.py` imports `anyio`. Add `import asyncio` to the top of `ipc.py` if not already present.

- [ ] **Step 3: Check if `asyncio` is already imported in `ipc.py`**

```bash
grep "^import asyncio" /home/rohit/mnemon/src/mnemon/daemon/ipc.py
```

If no output, add `import asyncio` after the `from __future__` line.

- [ ] **Step 4: Add convenience methods to `client.py`**

In `src/mnemon/daemon/cli/client.py`, add after `add_goal`:

```python
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
```

- [ ] **Step 5: Add `improve` subcommand to `cli/app.py`**

Read the bottom of `cli/app.py` to find the argument parser setup, then add:

```python
def cmd_improve(args: argparse.Namespace) -> None:
    """Run a supervised self-improvement cycle."""
    client = _get_client()

    if args.analyze:
        result = anyio.run(client.improve_analyze)
        print("\n=== Analysis ===")
        print(result.get("summary", "(no summary)"))
        print()
        print("Git status:", result.get("git_status", "(clean)") or "(clean)")
        return

    if args.abort:
        result = anyio.run(client.improve_abort)
        print("Aborted." if result.get("ok") else f"Error: {result.get('error')}")
        return

    if args.approve:
        result = anyio.run(client.improve_approve)
        if result.get("ok"):
            print(f"Merged branch '{result.get('branch')}' into HEAD.")
            print(f"Summary: {result.get('summary')}")
        else:
            print(f"Error: {result.get('error')}")
        return

    goal = args.goal or "improve code quality and fix any failing tests"
    print(f"Starting self-improvement session: {goal}")
    print("Run 'mnemon-daemon improve --status' to poll progress.")
    anyio.run(client.improve_start, goal)


def cmd_improve_status(args: argparse.Namespace) -> None:
    client = _get_client()
    result = anyio.run(client.improve_status)
    phase = result.get("phase", "idle")
    print(f"Phase: {phase}")
    if result.get("analysis_summary"):
        print(f"Analysis: {result['analysis_summary']}")
    if result.get("plan_summary"):
        print(f"Plan: {result['plan_summary']}")
    steps_total = result.get("patch_steps_total", 0)
    steps_done = result.get("steps_applied", 0)
    if steps_total:
        print(f"Patches: {steps_done}/{steps_total} applied")
    if result.get("verify_output"):
        print(f"Verify: {'PASS' if result.get('verify_passed') else 'FAIL'}")
        print(result["verify_output"][:800])
    if result.get("error"):
        print(f"Error: {result['error']}")
    if phase == "awaiting_approval":
        approval_id = result.get("approval_id")
        print(f"\nReady for approval (id={approval_id})")
        print("  mnemon-daemon improve --approve    # merge and clean up")
        print("  mnemon-daemon improve --abort      # discard changes")
```

In the argument parser section, add the `improve` subparser. Find where `subparsers.add_parser("approve"...)` is defined and add after it:

```python
    # improve
    p_improve = subparsers.add_parser("improve", help="Run a supervised self-improvement cycle")
    p_improve.add_argument("goal", nargs="?", help="Improvement goal description")
    p_improve.add_argument("--analyze", action="store_true", help="Analyze only, don't start")
    p_improve.add_argument("--status", action="store_true", help="Show current session status")
    p_improve.add_argument("--approve", action="store_true", help="Approve pending improvement")
    p_improve.add_argument("--abort", action="store_true", help="Abort current session")
    p_improve.set_defaults(func=lambda a: cmd_improve_status(a) if a.status else cmd_improve(a))
```

- [ ] **Step 6: Run full unit suite to confirm nothing broken**

```bash
cd /home/rohit/mnemon
.venv/bin/python -m pytest tests/unit/ -q --tb=short
```

Expected: 237+ passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
cd /home/rohit/mnemon
git add src/mnemon/daemon/ipc.py src/mnemon/daemon/cli/client.py src/mnemon/daemon/cli/app.py
git commit -m "feat(daemon): wire self-improvement orchestrator into IPC + CLI"
```

---

## Task 3 — Web UI: Goal Creation Form + Memory Search

**Files:**
- Modify: `src/mnemon/daemon/webui.py` — add POST `/api/goals` + GET `/api/memory/search` + HTML updates

The web UI is a single large file. We make surgical additions only.

- [ ] **Step 1: Add two new route handlers before `create_app`**

In `src/mnemon/daemon/webui.py`, add before the `create_app` function (around line 1448):

```python
async def goals_add_handler(request: web.Request) -> web.Response:
    """POST /api/goals — add a new goal via daemon IPC."""
    try:
        body = await request.json()
        description = body.get("description", "").strip()
        priority = float(body.get("priority", 0.5))
        if not description:
            return web.json_response({"error": "description is required"}, status=400)
        client = DaemonClient(SOCKET_PATH)
        result = await asyncio.wait_for(
            client.add_goal(description=description, priority=priority), timeout=10.0
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def memory_search_handler(request: web.Request) -> web.Response:
    """GET /api/memory/search?q=... — search episodic memory."""
    query = request.rel_url.query.get("q", "").strip()
    if not query:
        return web.json_response({"results": []})
    try:
        client = DaemonClient(SOCKET_PATH)
        result = await asyncio.wait_for(
            client.call("memory.search", query=query, top_k=10), timeout=15.0
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"results": [], "error": str(exc)}, status=502)
```

- [ ] **Step 2: Register the two routes in `create_app`**

Change the `create_app` function from:

```python
def create_app(socket_path: Path | None = None) -> web.Application:
    app = web.Application()
    if socket_path:
        app["socket_path"] = socket_path
    app.router.add_get("/", lambda r: web.Response(text=HTML, content_type="text/html"))
    app.router.add_get("/events", sse_stream)
    app.router.add_post("/chat", chat_handler)
    return app
```

To:

```python
def create_app(socket_path: Path | None = None) -> web.Application:
    app = web.Application()
    if socket_path:
        app["socket_path"] = socket_path
    app.router.add_get("/", lambda r: web.Response(text=HTML, content_type="text/html"))
    app.router.add_get("/events", sse_stream)
    app.router.add_post("/chat", chat_handler)
    app.router.add_post("/api/goals", goals_add_handler)
    app.router.add_get("/api/memory/search", memory_search_handler)
    return app
```

- [ ] **Step 3: Add HTML for goal creation form + memory search widget**

In the `HTML` string in `webui.py`, find the goals panel section (search for `goals-panel` or `id="goals"`). Add the goal creation form inside the goals panel body. Find the exact text to splice — look for where the goals list is rendered in the HTML template.

Find this string in `HTML` (it will be inside the `<script>` section, look for `function renderGoals`):

```javascript
function renderGoals(goals) {
```

Before that function, add a new function and wire the form:

The JavaScript additions go in the `<script>` block. Find `// --- Chat ---` or the end of the script block and add before it:

```javascript
// --- Goals form ---
document.getElementById('goal-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('goal-input');
  const desc = input.value.trim();
  if (!desc) return;
  input.disabled = true;
  try {
    const res = await fetch('/api/goals', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: desc, priority: 0.5}),
    });
    if (res.ok) {
      input.value = '';
    }
  } catch (e) { console.error(e); }
  input.disabled = false;
  input.focus();
});

// --- Memory search ---
const searchInput = document.getElementById('mem-search-input');
const searchResults = document.getElementById('mem-search-results');
let searchTimer = null;
if (searchInput) {
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(async () => {
      const q = searchInput.value.trim();
      if (!q) { searchResults.innerHTML = ''; return; }
      try {
        const res = await fetch(`/api/memory/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        const items = data.results || [];
        searchResults.innerHTML = items.length
          ? items.map(it => `<div class="search-result"><span class="search-score">${(it.score * 100).toFixed(0)}%</span> ${escHtml(it.content.slice(0, 200))}</div>`).join('')
          : '<div class="empty">No results</div>';
      } catch (e) { searchResults.innerHTML = '<div class="empty">Search failed</div>'; }
    }, 350);
  });
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
```

Find the goals panel HTML (search for `goals-panel-body` or equivalent class). The goals panel body currently shows a list of goals. Add a form at the bottom:

Find this text in `HTML` (the closing tag of the goals panel, something like `</div><!-- end goals panel -->`):

Look for where the goals list `<ul>` or goals empty state is. Add after the goals list container:

```html
<form id="goal-form" style="display:flex;gap:8px;padding:10px 14px 14px;flex-shrink:0;">
  <input id="goal-input" class="chat-input" style="flex:1;min-height:38px;" placeholder="Add a new goal…" autocomplete="off" />
  <button type="submit" class="send-btn" style="min-height:38px;padding:6px 14px;font-size:14px;">Add</button>
</form>
```

Add a memory search widget. Find the topbar section (the `<div class="topbar-inner">`) and add a search input after the brand div:

```html
<div style="display:flex;flex-direction:column;gap:4px;flex:0 0 260px;position:relative;">
  <input id="mem-search-input" class="chat-input" style="min-height:36px;font-size:13px;" placeholder="Search memory…" autocomplete="off" />
  <div id="mem-search-results" style="position:absolute;top:40px;left:0;right:0;background:#fff;border:1px solid var(--oat);border-radius:12px;box-shadow:var(--clay-shadow);z-index:20;max-height:240px;overflow-y:auto;"></div>
</div>
```

Add CSS for `.search-result` and `.search-score` in the `<style>` block (find the `.empty` selector and add after):

```css
  .search-result {
    padding: 8px 12px;
    border-bottom: 1px solid var(--oat-light);
    font-size: 13px;
    line-height: 1.4;
  }
  .search-result:last-child { border-bottom: none; }
  .search-score {
    display: inline-block;
    min-width: 36px;
    font-size: 11px;
    font-weight: 600;
    color: var(--matcha-600);
    font-family: var(--font-mono);
    margin-right: 6px;
  }
```

**Important:** The HTML modifications in this task are surgical. Read the exact text around each insertion point in `webui.py` before editing to get the precise `old_string` for each Edit call.

- [ ] **Step 4: Smoke-test by checking the Python file parses**

```bash
cd /home/rohit/mnemon
.venv/bin/python -c "from mnemon.daemon.webui import create_app; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /home/rohit/mnemon
git add src/mnemon/daemon/webui.py
git commit -m "feat(webui): add goal creation form and memory search widget"
```

---

## Task 4 — MCP Daemon Bridge Example

**Files:**
- Create: `examples/mcp_daemon_server.py`

This is a standalone MCP server that wraps the running daemon's IPC. It requires `mnemon[mcp]` and a running daemon. No changes to core code — this is an example only.

- [ ] **Step 1: Write `examples/mcp_daemon_server.py`**

```python
#!/usr/bin/env python3
"""
MCP Daemon Bridge — exposes the running Mnemon daemon as MCP tools.

Requires:
    pip install "mnemon[mcp]"
    mnemon-daemon start  # in another terminal

Tools:
    daemon_chat(message)       — send a message to Jarvis and get a reply
    daemon_goals_list()        — list active goals
    daemon_goals_add(desc)     — add a new goal
    daemon_memory_search(q)    — search episodic memory

Usage:
    python examples/mcp_daemon_server.py

Add to Claude Desktop / other MCP clients as a stdio server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mnemon.daemon.cli.client import DaemonClient
from mnemon.daemon.config import DaemonConfig

_config = DaemonConfig()
_client = DaemonClient(_config.socket_path)

app = Server("mnemon-daemon")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="daemon_chat",
            description="Send a message to the running Jarvis daemon and receive a reply. The daemon has persistent memory of past conversations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Your message to Jarvis"},
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="daemon_goals_list",
            description="List the daemon's current active goals.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="daemon_goals_add",
            description="Add a new goal to the daemon.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Goal description"},
                    "priority": {"type": "number", "description": "Priority 0.0–1.0 (default 0.5)"},
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="daemon_memory_search",
            description="Search the daemon's episodic memory for relevant past experiences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "daemon_chat":
            result = await _client.chat(arguments["message"])
            reply = result.get("reply", "(no reply)")
            return [TextContent(type="text", text=reply)]

        if name == "daemon_goals_list":
            goals = await _client.list_goals()
            if not goals:
                return [TextContent(type="text", text="No active goals.")]
            lines = [f"- [{g.get('status','?')}] {g.get('description','?')} (priority={g.get('priority',0):.1f})" for g in goals]
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "daemon_goals_add":
            result = await _client.add_goal(
                description=arguments["description"],
                priority=float(arguments.get("priority", 0.5)),
            )
            return [TextContent(type="text", text=f"Goal added: {result.get('description', arguments['description'])}")]

        if name == "daemon_memory_search":
            result = await _client.memory_search(
                query=arguments["query"],
                top_k=int(arguments.get("top_k", 10)),
            )
            items = result.get("results", [])
            if not items:
                return [TextContent(type="text", text="No memories found.")]
            lines = [f"[{item.get('score', 0):.2f}] {item.get('content', '')[:300]}" for item in items]
            return [TextContent(type="text", text="\n\n".join(lines))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/rohit/mnemon
.venv/bin/python -c "
import ast, pathlib
src = pathlib.Path('examples/mcp_daemon_server.py').read_text()
ast.parse(src)
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /home/rohit/mnemon
git add examples/mcp_daemon_server.py
git commit -m "feat(examples): add MCP daemon bridge server exposing chat + goals + memory search"
```

---

## Task 5 — Update README to Reflect True Remaining State

**Files:**
- Modify: `README.md`

The README was reverted by a linter to the old version (missing the daemon section). Restore the full version and update the Roadmap section to reflect what's now done.

- [ ] **Step 1: Replace README with the full version**

Rewrite `README.md` with the comprehensive version that covers both the core framework and the daemon layer. In the Roadmap section, mark the completed items as done:

```markdown
## Roadmap

**Bugs / correctness**
- [x] Consolidation engine: mark episodes as `FAILED` after N LLM extraction retries
- [x] Semantic store: atomic write between vector index and SQLite `_docs` table

**Self-improvement**
- [x] `daemon/improve.py` — 6-phase supervised workflow (analyze → plan → worktree → patch → verify → approve/abort)
- [x] IPC + CLI integration (`improve.analyze`, `improve.start`, `improve.status`, `improve.approve`, `improve.abort`)
- [ ] Structured planning memory: persist improvement plans across sessions in episodic store

**Interfaces**
- [x] Web UI: goal creation form
- [x] Web UI: memory search widget
- [x] MCP daemon bridge (`examples/mcp_daemon_server.py`)

**Distribution**
- [ ] PyPI release
- [ ] Docker image
- [ ] Homebrew formula (macOS)
```

- [ ] **Step 2: Verify file is valid markdown (just check it loads)**

```bash
wc -l /home/rohit/mnemon/README.md
```

Expected: 200+ lines

- [ ] **Step 3: Commit**

```bash
cd /home/rohit/mnemon
git add README.md
git commit -m "docs: update README with daemon layer and current roadmap status"
```

---

## Self-Review

**Spec coverage:**
- Self-improvement orchestration (analyze → plan → worktree → patch → verify → approve → abort) — covered by Task 1 + 2
- Web UI goal creation — covered by Task 3
- Web UI memory search — covered by Task 3
- MCP daemon bridge — covered by Task 4
- Updated README roadmap — covered by Task 5

**Placeholder scan:** All tasks have complete code. No "TBD" or "implement later."

**Type consistency:**
- `SelfImprovementOrchestrator.analyze()` returns `dict[str, Any]` — used in `_rpc_improve_analyze` which also returns `dict[str, Any]` ✓
- `ImprovementSession.phase` is `Phase` (StrEnum) — JSON-serialisable via `str(phase)` ✓
- `DaemonClient.improve_start(goal)` calls `call("improve.start", goal=goal)` — matches `_rpc_improve_start(self, goal: str)` ✓
- MCP `daemon_memory_search` calls `client.memory_search` — method added in Task 2 Step 4 ✓
