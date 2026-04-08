"""
SelfImprovementOrchestrator — supervised 6-phase self-improvement workflow.

The orchestrator lets Jarvis analyze its own codebase, plan targeted patches,
apply them in an isolated git worktree, verify with tests, and wait for human
approval before merging.

Phases:
  1. ANALYZING   — git status + pytest to assess current state; LLM summary
  2. PLANNING    — LLM produces ordered patch steps as structured JSON
  3. WORKTREE    — create isolated git worktree on a new branch
  4. PATCHING    — apply each patch step sequentially
  5. VERIFYING   — run pytest in worktree; record pass/fail
  6. AWAITING_APPROVAL — human reviews; approve → merge, abort → discard
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
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
    phase: Phase = field(default=Phase.IDLE)
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

    Analyzes the current repo, plans targeted patches, applies them in an
    isolated worktree, verifies with tests, and gates on human approval
    before merging.
    """

    def __init__(self, workspace: JarvisWorkspace, llm: Any) -> None:
        self._workspace = workspace
        self._llm = llm
        self._session: ImprovementSession | None = None

    @property
    def session(self) -> ImprovementSession | None:
        return self._session

    def status(self) -> dict[str, Any]:
        """Return a JSON-serialisable status dict."""
        if self._session is None:
            return {"phase": "idle", "session_id": None}
        s = self._session
        return {
            "session_id": str(s.id),
            "phase": str(s.phase),
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
        """Run analysis phase only — returns summary without starting a full session."""
        ws = self._workspace
        git_result = await ws.git_status()
        git_out = git_result.get("stdout", "")

        verify_result = await ws.verify(
            commands=[f"{sys.executable} -m pytest tests/unit -q --tb=no --no-header"],
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
        Raises RuntimeError if a session is already active.
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
                "Each step: {description, file (relative path from repo root), search (exact text to find), replace (replacement text)}.\n"
                "Only include steps where the search text will be found exactly as-is in the file.\n"
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
                session.error = wt_result.get("stderr", "worktree creation failed")[:500]
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
                commands=[f"{sys.executable} -m pytest tests/unit -q --tb=short --no-header"],
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
                "Improvement session awaiting approval (approval_id=%s, verify_passed=%s)",
                session.approval_id,
                session.verify_passed,
            )

        except Exception as exc:
            session.error = str(exc)
            session.phase = Phase.FAILED
            logger.exception("Self-improvement session failed unexpectedly")

    async def approve(self) -> dict[str, Any]:
        """Merge the worktree branch into HEAD and remove the worktree."""
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
            s.error = merge_result.get("stderr", "merge failed")[:500]
            s.phase = Phase.FAILED
            return {"ok": False, "error": s.error}

        try:
            await ws.remove_worktree(s.worktree_path, force=True)
        except Exception as exc:
            logger.warning("remove_worktree failed after merge: %s", exc)

        s.phase = Phase.DONE
        return {"ok": True, "branch": s.branch, "summary": s.plan_summary}

    async def abort(self) -> dict[str, Any]:
        """Remove worktree without merging and mark session aborted."""
        s = self._session
        if s is None or s.phase not in (
            Phase.AWAITING_APPROVAL, Phase.VERIFYING, Phase.PATCHING
        ):
            return {"ok": False, "error": "no active session to abort"}

        if s.worktree_path:
            try:
                await self._workspace.remove_worktree(s.worktree_path, force=True)
            except Exception as exc:
                logger.warning("abort: remove_worktree failed: %s", exc)

        s.phase = Phase.ABORTED
        return {"ok": True}
