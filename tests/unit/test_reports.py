from __future__ import annotations

from typing import Any

import pytest

from mnemon.daemon.reports import ReportEngine


class _FakeLLM:
    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: object,
    ) -> dict[str, Any]:
        del prompt, response_schema, kwargs
        return {
            "title": "Weekly report",
            "summary": "Deployment work and memory UX were the main themes.",
            "highlights": ["Improved deployment flow", "Expanded memory UX"],
            "risks": ["Too many parallel priorities"],
            "next_steps": ["Finish the deployment workflow"],
        }


@pytest.mark.asyncio
async def test_report_engine_returns_grounded_weekly_report() -> None:
    engine = ReportEngine(_FakeLLM())

    result = await engine.run(
        report_type="weekly",
        focus="mnemon",
        profile={"static": ["Prefers direct answers"], "dynamic": ["Working on mnemon"]},
        goals=[{"description": "Ship deployment workflow"}],
        memories=[{"id": "ep-1", "preview": "deploy with blue-green"}],
        workspace_items=[{"path": "docs/deploy.md"}],
    )

    assert result["type"] == "weekly"
    assert result["title"] == "Weekly report"
    assert result["summary"]
    assert result["highlights"] == ["Improved deployment flow", "Expanded memory UX"]
    assert result["risks"] == ["Too many parallel priorities"]
    assert result["next_steps"] == ["Finish the deployment workflow"]
    assert result["citations"] == ["[memory:ep-1]"]
