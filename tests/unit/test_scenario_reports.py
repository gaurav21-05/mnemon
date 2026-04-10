from __future__ import annotations

from typing import Any

import pytest

from mnemon.daemon.scenario import ScenarioEngine


class _FakeLLM:
    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: object,
    ) -> dict[str, Any]:
        del prompt, response_schema, kwargs
        return {
            "summary": "If you prioritize deployment work now, delivery confidence improves.",
            "assumptions": ["Current goal priority remains stable"],
            "risks": ["Other goals may slip"],
            "recommendations": ["Focus on the deployment workflow first"],
            "uncertainty": "Medium",
        }


@pytest.mark.asyncio
async def test_scenario_engine_returns_grounded_report() -> None:
    engine = ScenarioEngine(_FakeLLM())

    result = await engine.run(
        scenario="What happens if I prioritize deployment work this week?",
        profile={"static": ["Prefers direct answers"], "dynamic": ["Working on mnemon"]},
        goals=[{"description": "Ship deployment workflow"}],
        memories=[{"id": "ep-1", "preview": "deploy with blue-green"}],
        workspace_items=[{"path": "docs/deploy.md"}],
    )

    assert "deployment work" in result["scenario"]
    assert result["summary"]
    assert result["assumptions"] == ["Current goal priority remains stable"]
    assert result["risks"] == ["Other goals may slip"]
    assert result["recommendations"] == ["Focus on the deployment workflow first"]
    assert result["citations"] == ["[memory:ep-1]"]
