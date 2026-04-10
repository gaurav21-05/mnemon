"""Bounded scenario sandbox for grounded operator what-if analysis."""

from __future__ import annotations

from typing import Any

_SCENARIO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "string"},
    },
    "required": ["summary", "assumptions", "risks", "recommendations", "uncertainty"],
}


class ScenarioEngine:
    """Generate bounded scenario reports from grounded daemon context."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(
        self,
        *,
        scenario: str,
        profile: dict[str, Any],
        goals: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        workspace_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a grounded scenario report."""
        prompt = self._build_prompt(
            scenario=scenario,
            profile=profile,
            goals=goals,
            memories=memories,
            workspace_items=workspace_items,
        )

        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                response_schema=_SCENARIO_SCHEMA,
            )
        except Exception:
            result = {
                "summary": "",
                "assumptions": [],
                "risks": [],
                "recommendations": [],
                "uncertainty": "",
            }

        if not result.get("summary"):
            result["summary"] = f"Grounded scenario analysis for: {scenario}"
        if not result.get("uncertainty"):
            result["uncertainty"] = (
                "Medium — based on recent memory and current goals, "
                "not execution guarantees."
            )

        citations = []
        for memory in memories[:5]:
            memory_id = str(memory.get("id", "")).strip()
            if memory_id:
                citations.append(f"[memory:{memory_id}]")

        return {
            "scenario": scenario,
            "summary": result["summary"],
            "assumptions": list(result.get("assumptions", [])),
            "risks": list(result.get("risks", [])),
            "recommendations": list(result.get("recommendations", [])),
            "uncertainty": result["uncertainty"],
            "citations": citations,
        }

    @staticmethod
    def _build_prompt(
        *,
        scenario: str,
        profile: dict[str, Any],
        goals: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        workspace_items: list[dict[str, Any]],
    ) -> str:
        """Build a bounded grounded prompt for scenario generation."""
        static = ", ".join(profile.get("static", [])[:4]) or "(none)"
        dynamic = ", ".join(profile.get("dynamic", [])[:5]) or "(none)"
        goal_lines = (
            "\n".join(f"- {goal.get('description', '')}" for goal in goals[:5])
            or "- (none)"
        )
        memory_lines = "\n".join(
            f"- {memory.get('preview', memory.get('content', ''))}"
            for memory in memories[:6]
        ) or "- (none)"
        workspace_lines = "\n".join(
            f"- {item.get('path') or item.get('name') or ''}"
            for item in workspace_items[:6]
        ) or "- (none)"

        return (
            "You are generating a bounded what-if analysis for an operator.\n"
            "Stay grounded in the provided facts. Do not invent external events.\n"
            "Return concise scenario guidance with assumptions, risks, "
            "recommendations, and uncertainty.\n\n"
            f"Scenario:\n{scenario}\n\n"
            f"Static profile:\n{static}\n\n"
            f"Dynamic profile:\n{dynamic}\n\n"
            f"Active goals:\n{goal_lines}\n\n"
            f"Recent memories:\n{memory_lines}\n\n"
            f"Workspace context:\n{workspace_lines}\n"
        )
