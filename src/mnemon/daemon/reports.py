"""Bounded report generation for weekly and project briefs."""

from __future__ import annotations

from typing import Any

_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "highlights": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "highlights", "risks", "next_steps"],
}


class ReportEngine:
    """Generate grounded operator-facing reports."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(
        self,
        *,
        report_type: str,
        focus: str,
        profile: dict[str, Any],
        goals: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        workspace_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a bounded report grounded in daemon context."""
        prompt = self._build_prompt(
            report_type=report_type,
            focus=focus,
            profile=profile,
            goals=goals,
            memories=memories,
            workspace_items=workspace_items,
        )

        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                response_schema=_REPORT_SCHEMA,
            )
        except Exception:
            result = {
                "title": "",
                "summary": "",
                "highlights": [],
                "risks": [],
                "next_steps": [],
            }

        if not result.get("title"):
            result["title"] = f"{report_type.title()} report"
        if not result.get("summary"):
            result["summary"] = f"Grounded {report_type} report for {focus or 'current context'}."

        citations = []
        for memory in memories[:5]:
            memory_id = str(memory.get("id", "")).strip()
            if memory_id:
                citations.append(f"[memory:{memory_id}]")

        return {
            "type": report_type,
            "focus": focus,
            "title": result["title"],
            "summary": result["summary"],
            "highlights": list(result.get("highlights", [])),
            "risks": list(result.get("risks", [])),
            "next_steps": list(result.get("next_steps", [])),
            "citations": citations,
        }

    @staticmethod
    def _build_prompt(
        *,
        report_type: str,
        focus: str,
        profile: dict[str, Any],
        goals: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        workspace_items: list[dict[str, Any]],
    ) -> str:
        static = ", ".join(profile.get("static", [])[:4]) or "(none)"
        dynamic = ", ".join(profile.get("dynamic", [])[:5]) or "(none)"
        goal_lines = (
            "\n".join(f"- {goal.get('description', '')}" for goal in goals[:6])
            or "- (none)"
        )
        memory_lines = (
            "\n".join(
                f"- {memory.get('preview', memory.get('content', ''))}"
                for memory in memories[:8]
            )
            or "- (none)"
        )
        workspace_lines = (
            "\n".join(
                f"- {item.get('path') or item.get('name') or ''}"
                for item in workspace_items[:8]
            )
            or "- (none)"
        )

        return (
            "Generate a concise, grounded operator report.\n"
            "Do not invent work that is not present in the inputs.\n"
            f"Report type: {report_type}\n"
            f"Focus: {focus or '(general)'}\n\n"
            f"Static profile:\n{static}\n\n"
            f"Dynamic profile:\n{dynamic}\n\n"
            f"Goals:\n{goal_lines}\n\n"
            f"Recent memories:\n{memory_lines}\n\n"
            f"Workspace context:\n{workspace_lines}\n"
        )
