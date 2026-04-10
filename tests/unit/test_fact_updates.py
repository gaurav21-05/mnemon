from __future__ import annotations

import pytest

from tests.unit.test_memory_service import build_service


@pytest.mark.asyncio
async def test_profile_recall_prefers_current_fact_and_preserves_history() -> None:
    service = build_service(with_consolidation=False)
    first = await service.write_memory(
        content="I use OpenAI",
        tags=["profile_static"],
    )
    second = await service.write_memory(
        content="I now use Anthropic",
        tags=["profile_static"],
    )

    result = await service.profile_recall(query="what model provider do I use now?")

    assert result["profile"]["static"][0]["text"] == "I now use Anthropic"
    assert result["profile"]["static"][0]["current"] is True
    assert result["history"]["static"][0]["text"] == "I use OpenAI"
    assert result["history"]["static"][0]["superseded_by"] == second["episode_id"]
    assert result["history"]["static"][0]["current"] is False
    assert result["profile"]["static"][0]["supersedes"] == [first["episode_id"]]


@pytest.mark.asyncio
async def test_profile_recall_keeps_unrelated_static_facts() -> None:
    service = build_service(with_consolidation=False)
    await service.write_memory(content="I prefer dark mode", tags=["profile_static"])
    await service.write_memory(content="I now use Anthropic", tags=["profile_static"])

    result = await service.profile_recall(query="summarize my preferences")

    texts = [item["text"] for item in result["profile"]["static"]]
    assert "I prefer dark mode" in texts
    assert "I now use Anthropic" in texts
