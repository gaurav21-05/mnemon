from __future__ import annotations

from uuid import UUID

import pytest

from tests.unit.test_memory_service import build_service


@pytest.mark.asyncio
async def test_scoped_recall_returns_only_matching_scope() -> None:
    service = build_service(with_consolidation=False)
    await service.write_memory(
        content="deploy the api with blue-green",
        tags=["project_context"],
        scope_type="workspace",
        scope_id="repo-a",
        repo_name="repo-a",
    )
    await service.write_memory(
        content="deploy the api with canary",
        tags=["project_context"],
        scope_type="workspace",
        scope_id="repo-b",
        repo_name="repo-b",
    )

    result = await service.profile_recall(
        query="how do I deploy the api?",
        scope_type="workspace",
        scope_id="repo-a",
    )

    assert result["scope_type"] == "workspace"
    assert result["scope_id"] == "repo-a"
    assert any("blue-green" in item["content"] for item in result["results"])
    assert not any("canary" in item["content"] for item in result["results"])


@pytest.mark.asyncio
async def test_write_memory_records_scope_metadata() -> None:
    service = build_service(with_consolidation=False)
    write_result = await service.write_memory(
        content="remember this repo preference",
        tags=["profile_dynamic"],
        scope_type="workspace",
        scope_id="mnemon",
        workspace_path="/tmp/mnemon",
        repo_name="mnemon",
    )

    episode = await service.episodic.get(UUID(write_result["episode_id"]))

    assert episode is not None
    assert episode.scope_type == "workspace"
    assert episode.scope_id == "mnemon"
    assert episode.workspace_path == "/tmp/mnemon"
    assert episode.repo_name == "mnemon"
