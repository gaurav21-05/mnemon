from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from aiohttp.test_utils import make_mocked_request

from mnemon.daemon import webui

if TYPE_CHECKING:
    from pathlib import Path as PathType


def test_resolve_managed_file_initializes_identity_docs(
    tmp_path: PathType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(webui, '_state_dir', lambda: tmp_path)

    path, meta = webui._resolve_managed_file('master.md')

    assert path == tmp_path / 'master.md'
    assert path.exists() is True
    assert meta['format'] == 'markdown'


def test_validate_managed_file_content_checks_json() -> None:
    webui._validate_managed_file_content('goals.json', '{"ok": true}')
    with pytest.raises(json.JSONDecodeError):
        webui._validate_managed_file_content('goals.json', '{bad json}')


def test_managed_file_payload_includes_content(tmp_path: PathType) -> None:
    path = tmp_path / 'learnings.md'
    path.write_text('hello', encoding='utf-8')

    payload = webui._managed_file_payload(
        'learnings.md',
        path,
        webui._MANAGED_FILES['learnings.md'],
    )

    assert payload['name'] == 'learnings.md'
    assert payload['content'] == 'hello'
    assert payload['exists'] is True
    assert payload['bytes'] == 5


def test_resolve_managed_file_initializes_privacy_rules_json(
    tmp_path: PathType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(webui, "_state_dir", lambda: tmp_path)

    path, meta = webui._resolve_managed_file("privacy_rules.json")

    assert path == tmp_path / "privacy_rules.json"
    assert path.exists() is True
    assert meta["format"] == "json"


def test_unified_diff_contains_expected_headers_and_change() -> None:
    diff = webui._unified_diff("old\n", "new\n", "master.md")

    assert "master.md (saved)" in diff
    assert "master.md (draft)" in diff
    assert "-old" in diff
    assert "+new" in diff


def test_html_contains_memory_search_keyboard_and_quick_actions() -> None:
    assert "search-open-memory-tab" in webui.HTML
    assert "search-copy-id" in webui.HTML
    assert "moveSearchSelection" in webui.HTML
    assert "openSearchResult" in webui.HTML
    assert "j/k when results focused" in webui.HTML
    assert "showToast(successMessage)" in webui.HTML
    assert 'id="toast"' in webui.HTML


def test_html_contains_scrollable_sidebar_tab_list() -> None:
    assert ".sidebar-shell {" in webui.HTML
    assert "overflow: hidden;" in webui.HTML
    assert ".tab-list {" in webui.HTML
    assert "overflow-y: auto;" in webui.HTML
    assert "scrollbar-width: thin;" in webui.HTML
    assert ".tab-list::-webkit-scrollbar" in webui.HTML


def test_html_contains_extra_bottom_breathing_room_for_chat() -> None:
    assert "padding: 14px 0 20px;" in webui.HTML
    assert "height: calc(100vh - 82px);" in webui.HTML
    assert "padding: 0 14px 16px;" in webui.HTML


def test_html_contains_compact_sidebar_density_tweaks() -> None:
    assert "grid-template-columns: 248px minmax(0, 1fr);" in webui.HTML
    assert "padding: 14px;" in webui.HTML
    assert "font-size: 11px;" in webui.HTML


def test_html_uses_flattened_spacing_and_graph_canvas() -> None:
    assert "rgba(0,0,0,0.045) 0 8px 22px" in webui.HTML
    assert "rgba(0,0,0,0.14) 0 28px 70px" not in webui.HTML
    assert "height: min(54vh, 430px);" in webui.HTML


def test_html_contains_shortened_sidebar_copy() -> None:
    assert '<span class="tab-btn-title">Overview</span>' in webui.HTML
    assert "Talk to Jarvis" in webui.HTML
    assert "Where memories live, connect, and compress" in webui.HTML


def test_html_contains_profile_sources_and_recent_changes_ui() -> None:
    assert "fact-sources" in webui.HTML
    assert "Recent changes" in webui.HTML
    assert "profile.static_facts" in webui.HTML
    assert "Copy cite" in webui.HTML
    assert "search-copy-citation" in webui.HTML


def test_html_contains_richer_hover_and_active_visual_states() -> None:
    assert "--accent-strong" in webui.HTML
    assert "--accent-soft" in webui.HTML
    assert ".tab-btn.active {" in webui.HTML
    assert "border-color: var(--border-strong);" in webui.HTML
    assert ".file-btn:hover {" in webui.HTML
    assert ".mini-btn:hover {" in webui.HTML


def test_html_contains_second_pass_chat_toolbar_status_polish() -> None:
    assert ".status-pill.online {" in webui.HTML
    assert ".msg.user .msg-text {" in webui.HTML
    assert "background: var(--accent);" in webui.HTML
    assert ".file-toolbar {" in webui.HTML


def test_html_contains_third_pass_drawer_dropdown_empty_state_polish() -> None:
    assert ".memory-search-results {" in webui.HTML
    assert "background: var(--surface-soft);" in webui.HTML
    assert ".drawer-memory-link:hover {" in webui.HTML
    assert ".detail-drawer {" in webui.HTML
    assert "border: 1px dashed var(--border-strong);" in webui.HTML


def test_html_contains_memory_atlas_and_relationship_language() -> None:
    assert "Memory atlas" in webui.HTML
    assert "Storage model" in webui.HTML
    assert "Linked sources" in webui.HTML
    assert "workspace ·" in webui.HTML


def test_html_contains_cursor_inspired_theme_tokens() -> None:
    assert "--paper: #f2f1ed;" in webui.HTML
    assert "--surface: #f7f7f4;" in webui.HTML
    assert "--accent: #f54e00;" in webui.HTML
    assert '--font-sans: Georgia, \'Iowan Old Style\'' in webui.HTML
    assert ':root[data-theme="dark"]' in webui.HTML


def test_html_contains_theme_toggle() -> None:
    assert 'id="themeToggle"' in webui.HTML
    assert "localStorage.setItem('mnemon-theme'" in webui.HTML
    assert "function toggleTheme()" in webui.HTML


def test_html_contains_voice_toggle_for_new_thoughts() -> None:
    assert 'id="voiceToggle"' in webui.HTML
    assert "localStorage.setItem('mnemon-voice'" in webui.HTML
    assert "function speakThought" in webui.HTML
    assert "maybeSpeakNewestThought(thoughts)" in webui.HTML
    assert "voiceHydrated" in webui.HTML


def test_create_app_serves_built_vite_assets_when_available(
    tmp_path: PathType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=\"root\"></div>", encoding="utf-8")
    monkeypatch.setattr(webui, "FRONTEND_DIST", dist)

    app = webui.create_app()

    route_names = {resource.name for resource in app.router.resources()}
    assert "frontend_assets" in route_names


def test_html_contains_offline_recovery_state() -> None:
    assert "renderOfflinePanels" in webui.HTML
    assert "Expected socket: ~/.mnemon/daemon.sock" in webui.HTML
    assert "Waiting for daemon status" in webui.HTML


def test_html_defaults_to_memory_tab_and_has_drawer_minimap() -> None:
    assert "let activeTab = 'memory';" in webui.HTML
    assert 'data-tab="memory"' in webui.HTML
    assert "drawer-minimap" in webui.HTML
    assert "relationships</span>" in webui.HTML


def test_html_contains_memory_first_sidebar_information_architecture() -> None:
    assert "Primary" in webui.HTML
    assert "Workspace" in webui.HTML
    assert "Utilities" in webui.HTML
    assert "Where memories live, connect, and compress" in webui.HTML
    assert 'class="tab-btn compact"' in webui.HTML
    assert 'data-tab="graph"' in webui.HTML
    assert "Relationships between summaries, sources, and storage" in webui.HTML
    assert "sidebar-footer" not in webui.HTML


def test_html_contains_graph_controls_and_interactions() -> None:
    assert "graphResetBtn" in webui.HTML
    assert "graphRefreshBtn" in webui.HTML
    assert "graphZoomInBtn" in webui.HTML
    assert "graphZoomOutBtn" in webui.HTML
    assert "graphFullscreenBtn" in webui.HTML
    assert "memory-graph-shell.fullscreen" in webui.HTML
    assert "refreshMemoryGraph()" in webui.HTML
    assert "graph-filter-chip" in webui.HTML
    assert "d3GraphSvg" in webui.HTML


def test_html_contains_zoom_cluster_behavior() -> None:
    assert "graphViewData" in webui.HTML
    assert "Researched ${children.length}" in webui.HTML
    assert "graphCurrentZoom < 1.15" in webui.HTML
    assert "graphForceExpanded" in webui.HTML


@pytest.mark.asyncio
async def test_memory_profile_handler_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def memory_profile(self) -> dict[str, object]:
            return {"static": ["prefers direct answers"], "dynamic": ["working on mnemon"]}

    monkeypatch.setattr(webui, "_client_for", lambda _request: _FakeClient())
    request = make_mocked_request("GET", "/api/memory/profile")

    response = await webui.memory_profile_handler(request)
    payload = json.loads(response.text)

    assert payload["static"] == ["prefers direct answers"]
    assert payload["dynamic"] == ["working on mnemon"]


@pytest.mark.asyncio
async def test_memory_recall_handler_returns_profile_and_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def memory_recall(
            self,
            query: str,
            top_k: int,
            scope: str = "all",
            scope_id: str | None = None,
        ) -> dict[str, object]:
            return {
                "query": query,
                "profile": {"static": ["prefers direct answers"]},
                "results": [{"id": "ep-1"}],
                "top_k": top_k,
                "scope": scope,
                "scope_id": scope_id,
            }

    monkeypatch.setattr(webui, "_client_for", lambda _request: _FakeClient())
    request = make_mocked_request("GET", "/api/memory/recall?q=mnemon&top_k=7&scope=workspace")

    response = await webui.memory_recall_handler(request)
    payload = json.loads(response.text)

    assert payload["query"] == "mnemon"
    assert payload["profile"]["static"] == ["prefers direct answers"]
    assert payload["results"][0]["id"] == "ep-1"
    assert payload["scope"] == "workspace"


@pytest.mark.asyncio
async def test_memory_hybrid_handler_returns_combined_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def memory_hybrid(
            self,
            query: str,
            top_k: int,
            scope: str = "all",
            scope_id: str | None = None,
        ) -> dict[str, object]:
            return {
                "query": query,
                "profile": {"static": ["prefers direct answers"]},
                "hybrid_results": [{"kind": "goal", "title": "Ship deployment workflow"}],
                "scope": scope,
                "scope_id": scope_id,
                "top_k": top_k,
            }

    monkeypatch.setattr(webui, "_client_for", lambda _request: _FakeClient())
    request = make_mocked_request("GET", "/api/memory/hybrid?q=deploy&top_k=5&scope=workspace")

    response = await webui.memory_hybrid_handler(request)
    payload = json.loads(response.text)

    assert payload["query"] == "deploy"
    assert payload["profile"]["static"] == ["prefers direct answers"]
    assert payload["hybrid_results"][0]["kind"] == "goal"


@pytest.mark.asyncio
async def test_memory_graph_handler_returns_graph_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def memory_graph(
            self,
            limit: int = 40,
            scope: str = "all",
            scope_id: str | None = None,
        ) -> dict[str, object]:
            return {
                "scope": scope,
                "scope_id": scope_id,
                "nodes": [{"id": "scope:workspace:mnemon", "kind": "scope"}],
                "edges": [],
                "limit": limit,
            }

    monkeypatch.setattr(webui, "_client_for", lambda _request: _FakeClient())
    request = make_mocked_request("GET", "/api/memory/graph?limit=12&scope=workspace")

    response = await webui.memory_graph_handler(request)
    payload = json.loads(response.text)

    assert payload["scope"] == "workspace"
    assert payload["nodes"][0]["kind"] == "scope"


@pytest.mark.asyncio
async def test_scenario_handler_returns_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def run_scenario(
            self,
            scenario: str,
            scope: str = "all",
            scope_id: str | None = None,
        ) -> dict[str, object]:
            return {
                "scenario": scenario,
                "summary": "Focus on deployment first.",
                "assumptions": ["Current priorities remain stable"],
                "risks": ["Other work may slip"],
                "recommendations": ["Prioritize deployment"],
                "uncertainty": "Medium",
                "citations": ["[memory:ep-1]"],
                "scope": scope,
                "scope_id": scope_id,
            }

    monkeypatch.setattr(webui, "_client_for", lambda _request: _FakeClient())
    request = make_mocked_request("GET", "/api/scenario?q=deploy&scope=workspace")

    response = await webui.scenario_handler(request)
    payload = json.loads(response.text)

    assert payload["summary"] == "Focus on deployment first."
    assert payload["citations"] == ["[memory:ep-1]"]


@pytest.mark.asyncio
async def test_report_handler_returns_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def run_report(
            self,
            report_type: str = "weekly",
            focus: str = "",
            scope: str = "all",
            scope_id: str | None = None,
        ) -> dict[str, object]:
            return {
                "type": report_type,
                "focus": focus,
                "title": "Weekly report",
                "summary": "Deployment work dominated the week.",
                "highlights": ["Validated deployment approach"],
                "risks": ["Other work may slip"],
                "next_steps": ["Finish deployment workflow"],
                "citations": ["[memory:ep-1]"],
                "scope": scope,
                "scope_id": scope_id,
            }

    monkeypatch.setattr(webui, "_client_for", lambda _request: _FakeClient())
    request = make_mocked_request("GET", "/api/report?type=weekly&focus=deployment&scope=workspace")

    response = await webui.report_handler(request)
    payload = json.loads(response.text)

    assert payload["title"] == "Weekly report"
    assert payload["summary"] == "Deployment work dominated the week."
    assert payload["citations"] == ["[memory:ep-1]"]


@pytest.mark.asyncio
async def test_memory_get_and_timeline_handlers_forward_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def memory_get(self, ids: list[str]) -> dict[str, object]:
            return {"items": [{"id": ids[0], "context": "hello"}], "missing": []}

        async def memory_timeline(self, anchor_id: str, limit: int) -> dict[str, object]:
            return {
                "anchor_id": anchor_id,
                "items": [{"id": anchor_id, "anchor": True}],
                "limit": limit,
            }

    monkeypatch.setattr(webui, "_client_for", lambda _request: _FakeClient())

    get_request = make_mocked_request("GET", "/api/memory/item?id=abc")
    get_response = await webui.memory_get_handler(get_request)
    get_payload = json.loads(get_response.text)
    assert get_payload["items"][0]["id"] == "abc"

    timeline_request = make_mocked_request("GET", "/api/memory/timeline?id=abc&limit=7")
    timeline_response = await webui.memory_timeline_handler(timeline_request)
    timeline_payload = json.loads(timeline_response.text)
    assert timeline_payload["anchor_id"] == "abc"
    assert timeline_payload["items"][0]["anchor"] is True
