"""Unit tests for improvement- and memory-search IPC helpers."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import anyio
import pytest

from mnemon.core.models import GoalStatus
from mnemon.daemon.autonomy import AutonomyController
from mnemon.daemon.config import DaemonConfig
from mnemon.daemon.ipc import DaemonIPCServer
from mnemon.daemon.state import DaemonState

pytestmark = pytest.mark.asyncio


class _DummyIdleLoop:
    is_busy = False

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


class _FakeEpisodicMemory:
    def __init__(self, docs: dict[UUID, dict[str, object]] | None = None) -> None:
        self._docs = docs or {}
        self._document_store = SimpleNamespace(
            get=self._get_doc,
            query=self._query_docs,
        )

    async def _get_doc(self, episode_id: UUID) -> dict[str, object] | None:
        return self._docs.get(episode_id)

    async def _query_docs(
        self,
        filters: dict[str, object],
        limit: int,
        sort_by: str | None = None,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        del filters, sort_by
        values = list(self._docs.values())
        return values[offset : offset + limit]

    async def retrieve(self, query: object) -> object:
        top_k = getattr(query, "top_k", None)
        filters = getattr(query, "filters", None) or {}
        matching_pairs = [
            (doc_id, doc)
            for doc_id, doc in self._docs.items()
            if all(doc.get(key) == value for key, value in filters.items())
        ]
        episode_id, source_doc = (
            matching_pairs[0] if matching_pairs else next(iter(self._docs.items()), (uuid4(), {}))
        )
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    content=f"match for top_k={top_k}",
                    score=0.8,
                    source_store="episodic",
                    metadata={
                        "episode_id": str(episode_id),
                        "session_id": str(source_doc.get("session_id", "session-1")),
                        "timestamp": str(source_doc.get("timestamp", "2026-04-09T08:00:00+00:00")),
                        "importance": float(source_doc.get("importance", 0.6) or 0.6),
                        "tags": list(source_doc.get("tags", ["demo"])),
                    },
                )
            ]
        )


class _FakeSemanticMemory:
    def __init__(self, docs: dict[UUID, dict[str, object]] | None = None) -> None:
        self._docs_map = docs or {}
        self._docs = SimpleNamespace(
            get=self._get_doc,
            query=self._query_docs,
        )

    async def _get_doc(self, triple_id: UUID) -> dict[str, object] | None:
        return self._docs_map.get(triple_id)

    async def _query_docs(self, filters: dict[str, object], limit: int) -> list[dict[str, object]]:
        matched = [
            doc
            for doc in self._docs_map.values()
            if all(doc.get(key) == value for key, value in filters.items())
        ]
        return matched[:limit]


class _ClearableDocumentStore:
    def __init__(self, docs: dict[UUID, dict[str, object]] | None = None) -> None:
        self.docs = docs or {}

    async def count(self, filters: dict[str, object] | None = None) -> int:
        del filters
        return len(self.docs)

    async def query(self, filters: dict[str, object], limit: int) -> list[dict[str, object]]:
        del filters
        return list(self.docs.values())[:limit]

    async def delete(self, document_id: UUID) -> None:
        self.docs.pop(document_id, None)


class _ClearableVectorStore:
    def __init__(self, count: int = 0) -> None:
        self.count_value = count
        self.cleared = False

    async def count(self) -> int:
        return self.count_value

    async def clear(self) -> None:
        self.cleared = True
        self.count_value = 0


def _make_server(brain: object) -> DaemonIPCServer:
    return DaemonIPCServer(
        socket_path=Path("/tmp/mnemon-test.sock"),
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(DaemonConfig()),
        idle_loop=_DummyIdleLoop(),
    )


async def test_ipc_server_serves_fast_request_while_slow_request_is_running(
    tmp_path: Path,
) -> None:
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
    )
    server = DaemonIPCServer(
        socket_path=tmp_path / "daemon.sock",
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(DaemonConfig()),
        idle_loop=_DummyIdleLoop(),
    )
    slow_started = anyio.Event()
    slow_can_finish = anyio.Event()

    async def slow() -> dict[str, bool]:
        slow_started.set()
        await slow_can_finish.wait()
        return {"slow": True}

    async def fast() -> dict[str, bool]:
        return {"fast": True}

    server._handlers["slow"] = slow
    server._handlers["fast"] = fast

    async def call(method: str) -> dict[str, object]:
        stream = await anyio.connect_unix(str(server._socket_path))
        try:
            await stream.send(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": {},
                }).encode("utf-8")
            )
            response = json.loads((await stream.receive(65_536)).decode("utf-8"))
            return response["result"]
        finally:
            await stream.aclose()

    async with anyio.create_task_group() as task_group:
        await server.start(task_group)
        for _ in range(20):
            if server._socket_path.exists():
                break
            await anyio.sleep(0.01)

        task_group.start_soon(call, "slow")
        await slow_started.wait()

        started = time.monotonic()
        fast_result = await call("fast")
        elapsed = time.monotonic() - started

        slow_can_finish.set()
        await server.stop()
        task_group.cancel_scope.cancel()

    assert fast_result == {"fast": True}
    assert elapsed < 0.25


async def test_improve_status_is_idle_before_session_exists() -> None:
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
    )
    server = _make_server(brain)

    assert await server._rpc_improve_status() == {"phase": "idle", "session_id": None}


async def test_improve_approve_and_abort_fail_without_session() -> None:
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
    )
    server = _make_server(brain)

    assert await server._rpc_improve_approve() == {
        "ok": False,
        "error": "no session awaiting approval",
    }
    assert await server._rpc_improve_abort() == {
        "ok": False,
        "error": "no active session to abort",
    }


async def test_memory_search_returns_serialized_retrieval_results() -> None:
    episode_id = uuid4()
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(
            episodic=_FakeEpisodicMemory(
                {
                    episode_id: {
                        "id": str(episode_id),
                        "timestamp": "2026-04-09T08:00:00+00:00",
                        "context": "match for top_k=1",
                        "action": "Stored via memory service",
                        "outcome": "Available for future recall",
                        "importance": 0.6,
                        "tags": ["demo"],
                        "session_id": "session-1",
                    }
                }
            )
        ),
    )
    server = _make_server(brain)

    result = await server._rpc_memory_search(query="search term", top_k=0)

    assert result["query"] == "search term"
    assert result["results"] == [
        {
            "id": str(episode_id),
            "preview": (
                "match for top_k=1 — Stored via memory service — Available for future recall"
            ),
            "content": (
                "match for top_k=1 — Stored via memory service — Available for future recall"
            ),
            "score": 0.8,
            "timestamp": "2026-04-09T08:00:00+00:00",
            "importance": 0.6,
            "tags": ["demo"],
            "session_id": "session-1",
            "scope_type": "personal",
            "scope_id": "",
            "workspace_path": "",
            "repo_name": "",
            "citation": f"[memory:{episode_id}]",
            "caused_by": None,
            "led_to": [],
            "source_episode_ids": [],
            "summary_kind": "",
            "summary_of_count": 0,
            "source": "episodic",
        }
    ]


async def test_memory_get_and_timeline_return_full_records() -> None:
    session_id = "session-42"
    ids = [uuid4(), uuid4(), uuid4()]
    docs = {
        ids[0]: {
            "id": str(ids[0]),
            "timestamp": "2026-04-09T08:00:00+00:00",
            "context": "first",
            "action": "remembered",
            "outcome": "stored",
            "importance": 0.3,
            "tags": ["alpha"],
            "session_id": session_id,
        },
        ids[1]: {
            "id": str(ids[1]),
            "timestamp": "2026-04-09T08:05:00+00:00",
            "context": "second",
            "action": "updated",
            "outcome": "stored",
            "importance": 0.6,
            "tags": ["beta"],
            "session_id": session_id,
        },
        ids[2]: {
            "id": str(ids[2]),
            "timestamp": "2026-04-09T08:10:00+00:00",
            "context": "third",
            "action": "finished",
            "outcome": "stored",
            "importance": 0.9,
            "tags": ["beta"],
            "session_id": session_id,
        },
    }
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)

    fetched = await server._rpc_memory_get([str(ids[1])])
    assert fetched["missing"] == []
    assert fetched["items"][0]["context"] == "second"
    assert fetched["items"][0]["action"] == "updated"

    timeline = await server._rpc_memory_timeline(anchor_id=str(ids[1]), limit=3)
    assert [item["id"] for item in timeline["items"]] == [str(ids[0]), str(ids[1]), str(ids[2])]
    assert timeline["items"][1]["anchor"] is True


async def test_memory_explain_fact_returns_evidence_chain() -> None:
    episode_id = uuid4()
    triple_id = uuid4()
    subject_id = uuid4()
    episodic = _FakeEpisodicMemory(
        {
            episode_id: {
                "id": str(episode_id),
                "timestamp": "2026-04-09T08:00:00+00:00",
                "context": "Rohit works on mnemon",
                "action": "said",
                "outcome": "stored",
                "importance": 0.8,
                "tags": ["profile_dynamic"],
                "session_id": "session-1",
            }
        }
    )
    semantic = _FakeSemanticMemory(
        {
            triple_id: {
                "id": str(triple_id),
                "_type": "triple",
                "subject": {"entity_id": str(subject_id), "name": "Rohit"},
                "predicate": "works_on",
                "object": "mnemon",
                "confidence": 0.9,
                "source_episodes": [str(episode_id)],
                "last_confirmed": "2026-04-09T08:00:00+00:00",
                "current": True,
                "contradiction_group": None,
            }
        }
    )
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(episodic=episodic, semantic=semantic),
    )
    server = _make_server(brain)

    result = await server._rpc_memory_explain_fact(str(triple_id))

    assert result["triple_id"] == str(triple_id)
    assert result["fact"] == "Rohit works_on mnemon"
    assert result["source_episode_ids"] == [str(episode_id)]
    assert result["evidence_chain"][0]["episode_id"] == str(episode_id)
    assert result["evidence_chain"][0]["context"] == "Rohit works on mnemon"


async def test_timeline_recent_includes_memory_ids() -> None:
    episode_id = uuid4()
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(
            episodic=_FakeEpisodicMemory(
                {
                    episode_id: {
                        "id": str(episode_id),
                        "timestamp": "2026-04-09T08:00:00+00:00",
                        "context": "memory context",
                        "action": "remembered",
                        "outcome": "stored",
                        "importance": 0.8,
                        "tags": ["demo"],
                        "session_id": "session-1",
                    }
                }
            )
        ),
        get_state=lambda: {"ok": True},
    )
    server = _make_server(brain)

    result = await server._rpc_timeline_recent(limit=5)

    memory_items = [item for item in result["items"] if item["kind"] == "memory"]
    assert memory_items[0]["memory_id"] == str(episode_id)
    assert memory_items[0]["tags"] == ["demo"]


async def test_memory_profile_uses_identity_and_recent_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "master.md").write_text(
        "# Who Is My Master\n\n- Loves clean tools\n\n"
        "# What Drives Them\n\n- Shipping useful agents\n\n"
        "# What They're Working On\n\n- Mnemon daemon\n\n"
        "# Patterns I've Noticed\n\n- Prefers direct answers\n\n"
        "# Questions I Want to Ask Them\n\n- What should Mnemon remember next?\n",
        encoding="utf-8",
    )
    docs = {
        uuid4(): {
            "id": "1",
            "timestamp": "2026-04-09T08:00:00+00:00",
            "context": "worked on daemon",
            "action": "coded",
            "outcome": "stored",
            "importance": 0.5,
            "tags": ["mnemon", "daemon"],
            "session_id": "s1",
        },
        uuid4(): {
            "id": "2",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "reviewed memory search",
            "action": "tested",
            "outcome": "stored",
            "importance": 0.5,
            "tags": ["mnemon", "profile_dynamic"],
            "session_id": "s2",
        },
    }

    class _FakeDaemonConfig:
        state_path = state_dir

    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                get_active_goals=lambda: [SimpleNamespace(description="Ship mnemon memory UX")]
            )
        ),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)
    monkeypatch.setattr("mnemon.daemon.ipc.DaemonConfig", _FakeDaemonConfig)

    profile = await server._rpc_memory_profile()

    assert profile["static"] == ["Loves clean tools", "Shipping useful agents"]
    assert "Mnemon daemon" in profile["dynamic"]
    assert "Ship mnemon memory UX" in profile["dynamic"]
    assert profile["questions"] == ["What should Mnemon remember next?"]
    assert profile["top_tags"][0] == {"tag": "mnemon", "count": 2}
    assert profile["static_facts"][0]["source_ids"] == []
    assert profile["dynamic_facts"][0]["source_ids"] == []
    assert profile["static_facts"][0]["citations"] == []
    assert profile["recent_changes"][0]["text"] == "reviewed memory search"


async def test_memory_profile_builds_structured_facts_from_capture_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    docs = {
        uuid4(): {
            "id": "ep-static",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "I prefer dark mode",
            "action": "said",
            "outcome": "stored",
            "importance": 0.8,
            "tags": ["profile_static", "auto_capture"],
            "session_id": "s1",
        },
        uuid4(): {
            "id": "ep-dynamic",
            "timestamp": "2026-04-09T09:05:00+00:00",
            "context": "I'm working on mnemon memory UX this week",
            "action": "said",
            "outcome": "stored",
            "importance": 0.9,
            "tags": ["profile_dynamic", "project_context", "auto_capture"],
            "session_id": "s2",
        },
    }

    class _FakeDaemonConfig:
        state_path = state_dir

    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)
    monkeypatch.setattr("mnemon.daemon.ipc.DaemonConfig", _FakeDaemonConfig)

    profile = await server._rpc_memory_profile()
    recall = await server._rpc_memory_recall(query="mnemon", top_k=3)

    assert profile["static_facts"][0]["text"] == "I prefer dark mode"
    assert profile["static_facts"][0]["source_ids"] == ["ep-static"]
    assert profile["static_facts"][0]["citations"] == ["[memory:ep-static]"]
    assert profile["dynamic_facts"][0]["text"] == "I'm working on mnemon memory UX this week"
    assert profile["dynamic_facts"][0]["source_ids"] == ["ep-dynamic"]
    assert recall["profile"]["static_facts"][0]["source_ids"] == ["ep-static"]


async def test_memory_search_honors_workspace_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        uuid4(): {
            "id": "ep-a",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "deploy with blue-green",
            "action": "said",
            "outcome": "stored",
            "importance": 0.7,
            "tags": ["project_context"],
            "session_id": "s1",
            "scope_type": "workspace",
            "scope_id": "repo-a",
            "workspace_path": "/tmp/repo-a",
            "repo_name": "repo-a",
        },
        uuid4(): {
            "id": "ep-b",
            "timestamp": "2026-04-09T09:10:00+00:00",
            "context": "deploy with canary",
            "action": "said",
            "outcome": "stored",
            "importance": 0.8,
            "tags": ["project_context"],
            "session_id": "s2",
            "scope_type": "workspace",
            "scope_id": "repo-b",
            "workspace_path": "/tmp/repo-b",
            "repo_name": "repo-b",
        },
    }
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)

    class _FakeWorkspace:
        root = Path("/tmp/repo-a")

    server._workspace = _FakeWorkspace()

    result = await server._rpc_memory_search(query="deploy", top_k=5, scope="workspace")

    assert result["scope"] == "workspace"
    assert result["scope_id"] == "repo-a"
    assert result["results"][0]["scope_id"] == "repo-a"
    assert result["profile"]["active_scope"]["scope_id"] == "repo-a"


async def test_memory_hybrid_combines_memory_goal_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = {
        uuid4(): {
            "id": "ep-a",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "deploy with blue-green",
            "action": "said",
            "outcome": "stored",
            "importance": 0.7,
            "tags": ["project_context"],
            "session_id": "s1",
            "scope_type": "workspace",
            "scope_id": "repo-a",
            "workspace_path": "/tmp/repo-a",
            "repo_name": "repo-a",
        }
    }
    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                get_active_goals=lambda: [
                    SimpleNamespace(
                        id=uuid4(),
                        description="Ship deployment workflow",
                        priority=0.8,
                        status="active",
                    )
                ]
            )
        ),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)

    class _FakeWorkspace:
        root = Path("/tmp/repo-a")

        async def list_dir(self, path: str = ".") -> dict[str, object]:
            return {"entries": [{"path": "docs/deploy.md", "type": "file"}]}

    server._workspace = _FakeWorkspace()

    result = await server._rpc_memory_hybrid(query="deploy", top_k=5, scope="workspace")

    kinds = [item["kind"] for item in result["hybrid_results"]]
    assert "memory" in kinds
    assert "goal" in kinds
    assert "workspace" in kinds


async def test_memory_graph_returns_scope_and_summary_relationships() -> None:
    docs = {
        uuid4(): {
            "id": "ep-raw",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "deploy with blue-green",
            "action": "said",
            "outcome": "stored",
            "importance": 0.7,
            "tags": ["project_context"],
            "session_id": "s1",
            "scope_type": "workspace",
            "scope_id": "repo-a",
            "workspace_path": "/tmp/repo-a",
            "repo_name": "repo-a",
            "caused_by": "",
            "led_to": ["ep-summary"],
            "source_episode_ids": [],
            "summary_kind": "",
            "summary_of_count": 0,
        },
        uuid4(): {
            "id": "ep-summary",
            "timestamp": "2026-04-09T09:10:00+00:00",
            "context": "Summary of 2 related memories",
            "action": "Compressed episodic traces into a durable summary",
            "outcome": "deployment summary",
            "importance": 0.9,
            "tags": ["project_context", "summary"],
            "session_id": "s2",
            "scope_type": "workspace",
            "scope_id": "repo-a",
            "workspace_path": "/tmp/repo-a",
            "repo_name": "repo-a",
            "caused_by": "ep-raw",
            "led_to": [],
            "source_episode_ids": ["ep-raw"],
            "summary_kind": "episodic_cluster",
            "summary_of_count": 2,
        },
    }
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)

    class _FakeWorkspace:
        root = Path("/tmp/repo-a")

    server._workspace = _FakeWorkspace()

    result = await server._rpc_memory_graph(limit=20, scope="workspace")

    kinds = [node["kind"] for node in result["nodes"]]
    edge_kinds = [edge["kind"] for edge in result["edges"]]
    assert "scope" in kinds
    assert "summary" in kinds
    assert "stored_in" in edge_kinds
    assert "summarizes" in edge_kinds
    assert "caused_by" in edge_kinds
    assert "led_to" in edge_kinds


async def test_memory_causal_trace_returns_chain() -> None:
    docs = {
        uuid4(): {
            "id": "ep-a",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "checked logs",
            "action": "triaged issue",
            "outcome": "found root cause",
            "importance": 0.7,
            "tags": ["debug"],
            "session_id": "s1",
            "caused_by": "",
            "led_to": ["ep-b"],
        },
        uuid4(): {
            "id": "ep-b",
            "timestamp": "2026-04-09T09:05:00+00:00",
            "context": "patched config",
            "action": "restarted service",
            "outcome": "incident resolved",
            "importance": 0.9,
            "tags": ["debug"],
            "session_id": "s1",
            "caused_by": "ep-a",
            "led_to": [],
        },
    }
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)

    result = await server._rpc_memory_causal_trace(episode_id="ep-b")

    assert [item["id"] for item in result["chain"]] == ["ep-a", "ep-b"]


async def test_debug_snapshot_and_clear_all_reset_stores_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"

    class _FakeDaemonConfig:
        state_path = state_dir

    episode_id = uuid4()
    semantic_id = uuid4()
    procedural_id = uuid4()
    episodic_docs = _ClearableDocumentStore({episode_id: {"id": str(episode_id)}})
    semantic_docs = _ClearableDocumentStore({semantic_id: {"id": str(semantic_id)}})
    procedural_docs = _ClearableDocumentStore({procedural_id: {"id": str(procedural_id)}})
    fake_goals = SimpleNamespace(_goals={uuid4(): object()}, get_active_goals=lambda: [object()])
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=fake_goals),
        memory=SimpleNamespace(
            episodic=SimpleNamespace(
                _document_store=episodic_docs,
                _vector_store=_ClearableVectorStore(2),
            ),
            semantic=SimpleNamespace(
                _docs=semantic_docs,
                _vectors=_ClearableVectorStore(3),
                _graph=None,
            ),
            procedural=SimpleNamespace(
                _docs=procedural_docs,
                _vectors=_ClearableVectorStore(4),
            ),
        ),
    )
    server = _make_server(brain)
    server._state.daemon_started_at = datetime.now(UTC) - timedelta(hours=74)
    server._state.total_cycles = 13
    server._state.total_idle_ticks = 1811
    old_started_at = server._state.daemon_started_at
    server._state.add_thought(
        SimpleNamespace(activity="grow", summary="demo", details={}, timestamp="now")
    )
    server._chat_history.append({"role": "user", "content": "hello"})
    monkeypatch.setattr("mnemon.daemon.ipc.DaemonConfig", _FakeDaemonConfig)

    snapshot = await server._rpc_debug_db_snapshot(limit=1)
    cleared = await server._rpc_debug_clear_all(confirm=True)

    assert snapshot["episodic"]["count"] == 1
    assert cleared["ok"] is True
    assert episodic_docs.docs == {}
    assert semantic_docs.docs == {}
    assert procedural_docs.docs == {}
    assert fake_goals._goals == {}
    assert server._state.recent_thoughts == []
    assert server._state.total_cycles == 0
    assert server._state.total_idle_ticks == 0
    assert server._state.daemon_started_at > old_started_at
    assert list(server._chat_history) == []
    assert (state_dir / "soul.md").exists()
    assert (state_dir / "goals.json").read_text(encoding="utf-8") == "[]\n"
    assert (state_dir / "daemon_state.json").exists()


async def test_scenario_run_returns_grounded_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = {
        uuid4(): {
            "id": "ep-a",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "deploy with blue-green",
            "action": "said",
            "outcome": "stored",
            "importance": 0.7,
            "tags": ["project_context"],
            "session_id": "s1",
            "scope_type": "workspace",
            "scope_id": "repo-a",
            "workspace_path": "/tmp/repo-a",
            "repo_name": "repo-a",
        }
    }

    class _FakeScenarioLLM:
        async def generate_structured(
            self,
            prompt: str,
            response_schema: dict,
            **kwargs: object,
        ) -> dict:
            del prompt, response_schema, kwargs
            return {
                "summary": "Prioritizing deployment now should improve shipping confidence.",
                "assumptions": ["Current priorities remain stable"],
                "risks": ["Other work may slip"],
                "recommendations": ["Focus on deployment first"],
                "uncertainty": "Medium",
            }

    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                _llm=_FakeScenarioLLM(),
                get_active_goals=lambda: [
                    SimpleNamespace(
                        id=uuid4(),
                        description="Ship deployment workflow",
                        priority=0.8,
                        status="active",
                    )
                ],
            )
        ),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)

    class _FakeWorkspace:
        root = Path("/tmp/repo-a")

        async def list_dir(self, path: str = ".") -> dict[str, object]:
            return {"entries": [{"path": "docs/deploy.md", "type": "file"}]}

    server._workspace = _FakeWorkspace()

    result = await server._rpc_scenario_run(
        scenario="What happens if I prioritize deployment this week?",
        scope="workspace",
    )

    assert "deployment" in result["summary"].lower()
    assert result["assumptions"] == ["Current priorities remain stable"]
    assert result["risks"] == ["Other work may slip"]
    assert result["recommendations"] == ["Focus on deployment first"]
    assert result["citations"] == ["[memory:ep-a]"]


async def test_report_run_returns_grounded_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = {
        uuid4(): {
            "id": "ep-a",
            "timestamp": "2026-04-09T09:00:00+00:00",
            "context": "deploy with blue-green",
            "action": "said",
            "outcome": "stored",
            "importance": 0.7,
            "tags": ["project_context"],
            "session_id": "s1",
            "scope_type": "workspace",
            "scope_id": "repo-a",
            "workspace_path": "/tmp/repo-a",
            "repo_name": "repo-a",
        }
    }

    class _FakeReportLLM:
        async def generate_structured(
            self,
            prompt: str,
            response_schema: dict,
            **kwargs: object,
        ) -> dict:
            del prompt, response_schema, kwargs
            return {
                "title": "Weekly report",
                "summary": "Deployment work dominated the week.",
                "highlights": ["Validated deployment approach"],
                "risks": ["Other work may slip"],
                "next_steps": ["Finish deployment workflow"],
            }

    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                _llm=_FakeReportLLM(),
                get_active_goals=lambda: [
                    SimpleNamespace(
                        id=uuid4(),
                        description="Ship deployment workflow",
                        priority=0.8,
                        status="active",
                    )
                ],
            )
        ),
        memory=SimpleNamespace(episodic=_FakeEpisodicMemory(docs)),
    )
    server = _make_server(brain)

    class _FakeWorkspace:
        root = Path("/tmp/repo-a")

        async def list_dir(self, path: str = ".") -> dict[str, object]:
            return {"entries": [{"path": "docs/deploy.md", "type": "file"}]}

    server._workspace = _FakeWorkspace()

    result = await server._rpc_report_run(
        report_type="weekly",
        focus="deployment",
        scope="workspace",
    )

    assert result["title"] == "Weekly report"
    assert result["summary"] == "Deployment work dominated the week."
    assert result["highlights"] == ["Validated deployment approach"]
    assert result["risks"] == ["Other work may slip"]
    assert result["next_steps"] == ["Finish deployment workflow"]
    assert result["citations"] == ["[memory:ep-a]"]


async def test_status_includes_channel_management_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "telegram_chat_id.txt").write_text("123456", encoding="utf-8")
    monkeypatch.setenv("JARVIS_TELEGRAM_TOKEN", "token")

    class _FakeDaemonConfig:
        telegram_token = ""
        telegram_poll_interval_s = 30
        state_path = state_dir

    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
        get_state=lambda: {"ok": True},
    )
    server = _make_server(brain)
    monkeypatch.setattr("mnemon.daemon.ipc.DaemonConfig", _FakeDaemonConfig)

    status = await server._rpc_status()

    assert status["channels"]["telegram"] == {
        "configured": True,
        "paired": True,
        "chat_id": "123456",
        "poll_interval_s": 30,
    }
    assert status["config"]["state_path"] == str(state_dir)
    assert status["pending_approvals"] == []
    assert status["chat_history"] == []


async def test_status_includes_recent_chat_history() -> None:
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
        get_state=lambda: {"ok": True},
    )
    server = _make_server(brain)
    server._chat_history.append({"role": "user", "content": "hello"})
    server._chat_history.append({"role": "assistant", "content": "hi"})

    status = await server._rpc_status()

    assert status["chat_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


async def test_status_pending_approvals_include_context() -> None:
    from mnemon.daemon.autonomy import ProposedAction
    from mnemon.daemon.config import RiskLevel

    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace()),
        memory=SimpleNamespace(),
        get_state=lambda: {"ok": True},
    )
    server = _make_server(brain)
    server._autonomy.check(
        ProposedAction(
            description="write to notes",
            risk_level=RiskLevel.HIGH,
            source="test",
            context={"path": "notes.txt", "reason": "demo"},
        )
    )

    status = await server._rpc_status()

    assert status["pending_approvals"][0]["context"] == {
        "path": "notes.txt",
        "reason": "demo",
    }


async def test_goals_update_status_updates_goal_and_pending_clear_empties_queue() -> None:
    goal_id = uuid4()

    class _FakeGoals:
        def __init__(self) -> None:
            self.updated: list[tuple[UUID, GoalStatus]] = []

        def get_active_goals(self) -> list[object]:
            return [
                SimpleNamespace(
                    id=goal_id,
                    description="ship web ui controls",
                    priority=0.8,
                    status=GoalStatus.SUSPENDED,
                    progress=0.5,
                    parent_goal_id=None,
                    subgoals=[],
                    success_criteria="ship with polish",
                )
            ]

        async def update_status(self, incoming_id: UUID, status: GoalStatus) -> None:
            self.updated.append((incoming_id, status))

    fake_goals = _FakeGoals()
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=fake_goals),
        memory=SimpleNamespace(),
        get_state=lambda: {"ok": True},
    )
    server = _make_server(brain)

    result = await server._rpc_goals_update_status(str(goal_id), "active")

    assert result["ok"] is True
    assert fake_goals.updated == [(goal_id, GoalStatus.ACTIVE)]

    from mnemon.daemon.autonomy import ProposedAction
    from mnemon.daemon.config import RiskLevel

    server._autonomy.check(
        ProposedAction(
            description="needs approval",
            risk_level=RiskLevel.HIGH,
            source="test",
        )
    )
    cleared = await server._rpc_pending_clear()
    assert cleared["cleared"] == 1


async def test_goals_update_edits_description_and_priority() -> None:
    goal_id = uuid4()

    class _FakeGoals:
        async def update_goal(
            self,
            incoming_id: UUID,
            *,
            description: str | None = None,
            priority: float | None = None,
            success_criteria: str | None = None,
        ) -> object:
            return SimpleNamespace(
                id=incoming_id,
                description=description or "unchanged",
                priority=priority if priority is not None else 0.5,
                status=GoalStatus.ACTIVE,
                progress=0.2,
                parent_goal_id=None,
                subgoals=[],
                success_criteria=success_criteria or "",
            )

    brain = SimpleNamespace(
        control=SimpleNamespace(goals=_FakeGoals()),
        memory=SimpleNamespace(),
        get_state=lambda: {"ok": True},
    )
    server = _make_server(brain)

    result = await server._rpc_goals_update(
        goal_id=str(goal_id),
        description="updated goal",
        priority=0.9,
        success_criteria="ship without regressions",
    )

    assert result["ok"] is True
    assert result["goal"]["description"] == "updated goal"
    assert result["goal"]["priority"] == 0.9
    assert result["goal"]["success_criteria"] == "ship without regressions"
