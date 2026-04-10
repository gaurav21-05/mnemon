from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemon.daemon.autonomy import AutonomyController
from mnemon.daemon.config import DaemonConfig
from mnemon.daemon.ipc import DaemonIPCServer
from mnemon.daemon.privacy import PrivacyRules
from mnemon.daemon.state import DaemonState

pytestmark = pytest.mark.asyncio


class _DummyIdleLoop:
    is_busy = False

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


class _FakeGoalManager:
    def __init__(self, descriptions: list[str] | None = None) -> None:
        self._descriptions = descriptions or []

    def get_active_goals(self) -> list[object]:
        return [SimpleNamespace(description=item) for item in self._descriptions]


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.suppressed = False
        self.last_outcome: str | None = None
        self.last_metadata: dict[str, object] | None = None
        self.redactions: list[str] = []
        self.feedback_ids: list[str] = []

    def suppress_next_episode_storage(self) -> None:
        self.suppressed = True

    def configure_next_episode_redactions(self, redactions: list[str]) -> None:
        self.redactions = list(redactions)

    async def update_last_episode_outcome(self, outcome: str) -> None:
        self.last_outcome = outcome

    async def update_last_episode_metadata(self, **updates: object) -> None:
        self.last_metadata = updates

    async def record_retrieval_feedback(
        self,
        memory_ids: list[str],
        helpful: bool = True,
    ) -> None:
        if helpful:
            self.feedback_ids.extend(memory_ids)


class _FakeBrain:
    def __init__(self, goals: list[str] | None = None) -> None:
        self.orchestrator = _FakeOrchestrator()
        self.control = SimpleNamespace(goals=_FakeGoalManager(goals))

    async def run_cycle(self, raw_input: str | None = None) -> dict[str, object]:
        return {
            "cycle_number": 1,
            "phases_completed": ["perception", "learning"],
            "retrieved_count": 0,
            "deliberation": {"citation_ids": ["ep-1", "ep-2"]},
            "meta_evaluation": None,
            "raw_input": raw_input,
        }


def _make_server(goals: list[str] | None = None) -> tuple[DaemonIPCServer, _FakeBrain]:
    brain = _FakeBrain(goals)
    server = DaemonIPCServer(
        socket_path=Path("/tmp/mnemon-test.sock"),
        brain=brain,
        state=DaemonState(),
        autonomy=AutonomyController(DaemonConfig()),
        idle_loop=_DummyIdleLoop(),
    )
    return server, brain


async def test_rpc_chat_suppresses_private_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    server, brain = _make_server()

    async def fake_handle_tool_request(_message: str) -> None:
        return None

    async def fake_is_browse_request(_message: str) -> bool:
        return False

    async def fake_generate_reply(
        message: str,
        deliberation: dict[str, object],
        pending_curiosity: str = "",
        browse_result: str = "",
    ) -> str:
        del message, deliberation, pending_curiosity, browse_result
        return "I won't retain that."

    monkeypatch.setattr(server, "_handle_tool_request", fake_handle_tool_request)
    monkeypatch.setattr(server, "_is_browse_request", fake_is_browse_request)
    monkeypatch.setattr(server, "_generate_reply", fake_generate_reply)

    result = await server._rpc_chat("<private>do not remember this</private>")

    assert result["reply"] == "I won't retain that."
    assert brain.orchestrator.suppressed is True
    assert brain.orchestrator.last_outcome is None
    assert brain.orchestrator.last_metadata is None


async def test_rpc_chat_applies_auto_capture_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, brain = _make_server(["Ship memory UX"])

    async def fake_handle_tool_request(_message: str) -> None:
        return None

    async def fake_is_browse_request(_message: str) -> bool:
        return False

    async def fake_generate_reply(
        message: str,
        deliberation: dict[str, object],
        pending_curiosity: str = "",
        browse_result: str = "",
    ) -> str:
        del message, deliberation, pending_curiosity, browse_result
        return "Got it — you prefer dark mode and are working on mnemon."

    monkeypatch.setattr(server, "_handle_tool_request", fake_handle_tool_request)
    monkeypatch.setattr(server, "_is_browse_request", fake_is_browse_request)
    monkeypatch.setattr(server, "_generate_reply", fake_generate_reply)

    result = await server._rpc_chat("I prefer dark mode and I'm working on mnemon this week.")

    assert "Got it" in result["reply"]
    assert brain.orchestrator.suppressed is False
    assert brain.orchestrator.last_outcome == result["reply"]
    assert brain.orchestrator.last_metadata is not None
    tags = brain.orchestrator.last_metadata["tags"]
    assert isinstance(tags, list)
    assert "auto_capture" in tags
    assert "profile_static" in tags
    assert "profile_dynamic" in tags
    assert "project_context" in tags


async def test_rpc_chat_appends_citations_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, brain = _make_server(["Ship memory UX"])

    async def fake_handle_tool_request(_message: str) -> None:
        return None

    async def fake_is_browse_request(_message: str) -> bool:
        return False

    async def fake_generate_reply(
        message: str,
        deliberation: dict[str, object],
        pending_curiosity: str = "",
        browse_result: str = "",
    ) -> str:
        del message, deliberation, pending_curiosity, browse_result
        return "Here is the answer."

    monkeypatch.setattr(server, "_handle_tool_request", fake_handle_tool_request)
    monkeypatch.setattr(server, "_is_browse_request", fake_is_browse_request)
    monkeypatch.setattr(server, "_generate_reply", fake_generate_reply)

    result = await server._rpc_chat("answer with citations please")

    assert "Sources: [memory:ep-1] [memory:ep-2]" in result["reply"]
    assert result["citations"] == ["[memory:ep-1]", "[memory:ep-2]"]
    assert brain.orchestrator.last_outcome is not None
    assert brain.orchestrator.feedback_ids == ["ep-1", "ep-2"]


async def test_rpc_chat_uses_persisted_exclusion_and_redaction_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, brain = _make_server()

    async def fake_handle_tool_request(_message: str) -> None:
        return None

    async def fake_is_browse_request(_message: str) -> bool:
        return False

    async def fake_generate_reply(
        message: str,
        deliberation: dict[str, object],
        pending_curiosity: str = "",
        browse_result: str = "",
    ) -> str:
        del message, deliberation, pending_curiosity, browse_result
        return "Stored token API_KEY_123"

    monkeypatch.setattr(
        "mnemon.daemon.ipc.load_privacy_rules",
        lambda _state_path: PrivacyRules(
            excluded_phrases=["secret project"],
            redaction_phrases=["API_KEY_123"],
        ),
    )
    monkeypatch.setattr(server, "_handle_tool_request", fake_handle_tool_request)
    monkeypatch.setattr(server, "_is_browse_request", fake_is_browse_request)
    monkeypatch.setattr(server, "_generate_reply", fake_generate_reply)

    excluded = await server._rpc_chat("This concerns my secret project")
    assert brain.orchestrator.suppressed is True
    assert excluded["reply"] == "Stored token API_KEY_123"

    brain.orchestrator.suppressed = False
    redacted = await server._rpc_chat(
        "I'm working on mnemon and need to remember API_KEY_123 for this deployment"
    )
    assert brain.orchestrator.redactions == ["API_KEY_123"]
    assert brain.orchestrator.last_outcome == "Stored token [REDACTED]"
    assert redacted["reply"] == "Stored token [REDACTED]"
