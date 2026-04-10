from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from mnemon.core.models import (
    ConsolidationState,
    Episode,
    Goal,
    GoalStatus,
    MemoryLifecycleState,
    Skill,
    SkillType,
)
from mnemon.daemon.config import IdleLoopConfig
from mnemon.daemon.loop import IdleThinkingLoop
from mnemon.daemon.state import DaemonState


class _FakeDocumentStore:
    def __init__(self, docs: dict[str, dict[str, object]]) -> None:
        self._docs = docs

    async def query(self, filters: dict[str, object], limit: int) -> list[dict[str, object]]:
        matched = [
            doc
            for doc in self._docs.values()
            if all(doc.get(key) == value for key, value in filters.items())
        ]
        return matched[:limit]

    async def get(self, key: object) -> dict[str, object] | None:
        return self._docs.get(str(key))


class _FakeEpisodicMemory:
    def __init__(self, episodes: list[Episode]) -> None:
        self._docs = {str(episode.id): episode.model_dump(mode="json") for episode in episodes}
        self._document_store = _FakeDocumentStore(self._docs)
        self.decay_sweeps = 0

    async def update(self, episode_id, **updates: object) -> None:
        doc = self._docs[str(episode_id)]
        normalized_updates: dict[str, object] = {}
        for key, value in updates.items():
            if isinstance(value, UUID):
                normalized_updates[key] = str(value)
            elif isinstance(value, list) and all(isinstance(item, UUID) for item in value):
                normalized_updates[key] = [str(item) for item in value]
            else:
                normalized_updates[key] = value
        doc.update(normalized_updates)

    async def run_decay_sweep(self) -> int:
        self.decay_sweeps += 1
        return 2

    async def sample_for_consolidation(self, batch_size: int = 32) -> list[Episode]:
        del batch_size
        return [Episode.model_validate(doc) for doc in self._docs.values()]


class _FakeSemanticMemory:
    def __init__(self, docs: list[dict[str, object]]) -> None:
        self._docs = _FakeDocumentStore({str(doc["id"]): doc for doc in docs})
        self._entity_store = _FakeDocumentStore({})
        self.maintenance_runs = 0

    async def run_maintenance(self) -> None:
        self.maintenance_runs += 1


def _make_loop(brain: object) -> IdleThinkingLoop:
    return IdleThinkingLoop(
        brain=brain,
        config=IdleLoopConfig(),
        state=DaemonState(),
        state_dir=Path("/tmp"),
    )


def test_reframes_learning_homework_as_jarvis_owned_action() -> None:
    reframed = IdleThinkingLoop._reframe_master_homework(
        "Your master should research neural networks and schedule a meeting."
    )

    assert "I will investigate this myself" in reframed
    assert "master should" not in reframed.lower()
    assert "schedule a meeting" not in reframed.lower()


def test_replaces_ungrounded_external_source_claims() -> None:
    cleaned = IdleThinkingLoop._replace_ungrounded_external_claims(
        "I reviewed your email inbox and calendar to find a blocker.",
        "Recent conversations:\n- can you make money online for me",
        "I need grounded source data before making that claim.",
    )

    assert cleaned == "I need grounded source data before making that claim."


class _FakeConsolidation:
    async def run_cycle(self):
        return SimpleNamespace(
            episodes_processed=2,
            triples_extracted=1,
            entities_resolved=1,
            duration_ms=5.0,
        )


class _FakeSelfLearningLLM:
    async def generate(self, prompt: str, **kwargs: object) -> str:
        assert "Do not say 'I should learn'" in prompt
        return (
            "I learned that hippocampal replay strengthens useful episodic traces "
            "by reactivating them during offline periods. I will compare recent "
            "raw episodes against active goals and prioritize replay for goal-linked traces."
        )


class _FakeReflectionLLM:
    async def generate(self, prompt: str, **kwargs: object) -> str:
        assert "Actual workspace context" in prompt
        return "I am currently grounded in local workspace context and empty memory state."


class _FakeSkillNeed:
    description = "Reusable deploy checklist"
    trigger_pattern = "deploy tag appears repeatedly"
    evidence_count = 3


class _FakeSkillAcquirer:
    def __init__(self) -> None:
        self.acquired = False

    async def detect_skill_need(self, recent_episodes: list[Episode]) -> list[_FakeSkillNeed]:
        assert len(recent_episodes) >= 3
        return [_FakeSkillNeed()]

    async def acquire_skill(self, need: _FakeSkillNeed) -> Skill:
        self.acquired = True
        return Skill(
            name="deploy_checklist",
            description=need.description,
            type=SkillType.WORKFLOW_DAG,
            definition="Check logs, deploy, verify rollback.",
        )


@pytest.mark.asyncio
async def test_goal_linked_priority_prefers_active_goals() -> None:
    active_goal = Goal(description="ship feature", priority=0.9, status=GoalStatus.ACTIVE)
    failed_goal = Goal(description="abandoned idea", priority=0.1, status=GoalStatus.FAILED)
    failed_episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="failed trace",
        action="tried an abandoned path",
        outcome="dropped",
        goal_id=failed_goal.id,
    )
    active_episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="active trace",
        action="worked the critical path",
        outcome="progress",
        goal_id=active_goal.id,
    )

    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                _goals={
                    active_goal.id: active_goal,
                    failed_goal.id: failed_goal,
                }
            )
        ),
        memory=SimpleNamespace(),
    )
    loop = _make_loop(brain)

    prioritized = loop._prioritize_goal_linked_episodes([failed_episode, active_episode])

    assert [episode.id for episode in prioritized] == [active_episode.id, failed_episode.id]


@pytest.mark.asyncio
async def test_apply_goal_anchored_lifecycle_promotes_and_accelerates() -> None:
    completed_goal = Goal(description="completed goal", status=GoalStatus.COMPLETED)
    blocked_goal = Goal(description="blocked goal", status=GoalStatus.BLOCKED)
    completed_episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="completed work",
        action="finished the task",
        outcome="done",
        goal_id=completed_goal.id,
        lifecycle_state=MemoryLifecycleState.CONSOLIDATED,
        consolidation_state=ConsolidationState.CONSOLIDATED,
    )
    blocked_episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="blocked work",
        action="hit a dead end",
        outcome="stalled",
        goal_id=blocked_goal.id,
        decay_lambda=0.01,
    )
    episodic = _FakeEpisodicMemory([completed_episode, blocked_episode])
    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                _goals={
                    completed_goal.id: completed_goal,
                    blocked_goal.id: blocked_goal,
                }
            )
        ),
        memory=SimpleNamespace(episodic=episodic),
    )
    loop = _make_loop(brain)

    result = await loop._apply_goal_anchored_lifecycle()

    assert result == {"promoted_to_summary": 1, "accelerated_decay": 1}
    assert (
        episodic._docs[str(completed_episode.id)]["lifecycle_state"] == MemoryLifecycleState.SUMMARY
    )
    assert episodic._docs[str(blocked_episode.id)]["decay_lambda"] > 0.01


@pytest.mark.asyncio
async def test_explore_surfaces_contradiction_insight() -> None:
    semantic = _FakeSemanticMemory(
        [
            {
                "id": str(uuid4()),
                "_type": "triple",
                "subject": {"entity_id": str(uuid4()), "name": "Rohit"},
                "predicate": "uses_provider",
                "object": "OpenAI",
                "confidence": 0.6,
                "current": False,
                "contradiction_group": "provider",
                "source_episodes": [],
                "last_confirmed": "2026-04-08T09:00:00+00:00",
            },
            {
                "id": str(uuid4()),
                "_type": "triple",
                "subject": {"entity_id": str(uuid4()), "name": "Rohit"},
                "predicate": "uses_provider",
                "object": "Anthropic",
                "confidence": 0.9,
                "current": True,
                "contradiction_group": "provider",
                "source_episodes": [],
                "last_confirmed": "2026-04-09T09:00:00+00:00",
            },
        ]
    )
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(
            semantic=semantic,
            episodic=SimpleNamespace(_document_store=_FakeDocumentStore({})),
        ),
    )
    loop = _make_loop(brain)

    result = await loop._explore()

    assert semantic.maintenance_runs == 1
    assert result["insight_type"] == "contradiction"
    assert result["share_with_user"] is True
    assert "Anthropic" in result["summary"]


@pytest.mark.asyncio
async def test_explore_surfaces_cross_session_connection() -> None:
    first = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="solved deploy issue in repo a",
        action="used blue-green rollout",
        outcome="worked",
        tags=["deploy"],
        scope_type="workspace",
        scope_id="repo-a",
        repo_name="repo-a",
    )
    second = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="solved deploy issue in repo b",
        action="used canary rollout",
        outcome="worked",
        tags=["deploy"],
        scope_type="workspace",
        scope_id="repo-b",
        repo_name="repo-b",
    )
    semantic = _FakeSemanticMemory([])
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(semantic=semantic, episodic=_FakeEpisodicMemory([first, second])),
    )
    loop = _make_loop(brain)

    result = await loop._explore()

    assert result["insight_type"] == "cross_session_connection"
    assert "repo-a" in result["proactive_message"]
    assert "repo-b" in result["proactive_message"]


@pytest.mark.asyncio
async def test_explore_mines_causal_links_from_sequential_episodes() -> None:
    first = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="started fix",
        action="opened logs",
        outcome="found clue",
    )
    second = Episode(
        agent_id="test-agent",
        session_id=first.session_id,
        context="continued fix",
        action="patched config",
        outcome="resolved outage",
    )
    semantic = _FakeSemanticMemory([])
    episodic = _FakeEpisodicMemory([first, second])
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(semantic=semantic, episodic=episodic),
    )
    loop = _make_loop(brain)

    result = await loop._explore()

    assert result["causal_links"]["linked"] >= 2
    assert episodic._docs[str(second.id)]["caused_by"] == str(first.id)
    assert str(second.id) in episodic._docs[str(first.id)]["led_to"]


@pytest.mark.asyncio
async def test_repeated_pattern_acquires_skill_when_acquirer_available() -> None:
    episodes = [
        Episode(
            agent_id="test-agent",
            session_id=uuid4(),
            context=f"deploy step {idx}",
            action="ran deploy",
            outcome="stored",
            tags=["deploy"],
            scope_id="repo-a",
        )
        for idx in range(3)
    ]
    skill_acquirer = _FakeSkillAcquirer()
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(get_active_goals=lambda: [])),
        memory=SimpleNamespace(
            semantic=_FakeSemanticMemory([]),
            episodic=_FakeEpisodicMemory(episodes),
        ),
        learning=SimpleNamespace(skill_acquirer=skill_acquirer),
    )
    loop = _make_loop(brain)

    result = await loop._explore()

    assert skill_acquirer.acquired is True
    assert result["candidate_skill"]["acquired_skill_name"] == "deploy_checklist"


@pytest.mark.asyncio
async def test_consolidate_runs_decay_sweep() -> None:
    goal = Goal(description="ship feature", priority=0.9, status=GoalStatus.ACTIVE)
    episode = Episode(
        agent_id="test-agent",
        session_id=uuid4(),
        context="important task",
        action="made progress",
        outcome="stored",
        goal_id=goal.id,
    )
    episodic = _FakeEpisodicMemory([episode])
    brain = SimpleNamespace(
        control=SimpleNamespace(goals=SimpleNamespace(_goals={goal.id: goal})),
        memory=SimpleNamespace(episodic=episodic),
        learning=SimpleNamespace(
            replay_buffer=SimpleNamespace(add=lambda **kwargs: None),
            consolidation=_FakeConsolidation(),
        ),
    )
    loop = _make_loop(brain)

    result = await loop._consolidate()

    assert episodic.decay_sweeps == 1
    assert result["decay_pruned"] == 2


@pytest.mark.asyncio
async def test_grow_records_self_learning_without_assigning_master_homework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mnemon.daemon.loop.random.choice", lambda _items: "self_learning")
    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                _llm=_FakeSelfLearningLLM(),
                get_active_goals=lambda: [Goal(description="build autonomous memory")],
            )
        ),
        memory=SimpleNamespace(semantic=_FakeSemanticMemory([])),
    )
    loop = IdleThinkingLoop(
        brain=brain,
        config=IdleLoopConfig(),
        state=DaemonState(),
        state_dir=tmp_path,
    )

    result = await loop._grow()
    learnings = (tmp_path / "learnings.md").read_text(encoding="utf-8")
    soul = (tmp_path / "soul.md").read_text(encoding="utf-8")

    assert result["mode"] == "self_learning"
    assert "hippocampal replay" in learnings
    assert "I should learn" not in learnings
    assert "master should learn" not in learnings.lower()
    assert "hippocampal replay" not in soul


@pytest.mark.asyncio
async def test_grow_who_am_i_uses_workspace_context_without_goal_context_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mnemon.daemon.loop.random.choice", lambda _items: "who_am_i")
    brain = SimpleNamespace(
        control=SimpleNamespace(
            goals=SimpleNamespace(
                _llm=_FakeReflectionLLM(),
                get_active_goals=lambda: [],
            )
        ),
        memory=SimpleNamespace(semantic=_FakeSemanticMemory([])),
    )
    loop = IdleThinkingLoop(
        brain=brain,
        config=IdleLoopConfig(),
        state=DaemonState(),
        state_dir=tmp_path,
    )

    result = await loop._grow()

    assert result["mode"] == "who_am_i"
    assert "workspace context" in (tmp_path / "soul.md").read_text(encoding="utf-8")
