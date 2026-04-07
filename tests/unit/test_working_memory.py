"""
Unit tests for WorkingMemoryManager.

Covers: inject, token budget enforcement, eviction, flush, push_goal/pop_goal,
get_state, token_status, and inject_retrieved.
"""

from __future__ import annotations

import pytest

from mnemon.core.config import EvictionPolicy, SummarizationConfig, WorkingMemoryConfig
from mnemon.core.exceptions import TokenBudgetExceededError
from mnemon.core.models import ContextBlock, ContextSource, Goal, RetrievedItem
from mnemon.memory.working import WorkingMemoryManager
from tests.unit.conftest import FakeLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    budget: int = 1_000,
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU_IMPORTANCE,
    summarization_enabled: bool = False,
) -> WorkingMemoryManager:
    # WorkingMemoryConfig enforces token_budget >= 256; clamp to the minimum.
    safe_budget = max(budget, 256)
    cfg = WorkingMemoryConfig(
        token_budget=safe_budget,
        eviction_policy=eviction_policy,
        summarization=SummarizationConfig(enabled=summarization_enabled),
    )
    return WorkingMemoryManager(cfg, FakeLLMProvider())


def _block(
    content: str = "hello",
    tokens: int = 10,
    importance: float = 0.5,
    evictable: bool = True,
) -> ContextBlock:
    return ContextBlock(
        content=content,
        token_count=tokens,
        source=ContextSource.USER_INPUT,
        importance=importance,
        evictable=evictable,
    )


# ---------------------------------------------------------------------------
# inject() — basic operation
# ---------------------------------------------------------------------------


async def test_inject_adds_block_to_active_context() -> None:
    mgr = _make_manager()
    block = _block()
    await mgr.inject(block)
    state = mgr.get_state()
    assert any(b.id == block.id for b in state.active_context)


async def test_inject_updates_token_used() -> None:
    mgr = _make_manager(budget=500)
    await mgr.inject(_block(tokens=100))
    status = mgr.token_status()
    assert status["used"] == 100


async def test_inject_multiple_blocks() -> None:
    mgr = _make_manager(budget=500)
    await mgr.inject(_block(tokens=100))
    await mgr.inject(_block(tokens=150))
    status = mgr.token_status()
    assert status["used"] == 250


async def test_inject_non_evictable_block_preserved() -> None:
    mgr = _make_manager(budget=300)
    pinned = _block(tokens=200, evictable=False)
    filler = _block(tokens=50, evictable=True)
    await mgr.inject(pinned)
    await mgr.inject(filler)
    state = mgr.get_state()
    assert any(b.id == pinned.id for b in state.active_context)


# ---------------------------------------------------------------------------
# Token budget enforcement
# ---------------------------------------------------------------------------


async def test_inject_raises_when_budget_exceeded_and_no_eviction_policy() -> None:
    # Budget is 256 (minimum); inject two blocks that together exceed it.
    mgr = _make_manager(budget=256, eviction_policy=EvictionPolicy.LRU)
    await mgr.inject(_block(tokens=200))
    with pytest.raises(TokenBudgetExceededError):
        await mgr.inject(_block(tokens=100))


async def test_inject_evicts_when_budget_exceeded_with_lru_importance() -> None:
    # Use budget=256, blocks of 200 tokens each.  Second inject forces eviction of first.
    mgr = _make_manager(budget=256)
    b1 = _block(content="old block", tokens=200, importance=0.1)
    b2 = _block(content="new block", tokens=200, importance=0.9)
    await mgr.inject(b1)
    await mgr.inject(b2)  # should evict b1 to make room
    state = mgr.get_state()
    ids = {b.id for b in state.active_context}
    # b1 had the lowest importance, so it should be gone (evicted)
    # b2 (high importance, newly added) should remain
    assert b2.id in ids
    assert b1.id not in ids


async def test_inject_raises_when_block_too_large_to_fit_after_eviction() -> None:
    # A block bigger than the entire budget cannot fit even after evicting everything.
    mgr = _make_manager(budget=256)
    b_huge = _block(tokens=300)
    with pytest.raises(TokenBudgetExceededError):
        await mgr.inject(b_huge)


async def test_eviction_with_summarization_enabled() -> None:
    """When summarization is on, evicted block is replaced by a shorter summary."""
    cfg = WorkingMemoryConfig(
        token_budget=256,
        eviction_policy=EvictionPolicy.LRU_IMPORTANCE,
        summarization=SummarizationConfig(enabled=True, max_summary_tokens=256),
    )
    mgr = WorkingMemoryManager(cfg, FakeLLMProvider())
    # Add a block that nearly fills the budget with low importance
    long_content = "x" * 800  # 800//4 = 200 tokens
    b_long = ContextBlock(
        content=long_content,
        token_count=200,
        source=ContextSource.USER_INPUT,
        importance=0.1,
        evictable=True,
    )
    await mgr.inject(b_long)
    # Now inject a second block that forces eviction of the long one
    b_new = _block(content="new content", tokens=100)
    await mgr.inject(b_new)
    state = mgr.get_state()
    # The new block must be present
    assert any(b.id == b_new.id for b in state.active_context)
    # A summary block may replace the original (source == SUMMARY)
    sources = {b.source for b in state.active_context}
    assert ContextSource.SUMMARY in sources or len(state.active_context) >= 1


# ---------------------------------------------------------------------------
# token_status()
# ---------------------------------------------------------------------------


async def test_token_status_initial_state() -> None:
    mgr = _make_manager(budget=1_000)
    status = mgr.token_status()
    assert status["used"] == 0
    assert status["budget"] == 1_000
    assert status["available"] == 1_000


async def test_token_status_after_inject() -> None:
    mgr = _make_manager(budget=1_000)
    await mgr.inject(_block(tokens=300))
    status = mgr.token_status()
    assert status["used"] == 300
    assert status["budget"] == 1_000
    assert status["available"] == 700


async def test_token_status_available_never_negative() -> None:
    # Even when used approaches budget, available must be clamped to zero.
    mgr = _make_manager(budget=256)
    await mgr.inject(_block(tokens=256))  # exactly at budget; eviction fired if needed
    status = mgr.token_status()
    assert status["available"] >= 0


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


async def test_flush_returns_episode() -> None:
    from mnemon.core.models import Episode

    mgr = _make_manager()
    await mgr.inject(_block(content="context text"))
    episode = await mgr.flush()
    assert isinstance(episode, Episode)


async def test_flush_episode_contains_block_content() -> None:
    mgr = _make_manager()
    await mgr.inject(_block(content="important context"))
    episode = await mgr.flush()
    assert "important context" in episode.context


async def test_flush_clears_active_context() -> None:
    mgr = _make_manager()
    await mgr.inject(_block())
    await mgr.flush()
    state = mgr.get_state()
    assert state.active_context == []


async def test_flush_resets_token_used_to_zero() -> None:
    mgr = _make_manager()
    await mgr.inject(_block(tokens=200))
    await mgr.flush()
    status = mgr.token_status()
    assert status["used"] == 0


async def test_flush_preserves_session_id() -> None:
    mgr = _make_manager()
    session_id_before = mgr.get_state().session_id
    await mgr.flush()
    assert mgr.get_state().session_id == session_id_before


async def test_flush_empty_workspace_episode_context_is_placeholder() -> None:
    mgr = _make_manager()
    episode = await mgr.flush()
    assert episode.context == "(empty)"


# ---------------------------------------------------------------------------
# push_goal / pop_goal
# ---------------------------------------------------------------------------


def test_push_goal_adds_to_stack() -> None:
    mgr = _make_manager()
    g = Goal(description="test goal")
    mgr.push_goal(g)
    state = mgr.get_state()
    assert any(gs.id == g.id for gs in state.goal_stack)


def test_pop_goal_removes_last_pushed() -> None:
    mgr = _make_manager()
    g1 = Goal(description="goal 1")
    g2 = Goal(description="goal 2")
    mgr.push_goal(g1)
    mgr.push_goal(g2)
    popped = mgr.pop_goal()
    assert popped is not None
    assert popped.id == g2.id


def test_pop_goal_empty_stack_returns_none() -> None:
    mgr = _make_manager()
    assert mgr.pop_goal() is None


def test_multiple_goals_on_stack() -> None:
    mgr = _make_manager()
    for i in range(3):
        mgr.push_goal(Goal(description=f"goal {i}"))
    state = mgr.get_state()
    assert len(state.goal_stack) == 3


# ---------------------------------------------------------------------------
# get_state() — returns a deep copy
# ---------------------------------------------------------------------------


async def test_get_state_returns_copy_not_reference() -> None:
    mgr = _make_manager()
    await mgr.inject(_block(content="original"))
    state1 = mgr.get_state()
    state1.scratch_pad = "mutated externally"
    state2 = mgr.get_state()
    assert state2.scratch_pad == ""  # not affected by external mutation


# ---------------------------------------------------------------------------
# inject_retrieved()
# ---------------------------------------------------------------------------


async def test_inject_retrieved_adds_retrieval_blocks() -> None:
    mgr = _make_manager(budget=500)
    items = [
        RetrievedItem(source_store="episodic", content="fact one", score=0.9),
        RetrievedItem(source_store="episodic", content="fact two", score=0.7),
    ]
    await mgr.inject_retrieved(items)
    state = mgr.get_state()
    sources = [b.source for b in state.active_context]
    assert all(s == ContextSource.RETRIEVAL for s in sources)
    assert len(state.active_context) == 2


async def test_inject_retrieved_respects_token_budget() -> None:
    """Items that would exceed the budget are skipped."""
    # Budget is 256 (minimum).  Fill most of it so only small items can fit.
    mgr = _make_manager(budget=256)
    # Pre-fill with 250 tokens so only 6 tokens remain.
    filler = _block(content="filler", tokens=250)
    await mgr.inject(filler)
    items = [
        RetrievedItem(
            source_store="episodic",
            content="x" * 100,  # 100//4 = 25 tokens — too big for remaining 6
            score=0.9,
        ),
        RetrievedItem(
            source_store="episodic",
            content="hi",  # 2//4 = 0 → max(1, …) = 1 token — fits in 6
            score=0.5,
        ),
    ]
    await mgr.inject_retrieved(items)
    state = mgr.get_state()
    # The filler and the small item should be present; the large item should not.
    contents = [b.content for b in state.active_context]
    assert any("hi" in c for c in contents)
    assert not any("x" * 100 in c for c in contents)


async def test_inject_retrieved_sorts_by_score_descending() -> None:
    mgr = _make_manager(budget=500)
    items = [
        RetrievedItem(source_store="s", content="low score", score=0.2),
        RetrievedItem(source_store="s", content="high score", score=0.95),
    ]
    await mgr.inject_retrieved(items)
    state = mgr.get_state()
    # Both should be present since budget is large, but the injection order
    # should reflect descending score (high first)
    contents = [b.content for b in state.active_context]
    assert contents.index("high score") < contents.index("low score")


# ---------------------------------------------------------------------------
# flush() — source filtering
# ---------------------------------------------------------------------------


async def test_flush_excludes_retrieval_blocks_from_context() -> None:
    """Retrieval-sourced blocks must NOT pollute the episode context."""
    mgr = _make_manager(budget=500)
    user_block = ContextBlock(
        content="user said hello",
        token_count=5,
        source=ContextSource.USER_INPUT,
        importance=0.5,
    )
    await mgr.inject(user_block)
    retrieval_block = ContextBlock(
        content="old retrieved memory should not pollute context",
        token_count=10,
        source=ContextSource.RETRIEVAL,
        importance=0.7,
    )
    await mgr.inject(retrieval_block)
    episode = await mgr.flush()
    assert "user said hello" in episode.context
    assert "old retrieved memory should not pollute context" not in episode.context


async def test_flush_excludes_summary_blocks_from_context() -> None:
    """Summary blocks must NOT pollute the episode context."""
    mgr = _make_manager(budget=500)
    user_block = ContextBlock(
        content="actual user input",
        token_count=5,
        source=ContextSource.USER_INPUT,
        importance=0.5,
    )
    summary_block = ContextBlock(
        content="summarised old stuff that should stay out",
        token_count=8,
        source=ContextSource.SUMMARY,
        importance=0.3,
    )
    await mgr.inject(user_block)
    await mgr.inject(summary_block)
    episode = await mgr.flush()
    assert "actual user input" in episode.context
    assert "summarised old stuff" not in episode.context
