"""
Unit tests for SensoryBuffer.

Covers percept creation, buffering, capacity limiting (oldest evicted),
TTL expiry, and the peek/clear API.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mnemon.core.config import SensoryConfig
from mnemon.core.models import Modality, PerceptUnit
from mnemon.memory.sensory import SensoryBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buffer(capacity: int = 16, ttl_ms: int = 30_000) -> SensoryBuffer:
    # SensoryConfig enforces ttl_ms >= 100; clamp to the minimum so helper
    # callers can still pass arbitrary values for capacity-only tests.
    safe_ttl = max(ttl_ms, 100)
    cfg = SensoryConfig(capacity=capacity, ttl_ms=safe_ttl)
    return SensoryBuffer(cfg)


def _past_percept(ttl_ms: int = 100) -> PerceptUnit:
    """Return a PerceptUnit already expired (timestamp in the past)."""
    return PerceptUnit(
        modality=Modality.TEXT,
        raw_content="old",
        normalized="old",
        tokens=1,
        ttl_ms=ttl_ms,
        timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),  # definitely expired
    )


# ---------------------------------------------------------------------------
# process() — basic percept creation
# ---------------------------------------------------------------------------


async def test_process_returns_percept_unit() -> None:
    buf = _make_buffer()
    percept = await buf.process("Hello World", modality=Modality.TEXT)
    assert isinstance(percept, PerceptUnit)


async def test_process_sets_modality() -> None:
    buf = _make_buffer()
    percept = await buf.process("click!", modality=Modality.TOOL_OUTPUT)
    assert percept.modality == Modality.TOOL_OUTPUT


async def test_process_preserves_raw_content() -> None:
    buf = _make_buffer()
    percept = await buf.process("  Hello World  ")
    assert percept.raw_content == "  Hello World  "


async def test_process_strips_and_lowercases_normalized() -> None:
    buf = _make_buffer()
    percept = await buf.process("  Hello World  ")
    assert percept.normalized == "hello world"


async def test_process_computes_token_count() -> None:
    buf = _make_buffer()
    text = "a" * 40  # 40 chars → 40//4 = 10 tokens
    percept = await buf.process(text)
    assert percept.tokens == 10


async def test_process_token_count_minimum_one() -> None:
    buf = _make_buffer()
    percept = await buf.process("")  # empty → max(1, 0//4) = 1
    assert percept.tokens == 1


async def test_process_sets_ttl_from_config() -> None:
    buf = _make_buffer(ttl_ms=12_345)
    percept = await buf.process("test")
    assert percept.ttl_ms == 12_345


async def test_process_embedding_is_none_by_default() -> None:
    buf = _make_buffer()
    percept = await buf.process("test")
    assert percept.embedding is None


async def test_process_entities_empty_by_default() -> None:
    buf = _make_buffer()
    percept = await buf.process("test")
    assert percept.entities == []


async def test_process_sentiment_zero_by_default() -> None:
    buf = _make_buffer()
    percept = await buf.process("test")
    assert percept.sentiment == pytest.approx(0.0)


async def test_process_attended_false_by_default() -> None:
    buf = _make_buffer()
    percept = await buf.process("test")
    assert percept.attended is False


# ---------------------------------------------------------------------------
# peek() — returns buffered items
# ---------------------------------------------------------------------------


async def test_peek_returns_empty_list_initially() -> None:
    buf = _make_buffer()
    assert buf.peek() == []


async def test_peek_returns_buffered_percepts() -> None:
    buf = _make_buffer()
    await buf.process("alpha")
    await buf.process("beta")
    live = buf.peek()
    assert len(live) == 2


async def test_peek_does_not_consume_percepts() -> None:
    buf = _make_buffer()
    await buf.process("hello")
    _ = buf.peek()
    assert len(buf.peek()) == 1


# ---------------------------------------------------------------------------
# Capacity limit — oldest evicted when full
# ---------------------------------------------------------------------------


async def test_capacity_limit_enforced() -> None:
    buf = _make_buffer(capacity=3)
    for i in range(5):
        await buf.process(f"item {i}")
    assert len(buf.peek()) == 3


async def test_capacity_oldest_percept_evicted() -> None:
    """The buffer is a deque with maxlen; the first item added is dropped first."""
    buf = _make_buffer(capacity=2)
    p1 = await buf.process("first")
    p2 = await buf.process("second")
    p3 = await buf.process("third")  # should evict p1
    live = buf.peek()
    ids = {p.id for p in live}
    assert p1.id not in ids
    assert p2.id in ids
    assert p3.id in ids


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


async def test_expired_percepts_pruned_on_peek() -> None:
    buf = _make_buffer(ttl_ms=1)
    # Manually insert an already-expired percept
    expired = _past_percept(ttl_ms=1)
    buf._buffer.append(expired)
    # peek() should prune it
    live = buf.peek()
    assert len(live) == 0


async def test_live_percepts_not_pruned() -> None:
    buf = _make_buffer(ttl_ms=60_000)
    await buf.process("still alive")
    live = buf.peek()
    assert len(live) == 1


async def test_mixed_expired_and_live_percepts(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _make_buffer(ttl_ms=30_000)
    # Add a live percept normally
    await buf.process("live item")
    # Inject an already-expired percept directly
    expired = _past_percept(ttl_ms=1)
    buf._buffer.append(expired)

    live = buf.peek()
    assert len(live) == 1
    assert live[0].normalized == "live item"


async def test_process_prunes_expired_before_adding() -> None:
    """Expired items should be removed before the new item is inserted."""
    buf = _make_buffer(capacity=2, ttl_ms=30_000)
    # Force an expired item into the buffer
    expired = _past_percept(ttl_ms=1)
    buf._buffer.append(expired)
    # Now the buffer has one item (expired).  Adding two more should not
    # overflow the capacity because the expired one gets pruned first.
    await buf.process("item A")
    await buf.process("item B")
    live = buf.peek()
    assert len(live) == 2
    normalized = {p.normalized for p in live}
    assert "item a" in normalized
    assert "item b" in normalized


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


async def test_clear_empties_buffer() -> None:
    buf = _make_buffer()
    await buf.process("to be cleared")
    buf.clear()
    assert buf.peek() == []


async def test_clear_on_empty_buffer_is_noop() -> None:
    buf = _make_buffer()
    buf.clear()
    assert buf.peek() == []
