"""
Unit tests for CognitiveBus.

Covers:
- publish / subscribe delivery
- Multiple subscribers on the same topic
- Unsubscribe stops delivery
- Message filtering via EventFilter
- Direct vs broadcast routing
- start / stop lifecycle
- Publishing before start raises RuntimeError
- subscription_count / is_running diagnostics
"""

from __future__ import annotations

import anyio
import pytest

from mnemon.core.bus import CognitiveBus, EventFilter
from mnemon.core.models import CognitiveMessage, MessageType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(
    source: str = "sender",
    target: str = "*",
    msg_type: MessageType = MessageType.BROADCAST,
    priority: float = 0.5,
) -> CognitiveMessage:
    return CognitiveMessage(
        source=source,
        target=target,
        type=msg_type,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_is_running_before_start_is_false() -> None:
    bus = CognitiveBus()
    assert bus.is_running() is False


async def test_is_running_after_start_is_true() -> None:
    bus = CognitiveBus()
    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        assert bus.is_running() is True
        await bus.stop()


async def test_is_running_after_stop_is_false() -> None:
    bus = CognitiveBus()
    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.stop()
        assert bus.is_running() is False


async def test_double_start_is_noop() -> None:
    """Calling start() on an already-running bus should not raise."""
    bus = CognitiveBus()
    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.start(tg)  # second call is a no-op
        assert bus.is_running() is True
        await bus.stop()


async def test_stop_on_not_started_bus_is_noop() -> None:
    bus = CognitiveBus()
    await bus.stop()  # should not raise
    assert bus.is_running() is False


# ---------------------------------------------------------------------------
# publish before start
# ---------------------------------------------------------------------------


async def test_publish_before_start_raises_runtime_error() -> None:
    bus = CognitiveBus()
    with pytest.raises(RuntimeError, match="start()"):
        await bus.publish(_msg())


# ---------------------------------------------------------------------------
# subscribe / unsubscribe
# ---------------------------------------------------------------------------


def test_subscribe_returns_subscription_id() -> None:
    bus = CognitiveBus()

    async def handler(m: CognitiveMessage) -> None:
        pass

    sub_id = bus.subscribe(handler)
    assert isinstance(sub_id, str)
    assert len(sub_id) > 0


def test_subscription_count_increments() -> None:
    bus = CognitiveBus()

    async def h(m: CognitiveMessage) -> None:
        pass

    assert bus.subscription_count() == 0
    bus.subscribe(h)
    assert bus.subscription_count() == 1
    bus.subscribe(h)
    assert bus.subscription_count() == 2


def test_unsubscribe_decrements_count() -> None:
    bus = CognitiveBus()

    async def h(m: CognitiveMessage) -> None:
        pass

    sid = bus.subscribe(h)
    bus.unsubscribe(sid)
    assert bus.subscription_count() == 0


def test_unsubscribe_unknown_id_is_noop() -> None:
    bus = CognitiveBus()
    bus.unsubscribe("definitely-not-a-real-id")  # should not raise


# ---------------------------------------------------------------------------
# publish / subscribe message delivery
# ---------------------------------------------------------------------------


async def test_subscriber_receives_broadcast_message() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    bus.subscribe(handler)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        msg = _msg(target="*")
        await bus.publish(msg)
        await anyio.sleep(0.05)  # give dispatch loop time to process
        await bus.stop()

    assert len(received) == 1
    assert received[0].id == msg.id


async def test_multiple_subscribers_all_receive_broadcast() -> None:
    bus = CognitiveBus()
    received_a: list[CognitiveMessage] = []
    received_b: list[CognitiveMessage] = []

    async def handler_a(m: CognitiveMessage) -> None:
        received_a.append(m)

    async def handler_b(m: CognitiveMessage) -> None:
        received_b.append(m)

    bus.subscribe(handler_a)
    bus.subscribe(handler_b)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg(target="*"))
        await anyio.sleep(0.05)
        await bus.stop()

    assert len(received_a) == 1
    assert len(received_b) == 1


async def test_unsubscribed_handler_does_not_receive_messages() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    sid = bus.subscribe(handler)
    bus.unsubscribe(sid)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg())
        await anyio.sleep(0.05)
        await bus.stop()

    assert received == []


async def test_direct_message_delivered_to_matching_module() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    bus.subscribe(handler, module_id="module_A")

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg(target="module_A"))
        await anyio.sleep(0.05)
        await bus.stop()

    assert len(received) == 1


async def test_direct_message_not_delivered_to_wrong_module() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    bus.subscribe(handler, module_id="module_B")

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg(target="module_A"))  # send to A, not B
        await anyio.sleep(0.05)
        await bus.stop()

    assert received == []


async def test_broadcast_delivered_to_module_scoped_subscriber() -> None:
    """A module-scoped subscriber should still receive broadcast (target='*') messages."""
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    bus.subscribe(handler, module_id="module_C")

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg(target="*"))
        await anyio.sleep(0.05)
        await bus.stop()

    assert len(received) == 1


# ---------------------------------------------------------------------------
# EventFilter
# ---------------------------------------------------------------------------


async def test_event_filter_by_message_type() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    flt = EventFilter(message_type=MessageType.PERCEPT)
    bus.subscribe(handler, filter=flt)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        # This should be delivered
        await bus.publish(_msg(msg_type=MessageType.PERCEPT))
        # This should be filtered out
        await bus.publish(_msg(msg_type=MessageType.REWARD_SIGNAL))
        await anyio.sleep(0.05)
        await bus.stop()

    assert len(received) == 1
    assert received[0].type == MessageType.PERCEPT


async def test_event_filter_by_min_priority() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    flt = EventFilter(min_priority=0.8)
    bus.subscribe(handler, filter=flt)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg(priority=0.9))  # passes
        await bus.publish(_msg(priority=0.3))  # filtered
        await anyio.sleep(0.05)
        await bus.stop()

    assert len(received) == 1
    assert received[0].priority == pytest.approx(0.9)


async def test_event_filter_by_source_module() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    flt = EventFilter(source_module="trusted_source")
    bus.subscribe(handler, filter=flt)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg(source="trusted_source"))
        await bus.publish(_msg(source="untrusted_source"))
        await anyio.sleep(0.05)
        await bus.stop()

    assert len(received) == 1
    assert received[0].source == "trusted_source"


async def test_event_filter_custom_predicate() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    flt = EventFilter(custom_predicate=lambda m: m.payload.get("flag") is True)
    bus.subscribe(handler, filter=flt)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(
            CognitiveMessage(
                source="s", target="*", type=MessageType.BROADCAST, payload={"flag": True}
            )
        )
        await bus.publish(
            CognitiveMessage(
                source="s", target="*", type=MessageType.BROADCAST, payload={"flag": False}
            )
        )
        await anyio.sleep(0.05)
        await bus.stop()

    assert len(received) == 1
    assert received[0].payload["flag"] is True


# ---------------------------------------------------------------------------
# Multiple messages in sequence
# ---------------------------------------------------------------------------


async def test_multiple_messages_all_delivered_in_order() -> None:
    bus = CognitiveBus()
    received: list[CognitiveMessage] = []

    async def handler(m: CognitiveMessage) -> None:
        received.append(m)

    bus.subscribe(handler)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        msgs = [_msg(source=f"src_{i}") for i in range(5)]
        for m in msgs:
            await bus.publish(m)
        await anyio.sleep(0.1)
        await bus.stop()

    assert len(received) == 5
    received_ids = {m.id for m in received}
    expected_ids = {m.id for m in msgs}
    assert received_ids == expected_ids


# ---------------------------------------------------------------------------
# Handler error isolation
# ---------------------------------------------------------------------------


async def test_handler_exception_does_not_kill_bus() -> None:
    """A handler that raises must not crash the dispatch loop."""
    bus = CognitiveBus()
    received_by_good: list[CognitiveMessage] = []

    async def bad_handler(m: CognitiveMessage) -> None:
        raise RuntimeError("intentional test error")

    async def good_handler(m: CognitiveMessage) -> None:
        received_by_good.append(m)

    bus.subscribe(bad_handler)
    bus.subscribe(good_handler)

    async with anyio.create_task_group() as tg:
        await bus.start(tg)
        await bus.publish(_msg())
        await anyio.sleep(0.05)
        # Bus must still be running despite the handler error
        assert bus.is_running() is True
        # And the good handler still received the message
        assert len(received_by_good) == 1
        await bus.stop()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr_contains_running_state() -> None:
    bus = CognitiveBus()
    r = repr(bus)
    assert "running=False" in r
    assert "subscriptions=0" in r
