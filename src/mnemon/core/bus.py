"""
CognitiveBus — the thalamic relay for inter-module communication.

Brain analog
------------
The thalamus is not a passive wire; it actively gates and transforms
cortico-cortical signals, enforcing selective routing and synchrony.
CognitiveBus mirrors this by:
  - Routing direct messages (point-to-point)
  - Broadcasting high-salience signals to all subscribers (Global Workspace)
  - Fan-out / fan-in for parallel retrieval aggregation
  - Pipeline chaining for sequential cognitive processing

Implementation
--------------
All I/O is non-blocking via AnyIO memory object streams.
Subscriptions are held in-process (no external broker required).
The bus is fully injectable and stateless between ``start``/``stop`` pairs,
making it straightforward to test in isolation.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import anyio.abc

from mnemon.core.models import CognitiveMessage, MessageType

if TYPE_CHECKING:
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EventHandler = Callable[[CognitiveMessage], Awaitable[None]]

# ---------------------------------------------------------------------------
# EventFilter
# ---------------------------------------------------------------------------


@dataclass
class EventFilter:
    """Declarative predicate for scoping a subscription to a subset of messages.

    All non-None fields must match for the filter to pass.
    A ``None`` field means "any value is acceptable".

    Brain analog: Dendritic filtering — a neuron only fires when the
    right combination of pre-synaptic inputs is present.
    """

    source_module: str | None = None
    message_type: MessageType | None = None
    min_priority: float | None = None
    custom_predicate: Callable[[CognitiveMessage], bool] | None = None

    def matches(self, msg: CognitiveMessage) -> bool:
        """Return True only if *msg* satisfies every non-None constraint."""
        if self.source_module is not None and msg.source != self.source_module:
            return False
        if self.message_type is not None and msg.type != self.message_type:
            return False
        if self.min_priority is not None and msg.priority < self.min_priority:
            return False
        return not (self.custom_predicate is not None and not self.custom_predicate(msg))


# ---------------------------------------------------------------------------
# Internal subscription record
# ---------------------------------------------------------------------------


@dataclass
class _Subscription:
    subscription_id: str
    module_id: str | None
    handler: EventHandler
    event_filter: EventFilter | None

    def should_handle(self, msg: CognitiveMessage) -> bool:
        """Return True if this subscription should receive *msg*."""
        # If the subscription is scoped to a module, only deliver to that module
        if self.module_id is not None and msg.target not in ("*", self.module_id):
            # Direct messages must match the module_id;
            # broadcast/wildcard messages go to everyone
            return False
        if self.event_filter is not None:
            return self.event_filter.matches(msg)
        return True


# ---------------------------------------------------------------------------
# CognitiveBus
# ---------------------------------------------------------------------------

_BROADCAST_TARGET = "*"
# Internal stream buffer capacity — keep large enough to avoid back-pressure
# under normal operation but bounded to surface runaway publishers.
_STREAM_BUFFER = 256


class CognitiveBus:
    """Central event bus for inter-module communication.

    Brain analog: Thalamus — active relay that gates and routes
    cortico-cortical communication between cognitive modules.

    Supports four routing modes:
    - Direct: Point-to-point (module A → module B)
    - Broadcast: Global Workspace style (→ all subscribers)
    - Fan-out/Fan-in: Parallel retrieval with aggregation
    - Pipeline: Sequential chain processing

    Implementation uses AnyIO memory object streams for
    zero-dependency async pub/sub.

    Thread / task safety
    --------------------
    ``subscribe`` and ``unsubscribe`` may be called from any coroutine.
    ``publish``, ``request``, and ``fan_out`` must be called from within an
    active AnyIO event loop (i.e. after ``start`` or during a test task group).
    """

    def __init__(self, stream_buffer_size: int = _STREAM_BUFFER) -> None:
        self._buffer_size = stream_buffer_size
        self._subscriptions: dict[str, _Subscription] = {}

        # Main inbound channel — all published messages land here first.
        self._send_stream: MemoryObjectSendStream[CognitiveMessage] | None = None
        self._recv_stream: MemoryObjectReceiveStream[CognitiveMessage] | None = None

        # Pending request-response pairs: correlation_id -> response send stream
        self._pending_responses: dict[
            str, MemoryObjectSendStream[CognitiveMessage]
        ] = {}

        self._running = False
        self._task_group: anyio.abc.TaskGroup | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, task_group: anyio.abc.TaskGroup) -> None:
        """Start the bus dispatch loop within *task_group*.

        Must be called before any ``publish`` / ``request`` / ``fan_out`` calls.

        Parameters
        ----------
        task_group:
            AnyIO TaskGroup that owns the dispatch coroutine's lifetime.
            When the group is cancelled the bus shuts down cleanly.
        """
        if self._running:
            logger.warning("CognitiveBus.start() called on an already-running bus; ignoring.")
            return

        self._send_stream, self._recv_stream = anyio.create_memory_object_stream(
            max_buffer_size=self._buffer_size
        )
        self._running = True
        self._task_group = task_group
        task_group.start_soon(self._dispatch_loop)
        logger.info("CognitiveBus started (buffer_size=%d).", self._buffer_size)

    async def stop(self) -> None:
        """Gracefully drain remaining messages and stop the dispatch loop."""
        if not self._running:
            return
        self._running = False
        if self._send_stream is not None:
            await self._send_stream.aclose()
        if self._recv_stream is not None:
            await self._recv_stream.aclose()
        logger.info("CognitiveBus stopped.")

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(
        self,
        handler: EventHandler,
        filter: EventFilter | None = None,  # noqa: A002 — mirrors the public API name
        module_id: str | None = None,
    ) -> str:
        """Register *handler* to receive messages that match *filter*.

        Parameters
        ----------
        handler:
            Async callable invoked with each matching CognitiveMessage.
        filter:
            Optional EventFilter; if ``None`` the handler receives every message
            whose target matches *module_id* (or all messages if that is also ``None``).
        module_id:
            Logical name of the subscribing module.  Direct messages are only
            delivered to subscriptions with a matching *module_id*.

        Returns
        -------
        str
            Opaque subscription ID that can be passed to ``unsubscribe``.
        """
        sub_id = str(uuid.uuid4())
        self._subscriptions[sub_id] = _Subscription(
            subscription_id=sub_id,
            module_id=module_id,
            handler=handler,
            event_filter=filter,
        )
        logger.debug(
            "Subscribed handler %s (module=%s, filter=%s) → %s",
            getattr(handler, "__qualname__", repr(handler)),
            module_id,
            filter,
            sub_id,
        )
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove the subscription identified by *subscription_id*.

        Silently ignores unknown IDs so callers need not guard against
        double-unsubscribe races.
        """
        removed = self._subscriptions.pop(subscription_id, None)
        if removed is not None:
            logger.debug("Unsubscribed %s.", subscription_id)
        else:
            logger.debug("unsubscribe: unknown subscription_id=%s (ignored).", subscription_id)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, message: CognitiveMessage) -> None:
        """Fire-and-forget publish.  Routes based on ``message.target``.

        Messages with ``target == "*"`` are broadcast to every subscriber.
        Messages with a specific target are delivered only to subscriptions
        whose ``module_id`` matches.

        Parameters
        ----------
        message:
            The CognitiveMessage to route.

        Raises
        ------
        RuntimeError
            If the bus has not been started via ``start()``.
        """
        if self._send_stream is None or not self._running:
            raise RuntimeError(
                "CognitiveBus.publish() called before start() or after stop()."
            )
        logger.debug(
            "Publishing %s/%s  %s → %s  priority=%.2f",
            message.type,
            message.id,
            message.source,
            message.target,
            message.priority,
        )
        await self._send_stream.send(message)

    # ------------------------------------------------------------------
    # Request / response
    # ------------------------------------------------------------------

    async def request(
        self,
        message: CognitiveMessage,
        timeout: float = 5.0,
    ) -> CognitiveMessage:
        """Publish *message* and await a correlated response.

        The response must be published back onto the bus by the handler with
        ``message.trace_id`` used as the correlation key in ``metadata["reply_to"]``.

        Parameters
        ----------
        message:
            The request message.  ``message.trace_id`` is used as the
            correlation key.
        timeout:
            Seconds to wait for a response before raising ``TimeoutError``.

        Returns
        -------
        CognitiveMessage
            The first response message correlated to this request.

        Raises
        ------
        TimeoutError
            If no response arrives within *timeout* seconds.
        """
        correlation_id = str(message.trace_id)
        resp_send, resp_recv = anyio.create_memory_object_stream[CognitiveMessage](
            max_buffer_size=1
        )
        self._pending_responses[correlation_id] = resp_send

        try:
            await self.publish(message)
            with anyio.fail_after(timeout):
                response = await resp_recv.receive()
            return response
        finally:
            self._pending_responses.pop(correlation_id, None)
            await resp_send.aclose()
            await resp_recv.aclose()

    # ------------------------------------------------------------------
    # Fan-out / fan-in
    # ------------------------------------------------------------------

    async def fan_out(
        self,
        message: CognitiveMessage,
        target_modules: list[str],
        timeout: float = 5.0,
    ) -> list[CognitiveMessage]:
        """Dispatch *message* to each module in *target_modules* concurrently.

        Collects all responses (or tolerates timeouts on individual modules)
        and returns them as a list.  Modules that do not respond within
        *timeout* are skipped; their slot will be absent from the results.

        Parameters
        ----------
        message:
            Template message.  A copy is created for each target with its
            ``target`` field overwritten.
        target_modules:
            List of module IDs to fan the message out to.
        timeout:
            Per-module response timeout in seconds.

        Returns
        -------
        list[CognitiveMessage]
            Responses received, in the order they arrived (not the order of
            *target_modules*).
        """
        responses: list[CognitiveMessage] = []

        async def _dispatch_one(target: str) -> None:
            targeted = message.model_copy(update={"target": target, "trace_id": uuid.uuid4()})
            try:
                resp = await self.request(targeted, timeout=timeout)
                responses.append(resp)
            except TimeoutError:
                logger.warning(
                    "fan_out: no response from module '%s' within %.1fs (trace_id=%s).",
                    target,
                    timeout,
                    message.trace_id,
                )

        async with anyio.create_task_group() as tg:
            for mod in target_modules:
                tg.start_soon(_dispatch_one, mod)

        logger.debug(
            "fan_out collected %d/%d responses (trace_id=%s).",
            len(responses),
            len(target_modules),
            message.trace_id,
        )
        return responses

    # ------------------------------------------------------------------
    # Internal dispatch loop
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Consume messages from the inbound stream and deliver to subscribers.

        Runs for the lifetime of the bus inside the task group provided to
        ``start()``.  Handlers that raise exceptions are logged but do not
        kill the dispatch loop.
        """
        if self._recv_stream is None:
            return

        logger.debug("CognitiveBus dispatch loop started.")
        try:
            async with self._recv_stream:
                async for message in self._recv_stream:
                    await self._route(message)
        except anyio.ClosedResourceError:
            # Normal shutdown path — the send stream was closed by stop().
            pass
        except Exception:
            logger.exception("CognitiveBus dispatch loop encountered an unexpected error.")
            raise
        finally:
            logger.debug("CognitiveBus dispatch loop exited.")

    async def _route(self, message: CognitiveMessage) -> None:
        """Deliver *message* to all matching subscribers and pending requests."""
        # First check if this is a reply to a pending request()
        reply_to = message.metadata.get("reply_to")
        if reply_to is not None:
            resp_stream = self._pending_responses.get(str(reply_to))
            if resp_stream is not None:
                try:
                    await resp_stream.send(message)
                    return  # reply consumed; don't also broadcast
                except anyio.ClosedResourceError:
                    logger.debug(
                        "_route: response stream for reply_to=%s already closed.", reply_to
                    )

        # Deliver to all matching subscriptions concurrently
        matching = [
            sub for sub in self._subscriptions.values() if sub.should_handle(message)
        ]

        if not matching:
            logger.debug(
                "_route: no subscribers for %s/%s (target=%s).",
                message.type,
                message.id,
                message.target,
            )
            return

        async def _call(sub: _Subscription) -> None:
            try:
                await sub.handler(message)
            except Exception:
                logger.exception(
                    "Handler %s raised an exception for message %s (type=%s).",
                    sub.subscription_id,
                    message.id,
                    message.type,
                )

        async with anyio.create_task_group() as tg:
            for sub in matching:
                tg.start_soon(_call, sub)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def subscription_count(self) -> int:
        """Return the number of active subscriptions."""
        return len(self._subscriptions)

    def is_running(self) -> bool:
        """Return True if the dispatch loop is active."""
        return self._running

    def __repr__(self) -> str:
        return (
            f"CognitiveBus(running={self._running}, "
            f"subscriptions={len(self._subscriptions)})"
        )
