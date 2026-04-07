"""
WorkingMemoryManager — active context management with token-budget constraints.

Brain analog: Dorsolateral prefrontal cortex (dlPFC) — the limited-capacity
workspace that maintains task-relevant information in a readily accessible,
active state.  The token budget models the biological capacity limit
(~7 ± 2 chunks) of Baddeley's working memory model.  LRU-importance eviction
mirrors the way the brain preferentially retains high-salience or goal-relevant
information when the workspace becomes overloaded.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from mnemon.core.config import WorkingMemoryConfig
from mnemon.core.exceptions import TokenBudgetExceededError
from mnemon.core.interfaces import LLMProvider, WorkingMemoryInterface
from mnemon.core.models import (
    ContextBlock,
    ContextSource,
    Episode,
    EvictionPolicy,
    Goal,
    RetrievedItem,
    WorkingMemoryState,
)

logger = logging.getLogger(__name__)


class WorkingMemoryManager(WorkingMemoryInterface):
    """Token-budget-constrained working memory with LRU-importance eviction.

    Injection respects eviction policy: when the budget is tight and the policy
    is :attr:`~mnemon.core.models.EvictionPolicy.LRU_IMPORTANCE`, the manager
    scores each evictable block and removes the lowest-scoring one first.
    Optionally, before eviction the block can be summarised via an
    :class:`~mnemon.core.interfaces.LLMProvider` and replaced with a
    shorter summary block so that the gist is preserved.
    """

    def __init__(
        self,
        config: WorkingMemoryConfig,
        llm: LLMProvider,
    ) -> None:
        self._config = config
        self._llm = llm
        self._state = WorkingMemoryState(token_budget=config.token_budget)
        # Insertion-order tracking for LRU: list of block ids, oldest first.
        self._insertion_order: list[str] = []
        logger.debug(
            "WorkingMemoryManager initialised — budget=%d policy=%s",
            config.token_budget,
            config.eviction_policy,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recency_rank(self, block_id: str) -> float:
        """Normalised recency rank in [0, 1]; 0 = oldest, 1 = most recent."""
        order = self._insertion_order
        if not order:
            return 0.0
        try:
            idx = order.index(block_id)
        except ValueError:
            return 0.0
        return idx / max(len(order) - 1, 1)

    def _eviction_priority(self, block: ContextBlock) -> float:
        """Lower score → evict first."""
        recency = self._recency_rank(str(block.id))
        return recency * 0.4 + block.importance * 0.6

    def _evictable_blocks(self) -> list[ContextBlock]:
        return [b for b in self._state.active_context if b.evictable]

    async def _summarise_block(self, block: ContextBlock) -> ContextBlock:
        """Generate an LLM summary of *block* and return a replacement summary block."""
        max_tokens = self._config.summarization.max_summary_tokens
        prompt = (
            f"Summarise the following context in at most {max_tokens} tokens. "
            f"Be concise but preserve key facts.\n\n{block.content}"
        )
        try:
            summary_text = await self._llm.generate(
                prompt, max_tokens=max_tokens
            )
        except Exception:
            logger.warning(
                "LLM summarisation failed for block %s; using truncated content",
                block.id,
                exc_info=True,
            )
            summary_text = block.content[:500]

        # Rough token estimate for the summary.
        summary_tokens = max(1, len(summary_text) // 4)
        return ContextBlock(
            content=summary_text,
            token_count=summary_tokens,
            source=ContextSource.SUMMARY,
            importance=block.importance,
            evictable=True,
            summary=block.content[:200],
        )

    def _remove_block(self, block: ContextBlock) -> None:
        """Remove *block* from active_context and update token_used."""
        self._state.active_context = [
            b for b in self._state.active_context if b.id != block.id
        ]
        self._state.token_used = max(0, self._state.token_used - block.token_count)
        try:
            self._insertion_order.remove(str(block.id))
        except ValueError:
            pass

    def _add_block(self, block: ContextBlock) -> None:
        """Append *block* to active_context and update accounting."""
        self._state.active_context.append(block)
        self._state.token_used += block.token_count
        self._insertion_order.append(str(block.id))

    # ------------------------------------------------------------------
    # WorkingMemoryInterface implementation
    # ------------------------------------------------------------------

    async def inject(self, block: ContextBlock) -> None:
        """Add *block* to the active workspace, evicting if necessary.

        Raises :class:`~mnemon.core.exceptions.TokenBudgetExceededError` if the
        block cannot fit even after exhausting all evictable slots.
        """
        budget = self._config.token_budget
        needed = block.token_count

        # Fast path: fits without eviction.
        if self._state.token_used + needed <= budget:
            self._add_block(block)
            logger.debug(
                "WorkingMemory injected block %s (%d tokens); used=%d/%d",
                block.id,
                needed,
                self._state.token_used,
                budget,
            )
            return

        # Eviction required.
        if self._config.eviction_policy != EvictionPolicy.LRU_IMPORTANCE:
            raise TokenBudgetExceededError(needed, budget, self._state.token_used)

        logger.debug(
            "WorkingMemory budget pressure — need %d, used %d/%d; starting eviction",
            needed,
            self._state.token_used,
            budget,
        )

        # Evict until the block fits or no evictable slots remain.
        while self._state.token_used + needed > budget:
            candidates = self._evictable_blocks()
            if not candidates:
                raise TokenBudgetExceededError(needed, budget, self._state.token_used)

            # Sort ascending by eviction priority; lowest score → evict first.
            candidates.sort(key=lambda b: self._eviction_priority(b))
            victim = candidates[0]

            if self._config.summarization.enabled:
                summary_block = await self._summarise_block(victim)
                self._remove_block(victim)
                # Only add summary if it actually saves tokens.
                if summary_block.token_count < victim.token_count:
                    self._add_block(summary_block)
                    logger.debug(
                        "Evicted block %s → summary block %s (%d→%d tokens)",
                        victim.id,
                        summary_block.id,
                        victim.token_count,
                        summary_block.token_count,
                    )
                else:
                    logger.debug(
                        "Evicted block %s (summary not smaller; discarded)",
                        victim.id,
                    )
            else:
                self._remove_block(victim)
                logger.debug("Evicted block %s (%d tokens)", victim.id, victim.token_count)

        self._add_block(block)
        logger.debug(
            "WorkingMemory injected block %s after eviction; used=%d/%d",
            block.id,
            self._state.token_used,
            budget,
        )

    async def inject_retrieved(self, items: list[RetrievedItem]) -> None:
        """Inject retrieved items in descending score order up to the token budget."""
        sorted_items = sorted(items, key=lambda i: i.score, reverse=True)
        for item in sorted_items:
            token_count = max(1, len(item.content) // 4)
            if self._state.token_used + token_count > self._config.token_budget:
                logger.debug(
                    "inject_retrieved: skipping item score=%.3f — budget exhausted",
                    item.score,
                )
                continue
            block = ContextBlock(
                content=item.content,
                token_count=token_count,
                source=ContextSource.RETRIEVAL,
                importance=item.score,
                evictable=True,
            )
            # Use direct path to avoid eviction loop for retrieved items.
            self._add_block(block)
        logger.debug(
            "inject_retrieved complete; used=%d/%d",
            self._state.token_used,
            self._config.token_budget,
        )

    def get_state(self) -> WorkingMemoryState:
        """Return a snapshot of the current working memory state."""
        return self._state.model_copy(deep=True)

    async def generate_cues(self) -> list[str]:
        """Extract retrieval cues from the last three active context blocks."""
        context = self._state.active_context
        recent = context[-3:] if len(context) >= 3 else context[:]
        cues = [block.content[:200] for block in recent]
        logger.debug("generate_cues produced %d cue(s)", len(cues))
        return cues

    def push_goal(self, goal: Goal) -> None:
        """Push *goal* onto the active goal stack."""
        self._state.goal_stack.append(goal)
        logger.debug("Goal pushed: %s (priority=%.2f)", goal.description, goal.priority)

    def pop_goal(self) -> Goal | None:
        """Pop and return the top goal, or None if the stack is empty."""
        if not self._state.goal_stack:
            return None
        goal = self._state.goal_stack.pop()
        logger.debug("Goal popped: %s", goal.description)
        return goal

    async def flush(self) -> Episode:
        """Serialise current state into an Episode and clear the workspace.

        The context is serialised as the concatenation of all active block
        contents.  Action and outcome default to empty strings since the
        working memory layer has no knowledge of what the agent did or
        observed — the caller is expected to enrich the episode afterwards.
        """
        _EXCLUDED = {ContextSource.RETRIEVAL, ContextSource.SUMMARY}
        context_text = "\n\n".join(
            b.content
            for b in self._state.active_context
            if b.source not in _EXCLUDED
        )
        episode = Episode(
            agent_id="mnemon",
            session_id=self._state.session_id,
            timestamp=datetime.now(timezone.utc),
            context=context_text or "(empty)",
            action="",
            outcome="",
        )

        # Reset state, preserving session identity.
        session_id = self._state.session_id
        self._state = WorkingMemoryState(
            session_id=session_id,
            token_budget=self._config.token_budget,
        )
        self._insertion_order.clear()
        logger.debug("WorkingMemory flushed; episode id=%s", episode.id)
        return episode

    def token_status(self) -> dict[str, int]:
        """Return token usage summary."""
        budget = self._config.token_budget
        used = self._state.token_used
        return {
            "used": used,
            "budget": budget,
            "available": max(0, budget - used),
        }
