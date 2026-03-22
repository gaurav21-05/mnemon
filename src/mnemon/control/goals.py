"""
GoalManager — hierarchical goal management and LLM-driven task decomposition.

Brain analog: Anterior prefrontal cortex (aPFC) — maintains branching,
nested sub-goal representations and manages the temporal integration required
to pursue long-horizon objectives across multiple cognitive cycles. The aPFC
is uniquely human in its capacity to hold pending intentions while pursuing
intermediate sub-goals, precisely mirroring the parent/child goal hierarchy
implemented here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final
from uuid import UUID

from mnemon.core.exceptions import GoalError
from mnemon.core.interfaces import GoalManagerInterface, LLMProvider
from mnemon.core.models import Goal, GoalStatus

logger: Final = logging.getLogger(__name__)

_DECOMPOSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "priority": {"type": "number"},
                    "success_criteria": {"type": "string"},
                },
                "required": ["description"],
            },
        }
    },
    "required": ["subgoals"],
}


class GoalManager(GoalManagerInterface):
    """Hierarchical goal manager implementing the anterior prefrontal cortex analog.

    Maintains a flat store of all goals keyed by UUID, with parent/child
    linkage and sequential dependency chaining for decomposed sub-goals.
    LLM-driven decomposition converts high-level intentions into ordered,
    actionable sub-goal trees.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._goals: dict[UUID, Goal] = {}
        logger.info("GoalManager initialised")

    async def create_goal(
        self,
        description: str,
        priority: float = 0.5,
    ) -> Goal:
        """Instantiate, persist, and return a new top-level Goal.

        Parameters
        ----------
        description:
            Natural-language description of the desired outcome.
        priority:
            Initial priority weight in [0, 1]; higher = more urgent.

        Returns
        -------
        Goal
            The newly created goal with an assigned UUID.
        """
        goal = Goal(description=description, priority=priority)
        self._goals[goal.id] = goal
        logger.info(
            "Created goal %s (priority=%.3f): %r",
            goal.id,
            priority,
            description[:120],
        )
        return goal

    async def decompose(self, goal: Goal) -> list[Goal]:
        """Break *goal* into an ordered list of sub-goals via LLM planning.

        Uses structured output to guarantee parseable JSON. Sub-goals are
        linked sequentially so that each depends on its predecessor, modelling
        the aPFC's role in maintaining temporal ordering of intentions.

        Parameters
        ----------
        goal:
            The parent goal to decompose.

        Returns
        -------
        list[Goal]
            Ordered sub-goals (index 0 should be executed first).
            Returns an empty list on LLM failure.
        """
        prompt = (
            f"Break this goal into 2-5 concrete, actionable sub-goals:\n"
            f"Goal: {goal.description}\n"
            f"Success criteria: {goal.success_criteria}\n\n"
            f"Return a JSON object with key \"subgoals\" containing a list of objects,\n"
            f"each with: \"description\", \"priority\" (0.0-1.0), \"success_criteria\".\n"
            f"Order them by execution dependency (first subgoal should be done first)."
        )

        try:
            raw: dict[str, Any] = await self._llm.generate_structured(
                prompt=prompt,
                response_schema=_DECOMPOSE_SCHEMA,
            )
        except Exception as exc:
            logger.warning(
                "LLM decomposition failed for goal %s: %s — returning empty list",
                goal.id,
                exc,
            )
            return []

        subgoal_defs: list[dict[str, Any]] = raw.get("subgoals", [])
        if not subgoal_defs:
            logger.warning("LLM returned no subgoals for goal %s", goal.id)
            return []

        subgoals: list[Goal] = []
        for item in subgoal_defs:
            try:
                description: str = str(item["description"])
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping malformed subgoal item %r: %s", item, exc)
                continue

            priority: float = float(item.get("priority", goal.priority))
            priority = max(0.0, min(1.0, priority))
            success_criteria: str = str(item.get("success_criteria", ""))

            subgoal = Goal(
                description=description,
                priority=priority,
                success_criteria=success_criteria,
                parent_goal_id=goal.id,
            )
            subgoals.append(subgoal)

        # Sequential dependency chain: each subgoal depends on the previous one.
        for idx in range(1, len(subgoals)):
            subgoals[idx].dependencies = [subgoals[idx - 1].id]

        # Register child IDs on the parent goal and persist everything.
        goal.subgoals = [sg.id for sg in subgoals]
        self._goals[goal.id] = goal

        for subgoal in subgoals:
            self._goals[subgoal.id] = subgoal

        logger.info(
            "Decomposed goal %s into %d sub-goals: %s",
            goal.id,
            len(subgoals),
            [str(sg.id) for sg in subgoals],
        )
        return subgoals

    async def update_status(self, goal_id: UUID, status: GoalStatus) -> None:
        """Transition *goal_id* to *status* with cascading parent updates.

        Completing the last sibling auto-completes the parent (conjunction
        semantics). Failing any sibling immediately fails the parent
        (fail-fast, mirroring the aPFC's abort signal on intention failure).

        Parameters
        ----------
        goal_id:
            UUID of the goal to update.
        status:
            New lifecycle status to assign.

        Raises
        ------
        GoalError
            If *goal_id* does not exist in the store.
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            raise GoalError(f"Goal {goal_id} not found")

        previous_status = goal.status
        goal.status = status
        self._goals[goal_id] = goal

        logger.info(
            "Goal %s status transition: %s -> %s",
            goal_id,
            previous_status,
            status,
        )

        if goal.parent_goal_id is None:
            return

        parent = self._goals.get(goal.parent_goal_id)
        if parent is None:
            logger.warning(
                "Parent goal %s not found for child %s — skipping cascade",
                goal.parent_goal_id,
                goal_id,
            )
            return

        if status == GoalStatus.FAILED:
            if parent.status not in (GoalStatus.COMPLETED, GoalStatus.FAILED):
                parent.status = GoalStatus.FAILED
                self._goals[parent.id] = parent
                logger.info(
                    "Parent goal %s marked FAILED due to child %s failure (fail-fast)",
                    parent.id,
                    goal_id,
                )
            return

        if status == GoalStatus.COMPLETED:
            sibling_ids = parent.subgoals
            # All siblings must be present AND completed for auto-completion.
            # Missing siblings block completion to prevent premature cascading.
            all_present = all(sid in self._goals for sid in sibling_ids)
            all_complete = all_present and all(
                self._goals[sid].status == GoalStatus.COMPLETED
                for sid in sibling_ids
            )
            if all_complete and sibling_ids:
                if parent.status not in (GoalStatus.COMPLETED, GoalStatus.FAILED):
                    parent.status = GoalStatus.COMPLETED
                    self._goals[parent.id] = parent
                    logger.info(
                        "Parent goal %s auto-completed — all %d sub-goals done",
                        parent.id,
                        len(sibling_ids),
                    )

    def get_active_goals(self) -> list[Goal]:
        """Return all goals with status ACTIVE, sorted by priority descending.

        Returns
        -------
        list[Goal]
            Active goals ordered from highest to lowest priority.
        """
        active = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]
        active.sort(key=lambda g: g.priority, reverse=True)
        return active
