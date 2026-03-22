"""
Orchestrator — the central executive of the Mnemon cognitive framework.

Brain analog
------------
Lateral prefrontal cortex — the central executive that integrates input
from all cognitive subsystems, selects and sequences cognitive operations,
and drives goal-directed behaviour across multiple cycles.  It does not
store knowledge itself; it coordinates the specialised modules that do.

The Orchestrator implements a 6-phase cognitive cycle:
  1. Perception   — sensory buffer → PerceptUnit
  2. Attention    — salience scoring, gate decision, WM injection
  3. Retrieval    — cue-driven fan-out across episodic, semantic, procedural
  4. Deliberation — assemble context + goal state for action selection
  5. Execution    — placeholder; downstream agent framework consumes context
  6. Learning     — flush WM → episode, compute RPE, encode, update valence,
                    meta-cognitive self-evaluation
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from mnemon.core.bus import CognitiveBus
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import (
    AttentionControllerInterface,
    EmbeddingProvider,
    EpisodicMemoryInterface,
    GoalManagerInterface,
    MetaCognitionInterface,
    OrchestratorInterface,
    ProceduralMemoryInterface,
    RewardProcessorInterface,
    SemanticMemoryInterface,
    SensoryBufferInterface,
    ValenceMemoryInterface,
    WorkingMemoryInterface,
)
from mnemon.core.models import (
    ContextBlock,
    ContextSource,
    GateDecision,
    Goal,
    GoalStatus,
    RetrievalQuery,
    RetrievedItem,
)

logger = logging.getLogger(__name__)


class Orchestrator(OrchestratorInterface):
    """Central executive: runs the full 6-phase cognitive cycle.

    Brain analog: Lateral prefrontal cortex.

    All cognitive modules are injected at construction time; the Orchestrator
    holds no long-term state beyond a cycle counter and optional bus reference.
    """

    def __init__(
        self,
        config: MnemonConfig,
        sensory: SensoryBufferInterface,
        working_memory: WorkingMemoryInterface,
        episodic: EpisodicMemoryInterface,
        semantic: SemanticMemoryInterface,
        procedural: ProceduralMemoryInterface,
        valence: ValenceMemoryInterface,
        attention: AttentionControllerInterface,
        goal_manager: GoalManagerInterface,
        meta_cognition: MetaCognitionInterface,
        reward_processor: RewardProcessorInterface,
        embedding_provider: EmbeddingProvider,
        bus: CognitiveBus | None = None,
    ) -> None:
        self._config = config
        self._sensory = sensory
        self._working_memory = working_memory
        self._episodic = episodic
        self._semantic = semantic
        self._procedural = procedural
        self._valence = valence
        self._attention = attention
        self._goal_manager = goal_manager
        self._meta = meta_cognition
        self._reward = reward_processor
        self._embedding_provider = embedding_provider
        self._bus = bus
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    # OrchestratorInterface implementation
    # ------------------------------------------------------------------

    async def run_cycle(self, raw_input: str | None = None) -> dict[str, Any]:
        """Execute one full 6-phase cognitive cycle.

        Parameters
        ----------
        raw_input:
            Optional new stimulus.  When provided it is processed by the
            sensory buffer; otherwise the latest buffered percept is used.

        Returns
        -------
        dict[str, Any]
            Cycle summary containing cycle_id, phases_completed, and
            meta_evaluation fields.
        """
        cycle_id = uuid4()
        self._cycle_count += 1
        phases_completed: list[str] = []
        percept = None
        retrieved_items: list[RetrievedItem] = []
        deliberation: dict[str, Any] = {}
        action_result: dict[str, Any] = {}
        meta_eval = None
        wm_state = None

        logger.info(
            "Cycle %d started (cycle_id=%s, has_input=%s).",
            self._cycle_count,
            cycle_id,
            raw_input is not None,
        )

        # ------------------------------------------------------------------
        # Phase 1: PERCEPTION
        # ------------------------------------------------------------------
        try:
            if raw_input:
                percept = await self._sensory.process(raw_input)
            else:
                buffered = self._sensory.peek()
                percept = buffered[-1] if buffered else None

            phases_completed.append("perception")
            logger.debug(
                "Cycle %d / phase 1 PERCEPTION: percept_id=%s",
                self._cycle_count,
                percept.id if percept else None,
            )
        except Exception:
            logger.exception("Cycle %d / phase 1 PERCEPTION failed.", self._cycle_count)

        # ------------------------------------------------------------------
        # Phase 2: ATTENTION
        # ------------------------------------------------------------------
        try:
            if percept is not None:
                active_goals = self._goal_manager.get_active_goals()
                salience = await self._attention.score(percept, active_goals)
                gate_decision = self._attention.gate(salience)

                # Adjust thresholds based on WM load
                token_status = self._working_memory.token_status()
                load = token_status["used"] / max(token_status["budget"], 1)
                self._attention.adjust_thresholds(load)

                logger.debug(
                    "Cycle %d / phase 2 ATTENTION: gate=%s salience=%.3f load=%.3f",
                    self._cycle_count,
                    gate_decision,
                    salience.combined,
                    load,
                )

                if gate_decision == GateDecision.BROADCAST:
                    block = ContextBlock(
                        content=percept.normalized,
                        token_count=percept.tokens,
                        source=ContextSource.USER_INPUT,
                        importance=salience.combined,
                    )
                    await self._working_memory.inject(block)
                elif gate_decision == GateDecision.QUEUE:
                    # Store for potential later use — logged for observability
                    logger.debug(
                        "Cycle %d / phase 2 ATTENTION: percept %s queued (not broadcast).",
                        self._cycle_count,
                        percept.id,
                    )
                # GateDecision.DISCARD: do nothing

            phases_completed.append("attention")
        except Exception:
            logger.exception("Cycle %d / phase 2 ATTENTION failed.", self._cycle_count)

        # ------------------------------------------------------------------
        # Phase 3: RETRIEVAL
        # ------------------------------------------------------------------
        try:
            cues = await self._working_memory.generate_cues()
            logger.debug(
                "Cycle %d / phase 3 RETRIEVAL: %d cue(s) generated.",
                self._cycle_count,
                len(cues),
            )

            episodic_items: list[RetrievedItem] = []
            semantic_items: list[RetrievedItem] = []
            procedural_items: list[RetrievedItem] = []

            for cue in cues:
                try:
                    cue_embedding = await self._embedding_provider.embed(cue)
                    query = RetrievalQuery(
                        query_text=cue,
                        query_embedding=cue_embedding,
                    )

                    # Episodic retrieval
                    try:
                        episodic_result = await self._episodic.retrieve(query)
                        episodic_items.extend(episodic_result.items)
                        logger.debug(
                            "Cycle %d / RETRIEVAL episodic: %d item(s) for cue '%s'.",
                            self._cycle_count,
                            len(episodic_result.items),
                            cue[:60],
                        )
                    except Exception:
                        logger.exception(
                            "Cycle %d / RETRIEVAL episodic failed for cue '%s'.",
                            self._cycle_count,
                            cue[:60],
                        )

                    # Semantic retrieval
                    try:
                        sem_triples = await self._semantic.retrieve_by_similarity(
                            cue_embedding, top_k=5
                        )
                        for triple in sem_triples:
                            obj_str = (
                                triple.object.name
                                if hasattr(triple.object, "name")
                                else str(triple.object)
                            )
                            semantic_items.append(
                                RetrievedItem(
                                    source_store="semantic",
                                    content=f"{triple.subject.name} {triple.predicate} {obj_str}",
                                    score=triple.confidence,
                                )
                            )
                        logger.debug(
                            "Cycle %d / RETRIEVAL semantic: %d triple(s) for cue '%s'.",
                            self._cycle_count,
                            len(sem_triples),
                            cue[:60],
                        )
                    except Exception:
                        logger.exception(
                            "Cycle %d / RETRIEVAL semantic failed for cue '%s'.",
                            self._cycle_count,
                            cue[:60],
                        )

                    # Procedural retrieval
                    try:
                        skills = await self._procedural.retrieve(cue_embedding, top_k=3)
                        for skill in skills:
                            procedural_items.append(
                                RetrievedItem(
                                    source_store="procedural",
                                    content=f"Skill: {skill.name} — {skill.description}",
                                    score=skill.utility,
                                )
                            )
                        logger.debug(
                            "Cycle %d / RETRIEVAL procedural: %d skill(s) for cue '%s'.",
                            self._cycle_count,
                            len(skills),
                            cue[:60],
                        )
                    except Exception:
                        logger.exception(
                            "Cycle %d / RETRIEVAL procedural failed for cue '%s'.",
                            self._cycle_count,
                            cue[:60],
                        )

                except Exception:
                    logger.exception(
                        "Cycle %d / RETRIEVAL embedding failed for cue '%s'.",
                        self._cycle_count,
                        cue[:60],
                    )

            # Fuse per-source rankings with Reciprocal Rank Fusion and inject top items
            retrieved_items = self._rrf_fuse(
                {
                    "episodic": episodic_items,
                    "semantic": semantic_items,
                    "procedural": procedural_items,
                }
            )
            await self._working_memory.inject_retrieved(retrieved_items[:10])

            phases_completed.append("retrieval")
            logger.debug(
                "Cycle %d / phase 3 RETRIEVAL complete: %d total item(s) retrieved.",
                self._cycle_count,
                len(retrieved_items),
            )
        except Exception:
            logger.exception("Cycle %d / phase 3 RETRIEVAL failed.", self._cycle_count)

        # ------------------------------------------------------------------
        # Phase 4: DELIBERATION
        # ------------------------------------------------------------------
        try:
            wm_state = self._working_memory.get_state()

            context_parts: list[str] = []
            for block in wm_state.active_context:
                context_parts.append(block.content)
            for item in wm_state.retrieved_items:
                context_parts.append(f"[{item.source_store}] {item.content}")

            active_goals = self._goal_manager.get_active_goals()
            goal_text = (
                "; ".join(g.description for g in active_goals)
                if active_goals
                else "No specific goal"
            )

            deliberation = {
                "context": "\n".join(context_parts),
                "goal": goal_text,
                "retrieved_count": len(retrieved_items),
            }

            phases_completed.append("deliberation")
            logger.debug(
                "Cycle %d / phase 4 DELIBERATION: context_blocks=%d retrieved_items=%d",
                self._cycle_count,
                len(wm_state.active_context),
                len(wm_state.retrieved_items),
            )
        except Exception:
            logger.exception("Cycle %d / phase 4 DELIBERATION failed.", self._cycle_count)

        # ------------------------------------------------------------------
        # Phase 5: EXECUTION
        # ------------------------------------------------------------------
        try:
            # Placeholder: actual action execution depends on the agent framework.
            # The orchestrator prepares the augmented context; execution is delegated.
            action_result = {
                "action": "deliberation_complete",
                "context_tokens": wm_state.token_used if wm_state is not None else 0,
            }

            phases_completed.append("execution")
            logger.debug(
                "Cycle %d / phase 5 EXECUTION: action=%s context_tokens=%d",
                self._cycle_count,
                action_result["action"],
                action_result["context_tokens"],
            )
        except Exception:
            logger.exception("Cycle %d / phase 5 EXECUTION failed.", self._cycle_count)

        # ------------------------------------------------------------------
        # Phase 6: LEARNING
        # ------------------------------------------------------------------
        try:
            # Flush working memory to create an episode
            episode = await self._working_memory.flush()

            # Compute reward (simple heuristic: use episode importance as proxy)
            predicted_value = 0.5
            actual_reward = episode.importance
            reward_signal = await self._reward.compute_rpe(
                episode_id=episode.id,
                predicted_value=predicted_value,
                actual_reward=actual_reward,
            )

            # Update episode with reward signal
            episode_with_reward = episode.model_copy(
                update={
                    "reward_signal": reward_signal.rpe,
                    "emotional_valence": max(-1.0, min(1.0, reward_signal.rpe)),
                }
            )

            # Encode episode to episodic memory
            await self._episodic.encode(episode_with_reward)

            # Update valence associations
            if percept is not None:
                entities = [e.canonical_name for e in percept.entities]
                if entities:
                    await self._valence.update(entities, reward_signal.rpe)

            # Meta-cognitive self-evaluation
            meta_eval = await self._meta.evaluate_cycle(episode_with_reward, reward_signal.rpe)

            phases_completed.append("learning")
            logger.debug(
                "Cycle %d / phase 6 LEARNING: rpe=%.4f meta_confidence=%.3f",
                self._cycle_count,
                reward_signal.rpe,
                meta_eval.confidence,
            )
        except Exception:
            logger.exception("Cycle %d / phase 6 LEARNING failed.", self._cycle_count)

        logger.info(
            "Cycle %d completed (cycle_id=%s, phases=%s).",
            self._cycle_count,
            cycle_id,
            phases_completed,
        )

        return {
            "cycle_id": str(cycle_id),
            "cycle_number": self._cycle_count,
            "phases_completed": phases_completed,
            "percept_id": str(percept.id) if percept else None,
            "retrieved_count": len(retrieved_items),
            "deliberation": deliberation,
            "action_result": action_result,
            "meta_evaluation": (
                {
                    "confidence": meta_eval.confidence,
                    "prediction_error": meta_eval.prediction_error,
                    "strategy_recommended": meta_eval.strategy_recommended,
                    "lessons": meta_eval.lessons,
                }
                if meta_eval is not None
                else None
            ),
        }

    async def run_until_complete(
        self,
        goal: Goal,
        max_cycles: int = 10,
    ) -> dict[str, Any]:
        """Run consecutive cognitive cycles until *goal* reaches a terminal state.

        Parameters
        ----------
        goal:
            The goal whose status determines the stopping condition.
        max_cycles:
            Hard upper bound on cycle count to prevent infinite loops.

        Returns
        -------
        dict[str, Any]
            Summary containing cycle_count, success, and the final goal state.
        """
        logger.info(
            "run_until_complete: goal_id=%s description='%s' max_cycles=%d",
            goal.id,
            goal.description,
            max_cycles,
        )

        if max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {max_cycles}")

        # Push the goal into working memory so it biases retrieval and deliberation
        self._working_memory.push_goal(goal)

        cycle_results: list[dict[str, Any]] = []
        cycles_run = 0
        current_status: GoalStatus = goal.status

        for _ in range(max_cycles):
            result = await self.run_cycle()
            cycle_results.append(result)
            cycles_run += 1

            # Re-fetch goal status via the goal manager's active list
            active_goals = self._goal_manager.get_active_goals()
            matching = [g for g in active_goals if g.id == goal.id]

            # If the goal is no longer in the active list it was completed or failed
            if matching:
                current_status = matching[0].status
            else:
                # Goal removed from active list — check original object for mutation
                current_status = goal.status

            logger.debug(
                "run_until_complete: cycle=%d goal_status=%s",
                cycles_run,
                current_status,
            )

            if current_status in (GoalStatus.COMPLETED, GoalStatus.FAILED):
                break

        final_status = current_status
        success = final_status == GoalStatus.COMPLETED

        logger.info(
            "run_until_complete finished: cycles=%d success=%s goal_id=%s",
            cycles_run,
            success,
            goal.id,
        )

        return {
            "goal_id": str(goal.id),
            "goal_description": goal.description,
            "cycle_count": cycles_run,
            "max_cycles": max_cycles,
            "success": success,
            "final_goal_status": str(final_status),
            "cycle_results": cycle_results,
        }

    @staticmethod
    def _rrf_fuse(
        source_rankings: dict[str, list[RetrievedItem]],
        k: int = 60,
    ) -> list[RetrievedItem]:
        """Fuse ranked lists from multiple retrieval sources using Reciprocal Rank Fusion.

        For each unique item (keyed by content string), the RRF score is the sum
        of 1/(k + rank_i) across all sources in which the item appears, where
        rank_i is 1-based.  Items that surface in multiple sources receive
        additive boosts, making cross-source agreement a first-class signal.

        Parameters
        ----------
        source_rankings:
            Mapping of source store name to its ranked list of RetrievedItems.
            Items within each list are assumed to be ordered best-first.
        k:
            Smoothing constant that dampens the impact of top-ranked items.
            Defaults to 60 (standard RRF literature value).

        Returns
        -------
        list[RetrievedItem]
            Deduplicated items sorted by descending RRF score.  Each item's
            ``.score`` field is overwritten with its computed RRF score.
        """
        rrf_scores: dict[str, float] = {}
        # Keep one RetrievedItem representative per unique content string
        canonical: dict[str, RetrievedItem] = {}

        for _source, items in source_rankings.items():
            for rank, item in enumerate(items, start=1):
                key = item.content
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                if key not in canonical:
                    canonical[key] = item

        fused: list[RetrievedItem] = []
        for key, score in rrf_scores.items():
            fused.append(canonical[key].model_copy(update={"score": score}))

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the orchestrator's current internal state.

        Returns
        -------
        dict[str, Any]
            Includes token_status, active_goals, cycle_count, and bus status.
        """
        token_status = self._working_memory.token_status()
        active_goals = self._goal_manager.get_active_goals()

        return {
            "cycle_count": self._cycle_count,
            "working_memory": {
                "token_used": token_status["used"],
                "token_budget": token_status["budget"],
                "token_available": token_status["available"],
            },
            "active_goals": [
                {
                    "id": str(g.id),
                    "description": g.description,
                    "status": str(g.status),
                    "priority": g.priority,
                    "progress": g.progress,
                }
                for g in active_goals
            ],
            "bus": {
                "running": self._bus.is_running() if self._bus is not None else False,
                "subscriptions": (
                    self._bus.subscription_count() if self._bus is not None else 0
                ),
            },
        }
