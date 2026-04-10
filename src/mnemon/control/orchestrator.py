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
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

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

if TYPE_CHECKING:
    from mnemon.core.bus import CognitiveBus
    from mnemon.core.config import MnemonConfig

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
        self._last_episode_id: UUID | None = None
        self._suppress_next_episode_storage = False
        self._next_episode_redactions: list[str] = []
        self._last_episode_redactions: list[str] = []

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
        self._last_episode_id = None

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

                # Always inject explicit user input — user messages must never be
                # silently dropped by the attention gate, as they form the core
                # episodic memory of conversations.
                if raw_input or gate_decision == GateDecision.BROADCAST:
                    block = ContextBlock(
                        content=percept.normalized,
                        token_count=percept.tokens,
                        source=ContextSource.USER_INPUT,
                        importance=max(salience.combined, 0.5),  # floor for user input
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
                "citation_ids": [
                    str(item.metadata.get("episode_id") or item.metadata.get("triple_id") or "")
                    for item in retrieved_items
                    if item.metadata.get("episode_id") or item.metadata.get("triple_id")
                ],
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

            # Store raw_input as action if episode action is empty
            # This ensures user messages are captured even when execution is a placeholder
            if raw_input and not episode.action:
                episode = episode.model_copy(update={"action": raw_input})

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
                    "entities": percept.entities if percept is not None else [],
                    "tags": [
                        entity.canonical_name.lower()
                        for entity in (percept.entities if percept is not None else [])
                    ],
                }
            )

            # Meta-cognitive self-evaluation
            meta_eval = await self._meta.evaluate_cycle(episode_with_reward, reward_signal.rpe)

            if meta_eval.lessons:
                episode_with_reward = episode_with_reward.model_copy(
                    update={"reflection": " ".join(meta_eval.lessons)}
                )

            if self._suppress_next_episode_storage:
                logger.debug(
                    "Cycle %d / phase 6 LEARNING: suppressed episode storage for privacy.",
                    self._cycle_count,
                )
                self._last_episode_id = None
                self._suppress_next_episode_storage = False
                self._next_episode_redactions = []
                self._last_episode_redactions = []
            else:
                redactions = list(self._next_episode_redactions)
                if redactions:
                    episode_with_reward = self._redact_episode(episode_with_reward, redactions)
                self._next_episode_redactions = []
                self._last_episode_redactions = redactions
                # Encode episode to episodic memory
                await self._episodic.encode(episode_with_reward)
                self._last_episode_id = episode_with_reward.id

                # Update valence associations
                if percept is not None:
                    entities = [e.canonical_name for e in percept.entities]
                    if entities:
                        await self._valence.update(entities, reward_signal.rpe)

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

    async def update_last_episode_outcome(self, outcome: str) -> None:
        """Patch the outcome field of the most recently encoded episode.

        Called by the IPC layer after the LLM reply is generated, so that
        the stored episode reflects what Jarvis actually said.
        """
        if self._last_episode_id is None:
            return
        episode = await self._episodic.get(self._last_episode_id)
        if episode is None:
            return
        outcome = self._redact_text(outcome, self._last_episode_redactions)
        await self._episodic.update(episode.id, outcome=outcome)
        logger.debug(
            "Updated outcome for episode %s (len=%d)",
            self._last_episode_id,
            len(outcome),
        )

    async def update_last_episode_metadata(self, **updates: Any) -> None:
        """Patch tags / importance / reflection on the latest stored episode."""
        if self._last_episode_id is None:
            return
        episode = await self._episodic.get(self._last_episode_id)
        if episode is None:
            return

        merged_updates = dict(updates)
        if "tags" in merged_updates:
            incoming_tags = [str(tag) for tag in merged_updates["tags"]]
            merged_updates["tags"] = list(dict.fromkeys([*episode.tags, *incoming_tags]))
        if "importance" in merged_updates:
            merged_updates["importance"] = max(0.0, min(1.0, float(merged_updates["importance"])))

        await self._episodic.update(episode.id, **merged_updates)
        logger.debug(
            "Updated metadata for episode %s fields=%s",
            self._last_episode_id,
            sorted(merged_updates.keys()),
        )

    async def record_retrieval_feedback(
        self,
        memory_ids: list[str],
        helpful: bool = True,
    ) -> None:
        """Record that retrieved memories were actually used in the cycle."""
        for raw_id in memory_ids:
            try:
                episode_id = UUID(str(raw_id))
            except ValueError:
                continue
            episode = await self._episodic.get(episode_id)
            if episode is None:
                continue
            updates: dict[str, Any] = {
                "retrieval_uses": episode.retrieval_uses + 1,
                "retrieval_last_used_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ),
                "base_strength": min(episode.base_strength + 0.08, 3.0),
            }
            if helpful:
                updates["retrieval_help_count"] = episode.retrieval_help_count + 1
                updates["importance"] = min(1.0, episode.importance + 0.02)
                updates["decay_lambda"] = max(0.0001, episode.decay_lambda * 0.97)
            await self._episodic.update(episode_id, **updates)

    def suppress_next_episode_storage(self) -> None:
        """Skip persistence for the next cycle's flushed episode."""
        self._suppress_next_episode_storage = True

    def configure_next_episode_redactions(self, redactions: list[str]) -> None:
        """Apply literal phrase redactions to the next stored episode."""
        self._next_episode_redactions = [item for item in redactions if item.strip()]

    @staticmethod
    def _redact_text(text: str | None, redactions: list[str]) -> str | None:
        if text is None:
            return None
        redacted = text
        for phrase in redactions:
            redacted = redacted.replace(phrase, "[REDACTED]")
        return redacted

    def _redact_episode(self, episode: Any, redactions: list[str]) -> Any:
        """Redact literal phrases from the episode before persistence."""
        return episode.model_copy(
            update={
                "context": self._redact_text(episode.context, redactions),
                "action": self._redact_text(episode.action, redactions),
                "outcome": self._redact_text(episode.outcome, redactions),
                "reflection": self._redact_text(episode.reflection, redactions),
            }
        )

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
            current_status = matching[0].status if matching else goal.status

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
                "subscriptions": (self._bus.subscription_count() if self._bus is not None else 0),
            },
        }
