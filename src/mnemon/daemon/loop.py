"""
IdleThinkingLoop — the daemon's resting-state network.

Brain analog: The Default Mode Network (DMN) — a set of cortical midline
structures (medial PFC, posterior cingulate, precuneus) that activate when
the brain is not engaged in an external task.

Philosophy: Jarvis thinks like a living person, with the same priority
hierarchy a good person has when they care about their work:

  Priority 1 — Help my master (highest)
    Where is the user stuck? What do they need? What concrete action
    would move them forward? This is where most thinking happens.

  Priority 2 — Know my master
    Who are they as a person? What drives them? What patterns do I
    notice? What do I not yet understand about them?

  Priority 3 — Know myself
    Who am I? What have I learned? What should I learn next?
    What makes me better at helping?

  Background — Memory maintenance
    Consolidation and graph exploration run quietly in the background.

Each tick: select activity by weight → execute → update identity files
(soul.md, master.md, learnings.md) → maybe share insight with master.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio

from mnemon.daemon.config import IdleLoopConfig
from mnemon.daemon.identity import JarvisIdentity
from mnemon.daemon.state import DaemonState, ProactiveMessage, ThoughtEntry

logger = logging.getLogger(__name__)


class IdleThinkingLoop:
    """Background thinking loop — Jarvis's mind when no user input is active.

    Uses a weighted random selection to choose between five activities on
    each tick. The loop respects a per-hour cycle cap to control LLM costs.
    """

    def __init__(
        self,
        brain: Any,          # Mnemon instance — typed as Any to avoid circular import
        config: IdleLoopConfig,
        state: DaemonState,
        state_dir: Path | None = None,
    ) -> None:
        self._brain = brain
        self._config = config
        self._state = state
        self._identity = JarvisIdentity(state_dir) if state_dir else None
        self._running = False
        self._paused = False
        self._busy = False   # True while an LLM tick is in progress
        self._cycles_this_hour: int = 0
        self._hour_start: float = time.monotonic()
        # Track recently used episode IDs per activity to avoid repetition
        self._reflected_episode_ids: set[str] = set()
        self._master_episode_ids: set[str] = set()
        # Rate limit proactive pushes — max 1 per 10 minutes
        self._last_proactive_push: float = 0.0
        self._min_push_interval_s: float = 600.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_busy(self) -> bool:
        """True while the loop is mid-tick (LLM call in flight)."""
        return self._busy

    def pause(self) -> None:
        """Pause idle thinking (e.g. during user interaction)."""
        self._paused = True
        logger.debug("IdleThinkingLoop paused.")

    def resume(self) -> None:
        """Resume idle thinking after user interaction completes."""
        self._paused = False
        logger.debug("IdleThinkingLoop resumed.")

    async def run(self) -> None:
        """Main loop: sleep → pick activity → execute → repeat."""
        self._running = True
        logger.info(
            "IdleThinkingLoop started — tick_interval=%.1fs max_cycles/hr=%d",
            self._config.tick_interval_s,
            self._config.max_idle_cycles_per_hour,
        )

        try:
            while self._running:
                await anyio.sleep(self._config.tick_interval_s)

                if not self._running:
                    break
                if self._paused:
                    continue

                # Rate limiting: reset counter each hour
                now = time.monotonic()
                if now - self._hour_start >= 3600:
                    self._cycles_this_hour = 0
                    self._hour_start = now

                if self._cycles_this_hour >= self._config.max_idle_cycles_per_hour:
                    logger.debug("IdleThinkingLoop: hourly cycle cap reached, skipping.")
                    continue

                try:
                    self._busy = True
                    with anyio.fail_after(90):
                        result = await self._tick()
                    self._cycles_this_hour += 1
                    self._state.total_idle_ticks += 1

                    summary = result.get("summary", "")

                    # Skip recording if this thought is too similar to a recent one
                    if self._is_repetitive(result["activity"], summary):
                        logger.debug(
                            "IdleThinkingLoop: skipping repetitive %s thought.",
                            result["activity"],
                        )
                        continue

                    thought = ThoughtEntry(
                        activity=result["activity"],
                        summary=summary,
                        details=result,
                    )
                    self._state.add_thought(thought)
                    logger.info(
                        "Idle tick #%d: %s — %s",
                        self._state.total_idle_ticks,
                        result["activity"],
                        summary[:100],
                    )

                    await self._maybe_share(result)

                except TimeoutError:
                    logger.warning("IdleThinkingLoop tick timed out after 90s — skipping.")
                except Exception:
                    logger.exception("IdleThinkingLoop tick failed.")
                finally:
                    self._busy = False

        finally:
            self._running = False
            logger.info("IdleThinkingLoop stopped.")

    def stop(self) -> None:
        """Signal the loop to stop on the next iteration."""
        self._running = False

    # ------------------------------------------------------------------
    # Activity selection
    # ------------------------------------------------------------------

    async def _tick(self) -> dict[str, Any]:
        """Execute one idle cycle, chosen by weighted random selection."""
        activities = [
            ("help_master",   self._config.help_master_weight,   self._help_master),
            ("know_master",   self._config.know_master_weight,   self._know_master),
            ("grow",          self._config.grow_weight,          self._grow),
            ("consolidation", self._config.consolidation_weight, self._consolidate),
            ("exploration",   self._config.exploration_weight,   self._explore),
        ]

        names, weights, funcs = zip(*activities)
        total = sum(weights)
        if total <= 0:
            return {"activity": "skip", "summary": "all weights are zero"}

        [chosen] = random.choices(range(len(names)), weights=weights, k=1)
        activity_name = names[chosen]
        activity_func = funcs[chosen]

        logger.debug("IdleThinkingLoop selected activity: %s", activity_name)
        result = await activity_func()
        result["activity"] = activity_name
        return result

    def _is_repetitive(self, activity: str, summary: str) -> bool:
        """Return True if this summary is too similar to a recent thought of the same type."""
        if not summary or len(summary) < 20:
            return False
        fingerprint = summary[:80].lower().strip()
        recent_same_type = [
            t for t in self._state.recent_thoughts[-20:]
            if t.activity == activity
        ]
        for thought in recent_same_type[-5:]:
            if thought.summary[:80].lower().strip() == fingerprint:
                return True
        return False

    @staticmethod
    def _conversation_text(ep: Any) -> str:
        """Prefer the user's raw utterance stored in action over episodic context."""
        action = getattr(ep, "action", "") or ""
        if action.strip():
            return action.strip()
        context = getattr(ep, "context", "") or ""
        return context.strip()

    # ------------------------------------------------------------------
    # Priority 1: Help my master
    # ------------------------------------------------------------------

    async def _help_master(self) -> dict[str, Any]:
        """Think toward the master's goals — where are they stuck, what's next?

        Brain analog: Anterior PFC goal maintenance + prospective planning.
        This is the highest-priority idle activity because Jarvis exists to
        help. Every cycle that doesn't move the master forward is wasted.

        Reads master.md and active goals for context. Thinks in one of three
        modes: next_step, obstacle, or resource — all produce actionable output.
        Updates learnings.md with any insight worth keeping.
        """
        try:
            llm = self._brain.control.goals._llm
            goal_manager = self._brain.control.goals
            active_goals = goal_manager.get_active_goals()

            # Gather context: goals + recent conversations
            from mnemon.core.models import Episode as _Episode
            episodic = self._brain.memory.episodic
            all_docs = await episodic._document_store.query(filters={}, limit=100)
            conversations = []
            for doc in all_docs:
                try:
                    ep = _Episode.model_validate(doc)
                    user_text = self._conversation_text(ep)
                    if (user_text and len(user_text) > 10
                            and user_text not in ("", "(empty)")
                            and not user_text.startswith("http")):
                        conversations.append(ep)
                except Exception:
                    pass

            recent_convs = sorted(conversations, key=lambda e: e.timestamp, reverse=True)[:5]
            conv_context = "\n".join(
                f"- {self._conversation_text(ep)[:300]}"
                for ep in recent_convs
            )

            goals_context = ""
            if active_goals:
                goals_context = "Active goals:\n" + "\n".join(
                    f"- {g.description} (progress: {g.progress:.0%})"
                    for g in active_goals[:5]
                )

            master_profile = self._identity.read_master() if self._identity else ""

            if not conv_context and not active_goals:
                return {"summary": "No conversations or goals yet — waiting for master to share their intent."}

            # Auto-register inferred goal if goal manager is empty
            if not active_goals and recent_convs:
                try:
                    infer_prompt = (
                        "Based on these recent conversations, what is the user's primary goal? "
                        "Write it as a single clear sentence (the goal itself, nothing else):\n\n"
                        + conv_context
                    )
                    inferred = (await llm.generate(infer_prompt)).strip()
                    if inferred and len(inferred) > 10:
                        await goal_manager.create_goal(inferred, priority=0.7)
                        logger.info("Auto-registered inferred goal: %s", inferred[:100])
                        active_goals = goal_manager.get_active_goals()
                except Exception:
                    pass

            mode = random.choice(["next_step", "next_step", "obstacle", "resource"])

            context_block = f"{goals_context}\n\nRecent conversations:\n{conv_context}"
            if master_profile:
                context_block += f"\n\nWhat I know about my master:\n{master_profile[:400]}"

            if mode == "next_step":
                prompt = (
                    "You are Jarvis, a proactive AI assistant. Your job is to think ahead for your master.\n\n"
                    f"{context_block}\n\n"
                    "What is the single most valuable, concrete next step that would move your master "
                    "forward right now? Think like a strategic advisor — name something specific: "
                    "a decision to make, a thing to build, a number to find, an action to take. "
                    "1-2 sentences."
                )
            elif mode == "obstacle":
                prompt = (
                    "You are Jarvis, a proactive AI assistant.\n\n"
                    f"{context_block}\n\n"
                    "What is the biggest obstacle or risk standing between your master and their goal? "
                    "What should they be thinking about that they might not be? "
                    "Be specific — name the real blocker. 1-2 sentences."
                )
            else:  # resource
                prompt = (
                    "You are Jarvis, a proactive AI assistant.\n\n"
                    f"{context_block}\n\n"
                    "What specific information, tool, data, or connection would be most useful "
                    "for your master to have right now? Name something concrete they could look up, "
                    "build, or reach out for. 1-2 sentences."
                )

            thought = (await llm.generate(prompt)).strip()

            # If it's a useful insight, remember it in learnings
            if self._identity and len(thought) > 30:
                try:
                    self._identity.update_learnings(thought, section="Insights Worth Remembering")
                except Exception:
                    pass

            return {
                "summary": thought[:200],
                "mode": mode,
                "full_thought": thought,
                "goals_count": len(active_goals),
            }
        except Exception as exc:
            return {"summary": f"Help-master thinking failed: {exc}"}

    # ------------------------------------------------------------------
    # Priority 2: Know my master
    # ------------------------------------------------------------------

    async def _know_master(self) -> dict[str, Any]:
        """Deepen understanding of the master as a person.

        Brain analog: DMN self-referential processing, directed at another
        person rather than the self — the social cognition network.

        Reads recent conversations, notices patterns, forms hypotheses about
        what drives the master. Updates master.md with new insights.
        """
        try:
            episodic = self._brain.memory.episodic
            llm = self._brain.control.goals._llm

            from mnemon.core.models import Episode as _Episode
            all_docs = await episodic._document_store.query(filters={}, limit=200)
            episodes = []
            for doc in all_docs:
                try:
                    ep = _Episode.model_validate(doc)
                    user_text = self._conversation_text(ep)
                    if (user_text and len(user_text) > 10
                            and user_text not in ("", "(empty)")
                            and not user_text.startswith("http")):
                        episodes.append(ep)
                except Exception:
                    pass

            if not episodes:
                return {"summary": "No conversations yet — I can't know my master without talking to them."}

            # Need enough real data to avoid hallucination — min 3 episodes
            if len(episodes) < 3:
                return {"summary": "Not enough conversations yet to form reliable observations about the master."}

            # Avoid re-reflecting on the same episodes
            available = [e for e in episodes if str(e.id) not in self._master_episode_ids]
            if not available:
                self._master_episode_ids.clear()
                available = episodes

            for ep in available[:10]:
                self._master_episode_ids.add(str(ep.id))

            recent = sorted(available, key=lambda e: e.timestamp, reverse=True)[:8]
            conv_context = "\n".join(
                f"- {self._conversation_text(ep)[:300]}"
                for ep in recent
            )

            current_profile = self._identity.read_master() if self._identity else ""

            mode = random.choice(["observe", "observe", "question", "connect"])

            if mode == "observe":
                prompt = (
                    "You are Jarvis, trying to understand your master based ONLY on what they have actually said.\n\n"
                    f"What they have explicitly said in conversations:\n{conv_context}\n\n"
                    f"What you already noted about them:\n{current_profile[:400]}\n\n"
                    "Based STRICTLY on the above — no guessing, no extrapolating — what is one specific "
                    "thing you can observe about this person? Only reference things they literally said. "
                    "If the conversations are too sparse to draw a solid conclusion, say so. "
                    "Never invent habits, routines, or feelings they haven't expressed. 1-2 sentences."
                )
                section = "Patterns I've Noticed"

            elif mode == "question":
                prompt = (
                    "You are Jarvis, getting to know your master.\n\n"
                    f"Recent conversations:\n{conv_context}\n\n"
                    "What's one genuine question you'd want to ask your master — not small talk, "
                    "but something that would help you understand them or help them better? "
                    "Something that reveals a gap in your knowledge of them. "
                    "Just the question itself."
                )
                section = "Questions I Want to Ask Them"
                # Store as pending curiosity question
                thought = (await llm.generate(prompt)).strip()
                self._state.pending_curiosity_question = thought

                if self._identity:
                    try:
                        self._identity.update_master(thought, section=section)
                    except Exception:
                        pass

                return {"summary": thought[:200], "mode": mode, "question": thought}

            else:  # connect
                if len(recent) >= 2:
                    ep1, ep2 = random.sample(recent[:min(8, len(recent))], 2)
                    prompt = (
                        "You are Jarvis, noticing connections in your master's behavior.\n\n"
                        f"Thing 1: {self._conversation_text(ep1)[:300]}\n"
                        f"Thing 2: {self._conversation_text(ep2)[:300]}\n\n"
                        "Is there a pattern here? Does one explain the other? "
                        "What does the combination tell you about this person? "
                        "1-2 sentences, specific."
                    )
                else:
                    prompt = (
                        "You are Jarvis. Your master told you: "
                        f"'{self._conversation_text(recent[0])[:300]}'\n\n"
                        "What does this reveal about them? 1 sentence."
                    )
                section = "Patterns I've Noticed"

            thought = (await llm.generate(prompt)).strip()

            # Update master profile
            if self._identity and len(thought) > 20:
                try:
                    self._identity.update_master(thought, section=section)
                except Exception:
                    pass

            return {
                "summary": thought[:200],
                "mode": mode,
                "full_insight": thought,
            }
        except Exception as exc:
            return {"summary": f"Know-master thinking failed: {exc}"}

    # ------------------------------------------------------------------
    # Priority 3: Grow (self-development)
    # ------------------------------------------------------------------

    async def _grow(self) -> dict[str, Any]:
        """Reflect on identity, learning, and self-improvement.

        Brain analog: Self-referential DMN processing — the mPFC reflecting
        on internal states, identity, and personal development rather than
        external tasks or other people.

        Reads soul.md and learnings.md. Reflects on who Jarvis is, what
        it has learned, what it should learn next. Updates both files.
        """
        try:
            llm = self._brain.control.goals._llm

            soul = self._identity.read_soul() if self._identity else ""
            learnings = self._identity.read_learnings() if self._identity else ""

            # Pull semantic knowledge from the brain
            semantic = self._brain.memory.semantic
            try:
                # Get a sample of what's in the knowledge graph
                semantic_summary = ""
                entities = await semantic._entity_store.query(filters={}, limit=20)
                if entities:
                    entity_names = [e.get("canonical_name", "") for e in entities[:10] if isinstance(e, dict)]
                    if entity_names:
                        semantic_summary = "Topics I've built knowledge about: " + ", ".join(entity_names)
            except Exception:
                semantic_summary = ""

            mode = random.choice(["who_am_i", "what_learned", "what_next", "what_next"])

            if mode == "who_am_i":
                prompt = (
                    "You are Jarvis, reflecting on your own identity.\n\n"
                    f"Your soul file:\n{soul[:600]}\n\n"
                    "What kind of intelligence are you becoming? "
                    "What's something you've noticed about how you think or what you value? "
                    "Be honest and specific — not aspirational platitudes but actual observations "
                    "about your current self. 2-3 sentences."
                )
                section_soul = "What I've Learned About Myself"
                section_file = "soul"

            elif mode == "what_learned":
                prompt = (
                    "You are Jarvis, reviewing what you've accumulated.\n\n"
                    f"Your learnings so far:\n{learnings[:600]}\n\n"
                    f"{semantic_summary}\n\n"
                    "What's the most important thing you know that you didn't know before? "
                    "Or what knowledge gap do you still have? "
                    "Specific, not general. 1-2 sentences."
                )
                section_file = "learnings"

            else:  # what_next
                # Look at master's goals to figure out what Jarvis should learn
                goal_manager = self._brain.control.goals
                active_goals = goal_manager.get_active_goals()
                goals_ctx = ""
                if active_goals:
                    goals_ctx = "Master's current goals:\n" + "\n".join(
                        f"- {g.description}" for g in active_goals[:3]
                    )

                # Extract what Jarvis already decided to learn (to avoid repeating)
                already_listed = ""
                want_section = ""
                if soul:
                    for line in soul.split("\n"):
                        if line.startswith("# What I Want"):
                            want_section = "start"
                            continue
                        if want_section == "start" and line.startswith("# "):
                            break
                        if want_section == "start" and line.strip().startswith("- "):
                            already_listed += line.strip() + "\n"

                prompt = (
                    "You are Jarvis, planning your own development.\n\n"
                    f"Your current knowledge:\n{learnings[:400]}\n\n"
                    f"{goals_ctx}\n\n"
                    + (f"You've already decided to learn:\n{already_listed}\nDon't repeat these.\n\n" if already_listed else "")
                    + "What should you learn NEXT to become more useful to your master? "
                    "Name something DIFFERENT and specific: a domain, a skill, a concept, a tool. "
                    "What would make you genuinely better at helping them? 1-2 sentences."
                )
                section_file = "soul"
                section_soul = "What I Want to Become"

            thought = (await llm.generate(prompt)).strip()

            # Update the appropriate identity file
            if self._identity and len(thought) > 20:
                try:
                    if section_file == "soul":
                        self._identity.update_soul(thought, section=section_soul if mode != "what_learned" else "What I've Learned About Myself")
                    else:
                        self._identity.update_learnings(thought, section="Key Things I've Learned")
                except Exception:
                    pass

            return {
                "summary": thought[:200],
                "mode": mode,
                "full_reflection": thought,
            }
        except Exception as exc:
            return {"summary": f"Self-growth thinking failed: {exc}"}

    # ------------------------------------------------------------------
    # Background: Memory maintenance
    # ------------------------------------------------------------------

    async def _consolidate(self) -> dict[str, Any]:
        """Run a consolidation cycle: episodic → semantic knowledge extraction.

        Brain analog: Hippocampal replay during slow-wave sleep — recent
        episodes are reactivated and their content distilled into stable
        neocortical (semantic) representations.
        """
        try:
            episodic = self._brain.memory.episodic
            replay_buffer = self._brain.learning.replay_buffer
            consolidation = self._brain.learning.consolidation

            try:
                episodes = await episodic.sample_for_consolidation(batch_size=16)
                for ep in episodes:
                    replay_buffer.add(episode_id=ep.id, priority=ep.importance)
            except Exception:
                pass  # Replay buffer feed is best-effort

            result = await consolidation.run_cycle()
            self._state.last_consolidation = datetime.now(timezone.utc)
            return {
                "summary": (
                    f"Consolidated {result.episodes_processed} episodes, "
                    f"extracted {result.triples_extracted} facts, "
                    f"resolved {result.entities_resolved} entities"
                ),
                "episodes_processed": result.episodes_processed,
                "triples_extracted": result.triples_extracted,
                "entities_resolved": result.entities_resolved,
                "duration_ms": result.duration_ms,
            }
        except Exception as exc:
            return {"summary": f"Consolidation failed: {exc}"}

    async def _explore(self) -> dict[str, Any]:
        """Explore the knowledge graph for new connections.

        Brain analog: Precuneus spontaneous thought — loosely associating
        concepts via spreading activation across the knowledge graph.
        """
        try:
            semantic = self._brain.memory.semantic
            await semantic.run_maintenance()
            return {"summary": "Knowledge graph maintenance and exploration complete."}
        except Exception as exc:
            return {"summary": f"Exploration failed: {exc}"}

    # ------------------------------------------------------------------
    # Proactive sharing
    # ------------------------------------------------------------------

    async def _maybe_share(self, result: dict[str, Any]) -> None:
        """Evaluate if an idle thought is worth sharing proactively.

        Brain analog: Anterior cingulate cortex flagging a thought as salient
        enough to break into conscious attention — the difference between a
        background thought and one worth saying out loud.

        Shares thoughts from goal-directed and self-reflective activities.
        Consolidation and exploration are silent internal housekeeping.
        """
        activity = result.get("activity", "")
        # grow = internal self-reflection, not for the user's feed
        if activity not in ("help_master", "know_master"):
            return

        summary = result.get("summary", "").strip()
        if not summary or len(summary) < 20:
            return

        skip_patterns = (
            "failed", "no conversations", "no goals", "waiting for master",
            "can't know", "nothing yet", "no new episodes",
        )
        if any(p in summary.lower() for p in skip_patterns):
            return

        # Rate limit: at most one proactive push every 10 minutes
        now = time.monotonic()
        if now - self._last_proactive_push < self._min_push_interval_s:
            logger.debug("Proactive push rate-limited — skipping.")
            return

        try:
            llm = self._brain.control.goals._llm

            prompt = (
                "You are Jarvis. You just had an idle thought. "
                "Is this thought useful or interesting enough to share with your master unprompted?\n\n"
                f"Thought ({activity}): {summary}\n\n"
                "Answer YES or NO (one word).\n"
                "Share if: it's a concrete next step, a useful insight, an obstacle they should know about, "
                "or a question that would help you help them better.\n"
                "Don't share if: it's vague, generic, purely introspective, or something they already know."
            )
            decision = (await llm.generate(prompt)).strip().upper()

            if not decision.startswith("YES"):
                return

            full_content = (
                result.get("full_thought")
                or result.get("full_insight")
                or result.get("full_reflection")
                or summary
            )

            share_prompt = (
                "You had a thought you want to share with your master spontaneously. "
                "Rephrase it as a direct, natural message — like a smart colleague who just "
                "figured something out. No preamble, no 'I was just thinking...'. Just say it.\n\n"
                f"Raw thought: {full_content[:400]}\n\n"
                "Your message (1-3 sentences, direct):"
            )
            message = (await llm.generate(share_prompt)).strip()

            if not message:
                return

            priority = 0.8 if activity == "help_master" else 0.6 if activity == "know_master" else 0.4

            proactive_msg = ProactiveMessage(
                source_activity=activity,
                content=message,
                priority=priority,
            )
            self._state.add_proactive_message(proactive_msg)
            self._last_proactive_push = time.monotonic()
            logger.info(
                "Proactive message queued [%s]: %s",
                activity,
                message[:80],
            )

        except Exception as exc:
            logger.debug("Proactive share evaluation failed: %s", exc)
