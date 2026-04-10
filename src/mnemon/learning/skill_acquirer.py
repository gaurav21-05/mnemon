"""
Skill Acquirer — automated procedural skill learning from experience.

Brain analog: The basal ganglia's role in habit formation. Just as the
striatum detects repeated action patterns and compiles them into efficient
motor programs, this module monitors episodic experience for recurring
action sequences, synthesises them into reusable procedural skills, and
refines those skills through trial-and-error reinforcement. This mirrors
the cortico-striatal loop: cortex detects opportunities, striatum selects
and encodes the routine, and dopaminergic feedback strengthens successful
patterns while weakening failures.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from mnemon.core.models import (
    Condition,
    ConditionType,
    Episode,
    Skill,
    SkillStatus,
    SkillType,
)

if TYPE_CHECKING:
    from mnemon.core.config import ProceduralConfig
    from mnemon.core.interfaces import (
        EmbeddingProvider,
        EpisodicMemoryInterface,
        LLMProvider,
        ProceduralMemoryInterface,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


class SkillNeed(BaseModel):
    """A detected need for a new procedural skill."""

    description: str
    trigger_pattern: str
    evidence_count: int = Field(ge=1, default=1)
    source_episode_ids: list[UUID] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON schemas for LLM structured generation
# ---------------------------------------------------------------------------

_SKILL_NEED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "needs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "trigger_pattern": {"type": "string"},
                    "evidence_count": {"type": "integer", "minimum": 1},
                },
                "required": ["description", "trigger_pattern", "evidence_count"],
            },
        }
    },
    "required": ["needs"],
}

_SKILL_DEFINITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "definition": {"type": "string"},
        "preconditions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "postconditions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["name", "description", "definition", "preconditions", "postconditions"],
}

_SKILL_REFINEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "definition": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["definition", "description"],
}

_COMPOSITE_SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "definition": {"type": "string"},
                    "sub_skill_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "preconditions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "postconditions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "name",
                    "description",
                    "definition",
                    "sub_skill_names",
                    "preconditions",
                    "postconditions",
                ],
            },
        }
    },
    "required": ["skills"],
}


# ---------------------------------------------------------------------------
# SkillAcquirer
# ---------------------------------------------------------------------------


class SkillAcquirer:
    """
    Detects, synthesises, and refines procedural skills from episodic experience.

    Implements the basal ganglia habit-formation pipeline:
      1. detect_skill_need  — pattern detection over recent episodes
      2. synthesize_skill   — LLM-driven skill generation from a detected need
      3. acquire_skill      — full register-in-memory pipeline
      4. refine_skill       — failure-driven iterative improvement
      5. detect_composite_skills — macro-operator detection from co-occurring sequences
    """

    def __init__(
        self,
        config: ProceduralConfig,
        procedural_memory: ProceduralMemoryInterface,
        episodic_memory: EpisodicMemoryInterface,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._acquisition_config = config.skill_acquisition
        self._procedural_memory = procedural_memory
        self._episodic_memory = episodic_memory
        self._llm = llm
        self._embedding_provider = embedding_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect_skill_need(self, recent_episodes: list[Episode]) -> list[SkillNeed]:
        """
        Analyse recent episodes for recurring patterns that suggest a new skill.

        Examines action repetition, contextual similarity, and failure clusters
        to surface opportunities for skill compilation — analogous to the
        striatum detecting which cortical sequences are worth automating.

        Parameters
        ----------
        recent_episodes:
            Episodes to scan for actionable patterns.

        Returns
        -------
        list[SkillNeed]
            Detected needs, each backed by at least one supporting episode.
        """
        if not recent_episodes:
            return []

        formatted = self._format_episodes(recent_episodes)
        prompt = (
            "Analyze these recent agent experiences and identify any recurring action patterns\n"
            "that could be compiled into reusable skills.\n\n"
            f"Experiences:\n{formatted}\n\n"
            "Return a JSON object with key \"needs\" containing a list of objects, each with:\n"
            "- \"description\": what the skill should do\n"
            "- \"trigger_pattern\": when the skill should activate\n"
            "- \"evidence_count\": how many episodes support this pattern"
        )

        try:
            result = await self._llm.generate_structured(prompt, _SKILL_NEED_SCHEMA)
        except Exception:
            logger.exception("LLM call failed during detect_skill_need")
            return []

        needs: list[SkillNeed] = []
        episode_ids = [ep.id for ep in recent_episodes]

        for raw in result.get("needs", []):
            try:
                need = SkillNeed(
                    description=raw["description"],
                    trigger_pattern=raw["trigger_pattern"],
                    evidence_count=max(1, int(raw.get("evidence_count", 1))),
                    source_episode_ids=episode_ids,
                )
                needs.append(need)
            except Exception:
                logger.warning("Skipping malformed skill need entry: %s", raw)

        logger.info("Detected %d skill needs from %d episodes", len(needs), len(recent_episodes))
        return needs

    async def synthesize_skill(self, need: SkillNeed) -> Skill:
        """
        Generate a Skill definition from a detected SkillNeed via LLM.

        The LLM acts as the cortical planning system, drafting the procedure;
        the resulting skill starts as DRAFT with initial utility, awaiting
        real-world validation to graduate its status.

        Parameters
        ----------
        need:
            The detected pattern that motivates this skill.

        Returns
        -------
        Skill
            A fully populated Skill model ready for registration.

        Raises
        ------
        RuntimeError
            If the LLM call fails or returns an unparseable response.
        """
        prompt = (
            "Create a reusable skill/procedure for the following pattern:\n"
            f"Description: {need.description}\n"
            f"Trigger: {need.trigger_pattern}\n\n"
            "Return a JSON object with:\n"
            "- \"name\": short snake_case name\n"
            "- \"description\": what the skill does\n"
            "- \"definition\": step-by-step procedure as a prompt template\n"
            "- \"preconditions\": list of strings describing when this skill applies\n"
            "- \"postconditions\": list of strings describing expected outcomes"
        )

        result = await self._llm.generate_structured(prompt, _SKILL_DEFINITION_SCHEMA)

        preconditions = [
            Condition(type=ConditionType.STATE_CHECK, expression=expr)
            for expr in result.get("preconditions", [])
        ]
        postconditions = [
            Condition(type=ConditionType.STATE_CHECK, expression=expr)
            for expr in result.get("postconditions", [])
        ]

        description = result.get("description", need.description)
        embedding = await self._embedding_provider.embed(description)

        skill = Skill(
            id=uuid4(),
            name=result["name"],
            description=description,
            definition=result["definition"],
            type=SkillType.PROMPT_TEMPLATE,
            status=SkillStatus.DRAFT,
            utility=self._acquisition_config.initial_utility,
            creation_source="agent_learned",
            preconditions=preconditions,
            postconditions=postconditions,
            embedding=embedding,
        )

        logger.debug("Synthesized skill '%s' from need: %s", skill.name, need.description)
        return skill

    async def refine_skill(self, skill: Skill, failure_reason: str) -> Skill:
        """
        Improve a skill based on observed failure via LLM-driven critique.

        Mirrors dopaminergic negative prediction error: a worse-than-expected
        outcome triggers cortical re-planning and updated striatal encoding.
        Each refinement increments the version counter and preserves lineage
        via previous_version_id. Respects the configured max_refinement_attempts
        ceiling — beyond which callers should consider deprecating the skill.

        Parameters
        ----------
        skill:
            The skill instance that failed.
        failure_reason:
            Description of what went wrong during execution.

        Returns
        -------
        Skill
            A new version of the skill with updated definition and description.

        Raises
        ------
        RuntimeError
            If the LLM call fails or returns an unparseable response.
        ValueError
            If the skill has already reached the maximum refinement attempts.
        """
        if skill.version >= self._acquisition_config.max_refinement_attempts:
            raise ValueError(
                f"Skill '{skill.name}' has reached the maximum refinement attempts "
                f"({self._acquisition_config.max_refinement_attempts}). Consider deprecation."
            )

        prompt = (
            "This skill failed during execution:\n"
            f"Name: {skill.name}\n"
            f"Definition: {skill.definition}\n"
            f"Failure reason: {failure_reason}\n\n"
            "Provide an improved version. Return JSON with \"definition\" and \"description\" keys."
        )

        result = await self._llm.generate_structured(prompt, _SKILL_REFINEMENT_SCHEMA)

        new_description = result.get("description", skill.description)
        new_embedding = await self._embedding_provider.embed(new_description)

        refined = skill.model_copy(
            update={
                "id": uuid4(),
                "version": skill.version + 1,
                "previous_version_id": skill.id,
                "definition": result["definition"],
                "description": new_description,
                "embedding": new_embedding,
                "status": SkillStatus.DRAFT,
            }
        )

        logger.info(
            "Refined skill '%s' to version %d after failure: %s",
            refined.name,
            refined.version,
            failure_reason,
        )
        return refined

    async def acquire_skill(self, need: SkillNeed) -> Skill | None:
        """
        Full skill acquisition pipeline: synthesize, register, and return.

        Combines synthesis and memory registration into a single atomic
        operation — mirroring the consolidation step where a cortical routine
        is committed to stable striatal long-term memory.

        Parameters
        ----------
        need:
            The detected pattern to compile into a skill.

        Returns
        -------
        Skill | None
            The registered skill on success, or None if synthesis failed.
        """
        try:
            skill = await self.synthesize_skill(need)
        except Exception:
            logger.exception(
                "Skill synthesis failed for need '%s'; acquisition aborted", need.description
            )
            return None

        try:
            registered_id = await self._procedural_memory.register(skill)
            logger.info(
                "Acquired and registered skill '%s' (id=%s) from need: %s",
                skill.name,
                registered_id,
                need.description,
            )
        except Exception:
            logger.exception(
                "Failed to register skill '%s' in procedural memory", skill.name
            )
            return None

        return skill

    async def detect_composite_skills(
        self, recent_episodes: list[Episode]
    ) -> list[Skill]:
        """
        Identify frequently co-occurring skill sequences and propose macro-operators.

        Mirrors the hierarchical chunking behaviour of the basal ganglia, where
        repeated sub-skill sequences are progressively compressed into single
        high-level action chunks — reducing deliberate cognitive load over time.

        Parameters
        ----------
        recent_episodes:
            Episodes to mine for co-occurring skill invocations.

        Returns
        -------
        list[Skill]
            Proposed composite skills with sub_skills populated (as UUIDs of
            already-registered component skills where available).
        """
        if not recent_episodes:
            return []

        formatted = self._format_episodes(recent_episodes)
        prompt = (
            "Analyze these agent experiences and identify sequences of actions that frequently "
            "appear together in the same order. Propose composite macro-skills (higher-level "
            "operators) that combine these sub-sequences into single reusable routines.\n\n"
            f"Experiences:\n{formatted}\n\n"
            "Return a JSON object with key \"skills\" containing a list of composite skill "
            "proposals. Each item should have:\n"
            "- \"name\": short snake_case name for the composite skill\n"
            "- \"description\": what the composite skill accomplishes end-to-end\n"
            "- \"definition\": the combined step-by-step procedure as a prompt template\n"
            "- \"sub_skill_names\": list of component sub-skill names this macro combines\n"
            "- \"preconditions\": list of strings describing when this skill applies\n"
            "- \"postconditions\": list of strings describing expected outcomes"
        )

        try:
            result = await self._llm.generate_structured(prompt, _COMPOSITE_SKILL_SCHEMA)
        except Exception:
            logger.exception("LLM call failed during detect_composite_skills")
            return []

        composite_skills: list[Skill] = []

        for raw in result.get("skills", []):
            try:
                description = raw.get("description", "")
                embedding = await self._embedding_provider.embed(description)

                preconditions = [
                    Condition(type=ConditionType.STATE_CHECK, expression=expr)
                    for expr in raw.get("preconditions", [])
                ]
                postconditions = [
                    Condition(type=ConditionType.STATE_CHECK, expression=expr)
                    for expr in raw.get("postconditions", [])
                ]

                # sub_skill_names are resolved to UUIDs opportunistically;
                # if the procedural store has no record, the list stays empty.
                sub_skill_ids = await self._resolve_sub_skill_ids(
                    raw.get("sub_skill_names", [])
                )

                skill = Skill(
                    id=uuid4(),
                    name=raw["name"],
                    description=description,
                    definition=raw["definition"],
                    type=SkillType.PROMPT_TEMPLATE,
                    status=SkillStatus.DRAFT,
                    utility=self._acquisition_config.initial_utility,
                    creation_source="agent_learned",
                    preconditions=preconditions,
                    postconditions=postconditions,
                    sub_skills=sub_skill_ids,
                    embedding=embedding,
                )
                composite_skills.append(skill)
            except Exception:
                logger.warning("Skipping malformed composite skill entry: %s", raw)

        logger.info("Detected %d composite skill proposals", len(composite_skills))
        return composite_skills

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_episodes(self, episodes: list[Episode]) -> str:
        """Render episodes as a numbered plaintext block for LLM consumption."""
        lines: list[str] = []
        for i, ep in enumerate(episodes, start=1):
            lines.append(
                f"{i}. [Context] {ep.context}\n"
                f"   [Action]  {ep.action}\n"
                f"   [Outcome] {ep.outcome}\n"
                f"   [Reward]  {ep.reward_signal:.3f}  "
                f"[Valence] {ep.emotional_valence:.3f}"
            )
        return "\n\n".join(lines)

    async def _resolve_sub_skill_ids(self, skill_names: list[str]) -> list[UUID]:
        """
        Attempt to resolve a list of sub-skill names to registered UUIDs.

        Uses the procedural memory's embedding-based retrieval with the skill
        name as the query text. Only high-confidence matches are included;
        unresolvable names are silently skipped to keep composite skill
        registration non-blocking.
        """
        if not skill_names:
            return []

        resolved: list[UUID] = []
        for name in skill_names:
            try:
                embedding = await self._embedding_provider.embed(name)
                candidates = await self._procedural_memory.retrieve(
                    situation_embedding=embedding, top_k=1
                )
                if candidates:
                    resolved.append(candidates[0].id)
            except Exception:
                logger.debug("Could not resolve sub-skill name '%s' to a UUID", name)

        return resolved
