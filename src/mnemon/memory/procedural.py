"""
ProceduralMemoryStore — skill storage with reinforcement-learning utility scoring.

Brain analog: Basal ganglia (striatum) — encodes stimulus–action mappings and
selects actions based on expected reward.  Skills with high utility scores are
preferentially retrieved, mirroring the striatum's role in action selection and
habit formation.  Utility scores are updated via an exponential moving average
(TD-style) after each execution outcome, allowing the system to gradually
deprecate low-value skills and amplify reliable ones.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from mnemon.core.config import ProceduralConfig
from mnemon.core.exceptions import MemoryError, RetrievalError
from mnemon.core.interfaces import (
    DocumentStore,
    EmbeddingProvider,
    ProceduralMemoryInterface,
    VectorStore,
)
from mnemon.core.models import Skill, SkillStatus

logger = logging.getLogger(__name__)

# Document type discriminator stored in every skill document.
_TYPE_SKILL = "skill"


class ProceduralMemoryStore(ProceduralMemoryInterface):
    """Vector-backed procedural memory with RL utility scoring.

    Each :class:`~mnemon.core.models.Skill` is persisted in two stores:

    * :class:`~mnemon.core.interfaces.VectorStore` — dense embedding of the
      skill description for similarity-based retrieval (situation → skill).
    * :class:`~mnemon.core.interfaces.DocumentStore` — authoritative serialised
      record holding the full skill definition, utility score, and counters.

    Retrieval re-ranks ANN results by a weighted combination of vector
    similarity (60 %) and learned utility (40 %), mirroring how the
    striatum blends perceptual similarity with expected value.
    """

    def __init__(
        self,
        config: ProceduralConfig,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._config = config
        self._vectors = vector_store
        self._docs = document_store
        self._embedder = embedding_provider
        logger.debug(
            "ProceduralMemoryStore initialised — lr=%.4f deprecation_threshold=%.4f",
            config.utility.learning_rate,
            config.utility.deprecation_threshold,
        )

    # ------------------------------------------------------------------
    # ProceduralMemoryInterface implementation
    # ------------------------------------------------------------------

    async def register(self, skill: Skill) -> UUID:
        """Persist a new skill definition and return its UUID.

        If *skill* has no embedding, one is computed from its description via
        the configured :class:`~mnemon.core.interfaces.EmbeddingProvider`.

        Parameters
        ----------
        skill:
            The skill object to store.  ``skill.id`` is used as the primary key.

        Returns
        -------
        UUID
            The ID of the registered skill.
        """
        try:
            embedding = skill.embedding
            if embedding is None:
                embedding = await self._embedder.embed(skill.description)
                logger.debug(
                    "Computed embedding for skill id=%s name=%r", skill.id, skill.name
                )

            # Index the embedding for similarity retrieval.
            metadata = {
                "name": skill.name,
                "type": skill.type,
                "status": skill.status,
                "utility": skill.utility,
                "_type": _TYPE_SKILL,
                "skill_id": str(skill.id),
            }
            await self._vectors.insert(
                id=skill.id,
                embedding=embedding,
                metadata=metadata,
            )

            # Persist the full skill document (including computed embedding).
            skill_with_embedding = skill.model_copy(update={"embedding": embedding})
            doc = skill_with_embedding.model_dump(mode="json")
            doc["_type"] = _TYPE_SKILL
            await self._docs.put(skill.id, doc)

            logger.info("Registered skill id=%s name=%r", skill.id, skill.name)
            return skill.id

        except Exception as exc:
            raise MemoryError(f"Failed to register skill id={skill.id}: {exc}") from exc

    async def retrieve(
        self,
        situation_embedding: list[float],
        top_k: int = 5,
    ) -> list[Skill]:
        """Return the *top_k* skills most applicable to the current situation.

        Performs ANN search, loads full skill records, filters out deprecated
        skills, then re-ranks by:
            ``final_score = 0.6 * vector_similarity + 0.4 * skill.utility``

        Parameters
        ----------
        situation_embedding:
            Dense embedding of the current situational context.
        top_k:
            Maximum number of skills to return.

        Returns
        -------
        list[Skill]
            Skills sorted by descending final score, length ≤ top_k.
        """
        try:
            # Over-fetch to compensate for deprecated/missing skills.
            oversample_k = max(top_k * 3, 20)
            results = await self._vectors.search(
                query_embedding=situation_embedding,
                top_k=oversample_k,
                filters={"_type": _TYPE_SKILL},
            )

            ranked: list[tuple[float, Skill]] = []
            for result in results:
                skill_id_str = result.metadata.get("skill_id")
                if skill_id_str is None:
                    continue

                doc = await self._docs.get(UUID(skill_id_str))
                if doc is None or doc.get("_type") != _TYPE_SKILL:
                    continue

                skill = Skill.model_validate(doc)

                # Exclude deprecated skills.
                if skill.status == SkillStatus.DEPRECATED:
                    continue

                final_score = 0.6 * result.score + 0.4 * skill.utility
                ranked.append((final_score, skill))

            ranked.sort(key=lambda t: t[0], reverse=True)
            top_skills = [skill for _, skill in ranked[:top_k]]

            logger.debug(
                "retrieve situation_embedding top_k=%d returned=%d skills",
                top_k,
                len(top_skills),
            )
            return top_skills

        except MemoryError:
            raise
        except Exception as exc:
            raise RetrievalError(f"ProceduralMemoryStore.retrieve failed: {exc}") from exc

    async def get(self, skill_id: UUID) -> Skill | None:
        """Fetch a skill by its UUID.

        Returns
        -------
        Skill | None
            The validated skill object, or ``None`` if not found.
        """
        try:
            doc = await self._docs.get(skill_id)
            if doc is None or doc.get("_type") != _TYPE_SKILL:
                return None
            return Skill.model_validate(doc)
        except Exception as exc:
            raise RetrievalError(
                f"ProceduralMemoryStore.get failed for id={skill_id}: {exc}"
            ) from exc

    async def update_utility(
        self,
        skill_id: UUID,
        reward: float,
        success: bool,
    ) -> None:
        """Update a skill's utility using an exponential moving average.

        Applies the TD-style EMA rule:
            ``utility = (1 - lr) * utility + lr * reward``

        The updated utility is clamped to [0, 1].  Counters and timestamp
        are updated, and if utility falls below the deprecation threshold
        the skill is auto-deprecated (unless it is human-authored).

        Parameters
        ----------
        skill_id:
            UUID of the skill to update.
        reward:
            Observed reward signal (should be in a meaningful range, e.g. [0, 1]).
        success:
            Whether the skill execution was considered successful.
        """
        try:
            doc = await self._docs.get(skill_id)
            if doc is None or doc.get("_type") != _TYPE_SKILL:
                logger.warning(
                    "update_utility: skill id=%s not found — skipping", skill_id
                )
                return

            skill = Skill.model_validate(doc)
            lr = self._config.utility.learning_rate

            new_utility = (1.0 - lr) * skill.utility + lr * reward
            new_utility = max(0.0, min(1.0, new_utility))

            updates: dict = {
                "utility": new_utility,
                "last_used": datetime.now(timezone.utc),
            }

            if success:
                updates["success_count"] = skill.success_count + 1
            else:
                updates["failure_count"] = skill.failure_count + 1

            # Auto-deprecate non-human-authored skills with very low utility.
            deprecation_threshold = self._config.utility.deprecation_threshold
            is_human_authored = skill.creation_source == "human_authored"
            if new_utility < deprecation_threshold and not is_human_authored:
                updates["status"] = SkillStatus.DEPRECATED
                logger.info(
                    "Auto-deprecating skill id=%s name=%r utility=%.4f < threshold=%.4f",
                    skill_id,
                    skill.name,
                    new_utility,
                    deprecation_threshold,
                )

            updated_skill = skill.model_copy(update=updates)
            updated_doc = updated_skill.model_dump(mode="json")
            updated_doc["_type"] = _TYPE_SKILL
            await self._docs.put(skill_id, updated_doc)

            # Keep vector store metadata in sync.
            if updated_skill.embedding is not None:
                metadata = {
                    "name": updated_skill.name,
                    "type": updated_skill.type,
                    "status": updated_skill.status,
                    "utility": new_utility,
                    "_type": _TYPE_SKILL,
                    "skill_id": str(skill_id),
                }
                await self._vectors.update(
                    id=skill_id,
                    embedding=updated_skill.embedding,
                    metadata=metadata,
                )

            logger.debug(
                "update_utility skill id=%s reward=%.4f success=%s utility %.4f → %.4f",
                skill_id,
                reward,
                success,
                skill.utility,
                new_utility,
            )

        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(
                f"update_utility failed for skill id={skill_id}: {exc}"
            ) from exc

    async def deprecate(self, skill_id: UUID) -> None:
        """Mark a skill as deprecated so it is excluded from retrieval.

        Parameters
        ----------
        skill_id:
            UUID of the skill to deprecate.
        """
        try:
            doc = await self._docs.get(skill_id)
            if doc is None or doc.get("_type") != _TYPE_SKILL:
                logger.warning(
                    "deprecate: skill id=%s not found — skipping", skill_id
                )
                return

            skill = Skill.model_validate(doc)
            if skill.status == SkillStatus.DEPRECATED:
                logger.debug("deprecate: skill id=%s already deprecated", skill_id)
                return

            deprecated_skill = skill.model_copy(update={"status": SkillStatus.DEPRECATED})
            updated_doc = deprecated_skill.model_dump(mode="json")
            updated_doc["_type"] = _TYPE_SKILL
            await self._docs.put(skill_id, updated_doc)

            # Sync metadata in vector store so retrieval filters work correctly.
            if deprecated_skill.embedding is not None:
                metadata = {
                    "name": deprecated_skill.name,
                    "type": deprecated_skill.type,
                    "status": SkillStatus.DEPRECATED,
                    "utility": deprecated_skill.utility,
                    "_type": _TYPE_SKILL,
                    "skill_id": str(skill_id),
                }
                await self._vectors.update(
                    id=skill_id,
                    embedding=deprecated_skill.embedding,
                    metadata=metadata,
                )

            logger.info("Deprecated skill id=%s name=%r", skill_id, skill.name)

        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(
                f"deprecate failed for skill id={skill_id}: {exc}"
            ) from exc
