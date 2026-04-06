"""
KnowledgeBootstrap — seeds Mnemon's semantic KG with foundational knowledge.

Brain analog: Early childhood learning — a human child doesn't start with
zero knowledge. By age 5, they've absorbed ~14,000 words, basic causal
relationships, object permanence, number sense, and a working model of
the social world — all before formal schooling. This module does the same
for Mnemon: loads structured knowledge from freely available sources before
the system has had any real conversations.

Learning phases (in order of impact):
  Phase 1 — Wikipedia Summaries
    Uses the Wikipedia REST API to fetch clean article summaries for a
    configurable topic list. Each summary becomes an episode → consolidation
    extracts semantic triples. Fast: ~0.5s per article, no API key required.

  Phase 2 — ConceptNet Assertions
    Uses the ConceptNet public REST API to fetch entity relationships
    (IsA, UsedFor, CapableOf, etc.) and writes them DIRECTLY into the
    semantic store, bypassing the LLM consolidation step. Extremely fast:
    seeding 10k+ triples takes seconds.

  Phase 3 — Domain-Specific Articles (optional)
    A list of Wikipedia article titles or web URLs specific to the user's
    domain can be fed in for deeper knowledge in a narrow area.

Usage:
    from mnemon.learning.bootstrap import KnowledgeBootstrap

    bootstrap = KnowledgeBootstrap(brain, llm, embedding_provider)
    await bootstrap.run(phases=[1, 2])
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Wikipedia REST API — returns clean JSON, no scraping
_WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# ConceptNet public API — no auth required
_CONCEPTNET_API = "https://api.conceptnet.io/c/en/{concept}?limit=50&offset=0"

# Default topic seed for Phase 1 — broad foundational knowledge
FOUNDATION_TOPICS = [
    # Science & Nature
    "Artificial intelligence", "Machine learning", "Neuroscience",
    "Human brain", "Memory", "Cognition",
    # Technology
    "Computer science", "Software engineering", "Internet",
    "Natural language processing", "Deep learning",
    # Practical world knowledge
    "Economics", "Business", "Marketing", "Entrepreneurship",
    # Philosophy & thinking
    "Logic", "Philosophy of mind", "Learning",
]

# Core ConceptNet concepts for Phase 2
FOUNDATION_CONCEPTS = [
    "person", "place", "thing", "time", "idea", "knowledge",
    "computer", "software", "data", "internet", "business",
    "learning", "memory", "language", "thought", "goal",
    "action", "result", "problem", "solution",
]


class KnowledgeBootstrap:
    """Seeds Mnemon's memory stores with foundational structured knowledge.

    Phase 1 (Wikipedia) populates episodic memory with article content,
    which the background consolidation engine then processes into semantic
    triples over time. This mirrors how humans read books to build knowledge.

    Phase 2 (ConceptNet) writes entity relationship triples DIRECTLY into
    the semantic store without LLM processing. This mirrors the implicit
    world model every human has before they can even read — knowing that
    dogs bark, fire is hot, people have names.
    """

    def __init__(
        self,
        brain: Any,           # Mnemon instance
        topics: list[str] | None = None,
        concepts: list[str] | None = None,
    ) -> None:
        self._brain = brain
        self._topics = topics or FOUNDATION_TOPICS
        self._concepts = concepts or FOUNDATION_CONCEPTS

    async def run(
        self,
        phases: list[int] | None = None,
        on_progress: Any = None,
    ) -> dict[str, int]:
        """Run bootstrap phases.

        Parameters
        ----------
        phases:
            Which phases to run. Default: [1, 2].
            Phase 1 = Wikipedia summaries → episodic memory
            Phase 2 = ConceptNet assertions → semantic store directly
        on_progress:
            Optional async callable(message: str) for progress updates.

        Returns
        -------
        dict with counts: wikipedia_articles, conceptnet_triples
        """
        phases = phases or [1, 2]
        results: dict[str, int] = {"wikipedia_articles": 0, "conceptnet_triples": 0}

        async def _report(msg: str) -> None:
            logger.info("Bootstrap: %s", msg)
            if on_progress:
                await on_progress(msg)

        if 1 in phases:
            await _report(f"Phase 1: Loading {len(self._topics)} Wikipedia articles...")
            count = await self._bootstrap_wikipedia(on_progress=_report)
            results["wikipedia_articles"] = count
            await _report(f"Phase 1 complete — {count} articles encoded to episodic memory.")

        if 2 in phases:
            await _report(f"Phase 2: Loading ConceptNet for {len(self._concepts)} concepts...")
            count = await self._bootstrap_conceptnet(on_progress=_report)
            results["conceptnet_triples"] = count
            await _report(f"Phase 2 complete — {count} triples written to semantic store.")

        return results

    # ------------------------------------------------------------------
    # Phase 1: Wikipedia
    # ------------------------------------------------------------------

    async def _bootstrap_wikipedia(self, on_progress: Any = None) -> int:
        """Fetch Wikipedia summaries and encode as episodic memories."""
        from mnemon.core.models import Episode

        encoded = 0
        async with httpx.AsyncClient(timeout=10.0) as client:
            for topic in self._topics:
                try:
                    url = _WIKIPEDIA_API.format(title=topic.replace(" ", "_"))
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        logger.debug("Wikipedia: %s returned %d", topic, resp.status_code)
                        continue

                    data = resp.json()
                    title = data.get("title", topic)
                    summary = data.get("extract", "")
                    if not summary or len(summary) < 50:
                        continue

                    # Encode as a high-importance episode (important foundational knowledge)
                    episode = Episode(
                        agent_id="bootstrap",
                        session_id=uuid.uuid4(),
                        context=f"Wikipedia: {title}\n\n{summary[:800]}",
                        action="read",
                        outcome="foundational knowledge ingested",
                        importance=0.7,  # higher than web observer — this is curated
                    )
                    ep_id = await self._brain.memory.episodic.encode(episode)

                    # Push to replay buffer with high priority for early consolidation
                    try:
                        self._brain.learning.replay_buffer.add(ep_id, priority=0.7)
                    except Exception:
                        pass

                    encoded += 1
                    logger.debug("Bootstrap: encoded Wikipedia article '%s'", title)

                except Exception as exc:
                    logger.warning("Bootstrap: Wikipedia fetch failed for '%s': %s", topic, exc)

        return encoded

    # ------------------------------------------------------------------
    # Phase 2: ConceptNet
    # ------------------------------------------------------------------

    async def _bootstrap_conceptnet(self, on_progress: Any = None) -> int:
        """Fetch ConceptNet assertions and write directly to the semantic store."""
        written = 0
        semantic = self._brain.memory.semantic

        async with httpx.AsyncClient(timeout=10.0) as client:
            for concept in self._concepts:
                try:
                    url = _CONCEPTNET_API.format(concept=concept.lower().replace(" ", "_"))
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        logger.debug("ConceptNet: %s returned %d", concept, resp.status_code)
                        continue

                    data = resp.json()
                    edges = data.get("edges", [])

                    for edge in edges:
                        triple = _conceptnet_edge_to_triple(edge)
                        if triple is None:
                            continue
                        try:
                            await _write_triple_to_semantic(semantic, triple)
                            written += 1
                        except Exception as exc:
                            logger.debug("Bootstrap: triple write failed: %s", exc)

                    logger.debug(
                        "Bootstrap: ConceptNet '%s' — %d edges processed", concept, len(edges)
                    )

                except Exception as exc:
                    logger.warning("Bootstrap: ConceptNet fetch failed for '%s': %s", concept, exc)

        return written


# ------------------------------------------------------------------
# ConceptNet helpers
# ------------------------------------------------------------------


def _conceptnet_edge_to_triple(edge: dict) -> dict | None:
    """Extract a (subject, relation, object, confidence) triple from a ConceptNet edge."""
    try:
        rel = edge.get("rel", {}).get("label", "")
        start = edge.get("start", {})
        end = edge.get("end", {})

        # Only use English nodes
        if start.get("language", "en") != "en" or end.get("language", "en") != "en":
            return None

        subj = start.get("label", "").strip()
        obj = end.get("label", "").strip()
        weight = float(edge.get("weight", 1.0))

        if not subj or not obj or not rel:
            return None

        return {
            "subject": subj,
            "predicate": rel,
            "object": obj,
            "confidence": min(weight / 10.0, 1.0),  # normalise to [0, 1]
        }
    except Exception:
        return None


async def _write_triple_to_semantic(semantic: Any, triple: dict) -> None:
    """Write a raw triple dict directly to the semantic store.

    Bypasses the LLM consolidation step — we trust ConceptNet's
    crowd-sourced knowledge directly.
    """
    from mnemon.core.models import Entity, KnowledgeTriple

    subject_entity = Entity(
        name=triple["subject"],
        canonical_name=triple["subject"].lower(),
        entity_type="concept",
    )
    object_entity = Entity(
        name=triple["object"],
        canonical_name=triple["object"].lower(),
        entity_type="concept",
    )
    knowledge_triple = KnowledgeTriple(
        subject=subject_entity,
        predicate=triple["predicate"],
        object=object_entity,
        confidence=triple["confidence"],
        source="conceptnet",
    )
    await semantic.store_triple(knowledge_triple)
