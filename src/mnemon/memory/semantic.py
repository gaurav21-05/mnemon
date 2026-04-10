"""
SemanticMemoryStore — structured knowledge storage with graph-based retrieval.

Brain analog: Neocortical association areas — encode distilled, context-independent
facts about the world as a directed property graph of concepts and relations.
Populated primarily by hippocampal consolidation rather than direct encoding.
Spreading activation (Personalised PageRank) models how activation propagates
through cortical columns during associative recall.  Community detection mirrors
how the brain organises concepts into semantic neighbourhoods.
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from mnemon.core.exceptions import MemoryError, RetrievalError
from mnemon.core.interfaces import (
    DocumentStore,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    RankedNode,
    SemanticMemoryInterface,
    VectorItem,
    VectorStore,
)
from mnemon.core.models import (
    Community,
    Entity,
    EntityRef,
    SemanticCluster,
    SemanticTriple,
)

if TYPE_CHECKING:
    from mnemon.core.config import SemanticConfig

logger = logging.getLogger(__name__)

# Document type discriminator injected into every stored dict.
_TYPE_ENTITY = "entity"
_TYPE_TRIPLE = "triple"
_TYPE_COMMUNITY = "community"
_TYPE_CLUSTER = "cluster"


def _object_identity(value: EntityRef | str) -> str:
    """Return a stable comparison key for a triple object."""
    if isinstance(value, EntityRef):
        return f"entity:{value.entity_id}"
    return f"literal:{value.strip().lower()}"


class SemanticMemoryStore(SemanticMemoryInterface):
    """Graph-backed semantic memory with vector similarity and spreading activation.

    Each triple is persisted in three complementary stores:

    * :class:`~mnemon.core.interfaces.GraphStore` — directed property graph
      for multi-hop traversal and PageRank-based spreading activation.
    * :class:`~mnemon.core.interfaces.VectorStore` — dense embeddings for
      similarity-based triple retrieval.
    * :class:`~mnemon.core.interfaces.DocumentStore` — authoritative serialised
      record for full triple reconstruction.

    Entity nodes are stored both as graph nodes and as documents to support
    name-based coreference resolution without a separate entity store.
    """

    def __init__(
        self,
        config: SemanticConfig,
        graph_store: GraphStore,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._config = config
        self._graph = graph_store
        self._vectors = vector_store
        self._docs = document_store
        self._embedder = embedding_provider
        self._llm = llm_provider
        self._consistency_checked = False
        logger.debug(
            "SemanticMemoryStore initialised — graph_backend=%s llm=%s",
            config.graph_backend,
            type(llm_provider).__name__ if llm_provider is not None else "none",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _entity_doc_id(self, entity_id: UUID) -> UUID:
        """Stable document ID for an entity record (same as entity_id)."""
        return entity_id

    def _triple_doc_id(self, triple_id: UUID) -> UUID:
        """Stable document ID for a triple record (same as triple_id)."""
        return triple_id

    def _community_doc_id(self, community_id: UUID) -> UUID:
        """Stable document ID for a community record (same as community_id)."""
        return community_id

    async def _query_docs_by_type(self, doc_type: str) -> list[dict[str, Any]]:
        return await self._docs.query(filters={"_type": doc_type}, limit=100_000)

    def _build_entity_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": _TYPE_ENTITY,
            "entity_id": doc["entity_id"],
            "canonical_name": doc["canonical_name"],
        }

    def _build_triple_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        obj = doc.get("object")
        object_name = obj.get("name", "") if isinstance(obj, dict) else str(obj)
        subject = doc.get("subject", {})
        return {
            "triple_id": doc["id"],
            "predicate": doc.get("predicate", ""),
            "subject_name": subject.get("name", "") if isinstance(subject, dict) else "",
            "object_name": object_name,
            "_type": _TYPE_TRIPLE,
        }

    def _build_cluster_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": _TYPE_CLUSTER,
            "cluster_id": doc["id"],
            "level": doc.get("level", 0),
        }

    async def _expected_vector_items(self) -> list[VectorItem]:
        items: list[VectorItem] = []

        for doc in await self._query_docs_by_type(_TYPE_ENTITY):
            embedding = doc.get("embedding")
            if embedding is None:
                continue
            items.append(
                VectorItem(
                    id=UUID(str(doc["id"])),
                    embedding=embedding,
                    metadata=self._build_entity_metadata(doc),
                )
            )

        for doc in await self._query_docs_by_type(_TYPE_TRIPLE):
            embedding = doc.get("embedding")
            if embedding is None:
                continue
            items.append(
                VectorItem(
                    id=UUID(str(doc["id"])),
                    embedding=embedding,
                    metadata=self._build_triple_metadata(doc),
                )
            )

        for doc in await self._query_docs_by_type(_TYPE_CLUSTER):
            embedding = doc.get("embedding")
            if embedding is None:
                continue
            items.append(
                VectorItem(
                    id=UUID(str(doc["id"])),
                    embedding=embedding,
                    metadata=self._build_cluster_metadata(doc),
                )
            )

        return items

    def _vector_store_ids(self) -> set[str] | None:
        if hasattr(self._vectors, "_metadata"):
            metadata = getattr(self._vectors, "_metadata", {})
            if isinstance(metadata, dict):
                return set(metadata.keys())
        if hasattr(self._vectors, "_store"):
            store = getattr(self._vectors, "_store", {})
            if isinstance(store, dict):
                return {str(key) for key in store}
        return None

    async def _clear_vector_store(self) -> None:
        clear = getattr(self._vectors, "clear", None)
        if callable(clear):
            result = clear()
            if hasattr(result, "__await__"):
                await result
            return
        vector_ids = self._vector_store_ids()
        if vector_ids is None:
            raise MemoryError("Vector store does not support consistency repair")
        for vector_id in vector_ids:
            await self._vectors.delete(UUID(vector_id))

    async def _ensure_vector_doc_consistency(self) -> None:
        if self._consistency_checked:
            return

        expected_items = await self._expected_vector_items()
        expected_ids = {str(item.id) for item in expected_items}
        current_ids = self._vector_store_ids()

        if current_ids is None:
            current_count = await self._vectors.count()
            if current_count == len(expected_items):
                self._consistency_checked = True
                return
            logger.warning(
                "SemanticMemoryStore: vector/doc count mismatch (%d != %d) "
                "but vector IDs are unavailable; leaving store unchanged.",
                current_count,
                len(expected_items),
            )
            self._consistency_checked = True
            return

        if current_ids != expected_ids:
            logger.warning(
                "SemanticMemoryStore: repairing vector/document desync (vectors=%d docs=%d).",
                len(current_ids),
                len(expected_ids),
            )
            await self._clear_vector_store()
            if expected_items:
                await self._vectors.bulk_insert(expected_items)

        self._consistency_checked = True

    async def _ensure_entity_node(self, ref: EntityRef) -> None:
        """Insert an entity node into GraphStore + DocumentStore + VectorStore if absent."""
        existing = await self._graph.get_node(ref.entity_id)
        if existing is not None:
            return

        properties: dict[str, Any] = {"name": ref.name, "entity_id": str(ref.entity_id)}
        await self._graph.add_node(
            node_id=ref.entity_id,
            labels=["Entity"],
            properties=properties,
        )

        # Compute entity embedding for vector-based coreference resolution.
        try:
            entity_embedding = await self._embedder.embed(ref.name)
        except Exception:
            entity_embedding = None
            logger.debug("Could not compute embedding for entity %r", ref.name)

        # Persist a minimal entity document for coreference resolution.
        doc: dict[str, Any] = {
            "_type": _TYPE_ENTITY,
            "id": str(ref.entity_id),
            "entity_id": str(ref.entity_id),
            "canonical_name": ref.name,
            "aliases": [],
            "type": "unknown",
            "properties": {},
            "embedding": entity_embedding,
            "description": "",
            "importance": 0.5,
            "community_id": None,
        }
        await self._docs.put(self._entity_doc_id(ref.entity_id), doc)

        # Index entity embedding in VectorStore for similarity-based resolution.
        if entity_embedding is not None:
            await self._vectors.insert(
                id=ref.entity_id,
                embedding=entity_embedding,
                metadata={
                    "_type": _TYPE_ENTITY,
                    "entity_id": str(ref.entity_id),
                    "canonical_name": ref.name,
                },
            )

        logger.debug("Ensured entity node id=%s name=%r", ref.entity_id, ref.name)

    async def _ensure_literal_node(self, literal: str) -> UUID:
        """Create a synthetic literal node for string object values."""
        # Use a deterministic UUID based on the literal text so the same string
        # always maps to the same node.
        import hashlib

        digest = hashlib.sha256(literal.encode()).digest()[:16]
        literal_id = UUID(bytes=digest)

        existing = await self._graph.get_node(literal_id)
        if existing is None:
            await self._graph.add_node(
                node_id=literal_id,
                labels=["Literal"],
                properties={"value": literal},
            )
            logger.debug("Created literal node id=%s value=%r", literal_id, literal)
        return literal_id

    # ------------------------------------------------------------------
    # RAPTOR clustering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Return the cosine similarity between two equal-length vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _kmeans_assign(
        embeddings: list[list[float]],
        centers: list[list[float]],
    ) -> list[int]:
        """Assign each embedding to the index of its nearest center by cosine sim."""
        assignments: list[int] = []
        for emb in embeddings:
            best_idx = 0
            best_sim = -2.0
            for idx, center in enumerate(centers):
                sim = SemanticMemoryStore._cosine_similarity(emb, center)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            assignments.append(best_idx)
        return assignments

    @staticmethod
    def _kmeans_update_centers(
        embeddings: list[list[float]],
        assignments: list[int],
        k: int,
    ) -> list[list[float]]:
        """Recompute cluster centers as the mean of all assigned embeddings."""
        dim = len(embeddings[0])
        sums: list[list[float]] = [[0.0] * dim for _ in range(k)]
        counts: list[int] = [0] * k
        for emb, idx in zip(embeddings, assignments, strict=False):
            for d in range(dim):
                sums[idx][d] += emb[d]
            counts[idx] += 1

        centers: list[list[float]] = []
        for idx in range(k):
            if counts[idx] == 0:
                # Dead center — reinitialise from a random embedding.
                centers.append(list(random.choice(embeddings)))
            else:
                centers.append([v / counts[idx] for v in sums[idx]])
        return centers

    def _kmeans_cluster(
        self,
        embeddings: list[list[float]],
        k: int,
        max_iter: int = 20,
    ) -> list[int]:
        """Cluster *embeddings* into *k* groups using cosine-similarity k-means.

        Picks initial centers by randomly sampling k distinct embeddings, then
        iterates assignment and center-update steps up to *max_iter* times.

        Returns
        -------
        list[int]
            Per-embedding cluster assignments (integers in ``[0, k)``)
        """
        if k >= len(embeddings):
            return list(range(len(embeddings)))

        indices = random.sample(range(len(embeddings)), k)
        centers = [list(embeddings[i]) for i in indices]

        assignments: list[int] = []  # empty sentinel — never equals first iteration
        for _ in range(max_iter):
            new_assignments = self._kmeans_assign(embeddings, centers)
            if new_assignments == assignments:
                break
            assignments = new_assignments
            centers = self._kmeans_update_centers(embeddings, assignments, k)

        return assignments

    async def _summarise_triples(self, triples: list[SemanticTriple]) -> str:
        """Generate a natural-language summary for a cluster of triples.

        Uses the LLM provider when available; falls back to a simple
        concatenation of subject–predicate–object texts.
        """
        lines = [
            f"{t.subject.name} {t.predicate} "
            f"{t.object.name if isinstance(t.object, EntityRef) else t.object}"
            for t in triples
        ]
        if self._llm is None:
            return "; ".join(lines)

        prompt = (
            "Summarise the following knowledge triples into one concise paragraph "
            "that captures the key facts and relationships:\n\n"
            + "\n".join(f"- {line}" for line in lines)
        )
        try:
            return await self._llm.generate(prompt)
        except Exception:
            logger.debug(
                "LLM summarisation failed for %d triples — using concatenation fallback",
                len(triples),
            )
            return "; ".join(lines)

    async def _summarise_clusters(self, clusters: list[SemanticCluster]) -> str:
        """Generate a higher-level summary over a group of child clusters.

        Uses the LLM provider when available; falls back to joining child
        summaries directly.
        """
        summaries = [c.summary for c in clusters]
        if self._llm is None:
            return " | ".join(summaries)

        prompt = (
            "The following are summaries of related knowledge clusters. "
            "Synthesise them into a single, more abstract paragraph that "
            "captures the overarching themes:\n\n"
            + "\n".join(f"- {s}" for s in summaries)
        )
        try:
            return await self._llm.generate(prompt)
        except Exception:
            logger.debug(
                "LLM summarisation failed for %d clusters — using join fallback",
                len(clusters),
            )
            return " | ".join(summaries)

    # ------------------------------------------------------------------
    # SemanticMemoryInterface implementation
    # ------------------------------------------------------------------

    async def upsert_triples(self, triples: list[SemanticTriple]) -> int:
        """Insert or update semantic triples, deduplicating by triple ID.

        For each new triple the subject and object entity nodes are ensured in
        the graph, a directed edge is created, the embedding (if present) is
        indexed in the vector store, and the full triple dict is persisted to
        the document store.

        Returns
        -------
        int
            Number of triples actually written (skipping duplicates).
        """
        if not triples:
            return 0

        await self._ensure_vector_doc_consistency()

        written = 0
        for triple in triples:
            triple_doc_id = self._triple_doc_id(triple.id)
            now = datetime.now(UTC)

            # Exact ID deduplication: already persisted.
            existing = await self._docs.get(triple_doc_id)
            if existing is not None:
                logger.debug("Triple id=%s already exists — skipping", triple.id)
                continue

            # Semantic conflict scan: same subject+predicate with current truth.
            all_triple_docs = await self._docs.query(filters={"_type": _TYPE_TRIPLE}, limit=10_000)
            same_slot_docs = [
                doc
                for doc in all_triple_docs
                if doc.get("current", True)
                and isinstance(doc.get("subject"), dict)
                and doc["subject"].get("entity_id") == str(triple.subject.entity_id)
                and doc.get("predicate") == triple.predicate
            ]

            matching_doc = next(
                (
                    doc
                    for doc in same_slot_docs
                    if _object_identity(SemanticTriple.model_validate(doc).object)
                    == _object_identity(triple.object)
                ),
                None,
            )
            if matching_doc is not None:
                merged_sources = list(
                    dict.fromkeys(
                        [
                            *matching_doc.get("source_episodes", []),
                            *[str(item) for item in triple.source_episodes],
                        ]
                    )
                )
                matching_doc["source_episodes"] = merged_sources
                matching_doc["confidence"] = max(
                    float(matching_doc.get("confidence", 0.0) or 0.0),
                    triple.confidence,
                )
                matching_doc["last_confirmed"] = now.isoformat()
                await self._docs.put(UUID(str(matching_doc["id"])), matching_doc)
                written += 1
                continue

            contradiction_group = (
                f"{triple.subject.entity_id}:{triple.predicate}"
                if same_slot_docs
                else None
            )
            superseded_ids: list[str] = []
            for doc in same_slot_docs:
                doc["current"] = False
                doc["valid_to"] = now.isoformat()
                doc["superseded_by"] = str(triple.id)
                doc["contradiction_group"] = contradiction_group
                await self._docs.put(UUID(str(doc["id"])), doc)
                superseded_ids.append(str(doc["id"]))

            try:
                # Ensure subject entity node exists.
                await self._ensure_entity_node(triple.subject)

                # Resolve the object side.
                if isinstance(triple.object, EntityRef):
                    await self._ensure_entity_node(triple.object)
                    target_id = triple.object.entity_id
                    object_name = triple.object.name
                else:
                    # String literal — create a synthetic node.
                    target_id = await self._ensure_literal_node(triple.object)
                    object_name = triple.object

                # Add directed edge in the knowledge graph.
                await self._graph.add_edge(
                    source_id=triple.subject.entity_id,
                    target_id=target_id,
                    edge_type=triple.predicate,
                    properties={
                        "triple_id": str(triple.id),
                        "confidence": triple.confidence,
                    },
                )

                # Persist full triple document before indexing it so startup
                # repair can deterministically rebuild vectors from documents.
                doc = triple.model_dump(mode="json")
                doc["_type"] = _TYPE_TRIPLE
                doc["current"] = True
                doc["valid_from"] = now.isoformat()
                doc["valid_to"] = None
                doc["supersedes"] = superseded_ids
                doc["superseded_by"] = None
                doc["contradiction_group"] = contradiction_group
                await self._docs.put(triple_doc_id, doc)

                # Index the triple embedding when available.
                if triple.embedding is not None:
                    metadata: dict[str, Any] = {
                        "triple_id": str(triple.id),
                        "predicate": triple.predicate,
                        "subject_name": triple.subject.name,
                        "object_name": object_name,
                        "_type": _TYPE_TRIPLE,
                    }
                    await self._vectors.insert(
                        id=triple.id,
                        embedding=triple.embedding,
                        metadata=metadata,
                    )

                written += 1
                logger.debug(
                    "Upserted triple id=%s predicate=%r subject=%r",
                    triple.id,
                    triple.predicate,
                    triple.subject.name,
                )

            except Exception as exc:
                raise MemoryError(
                    f"Failed to upsert triple id={triple.id}: {exc}"
                ) from exc

        logger.info("upsert_triples: wrote %d/%d triples", written, len(triples))
        return written

    async def resolve_entity(
        self,
        name: str,
        embedding: list[float] | None = None,
    ) -> Entity | None:
        """Look up a canonical Entity by name or nearest embedding.

        First tries exact canonical_name match via DocumentStore.  If that
        misses and an embedding is supplied, falls back to vector similarity
        search with a conservative threshold (0.85) to prevent false merges.

        Returns
        -------
        Entity | None
            The best-matching entity or ``None`` if nothing is close enough.
        """
        try:
            await self._ensure_vector_doc_consistency()
            # Exact name match.
            docs = await self._docs.query(
                filters={"_type": _TYPE_ENTITY, "canonical_name": name},
                limit=1,
            )
            if docs:
                return Entity.model_validate(docs[0])

            # Vector similarity fallback.
            if embedding is not None:
                results = await self._vectors.search(
                    query_embedding=embedding,
                    top_k=5,
                    filters={"_type": _TYPE_ENTITY},
                )
                for result in results:
                    if result.score > 0.85:
                        entity_id_str = result.metadata.get("entity_id")
                        if entity_id_str is None:
                            continue
                        doc = await self._docs.get(UUID(entity_id_str))
                        if doc is not None and doc.get("_type") == _TYPE_ENTITY:
                            return Entity.model_validate(doc)

            return None

        except Exception as exc:
            raise RetrievalError(
                f"Entity resolution failed for name={name!r}: {exc}"
            ) from exc

    async def retrieve_by_entity(
        self,
        entity_ref: EntityRef,
        hops: int = 1,
    ) -> list[SemanticTriple]:
        """Return all triples within *hops* of *entity_ref* in the graph.

        Uses graph neighbour traversal to enumerate reachable nodes, then
        loads the connecting triple documents from the document store.
        """
        try:
            neighbors = await self._graph.get_neighbors(
                node_id=entity_ref.entity_id,
                max_hops=hops,
            )

            triples: list[SemanticTriple] = []
            seen_triple_ids: set[str] = set()

            # Collect all entity IDs to search for (seed + neighbors)
            entity_ids_to_search = {str(entity_ref.entity_id)}
            for neighbor in neighbors:
                entity_ids_to_search.add(str(neighbor.id))

            # Load all triples and filter in Python since DocumentStore
            # doesn't support nested dict matching.
            all_triple_docs = await self._docs.query(
                filters={"_type": _TYPE_TRIPLE},
                limit=10_000,
            )
            for doc in all_triple_docs:
                tid = doc.get("id")
                if tid and tid not in seen_triple_ids:
                    # Check if subject or object entity_id matches any of our targets
                    subject = doc.get("subject", {})
                    obj = doc.get("object", {})
                    subject_eid = subject.get("entity_id", "") if isinstance(subject, dict) else ""
                    object_eid = obj.get("entity_id", "") if isinstance(obj, dict) else ""
                    if subject_eid in entity_ids_to_search or object_eid in entity_ids_to_search:
                        seen_triple_ids.add(tid)
                        triples.append(SemanticTriple.model_validate(doc))

            triples.sort(
                key=lambda triple: (int(triple.current), triple.confidence, triple.last_confirmed),
                reverse=True,
            )

            logger.debug(
                "retrieve_by_entity entity=%s hops=%d found=%d triples",
                entity_ref.entity_id,
                hops,
                len(triples),
            )
            return triples

        except MemoryError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"retrieve_by_entity failed for entity_id={entity_ref.entity_id}: {exc}"
            ) from exc

    async def retrieve_by_similarity(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SemanticTriple]:
        """Find triples whose stored embedding is closest to *embedding*.

        Queries the vector store and hydrates full triple objects from the
        document store.  Silently skips results whose document has been
        deleted (e.g. pruned by maintenance).
        """
        try:
            await self._ensure_vector_doc_consistency()
            results = await self._vectors.search(
                query_embedding=embedding,
                top_k=top_k,
                filters={"_type": _TYPE_TRIPLE},
            )

            triples: list[SemanticTriple] = []
            for result in results:
                triple_id_str = result.metadata.get("triple_id")
                if triple_id_str is None:
                    continue
                doc = await self._docs.get(UUID(triple_id_str))
                if doc is not None and doc.get("_type") == _TYPE_TRIPLE:
                    triples.append(SemanticTriple.model_validate(doc))

            triples.sort(
                key=lambda triple: (int(triple.current), triple.confidence, triple.last_confirmed),
                reverse=True,
            )

            logger.debug("retrieve_by_similarity top_k=%d found=%d", top_k, len(triples))
            return triples

        except Exception as exc:
            raise RetrievalError(
                f"retrieve_by_similarity failed: {exc}"
            ) from exc

    async def spreading_activation(
        self,
        seed_entities: list[EntityRef],
        max_hops: int = 2,
    ) -> list[RankedNode]:
        """Propagate activation from *seed_entities* via Personalised PageRank.

        Returns nodes ranked by descending activation score, mirroring how
        the brain's associative network amplifies concepts related to the
        current focus of attention.
        """
        if not seed_entities:
            return []

        try:
            seed_ids = [e.entity_id for e in seed_entities]
            ranked = await self._graph.run_pagerank(seed_ids=seed_ids)
            logger.debug(
                "spreading_activation seeds=%d returned=%d ranked nodes",
                len(seed_ids),
                len(ranked),
            )
            return ranked

        except Exception as exc:
            raise RetrievalError(
                f"spreading_activation failed: {exc}"
            ) from exc

    async def get_community(self, community_id: UUID) -> Community | None:
        """Retrieve a detected concept community by its UUID."""
        try:
            doc = await self._docs.get(self._community_doc_id(community_id))
            if doc is None or doc.get("_type") != _TYPE_COMMUNITY:
                return None
            return Community.model_validate(doc)
        except Exception as exc:
            raise RetrievalError(
                f"get_community failed for id={community_id}: {exc}"
            ) from exc

    async def build_raptor_index(self) -> int:
        """Build a RAPTOR hierarchical cluster tree over all stored triples.

        The algorithm proceeds level by level:

        1. Load every triple that carries an embedding from the document store.
        2. Cluster the embeddings into ``ceil(n / 5)`` groups using cosine-
           similarity k-means (no external dependencies — stdlib only).
        3. Generate a natural-language summary for each cluster via the LLM
           provider (or plain concatenation if none is configured).
        4. Embed each summary and create a :class:`~mnemon.core.models.SemanticCluster`
           at level 0, persisting it to the document store and vector store.
        5. Recursively cluster the level-N clusters to produce level-(N+1)
           until ``config.raptor.max_levels`` is reached or only one cluster
           remains at the current level.

        Returns
        -------
        int
            Total number of :class:`~mnemon.core.models.SemanticCluster` objects
            created across all levels.
        """
        logger.info("build_raptor_index starting")
        total_created = 0

        try:
            # ----------------------------------------------------------------
            # Level 0 — cluster raw triples
            # ----------------------------------------------------------------
            triple_docs = await self._docs.query(
                filters={"_type": _TYPE_TRIPLE},
                limit=100_000,
            )

            # Keep only triples that have an embedding to cluster on.
            embedded_triples: list[SemanticTriple] = []
            for doc in triple_docs:
                if doc.get("embedding") is not None:
                    embedded_triples.append(SemanticTriple.model_validate(doc))

            if not embedded_triples:
                logger.info("build_raptor_index: no embedded triples found — skipping")
                return 0

            n = len(embedded_triples)
            k = max(1, math.ceil(n / 5))
            logger.debug(
                "build_raptor_index level=0 triples=%d k=%d", n, k
            )

            triple_embeddings = [t.embedding for t in embedded_triples]  # type: ignore[misc]
            assignments = self._kmeans_cluster(triple_embeddings, k)

            # Group triples by cluster assignment.
            cluster_groups: dict[int, list[SemanticTriple]] = {}
            for triple, assignment in zip(embedded_triples, assignments, strict=False):
                cluster_groups.setdefault(assignment, []).append(triple)

            current_level_clusters: list[SemanticCluster] = []

            for group in cluster_groups.values():
                summary = await self._summarise_triples(group)
                try:
                    summary_embedding = await self._embedder.embed(summary)
                except Exception:
                    summary_embedding = None
                    logger.debug(
                        "Could not embed cluster summary — cluster stored without embedding"
                    )

                cluster = SemanticCluster(
                    id=uuid4(),
                    level=0,
                    summary=summary,
                    embedding=summary_embedding,
                    children=[],
                    member_triples=[t.id for t in group],
                )
                doc = cluster.model_dump(mode="json")
                doc["_type"] = _TYPE_CLUSTER
                await self._docs.put(cluster.id, doc)

                if summary_embedding is not None:
                    await self._vectors.insert(
                        id=cluster.id,
                        embedding=summary_embedding,
                        metadata={
                            "_type": _TYPE_CLUSTER,
                            "cluster_id": str(cluster.id),
                            "level": cluster.level,
                        },
                    )

                current_level_clusters.append(cluster)
                total_created += 1

            logger.info(
                "build_raptor_index level=0 created=%d clusters",
                len(current_level_clusters),
            )

            # ----------------------------------------------------------------
            # Levels 1..max_levels — recursively cluster previous-level clusters
            # ----------------------------------------------------------------
            max_levels = self._config.raptor.max_levels
            for level in range(1, max_levels):
                if len(current_level_clusters) <= 1:
                    logger.debug(
                        "build_raptor_index stopping early at level=%d (only 1 cluster)",
                        level,
                    )
                    break

                embeddable = [
                    c for c in current_level_clusters if c.embedding is not None
                ]
                if not embeddable:
                    logger.debug(
                        "build_raptor_index stopping at level=%d — no embeddable clusters",
                        level,
                    )
                    break

                n_clusters = len(embeddable)
                k_clusters = max(1, math.ceil(n_clusters / 5))
                logger.debug(
                    "build_raptor_index level=%d clusters=%d k=%d",
                    level,
                    n_clusters,
                    k_clusters,
                )

                cluster_embeddings = [c.embedding for c in embeddable]  # type: ignore[misc]
                assignments = self._kmeans_cluster(cluster_embeddings, k_clusters)

                group_map: dict[int, list[SemanticCluster]] = {}
                for cluster, assignment in zip(embeddable, assignments, strict=False):
                    group_map.setdefault(assignment, []).append(cluster)

                next_level_clusters: list[SemanticCluster] = []
                for group_clusters in group_map.values():
                    summary = await self._summarise_clusters(group_clusters)
                    try:
                        summary_embedding = await self._embedder.embed(summary)
                    except Exception:
                        summary_embedding = None
                        logger.debug(
                            "Could not embed level-%d cluster summary", level
                        )

                    parent = SemanticCluster(
                        id=uuid4(),
                        level=level,
                        summary=summary,
                        embedding=summary_embedding,
                        children=[c.id for c in group_clusters],
                        member_triples=[],
                    )
                    doc = parent.model_dump(mode="json")
                    doc["_type"] = _TYPE_CLUSTER
                    await self._docs.put(parent.id, doc)

                    if summary_embedding is not None:
                        await self._vectors.insert(
                            id=parent.id,
                            embedding=summary_embedding,
                            metadata={
                                "_type": _TYPE_CLUSTER,
                                "cluster_id": str(parent.id),
                                "level": parent.level,
                            },
                        )

                    next_level_clusters.append(parent)
                    total_created += 1

                logger.info(
                    "build_raptor_index level=%d created=%d clusters",
                    level,
                    len(next_level_clusters),
                )
                current_level_clusters = next_level_clusters

        except Exception as exc:
            raise MemoryError(f"build_raptor_index failed: {exc}") from exc

        logger.info("build_raptor_index complete — total_clusters=%d", total_created)
        return total_created

    async def retrieve_raptor(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SemanticCluster]:
        """Retrieve the most relevant RAPTOR clusters for *query_embedding*.

        Searches cluster embeddings in the vector store and hydrates full
        :class:`~mnemon.core.models.SemanticCluster` objects from the document
        store.  Results are returned in descending similarity order.

        Parameters
        ----------
        query_embedding:
            Dense query vector to compare against stored cluster embeddings.
        top_k:
            Maximum number of clusters to return.

        Returns
        -------
        list[SemanticCluster]
            Best-matching clusters, ordered by descending similarity score.
        """
        try:
            results = await self._vectors.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters={"_type": _TYPE_CLUSTER},
            )

            clusters: list[SemanticCluster] = []
            for result in results:
                cluster_id_str = result.metadata.get("cluster_id")
                if cluster_id_str is None:
                    continue
                doc = await self._docs.get(UUID(cluster_id_str))
                if doc is not None and doc.get("_type") == _TYPE_CLUSTER:
                    clusters.append(SemanticCluster.model_validate(doc))

            logger.debug(
                "retrieve_raptor top_k=%d found=%d clusters", top_k, len(clusters)
            )
            return clusters

        except Exception as exc:
            raise RetrievalError(f"retrieve_raptor failed: {exc}") from exc

    async def run_maintenance(self) -> None:
        """Background maintenance: community detection, confidence decay, pruning.

        Steps performed in order:

        1. Run community detection on the knowledge graph (Leiden / Louvain).
        2. Create or update :class:`~mnemon.core.models.Community` objects in
           the document store for each detected partition.
        3. Apply confidence decay to triples whose ``last_confirmed`` timestamp
           has fallen outside ``config.decay.confirmation_window`` days.
        4. Soft-delete triples whose confidence drops below 0.05.
        """
        logger.info("SemanticMemoryStore.run_maintenance starting")

        try:
            # ----------------------------------------------------------------
            # Step 1 & 2 — community detection
            # ----------------------------------------------------------------
            communities = await self._graph.run_community_detection(
                algorithm=self._config.community_detection.algorithm,
                resolution=self._config.community_detection.resolution,
            )
            logger.info("Community detection found %d communities", len(communities))

            for member_ids in communities:
                community_id = uuid4()
                community = Community(
                    id=community_id,
                    name=f"Community-{community_id}",
                    member_entities=member_ids,
                    last_updated=datetime.now(UTC),
                )
                doc = community.model_dump(mode="json")
                doc["_type"] = _TYPE_COMMUNITY
                await self._docs.put(self._community_doc_id(community_id), doc)

            # ----------------------------------------------------------------
            # Step 3 — RAPTOR hierarchical index
            # ----------------------------------------------------------------
            raptor_clusters = 0
            if self._config.raptor.enabled:
                raptor_clusters = await self.build_raptor_index()

            # ----------------------------------------------------------------
            # Step 4 & 5 — confidence decay and pruning
            # ----------------------------------------------------------------
            decay_epsilon = self._config.decay.epsilon
            confirmation_window_days = self._config.decay.confirmation_window
            cutoff = datetime.now(UTC) - timedelta(days=confirmation_window_days)

            triple_docs = await self._docs.query(
                filters={"_type": _TYPE_TRIPLE},
                limit=10_000,
            )

            pruned = 0
            decayed = 0
            for doc in triple_docs:
                last_confirmed_raw = doc.get("last_confirmed")
                if last_confirmed_raw is None:
                    continue

                last_confirmed = datetime.fromisoformat(last_confirmed_raw)
                if last_confirmed.tzinfo is None:
                    last_confirmed = last_confirmed.replace(tzinfo=UTC)

                if last_confirmed >= cutoff:
                    continue  # Still within confirmation window; no decay.

                confidence = float(doc.get("confidence", 1.0))
                new_confidence = confidence * (1.0 - decay_epsilon)
                doc["confidence"] = new_confidence

                triple_id_str = doc.get("id")
                if triple_id_str is None:
                    continue

                triple_id = UUID(triple_id_str)

                if new_confidence < 0.05:
                    # Soft-delete: remove from all stores.
                    await self._docs.delete(triple_id)
                    with contextlib.suppress(Exception):
                        await self._vectors.delete(triple_id)
                    pruned += 1
                    logger.debug("Pruned low-confidence triple id=%s", triple_id)
                else:
                    await self._docs.put(triple_id, doc)
                    decayed += 1

            logger.info(
                "run_maintenance complete — communities=%d raptor_clusters=%d "
                "decayed=%d pruned=%d",
                len(communities),
                raptor_clusters,
                decayed,
                pruned,
            )

        except Exception as exc:
            raise MemoryError(f"run_maintenance failed: {exc}") from exc
