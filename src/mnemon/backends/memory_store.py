"""
Pure-Python in-memory storage backends for VectorStore, DocumentStore, and GraphStore.

Brain analog: The most primitive form of biological memory — transient synaptic
potentiation held entirely in RAM.  These backends require no external dependencies
and are ideal for unit testing, rapid prototyping, and single-process development.
They mirror the short-lived hippocampal working buffer before consolidation to
stable long-term storage, and share its defining limitation: all state is lost
when the process exits.

All three classes implement their respective abstract interfaces from
``mnemon.core.interfaces`` and are registered by default under the backend key
``"memory"`` in ``ModuleRegistry.from_config()``.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any
from uuid import UUID

from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import (
    DocumentStore,
    GraphNode,
    GraphStore,
    RankedNode,
    VectorItem,
    VectorSearchResult,
    VectorStore,
)

__all__ = [
    "InMemoryVectorStore",
    "InMemoryDocumentStore",
    "InMemoryGraphStore",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns 0.0 for zero-magnitude vectors to avoid division by zero.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches_filters(document: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Return True if *document* satisfies all equality predicates in *filters*."""
    if not filters:
        return True
    for key, expected in filters.items():
        if document.get(key) != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStore):
    """Exact cosine-similarity vector store backed by a plain Python dict.

    Brain analog: A simplified version of the hippocampal index that performs
    exhaustive pattern-matching across all stored engrams.  Unlike biological
    pattern completion (which uses sparse distributed representations and
    attractor dynamics), this implementation scans every vector on every query —
    acceptable for small-scale development but not for production workloads.

    Thread-safety: Not thread-safe.  Wrap with an asyncio.Lock if concurrent
    writes from multiple coroutines are expected.
    """

    def __init__(self, config: MnemonConfig) -> None:
        # _store: id -> (embedding, metadata)
        self._store: dict[UUID, tuple[list[float], dict[str, Any]]] = {}
        logger.debug("InMemoryVectorStore initialised")

    async def insert(
        self,
        id: UUID,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Store *embedding* and *metadata* under *id*."""
        self._store[id] = (embedding, dict(metadata))
        logger.debug("VectorStore.insert id=%s dim=%d", id, len(embedding))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Scan all stored vectors, apply metadata filters, return top_k by cosine score."""
        scored: list[tuple[float, UUID, dict[str, Any]]] = []
        for vec_id, (embedding, metadata) in self._store.items():
            if not _matches_filters(metadata, filters):
                continue
            score = _cosine_similarity(query_embedding, embedding)
            scored.append((score, vec_id, metadata))

        scored.sort(key=lambda t: t[0], reverse=True)
        results = [
            VectorSearchResult(id=vec_id, score=score, metadata=metadata)
            for score, vec_id, metadata in scored[:top_k]
        ]
        logger.debug(
            "VectorStore.search top_k=%d candidates=%d returned=%d",
            top_k,
            len(scored),
            len(results),
        )
        return results

    async def delete(self, id: UUID) -> None:
        """Remove the vector identified by *id*; no-op if not present."""
        self._store.pop(id, None)
        logger.debug("VectorStore.delete id=%s", id)

    async def update(
        self,
        id: UUID,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Overwrite the embedding and metadata for an existing entry."""
        self._store[id] = (embedding, dict(metadata))
        logger.debug("VectorStore.update id=%s", id)

    async def bulk_insert(self, items: list[VectorItem]) -> None:
        """Insert multiple vectors in a single call."""
        for item in items:
            await self.insert(item.id, item.embedding, item.metadata)
        logger.debug("VectorStore.bulk_insert count=%d", len(items))

    async def count(self) -> int:
        """Return the total number of vectors currently stored."""
        return len(self._store)

    async def clear(self) -> None:
        """Remove all stored vectors."""
        self._store.clear()
        logger.debug("VectorStore.clear")


# ---------------------------------------------------------------------------
# InMemoryDocumentStore
# ---------------------------------------------------------------------------


class InMemoryDocumentStore(DocumentStore):
    """Key-value document store backed by a plain Python dict.

    Brain analog: The raw synaptic weight matrix before consolidation — a
    flat, addressable store of serialised memory traces.  Every read and write
    is O(1); full-table scans are O(n) and acceptable for development-scale
    data sets.

    Documents are stored as shallow copies; callers should not mutate returned
    dicts directly if isolation is required.
    """

    def __init__(self, config: MnemonConfig) -> None:
        self._store: dict[UUID, dict[str, Any]] = {}
        logger.debug("InMemoryDocumentStore initialised")

    async def put(self, id: UUID, document: dict[str, Any]) -> None:
        """Insert or replace *document* under *id*."""
        self._store[id] = dict(document)
        logger.debug("DocumentStore.put id=%s", id)

    async def get(self, id: UUID) -> dict[str, Any] | None:
        """Return the document for *id*, or None if absent."""
        doc = self._store.get(id)
        return dict(doc) if doc is not None else None

    async def query(
        self,
        filters: dict[str, Any],
        sort_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Scan all documents, filter by equality, optionally sort, then page."""
        matching = [
            dict(doc)
            for doc in self._store.values()
            if _matches_filters(doc, filters)
        ]
        if sort_by is not None:
            matching.sort(key=lambda d: d.get(sort_by))  # type: ignore[return-value]
        result = matching[offset : offset + limit]
        logger.debug(
            "DocumentStore.query filters=%s sort_by=%s offset=%d limit=%d returned=%d",
            filters,
            sort_by,
            offset,
            limit,
            len(result),
        )
        return result

    async def delete(self, id: UUID) -> None:
        """Permanently remove the document identified by *id*; no-op if absent."""
        self._store.pop(id, None)
        logger.debug("DocumentStore.delete id=%s", id)

    async def bulk_put(self, items: list[tuple[UUID, dict[str, Any]]]) -> None:
        """Insert or replace multiple documents in a single call."""
        for id, document in items:
            await self.put(id, document)
        logger.debug("DocumentStore.bulk_put count=%d", len(items))

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Return the count of documents matching *filters* (all if None)."""
        if not filters:
            return len(self._store)
        return sum(1 for doc in self._store.values() if _matches_filters(doc, filters))


# ---------------------------------------------------------------------------
# InMemoryGraphStore
# ---------------------------------------------------------------------------


class InMemoryGraphStore(GraphStore):
    """Directed property graph backed by in-memory adjacency dicts.

    Brain analog: A greatly simplified model of the cortical association network —
    nodes are concept representations and edges are learned semantic relationships.
    Spreading activation (Personalised PageRank) is implemented via power-iteration,
    mirroring the propagation of neural activation through cortical columns during
    associative recall.

    Edge storage:
        _out_edges[source_id] = list of (target_id, edge_type, properties)
        _in_edges[target_id]  = list of (source_id, edge_type, properties)

    Node storage:
        _nodes[node_id] = {"id": UUID, "labels": list[str], "properties": dict}
    """

    def __init__(self, config: MnemonConfig) -> None:
        self._nodes: dict[UUID, dict[str, Any]] = {}
        self._out_edges: dict[UUID, list[tuple[UUID, str, dict[str, Any]]]] = {}
        self._in_edges: dict[UUID, list[tuple[UUID, str, dict[str, Any]]]] = {}
        logger.debug("InMemoryGraphStore initialised")

    async def add_node(
        self,
        node_id: UUID,
        labels: list[str],
        properties: dict[str, Any],
    ) -> None:
        """Insert or upsert a node; merges properties if the node already exists."""
        if node_id in self._nodes:
            self._nodes[node_id]["labels"] = list(labels)
            self._nodes[node_id]["properties"].update(properties)
        else:
            self._nodes[node_id] = {
                "id": node_id,
                "labels": list(labels),
                "properties": dict(properties),
            }
            self._out_edges.setdefault(node_id, [])
            self._in_edges.setdefault(node_id, [])
        logger.debug("GraphStore.add_node id=%s labels=%s", node_id, labels)

    async def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a directed edge from *source_id* to *target_id*."""
        props = dict(properties) if properties else {}
        self._out_edges.setdefault(source_id, []).append((target_id, edge_type, props))
        self._in_edges.setdefault(target_id, []).append((source_id, edge_type, props))
        logger.debug(
            "GraphStore.add_edge %s -[%s]-> %s", source_id, edge_type, target_id
        )

    async def get_node(self, node_id: UUID) -> dict[str, Any] | None:
        """Return the node dict for *node_id*, or None if absent."""
        node = self._nodes.get(node_id)
        if node is None:
            return None
        return {
            "id": node["id"],
            "labels": list(node["labels"]),
            "properties": dict(node["properties"]),
        }

    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: str | None = None,
        direction: str = "out",
        max_hops: int = 1,
    ) -> list[GraphNode]:
        """BFS traversal returning all nodes reachable within *max_hops* steps."""
        if node_id not in self._nodes:
            return []

        visited: set[UUID] = {node_id}
        frontier: deque[tuple[UUID, int]] = deque([(node_id, 0)])
        result: list[GraphNode] = []

        while frontier:
            current_id, hops = frontier.popleft()
            if hops >= max_hops:
                continue

            if direction in ("out", "both"):
                for (neighbor_id, etype, _props) in self._out_edges.get(current_id, []):
                    if edge_type is not None and etype != edge_type:
                        continue
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        frontier.append((neighbor_id, hops + 1))
                        node = self._nodes.get(neighbor_id)
                        if node is not None:
                            result.append(
                                GraphNode(
                                    id=node["id"],
                                    labels=list(node["labels"]),
                                    properties=dict(node["properties"]),
                                )
                            )

            if direction in ("in", "both"):
                for (neighbor_id, etype, _props) in self._in_edges.get(current_id, []):
                    if edge_type is not None and etype != edge_type:
                        continue
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        frontier.append((neighbor_id, hops + 1))
                        node = self._nodes.get(neighbor_id)
                        if node is not None:
                            result.append(
                                GraphNode(
                                    id=node["id"],
                                    labels=list(node["labels"]),
                                    properties=dict(node["properties"]),
                                )
                            )

        logger.debug(
            "GraphStore.get_neighbors node=%s hops=%d direction=%s returned=%d",
            node_id,
            max_hops,
            direction,
            len(result),
        )
        return result

    async def run_pagerank(
        self,
        seed_ids: list[UUID],
        damping: float = 0.85,
        max_iterations: int = 100,
    ) -> list[RankedNode]:
        """Personalised PageRank via power iteration seeded at *seed_ids*.

        Nodes not reachable from the graph are assigned score 0.  The
        personalisation vector places uniform weight on *seed_ids*.
        """
        all_nodes = list(self._nodes.keys())
        n = len(all_nodes)
        if n == 0:
            return []

        node_index: dict[UUID, int] = {nid: i for i, nid in enumerate(all_nodes)}

        # Personalisation vector: uniform weight on seeds, 1/n elsewhere
        if seed_ids:
            teleport = [0.0] * n
            valid_seeds = [sid for sid in seed_ids if sid in node_index]
            if valid_seeds:
                seed_weight = 1.0 / len(valid_seeds)
                for sid in valid_seeds:
                    teleport[node_index[sid]] = seed_weight
            else:
                teleport = [1.0 / n] * n
        else:
            teleport = [1.0 / n] * n

        # Build out-degree for normalisation
        out_degree: list[int] = [len(self._out_edges.get(nid, [])) for nid in all_nodes]

        scores = [1.0 / n] * n
        damping_comp = 1.0 - damping

        for _iteration in range(max_iterations):
            new_scores = [0.0] * n
            # Collect dangling node mass (nodes with no out-edges)
            dangling_mass = sum(
                scores[i] for i, nid in enumerate(all_nodes) if out_degree[i] == 0
            )
            dangling_contrib = dangling_mass / n

            for i, nid in enumerate(all_nodes):
                new_scores[i] = damping_comp * teleport[i] + damping * dangling_contrib

            for i, nid in enumerate(all_nodes):
                if out_degree[i] == 0:
                    continue
                share = damping * scores[i] / out_degree[i]
                for (target_id, _etype, _props) in self._out_edges.get(nid, []):
                    if target_id in node_index:
                        new_scores[node_index[target_id]] += share

            # Check convergence
            delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if delta < 1e-8:
                logger.debug("PageRank converged at iteration %d", _iteration + 1)
                break

        ranked: list[RankedNode] = [
            RankedNode(
                id=all_nodes[i],
                score=scores[i],
                properties=dict(self._nodes[all_nodes[i]]["properties"]),
            )
            for i in range(n)
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        logger.debug("GraphStore.run_pagerank seeds=%d nodes=%d", len(seed_ids), n)
        return ranked

    async def run_community_detection(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
    ) -> list[list[UUID]]:
        """Label-propagation community detection (algorithm param is ignored).

        Each node adopts the label held by the majority of its neighbours.
        Isolated nodes form singleton communities.
        """
        if not self._nodes:
            return []

        all_nodes = list(self._nodes.keys())
        # Initialise each node with its own label
        labels: dict[UUID, int] = {nid: i for i, nid in enumerate(all_nodes)}

        max_iters = 100
        for _iteration in range(max_iters):
            changed = False
            # Randomise traversal order to avoid ordering artefacts
            import random
            shuffled = list(all_nodes)
            random.shuffle(shuffled)

            for nid in shuffled:
                neighbor_ids: list[UUID] = []
                for (target, _etype, _props) in self._out_edges.get(nid, []):
                    neighbor_ids.append(target)
                for (source, _etype, _props) in self._in_edges.get(nid, []):
                    neighbor_ids.append(source)

                if not neighbor_ids:
                    continue

                # Count label frequencies among neighbours
                freq: dict[int, int] = {}
                for nb in neighbor_ids:
                    lbl = labels[nb]
                    freq[lbl] = freq.get(lbl, 0) + 1

                best_label = max(freq, key=lambda l: freq[l])
                if labels[nid] != best_label:
                    labels[nid] = best_label
                    changed = True

            if not changed:
                logger.debug("Label propagation converged at iteration %d", _iteration + 1)
                break

        # Group nodes by final label
        community_map: dict[int, list[UUID]] = {}
        for nid, lbl in labels.items():
            community_map.setdefault(lbl, []).append(nid)

        communities = list(community_map.values())
        logger.debug(
            "GraphStore.run_community_detection algorithm=%s communities=%d",
            algorithm,
            len(communities),
        )
        return communities

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Not supported by the in-memory backend."""
        raise NotImplementedError("In-memory store does not support Cypher queries")

    async def delete_node(self, node_id: UUID) -> None:
        """Remove *node_id* and all of its incident edges from the graph."""
        if node_id not in self._nodes:
            return

        # Remove all outgoing edges: clean up the target's in_edges list
        for (target_id, _etype, _props) in self._out_edges.pop(node_id, []):
            self._in_edges[target_id] = [
                (src, et, p)
                for (src, et, p) in self._in_edges.get(target_id, [])
                if src != node_id
            ]

        # Remove all incoming edges: clean up the source's out_edges list
        for (source_id, _etype, _props) in self._in_edges.pop(node_id, []):
            self._out_edges[source_id] = [
                (tgt, et, p)
                for (tgt, et, p) in self._out_edges.get(source_id, [])
                if tgt != node_id
            ]

        del self._nodes[node_id]
        logger.debug("GraphStore.delete_node id=%s", node_id)

    async def node_count(self) -> int:
        """Return the total number of nodes."""
        return len(self._nodes)

    async def edge_count(self) -> int:
        """Return the total number of directed edges."""
        return sum(len(edges) for edges in self._out_edges.values())
