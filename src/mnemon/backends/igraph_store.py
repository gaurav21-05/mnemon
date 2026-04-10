"""
igraph-backed in-process graph store with JSON persistence.

Brain analog: The neocortical association network where concepts are linked by
semantic relationships. igraph provides efficient in-process graph algorithms
(PageRank, community detection) that model spreading activation across the
knowledge graph — mirroring how pattern activation spreads through cortical
columns during associative recall.

All mutating operations persist the graph to disk as a custom JSON format,
ensuring the semantic knowledge network survives process restarts.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from mnemon.core.interfaces import GraphNode, GraphStore, RankedNode

__all__ = ["IGraphGraphStore"]

logger = logging.getLogger(__name__)


class IGraphGraphStore(GraphStore):
    """igraph-backed implementation of the GraphStore ABC.

    Stores the knowledge graph as a directed igraph.Graph in process memory,
    persisted to *graph_path* as a custom JSON file. All write operations
    save immediately to disk.

    Usage::

        store = IGraphGraphStore(graph_path="~/.mnemon/graphs/semantic.json")
        await store.initialize()
        await store.add_node(uuid, ["Concept"], {"name": "Python"})
        nodes = await store.get_neighbors(uuid, max_hops=2)
    """

    def __init__(self, graph_path: str) -> None:
        self._graph_path = graph_path
        self._graph: Any = None           # igraph.Graph(directed=True)
        self._uuid_to_vtx: dict[str, int] = {}   # str(uuid) -> vertex index

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load existing graph if file exists, else create parent dirs."""
        import igraph
        Path(self._graph_path).parent.mkdir(parents=True, exist_ok=True)

        path = Path(self._graph_path)
        if path.exists():
            try:
                self._load()
                logger.debug(
                    "IGraphGraphStore loaded graph=%s vertices=%d edges=%d",
                    self._graph_path,
                    self._graph.vcount(),
                    self._graph.ecount(),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load igraph graph from %s: %s — starting fresh.",
                    self._graph_path,
                    exc,
                )
                self._graph = igraph.Graph(directed=True)
                self._uuid_to_vtx = {}
        else:
            self._graph = igraph.Graph(directed=True)
            self._uuid_to_vtx = {}
            logger.debug(
                "IGraphGraphStore: no existing graph at %s — created empty graph.",
                self._graph_path,
            )

    # ------------------------------------------------------------------
    # GraphStore ABC
    # ------------------------------------------------------------------

    async def add_node(
        self,
        node_id: UUID,
        labels: list[str],
        properties: dict[str, Any],
    ) -> None:
        """Insert or upsert a node with the given labels and properties."""
        str_id = str(node_id)
        props = dict(properties)
        props["uuid"] = str_id
        props["labels"] = labels

        if str_id in self._uuid_to_vtx:
            vtx_idx = self._uuid_to_vtx[str_id]
            vtx = self._graph.vs[vtx_idx]
            for k, v in props.items():
                vtx[k] = v
        else:
            self._graph.add_vertex(**props)
            vtx_idx = self._graph.vcount() - 1
            self._uuid_to_vtx[str_id] = vtx_idx

        self._save()
        logger.debug("IGraphGraphStore.add_node id=%s labels=%s", node_id, labels)

    async def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a directed edge between two existing nodes."""
        src = self._uuid_to_vtx.get(str(source_id))
        tgt = self._uuid_to_vtx.get(str(target_id))
        if src is None or tgt is None:
            logger.debug(
                "IGraphGraphStore.add_edge: missing node(s) src=%s tgt=%s — skipping.",
                source_id,
                target_id,
            )
            return

        edge_attrs: dict[str, Any] = dict(properties or {})
        edge_attrs["type"] = edge_type
        self._graph.add_edge(src, tgt, **edge_attrs)
        self._save()
        logger.debug(
            "IGraphGraphStore.add_edge src=%s tgt=%s type=%s", source_id, target_id, edge_type
        )

    async def get_node(self, node_id: UUID) -> dict[str, Any] | None:
        """Fetch a single node by its UUID."""
        vtx_idx = self._uuid_to_vtx.get(str(node_id))
        if vtx_idx is None:
            return None
        vtx = self._graph.vs[vtx_idx]
        return dict(vtx.attributes())

    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: str | None = None,
        direction: str = "out",
        max_hops: int = 1,
    ) -> list[GraphNode]:
        """Return nodes reachable from node_id within max_hops steps via BFS."""
        str_id = str(node_id)
        start_idx = self._uuid_to_vtx.get(str_id)
        if start_idx is None:
            return []

        if direction == "out":
            mode = "out"
        elif direction == "in":
            mode = "in"
        else:
            mode = "all"

        visited: set[int] = {start_idx}
        frontier: list[int] = [start_idx]
        results: list[GraphNode] = []

        for _ in range(max_hops):
            next_frontier: list[int] = []
            for vtx_idx in frontier:
                if direction == "out":
                    neighbors = self._graph.successors(vtx_idx)
                elif direction == "in":
                    neighbors = self._graph.predecessors(vtx_idx)
                else:
                    neighbors = self._graph.neighbors(vtx_idx, mode=mode)

                for nbr_idx in neighbors:
                    if nbr_idx in visited:
                        continue

                    if edge_type is not None:
                        # Filter edges by type
                        if direction == "out":
                            edges = self._graph.es.select(_source=vtx_idx, _target=nbr_idx)
                        elif direction == "in":
                            edges = self._graph.es.select(_source=nbr_idx, _target=vtx_idx)
                        else:
                            edges = self._graph.es.select(
                                _between=([vtx_idx], [nbr_idx])
                            )
                        if not any(e["type"] == edge_type for e in edges):
                            continue

                    visited.add(nbr_idx)
                    next_frontier.append(nbr_idx)

                    nbr_vtx = self._graph.vs[nbr_idx]
                    attrs = dict(nbr_vtx.attributes())
                    uuid_str = attrs.get("uuid", "")
                    labels = attrs.pop("labels", []) if "labels" in attrs else []
                    if "labels" not in attrs:
                        # labels was already popped
                        pass
                    else:
                        attrs.pop("labels", None)

                    try:
                        node_uuid = UUID(uuid_str)
                        results.append(GraphNode(id=node_uuid, labels=labels, properties=attrs))
                    except (ValueError, AttributeError):
                        pass

            frontier = next_frontier
            if not frontier:
                break

        return results

    async def run_pagerank(
        self,
        seed_ids: list[UUID],
        damping: float = 0.85,
        max_iterations: int = 100,
    ) -> list[RankedNode]:
        """Compute Personalised PageRank seeded from seed_ids."""
        if self._graph.vcount() == 0:
            return []

        seed_indices = [
            self._uuid_to_vtx[str(sid)]
            for sid in seed_ids
            if str(sid) in self._uuid_to_vtx
        ]

        try:
            scores = self._graph.personalized_pagerank(
                vertices=seed_indices,
                damping=damping,
                directed=True,
                reset_vertices=seed_indices,
            )
        except Exception as exc:
            logger.debug("IGraphGraphStore.run_pagerank fallback to global PR: %s", exc)
            scores = self._graph.pagerank(damping=damping, directed=True)

        ranked: list[RankedNode] = []
        for vtx_idx, score in enumerate(scores):
            vtx = self._graph.vs[vtx_idx]
            attrs = dict(vtx.attributes())
            uuid_str = attrs.get("uuid", "")
            try:
                node_uuid = UUID(uuid_str)
                ranked.append(RankedNode(id=node_uuid, score=float(score), properties=attrs))
            except (ValueError, AttributeError):
                pass

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    async def run_community_detection(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
    ) -> list[list[UUID]]:
        """Partition the graph into communities of related concepts."""
        if self._graph.vcount() == 0:
            return []

        try:
            if algorithm == "leiden":
                clustering = self._graph.community_leiden(resolution_parameter=resolution)
            else:
                clustering = self._graph.community_multilevel()
        except Exception as exc:
            logger.warning("IGraphGraphStore.run_community_detection failed: %s", exc)
            return []

        communities: list[list[UUID]] = []
        for community in clustering:
            members: list[UUID] = []
            for vtx_idx in community:
                uuid_str = self._graph.vs[vtx_idx].attributes().get("uuid", "")
                with contextlib.suppress(ValueError, AttributeError):
                    members.append(UUID(uuid_str))
            if members:
                communities.append(members)

        return communities

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """No-op stub — igraph does not support Cypher queries."""
        logger.debug(
            "IGraphGraphStore.query: Cypher not supported (query=%r) — returning [].",
            cypher[:80] if cypher else "",
        )
        return []

    async def delete_node(self, node_id: UUID) -> None:
        """Remove a node and all its incident edges from the graph."""
        str_id = str(node_id)
        vtx_idx = self._uuid_to_vtx.pop(str_id, None)
        if vtx_idx is None:
            return

        self._graph.delete_vertices(vtx_idx)

        # Rebuild UUID->vertex index mapping after deletion (indices shift)
        new_mapping: dict[str, int] = {}
        for i, vtx in enumerate(self._graph.vs):
            uuid_val = vtx.attributes().get("uuid", "")
            if uuid_val:
                new_mapping[uuid_val] = i
        self._uuid_to_vtx = new_mapping

        self._save()
        logger.debug("IGraphGraphStore.delete_node id=%s", node_id)

    async def node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        if self._graph is None:
            return 0
        return self._graph.vcount()

    async def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        if self._graph is None:
            return 0
        return self._graph.ecount()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist graph to disk as custom JSON."""
        try:
            vertices = []
            for vtx in self._graph.vs:
                attrs = dict(vtx.attributes())
                uuid_str = attrs.get("uuid", "")
                labels = attrs.pop("labels", [])
                vertices.append({
                    "uuid": uuid_str,
                    "labels": labels,
                    "properties": attrs,
                })

            edges = []
            for edge in self._graph.es:
                attrs = dict(edge.attributes())
                edge_type = attrs.pop("type", "")
                src_uuid = self._graph.vs[edge.source].attributes().get("uuid", "")
                tgt_uuid = self._graph.vs[edge.target].attributes().get("uuid", "")
                edges.append({
                    "source_uuid": src_uuid,
                    "target_uuid": tgt_uuid,
                    "type": edge_type,
                    "properties": attrs,
                })

            data = {"vertices": vertices, "edges": edges}
            with open(self._graph_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception as exc:
            logger.warning("IGraphGraphStore._save failed: %s", exc)

    def _load(self) -> None:
        """Load graph from custom JSON format."""
        import igraph
        with open(self._graph_path, encoding="utf-8") as fh:
            data = json.load(fh)

        self._graph = igraph.Graph(directed=True)
        self._uuid_to_vtx = {}

        for v in data.get("vertices", []):
            uuid_str = v.get("uuid", "")
            labels = v.get("labels", [])
            props = dict(v.get("properties", {}))
            props["uuid"] = uuid_str
            props["labels"] = labels
            self._graph.add_vertex(**props)
            vtx_idx = self._graph.vcount() - 1
            if uuid_str:
                self._uuid_to_vtx[uuid_str] = vtx_idx

        for e in data.get("edges", []):
            src_uuid = e.get("source_uuid", "")
            tgt_uuid = e.get("target_uuid", "")
            edge_type = e.get("type", "")
            props = dict(e.get("properties", {}))
            props["type"] = edge_type

            src_idx = self._uuid_to_vtx.get(src_uuid)
            tgt_idx = self._uuid_to_vtx.get(tgt_uuid)
            if src_idx is not None and tgt_idx is not None:
                self._graph.add_edge(src_idx, tgt_idx, **props)
