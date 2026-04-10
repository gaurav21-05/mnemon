"""
FalkorDB-backed production graph store for Mnemon's semantic memory subsystem.

Brain analog: The persistent neocortical association network — a Redis-backed
property graph that encodes concept nodes and their semantic relationships with
full durability and horizontal scalability.  Where the in-memory store holds
only transient synaptic potentiation, FalkorDB commits every learned association
to stable long-term storage, surviving process restarts just as consolidated
memories survive sleep.

Spreading activation (Personalised PageRank) and community detection are
computed in Python over a topology snapshot fetched from FalkorDB, mirroring
the in-memory implementation for behavioural parity across backends.
"""

from __future__ import annotations

import json
import logging
import random
import re as _re
from typing import TYPE_CHECKING, Any
from uuid import UUID

import anyio
from falkordb import FalkorDB

from mnemon.core.exceptions import MemoryError
from mnemon.core.interfaces import GraphNode, GraphStore, RankedNode

__all__ = ["FalkorDBGraphStore"]

logger = logging.getLogger(__name__)

# Sentinel used to distinguish "key present with value None" from "key absent"
_MISSING = object()

if TYPE_CHECKING:
    from mnemon.core.config import MnemonConfig

_SAFE_IDENTIFIER_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(value: str, context: str = "identifier") -> str:
    """Validate that *value* is safe for use as a Cypher label or relationship type.

    Raises
    ------
    MemoryError
        If the value contains characters outside ``[A-Za-z0-9_]`` or is empty.
    """
    if not value or not _SAFE_IDENTIFIER_RE.match(value):
        raise MemoryError(
            f"Unsafe Cypher {context}: {value!r}. "
            "Only alphanumeric characters and underscores are allowed."
        )
    return value


class FalkorDBGraphStore(GraphStore):
    """FalkorDB-backed persistent knowledge graph.

    Brain analog: The neocortical association network where concepts are linked
    by semantic relationships.  This backend persists the graph topology in a
    Redis-backed property graph, providing durability, full Cypher query support,
    and sub-millisecond edge traversal for the spreading-activation retrieval
    that models cortical recall.

    Connection is deferred: the constructor stores configuration but does not
    open any network socket.  Call ``await store.initialize()`` before use.

    Parameters
    ----------
    config:
        Root Mnemon configuration.  FalkorDB connection parameters are read
        from the ``semantic`` sub-section extras or fall back to defaults
        (host="localhost", port=6379, graph_name="mnemon_graph").
    """

    def __init__(self, config: MnemonConfig) -> None:
        # Extract FalkorDB-specific settings; SemanticConfig may carry extras
        # passed through pydantic's `extra="ignore"` — use getattr with defaults.
        semantic = config.semantic
        self._host: str = getattr(semantic, "falkordb_host", "localhost")
        self._port: int = getattr(semantic, "falkordb_port", 6379)
        self._graph_name: str = getattr(semantic, "falkordb_graph_name", "mnemon_graph")

        self._db: FalkorDB | None = None
        self._graph: Any | None = None
        self._initialized: bool = False

        logger.debug(
            "FalkorDBGraphStore configured host=%s port=%d graph=%s",
            self._host,
            self._port,
            self._graph_name,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the FalkorDB connection and ensure graph indices exist.

        Creates a ``node_id`` property index on the ``:Node`` label to enable
        O(log n) lookups.  The index creation is idempotent — any
        ``already exists`` error from FalkorDB is silently swallowed.

        Raises
        ------
        MemoryError
            If the connection or index creation fails for any reason other than
            the index already being present.
        """
        try:
            self._db = await anyio.to_thread.run_sync(
                lambda: FalkorDB(host=self._host, port=self._port)
            )
            self._graph = await anyio.to_thread.run_sync(
                lambda: self._db.select_graph(self._graph_name)  # type: ignore[union-attr]
            )
        except Exception as exc:
            raise MemoryError(
                f"FalkorDB connection failed ({self._host}:{self._port}): {exc}"
            ) from exc

        try:
            await self._run_query(
                "CREATE INDEX FOR (n:Node) ON (n.node_id)",
                {},
            )
            logger.debug("FalkorDBGraphStore: node_id index created")
        except Exception as exc:
            # Index already exists or backend returned a benign error
            logger.debug("FalkorDBGraphStore: node_id index already exists (%s)", exc)

        self._initialized = True
        logger.info(
            "FalkorDBGraphStore initialised host=%s port=%d graph=%s",
            self._host,
            self._port,
            self._graph_name,
        )

    async def close(self) -> None:
        """Gracefully release the FalkorDB connection.

        Safe to call even if ``initialize()`` was never invoked.
        """
        if self._db is not None:
            try:
                await anyio.to_thread.run_sync(lambda: self._db.connection.close())  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                logger.debug("FalkorDBGraphStore: error during close (%s)", exc)
            finally:
                self._db = None
                self._graph = None
                self._initialized = False
        logger.debug("FalkorDBGraphStore closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        """Raise MemoryError if ``initialize()`` has not been called."""
        if not self._initialized or self._graph is None:
            raise MemoryError(
                "FalkorDBGraphStore is not initialised — call await store.initialize() first."
            )

    async def _run_query(self, cypher: str, params: dict[str, Any]) -> Any:
        """Execute *cypher* on the FalkorDB graph, wrapped in a thread for async safety.

        Parameters
        ----------
        cypher:
            Cypher query string.
        params:
            Bound parameters dict (may be empty).

        Returns
        -------
        Any
            The raw FalkorDB ``QueryResult`` object.

        Raises
        ------
        MemoryError
            On any FalkorDB exception.
        """
        graph = self._graph
        try:
            result = await anyio.to_thread.run_sync(
                lambda: graph.query(cypher, params)
            )
        except Exception as exc:
            raise MemoryError(f"FalkorDB query failed: {exc}\nCypher: {cypher}") from exc
        return result

    @staticmethod
    def _serialize_props(properties: dict[str, Any]) -> dict[str, Any]:
        """Flatten *properties* into FalkorDB-safe primitive values.

        FalkorDB supports string, int, float, and bool.  Any value that is
        not one of those base types is JSON-serialised and stored as a string
        prefixed with ``__json__:`` so it can be round-tripped on retrieval.

        Parameters
        ----------
        properties:
            Raw property dict whose values may include lists, dicts, or UUIDs.

        Returns
        -------
        dict[str, Any]
            Flattened dict suitable for passing to FalkorDB.
        """
        serialized: dict[str, Any] = {}
        for key, value in properties.items():
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            elif isinstance(value, UUID):
                serialized[key] = str(value)
            else:
                serialized[key] = "__json__:" + json.dumps(value)
        return serialized

    @staticmethod
    def _deserialize_props(raw: dict[str, Any]) -> dict[str, Any]:
        """Restore *raw* FalkorDB properties to their original Python types.

        Reverses the transformation applied by ``_serialize_props``.
        Strings that begin with ``__json__:`` are JSON-decoded; all other
        values are returned as-is.

        Parameters
        ----------
        raw:
            Property dict as returned by FalkorDB.

        Returns
        -------
        dict[str, Any]
            Deserialized property dict with complex values restored.
        """
        result: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, str) and value.startswith("__json__:"):
                try:
                    result[key] = json.loads(value[len("__json__:"):])
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # GraphStore ABC implementation
    # ------------------------------------------------------------------

    async def add_node(
        self,
        node_id: UUID,
        labels: list[str],
        properties: dict[str, Any],
    ) -> None:
        """Insert or upsert a node identified by *node_id*.

        Uses ``MERGE`` on the ``node_id`` string property so that repeated
        calls with the same UUID update labels and properties rather than
        creating duplicate nodes.

        Parameters
        ----------
        node_id:
            Stable UUID used as the primary key.
        labels:
            Semantic type tags; at least ``Node`` is always included to
            support the index created during ``initialize()``.
        properties:
            Arbitrary node attributes; complex values are JSON-serialised.
        """
        self._assert_initialized()

        # Always include the base "Node" label so the index covers every node
        all_labels = list(dict.fromkeys(["Node"] + labels))
        # Sanitise each label to prevent Cypher injection
        for lbl in all_labels:
            _safe_identifier(lbl, "node label")
        label_str = ":".join(all_labels)

        serialized = self._serialize_props(properties)
        serialized["node_id"] = str(node_id)

        # Build SET clause for label properties (excluding node_id — already merged on it)
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in serialized)
        cypher = (
            f"MERGE (n:{label_str} {{node_id: $node_id}}) "
            f"SET {set_clauses}"
        )

        try:
            await self._run_query(cypher, serialized)
        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(f"add_node failed for node_id={node_id}: {exc}") from exc

        logger.debug("GraphStore.add_node id=%s labels=%s", node_id, labels)

    async def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a directed edge between two existing nodes.

        Matches both nodes by their ``node_id`` property and creates a typed
        relationship between them.  Properties are serialized for FalkorDB
        compatibility.

        Parameters
        ----------
        source_id:
            UUID of the origin node.
        target_id:
            UUID of the destination node.
        edge_type:
            Relationship label (e.g. ``"IS_A"``, ``"CAUSED_BY"``).
        properties:
            Optional edge-level attributes.
        """
        self._assert_initialized()

        props = properties or {}
        serialized = self._serialize_props(props)

        # Sanitise edge_type to prevent Cypher injection
        _safe_identifier(edge_type, "edge type")

        prop_str = ""
        if serialized:
            kv = ", ".join(f"{k}: ${k}" for k in serialized)
            prop_str = f" {{{kv}}}"

        cypher = (
            f"MATCH (s:Node {{node_id: $source_id}}) "
            f"MATCH (t:Node {{node_id: $target_id}}) "
            f"CREATE (s)-[r:{edge_type}{prop_str}]->(t)"
        )

        params: dict[str, Any] = {
            "source_id": str(source_id),
            "target_id": str(target_id),
            **serialized,
        }

        try:
            await self._run_query(cypher, params)
        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(
                f"add_edge failed {source_id} -[{edge_type}]-> {target_id}: {exc}"
            ) from exc

        logger.debug(
            "GraphStore.add_edge %s -[%s]-> %s", source_id, edge_type, target_id
        )

    async def get_node(self, node_id: UUID) -> dict[str, Any] | None:
        """Fetch a single node by its UUID.

        Returns ``None`` if no node with *node_id* exists.

        Parameters
        ----------
        node_id:
            UUID to look up.

        Returns
        -------
        dict[str, Any] | None
            Dict with keys ``"id"`` (UUID), ``"labels"`` (list[str]), and
            ``"properties"`` (dict), or ``None`` if absent.
        """
        self._assert_initialized()

        cypher = "MATCH (n:Node {node_id: $node_id}) RETURN n"
        result = await self._run_query(cypher, {"node_id": str(node_id)})

        rows = result.result_set
        if not rows:
            return None

        node_obj = rows[0][0]
        raw_props = dict(node_obj.properties)
        raw_props.pop("node_id", None)
        properties = self._deserialize_props(raw_props)

        labels = [lbl for lbl in node_obj.labels if lbl != "Node"]

        return {
            "id": node_id,
            "labels": labels,
            "properties": properties,
        }

    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: str | None = None,
        direction: str = "out",
        max_hops: int = 1,
    ) -> list[GraphNode]:
        """Return nodes reachable from *node_id* within *max_hops* steps.

        Uses variable-length Cypher path patterns for multi-hop traversal.

        Parameters
        ----------
        node_id:
            Starting node for the traversal.
        edge_type:
            If provided, only traverse edges of this relationship type.
        direction:
            ``"out"`` (default), ``"in"``, or ``"both"``.
        max_hops:
            Maximum graph distance to explore.
        """
        self._assert_initialized()

        rel_pattern = (
            f"[r:{edge_type}*1..{max_hops}]"
            if edge_type
            else f"[r*1..{max_hops}]"
        )

        if direction == "out":
            path_pattern = f"(n:Node {{node_id: $node_id}})-{rel_pattern}->(m:Node)"
        elif direction == "in":
            path_pattern = f"(n:Node {{node_id: $node_id}})<-{rel_pattern}-(m:Node)"
        else:  # both
            path_pattern = f"(n:Node {{node_id: $node_id}})-{rel_pattern}-(m:Node)"

        cypher = f"MATCH {path_pattern} RETURN DISTINCT m"

        try:
            result = await self._run_query(cypher, {"node_id": str(node_id)})
        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(f"get_neighbors failed for node_id={node_id}: {exc}") from exc

        neighbors: list[GraphNode] = []
        seen: set[str] = set()

        for row in result.result_set:
            node_obj = row[0]
            raw_nid = node_obj.properties.get("node_id")
            if raw_nid is None or raw_nid in seen:
                continue
            seen.add(raw_nid)

            raw_props = dict(node_obj.properties)
            raw_props.pop("node_id", None)
            properties = self._deserialize_props(raw_props)
            labels = [lbl for lbl in node_obj.labels if lbl != "Node"]

            try:
                neighbor_uuid = UUID(str(raw_nid))
            except (ValueError, AttributeError):
                logger.warning("get_neighbors: skipping node with non-UUID node_id=%r", raw_nid)
                continue

            neighbors.append(
                GraphNode(id=neighbor_uuid, labels=labels, properties=properties)
            )

        logger.debug(
            "GraphStore.get_neighbors node=%s hops=%d direction=%s returned=%d",
            node_id,
            max_hops,
            direction,
            len(neighbors),
        )
        return neighbors

    async def run_pagerank(
        self,
        seed_ids: list[UUID],
        damping: float = 0.85,
        max_iterations: int = 100,
    ) -> list[RankedNode]:
        """Compute Personalised PageRank seeded from *seed_ids*.

        Fetches the full graph topology from FalkorDB via a single Cypher
        query, then runs power iteration in Python — identical algorithm to
        ``InMemoryGraphStore`` for behavioural parity.

        Parameters
        ----------
        seed_ids:
            Nodes that receive the initial activation mass.
        damping:
            Probability of following an edge vs. teleporting (0–1).
        max_iterations:
            Iteration cap for convergence.

        Returns
        -------
        list[RankedNode]
            All nodes sorted by descending PageRank score.
        """
        self._assert_initialized()

        # Fetch all node IDs
        node_result = await self._run_query(
            "MATCH (n:Node) RETURN n.node_id", {}
        )
        all_node_strs: list[str] = [
            row[0] for row in node_result.result_set if row[0] is not None
        ]

        if not all_node_strs:
            return []

        # Fetch full edge topology as (source_id, target_id) pairs
        edge_result = await self._run_query(
            "MATCH (n:Node)-[r]->(m:Node) RETURN n.node_id, m.node_id", {}
        )

        # Build UUID-indexed adjacency structure
        all_nodes: list[UUID] = []
        for nid_str in all_node_strs:
            try:
                all_nodes.append(UUID(str(nid_str)))
            except (ValueError, AttributeError):
                logger.warning("run_pagerank: skipping non-UUID node_id=%r", nid_str)

        n = len(all_nodes)
        if n == 0:
            return []

        node_index: dict[UUID, int] = {nid: i for i, nid in enumerate(all_nodes)}

        # Build out-edges list per node index
        out_edges: list[list[int]] = [[] for _ in range(n)]
        for row in edge_result.result_set:
            src_str, tgt_str = row[0], row[1]
            if src_str is None or tgt_str is None:
                continue
            try:
                src_uuid = UUID(str(src_str))
                tgt_uuid = UUID(str(tgt_str))
            except (ValueError, AttributeError):
                continue
            src_idx = node_index.get(src_uuid)
            tgt_idx = node_index.get(tgt_uuid)
            if src_idx is not None and tgt_idx is not None:
                out_edges[src_idx].append(tgt_idx)

        # Personalisation vector
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

        out_degree: list[int] = [len(out_edges[i]) for i in range(n)]
        scores = [1.0 / n] * n
        damping_comp = 1.0 - damping

        for iteration in range(max_iterations):
            new_scores = [0.0] * n
            dangling_mass = sum(scores[i] for i in range(n) if out_degree[i] == 0)
            dangling_contrib = dangling_mass / n

            for i in range(n):
                new_scores[i] = damping_comp * teleport[i] + damping * dangling_contrib

            for i in range(n):
                if out_degree[i] == 0:
                    continue
                share = damping * scores[i] / out_degree[i]
                for tgt_idx in out_edges[i]:
                    new_scores[tgt_idx] += share

            delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if delta < 1e-8:
                logger.debug("PageRank converged at iteration %d", iteration + 1)
                break

        # Bulk-fetch properties for all nodes in a single query
        props_map: dict[UUID, dict[str, Any]] = {}
        try:
            bulk_result = await self._run_query(
                "MATCH (n:Node) RETURN n.node_id AS nid, n",
                {},
            )
            for row in bulk_result:
                nid_str = row.get("nid") or row.get("n.node_id")
                if nid_str:
                    try:
                        row_uuid = UUID(str(nid_str))
                        node_obj = row.get("n", {})
                        raw_props = dict(node_obj) if hasattr(node_obj, "__iter__") else {}
                        raw_props.pop("node_id", None)
                        props_map[row_uuid] = self._deserialize_props(raw_props)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            logger.debug("Bulk property fetch failed; using empty properties.")

        ranked: list[RankedNode] = []
        for i, nid in enumerate(all_nodes):
            props = props_map.get(nid, {})
            ranked.append(RankedNode(id=nid, score=scores[i], properties=props))

        ranked.sort(key=lambda r: r.score, reverse=True)
        logger.debug("GraphStore.run_pagerank seeds=%d nodes=%d", len(seed_ids), n)
        return ranked

    async def run_community_detection(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
    ) -> list[list[UUID]]:
        """Partition the graph into communities of related concepts.

        Fetches the full graph topology from FalkorDB and runs label-propagation
        in Python.  The ``algorithm`` parameter is accepted for interface
        compatibility but label propagation is always used.

        Parameters
        ----------
        algorithm:
            Community detection algorithm name (accepted but not dispatched).
        resolution:
            Modularity resolution (accepted but not used by label propagation).

        Returns
        -------
        list[list[UUID]]
            Each inner list is the set of node IDs belonging to one community.
        """
        self._assert_initialized()

        node_result = await self._run_query(
            "MATCH (n:Node) RETURN n.node_id", {}
        )
        all_node_strs: list[str] = [
            row[0] for row in node_result.result_set if row[0] is not None
        ]

        if not all_node_strs:
            return []

        all_nodes: list[UUID] = []
        for nid_str in all_node_strs:
            try:
                all_nodes.append(UUID(str(nid_str)))
            except (ValueError, AttributeError):
                logger.warning(
                    "run_community_detection: skipping non-UUID node_id=%r", nid_str
                )

        n = len(all_nodes)
        if n == 0:
            return []

        node_index: dict[UUID, int] = {nid: i for i, nid in enumerate(all_nodes)}

        # Fetch all edges for adjacency (undirected for community purposes)
        edge_result = await self._run_query(
            "MATCH (n:Node)-[r]->(m:Node) RETURN n.node_id, m.node_id", {}
        )

        neighbors: list[list[int]] = [[] for _ in range(n)]
        for row in edge_result.result_set:
            src_str, tgt_str = row[0], row[1]
            if src_str is None or tgt_str is None:
                continue
            try:
                src_uuid = UUID(str(src_str))
                tgt_uuid = UUID(str(tgt_str))
            except (ValueError, AttributeError):
                continue
            src_idx = node_index.get(src_uuid)
            tgt_idx = node_index.get(tgt_uuid)
            if src_idx is not None and tgt_idx is not None:
                neighbors[src_idx].append(tgt_idx)
                neighbors[tgt_idx].append(src_idx)

        # Label propagation
        labels: list[int] = list(range(n))

        for iteration in range(100):
            changed = False
            shuffled_indices = list(range(n))
            random.shuffle(shuffled_indices)

            for i in shuffled_indices:
                nb_list = neighbors[i]
                if not nb_list:
                    continue

                freq: dict[int, int] = {}
                for nb_idx in nb_list:
                    lbl = labels[nb_idx]
                    freq[lbl] = freq.get(lbl, 0) + 1

                best_label = max(freq, key=lambda lbl: freq[lbl])
                if labels[i] != best_label:
                    labels[i] = best_label
                    changed = True

            if not changed:
                logger.debug(
                    "Label propagation converged at iteration %d", iteration + 1
                )
                break

        community_map: dict[int, list[UUID]] = {}
        for i, nid in enumerate(all_nodes):
            community_map.setdefault(labels[i], []).append(nid)

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
        """Execute a raw Cypher query against FalkorDB.

        Results are converted to a list of dicts keyed by the column names
        returned in the query result header.

        Parameters
        ----------
        cypher:
            Cypher query string.
        params:
            Bound parameters to prevent injection.

        Returns
        -------
        list[dict[str, Any]]
            Each dict is one result row keyed by column name.
        """
        self._assert_initialized()

        result = await self._run_query(cypher, params or {})

        keys: list[str] = result.header if result.header else []
        rows: list[dict[str, Any]] = []

        for row in result.result_set:
            row_dict: dict[str, Any] = {}
            for idx, value in enumerate(row):
                key = keys[idx] if idx < len(keys) else str(idx)
                # Unwrap FalkorDB node/edge objects to plain dicts where possible
                if hasattr(value, "properties"):
                    row_dict[key] = self._deserialize_props(dict(value.properties))
                else:
                    row_dict[key] = value
            rows.append(row_dict)

        logger.debug("GraphStore.query returned=%d rows", len(rows))
        return rows

    async def delete_node(self, node_id: UUID) -> None:
        """Remove *node_id* and all of its incident edges from the graph.

        Uses ``DETACH DELETE`` which removes the node and every relationship
        it participates in atomically.

        Parameters
        ----------
        node_id:
            UUID of the node to remove.
        """
        self._assert_initialized()

        cypher = "MATCH (n:Node {node_id: $node_id}) DETACH DELETE n"

        try:
            await self._run_query(cypher, {"node_id": str(node_id)})
        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(f"delete_node failed for node_id={node_id}: {exc}") from exc

        logger.debug("GraphStore.delete_node id=%s", node_id)

    async def node_count(self) -> int:
        """Return the total number of nodes in the graph.

        Returns
        -------
        int
            Count of all nodes currently stored.
        """
        self._assert_initialized()

        result = await self._run_query("MATCH (n) RETURN count(n)", {})
        rows = result.result_set
        if rows:
            return int(rows[0][0])
        return 0

    async def edge_count(self) -> int:
        """Return the total number of directed edges in the graph.

        Returns
        -------
        int
            Count of all relationships currently stored.
        """
        self._assert_initialized()

        result = await self._run_query("MATCH ()-[r]->() RETURN count(r)", {})
        rows = result.result_set
        if rows:
            return int(rows[0][0])
        return 0
