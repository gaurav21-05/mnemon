"""
Unit tests for InMemoryVectorStore, InMemoryDocumentStore, and InMemoryGraphStore.

All operations are tested against the in-memory backend, which is the canonical
reference implementation for all backend contracts.
"""

from __future__ import annotations

import math
from uuid import UUID, uuid4

import pytest

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.config import MnemonConfig
from mnemon.core.interfaces import VectorItem


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class TestInMemoryVectorStore:
    @pytest.fixture(autouse=True)
    def store(self, config: MnemonConfig) -> InMemoryVectorStore:
        self._store = InMemoryVectorStore(config)
        return self._store

    async def test_insert_and_count(self) -> None:
        await self._store.insert(uuid4(), [1.0, 0.0], {})
        assert await self._store.count() == 1

    async def test_insert_multiple_increments_count(self) -> None:
        for _ in range(5):
            await self._store.insert(uuid4(), [1.0, 0.0], {})
        assert await self._store.count() == 5

    async def test_delete_removes_entry(self) -> None:
        vid = uuid4()
        await self._store.insert(vid, [1.0, 0.0], {})
        await self._store.delete(vid)
        assert await self._store.count() == 0

    async def test_delete_nonexistent_is_noop(self) -> None:
        await self._store.delete(uuid4())  # should not raise
        assert await self._store.count() == 0

    async def test_update_overwrites_embedding(self) -> None:
        vid = uuid4()
        await self._store.insert(vid, [1.0, 0.0], {"tag": "old"})
        await self._store.update(vid, [0.0, 1.0], {"tag": "new"})
        results = await self._store.search([0.0, 1.0], top_k=1)
        assert len(results) == 1
        assert results[0].metadata["tag"] == "new"

    async def test_search_returns_top_k_in_descending_order(self) -> None:
        # Three vectors with known cosine similarity to query [1, 0]
        # similarity([1,0],[1,0]) = 1.0
        # similarity([1,0],[0,1]) = 0.0
        # similarity([1,0],[0.7,0.7]) ≈ 0.707
        query = [1.0, 0.0]
        ids = [uuid4(), uuid4(), uuid4()]
        await self._store.insert(ids[0], [1.0, 0.0], {"rank": 1})
        await self._store.insert(ids[1], [0.0, 1.0], {"rank": 3})
        await self._store.insert(ids[2], [0.7071, 0.7071], {"rank": 2})

        results = await self._store.search(query, top_k=3)
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].metadata["rank"] == 1
        assert results[1].metadata["rank"] == 2
        assert results[2].metadata["rank"] == 3

    async def test_search_top_k_limits_results(self) -> None:
        for _ in range(10):
            await self._store.insert(uuid4(), [1.0, 0.0], {})
        results = await self._store.search([1.0, 0.0], top_k=3)
        assert len(results) == 3

    async def test_search_with_filter_excludes_non_matching(self) -> None:
        await self._store.insert(uuid4(), [1.0, 0.0], {"type": "A"})
        await self._store.insert(uuid4(), [1.0, 0.0], {"type": "B"})
        results = await self._store.search([1.0, 0.0], top_k=10, filters={"type": "A"})
        assert len(results) == 1
        assert results[0].metadata["type"] == "A"

    async def test_search_empty_store_returns_empty(self) -> None:
        results = await self._store.search([1.0, 0.0], top_k=5)
        assert results == []

    async def test_bulk_insert(self) -> None:
        items = [
            VectorItem(id=uuid4(), embedding=[float(i), 0.0], metadata={"i": i})
            for i in range(5)
        ]
        await self._store.bulk_insert(items)
        assert await self._store.count() == 5

    async def test_cosine_similarity_zero_vector_returns_zero(self) -> None:
        vid = uuid4()
        await self._store.insert(vid, [0.0, 0.0], {})
        results = await self._store.search([1.0, 0.0], top_k=1)
        assert results[0].score == pytest.approx(0.0)

    async def test_search_result_has_correct_id(self) -> None:
        vid = uuid4()
        await self._store.insert(vid, [1.0, 0.0], {})
        results = await self._store.search([1.0, 0.0], top_k=1)
        assert results[0].id == vid


# ---------------------------------------------------------------------------
# InMemoryDocumentStore
# ---------------------------------------------------------------------------


class TestInMemoryDocumentStore:
    @pytest.fixture(autouse=True)
    def store(self, config: MnemonConfig) -> InMemoryDocumentStore:
        self._store = InMemoryDocumentStore(config)
        return self._store

    async def test_put_and_get(self) -> None:
        did = uuid4()
        await self._store.put(did, {"key": "value"})
        doc = await self._store.get(did)
        assert doc is not None
        assert doc["key"] == "value"

    async def test_get_nonexistent_returns_none(self) -> None:
        doc = await self._store.get(uuid4())
        assert doc is None

    async def test_put_overwrites_existing(self) -> None:
        did = uuid4()
        await self._store.put(did, {"x": 1})
        await self._store.put(did, {"x": 99})
        doc = await self._store.get(did)
        assert doc is not None
        assert doc["x"] == 99

    async def test_delete_removes_document(self) -> None:
        did = uuid4()
        await self._store.put(did, {"x": 1})
        await self._store.delete(did)
        assert await self._store.get(did) is None

    async def test_delete_nonexistent_is_noop(self) -> None:
        await self._store.delete(uuid4())  # should not raise

    async def test_count_all(self) -> None:
        for i in range(4):
            await self._store.put(uuid4(), {"i": i})
        assert await self._store.count() == 4

    async def test_count_with_filter(self) -> None:
        await self._store.put(uuid4(), {"type": "A"})
        await self._store.put(uuid4(), {"type": "A"})
        await self._store.put(uuid4(), {"type": "B"})
        assert await self._store.count(filters={"type": "A"}) == 2
        assert await self._store.count(filters={"type": "B"}) == 1

    async def test_query_no_filter_returns_all(self) -> None:
        for i in range(3):
            await self._store.put(uuid4(), {"i": i})
        docs = await self._store.query(filters={}, limit=100)
        assert len(docs) == 3

    async def test_query_with_filter(self) -> None:
        await self._store.put(uuid4(), {"color": "red", "v": 1})
        await self._store.put(uuid4(), {"color": "blue", "v": 2})
        await self._store.put(uuid4(), {"color": "red", "v": 3})
        docs = await self._store.query(filters={"color": "red"})
        assert len(docs) == 2
        assert all(d["color"] == "red" for d in docs)

    async def test_query_sort_by(self) -> None:
        for v in [3, 1, 2]:
            await self._store.put(uuid4(), {"v": v})
        docs = await self._store.query(filters={}, sort_by="v", limit=100)
        values = [d["v"] for d in docs]
        assert values == sorted(values)

    async def test_query_limit(self) -> None:
        for i in range(10):
            await self._store.put(uuid4(), {"i": i})
        docs = await self._store.query(filters={}, limit=3)
        assert len(docs) == 3

    async def test_query_offset(self) -> None:
        for i in range(5):
            await self._store.put(uuid4(), {"i": i})
        docs_all = await self._store.query(filters={}, sort_by="i", limit=100)
        docs_offset = await self._store.query(filters={}, sort_by="i", limit=100, offset=2)
        assert len(docs_offset) == len(docs_all) - 2
        assert docs_offset[0]["i"] == docs_all[2]["i"]

    async def test_bulk_put(self) -> None:
        items = [(uuid4(), {"n": i}) for i in range(5)]
        await self._store.bulk_put(items)
        assert await self._store.count() == 5

    async def test_returned_document_is_a_copy(self) -> None:
        """Mutating the returned doc should not affect the stored doc."""
        did = uuid4()
        await self._store.put(did, {"x": 1})
        doc = await self._store.get(did)
        assert doc is not None
        doc["x"] = 999
        original = await self._store.get(did)
        assert original is not None
        assert original["x"] == 1


# ---------------------------------------------------------------------------
# InMemoryGraphStore
# ---------------------------------------------------------------------------


class TestInMemoryGraphStore:
    @pytest.fixture(autouse=True)
    def store(self, config: MnemonConfig) -> InMemoryGraphStore:
        self._store = InMemoryGraphStore(config)
        return self._store

    async def test_add_node_and_get_node(self) -> None:
        nid = uuid4()
        await self._store.add_node(nid, ["Person"], {"name": "Alice"})
        node = await self._store.get_node(nid)
        assert node is not None
        assert node["labels"] == ["Person"]
        assert node["properties"]["name"] == "Alice"

    async def test_get_node_nonexistent_returns_none(self) -> None:
        assert await self._store.get_node(uuid4()) is None

    async def test_node_upsert_merges_properties(self) -> None:
        nid = uuid4()
        await self._store.add_node(nid, ["X"], {"a": 1})
        await self._store.add_node(nid, ["X"], {"b": 2})
        node = await self._store.get_node(nid)
        assert node is not None
        assert node["properties"]["a"] == 1
        assert node["properties"]["b"] == 2

    async def test_node_count(self) -> None:
        for _ in range(3):
            await self._store.add_node(uuid4(), ["N"], {})
        assert await self._store.node_count() == 3

    async def test_edge_count(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        await self._store.add_edge(a, b, "KNOWS")
        await self._store.add_edge(b, c, "KNOWS")
        assert await self._store.edge_count() == 2

    async def test_get_neighbors_out_direction(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        await self._store.add_edge(a, b, "REL")
        await self._store.add_edge(a, c, "REL")
        neighbors = await self._store.get_neighbors(a, direction="out")
        neighbor_ids = {n.id for n in neighbors}
        assert b in neighbor_ids
        assert c in neighbor_ids
        assert a not in neighbor_ids

    async def test_get_neighbors_in_direction(self) -> None:
        a, b = uuid4(), uuid4()
        for nid in (a, b):
            await self._store.add_node(nid, ["N"], {})
        await self._store.add_edge(a, b, "REL")
        neighbors = await self._store.get_neighbors(b, direction="in")
        assert any(n.id == a for n in neighbors)

    async def test_get_neighbors_both_direction(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        await self._store.add_edge(a, b, "REL")
        await self._store.add_edge(c, b, "REL")
        neighbors = await self._store.get_neighbors(b, direction="both")
        ids = {n.id for n in neighbors}
        assert a in ids
        assert c in ids

    async def test_get_neighbors_edge_type_filter(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        await self._store.add_edge(a, b, "KNOWS")
        await self._store.add_edge(a, c, "HATES")
        knows_neighbors = await self._store.get_neighbors(a, edge_type="KNOWS")
        assert any(n.id == b for n in knows_neighbors)
        assert not any(n.id == c for n in knows_neighbors)

    async def test_get_neighbors_nonexistent_node_returns_empty(self) -> None:
        neighbors = await self._store.get_neighbors(uuid4())
        assert neighbors == []

    async def test_delete_node_removes_node_and_incident_edges(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        await self._store.add_edge(a, b, "REL")
        await self._store.add_edge(b, c, "REL")

        await self._store.delete_node(b)

        assert await self._store.get_node(b) is None
        assert await self._store.node_count() == 2
        assert await self._store.edge_count() == 0

    async def test_delete_nonexistent_node_is_noop(self) -> None:
        await self._store.delete_node(uuid4())  # should not raise

    async def test_pagerank_simple_three_node_graph(self) -> None:
        """Verify that PageRank ranks the hub node highest in a simple graph.

        Graph:  A → B ← C   (B has two in-edges, A and C have one out-edge each)
        Seed: A.  The hub B should receive the most accumulated rank.
        """
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        await self._store.add_edge(a, b, "REL")
        await self._store.add_edge(c, b, "REL")

        ranked = await self._store.run_pagerank(seed_ids=[a])
        assert len(ranked) == 3
        # Scores must be in descending order
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)
        # All scores must be non-negative
        assert all(s >= 0.0 for s in scores)

    async def test_pagerank_empty_graph_returns_empty(self) -> None:
        ranked = await self._store.run_pagerank(seed_ids=[uuid4()])
        assert ranked == []

    async def test_pagerank_scores_sum_to_approximately_one(self) -> None:
        ids = [uuid4() for _ in range(4)]
        for nid in ids:
            await self._store.add_node(nid, ["N"], {})
        for i in range(len(ids) - 1):
            await self._store.add_edge(ids[i], ids[i + 1], "REL")

        ranked = await self._store.run_pagerank(seed_ids=[ids[0]])
        total = sum(r.score for r in ranked)
        assert total == pytest.approx(1.0, abs=1e-4)

    async def test_community_detection_isolated_nodes_form_singletons(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        # No edges → each node is its own community
        communities = await self._store.run_community_detection()
        assert len(communities) == 3
        assert all(len(c) == 1 for c in communities)

    async def test_community_detection_connected_pair_groups_together(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        for nid in (a, b, c):
            await self._store.add_node(nid, ["N"], {})
        # Strongly connect a and b; c is isolated
        await self._store.add_edge(a, b, "REL")
        await self._store.add_edge(b, a, "REL")

        communities = await self._store.run_community_detection()
        # a and b must end up in the same community
        flat: dict[UUID, int] = {}
        for i, comm in enumerate(communities):
            for nid in comm:
                flat[nid] = i
        assert flat[a] == flat[b]
        assert flat[c] != flat[a]

    async def test_community_detection_empty_graph_returns_empty(self) -> None:
        communities = await self._store.run_community_detection()
        assert communities == []

    async def test_query_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            await self._store.query("MATCH (n) RETURN n")
