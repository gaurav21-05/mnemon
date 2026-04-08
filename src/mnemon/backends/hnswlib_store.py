"""
HNSWLib-backed persistent vector store for durable, process-persistent similarity search.

Brain analog: The hippocampal indexing substrate — a persistent approximate nearest-neighbour
index that survives process restarts, encoding the spatial relationships between memory
embeddings in a navigable small-world graph. Where the in-memory store mirrors transient
CA3 pattern completion, this backend models the structural synaptic weight matrix that
persists across sleep cycles.

The index is saved to disk after every mutating operation, making it resilient to
crashes. Dimension is detected lazily on the first insert, enabling the store to be
instantiated before the embedding dimensionality is known.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from mnemon.core.interfaces import VectorItem, VectorSearchResult, VectorStore

__all__ = ["HNSWLibVectorStore"]

logger = logging.getLogger(__name__)


class HNSWLibVectorStore(VectorStore):
    """HNSWLib-backed implementation of the VectorStore ABC.

    Persists the HNSW index to *index_path* and metadata/ID mappings to
    *index_path* + ``.meta.json``.  The index dimension is inferred from the
    first inserted embedding; subsequent inserts must have the same dimension.

    Usage::

        store = HNSWLibVectorStore(index_path="~/.mnemon/vectors/episodic.hnsw")
        await store.initialize()
        await store.insert(uuid, embedding, {"type": "episode"})
        results = await store.search(query_vec, top_k=5)
    """

    def __init__(self, index_path: str, max_elements: int = 100_000) -> None:
        self._index_path = index_path
        self._meta_path = index_path + ".meta.json"
        self._max_elements = max_elements

        self._index: Any = None  # hnswlib.Index, created on first insert
        self._dim: int | None = None

        self._metadata: dict[str, dict] = {}          # str(uuid) -> metadata
        self._id_to_label: dict[str, int] = {}        # str(uuid) -> hnswlib label
        self._label_to_id: dict[int, str] = {}        # hnswlib label -> str(uuid)
        self._next_label: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load existing index+metadata if files exist, else create parent dirs."""
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)

        meta_path = Path(self._meta_path)
        index_path = Path(self._index_path)

        if meta_path.exists() and index_path.exists():
            try:
                self._load()
                logger.debug(
                    "HNSWLibVectorStore loaded index=%s dim=%s count=%d",
                    self._index_path,
                    self._dim,
                    self._index.get_current_count() if self._index else 0,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load HNSWLib index from %s: %s — starting fresh.",
                    self._index_path,
                    exc,
                )
                self._reset()
        else:
            logger.debug(
                "HNSWLibVectorStore: no existing index at %s — will create on first insert.",
                self._index_path,
            )

    # ------------------------------------------------------------------
    # VectorStore ABC
    # ------------------------------------------------------------------

    async def insert(self, id: UUID, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Persist a single embedding with associated metadata."""
        str_id = str(id)
        if str_id in self._id_to_label:
            await self.update(id, embedding, metadata)
            return

        if self._index is None:
            self._create_index(len(embedding))

        label = self._next_label
        self._next_label += 1

        import numpy as np
        vec = np.array([embedding], dtype=np.float32)
        self._index.add_items(vec, [label])

        self._id_to_label[str_id] = label
        self._label_to_id[label] = str_id
        self._metadata[str_id] = dict(metadata)

        self._save()
        logger.debug("HNSWLibVectorStore.insert id=%s label=%d", id, label)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Find the top_k nearest neighbours of query_embedding."""
        if self._index is None or self._index.get_current_count() == 0:
            return []

        import numpy as np
        vec = np.array([query_embedding], dtype=np.float32)

        k = min(top_k, self._index.get_current_count())
        if k == 0:
            return []

        labels, distances = self._index.knn_query(vec, k=k)

        results: list[VectorSearchResult] = []
        for label, dist in zip(labels[0], distances[0]):
            str_id = self._label_to_id.get(int(label))
            if str_id is None:
                continue
            meta = self._metadata.get(str_id, {})

            if filters:
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue

            # hnswlib returns L2 distance; convert to similarity score
            score = float(1.0 / (1.0 + dist))
            results.append(
                VectorSearchResult(id=UUID(str_id), score=score, metadata=meta)
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def delete(self, id: UUID) -> None:
        """Remove the vector identified by id from the index."""
        str_id = str(id)
        label = self._id_to_label.pop(str_id, None)
        if label is None:
            return

        self._label_to_id.pop(label, None)
        self._metadata.pop(str_id, None)

        if self._index is not None:
            try:
                self._index.mark_deleted(label)
            except Exception:
                pass  # Some versions may not support mark_deleted

        self._save()
        logger.debug("HNSWLibVectorStore.delete id=%s", id)

    async def update(
        self,
        id: UUID,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Replace the embedding and metadata for an existing entry."""
        await self.delete(id)
        await self.insert(id, embedding, metadata)

    async def bulk_insert(self, items: list[VectorItem]) -> None:
        """Batch-insert multiple vectors."""
        for item in items:
            str_id = str(item.id)
            if str_id in self._id_to_label:
                # update in-place without save on each step
                old_label = self._id_to_label.pop(str_id)
                self._label_to_id.pop(old_label, None)
                self._metadata.pop(str_id, None)
                if self._index is not None:
                    try:
                        self._index.mark_deleted(old_label)
                    except Exception:
                        pass

            if self._index is None:
                self._create_index(len(item.embedding))

            import numpy as np
            label = self._next_label
            self._next_label += 1
            vec = np.array([item.embedding], dtype=np.float32)
            self._index.add_items(vec, [label])
            self._id_to_label[str_id] = label
            self._label_to_id[label] = str_id
            self._metadata[str_id] = dict(item.metadata)

        if items:
            self._save()
        logger.debug("HNSWLibVectorStore.bulk_insert count=%d", len(items))

    async def count(self) -> int:
        """Return the total number of vectors currently in the index."""
        if self._index is None:
            return 0
        return self._index.get_current_count()

    async def clear(self) -> None:
        """Remove all vectors and delete persisted index files."""
        self._reset()
        for path_str in (self._index_path, self._meta_path):
            path = Path(path_str)
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                logger.warning("HNSWLibVectorStore.clear could not remove %s: %s", path, exc)
        logger.debug("HNSWLibVectorStore.clear index=%s", self._index_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_index(self, dim: int) -> None:
        """Initialise the hnswlib index for the given dimension."""
        import hnswlib
        self._dim = dim
        self._index = hnswlib.Index(space="l2", dim=dim)
        self._index.init_index(max_elements=self._max_elements, ef_construction=200, M=16)
        self._index.set_ef(50)
        logger.debug("HNSWLibVectorStore: created index dim=%d max_elements=%d", dim, self._max_elements)

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        if self._index is None:
            return
        try:
            self._index.save_index(self._index_path)
            meta = {
                "dim": self._dim,
                "max_elements": self._max_elements,
                "next_label": self._next_label,
                "metadata": self._metadata,
                "id_to_label": self._id_to_label,
                "label_to_id": {str(k): v for k, v in self._label_to_id.items()},
            }
            with open(self._meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
        except Exception as exc:
            logger.warning("HNSWLibVectorStore._save failed: %s", exc)

    def _load(self) -> None:
        """Load index and metadata from disk."""
        import hnswlib
        with open(self._meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self._dim = meta["dim"]
        self._max_elements = meta.get("max_elements", self._max_elements)
        self._next_label = meta["next_label"]
        self._metadata = meta["metadata"]
        self._id_to_label = meta["id_to_label"]
        self._label_to_id = {int(k): v for k, v in meta["label_to_id"].items()}

        self._index = hnswlib.Index(space="l2", dim=self._dim)
        self._index.load_index(self._index_path, max_elements=self._max_elements)
        self._index.set_ef(50)

    def _reset(self) -> None:
        """Reset all state to a clean empty store."""
        self._index = None
        self._dim = None
        self._metadata = {}
        self._id_to_label = {}
        self._label_to_id = {}
        self._next_label = 0
