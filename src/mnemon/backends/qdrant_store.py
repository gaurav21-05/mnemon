"""
Qdrant-backed vector store for production-grade hippocampal indexing.

Brain analog: The dentate gyrus / CA3 subfield of the hippocampus, where
sparse, high-dimensional representations are indexed using approximate nearest
neighbour (ANN) structures.  Qdrant's HNSW index mirrors the attractor dynamics
of CA3 — queries propagate through a navigable small-world graph to converge on
the most similar stored engrams, enabling sub-linear retrieval even across tens
of millions of vectors.

Unlike the InMemoryVectorStore (which performs exhaustive scan, analogous to
unstructured short-term potentiation), this backend persists vectors durably
and scales to production workloads.  The cosine distance metric reflects the
angular geometry used by transformer-derived embeddings; points on the same
semantic manifold cluster together regardless of magnitude.

Lifecycle
---------
1. Instantiate with a ``MnemonConfig``.
2. Call ``await store.initialize()`` before any read/write operation.
3. Use ``await store.close()`` on shutdown to release the gRPC connection pool.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from mnemon.core.exceptions import MemoryError, RetrievalError
from mnemon.core.interfaces import VectorItem, VectorSearchResult, VectorStore

if TYPE_CHECKING:
    from mnemon.core.config import MnemonConfig

__all__ = ["QdrantVectorStore"]

logger = logging.getLogger(__name__)

# Default vector dimension assumed when the collection must be created before
# any vectors have been inserted (matches OpenAI text-embedding-ada-002 / 3-small).
_DEFAULT_DIMENSION: int = 1536


class QdrantVectorStore(VectorStore):
    """Production vector store backed by Qdrant's HNSW index.

    Brain analog: The hippocampal CA3-dentate gyrus circuit — a persistent,
    high-capacity index for approximate nearest-neighbour retrieval using the
    Hierarchical Navigable Small World (HNSW) algorithm.  Supports optional
    metadata filtering (analogous to entorhinal context gating) and gRPC
    transport for low-latency communication with the Qdrant server.

    Thread-safety: AsyncQdrantClient is safe for concurrent coroutine use.
    Do not share a single instance across OS threads without an asyncio event
    loop per thread.

    Parameters
    ----------
    config:
        Root Mnemon configuration.  Qdrant-specific settings are read from
        ``config.qdrant`` if present; otherwise defaults are applied.
    """

    def __init__(self, config: MnemonConfig) -> None:
        qdrant_cfg = getattr(config, "qdrant", None)

        self._host: str = getattr(qdrant_cfg, "host", "localhost")
        self._port: int = getattr(qdrant_cfg, "port", 6333)
        self._grpc_port: int = getattr(qdrant_cfg, "grpc_port", 6334)
        self._prefer_grpc: bool = getattr(qdrant_cfg, "prefer_grpc", True)
        self._collection_name: str = getattr(qdrant_cfg, "collection_name", "mnemon_vectors")
        self._dimension: int | None = getattr(qdrant_cfg, "dimension", None)
        self._binary_quantization: bool = getattr(qdrant_cfg, "binary_quantization", False)
        self._on_disk: bool = getattr(qdrant_cfg, "on_disk", False)

        self._client: AsyncQdrantClient | None = None
        self._initialized: bool = False

        logger.debug(
            "QdrantVectorStore configured host=%s port=%d collection=%s prefer_grpc=%s",
            self._host,
            self._port,
            self._collection_name,
            self._prefer_grpc,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to Qdrant and ensure the target collection exists.

        Creates the collection with a COSINE distance HNSW index if it does
        not already exist.  Uses the configured dimension, or ``1536`` as a
        safe default when no dimension has been provided yet.

        Raises
        ------
        MemoryError
            If the connection cannot be established or the collection cannot
            be created or inspected.
        """
        try:
            self._client = AsyncQdrantClient(
                host=self._host,
                port=self._port,
                grpc_port=self._grpc_port,
                prefer_grpc=self._prefer_grpc,
            )
            await self._ensure_collection()
            self._initialized = True
            logger.info(
                "QdrantVectorStore initialised collection=%s dim=%s",
                self._collection_name,
                self._dimension,
            )
        except Exception as exc:
            raise MemoryError(
                f"Failed to initialise QdrantVectorStore: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the underlying gRPC / HTTP connection pool.

        Safe to call even if ``initialize()`` was never invoked.
        """
        if self._client is not None:
            try:
                await self._client.close()
                logger.debug("QdrantVectorStore connection closed")
            except Exception as exc:  # pragma: no cover
                logger.warning("Error while closing Qdrant client: %s", exc)
            finally:
                self._client = None
                self._initialized = False

    # ------------------------------------------------------------------
    # VectorStore ABC
    # ------------------------------------------------------------------

    async def insert(
        self,
        id: UUID,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a single embedding with associated metadata.

        Parameters
        ----------
        id:
            Stable UUID that links this vector to its source document.
        embedding:
            Dense float vector produced by an EmbeddingProvider.
        metadata:
            Arbitrary key/value payload stored alongside the vector and
            returned in search results.

        Raises
        ------
        RuntimeError
            If ``initialize()`` has not been called.
        MemoryError
            If the upsert operation fails.
        """
        self._assert_initialized()
        await self._auto_detect_dimension(embedding)
        try:
            await self._client.upsert(  # type: ignore[union-attr]
                collection_name=self._collection_name,
                points=[
                    models.PointStruct(
                        id=str(id),
                        vector=embedding,
                        payload=metadata,
                    )
                ],
            )
            logger.debug("VectorStore.insert id=%s dim=%d", id, len(embedding))
        except Exception as exc:
            raise MemoryError(f"Failed to insert vector id={id}: {exc}") from exc

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Find the *top_k* nearest neighbours of *query_embedding*.

        Parameters
        ----------
        query_embedding:
            The probe vector to compare against the HNSW index.
        top_k:
            Maximum number of results to return.
        filters:
            Optional metadata equality predicates.  Each key/value pair
            is translated to a Qdrant ``FieldCondition`` with ``MatchValue``.

        Returns
        -------
        list[VectorSearchResult]
            Results in descending cosine similarity order.

        Raises
        ------
        RuntimeError
            If ``initialize()`` has not been called.
        RetrievalError
            If the search operation fails.
        """
        self._assert_initialized()
        qdrant_filter = self._build_filter(filters) if filters else None
        try:
            hits = await self._client.search(  # type: ignore[union-attr]
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            results = [
                VectorSearchResult(
                    id=UUID(hit.id) if isinstance(hit.id, str) else UUID(str(hit.id)),
                    score=hit.score,
                    metadata=hit.payload or {},
                )
                for hit in hits
            ]
            logger.debug(
                "VectorStore.search top_k=%d filters=%s returned=%d",
                top_k,
                filters,
                len(results),
            )
            return results
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

    async def delete(self, id: UUID) -> None:
        """Remove the vector identified by *id* from the index.

        Parameters
        ----------
        id:
            UUID of the point to delete.  No-op if the point does not exist.

        Raises
        ------
        RuntimeError
            If ``initialize()`` has not been called.
        MemoryError
            If the delete operation fails.
        """
        self._assert_initialized()
        try:
            await self._client.delete(  # type: ignore[union-attr]
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(points=[str(id)]),
            )
            logger.debug("VectorStore.delete id=%s", id)
        except Exception as exc:
            raise MemoryError(f"Failed to delete vector id={id}: {exc}") from exc

    async def update(
        self,
        id: UUID,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Replace the embedding and metadata for an existing entry.

        Qdrant's upsert is idempotent; this method is semantically identical
        to ``insert`` and will create the point if it does not already exist.

        Parameters
        ----------
        id:
            UUID of the point to update.
        embedding:
            New dense float vector.
        metadata:
            New metadata payload that fully replaces the existing payload.

        Raises
        ------
        RuntimeError
            If ``initialize()`` has not been called.
        MemoryError
            If the upsert operation fails.
        """
        self._assert_initialized()
        await self._auto_detect_dimension(embedding)
        try:
            await self._client.upsert(  # type: ignore[union-attr]
                collection_name=self._collection_name,
                points=[
                    models.PointStruct(
                        id=str(id),
                        vector=embedding,
                        payload=metadata,
                    )
                ],
            )
            logger.debug("VectorStore.update id=%s", id)
        except Exception as exc:
            raise MemoryError(f"Failed to update vector id={id}: {exc}") from exc

    async def bulk_insert(self, items: list[VectorItem]) -> None:
        """Batch-upsert multiple vectors in a single Qdrant request.

        Exploits Qdrant's native batch upsert API for significantly higher
        throughput than repeated ``insert`` calls on large ingestion workloads.

        Parameters
        ----------
        items:
            List of ``VectorItem`` objects to upsert.  All items must share
            the same embedding dimensionality.

        Raises
        ------
        RuntimeError
            If ``initialize()`` has not been called.
        MemoryError
            If the batch upsert operation fails.
        """
        self._assert_initialized()
        if not items:
            return
        await self._auto_detect_dimension(items[0].embedding)
        try:
            points = [
                models.PointStruct(
                    id=str(item.id),
                    vector=item.embedding,
                    payload=item.metadata,
                )
                for item in items
            ]
            await self._client.upsert(  # type: ignore[union-attr]
                collection_name=self._collection_name,
                points=points,
            )
            logger.debug("VectorStore.bulk_insert count=%d", len(items))
        except Exception as exc:
            raise MemoryError(f"Failed to bulk insert {len(items)} vectors: {exc}") from exc

    async def count(self) -> int:
        """Return the total number of vectors currently in the collection.

        Returns
        -------
        int
            Exact point count as reported by Qdrant.

        Raises
        ------
        RuntimeError
            If ``initialize()`` has not been called.
        MemoryError
            If the count operation fails.
        """
        self._assert_initialized()
        try:
            result = await self._client.count(  # type: ignore[union-attr]
                collection_name=self._collection_name,
                exact=True,
            )
            return result.count
        except Exception as exc:
            raise MemoryError(f"Failed to count vectors: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        """Raise RuntimeError if ``initialize()`` has not completed successfully."""
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "QdrantVectorStore has not been initialised. "
                "Call `await store.initialize()` before use."
            )

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist.

        Uses the configured or default dimension.  If the collection already
        exists this method is a no-op; the existing collection parameters are
        left unchanged.
        """
        assert self._client is not None  # guaranteed by calling context
        try:
            await self._client.get_collection(self._collection_name)
            logger.debug("Qdrant collection '%s' already exists", self._collection_name)
            return
        except Exception as get_exc:
            # Only proceed to create if the error indicates collection absence.
            # Log the original error for diagnostics.
            logger.debug(
                "get_collection('%s') raised %s — attempting creation.",
                self._collection_name,
                get_exc,
            )

        try:
            dimension = self._dimension if self._dimension is not None else _DEFAULT_DIMENSION
            create_kwargs: dict[str, Any] = {
                "collection_name": self._collection_name,
                "vectors_config": models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=self._on_disk,
                ),
            }
            if self._binary_quantization:
                create_kwargs["quantization_config"] = models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True,
                    ),
                )
            await self._client.create_collection(**create_kwargs)
            logger.info(
                "Created Qdrant collection '%s' with dim=%d distance=COSINE "
                "binary_quantization=%s on_disk=%s",
                self._collection_name,
                dimension,
                self._binary_quantization,
                self._on_disk,
            )
        except Exception as create_exc:
            raise MemoryError(
                f"Failed to create Qdrant collection '{self._collection_name}': {create_exc}"
            ) from create_exc

    async def _auto_detect_dimension(self, embedding: list[float]) -> None:
        """Record the vector dimension from the first embedding seen.

        If the collection was created with the default dimension (1536) and
        actual embeddings are a different size, this method does **not**
        recreate the collection — callers are responsible for ensuring
        dimension consistency.  The stored ``_dimension`` value is used only
        for logging and first-time collection creation.
        """
        if self._dimension is None:
            self._dimension = len(embedding)
            logger.debug("Auto-detected embedding dimension: %d", self._dimension)

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> models.Filter:
        """Convert a flat equality-predicate dict into a Qdrant ``Filter``.

        Each key/value pair in *filters* becomes a ``FieldCondition`` with a
        ``MatchValue`` clause.  All conditions are joined with a logical AND
        (``must`` list).

        Parameters
        ----------
        filters:
            Mapping of payload field names to their expected values.

        Returns
        -------
        models.Filter
            A Qdrant filter object ready to pass to ``client.search``.
        """
        must_conditions = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
        return models.Filter(must=must_conditions)
