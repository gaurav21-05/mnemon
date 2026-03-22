"""
Backends subpackage — concrete storage implementations for VectorStore,
GraphStore, and DocumentStore interfaces.

Brain analog: The physical neural substrate — the actual synaptic weight
matrices, dendritic trees, and axonal fibres that store and transmit
information.  Backends are interchangeable because all cognitive modules
depend only on the abstract interfaces in ``mnemon.core.interfaces``.

Available backends (installed as optional extras)
--------------------------------------------------
Vector stores:
  - ``QdrantVectorStore``     — Qdrant distributed vector DB    [extra: qdrant]
  - ``HNSWLibVectorStore``    — In-process HNSW index           [extra: hnswlib]
  - ``InMemoryVectorStore``   — Pure-Python dict (dev/test)     [no extra]

Graph stores:
  - ``Neo4jGraphStore``       — Neo4j property graph DB         [extra: neo4j]
  - ``FalkorDBGraphStore``    — FalkorDB (Redis-backed graph)   [extra: falkordb]
  - ``IGraphGraphStore``      — python-igraph (in-process)      [extra: igraph]
  - ``InMemoryGraphStore``    — Pure-Python adjacency dict      [no extra]

Document stores:
  - ``PostgresDocumentStore`` — asyncpg + PostgreSQL            [extra: postgres]
  - ``SQLiteDocumentStore``   — aiosqlite + SQLite              [extra: sqlite]
  - ``InMemoryDocumentStore`` — Pure-Python dict (dev/test)     [no extra]

The ``ModuleRegistry.from_config()`` factory auto-detects which extras are
installed and wires in the appropriate backend without any manual config.
"""

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)

# Optional production backends — imported lazily to avoid hard dependency on
# qdrant-client, falkordb, or aiosqlite when they are not installed.
try:
    from mnemon.backends.qdrant_store import QdrantVectorStore
except ImportError:
    QdrantVectorStore = None  # type: ignore[assignment,misc]

try:
    from mnemon.backends.falkordb_store import FalkorDBGraphStore
except ImportError:
    FalkorDBGraphStore = None  # type: ignore[assignment,misc]

try:
    from mnemon.backends.sqlite_store import SQLiteDocumentStore
except ImportError:
    SQLiteDocumentStore = None  # type: ignore[assignment,misc]

__all__ = [
    # Always available (no extra dependencies)
    "InMemoryVectorStore",
    "InMemoryDocumentStore",
    "InMemoryGraphStore",
    # Optional production backends
    "QdrantVectorStore",
    "FalkorDBGraphStore",
    "SQLiteDocumentStore",
]
