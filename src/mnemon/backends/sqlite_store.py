"""
SQLite-backed document store for durable, process-persistent memory storage.

Brain analog: The physical substrate of long-term potentiation — once a synaptic
trace is committed to SQLite, it survives process restarts, power cycles, and
system reboots.  Where the in-memory store mirrors the transient hippocampal
working buffer, this backend models the structural synaptic modifications that
consolidation produces in the neocortex: slow to form, slow to decay, and
retrievable long after the originating experience has faded from active working
memory.

Just as LTP converts short-lived post-synaptic potentiation into stable AMPA
receptor insertion, ``SQLiteDocumentStore`` converts ephemeral Python dicts into
durable rows indexed by a UUID primary key — a one-to-one analogue of the
engram's transition from labile to consolidated state.

Thread-safety note: aiosqlite serialises all access through the event loop.
SQLite's WAL mode permits concurrent readers but only one writer at a time.
For multi-process workloads, an external database (e.g. PostgreSQL) should be
preferred.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import aiosqlite

from mnemon.core.exceptions import ConfigError, MemoryError
from mnemon.core.interfaces import DocumentStore

if TYPE_CHECKING:
    from uuid import UUID

    from mnemon.core.config import MnemonConfig

__all__ = ["SQLiteDocumentStore"]

logger = logging.getLogger(__name__)

_UNINITIALIZED_MSG = (
    "SQLiteDocumentStore has not been initialised. "
    "Call await store.initialize() before use."
)


class SQLiteDocumentStore(DocumentStore):
    """SQLite-backed implementation of the DocumentStore ABC.

    Brain analog: The durable synaptic substrate — a persistent store of memory
    traces encoded as JSON rows in a local SQLite database file.  Unlike the
    in-memory store (transient potentiation), every write here survives process
    termination, mirroring the conversion of short-term synaptic changes into
    stable structural modifications during memory consolidation.

    Documents are stored as JSON-serialised dicts in a single ``data`` TEXT
    column.  The UUID primary key is also embedded in the JSON payload so that
    a retrieved row is self-contained.  SQLite's ``json_extract()`` function
    powers field-level filtering and sorting without requiring a schema change
    as document shapes evolve.

    WAL mode is enabled on initialisation to allow concurrent readers without
    blocking the single serialised writer — an important property when multiple
    cognitive subsystems issue simultaneous read queries during a retrieval fan-out.

    Usage::

        store = SQLiteDocumentStore(config)
        await store.initialize()
        await store.put(some_uuid, {"type": "episode", "content": "..."})
        doc = await store.get(some_uuid)
        await store.close()
    """

    def __init__(
        self,
        config: MnemonConfig,
        db_path: str | None = None,
        table_name: str | None = None,
    ) -> None:
        """Prepare settings but do not open a database connection.

        Parameters
        ----------
        config:
            Root Mnemon configuration.  SQLite-specific settings are read from
            ``config.model_extra`` if present; otherwise defaults are used.
            Supported extra keys: ``sqlite_db_path``, ``sqlite_table_name``.
        db_path:
            Explicit path to the SQLite database file.  When provided, takes
            precedence over ``config.model_extra["sqlite_db_path"]``.
        table_name:
            Explicit table name.  When provided, takes precedence over
            ``config.model_extra["sqlite_table_name"]``.
        """
        extra: dict[str, Any] = {}
        if hasattr(config, "model_extra") and config.model_extra:
            extra = config.model_extra

        self._db_path: str = db_path or str(extra.get("sqlite_db_path", "mnemon_documents.db"))
        self._table_name: str = table_name or str(extra.get("sqlite_table_name", "documents"))

        # Validate table name to prevent SQL injection — only alphanumeric + underscore
        if not self._table_name or not all(
            ch.isalnum() or ch == "_" for ch in self._table_name
        ):
            raise ConfigError(
                f"Invalid sqlite_table_name: {self._table_name!r}. "
                "Only alphanumeric characters and underscores are allowed."
            )

        self._db: aiosqlite.Connection | None = None
        self._initialized: bool = False

        logger.debug(
            "SQLiteDocumentStore configured db_path=%s table=%s",
            self._db_path,
            self._table_name,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the database connection and create the schema if absent.

        Must be called before any other method.  Idempotent — safe to call
        multiple times; subsequent calls are no-ops.

        Raises
        ------
        MemoryError
            If the database file cannot be opened or the schema cannot be created.
        """
        if self._initialized:
            return

        try:
            from pathlib import Path
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row

            # WAL mode: readers don't block the writer and vice-versa.
            await self._db.execute("PRAGMA journal_mode=WAL")

            # Main document table.
            await self._db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id         TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )

            # Index on created_at to speed up chronological sorts.
            await self._db.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_created_at
                ON {self._table_name} (created_at)
                """
            )

            await self._db.commit()
            self._initialized = True
            logger.debug(
                "SQLiteDocumentStore initialised db_path=%s table=%s",
                self._db_path,
                self._table_name,
            )
        except Exception as exc:
            raise MemoryError(
                f"Failed to initialise SQLiteDocumentStore at '{self._db_path}': {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the underlying database connection.

        Safe to call even if :meth:`initialize` was never called.  After
        closing, all subsequent method calls will raise ``MemoryError``.
        """
        if self._db is not None:
            try:
                await self._db.close()
                logger.debug("SQLiteDocumentStore closed db_path=%s", self._db_path)
            except Exception as exc:
                logger.warning("Error closing SQLiteDocumentStore: %s", exc)
            finally:
                self._db = None
                self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> aiosqlite.Connection:
        """Return the live connection or raise MemoryError."""
        if not self._initialized or self._db is None:
            raise MemoryError(_UNINITIALIZED_MSG)
        return self._db

    def _serialize(self, id: UUID, document: dict[str, Any]) -> str:
        """Merge *id* into *document* and return a JSON string."""
        payload = dict(document)
        payload["id"] = str(id)
        return json.dumps(payload)

    def _deserialize(self, data: str) -> dict[str, Any]:
        """Parse a JSON string back into a document dict."""
        return json.loads(data)

    def _build_where_clause(
        self, filters: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        """Translate a flat equality filter dict into a SQL WHERE fragment.

        Uses SQLite's ``json_extract()`` to reach inside the JSON ``data``
        column without requiring a schema change per field.

        Returns
        -------
        tuple[str, list[Any]]
            ``(clause, params)`` where *clause* is the WHERE expression
            (without the ``WHERE`` keyword) and *params* is the list of
            bound parameter values aligned with the ``?`` placeholders.
        """
        if not filters:
            return "1", []

        clauses: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            # Sanitise key to prevent SQL injection via json_extract path
            safe_key = "".join(ch for ch in key if ch.isalnum() or ch == "_")
            if safe_key != key:
                logger.warning(
                    "Skipping filter key with unsafe characters: %r (sanitised=%r)",
                    key,
                    safe_key,
                )
                continue
            clauses.append(f"json_extract(data, '$.{safe_key}') = ?")
            params.append(value)
        if not clauses:
            return "1", []
        return " AND ".join(clauses), params

    # ------------------------------------------------------------------
    # DocumentStore interface
    # ------------------------------------------------------------------

    async def put(self, id: UUID, document: dict[str, Any]) -> None:
        """Insert or replace *document* under *id*.

        The UUID is embedded in the JSON payload so that consumers can
        reconstruct a self-contained document from the ``data`` column alone.
        ``updated_at`` is refreshed to ``datetime('now')`` on every upsert.

        Parameters
        ----------
        id:
            Document identifier.
        document:
            Arbitrary JSON-serialisable dict.

        Raises
        ------
        MemoryError
            If the store is uninitialised or the write fails.
        """
        db = self._assert_initialized()
        serialized = self._serialize(id, document)
        try:
            await db.execute(
                f"""
                INSERT INTO {self._table_name} (id, data, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    data       = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (str(id), serialized),
            )
            await db.commit()
            logger.debug("DocumentStore.put id=%s", id)
        except Exception as exc:
            raise MemoryError(f"put failed for id={id}: {exc}") from exc

    async def get(self, id: UUID) -> dict[str, Any] | None:
        """Fetch the document identified by *id*, or ``None`` if absent.

        Parameters
        ----------
        id:
            UUID of the document to retrieve.

        Returns
        -------
        dict[str, Any] | None
            Deserialised document dict, or ``None`` if no row exists for *id*.

        Raises
        ------
        MemoryError
            If the store is uninitialised or the query fails.
        """
        db = self._assert_initialized()
        try:
            async with db.execute(
                f"SELECT data FROM {self._table_name} WHERE id = ?",
                (str(id),),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            doc = self._deserialize(row["data"])
            logger.debug("DocumentStore.get id=%s found=True", id)
            return doc
        except Exception as exc:
            raise MemoryError(f"get failed for id={id}: {exc}") from exc

    async def query(
        self,
        filters: dict[str, Any],
        sort_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query documents matching *filters* with optional sort and pagination.

        Field-level equality filtering is implemented via SQLite's
        ``json_extract(data, '$.field') = ?``.  All filter predicates are
        combined with AND.  Parameterized queries are used throughout to
        prevent SQL injection.

        Parameters
        ----------
        filters:
            Flat dict of field name → expected value equality predicates.
        sort_by:
            JSON field name to sort by ascending.  ``None`` leaves the order
            unspecified (insertion order in WAL mode is typically stable but
            not guaranteed).
        limit:
            Maximum number of documents to return.
        offset:
            Number of matching documents to skip (for pagination).

        Returns
        -------
        list[dict[str, Any]]
            Deserialised document dicts.

        Raises
        ------
        MemoryError
            If the store is uninitialised or the query fails.
        """
        db = self._assert_initialized()
        where_clause, params = self._build_where_clause(filters)

        order_clause = ""
        if sort_by is not None:
            # sort_by is a field name from the application layer, not user input
            # in the SQL-injection sense, but we still keep it out of params
            # because SQLite does not support bind parameters for ORDER BY columns.
            # The field name is sanitised by restricting to alphanumeric + underscore.
            safe_sort_field = "".join(
                ch for ch in sort_by if ch.isalnum() or ch == "_"
            )
            order_clause = f"ORDER BY json_extract(data, '$.{safe_sort_field}') ASC"

        sql = (
            f"SELECT data FROM {self._table_name} "
            f"WHERE {where_clause} "
            f"{order_clause} "
            f"LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        try:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
            results = [self._deserialize(row["data"]) for row in rows]
            logger.debug(
                "DocumentStore.query filters=%s sort_by=%s limit=%d offset=%d returned=%d",
                filters,
                sort_by,
                limit,
                offset,
                len(results),
            )
            return results
        except Exception as exc:
            raise MemoryError(f"query failed (filters={filters}): {exc}") from exc

    async def delete(self, id: UUID) -> None:
        """Permanently remove the document identified by *id*.

        No-op if no document with *id* exists.

        Parameters
        ----------
        id:
            UUID of the document to delete.

        Raises
        ------
        MemoryError
            If the store is uninitialised or the delete fails.
        """
        db = self._assert_initialized()
        try:
            await db.execute(
                f"DELETE FROM {self._table_name} WHERE id = ?",
                (str(id),),
            )
            await db.commit()
            logger.debug("DocumentStore.delete id=%s", id)
        except Exception as exc:
            raise MemoryError(f"delete failed for id={id}: {exc}") from exc

    async def bulk_put(self, items: list[tuple[UUID, dict[str, Any]]]) -> None:
        """Insert or replace multiple documents in a single atomic transaction.

        Uses ``executemany`` with a single ``INSERT OR REPLACE`` statement,
        which is significantly more efficient than repeated :meth:`put` calls
        for large ingestion workloads.  The entire batch is committed atomically
        — either all rows land or none do.

        Parameters
        ----------
        items:
            Sequence of ``(id, document)`` pairs.

        Raises
        ------
        MemoryError
            If the store is uninitialised or any part of the batch fails.
        """
        if not items:
            return

        db = self._assert_initialized()
        rows = [
            (str(id_), self._serialize(id_, doc))
            for id_, doc in items
        ]
        try:
            # Use aiosqlite's connection as a transaction context manager:
            # __aenter__ begins a transaction, __aexit__ commits on success
            # or rolls back on exception.
            await db.execute("BEGIN")
            try:
                await db.executemany(
                    f"""
                    INSERT INTO {self._table_name} (id, data, updated_at)
                    VALUES (?, ?, datetime('now'))
                    ON CONFLICT(id) DO UPDATE SET
                        data       = excluded.data,
                        updated_at = excluded.updated_at
                    """,
                    rows,
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            logger.debug("DocumentStore.bulk_put count=%d", len(items))
        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(f"bulk_put failed for {len(items)} items: {exc}") from exc

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Return the number of documents matching *filters*.

        Parameters
        ----------
        filters:
            Optional flat dict of equality predicates.  ``None`` (or an empty
            dict) counts all documents in the table.

        Returns
        -------
        int
            Document count.

        Raises
        ------
        MemoryError
            If the store is uninitialised or the query fails.
        """
        db = self._assert_initialized()
        effective_filters: dict[str, Any] = filters or {}
        where_clause, params = self._build_where_clause(effective_filters)

        sql = f"SELECT COUNT(*) FROM {self._table_name} WHERE {where_clause}"
        try:
            async with db.execute(sql, params) as cursor:
                row = await cursor.fetchone()
            result: int = row[0] if row is not None else 0
            logger.debug(
                "DocumentStore.count filters=%s result=%d", filters, result
            )
            return result
        except Exception as exc:
            raise MemoryError(f"count failed (filters={filters}): {exc}") from exc
