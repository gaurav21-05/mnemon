"""
ModuleRegistry — dependency injection container for Mnemon cognitive modules.

Design
------
Follows the Dependency Inversion Principle: high-level cognitive modules
request interfaces (ABCs); the registry resolves them to concrete
implementations that were registered at startup.

The registry is deliberately simple — it is a typed service locator, not
a full IoC container.  Auto-detection logic in ``from_config`` inspects
which optional extras are installed and wires in the appropriate backends,
providing sensible defaults for local development.

Brain analog: The connectome — the wiring diagram that determines which
brain regions can communicate and via which pathways.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from mnemon.core.exceptions import BackendNotAvailableError, ConfigError

if TYPE_CHECKING:
    from mnemon.core.config import MnemonConfig

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Service locator / dependency injection container for Mnemon modules.

    Usage
    -----
    ::

        registry = ModuleRegistry()
        registry.register(EmbeddingProvider, my_embedding_impl)
        registry.register(VectorStore, my_vector_store_impl)

        # Later, in a cognitive module:
        embedder = registry.resolve(EmbeddingProvider)

    The *interface* key is the ABC class itself.  Concrete implementations
    do not need to be registered under a string name — the type system
    provides the necessary uniqueness.
    """

    def __init__(self) -> None:
        self._services: dict[type[Any], Any] = {}

    # ------------------------------------------------------------------
    # Core registry operations
    # ------------------------------------------------------------------

    def register(self, interface: type[Any], implementation: Any) -> None:
        """Register *implementation* as the provider for *interface*.

        Parameters
        ----------
        interface:
            The ABC or protocol class that consumers will resolve against.
        implementation:
            A concrete instance that implements *interface*.

        Notes
        -----
        Registering a second implementation for the same interface silently
        replaces the first.  Log a warning so ops teams can spot accidental
        double-registration in complex startup sequences.
        """
        if interface in self._services:
            logger.warning(
                "ModuleRegistry: replacing existing registration for %s.",
                interface.__name__,
            )
        self._services[interface] = implementation
        logger.debug(
            "Registered %s → %s.",
            interface.__name__,
            type(implementation).__name__,
        )

    def resolve(self, interface: type[Any]) -> Any:
        """Return the implementation registered for *interface*.

        Parameters
        ----------
        interface:
            The ABC or protocol class to look up.

        Returns
        -------
        Any
            The registered implementation instance.

        Raises
        ------
        ConfigError
            If no implementation has been registered for *interface*.
        """
        try:
            return self._services[interface]
        except KeyError:
            raise ConfigError(
                f"No implementation registered for interface '{interface.__name__}'. "
                f"Call registry.register({interface.__name__}, <impl>) before resolving."
            ) from None

    def has(self, interface: type[Any]) -> bool:
        """Return True if *interface* has a registered implementation."""
        return interface in self._services

    def registered_interfaces(self) -> list[type[Any]]:
        """Return all interface types currently registered."""
        return list(self._services.keys())

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: MnemonConfig) -> ModuleRegistry:
        """Build a registry from *config*, auto-detecting available backends.

        Detection strategy (in priority order for each resource type):
        1. Explicit backend name in config (e.g. ``config.vector_backend = "qdrant"``).
        2. First available optional extra, tested by import probe.
        3. In-memory fallback (always available, no extra required).

        Parameters
        ----------
        config:
            Fully populated MnemonConfig (typically loaded via ``load_config()``).

        Returns
        -------
        ModuleRegistry
            Registry with all standard cognitive module interfaces wired.

        Raises
        ------
        ConfigError
            If a required backend is configured but the extra is not installed.
        BackendNotAvailableError
            If an explicitly configured backend's package is absent.
        """
        registry = cls()
        registry._wire_llm_providers(config)
        registry._wire_vector_store(config)
        registry._wire_graph_store(config)
        registry._wire_document_store(config)
        return registry

    # ------------------------------------------------------------------
    # Private wiring helpers (called by from_config)
    # ------------------------------------------------------------------

    def _wire_llm_providers(self, config: MnemonConfig) -> None:
        """Register LLMProvider and EmbeddingProvider from config.

        Reads the LLMConfig provider map to determine the default model and
        any additional per-provider kwargs (api_key, api_base, etc.).
        """
        from mnemon.core.interfaces import EmbeddingProvider, LLMProvider

        try:
            from mnemon.providers.litellm_provider import (
                LiteLLMEmbeddingProvider,
                LiteLLMProvider,
            )
        except ImportError as exc:
            raise ConfigError(
                "litellm is required for LLM/Embedding providers. "
                "Install it with: pip install litellm"
            ) from exc

        provider_name = config.llm.default_provider
        provider_cfg: dict[str, Any] = dict(config.llm.providers.get(provider_name, {}))

        # Extract well-known keys; forward everything else as **kwargs
        known_keys = {
            "model",
            "embedding_model",
            "embedding_dimensions",
            "temperature",
            "max_tokens",
        }
        model: str = provider_cfg.get("model", "gpt-4o-mini")
        embedding_model: str = provider_cfg.get("embedding_model", "text-embedding-3-small")
        embedding_dimensions: int = provider_cfg.get("embedding_dimensions", 1536)
        temperature: float = provider_cfg.get("temperature", 0.0)
        max_tokens: int = provider_cfg.get("max_tokens", 2048)
        # Remaining keys are provider-specific (api_key, api_base, etc.)
        extra_kwargs: dict[str, Any] = {
            k: v for k, v in provider_cfg.items() if k not in known_keys
        }

        llm_impl = LiteLLMProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        embedding_impl = LiteLLMEmbeddingProvider(
            model=embedding_model,
            dimensions=embedding_dimensions,
            **extra_kwargs,
        )
        self.register(LLMProvider, llm_impl)
        self.register(EmbeddingProvider, embedding_impl)
        logger.info(
            "LLM provider: %s (via %s) | Embedding provider: %s",
            model,
            provider_name,
            embedding_model,
        )

    def _wire_vector_store(self, config: MnemonConfig) -> None:
        """Register VectorStore implementation based on available extras."""
        from mnemon.core.interfaces import VectorStore

        backend = getattr(config.episodic, "backend", None) or "auto"

        if backend in ("qdrant", "auto") and _is_importable("qdrant_client"):
            impl = _lazy_init("mnemon.backends.qdrant_store", "QdrantVectorStore", config)
            self.register(VectorStore, impl)
            logger.info("VectorStore: QdrantVectorStore")
        elif backend in ("hnswlib", "auto") and _is_importable("hnswlib"):
            impl = _lazy_init("mnemon.backends.hnswlib_store", "HNSWLibVectorStore", config)
            self.register(VectorStore, impl)
            logger.info("VectorStore: HNSWLibVectorStore")
        elif backend == "auto" or backend == "memory":
            # In-memory fallback — always available
            impl = _lazy_init(
                "mnemon.backends.memory_store", "InMemoryVectorStore", config
            )
            self.register(VectorStore, impl)
            logger.info("VectorStore: InMemoryVectorStore (fallback)")
        else:
            raise BackendNotAvailableError(backend, "qdrant or hnswlib")

    def _wire_graph_store(self, config: MnemonConfig) -> None:
        """Register GraphStore implementation based on available extras."""
        from mnemon.core.interfaces import GraphStore

        backend = getattr(config.semantic, "graph_backend", None) or "auto"

        if backend in ("neo4j", "auto") and _is_importable("neo4j"):
            impl = _lazy_init("mnemon.backends.neo4j_store", "Neo4jGraphStore", config)
            self.register(GraphStore, impl)
            logger.info("GraphStore: Neo4jGraphStore")
        elif backend in ("falkordb", "auto") and _is_importable("falkordb"):
            impl = _lazy_init("mnemon.backends.falkordb_store", "FalkorDBGraphStore", config)
            self.register(GraphStore, impl)
            logger.info("GraphStore: FalkorDBGraphStore")
        elif backend in ("igraph", "auto") and _is_importable("igraph"):
            impl = _lazy_init("mnemon.backends.igraph_store", "IGraphGraphStore", config)
            self.register(GraphStore, impl)
            logger.info("GraphStore: IGraphGraphStore")
        elif backend == "auto" or backend == "memory":
            impl = _lazy_init(
                "mnemon.backends.memory_store", "InMemoryGraphStore", config
            )
            self.register(GraphStore, impl)
            logger.info("GraphStore: InMemoryGraphStore (fallback)")
        else:
            raise BackendNotAvailableError(backend, "neo4j, falkordb, or igraph")

    def _wire_document_store(self, config: MnemonConfig) -> None:
        """Register DocumentStore implementation based on available extras."""
        from mnemon.core.interfaces import DocumentStore

        # The document store shares the episodic backend selection.
        backend = getattr(config.episodic, "backend", None) or "auto"

        if backend in ("postgres", "auto") and _is_importable("asyncpg"):
            impl = _lazy_init(
                "mnemon.backends.postgres_store", "PostgresDocumentStore", config
            )
            self.register(DocumentStore, impl)
            logger.info("DocumentStore: PostgresDocumentStore")
        elif backend in ("sqlite", "auto") and _is_importable("aiosqlite"):
            impl = _lazy_init(
                "mnemon.backends.sqlite_store", "SQLiteDocumentStore", config
            )
            self.register(DocumentStore, impl)
            logger.info("DocumentStore: SQLiteDocumentStore")
        elif backend == "auto" or backend == "memory":
            impl = _lazy_init(
                "mnemon.backends.memory_store", "InMemoryDocumentStore", config
            )
            self.register(DocumentStore, impl)
            logger.info("DocumentStore: InMemoryDocumentStore (fallback)")
        else:
            raise BackendNotAvailableError(backend, "postgres or sqlite")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        names = [t.__name__ for t in self._services]
        return f"ModuleRegistry(registered={names})"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_importable(module_name: str) -> bool:
    """Return True if *module_name* can be imported (i.e. the extra is installed)."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _lazy_init(module_path: str, class_name: str, config: MnemonConfig) -> Any:
    """Import *class_name* from *module_path* and instantiate it with *config*.

    Parameters
    ----------
    module_path:
        Dotted Python module path (e.g. ``"mnemon.backends.qdrant_store"``).
    class_name:
        Name of the class to instantiate within that module.
    config:
        MnemonConfig passed as the sole constructor argument.

    Returns
    -------
    Any
        A new instance of the requested class.

    Raises
    ------
    ConfigError
        If the module or class cannot be imported.
    """
    try:
        module = importlib.import_module(module_path)
        klass = getattr(module, class_name)
        return klass(config)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(
            f"Failed to initialise backend {module_path}.{class_name}: {exc}"
        ) from exc
