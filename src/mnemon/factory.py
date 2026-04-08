"""
MnemonFactory — embryonic development of the cognitive framework.

Brain analog
------------
Just as a developing brain grows from a genetic blueprint (the genome) into a
fully wired connectome (trillions of synaptic connections), the factory reads a
declarative *config* (the blueprint) and assembles every cognitive module with
all inter-module dependencies correctly wired. The resulting ``Mnemon`` instance
is the finished brain: regions allocated, pathways connected, ready to run
cognitive cycles.

The factory is intentionally a one-shot constructor — call ``build()`` once at
startup, keep the ``Mnemon`` instance alive for the process lifetime, and call
``close()`` at shutdown.  Re-creating a factory mid-lifecycle is not supported
and will produce a fresh, amnesiac instance with empty memory stores.

Usage
-----
::

    from mnemon.factory import MnemonFactory

    factory = MnemonFactory()           # uses default MnemonConfig
    brain   = await factory.build()

    result  = await brain.run_cycle("What is the capital of France?")
    await brain.close()

To customise configuration::

    from mnemon.core.config import load_config, MnemonConfig
    from mnemon.factory import MnemonFactory

    config  = load_config("/path/to/mnemon.toml")
    factory = MnemonFactory(config)
    brain   = await factory.build()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio

from mnemon.core.bus import CognitiveBus
from mnemon.core.config import MnemonConfig
from mnemon.core.exceptions import ConfigError
from mnemon.core.interfaces import (
    DocumentStore,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    VectorStore,
)
from mnemon.core.registry import ModuleRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Namespace containers — thin attribute bags surfaced on the Mnemon facade
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MemoryNamespace:
    """Groups all memory-store instances under a single attribute namespace.

    Brain analog: The distributed memory network — hippocampus, neocortex,
    basal ganglia, amygdala, and their sensory/working entry points.
    """

    sensory: Any
    working: Any
    episodic: Any
    semantic: Any
    procedural: Any
    valence: Any


@dataclass(slots=True)
class _LearningNamespace:
    """Groups all learning-module instances.

    Brain analog: The synaptic plasticity machinery — dopaminergic RPE signals,
    hippocampal replay, and cortico-striatal habit formation circuits.
    """

    consolidation: Any
    reward: Any
    skill_acquirer: Any
    replay_buffer: Any
    scheduler: Any = None


@dataclass(slots=True)
class _ControlNamespace:
    """Groups cognitive-control module instances.

    Brain analog: The prefrontal executive network — lateral PFC for cycle
    orchestration, basal forebrain for attention gating, anterior PFC for goal
    management, and ACC for error monitoring and strategy adjustment.
    """

    attention: Any
    goals: Any
    meta_cognition: Any


# ---------------------------------------------------------------------------
# Mnemon facade
# ---------------------------------------------------------------------------


class Mnemon:
    """The assembled Mnemon cognitive framework.

    A high-level facade that exposes the entire cognitive framework through
    four top-level attributes (``memory``, ``learning``, ``control``, ``bus``)
    and delegates the core execution API to the underlying ``Orchestrator``.

    Brain analog: The complete, mature brain — all regions formed, all pathways
    myelinated, fully capable of integrated perception, memory, and action.

    Attributes
    ----------
    orchestrator:
        The central executive that drives the cognitive cycle.
    memory:
        Namespace exposing ``.sensory``, ``.working``, ``.episodic``,
        ``.semantic``, ``.procedural``, and ``.valence`` stores.
    learning:
        Namespace exposing ``.consolidation``, ``.reward``,
        ``.skill_acquirer``, and ``.replay_buffer``.
    control:
        Namespace exposing ``.attention``, ``.goals``, and ``.meta_cognition``.
    bus:
        The thalamic relay bus for inter-module messaging.
    config:
        The configuration snapshot used to build this instance.
    """

    def __init__(
        self,
        orchestrator: Any,
        memory: _MemoryNamespace,
        learning: _LearningNamespace,
        control: _ControlNamespace,
        bus: CognitiveBus,
        config: MnemonConfig,
        _backends: list[Any],
    ) -> None:
        self.orchestrator = orchestrator
        self.memory = memory
        self.learning = learning
        self.control = control
        self.bus = bus
        self.config = config
        self._backends = _backends
        self._task_group: anyio.abc.TaskGroup | None = None
        self._cancel_scope: anyio.CancelScope | None = None

    # ------------------------------------------------------------------
    # Async context manager — starts/stops the bus with a managed TaskGroup
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "Mnemon":
        """Start the CognitiveBus within a managed TaskGroup.

        Use ``async with`` to ensure the bus is properly started and stopped::

            brain = await factory.build()
            async with brain:
                result = await brain.run_cycle("Hello")
        """
        self._cancel_scope = anyio.CancelScope()
        # Create a task group for the bus dispatch loop.
        # We store a reference so close() can cancel it.
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        await self.bus.start(self._task_group)
        logger.info("Mnemon context entered — CognitiveBus started.")
        if self.learning.scheduler is not None:
            try:
                await self.learning.scheduler.start()
            except Exception as exc:
                logger.warning(
                    "ConsolidationScheduler failed to start (non-fatal): %s", exc
                )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the bus, scheduler, and clean up the task group."""
        if self.learning.scheduler is not None:
            try:
                await self.learning.scheduler.stop()
            except Exception as exc:
                logger.warning(
                    "ConsolidationScheduler failed to stop cleanly (non-fatal): %s", exc
                )
        await self.bus.stop()
        if self._task_group is not None:
            # Cancel the task group to stop the dispatch loop coroutine
            self._task_group.cancel_scope.cancel()
            try:
                await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass  # Task group may raise on cancel; safe to suppress
            self._task_group = None
        logger.info("Mnemon context exited — CognitiveBus stopped.")

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    async def run_cycle(self, raw_input: str | None = None) -> dict[str, Any]:
        """Execute one full perception–retrieval–reasoning–action cycle.

        Delegates entirely to the underlying ``Orchestrator``.

        Parameters
        ----------
        raw_input:
            Optional stimulus string injected at the start of the cycle.
            Pass ``None`` to run an internally-driven cycle (e.g. for
            background consolidation or goal pursuit without new input).

        Returns
        -------
        dict[str, Any]
            Cycle summary produced by the Orchestrator, including actions
            taken, memories updated, and current goal status.
        """
        return await self.orchestrator.run_cycle(raw_input=raw_input)

    async def run_until_complete(
        self,
        goal: Any,
        max_cycles: int = 10,
    ) -> dict[str, Any]:
        """Run consecutive cognitive cycles until *goal* is completed.

        Delegates to the underlying ``Orchestrator``.

        Parameters
        ----------
        goal:
            Terminal ``Goal`` that determines when to stop cycling.
        max_cycles:
            Hard upper bound on cycle count to prevent infinite loops.

        Returns
        -------
        dict[str, Any]
            Final summary including success/failure status and total
            cycle count.
        """
        return await self.orchestrator.run_until_complete(
            goal=goal, max_cycles=max_cycles
        )

    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the orchestrator's current internal state.

        Returns
        -------
        dict[str, Any]
            State dictionary produced by ``Orchestrator.get_state()``.
        """
        return self.orchestrator.get_state()

    async def close(self) -> None:
        """Gracefully shut down all cognitive framework components.

        Sequence:

        1. Stop the ``CognitiveBus`` dispatch loop (drain pending messages).
        2. Call ``close()`` on any backends that expose it (e.g. SQLite
           connection pools, Qdrant client sessions, FalkorDB connections).

        Safe to call multiple times — subsequent calls are no-ops.
        """
        logger.info("Mnemon.close() — initiating graceful shutdown.")

        await self.bus.stop()
        logger.debug("CognitiveBus stopped.")

        for backend in self._backends:
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                try:
                    result = close_fn()
                    # Support both sync and async close methods.
                    if hasattr(result, "__await__"):
                        await result
                    logger.debug("Closed backend %s.", type(backend).__name__)
                except Exception as exc:
                    # Log but do not re-raise — best-effort shutdown.
                    logger.warning(
                        "Error closing backend %s: %s",
                        type(backend).__name__,
                        exc,
                    )

        logger.info("Mnemon shutdown complete.")


# ---------------------------------------------------------------------------
# MnemonFactory
# ---------------------------------------------------------------------------


class MnemonFactory:
    """Assembles the entire Mnemon cognitive framework from configuration.

    Brain analog: Embryonic neural development — the factory encodes the
    developmental program that grows a blank substrate (raw config) into a
    fully connected cognitive system.  Each ``build()`` call is equivalent
    to one developmental pass: region allocation (module instantiation),
    axon growth (dependency injection), and synaptogenesis (bus wiring).

    Parameters
    ----------
    config:
        Root configuration object.  Defaults to ``MnemonConfig()`` (all
        built-in defaults, with environment-variable overrides applied).

    Examples
    --------
    Default build (suitable for local development)::

        brain = await MnemonFactory().build()

    Production build from a TOML file::

        from mnemon.core.config import load_config
        config = load_config("/etc/mnemon/production.toml")
        brain  = await MnemonFactory(config).build()
    """

    def __init__(self, config: MnemonConfig | None = None) -> None:
        self._config: MnemonConfig = config if config is not None else MnemonConfig()
        logger.debug(
            "MnemonFactory created (llm_provider=%s).",
            self._config.llm.default_provider,
        )

    # ------------------------------------------------------------------
    # Public factory method
    # ------------------------------------------------------------------

    async def build(self) -> Mnemon:
        """Construct and return a fully-wired ``Mnemon`` instance.

        Stages
        ------
        1.  Build ``ModuleRegistry`` — wires storage backends and LLM
            providers from config, with auto-detection and fallbacks.
        2.  Resolve infrastructure providers from the registry.
        3.  Instantiate all six memory stores.
        4.  Instantiate learning modules (replay buffer first, because
            ``ConsolidationEngine`` depends on it).
        5.  Instantiate cognitive-control modules.
        6.  Create ``CognitiveBus``.
        7.  Assemble ``Orchestrator`` with full dependency graph.
        8.  Call ``initialize()`` on any backends that expose it.
        9.  Wrap everything in a ``Mnemon`` facade and return.

        Returns
        -------
        Mnemon
            Ready-to-use cognitive framework instance.

        Raises
        ------
        ConfigError
            If registry wiring, module instantiation, or backend
            initialisation fails for any reason.
        """
        config = self._config

        # ----------------------------------------------------------
        # Stage 1: Registry for LLM/embedding providers only
        # ----------------------------------------------------------
        try:
            logger.info("MnemonFactory.build() — wiring ModuleRegistry (LLM/embedding only).")
            registry: ModuleRegistry = ModuleRegistry()
            registry._wire_llm_providers(config)
            llm: LLMProvider = registry.resolve(LLMProvider)
            embedder: EmbeddingProvider = registry.resolve(EmbeddingProvider)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"ModuleRegistry assembly failed: {exc}"
            ) from exc

        logger.info(
            "Resolved LLM/embedding providers — LLM=%s Embedder=%s",
            type(llm).__name__,
            type(embedder).__name__,
        )

        # ----------------------------------------------------------
        # Stage 2: Per-memory-type isolated stores (no sharing)
        # ----------------------------------------------------------
        try:
            state_dir = Path("~/.mnemon/").expanduser()
            episodic_vs, episodic_ds = self._make_stores(config, "episodic", state_dir)
            semantic_vs, semantic_ds = self._make_stores(config, "semantic", state_dir)
            procedural_vs, procedural_ds = self._make_stores(config, "procedural", state_dir)
            semantic_gs = self._make_graph_store(config, "semantic", state_dir)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"Isolated store creation failed: {exc}"
            ) from exc

        logger.info(
            "Created isolated stores — "
            "episodic: VectorStore=%s DocumentStore=%s | "
            "semantic: VectorStore=%s DocumentStore=%s GraphStore=%s | "
            "procedural: VectorStore=%s DocumentStore=%s",
            type(episodic_vs).__name__,
            type(episodic_ds).__name__,
            type(semantic_vs).__name__,
            type(semantic_ds).__name__,
            type(semantic_gs).__name__,
            type(procedural_vs).__name__,
            type(procedural_ds).__name__,
        )

        # ----------------------------------------------------------
        # Stage 3: Memory stores (using isolated backends)
        # ----------------------------------------------------------
        try:
            sensory = self._build_sensory(config, embedder)
            working = self._build_working(config, llm)
            episodic = self._build_episodic(config, episodic_vs, episodic_ds, embedder)
            semantic = self._build_semantic(config, semantic_gs, semantic_vs, semantic_ds, embedder, llm)
            procedural = self._build_procedural(config, procedural_vs, procedural_ds, embedder)
            valence = self._build_valence(config, embedder)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"Memory store instantiation failed: {exc}"
            ) from exc

        logger.info("All six memory stores instantiated.")

        # ----------------------------------------------------------
        # Stage 4: Learning modules
        # ----------------------------------------------------------
        try:
            replay_buffer = self._build_replay_buffer(config)
            reward_processor = self._build_reward_processor(config)
            consolidation = self._build_consolidation(
                config, episodic, semantic, llm, embedder, replay_buffer
            )
            skill_acquirer = self._build_skill_acquirer(
                config, procedural, episodic, llm, embedder
            )
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"Learning module instantiation failed: {exc}"
            ) from exc

        logger.info("Learning modules instantiated.")

        # ----------------------------------------------------------
        # Stage 4b: Consolidation scheduler (optional — requires apscheduler)
        # ----------------------------------------------------------
        scheduler = self._build_scheduler(config, consolidation)

        # ----------------------------------------------------------
        # Stage 5: Cognitive-control modules
        # ----------------------------------------------------------
        try:
            attention = self._build_attention(config, valence)
            goal_manager = self._build_goal_manager(llm)
            meta_cognition = self._build_meta_cognition(config, llm)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"Cognitive-control module instantiation failed: {exc}"
            ) from exc

        logger.info("Cognitive-control modules instantiated.")

        # ----------------------------------------------------------
        # Stage 6: CognitiveBus
        # ----------------------------------------------------------
        bus = CognitiveBus()
        logger.info("CognitiveBus created.")

        # ----------------------------------------------------------
        # Stage 7: Orchestrator
        # ----------------------------------------------------------
        try:
            orchestrator = self._build_orchestrator(
                config=config,
                sensory=sensory,
                working_memory=working,
                episodic=episodic,
                semantic=semantic,
                procedural=procedural,
                valence=valence,
                attention=attention,
                goal_manager=goal_manager,
                meta_cognition=meta_cognition,
                reward_processor=reward_processor,
                embedding_provider=embedder,
                bus=bus,
            )
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"Orchestrator instantiation failed: {exc}"
            ) from exc

        logger.info("Orchestrator assembled.")

        # ----------------------------------------------------------
        # Stage 8: Backend initialisation
        # ----------------------------------------------------------
        backends = [
            episodic_vs, episodic_ds,
            semantic_vs, semantic_ds, semantic_gs,
            procedural_vs, procedural_ds,
        ]
        await self._initialize_backends(backends)

        # ----------------------------------------------------------
        # Stage 9: Assemble and return Mnemon facade
        # ----------------------------------------------------------
        memory_ns = _MemoryNamespace(
            sensory=sensory,
            working=working,
            episodic=episodic,
            semantic=semantic,
            procedural=procedural,
            valence=valence,
        )
        learning_ns = _LearningNamespace(
            consolidation=consolidation,
            reward=reward_processor,
            skill_acquirer=skill_acquirer,
            replay_buffer=replay_buffer,
            scheduler=scheduler,
        )
        control_ns = _ControlNamespace(
            attention=attention,
            goals=goal_manager,
            meta_cognition=meta_cognition,
        )

        mnemon = Mnemon(
            orchestrator=orchestrator,
            memory=memory_ns,
            learning=learning_ns,
            control=control_ns,
            bus=bus,
            config=config,
            _backends=backends,
        )

        logger.info(
            "MnemonFactory.build() complete — cognitive framework is online."
        )
        return mnemon

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Per-memory-type isolated store factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stores(
        config: MnemonConfig, namespace: str, state_dir: Path
    ) -> tuple[Any, Any]:
        """Create isolated VectorStore + DocumentStore for a memory namespace."""
        vector_store = MnemonFactory._make_vector_store(config, namespace, state_dir)
        document_store = MnemonFactory._make_document_store(config, namespace, state_dir)
        return vector_store, document_store

    @staticmethod
    def _make_vector_store(
        config: MnemonConfig, namespace: str, state_dir: Path
    ) -> Any:
        """Select and instantiate the appropriate VectorStore for a namespace."""
        import importlib

        backend = getattr(config.episodic, "backend", "auto")

        def _importable(m: str) -> bool:
            try:
                importlib.import_module(m)
                return True
            except ImportError:
                return False

        if _importable("hnswlib") and backend not in ("memory", "postgres", "sqlite"):
            from mnemon.backends.hnswlib_store import HNSWLibVectorStore
            path = state_dir / "vectors" / f"{namespace}.hnsw"
            return HNSWLibVectorStore(index_path=str(path))
        elif _importable("qdrant_client") and backend == "qdrant":
            from mnemon.backends.qdrant_store import QdrantVectorStore
            return QdrantVectorStore(config)
        else:
            from mnemon.backends.memory_store import InMemoryVectorStore
            return InMemoryVectorStore(config)

    @staticmethod
    def _make_document_store(
        config: MnemonConfig, namespace: str, state_dir: Path
    ) -> Any:
        """Select and instantiate the appropriate DocumentStore for a namespace."""
        import importlib

        def _importable(m: str) -> bool:
            try:
                importlib.import_module(m)
                return True
            except ImportError:
                return False

        if _importable("aiosqlite"):
            from mnemon.backends.sqlite_store import SQLiteDocumentStore
            db_path = str(state_dir / "documents" / f"{namespace}.db")
            return SQLiteDocumentStore(config, db_path=db_path)
        else:
            from mnemon.backends.memory_store import InMemoryDocumentStore
            return InMemoryDocumentStore(config)

    @staticmethod
    def _make_graph_store(
        config: MnemonConfig, namespace: str, state_dir: Path
    ) -> Any:
        """Select and instantiate the appropriate GraphStore for a namespace."""
        import importlib

        backend = getattr(config.semantic, "graph_backend", "auto")

        def _importable(m: str) -> bool:
            try:
                importlib.import_module(m)
                return True
            except ImportError:
                return False

        if _importable("igraph") and backend not in ("memory", "falkordb", "neo4j"):
            from mnemon.backends.igraph_store import IGraphGraphStore
            path = state_dir / "graphs" / f"{namespace}.json"
            return IGraphGraphStore(graph_path=str(path))
        elif _importable("falkordb") and backend == "falkordb":
            from mnemon.backends.falkordb_store import FalkorDBGraphStore
            return FalkorDBGraphStore(config)
        else:
            from mnemon.backends.memory_store import InMemoryGraphStore
            return InMemoryGraphStore(config)

    # ------------------------------------------------------------------
    # Private stage helpers — each is a thin wrapper that provides a
    # single place for per-module error context without polluting build()
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sensory(config: MnemonConfig, embedder: EmbeddingProvider) -> Any:
        """Instantiate the SensoryBuffer (primary sensory cortex analog)."""
        from mnemon.memory.sensory import SensoryBuffer

        # SensoryBuffer's current implementation does not use the embedder
        # directly but we accept it here to honour the documented interface
        # and leave room for future modality-specific encoders.
        try:
            return SensoryBuffer(config=config.sensory)
        except Exception as exc:
            raise ConfigError(
                f"SensoryBuffer initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_working(config: MnemonConfig, llm: LLMProvider) -> Any:
        """Instantiate WorkingMemoryManager (dlPFC analog)."""
        from mnemon.memory.working import WorkingMemoryManager

        try:
            return WorkingMemoryManager(config=config.working_memory, llm=llm)
        except Exception as exc:
            raise ConfigError(
                f"WorkingMemoryManager initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_episodic(
        config: MnemonConfig,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedder: EmbeddingProvider,
    ) -> Any:
        """Instantiate EpisodicMemoryStore (hippocampal formation analog)."""
        from mnemon.memory.episodic import EpisodicMemoryStore

        try:
            return EpisodicMemoryStore(
                config=config.episodic,
                vector_store=vector_store,
                document_store=document_store,
                embedding_provider=embedder,
            )
        except Exception as exc:
            raise ConfigError(
                f"EpisodicMemoryStore initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_semantic(
        config: MnemonConfig,
        graph_store: GraphStore,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedder: EmbeddingProvider,
        llm: LLMProvider | None = None,
    ) -> Any:
        """Instantiate SemanticMemoryStore (neocortical association area analog)."""
        from mnemon.memory.semantic import SemanticMemoryStore

        try:
            return SemanticMemoryStore(
                config=config.semantic,
                graph_store=graph_store,
                vector_store=vector_store,
                document_store=document_store,
                embedding_provider=embedder,
                llm_provider=llm,
            )
        except Exception as exc:
            raise ConfigError(
                f"SemanticMemoryStore initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_procedural(
        config: MnemonConfig,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedder: EmbeddingProvider,
    ) -> Any:
        """Instantiate ProceduralMemoryStore (basal ganglia / cerebellum analog)."""
        from mnemon.memory.procedural import ProceduralMemoryStore

        try:
            return ProceduralMemoryStore(
                config=config.procedural,
                vector_store=vector_store,
                document_store=document_store,
                embedding_provider=embedder,
            )
        except Exception as exc:
            raise ConfigError(
                f"ProceduralMemoryStore initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_valence(
        config: MnemonConfig,
        embedder: EmbeddingProvider,
    ) -> Any:
        """Instantiate ValenceMemoryStore (amygdala analog)."""
        from mnemon.memory.valence import ValenceMemoryStore

        try:
            return ValenceMemoryStore(
                config=config.valence,
                embedding_provider=embedder,
            )
        except Exception as exc:
            raise ConfigError(
                f"ValenceMemoryStore initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_reward_processor(config: MnemonConfig) -> Any:
        """Instantiate RewardProcessor (VTA/Substantia Nigra dopaminergic analog)."""
        from mnemon.learning.reward import RewardProcessor

        try:
            return RewardProcessor(config=config.reward)
        except Exception as exc:
            raise ConfigError(
                f"RewardProcessor initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_replay_buffer(config: MnemonConfig) -> Any:
        """Instantiate PrioritizedReplayBuffer (hippocampal sharp-wave ripple analog).

        Capacity is derived from ``episodic.capacity.max_episodes`` so the
        replay buffer is never larger than the episodic store itself.
        """
        from mnemon.learning.replay import PrioritizedReplayBuffer

        capacity = config.episodic.capacity.max_episodes
        alpha = config.consolidation.replay.alpha
        beta_start = config.consolidation.replay.beta_start
        try:
            return PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=alpha,
                beta_start=beta_start,
            )
        except Exception as exc:
            raise ConfigError(
                f"PrioritizedReplayBuffer initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_consolidation(
        config: MnemonConfig,
        episodic: Any,
        semantic: Any,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        replay_buffer: Any,
    ) -> Any:
        """Instantiate ConsolidationEngine (slow-wave sleep replay analog)."""
        from mnemon.learning.consolidation import ConsolidationEngine

        try:
            return ConsolidationEngine(
                config=config.consolidation,
                episodic_memory=episodic,
                semantic_memory=semantic,
                llm=llm,
                embedding_provider=embedder,
                replay_buffer=replay_buffer,
            )
        except Exception as exc:
            raise ConfigError(
                f"ConsolidationEngine initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_scheduler(config: MnemonConfig, consolidation: Any) -> Any:
        """Attempt to create a ConsolidationScheduler if apscheduler is installed.

        Returns ``None`` silently when APScheduler is not available so the
        rest of the factory pipeline is unaffected.

        Brain analog: Wiring the suprachiasmatic nucleus into the consolidation
        pathway — present and active only when the circadian clock is enabled.
        """
        try:
            from mnemon.scheduling.scheduler import (
                ConsolidationScheduler,
                _APSCHEDULER_AVAILABLE,
            )
        except ImportError:
            logger.debug(
                "mnemon.scheduling not importable — ConsolidationScheduler skipped."
            )
            return None

        if not _APSCHEDULER_AVAILABLE:
            logger.debug(
                "APScheduler not installed — ConsolidationScheduler will be unavailable. "
                "Install with: pip install 'mnemon[scheduler]'"
            )
            return None

        try:
            scheduler = ConsolidationScheduler(
                config=config.consolidation.schedule,
                consolidation_engine=consolidation,
            )
            logger.info("ConsolidationScheduler created (mode=%s).", config.consolidation.schedule.mode)
            return scheduler
        except Exception as exc:
            logger.warning(
                "ConsolidationScheduler instantiation failed (non-fatal): %s", exc
            )
            return None

    @staticmethod
    def _build_skill_acquirer(
        config: MnemonConfig,
        procedural: Any,
        episodic: Any,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
    ) -> Any:
        """Instantiate SkillAcquirer (cortico-striatal habit formation analog)."""
        from mnemon.learning.skill_acquirer import SkillAcquirer

        try:
            return SkillAcquirer(
                config=config.procedural,
                procedural_memory=procedural,
                episodic_memory=episodic,
                llm=llm,
                embedding_provider=embedder,
            )
        except Exception as exc:
            raise ConfigError(
                f"SkillAcquirer initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_attention(config: MnemonConfig, valence: Any) -> Any:
        """Instantiate AttentionController (basal forebrain cholinergic analog)."""
        from mnemon.control.attention import AttentionController

        try:
            return AttentionController(config=config.attention, valence=valence)
        except Exception as exc:
            raise ConfigError(
                f"AttentionController initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_goal_manager(llm: LLMProvider) -> Any:
        """Instantiate GoalManager (anterior prefrontal cortex analog)."""
        from mnemon.control.goals import GoalManager

        try:
            return GoalManager(llm=llm)
        except Exception as exc:
            raise ConfigError(
                f"GoalManager initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_meta_cognition(config: MnemonConfig, llm: LLMProvider) -> Any:
        """Instantiate MetaCognitionController (ACC analog)."""
        from mnemon.control.metacognition import MetaCognitionController

        try:
            return MetaCognitionController(config=config.meta_cognition, llm=llm)
        except Exception as exc:
            raise ConfigError(
                f"MetaCognitionController initialisation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_orchestrator(
        config: MnemonConfig,
        sensory: Any,
        working_memory: Any,
        episodic: Any,
        semantic: Any,
        procedural: Any,
        valence: Any,
        attention: Any,
        goal_manager: Any,
        meta_cognition: Any,
        reward_processor: Any,
        embedding_provider: EmbeddingProvider,
        bus: CognitiveBus,
    ) -> Any:
        """Instantiate the Orchestrator with the full dependency graph injected."""
        from mnemon.control.orchestrator import Orchestrator

        try:
            return Orchestrator(
                config=config,
                sensory=sensory,
                working_memory=working_memory,
                episodic=episodic,
                semantic=semantic,
                procedural=procedural,
                valence=valence,
                attention=attention,
                goal_manager=goal_manager,
                meta_cognition=meta_cognition,
                reward_processor=reward_processor,
                embedding_provider=embedding_provider,
                bus=bus,
            )
        except Exception as exc:
            raise ConfigError(
                f"Orchestrator instantiation failed: {exc}"
            ) from exc

    @staticmethod
    async def _initialize_backends(backends: list[Any]) -> None:
        """Call ``initialize()`` on each backend that exposes it.

        Backends such as ``QdrantVectorStore``, ``FalkorDBGraphStore``, and
        ``SQLiteDocumentStore`` require an async ``initialize()`` call to
        create collections / tables and establish connection pools before
        any memory operations can proceed.

        Parameters
        ----------
        backends:
            List of storage backend instances to initialise in sequence.

        Raises
        ------
        ConfigError
            If any backend's ``initialize()`` raises an exception.
        """
        for backend in backends:
            init_fn = getattr(backend, "initialize", None)
            if callable(init_fn):
                backend_name = type(backend).__name__
                try:
                    logger.info("Initialising backend %s …", backend_name)
                    result = init_fn()
                    if hasattr(result, "__await__"):
                        await result
                    logger.info("Backend %s initialised.", backend_name)
                except Exception as exc:
                    raise ConfigError(
                        f"Backend '{backend_name}' failed during initialize(): {exc}"
                    ) from exc
