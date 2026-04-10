"""
Abstract interfaces (ABCs) for all Mnemon cognitive modules and storage backends.

Design philosophy
-----------------
Every interface in this module has a single responsibility (SRP) and maps to a
distinct functional region of the brain.  High-level cognitive modules depend
*only* on these abstractions — never on concrete backends — enforcing the
Dependency Inversion Principle throughout the framework.

Adding a new backend means implementing one of these ABCs; no existing code
needs to change (Open/Closed Principle).  Any conforming implementation can be
swapped in place of any other (Liskov Substitution Principle).  VectorStore,
GraphStore, and DocumentStore are intentionally separate rather than merged
into a monolithic store interface (Interface Segregation Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel

from mnemon.core.models import (
    Community,
    ConsolidationResult,
    ContextBlock,
    Entity,
    EntityRef,
    Episode,
    GateDecision,
    Goal,
    GoalStatus,
    MetaEvaluation,
    Modality,
    PerceptUnit,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    RewardSignal,
    SalienceScore,
    SemanticTriple,
    Skill,
    Strategy,
    ValenceAssociation,
    WorkingMemoryState,
)

# ---------------------------------------------------------------------------
# Supplementary types used by storage interfaces
# (kept here — not in models.py — to avoid circular imports)
# ---------------------------------------------------------------------------


class VectorSearchResult(BaseModel):
    """A single hit returned by a vector-similarity search."""

    id: UUID
    score: float
    metadata: dict[str, Any]


class VectorItem(BaseModel):
    """A unit of data to be inserted into a vector store."""

    id: UUID
    embedding: list[float]
    metadata: dict[str, Any]


class GraphNode(BaseModel):
    """A node retrieved from the knowledge graph."""

    id: UUID
    labels: list[str]
    properties: dict[str, Any]


class RankedNode(BaseModel):
    """A knowledge-graph node with an associated activation / PageRank score."""

    id: UUID
    score: float
    properties: dict[str, Any]


# ---------------------------------------------------------------------------
# Storage Provider Interfaces (SPIs)
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Storage backend for dense vector embeddings.

    Brain analog: The indexing substrate used by hippocampal and neocortical
    memory systems for similarity-based retrieval (pattern completion).
    Implementations may use approximate nearest-neighbour indices (HNSW,
    IVF-PQ) or exact search depending on scale requirements.
    """

    @abstractmethod
    async def insert(
        self,
        id: UUID,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Persist a single embedding with associated metadata.

        Parameters
        ----------
        id:
            Stable identifier that links this vector to its source document.
        embedding:
            Dense float vector produced by an EmbeddingProvider.
        metadata:
            Arbitrary key/value payload stored alongside the vector and
            returned in search results for post-processing.
        """

    @abstractmethod
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
            The probe vector to compare against the index.
        top_k:
            Maximum number of results to return.
        filters:
            Optional metadata predicates applied before or after ANN search
            (implementation-dependent).

        Returns
        -------
        list[VectorSearchResult]
            Results in descending similarity order.
        """

    @abstractmethod
    async def delete(self, id: UUID) -> None:
        """Remove the vector identified by *id* from the index."""

    @abstractmethod
    async def update(
        self,
        id: UUID,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Replace the embedding and metadata for an existing entry.

        Implementations that do not support in-place updates should
        perform a delete followed by insert atomically where possible.
        """

    @abstractmethod
    async def bulk_insert(self, items: list[VectorItem]) -> None:
        """Batch-insert multiple vectors in a single operation.

        Preferred over repeated ``insert`` calls for large ingestion
        workloads; backends may exploit batched upsert APIs.
        """

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of vectors currently in the index."""


class GraphStore(ABC):
    """Storage backend for knowledge graph operations.

    Brain analog: The neocortical association network where concepts are
    linked by semantic relationships.  Supports spreading activation
    (Personalized PageRank) for associative retrieval — mirroring how
    activation spreads through cortical columns during recall.
    """

    @abstractmethod
    async def add_node(
        self,
        node_id: UUID,
        labels: list[str],
        properties: dict[str, Any],
    ) -> None:
        """Insert or upsert a node with the given labels and properties.

        Parameters
        ----------
        node_id:
            Stable UUID used as the primary key across all stores.
        labels:
            Semantic type tags (e.g. ``["Person"]``, ``["Concept"]``).
        properties:
            Arbitrary node attributes persisted alongside the labels.
        """

    @abstractmethod
    async def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a directed edge between two existing nodes.

        Parameters
        ----------
        source_id:
            UUID of the edge's origin node.
        target_id:
            UUID of the edge's destination node.
        edge_type:
            Relationship label (e.g. ``"IS_A"``, ``"CAUSED_BY"``).
        properties:
            Optional edge-level attributes such as weight or timestamp.
        """

    @abstractmethod
    async def get_node(self, node_id: UUID) -> dict[str, Any] | None:
        """Fetch a single node by its UUID.

        Returns ``None`` if no node with *node_id* exists.
        """

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: str | None = None,
        direction: str = "out",
        max_hops: int = 1,
    ) -> list[GraphNode]:
        """Return nodes reachable from *node_id* within *max_hops* steps.

        Parameters
        ----------
        node_id:
            Starting node for the traversal.
        edge_type:
            If provided, only traverse edges of this type.
        direction:
            ``"out"`` (default), ``"in"``, or ``"both"``.
        max_hops:
            Maximum graph distance to explore.
        """

    @abstractmethod
    async def run_pagerank(
        self,
        seed_ids: list[UUID],
        damping: float = 0.85,
        max_iterations: int = 100,
    ) -> list[RankedNode]:
        """Compute Personalised PageRank seeded from *seed_ids*.

        Models spreading activation: high-ranked nodes are those most
        closely associated with the seed concepts through the graph topology.

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
            All reachable nodes sorted by descending PageRank score.
        """

    @abstractmethod
    async def run_community_detection(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
    ) -> list[list[UUID]]:
        """Partition the graph into communities of related concepts.

        Parameters
        ----------
        algorithm:
            Community detection algorithm to use (e.g. ``"louvain"``,
            ``"leiden"``, ``"label_propagation"``).
        resolution:
            Modularity resolution parameter; higher values produce more,
            smaller communities.

        Returns
        -------
        list[list[UUID]]
            Each inner list is the set of node IDs belonging to one community.
        """

    @abstractmethod
    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a raw graph query (Cypher or GQL dialect).

        Parameters
        ----------
        cypher:
            Query string in the backend's supported dialect.
        params:
            Bound parameters for the query (avoids injection).

        Returns
        -------
        list[dict[str, Any]]
            Each dict is one result row keyed by column name.
        """

    @abstractmethod
    async def delete_node(self, node_id: UUID) -> None:
        """Remove a node and all its incident edges from the graph."""

    @abstractmethod
    async def node_count(self) -> int:
        """Return the total number of nodes in the graph."""

    @abstractmethod
    async def edge_count(self) -> int:
        """Return the total number of edges in the graph."""


class DocumentStore(ABC):
    """Storage backend for structured document/record persistence.

    Brain analog: The physical substrate of long-term memory storage —
    the synaptic patterns that encode specific memories or knowledge.
    Implementations may be backed by SQLite, PostgreSQL, DynamoDB, etc.
    """

    @abstractmethod
    async def put(self, id: UUID, document: dict[str, Any]) -> None:
        """Insert or replace a document identified by *id*."""

    @abstractmethod
    async def get(self, id: UUID) -> dict[str, Any] | None:
        """Fetch a single document by its UUID.

        Returns ``None`` if no document with *id* exists.
        """

    @abstractmethod
    async def query(
        self,
        filters: dict[str, Any],
        sort_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query documents matching *filters*.

        Parameters
        ----------
        filters:
            Key/value equality predicates.  Backends may extend this with
            range operators or full-text search.
        sort_by:
            Field name to sort ascending.  ``None`` means unspecified order.
        limit:
            Maximum number of results.
        offset:
            Number of matching documents to skip (for pagination).
        """

    @abstractmethod
    async def delete(self, id: UUID) -> None:
        """Permanently remove a document from the store."""

    @abstractmethod
    async def bulk_put(self, items: list[tuple[UUID, dict[str, Any]]]) -> None:
        """Insert or replace multiple documents in a single operation."""

    @abstractmethod
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Return the number of documents matching *filters* (all if ``None``)."""


# ---------------------------------------------------------------------------
# Provider Interfaces
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Provides dense vector embeddings for text.

    Brain analog: The neural encoding process that transforms raw sensory
    input into distributed representations suitable for storage and comparison.
    Different models produce different representational geometries, analogous
    to how different cortical areas encode the same stimulus differently.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Produce a single embedding vector for *text*."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Produce embeddings for a batch of texts.

        Implementations should use the provider's native batching API
        where available for efficiency.  The returned list is aligned
        with the input: ``result[i]`` is the embedding of ``texts[i]``.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the vectors produced by this provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for the underlying model."""


class LLMProvider(ABC):
    """Provides language model inference for cognitive operations.

    Brain analog: The general-purpose neural computation substrate used
    for reasoning, extraction, summarisation, and deliberation.  Analogous
    to the associative neocortex that integrates information across modalities
    to produce coherent, contextualised outputs.
    """

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a free-form text completion for *prompt*.

        Parameters
        ----------
        prompt:
            Full prompt string including any system context.
        **kwargs:
            Provider-specific overrides (temperature, max_tokens, etc.).

        Returns
        -------
        str
            The generated text, stripped of any wrapping metadata.
        """

    async def generate_chat(
        self,
        system: str,
        history: list[dict[str, str]],
        message: str,
        **kwargs: Any,
    ) -> str:
        """Generate a reply given a system prompt, conversation history, and new message.

        Default implementation concatenates everything into a single prompt
        and calls ``generate``.  Subclasses may override to use native
        multi-turn message APIs.

        Parameters
        ----------
        system:
            System / persona prompt.
        history:
            List of ``{"role": "user"|"assistant", "content": str}`` dicts,
            oldest first.
        message:
            The latest user message.
        """
        parts = [system]
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append(f"User: {message}")
        parts.append("Assistant:")
        return await self.generate("\n\n".join(parts), **kwargs)

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a response constrained to *response_schema* (JSON Schema).

        Used for extraction tasks where the output must conform to a known
        structure.  Implementations should use native structured-output APIs
        (e.g. OpenAI response_format, Anthropic tool use) where available.

        Parameters
        ----------
        prompt:
            Full prompt string.
        response_schema:
            JSON Schema dict describing the expected response shape.
        **kwargs:
            Provider-specific overrides.

        Returns
        -------
        dict[str, Any]
            Parsed JSON conforming to *response_schema*.
        """

    @abstractmethod
    async def token_count(self, text: str) -> int:
        """Return the number of tokens *text* would consume for this model."""


# ---------------------------------------------------------------------------
# Memory Module Interfaces
# ---------------------------------------------------------------------------


class SensoryBufferInterface(ABC):
    """Pre-attentive input processing.

    Brain analog: Primary sensory cortices (V1, A1) — the earliest
    cortical processing stages that transduce raw stimuli into
    normalised, modality-specific representations before attention
    gates further processing.
    """

    @abstractmethod
    async def process(
        self,
        raw_input: str,
        modality: Modality = Modality.TEXT,
    ) -> PerceptUnit:
        """Transform *raw_input* into a normalised PerceptUnit.

        Parameters
        ----------
        raw_input:
            The unprocessed stimulus string.
        modality:
            The sensory channel through which the input arrived.

        Returns
        -------
        PerceptUnit
            Normalised percept with computed salience ready for attention gating.
        """

    @abstractmethod
    def peek(self) -> list[PerceptUnit]:
        """Return buffered percepts without consuming them."""

    @abstractmethod
    def clear(self) -> None:
        """Discard all buffered percepts (sensory decay)."""


class WorkingMemoryInterface(ABC):
    """Active context management with token-budget constraints.

    Brain analog: Dorsolateral prefrontal cortex (dlPFC) — the limited-
    capacity workspace that holds task-relevant information in an active,
    accessible state.  Token budget models the biological capacity limit
    of approximately 7 ± 2 chunks in human working memory.
    """

    @abstractmethod
    async def inject(self, block: ContextBlock) -> None:
        """Add a ContextBlock to the active workspace.

        Raises TokenBudgetExceededError if the block would push total
        usage over the configured budget.
        """

    @abstractmethod
    async def inject_retrieved(self, items: list[RetrievedItem]) -> None:
        """Inject a batch of retrieved memory items into working memory.

        Items are ranked and admitted up to the remaining token budget.
        """

    @abstractmethod
    def get_state(self) -> WorkingMemoryState:
        """Snapshot the current contents and token usage of working memory."""

    @abstractmethod
    async def generate_cues(self) -> list[str]:
        """Derive retrieval cues from the current working memory state.

        Returns a list of query strings suitable for driving memory
        retrieval across all long-term memory subsystems.
        """

    @abstractmethod
    def push_goal(self, goal: Goal) -> None:
        """Push a goal onto the active goal stack."""

    @abstractmethod
    def pop_goal(self) -> Goal | None:
        """Pop and return the top goal, or ``None`` if the stack is empty."""

    @abstractmethod
    async def flush(self) -> Episode:
        """Serialise the current workspace into an Episode and clear it.

        Called at the end of each cognitive cycle to hand off the
        session's context to the episodic encoding pipeline.
        """

    @abstractmethod
    def token_status(self) -> dict[str, int]:
        """Return ``{"used": int, "budget": int, "available": int}``."""


class EpisodicMemoryInterface(ABC):
    """Fast encoding and cue-based retrieval of experiences.

    Brain analog: Hippocampal formation — performs rapid, one-shot
    encoding of episodic events and supports pattern completion for
    context-dependent recall.  Also the source of memory traces
    replayed during consolidation into neocortical semantic memory.
    """

    @abstractmethod
    async def encode(self, episode: Episode) -> UUID:
        """Persist *episode* and return its assigned UUID."""

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Retrieve episodes matching *query* via multi-modal similarity."""

    @abstractmethod
    async def get(self, episode_id: UUID) -> Episode | None:
        """Fetch a specific episode by its UUID."""

    @abstractmethod
    async def update(self, episode_id: UUID, **updates: Any) -> None:
        """Apply partial updates to an existing episode record."""

    @abstractmethod
    async def sample_for_consolidation(self, batch_size: int = 32) -> list[Episode]:
        """Sample episodes due for offline consolidation into semantic memory.

        Implementations should prioritise recent, high-salience, or
        not-yet-consolidated episodes.
        """

    @abstractmethod
    async def mark_consolidated(self, episode_ids: list[UUID]) -> None:
        """Flag *episode_ids* as successfully consolidated.

        Prevents duplicate extraction during subsequent consolidation runs.
        """

    @abstractmethod
    async def run_decay_sweep(self) -> int:
        """Archive or delete stale episodes according to the decay policy.

        Returns
        -------
        int
            Number of episodes archived or deleted during the sweep.
        """


class SemanticMemoryInterface(ABC):
    """Structured knowledge storage with graph-based retrieval.

    Brain analog: Neocortical association areas — encode distilled,
    context-independent facts about the world as a network of concepts
    and relations.  Populated primarily by hippocampal consolidation
    rather than direct encoding.
    """

    @abstractmethod
    async def upsert_triples(self, triples: list[SemanticTriple]) -> int:
        """Insert or update semantic triples (subject–predicate–object).

        Returns
        -------
        int
            Number of triples actually written (deduplicated).
        """

    @abstractmethod
    async def resolve_entity(
        self,
        name: str,
        embedding: list[float] | None = None,
    ) -> Entity | None:
        """Look up a canonical Entity by name or nearest embedding.

        Performs coreference resolution to avoid duplicate nodes.
        Returns ``None`` if no sufficiently similar entity is found.
        """

    @abstractmethod
    async def retrieve_by_entity(
        self,
        entity_ref: EntityRef,
        hops: int = 1,
    ) -> list[SemanticTriple]:
        """Return all triples within *hops* of *entity_ref* in the graph."""

    @abstractmethod
    async def retrieve_by_similarity(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SemanticTriple]:
        """Find triples whose subject or object embedding is closest to *embedding*."""

    @abstractmethod
    async def spreading_activation(
        self,
        seed_entities: list[EntityRef],
        max_hops: int = 2,
    ) -> list[RankedNode]:
        """Propagate activation from *seed_entities* through the knowledge graph.

        Uses Personalised PageRank to rank associated concepts by
        relevance to the seed set.

        Returns
        -------
        list[RankedNode]
            Ranked nodes in descending activation order.
        """

    @abstractmethod
    async def get_community(self, community_id: UUID) -> Community | None:
        """Retrieve a detected concept community by its UUID."""

    @abstractmethod
    async def run_maintenance(self) -> None:
        """Execute background maintenance: community detection, deduplication, pruning."""


class ProceduralMemoryInterface(ABC):
    """Skill storage with RL-based utility scoring.

    Brain analog: Basal ganglia (striatum) — encodes stimulus–action
    mappings and selects actions based on expected reward.  Skills
    with high utility scores are preferentially retrieved, mirroring
    the striatum's role in action selection.
    """

    @abstractmethod
    async def register(self, skill: Skill) -> UUID:
        """Persist a new skill definition and return its UUID."""

    @abstractmethod
    async def retrieve(
        self,
        situation_embedding: list[float],
        top_k: int = 5,
    ) -> list[Skill]:
        """Return the *top_k* skills most applicable to the current situation.

        Applicability is determined by embedding similarity combined with
        utility score weighting.
        """

    @abstractmethod
    async def get(self, skill_id: UUID) -> Skill | None:
        """Fetch a skill by its UUID."""

    @abstractmethod
    async def update_utility(
        self,
        skill_id: UUID,
        reward: float,
        success: bool,
    ) -> None:
        """Update a skill's utility score using the provided reward signal.

        Implementations should apply a temporal-difference update rule
        (e.g. exponential moving average) to blend the new signal into
        the existing score.
        """

    @abstractmethod
    async def deprecate(self, skill_id: UUID) -> None:
        """Mark a skill as deprecated so it is excluded from retrieval."""


class ValenceMemoryInterface(ABC):
    """Emotional salience tagging and rapid appraisal.

    Brain analog: Amygdala — rapidly evaluates stimuli for emotional
    significance, modulates memory encoding strength based on arousal,
    and stores learned fear/reward associations that guide future appraisal.
    """

    @abstractmethod
    async def appraise(self, percept: PerceptUnit) -> SalienceScore:
        """Compute the emotional valence and arousal of *percept*.

        Returns
        -------
        SalienceScore
            Combined salience score incorporating valence, arousal,
            and learned associations.
        """

    @abstractmethod
    async def update(self, triggers: list[str], reward_signal: float) -> None:
        """Update valence associations for *triggers* given *reward_signal*.

        Called after outcome evaluation to strengthen or weaken the
        learned emotional associations for the triggering stimuli.
        """

    @abstractmethod
    async def get_associations(self, trigger: str) -> list[ValenceAssociation]:
        """Retrieve all learned valence associations for *trigger*."""

    @abstractmethod
    async def run_extinction_sweep(self) -> int:
        """Decay or remove associations that have not been reinforced.

        Models extinction learning: unreinforced conditioned responses
        gradually weaken over time.

        Returns
        -------
        int
            Number of associations weakened or removed.
        """


# ---------------------------------------------------------------------------
# Cognitive Control Interfaces
# ---------------------------------------------------------------------------


class OrchestratorInterface(ABC):
    """Central executive that runs the cognitive cycle.

    Brain analog: Prefrontal cortex executive network — integrates input
    from all cognitive subsystems, selects and sequences cognitive
    operations, and drives goal-directed behaviour across multiple cycles.
    """

    @abstractmethod
    async def run_cycle(self, raw_input: str | None = None) -> dict[str, Any]:
        """Execute one full perception–retrieval–reasoning–action cycle.

        Parameters
        ----------
        raw_input:
            Optional new stimulus to inject at the start of the cycle.

        Returns
        -------
        dict[str, Any]
            Cycle summary including actions taken, memories updated,
            and current goal status.
        """

    @abstractmethod
    async def run_until_complete(
        self,
        goal: Goal,
        max_cycles: int = 10,
    ) -> dict[str, Any]:
        """Run consecutive cognitive cycles until *goal* is completed.

        Parameters
        ----------
        goal:
            The terminal goal that determines when to stop.
        max_cycles:
            Hard limit on cycles to prevent infinite loops.

        Returns
        -------
        dict[str, Any]
            Final summary including success/failure status and cycle count.
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the orchestrator's current internal state."""


class AttentionControllerInterface(ABC):
    """Selective attention and Global Workspace broadcasting.

    Brain analog: Basal forebrain cholinergic system — modulates cortical
    excitability and signal-to-noise ratio, determining which stimuli are
    amplified to global workspace visibility and which are suppressed.
    """

    @abstractmethod
    async def score(
        self,
        percept: PerceptUnit,
        active_goals: list[Goal],
    ) -> SalienceScore:
        """Compute the attentional salience of *percept* given *active_goals*.

        Parameters
        ----------
        percept:
            The incoming percept to evaluate.
        active_goals:
            Goals currently active in working memory that bias attention.

        Returns
        -------
        SalienceScore
            Combined score reflecting bottom-up salience and top-down relevance.
        """

    @abstractmethod
    def gate(self, salience: SalienceScore) -> GateDecision:
        """Decide whether *salience* clears the current attention threshold.

        Returns a GateDecision indicating admit, suppress, or defer.
        """

    @abstractmethod
    def adjust_thresholds(self, cognitive_load: float) -> None:
        """Adapt attention thresholds based on current cognitive load (0.0–1.0).

        Under high load the threshold rises, increasing selectivity.
        Under low load the threshold falls, allowing broader scanning.
        """


class MetaCognitionInterface(ABC):
    """Self-monitoring, error detection, and strategy adjustment.

    Brain analog: Anterior cingulate cortex (ACC) — monitors for
    conflicts and prediction errors in ongoing cognitive processes,
    signalling the need for increased cognitive control and triggering
    strategy switches when current approaches are failing.
    """

    @abstractmethod
    async def evaluate_cycle(
        self,
        episode: Episode,
        rpe: float,
    ) -> MetaEvaluation:
        """Evaluate the quality of the just-completed cognitive cycle.

        Parameters
        ----------
        episode:
            The episode produced by flushing working memory at cycle end.
        rpe:
            Reward prediction error from the RewardProcessor for this cycle.

        Returns
        -------
        MetaEvaluation
            Assessment including detected errors, confidence, and flags.
        """

    @abstractmethod
    def recommend_strategy(self, state: dict[str, Any]) -> Strategy | None:
        """Suggest a strategy change based on recent performance signals.

        Returns ``None`` if no strategy switch is warranted.
        """

    @abstractmethod
    async def record_lesson(self, lesson: str, context: str) -> None:
        """Persist a meta-cognitive lesson learned from a failure or surprise.

        Lessons are stored for future retrieval when similar contexts arise.
        """


class GoalManagerInterface(ABC):
    """Hierarchical goal management and task decomposition.

    Brain analog: Anterior prefrontal cortex — maintains representations
    of branching, nested sub-goals and manages the temporal integration
    required to pursue long-horizon objectives across multiple cycles.
    """

    @abstractmethod
    async def create_goal(
        self,
        description: str,
        priority: float = 0.5,
    ) -> Goal:
        """Instantiate and persist a new Goal.

        Parameters
        ----------
        description:
            Natural-language description of the goal.
        priority:
            Initial priority weight in [0, 1].

        Returns
        -------
        Goal
            The newly created goal with an assigned UUID.
        """

    @abstractmethod
    async def decompose(self, goal: Goal) -> list[Goal]:
        """Break *goal* into an ordered list of sub-goals via LLM planning.

        The returned sub-goals are persisted and linked to *goal* as children.
        """

    @abstractmethod
    async def update_status(self, goal_id: UUID, status: GoalStatus) -> None:
        """Transition the goal identified by *goal_id* to *status*."""

    @abstractmethod
    def get_active_goals(self) -> list[Goal]:
        """Return all goals currently in ACTIVE status, sorted by priority."""


class ConsolidationEngineInterface(ABC):
    """Sleep-like memory consolidation from episodic to semantic memory.

    Brain analog: Hippocampal replay during slow-wave sleep — episodic
    traces are reactivated and transferred to stable neocortical
    representations, building a compressed, generalised world model.
    """

    @abstractmethod
    async def run_cycle(self) -> ConsolidationResult:
        """Execute one consolidation pass over the pending episode queue.

        Returns
        -------
        ConsolidationResult
            Summary of triples extracted, nodes created, and episodes processed.
        """

    @abstractmethod
    def queue_status(self) -> dict[str, Any]:
        """Return the current state of the consolidation queue.

        Typical keys: ``{"pending": int, "processing": int, "failed": int}``.
        """

    @abstractmethod
    def schedule(self, trigger: str, **kwargs: Any) -> None:
        """Register a future consolidation trigger.

        Parameters
        ----------
        trigger:
            Named event that will initiate consolidation
            (e.g. ``"idle"``, ``"episode_count_threshold"``).
        **kwargs:
            Trigger-specific configuration (threshold values, cron expressions).
        """


class RewardProcessorInterface(ABC):
    """Reward prediction error computation.

    Brain analog: VTA/Substantia Nigra dopaminergic system — computes
    reward prediction errors (RPE) that drive reinforcement learning
    updates across procedural memory and valence associations.  A
    positive RPE signals better-than-expected outcomes; a negative RPE
    signals worse-than-expected outcomes.
    """

    @abstractmethod
    async def compute_rpe(
        self,
        episode_id: UUID,
        predicted_value: float,
        actual_reward: float,
        next_value: float = 0.0,
    ) -> RewardSignal:
        """Compute the temporal-difference reward prediction error.

        Uses the TD(0) update rule:
            RPE = actual_reward + γ * next_value - predicted_value

        Parameters
        ----------
        predicted_value:
            The value estimate made before the outcome was observed.
        actual_reward:
            The reward actually received at this timestep.
        next_value:
            Bootstrap value of the successor state (0 for terminal states).

        Returns
        -------
        RewardSignal
            Full signal object carrying the raw RPE and metadata.
        """
