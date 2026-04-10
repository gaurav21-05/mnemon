"""
Core data models for the Mnemon cognitive memory framework.

Each model maps to a specific structure in the brain-inspired architecture:
- Sensory buffer   → PerceptUnit
- Working memory   → WorkingMemoryState, Goal, ContextBlock
- Episodic memory  → Episode  (hippocampus)
- Semantic memory  → SemanticTriple, Entity, Community  (neocortex)
- Procedural memory → Skill  (basal ganglia / cerebellum)
- Valence memory   → ValenceAssociation  (amygdala)
- Cognitive bus    → CognitiveMessage
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Modality(StrEnum):
    """Sensory channel through which a percept arrived."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    STRUCTURED_DATA = "structured_data"
    TOOL_OUTPUT = "tool_output"


class MessageType(StrEnum):
    """
    Type tag for messages on the cognitive bus.

    Determines routing and priority rules applied by the attention gate.
    """

    PERCEPT = "percept"
    RETRIEVAL_CUE = "retrieval_cue"
    RETRIEVAL_RESULT = "retrieval_result"
    ACTION_CANDIDATE = "action_candidate"
    ACTION_SELECTED = "action_selected"
    REWARD_SIGNAL = "reward_signal"
    CONSOLIDATION_TRIGGER = "consolidation_trigger"
    META_SIGNAL = "meta_signal"
    GOAL_UPDATE = "goal_update"
    ATTENTION_GATE = "attention_gate"
    BROADCAST = "broadcast"


class GoalStatus(StrEnum):
    """
    Lifecycle state of a goal on the goal stack.

    Mirrors motivational states tracked by the prefrontal cortex and ACC.
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    DROPPED = "dropped"


class ConsolidationState(StrEnum):
    """
    Processing stage of an episodic memory during offline consolidation.

    Mirrors the sleep-dependent hippocampo-neocortical transfer pipeline.
    """

    RAW = "raw"
    PROCESSING = "processing"
    FAILED = "failed"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"


class MemoryLifecycleState(StrEnum):
    """High-level lifecycle phase of an episodic memory."""

    INGESTED = "ingested"
    DURABLE = "durable"
    CONSOLIDATED = "consolidated"
    SUMMARY = "summary"
    HISTORICAL = "historical"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"
    EXCLUDED = "excluded"


class SkillType(StrEnum):
    """Representation format of a stored procedural skill."""

    CODE = "code"
    PROMPT_TEMPLATE = "prompt_template"
    WORKFLOW_DAG = "workflow_dag"
    TOOL_SEQUENCE = "tool_sequence"


class SkillStatus(StrEnum):
    """Validation lifecycle stage of a procedural skill."""

    DRAFT = "draft"
    TESTED = "tested"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"


class EvictionPolicy(StrEnum):
    """Policy governing which context blocks are evicted from working memory."""

    LRU = "lru"
    LRU_IMPORTANCE = "lru_importance"
    CUSTOM = "custom"


class ContextSource(StrEnum):
    """Origin of a context block admitted into working memory."""

    USER_INPUT = "user_input"
    AGENT_OUTPUT = "agent_output"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    SUMMARY = "summary"


class GateDecision(StrEnum):
    """Decision made by the attention gate for an incoming cognitive message."""

    BROADCAST = "broadcast"
    QUEUE = "queue"
    DISCARD = "discard"


class ConsolidationTrigger(StrEnum):
    """Event that initiates an offline consolidation run."""

    IDLE = "idle"
    THRESHOLD = "threshold"
    PERIODIC = "periodic"
    MANUAL = "manual"


class ConditionType(StrEnum):
    """Logical form of a precondition or postcondition on a skill."""

    STATE_CHECK = "state_check"
    ENTITY_PRESENT = "entity_present"
    GOAL_ACTIVE = "goal_active"
    CAPABILITY_AVAILABLE = "capability_available"


# ---------------------------------------------------------------------------
# Cognitive Bus
# ---------------------------------------------------------------------------


class CognitiveMessage(BaseModel):
    """
    A message on the global cognitive bus (Global Workspace Theory).

    Enables loosely coupled communication between cognitive modules.
    The bus is the substrate for attention broadcasting — messages with
    sufficient salience are broadcast to all listening modules.
    """

    id: UUID = Field(default_factory=uuid4)
    source: str = Field(description="Module ID of the sender.")
    target: str = Field(description="Module ID of the recipient, or '*' for broadcast.")
    type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID linking all messages in a single cognitive cycle.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Sensory models
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """
    A named entity extracted from perception or long-term knowledge.

    Entities are the nodes of the semantic knowledge graph, analogous to
    concept representations in the temporal lobe.
    """

    id: UUID = Field(default_factory=uuid4)
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    type: str = Field(description="Entity type label, e.g. 'person', 'org', 'concept'.")
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    description: str = ""
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    community_id: UUID | None = None


class PerceptUnit(BaseModel):
    """
    A single processed sensory input — the output of the sensory buffer.

    Analogous to a percept in the sensory cortex: raw signal → normalized
    representation → optionally embedded and entity-tagged for routing.
    TTL governs how long the percept stays available before it decays.
    """

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modality: Modality
    raw_content: str
    normalized: str
    embedding: list[float] | None = None
    entities: list[Entity] = Field(default_factory=list)
    intent: str | None = None
    sentiment: float = Field(ge=-1.0, le=1.0, default=0.0)
    tokens: int = Field(ge=0)
    ttl_ms: int = Field(default=30_000, ge=0, description="Time-to-live in milliseconds.")
    attended: bool = False


# ---------------------------------------------------------------------------
# Working Memory models
# ---------------------------------------------------------------------------


class Goal(BaseModel):
    """
    An active motivational target held in working memory.

    The goal stack mirrors the prefrontal cortex's role in maintaining
    hierarchical intentions. Goals can nest (parent/subgoals) and track
    their own progress and retry budget.
    """

    id: UUID = Field(default_factory=uuid4)
    description: str
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    status: GoalStatus = GoalStatus.ACTIVE
    parent_goal_id: UUID | None = None
    subgoals: list[UUID] = Field(default_factory=list)
    dependencies: list[UUID] = Field(default_factory=list)
    deadline: datetime | None = None
    success_criteria: str = ""
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    attempts: int = Field(ge=0, default=0)
    max_attempts: int = Field(ge=1, default=10)


class ContextBlock(BaseModel):
    """
    A chunk of textual context occupying working memory capacity.

    Mirrors the idea of items held in the phonological loop or visuospatial
    sketchpad of Baddeley's working memory model. Each block carries an
    importance score used by the eviction policy.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str
    token_count: int = Field(ge=0)
    source: ContextSource
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    evictable: bool = True
    summary: str | None = None


class RetrievedItem(BaseModel):
    """
    A single item surfaced by a retrieval query from any long-term store.

    The score is normalised to [0, 1] and reflects combined vector,
    keyword, recency, and importance signals.
    """

    model_config = {"frozen": True}

    source_store: str = Field(description="Identifier of the originating memory store.")
    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkingMemoryState(BaseModel):
    """
    A snapshot of the agent's working memory at a point in time.

    Working memory (prefrontal cortex analog) holds the current goal stack,
    active context, scratch-pad reasoning space, and recently retrieved long-
    term memories — everything needed to act coherently in the present moment.
    """

    session_id: UUID = Field(default_factory=uuid4)
    token_budget: int = Field(ge=1, description="Maximum tokens the context window can hold.")
    token_used: int = Field(ge=0, default=0)
    goal_stack: list[Goal] = Field(default_factory=list)
    active_context: list[ContextBlock] = Field(default_factory=list)
    scratch_pad: str = ""
    retrieved_items: list[RetrievedItem] = Field(default_factory=list)
    system_prompt: str = ""
    pinned_items: list[ContextBlock] = Field(default_factory=list)

    @property
    def is_over_budget(self) -> bool:
        """Check if token usage exceeds budget (for eviction decisions)."""
        return self.token_used > self.token_budget

    @property
    def available_tokens(self) -> int:
        """Tokens remaining before budget is exceeded."""
        return max(0, self.token_budget - self.token_used)


# ---------------------------------------------------------------------------
# Episodic Memory models
# ---------------------------------------------------------------------------


class Episode(BaseModel):
    """
    A single autobiographical memory — one complete agent experience.

    Analogous to a hippocampal episodic trace: context → action → outcome
    with affective colouring. Strength decays exponentially over time unless
    reinforced by retrieval or consolidation.
    """

    id: UUID = Field(default_factory=uuid4)
    agent_id: str
    session_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: str = Field(description="Situational context at the time of the episode.")
    action: str = Field(description="Action taken by the agent.")
    outcome: str = Field(description="Result / consequence observed.")
    reflection: str | None = None
    embedding: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    goal_id: UUID | None = None
    scope_type: str = Field(default="personal", description="Memory scope category.")
    scope_id: str = Field(default="personal", description="Stable scope identifier.")
    workspace_path: str | None = Field(default=None, description="Associated workspace path.")
    repo_name: str | None = Field(default=None, description="Associated repository name.")
    caused_by: UUID | None = None
    led_to: list[UUID] = Field(default_factory=list)
    source_episode_ids: list[UUID] = Field(default_factory=list)
    summary_kind: str | None = None
    summary_of_count: int = Field(ge=0, default=0)
    lifecycle_state: MemoryLifecycleState = MemoryLifecycleState.DURABLE
    retrieval_uses: int = Field(ge=0, default=0)
    retrieval_help_count: int = Field(ge=0, default=0)
    retrieval_last_used_at: datetime | None = None
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    emotional_valence: float = Field(
        ge=-1.0,
        le=1.0,
        default=0.0,
        description="Affective tone: -1 = very negative, +1 = very positive.",
    )
    reward_signal: float = Field(default=0.0)
    access_count: int = Field(ge=0, default=0)
    consolidation_state: ConsolidationState = ConsolidationState.RAW
    consolidation_attempts: int = Field(
        ge=0,
        default=0,
        description="Number of failed consolidation extraction attempts.",
    )
    decay_lambda: float = Field(
        ge=0.0,
        default=0.001,
        description="Exponential decay rate; higher = faster forgetting.",
    )
    base_strength: float = Field(
        ge=0.0,
        default=1.0,
        description="Initial memory strength before any decay.",
    )


# ---------------------------------------------------------------------------
# Semantic Memory models
# ---------------------------------------------------------------------------


class EntityRef(BaseModel):
    """
    A lightweight reference to an entity node in the knowledge graph.

    Used inside triples to avoid embedding full Entity objects, keeping
    triple serialization compact while preserving lookup identity.
    """

    entity_id: UUID
    name: str

    model_config = {"frozen": True}


class SemanticTriple(BaseModel):
    """
    A subject–predicate–object fact in the semantic knowledge graph.

    Semantic memory (neocortex analog) encodes generalised world knowledge
    as a directed property graph. Each triple accumulates evidence from
    multiple episodes and carries a confidence score that degrades without
    confirmation.
    """

    id: UUID = Field(default_factory=uuid4)
    subject: EntityRef
    predicate: str
    object: EntityRef | str
    confidence: float = Field(ge=0.0, le=1.0)
    source_episodes: list[UUID] = Field(default_factory=list)
    first_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_confirmed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    current: bool = True
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    supersedes: list[UUID] = Field(default_factory=list)
    superseded_by: UUID | None = None
    contradiction_group: str | None = None
    access_count: int = Field(ge=0, default=0)
    embedding: list[float] | None = None


class SemanticCluster(BaseModel):
    """
    A hierarchical summary node in the RAPTOR-style semantic tree.

    Clusters group related triples and sub-clusters into progressively
    more abstract summaries — analogous to cortical column hierarchies.
    """

    id: UUID = Field(default_factory=uuid4)
    level: int = Field(ge=0, description="0 = leaf (raw triples), higher = more abstract.")
    summary: str
    embedding: list[float] | None = None
    children: list[UUID] = Field(default_factory=list)
    member_triples: list[UUID] = Field(default_factory=list)


class Community(BaseModel):
    """
    A detected community of densely connected entities in the knowledge graph.

    Community detection (Leiden / Louvain) mirrors how the brain organises
    concepts into semantic neighbourhoods. Each community gets a generated
    summary useful for coarse-grained retrieval.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    member_entities: list[UUID] = Field(default_factory=list)
    summary: str = ""
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Procedural Memory models
# ---------------------------------------------------------------------------


class ParameterSchema(BaseModel):
    """Schema descriptor for a single parameter of a procedural skill."""

    name: str
    type: str = Field(description="Python type annotation string, e.g. 'str', 'int', 'list[str]'.")
    required: bool = True
    description: str = ""
    default_value: Any | None = None

    model_config = {"frozen": True}


class Condition(BaseModel):
    """
    A logical precondition or postcondition attached to a skill.

    Conditions guard skill execution (preconditions) and verify outcomes
    (postconditions), mirroring cerebellar forward/inverse model checking.
    """

    type: ConditionType
    expression: str = Field(description="DSL expression evaluated at runtime.")

    model_config = {"frozen": True}


class Skill(BaseModel):
    """
    A reusable learned action sequence stored in procedural memory.

    Analogous to motor programs in the basal ganglia / cerebellum:
    once learned, skills execute with minimal deliberate attention.
    Utility scores are updated via reinforcement after each execution.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    embedding: list[float] | None = None
    type: SkillType
    definition: str = Field(description="Serialised skill body (code, template, DAG JSON, etc.).")
    parameters: list[ParameterSchema] = Field(default_factory=list)
    preconditions: list[Condition] = Field(default_factory=list)
    postconditions: list[Condition] = Field(default_factory=list)
    sub_skills: list[UUID] = Field(default_factory=list)
    parent_skills: list[UUID] = Field(default_factory=list)
    utility: float = Field(ge=0.0, le=1.0, default=0.5)
    success_count: int = Field(ge=0, default=0)
    failure_count: int = Field(ge=0, default=0)
    avg_latency_ms: int = Field(ge=0, default=0)
    last_used: datetime | None = None
    creation_source: str = Field(default="human_authored")
    version: int = Field(ge=1, default=1)
    previous_version_id: UUID | None = None
    status: SkillStatus = SkillStatus.DRAFT


# ---------------------------------------------------------------------------
# Valence Memory models
# ---------------------------------------------------------------------------


class ValenceAssociation(BaseModel):
    """
    An affective association between a stimulus pattern and an emotional response.

    Valence memory (amygdala analog) stores learned emotional responses to
    triggers. Associations strengthen with repeated exposure and weaken via
    extinction learning when outcomes change.
    """

    id: UUID = Field(default_factory=uuid4)
    trigger: str = Field(description="Canonical text form of the stimulus.")
    trigger_embedding: list[float] | None = None
    valence: float = Field(
        ge=-1.0,
        le=1.0,
        description="Affective valence: -1 = aversive, 0 = neutral, +1 = appetitive.",
    )
    arousal: float = Field(ge=0.0, le=1.0, description="Activation level associated with trigger.")
    source_episodes: list[UUID] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    exposure_count: int = Field(ge=0, default=0)
    last_encountered: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class SalienceScore(BaseModel):
    """
    Decomposed salience score for an incoming percept.

    The attention gate combines sub-scores (valence, goal relevance, novelty)
    into a single combined score that determines broadcast/queue/discard.
    Mirrors the role of the superior colliculus and thalamic reticular nucleus
    in selective attention.
    """

    percept_id: UUID
    raw_salience: float = Field(ge=0.0, le=1.0)
    goal_relevance: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    combined: float = Field(ge=0.0, le=1.0)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


class RetrievalQuery(BaseModel):
    """
    A query issued to one or more long-term memory stores.

    Supports hybrid retrieval: dense vector search, sparse BM25,
    recency bias, and importance weighting can all be applied in combination.
    """

    query_text: str
    query_embedding: list[float] | None = None
    top_k: int = Field(ge=1, default=10)
    filters: dict[str, Any] = Field(default_factory=dict)
    time_range: tuple[datetime, datetime] | None = None
    min_score: float = Field(ge=0.0, le=1.0, default=0.0)

    @field_validator("time_range")
    @classmethod
    def time_range_ordered(
        cls, v: tuple[datetime, datetime] | None
    ) -> tuple[datetime, datetime] | None:
        if v is not None and v[0] > v[1]:
            raise ValueError("time_range start must be before end")
        return v


class RetrievalResult(BaseModel):
    """
    The result of a retrieval query from a single named store.

    Aggregation across multiple stores is the responsibility of the
    retrieval coordinator, which re-ranks and deduplicates results.
    """

    items: list[RetrievedItem]
    query_time_ms: float = Field(ge=0.0)
    store_name: str

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Learning models
# ---------------------------------------------------------------------------


class RewardSignal(BaseModel):
    """
    A temporal-difference reward signal for updating value estimates.

    Mirrors dopaminergic RPE (reward prediction error) signals in the
    mesolimbic system. The rpe field is the signed prediction error used
    to update both episodic importance and valence associations.
    """

    episode_id: UUID
    predicted_value: float
    actual_reward: float
    rpe: float = Field(description="Reward prediction error: actual_reward - predicted_value.")
    sources: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-source reward decomposition "
            "(e.g. {'task_success': 0.8, 'efficiency': 0.2})."
        ),
    )

    model_config = {"frozen": True}


class ConsolidationResult(BaseModel):
    """
    Summary statistics from a completed consolidation run.

    Produced by the offline consolidation pipeline after processing a batch
    of raw episodes into structured semantic knowledge.
    """

    episodes_processed: int = Field(ge=0)
    triples_extracted: int = Field(ge=0)
    entities_resolved: int = Field(ge=0)
    conflicts_detected: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Meta-cognition models
# ---------------------------------------------------------------------------


class Strategy(BaseModel):
    """
    A named meta-cognitive strategy that can be selected at runtime.

    Strategies are stored heuristics — if the trigger condition matches the
    current cognitive state, the associated action is recommended to the
    executive module.
    """

    name: str
    trigger: str = Field(description="Condition string that activates this strategy.")
    action: str = Field(description="Recommended action or module invocation.")
    weight: float = Field(ge=0.0, default=1.0)


class MetaEvaluation(BaseModel):
    """
    The output of a meta-cognitive self-assessment for a cognitive cycle.

    Mirrors the role of the anterior cingulate cortex in error monitoring
    and strategy adjustment. A high prediction_error triggers reflexion.
    """

    cycle_id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    prediction_error: float
    strategy_recommended: str | None = None
    lessons: list[str] = Field(default_factory=list)
