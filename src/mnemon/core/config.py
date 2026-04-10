"""
Hierarchical configuration system for Mnemon.

Each cognitive subsystem has its own config section, enabling independent
tuning and making defaults explicit. The top-level MnemonConfig composes
all sections and can be hydrated from a TOML file or environment variables.

Environment variable prefix: MNEMON__  (double underscore for nesting)
Example: MNEMON__EPISODIC__CAPACITY__MAX_EPISODES=50000
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigError
from .models import EvictionPolicy

# ---------------------------------------------------------------------------
# Sub-section configs
# ---------------------------------------------------------------------------


class CycleConfig(BaseSettings):
    """
    Controls the main cognitive cycle loop.

    The cycle is the heartbeat of the agent — each tick processes percepts,
    retrieves memories, selects actions, and schedules consolidation.
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__CYCLE__", env_nested_delimiter="__")

    max_cycles_per_request: int = Field(
        default=10,
        ge=1,
        description="Hard cap on cognitive cycles per user request.",
    )
    cycle_timeout_ms: int = Field(
        default=5_000,
        ge=100,
        description="Maximum wall-clock time for a single cycle in ms.",
    )
    parallel_retrieval: bool = Field(
        default=True,
        description="Fan out retrieval queries across stores concurrently.",
    )


class SensoryConfig(BaseSettings):
    """
    Governs the sensory buffer (thalamic relay analog).

    The buffer holds raw percepts before attention gating. Items that are
    not attended expire after ttl_ms milliseconds.
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__SENSORY__", env_nested_delimiter="__")

    capacity: int = Field(default=64, ge=1, description="Maximum number of live percepts.")
    ttl_ms: int = Field(default=30_000, ge=100, description="Default percept TTL in milliseconds.")


class SummarizationConfig(BaseSettings):
    """Working memory summarization sub-config."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__WORKING_MEMORY__SUMMARIZATION__", env_nested_delimiter="__"
    )

    enabled: bool = True
    max_summary_tokens: int = Field(
        default=256,
        ge=32,
        description="Token budget for each generated context summary.",
    )


class WorkingMemoryConfig(BaseSettings):
    """
    Configuration for working memory (prefrontal cortex analog).

    Token budget is the primary capacity constraint. When exceeded, the
    eviction policy selects which ContextBlocks to remove or summarize.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__WORKING_MEMORY__", env_nested_delimiter="__"
    )

    token_budget: int = Field(
        default=8_192,
        ge=256,
        description="Total token capacity of working memory.",
    )
    eviction_policy: EvictionPolicy = Field(
        default=EvictionPolicy.LRU_IMPORTANCE,
        description="Policy used to select context blocks for eviction.",
    )
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)


class EpisodicRetrievalWeights(BaseSettings):
    """Relative weights for hybrid episodic retrieval scoring."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__EPISODIC__RETRIEVAL__", env_nested_delimiter="__"
    )

    semantic: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25: float = Field(default=0.2, ge=0.0, le=1.0)
    recency: float = Field(default=0.2, ge=0.0, le=1.0)
    importance: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> EpisodicRetrievalWeights:
        total = self.semantic + self.bm25 + self.recency + self.importance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Episodic retrieval weights must sum to 1.0, got {total:.4f}"
            )
        return self


class EpisodicDecayConfig(BaseSettings):
    """Ebbinghaus-style exponential decay parameters for episodic memory."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__EPISODIC__DECAY__", env_nested_delimiter="__"
    )

    base_lambda: float = Field(
        default=0.001,
        ge=0.0,
        description="Default forgetting rate (higher = faster decay).",
    )
    forget_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Strength below which an episode is eligible for archival.",
    )
    sweep_interval_s: int = Field(
        default=3_600,
        ge=60,
        description="How often the decay sweep runs, in seconds.",
    )


class EpisodicCapacityConfig(BaseSettings):
    """Hard capacity limits for the episodic store."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__EPISODIC__CAPACITY__", env_nested_delimiter="__"
    )

    max_episodes: int = Field(
        default=100_000,
        ge=100,
        description="Maximum number of episodes retained before forced archival.",
    )


class EpisodicConfig(BaseSettings):
    """
    Configuration for episodic memory (hippocampus analog).

    Episodic memory stores autobiographical agent experiences indexed by
    time and context. Retrieval blends vector similarity, keyword overlap,
    recency bias, and importance weighting.
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__EPISODIC__", env_nested_delimiter="__")

    backend: str = Field(
        default="hnswlib",
        description="Vector store backend identifier (e.g. 'qdrant', 'hnswlib', 'memory').",
    )
    retrieval_weights: EpisodicRetrievalWeights = Field(
        default_factory=EpisodicRetrievalWeights
    )
    decay: EpisodicDecayConfig = Field(default_factory=EpisodicDecayConfig)
    capacity: EpisodicCapacityConfig = Field(default_factory=EpisodicCapacityConfig)


class RaptorConfig(BaseSettings):
    """RAPTOR hierarchical summarization config for semantic memory."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__SEMANTIC__RAPTOR__", env_nested_delimiter="__"
    )

    enabled: bool = True
    max_levels: int = Field(
        default=3,
        ge=1,
        description="Maximum depth of the RAPTOR cluster tree.",
    )


class CommunityDetectionConfig(BaseSettings):
    """Graph community detection parameters."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__SEMANTIC__COMMUNITY__", env_nested_delimiter="__"
    )

    algorithm: str = Field(
        default="leiden",
        description="Community detection algorithm: 'leiden' or 'louvain'.",
    )
    resolution: float = Field(
        default=1.0,
        gt=0.0,
        description="Resolution parameter; higher = smaller communities.",
    )


class SemanticDecayConfig(BaseSettings):
    """Confidence decay for semantic triples without recent confirmation."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__SEMANTIC__DECAY__", env_nested_delimiter="__"
    )

    epsilon: float = Field(
        default=0.01,
        ge=0.0,
        description="Confidence decay per unconfirmed interval.",
    )
    confirmation_window: int = Field(
        default=7,
        ge=1,
        description="Days before an unconfirmed triple starts decaying.",
    )


class SemanticConfig(BaseSettings):
    """
    Configuration for semantic memory (neocortex analog).

    Semantic memory stores generalised world knowledge as a directed property
    graph. RAPTOR clustering provides hierarchical summarization for efficient
    coarse-to-fine retrieval.
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__SEMANTIC__", env_nested_delimiter="__")

    graph_backend: str = Field(
        default="igraph",
        description=(
            "Graph database backend identifier "
            "(e.g. 'falkordb', 'neo4j', 'igraph', 'memory')."
        ),
    )
    raptor: RaptorConfig = Field(default_factory=RaptorConfig)
    community_detection: CommunityDetectionConfig = Field(
        default_factory=CommunityDetectionConfig
    )
    decay: SemanticDecayConfig = Field(default_factory=SemanticDecayConfig)


class SkillAcquisitionConfig(BaseSettings):
    """Parameters governing automated skill learning and validation."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__PROCEDURAL__ACQUISITION__", env_nested_delimiter="__"
    )

    enabled: bool = True
    max_refinement_attempts: int = Field(
        default=3,
        ge=1,
        description="How many LLM-driven refinement rounds before giving up.",
    )
    initial_utility: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Starting utility score for a newly acquired skill.",
    )
    validation_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum success rate to graduate a skill from TESTED to VALIDATED.",
    )


class SkillUtilityConfig(BaseSettings):
    """Reinforcement learning parameters for skill utility updates."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__PROCEDURAL__UTILITY__", env_nested_delimiter="__"
    )

    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    decay_epsilon: float = Field(
        default=0.001,
        ge=0.0,
        description="Utility decay applied per time step to unused skills.",
    )
    deprecation_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Utility below which a skill is automatically deprecated.",
    )


class ProceduralConfig(BaseSettings):
    """
    Configuration for procedural memory (basal ganglia / cerebellum analog).

    Procedural memory stores reusable action sequences. Utility scores are
    updated via TD-learning after each execution outcome is observed.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__PROCEDURAL__", env_nested_delimiter="__"
    )

    skill_acquisition: SkillAcquisitionConfig = Field(default_factory=SkillAcquisitionConfig)
    utility: SkillUtilityConfig = Field(default_factory=SkillUtilityConfig)


class ValenceConfig(BaseSettings):
    """
    Configuration for valence / affective memory (amygdala analog).

    Controls how quickly emotional associations are learned (learning_rate)
    and how fast they fade when the trigger stops producing the expected
    outcome (extinction_rate).
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__VALENCE__", env_nested_delimiter="__")

    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    extinction_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Rate at which unreinforced associations decay.",
    )


class ConsolidationScheduleConfig(BaseSettings):
    """When and how often consolidation runs are triggered."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__CONSOLIDATION__SCHEDULE__", env_nested_delimiter="__"
    )

    mode: str = Field(
        default="idle",
        description="Trigger mode: 'idle', 'threshold', 'periodic', or 'manual'.",
    )
    idle_timeout_s: int = Field(
        default=60,
        ge=5,
        description="Seconds of inactivity before an idle-triggered consolidation run.",
    )
    episode_threshold: int = Field(
        default=100,
        ge=1,
        description="Raw episode count that triggers threshold-based consolidation.",
    )
    periodic_interval_s: int = Field(
        default=3_600,
        ge=60,
        description="Wall-clock interval for periodic consolidation, in seconds.",
    )


class ReplayConfig(BaseSettings):
    """Prioritised experience replay parameters for consolidation."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__CONSOLIDATION__REPLAY__", env_nested_delimiter="__"
    )

    alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Priority exponent: 0 = uniform, 1 = fully prioritised.",
    )
    beta_start: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="IS correction weight at start of training (anneals to 1.0).",
    )


class ConsolidationConfig(BaseSettings):
    """
    Configuration for the offline consolidation pipeline.

    Consolidation (sleep-replay analog) converts raw episodic traces into
    semantic triples, resolves entity coreferences, and prunes weak memories.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__CONSOLIDATION__", env_nested_delimiter="__"
    )

    enabled: bool = True
    schedule: ConsolidationScheduleConfig = Field(default_factory=ConsolidationScheduleConfig)
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Number of episodes processed per consolidation batch.",
    )
    max_extraction_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum failed LLM extraction attempts before an episode is marked failed.",
    )
    replay: ReplayConfig = Field(default_factory=ReplayConfig)


class RewardSourceWeights(BaseSettings):
    """Per-source weights for composing the final reward signal."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__REWARD__WEIGHTS__", env_nested_delimiter="__"
    )

    task_success: float = Field(default=0.6, ge=0.0)
    efficiency: float = Field(default=0.2, ge=0.0)
    user_feedback: float = Field(default=0.15, ge=0.0)
    goal_progress: float = Field(default=0.05, ge=0.0)


class RewardConfig(BaseSettings):
    """
    Configuration for the reward / value learning subsystem.

    Reward signals drive TD-learning updates across episodic importance,
    skill utility, and valence associations — the mesolimbic pathway analog.
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__REWARD__", env_nested_delimiter="__")

    discount_factor: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Temporal discount γ for future reward.",
    )
    source_weights: RewardSourceWeights = Field(default_factory=RewardSourceWeights)


class AttentionWeightsConfig(BaseSettings):
    """Relative weights for the attention gate's salience computation."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__ATTENTION__WEIGHTS__", env_nested_delimiter="__"
    )

    valence: float = Field(default=0.3, ge=0.0, le=1.0)
    goal_relevance: float = Field(default=0.4, ge=0.0, le=1.0)
    novelty: float = Field(default=0.2, ge=0.0, le=1.0)
    urgency: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> AttentionWeightsConfig:
        total = self.valence + self.goal_relevance + self.novelty + self.urgency
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Attention weights must sum to 1.0, got {total:.4f}"
            )
        return self


class AttentionConfig(BaseSettings):
    """
    Configuration for the attention gate (thalamus / reticular nucleus analog).

    The gate scores each incoming percept and routes it to BROADCAST (all
    modules receive it), QUEUE (deferred processing), or DISCARD.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__ATTENTION__", env_nested_delimiter="__"
    )

    broadcast_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum combined salience for a percept to be broadcast.",
    )
    attention_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum salience to queue rather than discard.",
    )
    weights: AttentionWeightsConfig = Field(default_factory=AttentionWeightsConfig)
    adaptive_thresholds: bool = Field(
        default=True,
        description="Dynamically adjust thresholds based on cognitive load.",
    )


class ReflexionConfig(BaseSettings):
    """Reflexion-style self-critique sub-config."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__META__REFLEXION__", env_nested_delimiter="__"
    )

    enabled: bool = True
    trigger: str = Field(
        default="prediction_error > 0.5",
        description="Expression evaluated against MetaEvaluation fields.",
    )


class MetaCognitionConfig(BaseSettings):
    """
    Configuration for the meta-cognition module (ACC / prefrontal analog).

    Meta-cognition monitors prediction errors, tracks confidence, and
    recommends strategy switches when the current approach is failing.
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__META__", env_nested_delimiter="__")

    enabled: bool = True
    confidence_tracking: bool = True
    reflexion: ReflexionConfig = Field(default_factory=ReflexionConfig)
    max_strategy_switches: int = Field(
        default=3,
        ge=1,
        description="Maximum strategy switches allowed per cognitive cycle.",
    )


class QdrantConfig(BaseSettings):
    """Qdrant vector store connection and performance settings."""

    model_config = SettingsConfigDict(env_prefix="MNEMON__QDRANT__", env_nested_delimiter="__")

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)
    prefer_grpc: bool = Field(default=True)
    collection_name: str = Field(default="mnemon_vectors")
    dimension: int | None = Field(default=None)
    binary_quantization: bool = Field(
        default=False,
        description="Enable binary quantization for 40x speedup on large collections.",
    )
    on_disk: bool = Field(
        default=False,
        description="Store vectors on disk for memory-constrained environments.",
    )


class LLMConfig(BaseSettings):
    """
    Configuration for LLM provider routing via LiteLLM.

    All LLM calls go through LiteLLM, which handles provider selection,
    retries, and cost tracking. Provider-specific settings (api_key, model,
    base_url) live in the providers dict.
    """

    model_config = SettingsConfigDict(env_prefix="MNEMON__LLM__", env_nested_delimiter="__")

    default_provider: str = Field(
        default="openai",
        description="LiteLLM provider identifier used when no override is specified.",
    )
    providers: dict[str, Any] = Field(
        default_factory=lambda: {
            "openai": {"model": "gpt-4o-mini"},
            "anthropic": {"model": "claude-3-5-haiku-20241022"},
        },
        description="Provider-specific config dicts passed through to LiteLLM.",
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class MnemonConfig(BaseSettings):
    """
    Root configuration for the Mnemon cognitive memory framework.

    Composes all subsystem configs into a single loadable object. Can be
    hydrated from a TOML file, environment variables, or programmatically.
    Each subsystem section can be overridden independently.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMON__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cycle: CycleConfig = Field(default_factory=CycleConfig)
    sensory: SensoryConfig = Field(default_factory=SensoryConfig)
    working_memory: WorkingMemoryConfig = Field(default_factory=WorkingMemoryConfig)
    episodic: EpisodicConfig = Field(default_factory=EpisodicConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    procedural: ProceduralConfig = Field(default_factory=ProceduralConfig)
    valence: ValenceConfig = Field(default_factory=ValenceConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    meta_cognition: MetaCognitionConfig = Field(default_factory=MetaCognitionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | None = None) -> MnemonConfig:
    """
    Load a MnemonConfig from a TOML file path or fall back to env vars / defaults.

    If path is provided, the file must exist and be valid TOML. The TOML
    structure mirrors the MnemonConfig field hierarchy. After loading the
    TOML dict, the result is merged with any environment variable overrides
    (env vars take precedence over file values).

    Args:
        path: Absolute or relative path to a ``mnemon.toml`` config file.
              If None, only environment variables and built-in defaults are used.

    Returns:
        A fully validated MnemonConfig instance.

    Raises:
        ConfigError: If the file does not exist, is not valid TOML, or the
                     resulting config fails Pydantic validation.
    """
    if path is None:
        try:
            return MnemonConfig()
        except Exception as exc:
            raise ConfigError(f"Failed to build default config: {exc}") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")

    try:
        with config_path.open("rb") as fh:
            raw: dict[str, Any] = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in config file '{config_path}': {exc}") from exc

    try:
        return MnemonConfig(**raw)  # BaseSettings.__init__ layers env vars on top
    except Exception as exc:
        raise ConfigError(f"Config validation failed for '{config_path}': {exc}") from exc
