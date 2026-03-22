"""
Core subpackage for Mnemon — data models, config, and exceptions.

This is the foundation layer. Everything else in the framework depends on
these abstractions; nothing here depends on any other Mnemon subpackage.

Brain analog: The fundamental architectural blueprint of the cognitive system —
the axonal highways, synaptic contracts, and relay nuclei that allow
specialised brain regions to communicate reliably.
"""

from mnemon.core.config import (
    AttentionConfig,
    AttentionWeightsConfig,
    ConsolidationConfig,
    ConsolidationScheduleConfig,
    CycleConfig,
    EpisodicConfig,
    EpisodicDecayConfig,
    EpisodicRetrievalWeights,
    LLMConfig,
    MetaCognitionConfig,
    MnemonConfig,
    ProceduralConfig,
    ReplayConfig,
    RewardConfig,
    SemanticConfig,
    SensoryConfig,
    SkillAcquisitionConfig,
    SkillUtilityConfig,
    SummarizationConfig,
    ValenceConfig,
    WorkingMemoryConfig,
    load_config,
)
from mnemon.core.exceptions import (
    BackendNotAvailableError,
    ConfigError,
    ConsolidationError,
    GoalError,
    MemoryError,
    MnemonError,
    RetrievalError,
    SkillExecutionError,
    TokenBudgetExceededError,
)
from mnemon.core.models import (
    CognitiveMessage,
    Community,
    Condition,
    ConditionType,
    ConsolidationResult,
    ConsolidationState,
    ConsolidationTrigger,
    ContextBlock,
    ContextSource,
    Entity,
    EntityRef,
    Episode,
    EvictionPolicy,
    GateDecision,
    Goal,
    GoalStatus,
    MessageType,
    MetaEvaluation,
    Modality,
    ParameterSchema,
    PerceptUnit,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    RewardSignal,
    SalienceScore,
    SemanticCluster,
    SemanticTriple,
    Skill,
    SkillStatus,
    SkillType,
    Strategy,
    ValenceAssociation,
    WorkingMemoryState,
)

__all__ = [
    # Models — enums
    "Modality",
    "MessageType",
    "GoalStatus",
    "ConsolidationState",
    "SkillType",
    "SkillStatus",
    "EvictionPolicy",
    "ContextSource",
    "GateDecision",
    "ConsolidationTrigger",
    "ConditionType",
    # Models — cognitive bus
    "CognitiveMessage",
    # Models — sensory
    "Entity",
    "PerceptUnit",
    # Models — working memory
    "Goal",
    "ContextBlock",
    "RetrievedItem",
    "WorkingMemoryState",
    # Models — episodic
    "Episode",
    # Models — semantic
    "EntityRef",
    "SemanticTriple",
    "SemanticCluster",
    "Community",
    # Models — procedural
    "ParameterSchema",
    "Condition",
    "Skill",
    # Models — valence
    "ValenceAssociation",
    "SalienceScore",
    # Models — retrieval
    "RetrievalQuery",
    "RetrievalResult",
    # Models — learning
    "RewardSignal",
    "ConsolidationResult",
    # Models — meta-cognition
    "Strategy",
    "MetaEvaluation",
    # Config
    "MnemonConfig",
    "CycleConfig",
    "SensoryConfig",
    "WorkingMemoryConfig",
    "SummarizationConfig",
    "EpisodicConfig",
    "EpisodicRetrievalWeights",
    "EpisodicDecayConfig",
    "SemanticConfig",
    "ProceduralConfig",
    "SkillAcquisitionConfig",
    "SkillUtilityConfig",
    "ValenceConfig",
    "ConsolidationConfig",
    "ConsolidationScheduleConfig",
    "ReplayConfig",
    "RewardConfig",
    "AttentionConfig",
    "AttentionWeightsConfig",
    "MetaCognitionConfig",
    "LLMConfig",
    "load_config",
    # Exceptions
    "MnemonError",
    "MemoryError",
    "RetrievalError",
    "ConsolidationError",
    "ConfigError",
    "BackendNotAvailableError",
    "TokenBudgetExceededError",
    "SkillExecutionError",
    "GoalError",
]
