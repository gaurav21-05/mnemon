"""
Unit tests for MnemonConfig and all sub-section configs.

Covers:
- Default value correctness
- TOML file loading
- Validation errors (weight sum constraints, out-of-range values)
- Subsection access patterns
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mnemon.core.config import (
    AttentionWeightsConfig,
    EpisodicRetrievalWeights,
    MnemonConfig,
    load_config,
)
from mnemon.core.exceptions import ConfigError
from mnemon.core.models import EvictionPolicy


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
    cfg = MnemonConfig()
    assert cfg is not None


def test_cycle_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.cycle.max_cycles_per_request == 10
    assert cfg.cycle.cycle_timeout_ms == 5_000
    assert cfg.cycle.parallel_retrieval is True


def test_sensory_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.sensory.capacity == 64
    assert cfg.sensory.ttl_ms == 30_000


def test_working_memory_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.working_memory.token_budget == 8_192
    assert cfg.working_memory.eviction_policy == EvictionPolicy.LRU_IMPORTANCE
    assert cfg.working_memory.summarization.enabled is True
    assert cfg.working_memory.summarization.max_summary_tokens == 256


def test_episodic_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.episodic.backend == "qdrant"
    weights = cfg.episodic.retrieval_weights
    assert abs(weights.semantic + weights.bm25 + weights.recency + weights.importance - 1.0) < 1e-6
    assert cfg.episodic.decay.base_lambda == pytest.approx(0.001)
    assert cfg.episodic.decay.forget_threshold == pytest.approx(0.05)
    assert cfg.episodic.capacity.max_episodes == 100_000


def test_attention_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.attention.broadcast_threshold == pytest.approx(0.7)
    assert cfg.attention.attention_threshold == pytest.approx(0.3)
    w = cfg.attention.weights
    total = w.valence + w.goal_relevance + w.novelty + w.urgency
    assert abs(total - 1.0) < 1e-6


def test_reward_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.reward.discount_factor == pytest.approx(0.95)
    sw = cfg.reward.source_weights
    assert sw.task_success == pytest.approx(0.6)
    assert sw.efficiency == pytest.approx(0.2)
    assert sw.user_feedback == pytest.approx(0.15)
    assert sw.goal_progress == pytest.approx(0.05)


def test_valence_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.valence.learning_rate == pytest.approx(0.05)
    assert cfg.valence.extinction_rate == pytest.approx(0.01)


def test_semantic_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.semantic.graph_backend == "falkordb"
    assert cfg.semantic.raptor.enabled is True
    assert cfg.semantic.raptor.max_levels == 3


def test_procedural_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.procedural.skill_acquisition.enabled is True
    assert cfg.procedural.skill_acquisition.initial_utility == pytest.approx(0.5)
    assert cfg.procedural.utility.learning_rate == pytest.approx(0.1)


def test_consolidation_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.consolidation.enabled is True
    assert cfg.consolidation.batch_size == 32
    assert cfg.consolidation.replay.alpha == pytest.approx(0.6)


def test_llm_defaults() -> None:
    cfg = MnemonConfig()
    assert cfg.llm.default_provider == "openai"
    assert "openai" in cfg.llm.providers
    assert "anthropic" in cfg.llm.providers


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


def test_load_config_from_toml(tmp_path: pytest.fixture) -> None:
    toml_file = tmp_path / "mnemon.toml"
    toml_file.write_text(
        """
[cycle]
max_cycles_per_request = 5
cycle_timeout_ms = 2000

[sensory]
capacity = 32
ttl_ms = 15000

[episodic]
backend = "hnswlib"
""",
        encoding="utf-8",
    )
    cfg = load_config(str(toml_file))
    assert cfg.cycle.max_cycles_per_request == 5
    assert cfg.cycle.cycle_timeout_ms == 2_000
    assert cfg.sensory.capacity == 32
    assert cfg.sensory.ttl_ms == 15_000
    assert cfg.episodic.backend == "hnswlib"


def test_load_config_no_path_returns_defaults() -> None:
    cfg = load_config(None)
    assert cfg.cycle.max_cycles_per_request == 10


def test_load_config_missing_file_raises_config_error(tmp_path: pytest.fixture) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nonexistent.toml"))


def test_load_config_invalid_toml_raises_config_error(tmp_path: pytest.fixture) -> None:
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("this is not : valid toml ===", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(str(bad_toml))


def test_load_config_directory_raises_config_error(tmp_path: pytest.fixture) -> None:
    with pytest.raises(ConfigError, match="not a file"):
        load_config(str(tmp_path))


def test_load_config_toml_overrides_nested_section(tmp_path: pytest.fixture) -> None:
    toml_file = tmp_path / "mnemon.toml"
    toml_file.write_text(
        """
[working_memory]
token_budget = 4096
""",
        encoding="utf-8",
    )
    cfg = load_config(str(toml_file))
    assert cfg.working_memory.token_budget == 4_096


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_episodic_retrieval_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        EpisodicRetrievalWeights(semantic=0.5, bm25=0.5, recency=0.5, importance=0.5)


def test_episodic_retrieval_weights_negative_value_rejected() -> None:
    with pytest.raises(ValidationError):
        EpisodicRetrievalWeights(semantic=-0.1, bm25=0.4, recency=0.4, importance=0.3)


def test_attention_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        AttentionWeightsConfig(valence=0.5, goal_relevance=0.5, novelty=0.5, urgency=0.5)


def test_sensory_capacity_must_be_positive() -> None:
    from mnemon.core.config import SensoryConfig

    with pytest.raises(ValidationError):
        SensoryConfig(capacity=0)


def test_working_memory_budget_too_small() -> None:
    from mnemon.core.config import WorkingMemoryConfig

    with pytest.raises(ValidationError):
        WorkingMemoryConfig(token_budget=100)  # min is 256


def test_episodic_capacity_min_enforced() -> None:
    from mnemon.core.config import EpisodicCapacityConfig

    with pytest.raises(ValidationError):
        EpisodicCapacityConfig(max_episodes=50)  # min is 100


def test_cycle_max_cycles_minimum() -> None:
    from mnemon.core.config import CycleConfig

    with pytest.raises(ValidationError):
        CycleConfig(max_cycles_per_request=0)


# ---------------------------------------------------------------------------
# Subsection access patterns
# ---------------------------------------------------------------------------


def test_episodic_retrieval_weights_access() -> None:
    cfg = MnemonConfig()
    w = cfg.episodic.retrieval_weights
    assert hasattr(w, "semantic")
    assert hasattr(w, "bm25")
    assert hasattr(w, "recency")
    assert hasattr(w, "importance")


def test_attention_weights_access() -> None:
    cfg = MnemonConfig()
    w = cfg.attention.weights
    assert hasattr(w, "valence")
    assert hasattr(w, "goal_relevance")
    assert hasattr(w, "novelty")
    assert hasattr(w, "urgency")


def test_meta_cognition_reflexion_access() -> None:
    cfg = MnemonConfig()
    assert cfg.meta_cognition.enabled is True
    assert cfg.meta_cognition.reflexion.enabled is True
    assert cfg.meta_cognition.max_strategy_switches == 3


def test_consolidation_schedule_access() -> None:
    cfg = MnemonConfig()
    s = cfg.consolidation.schedule
    assert s.mode == "idle"
    assert s.idle_timeout_s == 60
    assert s.episode_threshold == 100


def test_semantic_community_detection_access() -> None:
    cfg = MnemonConfig()
    cd = cfg.semantic.community_detection
    assert cd.algorithm == "leiden"
    assert cd.resolution == pytest.approx(1.0)
