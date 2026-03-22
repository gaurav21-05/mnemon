"""
Unit tests for RewardProcessor.

Covers:
- compute_rpe: TD(0) formula correctness, terminal and non-terminal states
- Statistics tracking: running mean, cycle count, net RPE
- compute_composite_reward: weighted combination, unknown keys ignored, clamping
- get_stats: structure and types
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from mnemon.core.config import RewardConfig, RewardSourceWeights
from mnemon.learning.reward import RewardProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(
    discount: float = 0.95,
    task_success: float = 0.6,
    efficiency: float = 0.2,
    user_feedback: float = 0.15,
    goal_progress: float = 0.05,
) -> RewardProcessor:
    weights = RewardSourceWeights(
        task_success=task_success,
        efficiency=efficiency,
        user_feedback=user_feedback,
        goal_progress=goal_progress,
    )
    cfg = RewardConfig(discount_factor=discount, source_weights=weights)
    return RewardProcessor(cfg)


# ---------------------------------------------------------------------------
# compute_rpe — TD(0) formula
# ---------------------------------------------------------------------------


async def test_compute_rpe_terminal_state_no_next_value() -> None:
    """RPE = actual_reward + γ*0 - predicted_value = actual - predicted."""
    proc = _make_processor(discount=0.95)
    eid = uuid4()
    signal = await proc.compute_rpe(eid, predicted_value=0.5, actual_reward=1.0)
    # rpe = 1.0 + 0.95*0 - 0.5 = 0.5
    assert signal.rpe == pytest.approx(0.5)


async def test_compute_rpe_with_next_value() -> None:
    proc = _make_processor(discount=0.9)
    eid = uuid4()
    signal = await proc.compute_rpe(
        eid, predicted_value=0.5, actual_reward=0.2, next_value=0.8
    )
    # rpe = 0.2 + 0.9*0.8 - 0.5 = 0.2 + 0.72 - 0.5 = 0.42
    assert signal.rpe == pytest.approx(0.42, rel=1e-6)


async def test_compute_rpe_negative_error() -> None:
    """Worse-than-expected outcome produces a negative RPE."""
    proc = _make_processor(discount=0.95)
    eid = uuid4()
    signal = await proc.compute_rpe(eid, predicted_value=0.9, actual_reward=0.1)
    # rpe = 0.1 + 0 - 0.9 = -0.8
    assert signal.rpe < 0.0
    assert signal.rpe == pytest.approx(-0.8)


async def test_compute_rpe_zero_error() -> None:
    proc = _make_processor(discount=0.5)
    eid = uuid4()
    signal = await proc.compute_rpe(eid, predicted_value=0.5, actual_reward=0.5)
    assert signal.rpe == pytest.approx(0.0)


async def test_compute_rpe_episode_id_preserved() -> None:
    proc = _make_processor()
    eid = uuid4()
    signal = await proc.compute_rpe(eid, predicted_value=0.0, actual_reward=1.0)
    assert signal.episode_id == eid


async def test_compute_rpe_predicted_and_actual_preserved() -> None:
    proc = _make_processor()
    eid = uuid4()
    signal = await proc.compute_rpe(eid, predicted_value=0.3, actual_reward=0.7)
    assert signal.predicted_value == pytest.approx(0.3)
    assert signal.actual_reward == pytest.approx(0.7)


async def test_compute_rpe_signal_is_frozen() -> None:
    """RewardSignal should be immutable (frozen Pydantic model)."""
    proc = _make_processor()
    signal = await proc.compute_rpe(uuid4(), predicted_value=0.0, actual_reward=1.0)
    with pytest.raises(Exception):
        signal.rpe = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------


async def test_get_stats_initial_state() -> None:
    proc = _make_processor()
    stats = proc.get_stats()
    assert stats["cycle_count"] == pytest.approx(0.0)
    assert stats["total_positive_rpe"] == pytest.approx(0.0)
    assert stats["total_negative_rpe"] == pytest.approx(0.0)
    assert stats["mean_rpe"] == pytest.approx(0.0)
    assert stats["net_rpe"] == pytest.approx(0.0)


async def test_stats_cycle_count_increments() -> None:
    proc = _make_processor()
    for _ in range(3):
        await proc.compute_rpe(uuid4(), predicted_value=0.5, actual_reward=0.5)
    assert proc.get_stats()["cycle_count"] == pytest.approx(3.0)


async def test_stats_positive_rpe_accumulated() -> None:
    proc = _make_processor(discount=0.0)
    # Two positive RPE signals of +0.5 each
    await proc.compute_rpe(uuid4(), predicted_value=0.0, actual_reward=0.5)
    await proc.compute_rpe(uuid4(), predicted_value=0.0, actual_reward=0.5)
    stats = proc.get_stats()
    assert stats["total_positive_rpe"] == pytest.approx(1.0)
    assert stats["total_negative_rpe"] == pytest.approx(0.0)


async def test_stats_negative_rpe_accumulated() -> None:
    proc = _make_processor(discount=0.0)
    # Two negative RPE signals of -0.3 each
    await proc.compute_rpe(uuid4(), predicted_value=0.5, actual_reward=0.2)
    await proc.compute_rpe(uuid4(), predicted_value=0.5, actual_reward=0.2)
    stats = proc.get_stats()
    assert stats["total_negative_rpe"] == pytest.approx(-0.6, abs=1e-6)


async def test_stats_mean_rpe_computed_correctly() -> None:
    proc = _make_processor(discount=0.0)
    # RPE values: +1.0 and -0.5  → net = 0.5, mean = 0.25
    await proc.compute_rpe(uuid4(), predicted_value=0.0, actual_reward=1.0)
    await proc.compute_rpe(uuid4(), predicted_value=0.5, actual_reward=0.0)
    stats = proc.get_stats()
    assert stats["mean_rpe"] == pytest.approx(0.25, abs=1e-6)


async def test_stats_net_rpe_equals_sum_of_both_accumulators() -> None:
    proc = _make_processor(discount=0.0)
    await proc.compute_rpe(uuid4(), predicted_value=0.0, actual_reward=0.8)  # +0.8
    await proc.compute_rpe(uuid4(), predicted_value=0.9, actual_reward=0.2)  # -0.7
    stats = proc.get_stats()
    expected_net = stats["total_positive_rpe"] + stats["total_negative_rpe"]
    assert stats["net_rpe"] == pytest.approx(expected_net, abs=1e-9)


async def test_stats_keys_present() -> None:
    proc = _make_processor()
    stats = proc.get_stats()
    required_keys = {
        "total_positive_rpe",
        "total_negative_rpe",
        "cycle_count",
        "mean_rpe",
        "net_rpe",
    }
    assert required_keys.issubset(stats.keys())


# ---------------------------------------------------------------------------
# compute_composite_reward
# ---------------------------------------------------------------------------


def test_composite_reward_all_sources() -> None:
    """With all four sources provided and signal = 1.0, composite should be 1.0."""
    proc = _make_processor(
        task_success=0.6,
        efficiency=0.2,
        user_feedback=0.15,
        goal_progress=0.05,
    )
    result = proc.compute_composite_reward(
        {
            "task_success": 1.0,
            "efficiency": 1.0,
            "user_feedback": 1.0,
            "goal_progress": 1.0,
        }
    )
    assert result == pytest.approx(1.0)


def test_composite_reward_no_sources_returns_zero() -> None:
    proc = _make_processor()
    result = proc.compute_composite_reward({})
    assert result == pytest.approx(0.0)


def test_composite_reward_unknown_keys_ignored() -> None:
    proc = _make_processor()
    result = proc.compute_composite_reward({"unknown_key": 999.0})
    assert result == pytest.approx(0.0)


def test_composite_reward_partial_sources_normalised() -> None:
    """If only task_success is provided it should dominate fully → result = signal."""
    proc = _make_processor(task_success=0.6)
    result = proc.compute_composite_reward({"task_success": 0.8})
    assert result == pytest.approx(0.8)


def test_composite_reward_clamped_to_minus_one() -> None:
    proc = _make_processor()
    result = proc.compute_composite_reward({"task_success": -100.0})
    assert result == pytest.approx(-1.0)


def test_composite_reward_clamped_to_plus_one() -> None:
    proc = _make_processor()
    result = proc.compute_composite_reward({"task_success": 100.0})
    assert result == pytest.approx(1.0)


def test_composite_reward_weighted_average() -> None:
    """Verify exact weighted arithmetic with two sources."""
    proc = _make_processor(task_success=0.6, efficiency=0.4)
    # Only task_success and efficiency; weighted_sum = 0.6*1.0 + 0.4*0.5 = 0.8
    # total_weight = 1.0; normalised = 0.8
    result = proc.compute_composite_reward(
        {"task_success": 1.0, "efficiency": 0.5}
    )
    assert result == pytest.approx(0.8, rel=1e-6)


def test_composite_reward_mixed_positive_negative() -> None:
    proc = _make_processor(task_success=0.5, efficiency=0.5)
    result = proc.compute_composite_reward(
        {"task_success": 1.0, "efficiency": -1.0}
    )
    assert result == pytest.approx(0.0, abs=1e-9)
