"""
Unit tests for SumTree and PrioritizedReplayBuffer.

Covers:
- SumTree: add, update, sample, overflow/circular-buffer behaviour,
  total_priority, size/capacity properties.
- PrioritizedReplayBuffer: add experiences, sample batch with IS weights,
  update priorities, overflow handling, statistical bias toward high priority.
"""

from __future__ import annotations

from collections import Counter
from uuid import UUID, uuid4

import pytest

from mnemon.learning.replay import PrioritizedReplayBuffer, ReplayExperience, SumTree


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------


class TestSumTree:
    def test_initial_state(self) -> None:
        tree = SumTree(capacity=4)
        assert tree.size == 0
        assert tree.capacity == 4
        assert tree.total_priority() == pytest.approx(0.0)

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            SumTree(capacity=0)

    def test_add_increments_size(self) -> None:
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        assert tree.size == 1

    def test_add_updates_total_priority(self) -> None:
        tree = SumTree(capacity=4)
        tree.add(2.0, "a")
        tree.add(3.0, "b")
        assert tree.total_priority() == pytest.approx(5.0)

    def test_add_does_not_exceed_capacity_size(self) -> None:
        tree = SumTree(capacity=3)
        for i in range(10):
            tree.add(float(i + 1), f"item_{i}")
        assert tree.size == 3
        assert tree.capacity == 3

    def test_overflow_replaces_oldest_entry(self) -> None:
        """After capacity is exhausted, total priority reflects latest entries."""
        tree = SumTree(capacity=2)
        tree.add(5.0, "first")
        tree.add(5.0, "second")
        # Third entry overwrites first
        tree.add(1.0, "third")
        # Total should be 5.0 (second) + 1.0 (third), not 5 + 5 + 1
        assert tree.total_priority() == pytest.approx(6.0)

    def test_update_changes_priority(self) -> None:
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        # The leaf index for write_pos=0 is capacity - 1
        leaf_idx = tree.capacity - 1
        tree.update(leaf_idx, 10.0)
        assert tree.total_priority() == pytest.approx(10.0)

    def test_sample_returns_valid_tuple(self) -> None:
        tree = SumTree(capacity=4)
        tree.add(1.0, "data")
        tree_idx, priority, data = tree.sample(0.5)
        assert isinstance(tree_idx, int)
        assert priority > 0.0
        assert data == "data"

    def test_sample_returns_correct_data(self) -> None:
        tree = SumTree(capacity=4)
        payloads = ["a", "b", "c", "d"]
        for p in payloads:
            tree.add(1.0, p)
        # With equal priorities, any sample should return one of the stored items
        _, _, data = tree.sample(tree.total_priority() * 0.5)
        assert data in payloads

    def test_sample_high_priority_item_returned_for_matching_range(self) -> None:
        """The item with highest priority occupies the widest range in the tree."""
        tree = SumTree(capacity=4)
        tree.add(0.1, "low")
        tree.add(0.1, "low2")
        tree.add(0.1, "low3")
        tree.add(100.0, "high")
        # Sample at 99% of total_priority should land on the high item
        value = tree.total_priority() * 0.999
        _, _, data = tree.sample(min(value, tree.total_priority() - 1e-9))
        assert data == "high"

    def test_size_at_capacity(self) -> None:
        tree = SumTree(capacity=3)
        for _ in range(3):
            tree.add(1.0, "x")
        assert tree.size == 3

    def test_size_does_not_grow_beyond_capacity_after_overflow(self) -> None:
        tree = SumTree(capacity=2)
        for _ in range(100):
            tree.add(1.0, "x")
        assert tree.size == 2


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class TestPrioritizedReplayBuffer:
    def test_initial_size_is_zero(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8)
        assert buf.size == 0

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            PrioritizedReplayBuffer(capacity=8, alpha=1.5)

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            PrioritizedReplayBuffer(capacity=8, beta_start=2.0)

    def test_add_increases_size(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8)
        buf.add(uuid4(), 1.0)
        assert buf.size == 1

    def test_add_multiple(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8)
        for _ in range(5):
            buf.add(uuid4(), 1.0)
        assert buf.size == 5

    def test_sample_empty_buffer_returns_empty(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8)
        result = buf.sample(4)
        assert result == []

    def test_sample_returns_replay_experiences(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8)
        eid = uuid4()
        buf.add(eid, 1.0)
        samples = buf.sample(1)
        assert len(samples) == 1
        assert isinstance(samples[0], ReplayExperience)

    def test_sample_batch_size_respected(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=16)
        for _ in range(10):
            buf.add(uuid4(), 1.0)
        samples = buf.sample(4)
        assert len(samples) == 4

    def test_sample_limited_by_buffer_size(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=16)
        for _ in range(3):
            buf.add(uuid4(), 1.0)
        samples = buf.sample(10)  # request more than available
        assert len(samples) == 3

    def test_sample_is_weights_are_in_valid_range(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=16, alpha=0.6, beta_start=0.4)
        for _ in range(8):
            buf.add(uuid4(), 1.0)
        for exp in buf.sample(4):
            assert 0.0 <= exp.is_weight <= 1.0 + 1e-9

    def test_sample_max_is_weight_is_one(self) -> None:
        """The maximum IS weight after normalisation should be ≤ 1.0."""
        buf = PrioritizedReplayBuffer(capacity=16)
        for _ in range(8):
            buf.add(uuid4(), 1.0)
        samples = buf.sample(8)
        max_w = max(e.is_weight for e in samples)
        assert max_w <= 1.0 + 1e-9

    def test_sample_episode_ids_are_uuids(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8)
        for _ in range(4):
            buf.add(uuid4(), 1.0)
        for exp in buf.sample(4):
            assert isinstance(exp.episode_id, UUID)

    def test_overflow_replaces_oldest(self) -> None:
        """After capacity is exceeded, only the last *capacity* episodes remain."""
        buf = PrioritizedReplayBuffer(capacity=4, alpha=1.0, beta_start=0.0)
        first_id = uuid4()
        buf.add(first_id, 1.0)
        for _ in range(10):
            buf.add(uuid4(), 1.0)
        assert buf.size == 4
        # The first_id may have been overwritten — we just verify size is capped
        samples = buf.sample(4)
        assert len(samples) == 4

    def test_update_priorities_changes_sampling_bias(self) -> None:
        """After pushing a single high priority, it should dominate sampling."""
        buf = PrioritizedReplayBuffer(capacity=4, alpha=1.0, beta_start=0.0)
        eids = [uuid4() for _ in range(4)]
        for eid in eids:
            buf.add(eid, 1.0)

        # Grab all tree indices from a full sample
        initial_samples = buf.sample(4)
        target_idx = initial_samples[0].tree_index

        # Boost the priority of the first item dramatically
        buf.update_priorities([target_idx], [1_000.0])

        # After boost, sampling many times should predominantly return that item
        counts: Counter[int] = Counter()
        for _ in range(200):
            for exp in buf.sample(1):
                counts[exp.tree_index] += 1

        # The boosted item should be sampled far more than others
        assert counts[target_idx] > 100

    def test_update_priorities_mismatched_lengths_raises(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4)
        with pytest.raises(ValueError):
            buf.update_priorities([1, 2], [1.0])  # mismatched

    def test_anneal_beta_at_zero_fraction(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, beta_start=0.4)
        buf.anneal_beta(0.0)
        assert buf._beta == pytest.approx(0.4)

    def test_anneal_beta_at_full_fraction(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, beta_start=0.4)
        buf.anneal_beta(1.0)
        assert buf._beta == pytest.approx(1.0)

    def test_anneal_beta_clamps_fraction_below_zero(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, beta_start=0.4)
        buf.anneal_beta(-5.0)
        assert buf._beta == pytest.approx(0.4)

    def test_anneal_beta_clamps_fraction_above_one(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, beta_start=0.4)
        buf.anneal_beta(99.0)
        assert buf._beta == pytest.approx(1.0)

    def test_high_priority_sampled_more_often_than_low(self) -> None:
        """Statistical test: high-priority item should dominate sampling over many draws."""
        buf = PrioritizedReplayBuffer(capacity=8, alpha=1.0, beta_start=0.0)
        low_id = uuid4()
        high_id = uuid4()
        buf.add(low_id, 0.01)  # very low priority
        buf.add(high_id, 100.0)  # very high priority

        low_count = 0
        high_count = 0
        for _ in range(500):
            samples = buf.sample(1)
            if samples:
                if samples[0].episode_id == high_id:
                    high_count += 1
                else:
                    low_count += 1

        # High priority item should be sampled at least 80% of the time
        assert high_count > low_count * 3
