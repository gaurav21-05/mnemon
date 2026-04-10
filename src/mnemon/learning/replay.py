"""
Prioritized Experience Replay for hippocampal consolidation.

Brain analog: The hippocampal indexing and replay system that preferentially
reactivates high-salience experiences during offline consolidation (slow-wave
sleep). Surprising or emotionally charged memories receive higher priority,
mirroring the modulatory role of norepinephrine and dopamine in tagging
episodic traces for preferential reactivation.

References:
    Schaul et al. (2016) "Prioritized Experience Replay". ICLR 2016.
    https://arxiv.org/abs/1511.05952
"""

from __future__ import annotations

import logging
import random
from typing import Any
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class ReplayExperience(BaseModel):
    """A single experience sampled from the prioritized replay buffer."""

    model_config = {"frozen": True}

    episode_id: UUID
    tree_index: int
    priority: float
    is_weight: float  # importance-sampling weight


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------


class SumTree:
    """Binary sum-tree for O(log N) prioritized experience replay.

    Brain analog: The hippocampal indexing system that preferentially
    replays high-salience experiences during consolidation (sleep).

    The tree is backed by a flat array of size ``2 * capacity - 1``.
    Leaf nodes occupy indices ``[capacity - 1, 2 * capacity - 1)``.
    Internal nodes store the sum of their children; the root (index 0)
    holds the total priority.

    The leaf region acts as a circular buffer: writes wrap around via
    ``_write_pos``, overwriting the oldest leaf when capacity is reached.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"SumTree capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        # Tree array: indices 0..(capacity-2) are internal; (capacity-1)..(2*capacity-2) are leaves
        self._tree: list[float] = [0.0] * (2 * capacity - 1)
        # Circular data buffer storing episode IDs (or None for empty slots)
        self._data: list[Any] = [None] * capacity
        self._write_pos: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of experiences currently stored."""
        return self._size

    @property
    def capacity(self) -> int:
        """Maximum number of experiences."""
        return self._capacity

    def total_priority(self) -> float:
        """Total priority sum (root node value)."""
        return self._tree[0]

    def add(self, priority: float, data: Any) -> None:
        """Add experience with given priority. Overwrites oldest entry if full."""
        leaf_index = self._write_pos + (self._capacity - 1)
        self._data[self._write_pos] = data
        self.update(leaf_index, priority)

        self._write_pos = (self._write_pos + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def update(self, tree_index: int, priority: float) -> None:
        """Update priority at a specific tree index and propagate the delta upward."""
        delta = priority - self._tree[tree_index]
        self._tree[tree_index] = priority
        # Propagate delta to all ancestors
        idx = tree_index
        while idx > 0:
            idx = (idx - 1) // 2
            self._tree[idx] += delta

    def sample(self, value: float) -> tuple[int, float, Any]:
        """Sample one experience by traversing the tree with the given priority value.

        Parameters
        ----------
        value:
            A uniform random value in ``[0, total_priority()]``.

        Returns
        -------
        tuple[int, float, Any]
            ``(tree_index, priority, data)`` for the sampled leaf.
        """
        idx = self._retrieve(0, value)
        data_index = idx - (self._capacity - 1)
        return idx, self._tree[idx], self._data[data_index]

    def active_leaves(self) -> list[tuple[int, float, Any]]:
        """Return active leaf entries as ``(tree_index, priority, data)`` tuples."""
        leaf_start = self._capacity - 1
        leaves: list[tuple[int, float, Any]] = []
        for data_index, data in enumerate(self._data):
            tree_index = leaf_start + data_index
            priority = self._tree[tree_index]
            if data is not None and priority > 0.0:
                leaves.append((tree_index, priority, data))
        return leaves

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, idx: int, value: float) -> int:
        """Recursively traverse toward the leaf whose cumulative range contains value."""
        left = 2 * idx + 1
        right = left + 1

        # Reached a leaf node
        if left >= len(self._tree):
            return idx

        if value <= self._tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self._tree[left])


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree.

    Implements Schaul et al. (2016) prioritized sampling with
    importance-sampling (IS) correction for the bias introduced by
    non-uniform sampling.

    Brain analog: The hippocampal replay mechanism during sleep that
    preferentially reactivates surprising or emotionally salient memories.
    The IS weights correct for the statistical distortion introduced by
    prioritized sampling — analogous to the brain's normalisation of
    dopaminergic signals to avoid runaway potentiation.

    Parameters
    ----------
    capacity:
        Maximum number of experiences to retain.
    alpha:
        Priority exponent. 0 = uniform sampling; 1 = fully proportional.
    beta_start:
        Initial IS correction exponent. Anneals toward 1.0 over training.
    """

    _EPSILON: float = 1e-6  # prevents zero-priority experiences

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not (0.0 <= beta_start <= 1.0):
            raise ValueError(f"beta_start must be in [0, 1], got {beta_start}")

        self._tree = SumTree(capacity)
        self._alpha = alpha
        self._beta = beta_start
        self._beta_start = beta_start
        logger.debug(
            "PrioritizedReplayBuffer initialised (capacity=%d, alpha=%.2f, beta_start=%.2f)",
            capacity,
            alpha,
            beta_start,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of stored experiences."""
        return self._tree.size

    def add(self, episode_id: UUID, priority: float) -> None:
        """Add episode with computed priority = (|priority| + epsilon)^alpha."""
        scaled = (abs(priority) + self._EPSILON) ** self._alpha
        self._tree.add(scaled, episode_id)
        logger.debug("Added episode %s with scaled priority %.6f", episode_id, scaled)

    def sample(self, batch_size: int) -> list[ReplayExperience]:
        """Sample *batch_size* experiences with proportional prioritization.

        The total priority range is divided into *batch_size* equal segments.
        One experience is drawn uniformly at random from each segment,
        ensuring broad coverage of the replay buffer while still favouring
        high-priority experiences.

        Importance-sampling weights correct for the non-uniform sampling bias:
            w_i = (1 / (N * P(i)))^beta  /  max_weight

        Returns
        -------
        list[ReplayExperience]
            Sampled experiences with their IS weights.  Returns an empty list
            if the buffer is empty or total priority is zero.
        """
        if self._tree.size == 0:
            logger.warning("sample() called on empty PrioritizedReplayBuffer")
            return []

        total = self._tree.total_priority()
        if total <= 0.0:
            logger.warning("sample() called with zero total priority")
            return []

        n = self._tree.size
        effective_batch = min(batch_size, n)
        active_leaves = self._tree.active_leaves()
        if effective_batch >= n:
            return self._experiences_with_weights(active_leaves, total, n)

        segment = total / effective_batch

        sampled: list[tuple[int, float, Any]] = []

        for i in range(effective_batch):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)  # noqa: S311 — not security-sensitive
            # Clamp to avoid floating-point overshoot past root sum
            value = max(0.0, min(value, total - 1e-9))
            tree_index, priority, episode_id = self._tree.sample(value)
            # Guard against sampling an uninitialised or invalid leaf.
            if episode_id is None:
                logger.warning("sample: None episode_id at tree_index=%d", tree_index)
                continue
            sampled.append((tree_index, priority, episode_id))

        weighted = self._experiences_with_weights(sampled, total, n)
        if not weighted:
            logger.warning("sample: all sampled leaves were uninitialised — returning empty")
            return []
        return weighted

    def _experiences_with_weights(
        self,
        sampled: list[tuple[int, float, Any]],
        total: float,
        population_size: int,
    ) -> list[ReplayExperience]:
        """Build validated replay experiences and normalised IS weights."""
        experiences: list[ReplayExperience] = []
        priorities: list[float] = []

        for tree_index, priority, episode_id in sampled:
            try:
                experiences.append(
                    ReplayExperience(
                        episode_id=episode_id,
                        tree_index=tree_index,
                        priority=priority,
                        is_weight=0.0,
                    )
                )
            except Exception:
                logger.warning(
                    "sample: ReplayExperience rejected episode_id=%r type=%s tree_index=%d",
                    episode_id,
                    type(episode_id).__name__,
                    tree_index,
                )
                continue
            priorities.append(priority)

        if not experiences:
            return []

        # Compute IS weights: w_i = (1/(N*P(i)))^beta, normalised by max weight
        min_prob = min(p / total for p in priorities)
        max_weight = (1.0 / (population_size * min_prob)) ** self._beta

        weighted: list[ReplayExperience] = []
        for exp, p in zip(experiences, priorities, strict=False):
            prob = p / total
            weight = (1.0 / (population_size * prob)) ** self._beta / max_weight
            weighted.append(
                ReplayExperience(
                    episode_id=exp.episode_id,
                    tree_index=exp.tree_index,
                    priority=exp.priority,
                    is_weight=weight,
                )
            )

        logger.debug("Sampled %d experiences (beta=%.3f)", len(weighted), self._beta)
        return weighted

    def update_priorities(
        self,
        tree_indices: list[int],
        new_priorities: list[float],
    ) -> None:
        """Update priorities after learning (e.g., based on consolidation yield).

        Parameters
        ----------
        tree_indices:
            Tree-level indices returned in the sampled ``ReplayExperience`` objects.
        new_priorities:
            Corresponding new raw priority values (before alpha scaling).
        """
        if len(tree_indices) != len(new_priorities):
            raise ValueError(
                f"tree_indices and new_priorities must have the same length, "
                f"got {len(tree_indices)} vs {len(new_priorities)}"
            )
        for idx, p in zip(tree_indices, new_priorities, strict=False):
            scaled = (abs(p) + self._EPSILON) ** self._alpha
            self._tree.update(idx, scaled)
        logger.debug("Updated %d priorities", len(tree_indices))

    def anneal_beta(self, fraction: float) -> None:
        """Anneal beta from *beta_start* toward 1.0.

        Parameters
        ----------
        fraction:
            Training progress in ``[0, 1]``.  At ``fraction=1.0``,
            beta reaches 1.0, providing full bias correction.
        """
        fraction = max(0.0, min(1.0, fraction))
        self._beta = self._beta_start + fraction * (1.0 - self._beta_start)
        logger.debug("Annealed beta to %.4f (fraction=%.4f)", self._beta, fraction)
