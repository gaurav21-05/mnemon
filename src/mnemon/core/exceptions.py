"""
Custom exception hierarchy for Mnemon.

Each exception maps to a distinct failure domain in the cognitive architecture,
allowing callers to catch at the right level of granularity without over-catching.
"""

from __future__ import annotations


class MnemonError(Exception):
    """
    Root exception for all Mnemon errors.

    Analogous to an unhandled interrupt in the brain — something went wrong
    at the framework level. Catch this only when you want a catch-all.
    """


class MemoryError(MnemonError):
    """
    Raised when a memory store operation fails.

    Covers failures in any of the long-term or working memory subsystems —
    read, write, or update operations that cannot complete successfully.
    """


class RetrievalError(MemoryError):
    """
    Raised when a retrieval query fails to complete.

    Distinct from an empty result: this indicates the retrieval pipeline
    itself broke (backend unavailable, embedding failure, timeout, etc.).
    """


class ConsolidationError(MnemonError):
    """
    Raised when the offline consolidation pipeline encounters a fatal error.

    Consolidation is the sleep-like process that moves raw episodic memory
    into structured semantic knowledge. Failures here risk knowledge loss.
    """


class ConfigError(MnemonError):
    """
    Raised when configuration is invalid, missing, or cannot be loaded.

    Caught at startup to fail fast before any cognitive cycle begins.
    """


class BackendNotAvailableError(MnemonError):
    """
    Raised when an optional storage backend dependency is not installed.

    Signals that the requested backend (e.g. Qdrant, FalkorDB) requires
    an optional extra that was not included in the installation.
    """

    def __init__(self, backend: str, extra: str) -> None:
        self.backend = backend
        self.extra = extra
        super().__init__(
            f"Backend '{backend}' is not available. "
            f"Install it with: pip install mnemon[{extra}]"
        )


class TokenBudgetExceededError(MemoryError):
    """
    Raised when working memory would exceed its token budget.

    Working memory (prefrontal cortex analog) has a hard capacity limit.
    This error signals that the context window is full and eviction or
    summarization must occur before new content can be admitted.
    """

    def __init__(self, requested: int, budget: int, used: int) -> None:
        self.requested = requested
        self.budget = budget
        self.used = used
        available = budget - used
        super().__init__(
            f"Token budget exceeded: requested {requested} tokens "
            f"but only {available} available ({used}/{budget} used)."
        )


class SkillExecutionError(MnemonError):
    """
    Raised when a procedural skill fails during execution.

    Procedural memory (basal ganglia analog) encodes learned action sequences.
    This error fires when a skill's preconditions are unmet, the definition
    is malformed, or execution raises an unhandled exception.
    """

    def __init__(self, skill_name: str, reason: str) -> None:
        self.skill_name = skill_name
        self.reason = reason
        super().__init__(f"Skill '{skill_name}' execution failed: {reason}")


class GoalError(MnemonError):
    """
    Raised when goal management encounters an inconsistent or invalid state.

    Covers illegal goal transitions (e.g. completing an already-failed goal),
    circular goal dependencies, or attempts to push goals onto a full stack.
    """
