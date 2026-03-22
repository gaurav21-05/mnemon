"""
Mnemon — a brain-like cognitive memory framework for AI agents.

Provides a modular, bio-inspired memory architecture with episodic,
semantic, procedural, and working memory subsystems connected via
a thalamic event bus.

Public API surface is intentionally minimal at the top level. Import
directly from subpackages (e.g. ``mnemon.core``) for full access.
Heavy modules (backends, LLM providers) are loaded lazily on first use.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("mnemon")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Core models and config are lightweight — safe to import eagerly.
# Everything else (vector backends, graph backends, LiteLLM) is deferred.
from mnemon.core import (
    # Config
    MnemonConfig,
    load_config,
    # Exceptions — always useful at the top level
    BackendNotAvailableError,
    ConfigError,
    ConsolidationError,
    GoalError,
    MemoryError,
    MnemonError,
    RetrievalError,
    SkillExecutionError,
    TokenBudgetExceededError,
    # Key model types callers need without digging into subpackages
    CognitiveMessage,
    ContextBlock,
    Episode,
    Entity,
    Goal,
    PerceptUnit,
    Skill,
    WorkingMemoryState,
)


def __getattr__(name: str) -> object:
    """
    Lazy loader for heavy optional submodules.

    Allows ``import mnemon; mnemon.memory`` without paying the import cost
    of vector or graph backend dependencies at startup.
    """
    _lazy_modules = {
        "memory": "mnemon.memory",
        "backends": "mnemon.backends",
        "providers": "mnemon.providers",
        "learning": "mnemon.learning",
        "control": "mnemon.control",
        "evaluation": "mnemon.evaluation",
    }

    # Lazy-loaded classes from mnemon.factory
    _lazy_classes = {
        "Mnemon": ("mnemon.factory", "Mnemon"),
        "MnemonFactory": ("mnemon.factory", "MnemonFactory"),
    }
    if name in _lazy_classes:
        import importlib

        mod_path, cls_name = _lazy_classes[name]
        module = importlib.import_module(mod_path)
        obj = getattr(module, cls_name)
        globals()[name] = obj
        return obj
    if name in _lazy_modules:
        import importlib

        module = importlib.import_module(_lazy_modules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'mnemon' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Config
    "MnemonConfig",
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
    # Core models
    "CognitiveMessage",
    "ContextBlock",
    "Episode",
    "Entity",
    "Goal",
    "PerceptUnit",
    "Skill",
    "WorkingMemoryState",
    # Factory (lazy)
    "Mnemon",
    "MnemonFactory",
    # Lazy submodules
    "memory",
    "backends",
    "providers",
    "learning",
    "control",
    "evaluation",
]
