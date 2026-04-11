"""Heuristics for automatic daemon-side memory capture."""

from __future__ import annotations

from dataclasses import dataclass

_PRIVATE_MARKERS = (
    "<private>",
    "</private>",
    "don't remember this",
    "do not remember this",
    "off the record",
    "private:",
)

_LOW_SIGNAL_MESSAGES = {
    "ok",
    "okay",
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "no",
    "n",
    "nope",
    "thanks",
    "thank you",
    "ok thanks",
    "cool",
    "nice",
    "great",
    "sounds good",
}

_STATIC_MARKERS = (
    "my name is",
    "call me",
    "i prefer",
    "i like",
    "i love",
    "my favorite",
    "i use",
    "i'm a",
    "i am a",
    "i live",
)

_DYNAMIC_MARKERS = (
    "i'm working on",
    "i am working on",
    "currently",
    "this week",
    "today",
    "right now",
    "next step",
    "trying to",
    "need to ship",
)

_PROJECT_MARKERS = (
    "repo",
    "repository",
    "project",
    "workspace",
    "codebase",
    "pr",
    "daemon",
    "mnemon",
    "test",
    "bug",
    "feature",
)

_ROLE_MARKERS = (
    "developer",
    "engineer",
    "designer",
    "writer",
    "founder",
    "student",
    "researcher",
    "manager",
)


@dataclass(frozen=True)
class CaptureDecision:
    """Result of automatic capture classification."""

    store_memory: bool
    importance: float
    tags: list[str]


def classify_interaction(
    *,
    user_message: str,
    assistant_reply: str = "",
    active_goals: list[str] | None = None,
    source: str = "chat",
    excluded_phrases: list[str] | None = None,
) -> CaptureDecision:
    """Classify whether a daemon interaction should become durable memory."""
    active_goals = active_goals or []
    excluded_phrases = excluded_phrases or []
    combined = user_message.strip()
    lowered = combined.lower()
    user_lower = user_message.strip().lower()

    tags = ["auto_capture", f"source:{source}"]

    if any(marker in lowered for marker in _PRIVATE_MARKERS):
        tags.append("private_excluded")
        return CaptureDecision(store_memory=False, importance=0.0, tags=tags)

    if any(phrase.strip().lower() in lowered for phrase in excluded_phrases if phrase.strip()):
        tags.append("private_excluded")
        return CaptureDecision(store_memory=False, importance=0.0, tags=tags)

    if user_lower in _LOW_SIGNAL_MESSAGES and len(user_lower) <= 24:
        tags.append("ephemeral")
        return CaptureDecision(store_memory=False, importance=0.1, tags=tags)

    importance = 0.35

    if any(marker in lowered for marker in _STATIC_MARKERS):
        tags.append("profile_static")
        importance += 0.25

    if (
        user_lower.startswith(("i am ", "i'm "))
        and any(marker in user_lower for marker in _ROLE_MARKERS)
    ):
        tags.append("profile_static")
        importance += 0.25

    if any(marker in lowered for marker in _DYNAMIC_MARKERS):
        tags.append("profile_dynamic")
        importance += 0.2

    if any(marker in lowered for marker in _PROJECT_MARKERS):
        tags.append("project_context")
        importance += 0.15

    # Once Jarvis has an active goal, short follow-up turns often carry key
    # task-shaping decisions such as style, hosting, scope, or stack choices.
    if active_goals and 1 < len(user_lower) <= 80 and user_lower not in _LOW_SIGNAL_MESSAGES:
        tags.append("project_context")
        importance += 0.1

    if "remember" in lowered or "important" in lowered:
        importance += 0.1

    if len(combined) > 180:
        importance += 0.05

    if tags == ["auto_capture", f"source:{source}"]:
        tags.append("ephemeral")
        if len(combined) < 60:
            return CaptureDecision(store_memory=False, importance=0.15, tags=tags)

    deduped_tags = list(dict.fromkeys(tags))
    return CaptureDecision(
        store_memory=True,
        importance=min(1.0, max(0.0, importance)),
        tags=deduped_tags,
    )
