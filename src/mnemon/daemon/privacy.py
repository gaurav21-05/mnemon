"""Privacy rule storage and enforcement helpers for Mnemon daemon memory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class PrivacyRules(BaseModel):
    """Persisted exclusion and redaction controls for memory capture."""

    excluded_phrases: list[str] = Field(default_factory=list)
    redaction_phrases: list[str] = Field(default_factory=list)


def privacy_rules_path(state_dir: Path) -> Path:
    """Return the canonical privacy rules file path."""
    return state_dir / "privacy_rules.json"


def load_privacy_rules(state_dir: Path) -> PrivacyRules:
    """Load privacy rules, creating a default file if needed."""
    path = privacy_rules_path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        save_privacy_rules(state_dir, PrivacyRules())
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PrivacyRules.model_validate(raw)


def save_privacy_rules(state_dir: Path, rules: PrivacyRules) -> None:
    """Persist privacy rules as formatted JSON."""
    path = privacy_rules_path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(rules.model_dump_json(indent=2), encoding="utf-8")


def should_exclude_text(text: str, rules: PrivacyRules) -> bool:
    """Return True when any configured exclusion phrase matches."""
    lowered = text.lower()
    return any(
        phrase.strip().lower() in lowered
        for phrase in rules.excluded_phrases
        if phrase.strip()
    )


def apply_redactions(text: str, rules: PrivacyRules) -> str:
    """Replace configured redaction phrases with a stable token."""
    redacted = text
    for phrase in rules.redaction_phrases:
        cleaned = phrase.strip()
        if not cleaned:
            continue
        redacted = redacted.replace(cleaned, "[REDACTED]")
    return redacted
