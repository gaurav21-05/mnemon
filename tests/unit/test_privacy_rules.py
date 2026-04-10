from __future__ import annotations

from mnemon.daemon.privacy import (
    PrivacyRules,
    apply_redactions,
    load_privacy_rules,
    should_exclude_text,
)


def test_load_privacy_rules_creates_default_file(tmp_path) -> None:
    rules = load_privacy_rules(tmp_path)

    assert rules == PrivacyRules()
    assert (tmp_path / "privacy_rules.json").exists() is True


def test_should_exclude_text_matches_configured_phrase() -> None:
    rules = PrivacyRules(excluded_phrases=["secret project"])

    assert should_exclude_text("This is about my Secret Project roadmap", rules) is True


def test_apply_redactions_replaces_configured_phrases() -> None:
    rules = PrivacyRules(redaction_phrases=["API_KEY_123"])

    assert (
        apply_redactions("token API_KEY_123 should be hidden", rules)
        == "token [REDACTED] should be hidden"
    )
