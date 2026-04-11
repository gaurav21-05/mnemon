from __future__ import annotations

from mnemon.daemon.capture_policy import classify_interaction


def test_classify_interaction_marks_static_and_dynamic_profile_content() -> None:
    decision = classify_interaction(
        user_message="My name is Rohit and I prefer dark mode. I'm working on mnemon this week.",
        assistant_reply="Got it.",
        active_goals=["Ship memory UX"],
        source="chat",
    )

    assert decision.store_memory is True
    assert "auto_capture" in decision.tags
    assert "profile_static" in decision.tags
    assert "profile_dynamic" in decision.tags
    assert "project_context" in decision.tags
    assert decision.importance >= 0.7


def test_classify_interaction_excludes_private_content() -> None:
    decision = classify_interaction(
        user_message="<private>my banking password is secret</private>",
        assistant_reply="I won't retain that.",
        source="chat",
    )

    assert decision.store_memory is False
    assert "private_excluded" in decision.tags


def test_classify_interaction_skips_low_signal_chatter() -> None:
    decision = classify_interaction(
        user_message="ok thanks",
        assistant_reply="Any time.",
        source="chat",
    )

    assert decision.store_memory is False
    assert "ephemeral" in decision.tags


def test_classify_interaction_keeps_short_goal_shaping_turns() -> None:
    decision = classify_interaction(
        user_message="Vercel",
        assistant_reply="We'll use Vercel.",
        active_goals=["Build a portfolio website"],
        source="chat",
    )

    assert decision.store_memory is True
    assert "project_context" in decision.tags


def test_classify_interaction_marks_role_identity_statements_static() -> None:
    decision = classify_interaction(
        user_message="I am a full stack TypeScript developer and agentic AI developer",
        assistant_reply="Understood.",
        source="chat",
    )

    assert decision.store_memory is True
    assert "profile_static" in decision.tags
