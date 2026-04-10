from __future__ import annotations

import json

from mnemon.daemon.identity import JarvisIdentity, MasterProfile, ProfileFact


def test_write_master_profile_renders_markdown_and_json(tmp_path) -> None:
    identity = JarvisIdentity(tmp_path)
    profile = MasterProfile(
        facts=[
            ProfileFact(
                text="Rohit prefers dark mode",
                section="What Drives Them",
                source_ids=["ep-1"],
                updated_at="2026-04-09T09:00:00+00:00",
            ),
            ProfileFact(
                text="Working on mnemon memory UX",
                section="What They're Working On",
                source_ids=["ep-2"],
                updated_at="2026-04-09T09:05:00+00:00",
            ),
        ]
    )

    identity.write_master_profile(profile)

    payload = identity.read_master_profile()
    raw_json = json.loads((tmp_path / "master_profile.json").read_text(encoding="utf-8"))
    rendered = (tmp_path / "master.md").read_text(encoding="utf-8")

    assert raw_json["facts"][0]["source_ids"] == ["ep-1"]
    assert payload["static"] == ["Rohit prefers dark mode"]
    assert payload["dynamic"] == ["Working on mnemon memory UX"]
    assert payload["static_facts"][0]["source_ids"] == ["ep-1"]
    assert "sources: ep-1" in rendered
    assert "Working on mnemon memory UX" in rendered


def test_update_master_uses_structured_profile_source_of_truth(tmp_path) -> None:
    identity = JarvisIdentity(tmp_path)

    identity.update_master("Prefers direct answers", section="Patterns I've Noticed")

    profile = identity.read_master_profile_model()
    assert [fact.text for fact in profile.facts_for_section("Patterns I've Noticed")] == [
        "Prefers direct answers"
    ]
