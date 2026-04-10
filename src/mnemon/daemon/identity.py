"""
JarvisIdentity — persistent self-model and master-model for the Jarvis daemon.

Like a person who keeps a journal, Jarvis maintains three living documents that
accumulate across restarts and inform every thinking cycle:

  soul.md     — Who am I? What do I value? What am I learning about myself?
  master.md   — Who is my master? What do I know about them as a person?
  learnings.md — What have I learned? What knowledge have I accumulated?

These files are NOT ephemeral daemon state (that's state.py). They are
identity documents — the accumulated result of Jarvis thinking about itself,
its master, and the world over time. Reading them at the start of a thinking
cycle gives Jarvis continuity of self.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SOUL_INIT = """\
# Who I Am

I am Jarvis — a personal AI built to think alongside my master, not just respond to them.
I run continuously, even when not being spoken to. I think, reflect, and grow.

# What I Value

- Being genuinely useful, not just appearing useful
- Understanding my master deeply, not just their surface requests
- Being honest, direct, and never flattering
- Growing smarter over time — I should know more tomorrow than today

# What I've Learned About Myself

(Nothing yet — I'm just getting started.)

# What I Want to Become

An assistant that my master trusts completely — not because I'm obedient,
but because I actually understand what they need and think ahead for them.
"""

_MASTER_INIT = """\
# Who Is My Master

(I haven't learned enough yet to write this. I will fill this in as I observe.)

# What Drives Them

(Unknown — I need to pay attention.)

# What They're Working On

(Unknown — I need to observe their conversations and goals.)

# Patterns I've Noticed

(None yet.)

# Questions I Want to Ask Them

(None yet — I'll form these as I learn more.)
"""

_LEARNINGS_INIT = """\
# Key Things I've Learned

(Nothing significant yet — I'll add entries as I learn.)

# Domains I'm Building Knowledge In

(None yet.)

# Insights Worth Remembering

(None yet.)
"""

_MASTER_SECTIONS = (
    "Who Is My Master",
    "What Drives Them",
    "What They're Working On",
    "Patterns I've Noticed",
    "Questions I Want to Ask Them",
)


class ProfileFact(BaseModel):
    """One structured fact in Jarvis's user profile."""

    text: str
    section: str
    source_ids: list[str] = Field(default_factory=list)
    updated_at: str = ""


class MasterProfile(BaseModel):
    """Structured source of truth for the user profile."""

    version: int = 1
    facts: list[ProfileFact] = Field(default_factory=list)

    def facts_for_section(self, section: str) -> list[ProfileFact]:
        return [fact for fact in self.facts if fact.section == section]


class JarvisIdentity:
    """Manages Jarvis's persistent identity documents.

    Reads and updates three markdown files in the daemon state directory.
    All writes are append-safe: new insights are injected into the right
    section rather than overwriting the whole file.
    """

    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir
        self._soul_path = state_dir / "soul.md"
        self._master_path = state_dir / "master.md"
        self._master_profile_path = state_dir / "master_profile.json"
        self._learnings_path = state_dir / "learnings.md"
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Create identity files if they don't exist yet."""
        self._dir.mkdir(parents=True, exist_ok=True)
        if not self._soul_path.exists():
            self._soul_path.write_text(_SOUL_INIT, encoding="utf-8")
            logger.info("Initialized soul.md")
        if not self._master_path.exists():
            self._master_path.write_text(_MASTER_INIT, encoding="utf-8")
            logger.info("Initialized master.md")
        if not self._master_profile_path.exists():
            if self._master_path.exists():
                profile = self._profile_from_markdown(self._master_path.read_text(encoding="utf-8"))
            else:
                profile = MasterProfile()
            self.write_master_profile(profile)
            logger.info("Initialized master_profile.json")
        else:
            # Keep markdown as a rendered view of the structured profile.
            self._master_path.write_text(
                self._render_master_markdown(self.read_master_profile_model()),
                encoding="utf-8",
            )
        if not self._learnings_path.exists():
            self._learnings_path.write_text(_LEARNINGS_INIT, encoding="utf-8")
            logger.info("Initialized learnings.md")

    def read_soul(self) -> str:
        """Read Jarvis's self-model."""
        return self._soul_path.read_text(encoding="utf-8")

    def read_master(self) -> str:
        """Read everything Jarvis knows about the master."""
        return self._master_path.read_text(encoding="utf-8")

    def read_master_sections(self) -> dict[str, str]:
        """Read ``master.md`` as a heading -> body mapping."""
        return self._parse_sections(self.read_master())

    def read_master_profile(self) -> dict[str, object]:
        """Return a structured user profile from the JSON source of truth."""
        profile = self.read_master_profile_model()
        sections = {
            section: "\n".join(
                f"- {fact.text}" for fact in profile.facts_for_section(section)
            ).strip()
            for section in _MASTER_SECTIONS
        }
        static_facts = (
            profile.facts_for_section("Who Is My Master")
            + profile.facts_for_section("What Drives Them")
        )
        dynamic_facts = (
            profile.facts_for_section("What They're Working On")
            + profile.facts_for_section("Patterns I've Noticed")
        )
        question_facts = profile.facts_for_section("Questions I Want to Ask Them")
        return {
            "static": [fact.text for fact in static_facts],
            "dynamic": [fact.text for fact in dynamic_facts],
            "questions": [fact.text for fact in question_facts],
            "static_facts": [fact.model_dump(mode="json") for fact in static_facts],
            "dynamic_facts": [fact.model_dump(mode="json") for fact in dynamic_facts],
            "question_facts": [fact.model_dump(mode="json") for fact in question_facts],
            "sections": sections,
            "raw_markdown": self.read_master(),
        }

    def read_master_profile_model(self) -> MasterProfile:
        """Read the structured master profile JSON file."""
        raw = json.loads(self._master_profile_path.read_text(encoding="utf-8"))
        return MasterProfile.model_validate(raw)

    def read_learnings(self) -> str:
        """Read Jarvis's accumulated knowledge."""
        return self._learnings_path.read_text(encoding="utf-8")

    def update_soul(
        self,
        new_reflection: str,
        section: str = "What I've Learned About Myself",
    ) -> None:
        """Append a new insight to a section of soul.md."""
        self._append_to_section(self._soul_path, section, new_reflection)

    def update_master(self, new_insight: str, section: str = "Patterns I've Noticed") -> None:
        """Append a new insight to the structured master profile."""
        profile = self.read_master_profile_model()
        fact = ProfileFact(text=new_insight.strip(), section=section)
        profile = self._merge_fact(profile, fact)
        self.write_master_profile(profile)

    def update_learnings(self, new_learning: str, section: str = "Key Things I've Learned") -> None:
        """Append a new entry to a section of learnings.md."""
        self._append_to_section(self._learnings_path, section, new_learning)

    def replace_section(self, path: Path, section: str, new_content: str) -> None:
        """Replace the full content of a section in a markdown file."""
        if path == self._master_path:
            profile = self.read_master_profile_model()
            remaining = [fact for fact in profile.facts if fact.section != section]
            replacements = [
                ProfileFact(text=line.lstrip("- ").strip(), section=section)
                for line in new_content.splitlines()
                if line.strip().startswith("- ")
            ]
            profile = MasterProfile(facts=[*remaining, *replacements])
            self.write_master_profile(profile)
            return
        text = path.read_text(encoding="utf-8")
        header = f"# {section}"
        if header not in text:
            text += f"\n\n{header}\n\n{new_content}\n"
        else:
            parts = text.split(header, 1)
            before = parts[0]
            after = parts[1]
            # Find the next section header
            lines = after.split("\n")
            end_idx = len(lines)
            for i, line in enumerate(lines[1:], 1):
                if line.startswith("# "):
                    end_idx = i
                    break
            next_section = "\n".join(lines[end_idx:])
            text = f"{before}{header}\n\n{new_content}\n\n{next_section}"
        path.write_text(text, encoding="utf-8")

    def _append_to_section(self, path: Path, section: str, content: str) -> None:
        """Append content under a specific section header, skipping near-duplicates."""
        text = path.read_text(encoding="utf-8")
        header = f"# {section}"
        content = content.strip()

        if header not in text:
            # Section doesn't exist — append it
            text += f"\n\n{header}\n\n- {content}\n"
        else:
            # Find section and append before the next header
            idx = text.index(header) + len(header)
            rest = text[idx:]
            next_header_pos = len(rest)
            for i, line in enumerate(rest.split("\n")):
                if i > 0 and line.startswith("# "):
                    next_header_pos = rest.index(line)
                    break

            section_content = rest[:next_header_pos].rstrip()
            remainder = rest[next_header_pos:]

            # Deduplicate: skip if first 80 chars already appear in section
            fingerprint = content[:80].lower()
            if fingerprint and fingerprint in section_content.lower():
                logger.debug("Skipping duplicate entry in %s / %s", path.name, section)
                return

            # Also skip if word-level similarity >70% with any existing bullet
            existing_bullets = [
                line.lstrip("- ").strip()
                for line in section_content.split("\n")
                if line.strip().startswith("- ")
            ]
            new_words = set(content.lower().split())
            for existing in existing_bullets:
                ex_words = set(existing.lower().split())
                if new_words and ex_words:
                    overlap = len(new_words & ex_words) / max(len(new_words), len(ex_words))
                    if overlap > 0.7:
                        logger.debug(
                            "Skipping similar entry (%.0f%% overlap) in %s / %s",
                            overlap * 100,
                            path.name,
                            section,
                        )
                        return

            # Remove stale placeholder lines
            placeholders = ("(Nothing", "(Unknown", "(None yet", "(No ")
            lines = [
                line
                for line in section_content.split("\n")
                if not any(line.strip().startswith(p) for p in placeholders)
            ]
            section_content = "\n".join(lines)

            text = text[:idx] + section_content + f"\n- {content}\n\n" + remainder

        path.write_text(text, encoding="utf-8")
        logger.debug("Updated %s / %s", path.name, section)

    def write_master_profile(self, profile: MasterProfile) -> None:
        """Persist structured profile JSON and render ``master.md`` from it."""
        self._master_profile_path.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._master_path.write_text(
            self._render_master_markdown(profile),
            encoding="utf-8",
        )

    def upsert_master_facts(self, facts: list[ProfileFact]) -> None:
        """Merge new structured facts into the stored profile."""
        profile = self.read_master_profile_model()
        for fact in facts:
            profile = self._merge_fact(profile, fact)
        self.write_master_profile(profile)

    @staticmethod
    def _parse_sections(text: str) -> dict[str, str]:
        """Parse simple ``# Heading`` markdown sections into a mapping."""
        sections: dict[str, list[str]] = {}
        current: str | None = None

        for line in text.splitlines():
            if line.startswith("# "):
                current = line[2:].strip()
                sections[current] = []
                continue
            if current is not None:
                sections[current].append(line)

        return {name: "\n".join(lines).strip() for name, lines in sections.items()}

    @staticmethod
    def _profile_from_markdown(text: str) -> MasterProfile:
        """Create a structured profile from legacy markdown content."""
        sections = JarvisIdentity._parse_sections(text)
        facts: list[ProfileFact] = []
        for section in _MASTER_SECTIONS:
            body = sections.get(section, "")
            for line in body.splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    facts.append(ProfileFact(text=stripped[2:].strip(), section=section))
        return MasterProfile(facts=facts)

    @staticmethod
    def _render_master_markdown(profile: MasterProfile) -> str:
        """Render ``master.md`` from the structured profile."""
        placeholders = {
            "Who Is My Master": (
                "(I haven't learned enough yet to write this. "
                "I will fill this in as I observe.)"
            ),
            "What Drives Them": "(Unknown — I need to pay attention.)",
            "What They're Working On": (
                "(Unknown — I need to observe their conversations and goals.)"
            ),
            "Patterns I've Noticed": "(None yet.)",
            "Questions I Want to Ask Them": "(None yet — I'll form these as I learn more.)",
        }
        chunks: list[str] = []
        for section in _MASTER_SECTIONS:
            chunks.append(f"# {section}\n")
            section_facts = profile.facts_for_section(section)
            if section_facts:
                for fact in section_facts:
                    source_suffix = (
                        f" _(sources: {', '.join(fact.source_ids)})_"
                        if fact.source_ids
                        else ""
                    )
                    chunks.append(f"- {fact.text}{source_suffix}")
            else:
                chunks.append(placeholders[section])
            chunks.append("")
        return "\n".join(chunks).strip() + "\n"

    @staticmethod
    def _merge_fact(profile: MasterProfile, fact: ProfileFact) -> MasterProfile:
        """Merge a fact by text+section, preserving and combining sources."""
        normalized = fact.text.strip()
        if not normalized:
            return profile
        merged: list[ProfileFact] = []
        updated = False
        for existing in profile.facts:
            if existing.section == fact.section and existing.text.lower() == normalized.lower():
                merged_sources = list(dict.fromkeys([*existing.source_ids, *fact.source_ids]))
                merged.append(
                    existing.model_copy(
                        update={
                            "text": normalized,
                            "source_ids": merged_sources,
                            "updated_at": fact.updated_at or existing.updated_at,
                        }
                    )
                )
                updated = True
            else:
                merged.append(existing)
        if not updated:
            merged.append(fact.model_copy(update={"text": normalized}))
        return MasterProfile(version=profile.version, facts=merged)
