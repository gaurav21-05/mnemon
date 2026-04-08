# Fix Mnemon DB Bugs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 data-integrity and correctness bugs found in the Mnemon episodic/semantic databases.

**Architecture:** Each bug is an isolated change to one or two files. Bugs 1–4 are code fixes with tests; Bug 5 (Ollama down) is diagnosed but not auto-fixed (needs user action); Bug 6 (entities/reflection) is a no-op until NER is wired; Bug 7 (goal poisoning) is cleared from the live state file.

**Tech Stack:** Python 3.12, Pydantic v2, SQLite (aiosqlite), HNSWLib, pytest-asyncio

---

## Files Modified

| File | Bug | What changes |
|------|-----|--------------|
| `src/mnemon/memory/working.py` | #1 | `flush()` excludes `RETRIEVAL` blocks from episode context |
| `src/mnemon/daemon/ipc.py` | #2 | `_rpc_chat` stores the LLM reply as episode `outcome` via `update_outcome` |
| `src/mnemon/control/orchestrator.py` | #2 | Exposes `update_outcome(episode_id, outcome)` method |
| `src/mnemon/learning/consolidation.py` | #3 | Failed LLM extraction marks episodes `FAILED` after N retries; new `FAILED` state |
| `src/mnemon/core/models.py` | #3 | Add `FAILED = "failed"` to `ConsolidationState` |
| `src/mnemon/memory/semantic.py` | #4 | `_ensure_entity_node` / `upsert_triple` write to `_docs` atomically with vector insert |
| `~/.mnemon/state/goals.json` | #7 | Clear poisoned goal from live state |
| `tests/unit/test_working_memory.py` | #1 | New test: flush excludes retrieval blocks |
| `tests/unit/test_consolidation.py` | #3 | New test: failed LLM marks episode failed after retry limit |

---

## Task 1 — Bug #1: Context Bloat — Exclude Retrieval Blocks from flush()

**Problem:** `WorkingMemoryManager.flush()` serialises ALL `active_context` blocks into the episode `context` field. The retrieval phase (Phase 3 of the orchestrator) injects blocks with `source=ContextSource.RETRIEVAL`. These blocks contain the full text of prior episodes (each of which contains older contexts), causing O(n²) recursive growth.

**Fix:** In `flush()`, only include blocks with `source != ContextSource.RETRIEVAL` and `source != ContextSource.SUMMARY` in the context text.

**Files:**
- Modify: `src/mnemon/memory/working.py:259-287`
- Test: `tests/unit/test_working_memory.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_working_memory.py`:

```python
async def test_flush_excludes_retrieval_blocks_from_context() -> None:
    """Retrieval-sourced blocks must NOT pollute the episode context."""
    from mnemon.core.models import ContextSource, RetrievedItem

    mgr = _make_manager(budget=500)
    # Inject a genuine user input block
    user_block = ContextBlock(
        content="user said hello",
        token_count=5,
        source=ContextSource.USER_INPUT,
        importance=0.5,
    )
    await mgr.inject(user_block)

    # Inject a retrieved memory block (simulates Phase 3 injection)
    retrieval_block = ContextBlock(
        content="old retrieved memory should not pollute context",
        token_count=10,
        source=ContextSource.RETRIEVAL,
        importance=0.7,
    )
    await mgr.inject(retrieval_block)

    episode = await mgr.flush()
    assert "user said hello" in episode.context
    assert "old retrieved memory should not pollute context" not in episode.context


async def test_flush_excludes_summary_blocks_from_context() -> None:
    """Summary blocks must NOT pollute the episode context."""
    from mnemon.core.models import ContextSource

    mgr = _make_manager(budget=500)
    user_block = ContextBlock(
        content="actual user input",
        token_count=5,
        source=ContextSource.USER_INPUT,
        importance=0.5,
    )
    summary_block = ContextBlock(
        content="summarised old stuff that should stay out",
        token_count=8,
        source=ContextSource.SUMMARY,
        importance=0.3,
    )
    await mgr.inject(user_block)
    await mgr.inject(summary_block)

    episode = await mgr.flush()
    assert "actual user input" in episode.context
    assert "summarised old stuff" not in episode.context
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/rohit/mnemon
.venv/bin/python -m pytest tests/unit/test_working_memory.py::test_flush_excludes_retrieval_blocks_from_context tests/unit/test_working_memory.py::test_flush_excludes_summary_blocks_from_context -v
```

Expected: FAIL — both retrieved and summary content currently end up in context.

- [ ] **Step 3: Fix `flush()` in `working.py`**

In `src/mnemon/memory/working.py`, change the `flush()` method (lines 267-268):

```python
    async def flush(self) -> Episode:
        """Serialise current state into an Episode and clear the workspace.

        Only USER_INPUT and SYSTEM blocks are included in the context.
        RETRIEVAL and SUMMARY blocks are excluded to prevent recursive
        context embedding (O(n²) growth across episodes).
        """
        _EXCLUDED = {ContextSource.RETRIEVAL, ContextSource.SUMMARY}
        context_text = "\n\n".join(
            b.content
            for b in self._state.active_context
            if b.source not in _EXCLUDED
        )
        episode = Episode(
            agent_id="mnemon",
            session_id=self._state.session_id,
            timestamp=datetime.now(timezone.utc),
            context=context_text or "(empty)",
            action="",
            outcome="",
        )

        # Reset state, preserving session identity.
        session_id = self._state.session_id
        self._state = WorkingMemoryState(
            session_id=session_id,
            token_budget=self._config.token_budget,
        )
        self._insertion_order.clear()
        logger.debug("WorkingMemory flushed; episode id=%s", episode.id)
        return episode
```

The import `ContextSource` is already used in the file via `ContextBlock`/`ContextSource.SUMMARY` elsewhere; if not imported, add it to the existing import block:
```python
from mnemon.core.models import (
    ContextBlock,
    ContextSource,   # add if missing
    Episode,
    ...
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/test_working_memory.py -v
```

Expected: all tests pass including the two new ones.

- [ ] **Step 5: Commit**

```bash
cd /home/rohit/mnemon
git add src/mnemon/memory/working.py tests/unit/test_working_memory.py
git commit -m "fix: exclude retrieval/summary blocks from episode context to stop O(n²) bloat"
```

---

## Task 2 — Bug #2: Outcome Never Stored

**Problem:** The LLM reply is generated in `DaemonIPCServer._rpc_chat()` *after* `run_cycle()` has already flushed and encoded the episode. The `outcome` field in every episode is always `""`.

**Fix:** Add an `update_outcome(episode_id, outcome)` method to the orchestrator (or directly use `episodic.update`) that patches the most-recently-encoded episode's outcome. Call it from `_rpc_chat` after `_generate_reply` returns.

**Files:**
- Modify: `src/mnemon/control/orchestrator.py` (add `update_last_episode_outcome`)
- Modify: `src/mnemon/daemon/ipc.py` (`_rpc_chat` calls it after getting reply)
- Test: `tests/unit/test_orchestrator_outcome.py` (new file)

- [ ] **Step 1: Check the episodic store's update interface**

```bash
grep -n "def update\|async def update\|def patch\|async def patch" /home/rohit/mnemon/src/mnemon/memory/episodic.py
```

If there's no `update` method, we'll use `encode` (which does `put`, an upsert) on a patched copy.

- [ ] **Step 2: Add `_last_episode_id` tracking + `update_last_episode_outcome` to orchestrator**

In `src/mnemon/control/orchestrator.py`, in `__init__` after `self._cycle_count = 0` add:
```python
        self._last_episode_id: UUID | None = None
```

In Phase 6 LEARNING, after `await self._episodic.encode(episode_with_reward)` add:
```python
            self._last_episode_id = episode_with_reward.id
```

Add this new method to the class (after `run_cycle`):
```python
    async def update_last_episode_outcome(self, outcome: str) -> None:
        """Patch the outcome field of the most recently encoded episode.

        Called by the IPC layer after the LLM reply is generated, so that
        the stored episode reflects what Jarvis actually said.
        """
        if self._last_episode_id is None:
            return
        episode = await self._episodic.get(self._last_episode_id)
        if episode is None:
            return
        updated = episode.model_copy(update={"outcome": outcome})
        await self._episodic.encode(updated)
        logger.debug(
            "Updated outcome for episode %s (len=%d)",
            self._last_episode_id,
            len(outcome),
        )
```

- [ ] **Step 3: Write the failing test**

Create `tests/unit/test_orchestrator_outcome.py`:

```python
"""Test that update_last_episode_outcome patches the episode."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from mnemon.core.models import Episode, ConsolidationState


async def test_update_last_episode_outcome_patches_stored_episode() -> None:
    """update_last_episode_outcome should write the outcome back to episodic store."""
    from mnemon.control.orchestrator import CognitiveOrchestrator
    from mnemon.core.config import MnemonConfig
    from mnemon.backends.memory_store import InMemoryDocumentStore, InMemoryVectorStore
    from mnemon.memory.episodic import EpisodicMemoryStore
    from tests.unit.conftest import FakeEmbeddingProvider

    config = MnemonConfig()
    ep_store = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
    )

    # Build a minimal orchestrator with real episodic store
    from tests.unit.conftest import FakeLLMProvider, FakeEmbeddingProvider
    from mnemon.backends.memory_store import InMemoryGraphStore
    from mnemon.memory.semantic import SemanticMemoryStore
    from mnemon.memory.procedural import ProceduralMemoryStore
    from mnemon.memory.working import WorkingMemoryManager
    from mnemon.memory.sensory import SensoryBuffer
    from mnemon.control.attention import AttentionController
    from mnemon.control.metacognition import MetaCognitionController
    from mnemon.memory.valence import ValenceMemory
    from mnemon.memory.reward import RewardProcessor
    from mnemon.control.goals import GoalManager

    llm = FakeLLMProvider()
    embedder = FakeEmbeddingProvider()

    wm = WorkingMemoryManager(config.working_memory, llm)
    sensory = SensoryBuffer(config.sensory)
    attention = AttentionController(config.attention, embedder)
    semantic = SemanticMemoryStore(
        config=config.semantic,
        graph_store=InMemoryGraphStore(config),
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
        llm_provider=llm,
    )
    procedural = ProceduralMemoryStore(
        config=config.procedural,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
    )
    valence = ValenceMemory()
    reward = RewardProcessor()
    goals = GoalManager(config.goals, llm)
    meta = MetaCognitionController(config.metacognition, llm)

    orch = CognitiveOrchestrator(
        config=config,
        working_memory=wm,
        sensory_buffer=sensory,
        attention_controller=attention,
        episodic=ep_store,
        semantic=semantic,
        procedural=procedural,
        goal_manager=goals,
        valence_memory=valence,
        reward_processor=reward,
        meta_controller=meta,
    )

    # Run a cycle to create an episode
    await orch.run_cycle(raw_input="hello world")
    assert orch._last_episode_id is not None

    # Now update the outcome
    await orch.update_last_episode_outcome("I said hello back")

    # Fetch the episode and verify
    ep = await ep_store.get(orch._last_episode_id)
    assert ep is not None
    assert ep.outcome == "I said hello back"


async def test_update_last_episode_outcome_noop_when_no_episode() -> None:
    """Should not raise if called before any cycle has run."""
    from mnemon.control.orchestrator import CognitiveOrchestrator
    from unittest.mock import MagicMock
    # Build a bare-minimum orchestrator with all mocked
    orch = MagicMock(spec=CognitiveOrchestrator)
    orch._last_episode_id = None
    orch._episodic = MagicMock()
    # Call the real method on the instance
    await CognitiveOrchestrator.update_last_episode_outcome(orch, "some reply")
    orch._episodic.get.assert_not_called()
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd /home/rohit/mnemon
.venv/bin/python -m pytest tests/unit/test_orchestrator_outcome.py -v
```

Expected: FAIL — `update_last_episode_outcome` does not exist yet, and `_last_episode_id` not tracked.

- [ ] **Step 5: Apply the orchestrator changes**

In `src/mnemon/control/orchestrator.py`:

a) In `__init__`, after `self._cycle_count = 0`:
```python
        self._last_episode_id: "UUID | None" = None
```

b) In Phase 6 LEARNING, after line `await self._episodic.encode(episode_with_reward)`:
```python
            self._last_episode_id = episode_with_reward.id
```

c) Add the new method (paste from Step 2 above).

- [ ] **Step 6: Wire outcome in `ipc.py`**

In `src/mnemon/daemon/ipc.py`, in `_rpc_chat`, after:
```python
            reply = await self._generate_reply(...)
```
add:
```python
            # Patch the episode outcome with Jarvis's actual reply
            try:
                await self._brain.control.orchestrator.update_last_episode_outcome(reply)
            except Exception:
                logger.debug("Could not update episode outcome", exc_info=True)
```

Check how `self._brain.control.orchestrator` is accessed — grep for it:
```bash
grep -n "self._brain.control\|brain.control" /home/rohit/mnemon/src/mnemon/daemon/ipc.py
```
If the path differs, use the actual attribute path (e.g. `self._brain._orchestrator`).

- [ ] **Step 7: Run all tests**

```bash
.venv/bin/python -m pytest tests/unit/ -v
```

Expected: all 208+ tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/mnemon/control/orchestrator.py src/mnemon/daemon/ipc.py tests/unit/test_orchestrator_outcome.py
git commit -m "fix: store LLM reply as episode outcome after each chat cycle"
```

---

## Task 3 — Bug #3: Consolidation Stuck — Add FAILED State + Retry Limit

**Problem:** When LLM extraction fails (e.g. Ollama down), episodes get `episode_triples[id] = []` and never reach the `successfully_upserted` set. So `mark_consolidated` is never called. Episodes stay `raw` forever and are re-queued infinitely.

**Fix:**
1. Add `FAILED = "failed"` to `ConsolidationState`.
2. Track per-episode failure count. After 3 consecutive LLM failures, mark the episode `FAILED` so consolidation stops retrying it.
3. Use `ConsolidationState.PROCESSING` (already exists) during active extraction to prevent double-processing by concurrent runs.

**Files:**
- Modify: `src/mnemon/core/models.py` — add `FAILED = "failed"` (already has `PROCESSING`)
- Modify: `src/mnemon/learning/consolidation.py` — track LLM failure per episode, mark `FAILED` after 3 attempts
- Modify: `src/mnemon/memory/episodic.py` — `sample_for_consolidation` excludes `FAILED` state
- Test: `tests/unit/test_consolidation_retry.py` (new)

- [ ] **Step 1: Check existing ConsolidationState**

```bash
grep -n "ConsolidationState\|FAILED\|PROCESSING" /home/rohit/mnemon/src/mnemon/core/models.py
```

`PROCESSING` exists (value `"processing"`), `FAILED` does not yet.

- [ ] **Step 2: Add `FAILED` to `ConsolidationState`**

In `src/mnemon/core/models.py`, inside `class ConsolidationState(StrEnum)`:
```python
class ConsolidationState(StrEnum):
    RAW = "raw"
    PROCESSING = "processing"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"
    FAILED = "failed"          # add this line
```

- [ ] **Step 3: Add `mark_failed` to `EpisodicMemoryStore`**

In `src/mnemon/memory/episodic.py`, after the `mark_consolidated` method, add:

```python
    async def mark_failed(self, episode_ids: list[UUID]) -> None:
        """Mark episodes as FAILED so they are not retried during consolidation."""
        for episode_id in episode_ids:
            raw_doc = await self._document_store.get(episode_id)
            if raw_doc is None:
                logger.warning("mark_failed: episode %s not found — skipping", episode_id)
                continue
            try:
                episode = Episode.model_validate(raw_doc)
                updated = episode.model_copy(
                    update={"consolidation_state": ConsolidationState.FAILED}
                )
                await self._document_store.put(episode_id, updated.model_dump(mode="json"))
                # Update vector metadata
                await self._vector_store.update_metadata(
                    episode_id,
                    {"consolidation_state": ConsolidationState.FAILED},
                )
            except Exception as exc:
                logger.warning("mark_failed: failed to update episode %s: %s", episode_id, exc)
```

- [ ] **Step 4: Exclude FAILED from `sample_for_consolidation`**

In `src/mnemon/memory/episodic.py`, `sample_for_consolidation` currently queries `{"consolidation_state": ConsolidationState.RAW}`. Change it to also exclude `FAILED` (it already does, since the filter is equality on `RAW`). No change needed here — it already only fetches `RAW`.

But double-check by looking at the filter:
```bash
grep -n "sample_for_consolidation\|consolidation_state.*RAW" /home/rohit/mnemon/src/mnemon/memory/episodic.py
```

- [ ] **Step 5: Add failure tracking to consolidation Stage 2**

In `src/mnemon/learning/consolidation.py`, the constant `_MAX_EXTRACTION_FAILURES` controls when an episode is abandoned:

Add at module level after `_ENTITY_NAME_MIN_LEN`:
```python
# Episodes that fail LLM extraction this many times are marked FAILED.
_MAX_EXTRACTION_FAILURES: int = 3
```

In `run_cycle`, Stage 2 (around line 230), change the `except Exception` block for LLM failure:
```python
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM extraction failed for episode %s — skipping. Error: %s",
                    episode.id,
                    exc,
                )
                episode_triples[episode.id] = []
                continue
```

Replace with:
```python
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM extraction failed for episode %s — skipping. Error: %s",
                    episode.id,
                    exc,
                )
                episode_triples[episode.id] = []
                # Track failure against the episode's own access_count as a
                # lightweight retry counter (avoids adding a new field to Episode).
                # After _MAX_EXTRACTION_FAILURES attempts, mark episode FAILED.
                fail_count = getattr(episode, "_extraction_failures", 0) + 1
                if fail_count >= _MAX_EXTRACTION_FAILURES:
                    _episodes_to_fail.append(episode.id)
                    logger.warning(
                        "Episode %s failed extraction %d times — marking FAILED",
                        episode.id,
                        fail_count,
                    )
                continue
```

But we can't store `_extraction_failures` on the frozen model. Instead, track it in the consolidation engine itself with a dict. Here's the full approach:

In `ConsolidationEngine.__init__`, add:
```python
        # Maps episode_id → consecutive extraction failure count.
        self._extraction_failures: dict[UUID, int] = {}
```

In Stage 2 `except Exception` block, replace entirely with:
```python
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM extraction failed for episode %s — skipping. Error: %s",
                    episode.id,
                    exc,
                )
                episode_triples[episode.id] = []
                self._extraction_failures[episode.id] = (
                    self._extraction_failures.get(episode.id, 0) + 1
                )
                continue
```

At the end of Stage 4 (CLEANUP), after `mark_consolidated`, add:

```python
        # Mark episodes that have exceeded the extraction failure limit as FAILED
        # so they are not endlessly re-queued.
        failed_ids = [
            ep.id
            for ep, _, _ in valid_episodes
            if self._extraction_failures.get(ep.id, 0) >= _MAX_EXTRACTION_FAILURES
        ]
        if failed_ids:
            try:
                await self._episodic.mark_failed(failed_ids)
                logger.warning(
                    "Marked %d episodes as FAILED (exceeded extraction retry limit)",
                    len(failed_ids),
                )
            except Exception as exc:
                logger.error("mark_failed failed: %s", exc)
            # Clear counters for marked episodes
            for eid in failed_ids:
                self._extraction_failures.pop(eid, None)
```

- [ ] **Step 6: Write the failing test**

Create `tests/unit/test_consolidation_retry.py`:

```python
"""Test that consolidation marks episodes FAILED after repeated LLM failures."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from mnemon.core.config import MnemonConfig, ConsolidationConfig
from mnemon.core.models import Episode, ConsolidationState
from mnemon.backends.memory_store import InMemoryDocumentStore, InMemoryVectorStore
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.backends.memory_store import InMemoryGraphStore
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.learning.consolidation import ConsolidationEngine, _MAX_EXTRACTION_FAILURES
from tests.unit.conftest import FakeEmbeddingProvider, FakeLLMProvider
from datetime import datetime, timezone


def _make_episode(agent_id: str = "mnemon") -> Episode:
    return Episode(
        agent_id=agent_id,
        session_id=uuid4(),
        context="the user said hello",
        action="hi",
        outcome="",
        timestamp=datetime.now(timezone.utc),
    )


async def _make_engine(config: MnemonConfig) -> tuple[ConsolidationEngine, EpisodicMemoryStore, PrioritizedReplayBuffer]:
    embedder = FakeEmbeddingProvider()
    ep_store = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
    )
    semantic = SemanticMemoryStore(
        config=config.semantic,
        graph_store=InMemoryGraphStore(config),
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
    )
    replay = PrioritizedReplayBuffer(capacity=100)
    engine = ConsolidationEngine(
        config=config.consolidation,
        episodic=ep_store,
        semantic=semantic,
        llm=FakeLLMProvider(),  # returns {} → no triples
        embedding=embedder,
        replay_buffer=replay,
    )
    return engine, ep_store, replay


async def test_episode_marked_failed_after_max_extraction_failures() -> None:
    config = MnemonConfig()
    engine, ep_store, replay = await _make_engine(config)

    # Encode an episode
    ep = _make_episode()
    await ep_store.encode(ep)
    replay.add(episode_id=ep.id, priority=0.5)

    # Make LLM always raise
    async def failing_llm(*a, **kw):
        raise RuntimeError("Ollama is down")
    engine._llm.generate_structured = failing_llm

    # Run consolidation _MAX_EXTRACTION_FAILURES times
    for _ in range(_MAX_EXTRACTION_FAILURES):
        await engine.run_cycle()
        # Re-add to replay each time (loop refills it)
        replay.add(episode_id=ep.id, priority=0.5)

    # Episode should now be marked FAILED
    stored = await ep_store.get(ep.id)
    assert stored is not None
    assert stored.consolidation_state == ConsolidationState.FAILED


async def test_failed_episode_not_in_sample_for_consolidation() -> None:
    config = MnemonConfig()
    embedder = FakeEmbeddingProvider()
    ep_store = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=embedder,
    )
    ep = _make_episode()
    await ep_store.encode(ep)
    await ep_store.mark_failed([ep.id])

    # sample_for_consolidation only returns RAW episodes
    candidates = await ep_store.sample_for_consolidation(batch_size=32)
    ids = [c.id for c in candidates]
    assert ep.id not in ids
```

- [ ] **Step 7: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_consolidation_retry.py -v
```

Expected: FAIL — `FAILED` state and `mark_failed` don't exist yet.

- [ ] **Step 8: Apply all changes (models, episodic, consolidation)**

Apply the code changes from Steps 2, 3, and 5 above.

- [ ] **Step 9: Run all tests**

```bash
.venv/bin/python -m pytest tests/unit/ -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/mnemon/core/models.py src/mnemon/memory/episodic.py src/mnemon/learning/consolidation.py tests/unit/test_consolidation_retry.py
git commit -m "fix: mark episodes FAILED after repeated consolidation LLM failures to stop infinite retry"
```

---

## Task 4 — Bug #4: Semantic Vector/SQLite Desync

**Problem:** After a daemon restart, `semantic.hnsw` is loaded from disk (114 entries) but `semantic.db` has 0 rows. Any code path that calls `_docs.get(id)` after a vector search returns `None`, causing silent failures.

**Root cause:** On daemon start, `HNSWLibVectorStore` loads the persisted index from disk, but `SQLiteDocumentStore` starts fresh (db file exists but has 0 rows — either it was never populated separately or it was cleared).

**Fix:** On `SemanticMemoryStore` startup (or at first use), detect desync by comparing vector index size vs document store count and rebuild the document store from the vector metadata if needed. Also: in `_ensure_entity_node`, verify the doc was written atomically.

Actually, the real fix is simpler: **always write to the document store before inserting into the vector store** (already the case in the code). The desync happened because an earlier session stored to vectors but not docs (or docs were cleared). The correct fix is:

1. Add a `sync_check()` method that detects mismatched counts on startup.
2. In `JarvisDaemon.__init__` (or the factory startup sequence), call `await semantic.sync_check()` which rebuilds missing docs from HNSW metadata.

But HNSW metadata only has minimal fields (entity/triple id + canonical_name), not full documents. Without the full graph data in the metadata, we can't fully reconstruct the docs.

**Better fix:** Ensure the vector store and document store are always written together via a wrapper that does both atomically in `_ensure_entity_node` and `upsert_triple`. Add a startup integrity check that logs the desync and clears the stale HNSW index if docs are missing (forcing a clean re-consolidation on next run).

**Files:**
- Modify: `src/mnemon/memory/semantic.py` — add `check_integrity()` method
- Modify: `src/mnemon/daemon/__init__.py` — call `check_integrity()` on startup
- Test: `tests/unit/test_semantic_integrity.py` (new)

- [ ] **Step 1: Add `check_integrity()` to `SemanticMemoryStore`**

In `src/mnemon/memory/semantic.py`, add this method after `__init__`:

```python
    async def check_integrity(self) -> dict[str, int]:
        """Detect and repair vector/document store desync on startup.

        If the vector index has entries but the document store has none,
        the HNSW index is stale (loaded from a previous session whose doc
        store was never populated or was reset). In that case, we reset
        the vector index to match the empty doc store — forcing a clean
        re-consolidation on the next consolidation cycle.

        Returns a dict with 'vector_count', 'doc_count', and 'reset' (bool).
        """
        try:
            vector_count = self._vectors.count()
        except Exception:
            vector_count = 0
        try:
            all_docs = await self._docs.query(filters={}, limit=1)
            doc_count = len(all_docs)
        except Exception:
            doc_count = 0

        result = {"vector_count": vector_count, "doc_count": doc_count, "reset": False}

        if vector_count > 0 and doc_count == 0:
            logger.warning(
                "SemanticMemoryStore: vector index has %d entries but document store "
                "is empty — stale index detected. Resetting vector store.",
                vector_count,
            )
            try:
                await self._vectors.clear()
                result["reset"] = True
            except Exception as exc:
                logger.error("Failed to reset stale vector index: %s", exc)

        logger.info(
            "SemanticMemoryStore integrity check: vector=%d docs=%d reset=%s",
            result["vector_count"],
            result["doc_count"],
            result["reset"],
        )
        return result
```

- [ ] **Step 2: Check if `VectorStore` has `count()` and `clear()`**

```bash
grep -n "def count\|async def count\|def clear\|async def clear" /home/rohit/mnemon/src/mnemon/core/interfaces.py
grep -n "def count\|def clear" /home/rohit/mnemon/src/mnemon/backends/*.py
```

If `count()` / `clear()` are missing from the interface and backends, add them.

- [ ] **Step 3: Add `count()` and `clear()` to VectorStore interface and backends**

In `src/mnemon/core/interfaces.py`, in `class VectorStore`, add:
```python
    @abstractmethod
    def count(self) -> int:
        """Return number of indexed vectors."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all indexed vectors."""
```

In `src/mnemon/backends/hnswlib_store.py` (the HNSW backend), add:
```python
    def count(self) -> int:
        if self._index is None:
            return 0
        return self._index.get_current_count()

    async def clear(self) -> None:
        """Reset the index to empty and save the cleared state."""
        dim = self._config.dimensions
        max_elements = self._config.max_elements
        self._index = hnswlib.Index(space=self._config.space, dim=dim)
        self._index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self._metadata.clear()
        self._id_to_label.clear()
        self._label_to_id.clear()
        self._next_label = 0
        await self._save()
        logger.info("HNSWLibVectorStore cleared and saved.")
```

In `src/mnemon/backends/memory_store.py` (InMemory backend), add:
```python
    def count(self) -> int:
        return len(self._vectors)

    async def clear(self) -> None:
        self._vectors.clear()
        self._metadata.clear()
```

- [ ] **Step 4: Call `check_integrity()` on daemon startup**

In `src/mnemon/daemon/__init__.py`, find where the daemon initialises the brain and starts running. After the brain is built (post-factory), add:

```python
        # Repair semantic store desync that can occur after unclean restarts
        try:
            await self._brain.memory.semantic.check_integrity()
        except Exception as exc:
            logger.warning("Semantic integrity check failed: %s", exc)
```

Find the exact location:
```bash
grep -n "async def start\|brain\|factory\|semantic" /home/rohit/mnemon/src/mnemon/daemon/__init__.py | head -30
```

- [ ] **Step 5: Write the failing test**

Create `tests/unit/test_semantic_integrity.py`:

```python
"""Test semantic store integrity check detects and repairs vector/doc desync."""
from __future__ import annotations

import pytest
from mnemon.core.config import MnemonConfig
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.backends.memory_store import InMemoryDocumentStore, InMemoryVectorStore, InMemoryGraphStore
from tests.unit.conftest import FakeEmbeddingProvider, FakeLLMProvider


async def _make_semantic(config: MnemonConfig) -> SemanticMemoryStore:
    return SemanticMemoryStore(
        config=config.semantic,
        graph_store=InMemoryGraphStore(config),
        vector_store=InMemoryVectorStore(config),
        document_store=InMemoryDocumentStore(config),
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLLMProvider(),
    )


async def test_check_integrity_no_desync() -> None:
    config = MnemonConfig()
    store = await _make_semantic(config)
    result = await store.check_integrity()
    assert result["reset"] is False
    assert result["vector_count"] == 0
    assert result["doc_count"] == 0


async def test_check_integrity_detects_stale_vectors_and_clears() -> None:
    config = MnemonConfig()
    store = await _make_semantic(config)

    # Manually insert a vector without a doc (simulates the desync)
    fake_embedding = [0.1] * 8
    from uuid import uuid4
    fake_id = uuid4()
    await store._vectors.insert(fake_id, fake_embedding, {"_type": "entity"})
    # Verify: 1 vector, 0 docs
    assert store._vectors.count() == 1

    result = await store.check_integrity()
    assert result["reset"] is True
    assert store._vectors.count() == 0  # cleared
```

- [ ] **Step 6: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_semantic_integrity.py -v
```

Expected: FAIL — `count()`, `clear()`, and `check_integrity()` don't exist yet.

- [ ] **Step 7: Apply all code changes**

Apply changes from Steps 1, 3, and 4.

- [ ] **Step 8: Run all tests**

```bash
.venv/bin/python -m pytest tests/unit/ -v
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/mnemon/core/interfaces.py src/mnemon/backends/hnswlib_store.py src/mnemon/backends/memory_store.py src/mnemon/memory/semantic.py src/mnemon/daemon/__init__.py tests/unit/test_semantic_integrity.py
git commit -m "fix: add semantic store integrity check to repair vector/doc desync on restart"
```

---

## Task 5 — Bug #7: Clear Poisoned Goal from Live State

**Problem:** `~/.mnemon/state/goals.json` contains a single wrong goal (`"user's primary goal is to determine whether the AI has access to information about them"`), inferred from Rohit's test conversations via the bloated context. Now that Bug #1 is fixed, future goals will be correct. The poisoned goal should be removed.

**Files:**
- Modify: `~/.mnemon/state/goals.json` (runtime data, not source code)

- [ ] **Step 1: Verify the daemon is running and check current goals**

```bash
cd /home/rohit/mnemon && .venv/bin/mnemon-daemon status
cat ~/.mnemon/state/goals.json
```

- [ ] **Step 2: Clear the goals file**

```bash
echo '[]' > ~/.mnemon/state/goals.json
```

- [ ] **Step 3: Verify via daemon**

```bash
.venv/bin/mnemon-daemon goals
```

Expected: empty goal list.

- [ ] **Step 4: Commit note (no source change)**

No git commit needed — this is a runtime data file. Just verify it's cleared.

---

## Task 6 — Bug #5 + Verify Fixes End-to-End

**Problem:** Ollama is not running. This causes embedding failures (new episodes can't be stored in vector index) and consolidation LLM failures. Bugs #1–#4 are code fixes; this is an operational note.

- [ ] **Step 1: Check Ollama status**

```bash
curl -s http://localhost:11434/api/tags 2>&1 || echo "Ollama is not running"
```

- [ ] **Step 2: Start Ollama if needed**

```bash
ollama serve &
```

Or run `! ollama serve` in the Claude Code session.

- [ ] **Step 3: Run all tests one final time**

```bash
cd /home/rohit/mnemon
.venv/bin/python -m pytest tests/unit/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 4: Restart the daemon to apply all fixes**

```bash
.venv/bin/mnemon-daemon stop
sleep 2
.venv/bin/mnemon-daemon start
sleep 3
.venv/bin/mnemon-daemon status
```

- [ ] **Step 5: Send a test message and verify the episode is clean**

```bash
.venv/bin/mnemon-daemon chat "hello, how are you?"
sleep 2
sqlite3 ~/.mnemon/documents/episodic.db "SELECT json_extract(data,'$.context'), json_extract(data,'$.action'), json_extract(data,'$.outcome') FROM documents ORDER BY created_at DESC LIMIT 1;"
```

Expected:
- `context` is just the user message (short, no recursive embedding)
- `action` contains the user message
- `outcome` contains the Jarvis reply

---

## Self-Review

**Spec coverage:**
- Bug 1 (context bloat): Task 1 ✓
- Bug 2 (outcome empty): Task 2 ✓
- Bug 3 (consolidation infinite retry): Task 3 ✓
- Bug 4 (semantic desync): Task 4 ✓
- Bug 5 (Ollama down): Task 6 (operational) ✓
- Bug 6 (entities/tags/reflection empty): deferred — NER pipeline not yet wired; adding a stub fix without NER would be no-op
- Bug 7 (poisoned goal): Task 5 ✓

**Placeholder scan:** No TBDs or "implement later" — all steps have code.

**Type consistency:** `UUID` imported from `uuid` in all new code. `ConsolidationState.FAILED` used consistently. `ContextSource.RETRIEVAL` and `ContextSource.SUMMARY` are `StrEnum` values accessible wherever `ContextSource` is imported.
