# Mnemon Memory Strategy from Similar Repo Research — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve Mnemon from a useful daemon with episodic memory into a stronger persistent memory system tailored to a personal assistant / operator workflow. The plan selectively adopts the best ideas from Supermemory, Claude-Mem, and MiroFish without turning Mnemon into a generic memory SaaS or a heavyweight swarm simulator.

**Recommended product direction:** Mnemon should primarily be a **persistent personal memory daemon for agents and operators**. That means:
- take **profile modeling, contradiction handling, and scoped retrieval** from Supermemory
- take **automatic capture, progressive disclosure, citations, and privacy controls** from Claude-Mem
- take **scenario/report workflows** from MiroFish only as an optional advanced capability later

**Architecture:** Build this in three phases. Phase A strengthens Mnemon's core memory model and retrieval UX. Phase B deepens hybrid recall, privacy, and consolidation. Phase C adds optional scenario/report features. The current daemon/UI work in `src/mnemon/daemon/ipc.py`, `src/mnemon/daemon/webui.py`, `src/mnemon/daemon/identity.py`, and `src/mnemon/control/orchestrator.py` is already the right foundation.

**Research basis:**
- **Supermemory** — strong fit for profile memory, static vs dynamic context, contradiction-aware updates, and project-scoped retrieval
- **Claude-Mem** — strong fit for automatic observation capture, layered retrieval, citations, and memory compression
- **MiroFish** — useful mostly for a later scenario / what-if / report layer, not as Mnemon's central architecture

**Tech Stack:** Python 3.12, aiohttp web UI, anyio/asyncio daemon runtime, Mnemon episodic + semantic memory stores, LiteLLM-backed extraction / summarization, pytest / pytest-asyncio.

---

## Why these repos matter for Mnemon

### What Supermemory contributes
- **Static vs dynamic user profile split**
- **Workspace / container scoping**
- **Contradiction and recency handling**
- **One-call profile + recall flow**

### What Claude-Mem contributes
- **Automatic memory capture from interaction flow**
- **Progressive disclosure retrieval** (`search` -> `timeline` -> `details`)
- **Observation IDs / citations**
- **Compression / summarization of accumulated observations**
- **Privacy rules and exclusion controls**

### What MiroFish contributes
- **Simulation / scenario framing**
- **Report-oriented synthesis**
- **Entity / persona / world-state modeling**

### What to avoid
- [ ] Do **not** make Mnemon a full swarm-simulation platform by default
- [ ] Do **not** overbuild connectors before core recall quality is reliable
- [ ] Do **not** rely only on vector search; preserve structured profile state
- [ ] Do **not** surface persistent memory without privacy / exclusion controls

---

## Existing foundation in Mnemon

Mnemon already has the key seams needed for this roadmap:

| Area | Current foundation | Why it matters |
|------|--------------------|----------------|
| Identity/profile docs | `src/mnemon/daemon/identity.py` | Can hold structured user/profile summaries now |
| Episodic retrieval | `src/mnemon/memory/episodic.py` | Supports ranked search + metadata |
| External memory API | `src/mnemon/services/memory_service.py` | Natural place to expose profile-aware APIs |
| Daemon RPC | `src/mnemon/daemon/ipc.py` | Already exposes search/recent/timeline/profile-like routes |
| Operator UI | `src/mnemon/daemon/webui.py` | Already supports search, profile snapshot, timeline, drawer |
| Goal loop | `src/mnemon/control/goals.py` and daemon loop | Can feed dynamic profile / active work state |
| Consolidation | `src/mnemon/learning/consolidation.py` | Natural place for memory compression and stale-fact updates |

---

## Files likely to change across the roadmap

| File | Why it changes |
|------|----------------|
| `src/mnemon/daemon/identity.py` | Move from markdown append-only identity docs toward structured profile sections + conflict-aware updates |
| `src/mnemon/services/memory_service.py` | Add profile-aware retrieval, scoped recall, contradiction-aware writes |
| `src/mnemon/memory/episodic.py` | Extend metadata and filters for scoped/projected search |
| `src/mnemon/memory/semantic.py` | Store durable facts, stale/updated fact relationships, and contradiction handling |
| `src/mnemon/learning/consolidation.py` | Summarize repeated events into compressed durable memory |
| `src/mnemon/control/orchestrator.py` | Mark important events and emit richer capture signals |
| `src/mnemon/daemon/ipc.py` | Add APIs for citations, privacy rules, scope, and hybrid recall |
| `src/mnemon/daemon/webui.py` | Add privacy/configuration, profile diffs, citations, hybrid recall inspection |
| `src/mnemon/daemon/cli/app.py` | Add memory admin / privacy / scope / context commands |
| `tests/unit/test_ipc_improve.py` | Extend daemon memory API tests |
| `tests/unit/test_webui_files.py` | Extend UI contract tests |
| `tests/unit/` new files | Add tests for contradiction handling, automatic capture policy, scoped recall |

---

# Phase A — Core Memory Upgrade (highest ROI)

## Task 1 — Automatic capture policy (Claude-Mem pattern)

**Goal:** Stop depending on purely explicit memory writes. Mnemon should automatically classify interaction fragments into:
- durable user facts/preferences
- active project/work context
- ephemeral short-lived notes
- do-not-store content

**Files:**
- Modify: `src/mnemon/control/orchestrator.py`
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/services/memory_service.py`
- Test: new `tests/unit/test_memory_capture_policy.py`

- [ ] **Step 1: Add a capture policy abstraction**
  - Create a helper that scores whether an event should be stored, how strongly, and in what category.
  - Inputs should include raw user text, assistant reply, active goals, tags, and source channel.

- [ ] **Step 2: Mark categories explicitly**
  - Add category metadata such as `profile_static`, `profile_dynamic`, `project_context`, `ephemeral`, `private_excluded`.

- [ ] **Step 3: Wire capture into daemon chat / tool / browse flow**
  - Ensure daemon-side interactions can generate observation candidates automatically.

- [ ] **Step 4: Write tests**
  - durable personal preference -> stored
  - transient one-off chatter -> low importance / ephemeral
  - explicitly private content -> excluded

**Acceptance criteria:**
- important user facts are captured without manual calls
- trivial chatter is not over-stored
- excluded content never reaches persistent storage

---

## Task 2 — Static vs dynamic profile model (Supermemory pattern)

**Goal:** Turn `master.md` and current identity output into a clearer profile model:
- **static**: stable preferences, identity, recurring constraints
- **dynamic**: current work, recent focus, temporary context
- **open questions**: unresolved items to clarify later

**Files:**
- Modify: `src/mnemon/daemon/identity.py`
- Modify: `src/mnemon/services/memory_service.py`
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/daemon/webui.py`
- Test: `tests/unit/test_profile_model.py`

- [ ] **Step 1: Formalize profile schema**
  - Define a typed structure for `static`, `dynamic`, `questions`, `top_tags`, `sources`.

- [ ] **Step 2: Make markdown identity docs a view, not the source of truth**
  - Keep `master.md` for readability, but derive it from structured data rather than only append-text mutation.

- [ ] **Step 3: Expose a single profile-aware recall endpoint**
  - One daemon/API call should return profile + relevant memories together.

- [ ] **Step 4: Update UI**
  - Show profile changes over time and source citations for profile facts.

**Acceptance criteria:**
- profile data is explicitly separated into stable vs current context
- daemon/UI can fetch a single profile snapshot without manual parsing
- profile facts point back to source memories

---

## Task 3 — Contradiction-aware updates and stale fact handling (Supermemory pattern)

**Goal:** Support "used to / now / current" semantics instead of duplicating conflicting facts forever.

**Files:**
- Modify: `src/mnemon/memory/semantic.py`
- Modify: `src/mnemon/learning/consolidation.py`
- Modify: `src/mnemon/services/memory_service.py`
- Test: new `tests/unit/test_fact_updates.py`

- [ ] **Step 1: Add update semantics**
  - Facts need metadata like `supersedes`, `superseded_by`, `current`, `valid_from`, `valid_to`.

- [ ] **Step 2: Detect contradictions during consolidation**
  - When a new fact conflicts with an existing one in the same category/entity, mark the older fact stale or historical rather than keeping both equally live.

- [ ] **Step 3: Prefer current facts in retrieval**
  - Retrieval should still preserve history but rank current facts first.

- [ ] **Step 4: Write tests**
  - "I use OpenAI" followed by "I now use Anthropic" should preserve history while surfacing current state.

**Acceptance criteria:**
- current fact wins in recall
- old fact remains inspectable as history
- no flat contradiction pile-up in profile output

---

## Task 4 — Scope / project-aware memory (Supermemory pattern)

**Goal:** Separate personal memory from workspace/project memory.

**Files:**
- Modify: `src/mnemon/services/memory_service.py`
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/daemon/cli/app.py`
- Modify: `src/mnemon/daemon/webui.py`
- Test: new `tests/unit/test_memory_scopes.py`

- [ ] **Step 1: Add scope metadata**
  - At minimum: `scope_type`, `scope_id`, `workspace_path`, `repo_name`, `session_id`.

- [ ] **Step 2: Support scoped search**
  - CLI / IPC / web should support `all`, `personal`, and current-workspace scope filters.

- [ ] **Step 3: Auto-infer scope from daemon workspace**
  - When running in or against a repo, tag memories with current workspace identity by default.

- [ ] **Step 4: Add tests**
  - same phrase in two scopes should retrieve only the scoped result when requested.

**Acceptance criteria:**
- personal and project memory can be separated cleanly
- current workspace search is narrower than global search
- UI exposes scope clearly

---

## Task 5 — Citation-first recall (Claude-Mem pattern)

**Goal:** Every important recallable result should be traceable to source observations/memories.

**Files:**
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/daemon/webui.py`
- Modify: `src/mnemon/daemon/cli/app.py`
- Test: extend `tests/unit/test_ipc_improve.py`

- [ ] **Step 1: Standardize IDs in all recall outputs**
- [ ] **Step 2: Ensure profile facts include source IDs**
- [ ] **Step 3: Add copy/open/cite affordances to UI and CLI**
- [ ] **Step 4: Make assistant replies optionally emit citations in verbose mode**

**Acceptance criteria:**
- profile and recall items always have inspectable IDs
- operator can trace claims to source memory records
- citation mode is available without cluttering default chat replies

---

# Phase B — Hybrid Recall, Privacy, Compression

## Task 6 — Hybrid recall (Supermemory + Mnemon workspace)

**Goal:** Combine:
- profile facts
- episodic memories
- goals
- recent files / workspace context
- optional semantic facts

into one ranked operator recall flow.

**Files:**
- Modify: `src/mnemon/services/memory_service.py`
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/daemon/tools/workspace.py`
- Modify: `src/mnemon/daemon/webui.py`

- [ ] **Step 1: Define hybrid result schema**
- [ ] **Step 2: Merge profile + episodic + goals + workspace snippets**
- [ ] **Step 3: Rank and label each source type**
- [ ] **Step 4: Add recall mode selection (`memory-only`, `workspace-only`, `hybrid`)**

**Acceptance criteria:**
- one query can retrieve both memory and current workspace relevance
- results are labeled by origin
- hybrid mode improves operator usefulness without hiding provenance

---

## Task 7 — Privacy / exclusion controls (Claude-Mem pattern)

**Goal:** Users need explicit control over what Mnemon must never persist.

**Files:**
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/daemon/webui.py`
- Modify: `src/mnemon/daemon/cli/app.py`
- Possibly add: `src/mnemon/daemon/privacy.py`
- Test: new `tests/unit/test_privacy_rules.py`

- [ ] **Step 1: Add exclusion rules**
  - explicit command, tag, or pattern-based rules
- [ ] **Step 2: Add redaction support**
  - replace sensitive spans before persistence
- [ ] **Step 3: Add web/CLI settings UI**
- [ ] **Step 4: Add tests proving excluded content never persists**

**Acceptance criteria:**
- users can mark content/rules as private
- excluded content is not stored in docs, vectors, or profile summaries
- UI clearly communicates exclusion behavior

---

## Task 8 — Memory compression and durable summaries (Claude-Mem pattern)

**Goal:** Compress repetitive episodic traces into durable summaries without losing citations.

**Files:**
- Modify: `src/mnemon/learning/consolidation.py`
- Modify: `src/mnemon/memory/episodic.py`
- Modify: `src/mnemon/daemon/ipc.py`
- Test: new `tests/unit/test_memory_compression.py`

- [ ] **Step 1: Group repetitive / related episodes**
- [ ] **Step 2: Produce summary nodes with source links**
- [ ] **Step 3: Surface compressed summaries before raw details in some retrieval modes**
- [ ] **Step 4: Preserve drill-down to raw sources**

**Acceptance criteria:**
- repeated patterns become concise summaries
- summaries keep links to underlying source memories
- progressive disclosure still works: summary first, details later

---

# Phase C — Optional Scenario / Report Layer (MiroFish-inspired, but smaller)

## Task 9 — Scenario sandbox for operator what-if analysis

**Goal:** Support questions like:
- "What happens if I prioritize repo A over repo B this week?"
- "What are likely consequences if I ignore this goal cluster?"

**Important:** This must be a bounded report feature, **not** a default giant swarm engine.

**Files:**
- New: `src/mnemon/daemon/scenario.py`
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/daemon/webui.py`
- Test: new `tests/unit/test_scenario_reports.py`

- [ ] **Step 1: Define scenario input schema**
- [ ] **Step 2: Reuse profile + goals + recent memory as input state**
- [ ] **Step 3: Generate report-style output with assumptions and risks**
- [ ] **Step 4: Add source citations and explicit uncertainty language**

**Acceptance criteria:**
- scenario mode is optional and bounded
- reports are grounded in Mnemon memory/goals, not freeform hallucination
- output includes assumptions, uncertainty, and cited supporting memories

---

## Task 10 — Weekly / project report agent

**Goal:** Produce useful summaries like:
- weekly memory brief
- current project brief
- unresolved questions brief

**Files:**
- New: `src/mnemon/daemon/reports.py`
- Modify: `src/mnemon/daemon/ipc.py`
- Modify: `src/mnemon/daemon/webui.py`

- [ ] **Step 1: Add report templates**
- [ ] **Step 2: Pull from profile, goals, and recent memory summaries**
- [ ] **Step 3: Add UI/CLI entrypoints**
- [ ] **Step 4: Test deterministic structure and source coverage**

**Acceptance criteria:**
- reports are concise, useful, and sourced
- reports help the operator understand "what matters now"

---

# Rollout order

## Recommended sequence
1. **Automatic capture policy**
2. **Static/dynamic profile model**
3. **Contradiction-aware updates**
4. **Scope / project-aware memory**
5. **Citation-first recall**
6. **Hybrid recall**
7. **Privacy / exclusion rules**
8. **Compression / durable summaries**
9. **Scenario sandbox**
10. **Report agent**

This order keeps Mnemon focused on its strongest use case first: persistent personal memory + operator usefulness.

---

# Risks and mitigations

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Over-storing noise | Bad memories poison recall | Capture policy + durable/ephemeral classes |
| Contradiction sprawl | Profile becomes untrustworthy | Explicit supersession model |
| Privacy leakage | High trust failure | Exclusion/redaction before persistence |
| Scope confusion | Wrong workspace memories retrieved | Strong scope metadata + explicit UI filters |
| Overbuilding simulation | Product drifts from core value | Keep scenario layer optional and late |

---

# Verification strategy

## Unit tests
- capture classification
- profile update semantics
- contradiction resolution
- scoped recall filtering
- privacy exclusion/redaction
- compression and citation integrity

## Integration tests
- daemon chat -> automatic memory capture -> profile update -> recall
- workspace scope -> hybrid recall -> UI/API rendering
- privacy rules -> no persistence in memory/profile outputs

## UI checks
- profile provenance visible
- scope visible
- citations inspectable
- privacy settings understandable
- progressive disclosure stays simple: search -> nearby -> details

---

# Success criteria

Mnemon should feel measurably closer to a real persistent assistant when this roadmap lands:
- it remembers stable personal facts correctly
- it distinguishes current work from long-term identity
- it updates stale facts instead of piling up contradictions
- it separates personal and workspace memory cleanly
- it can explain why it "knows" something via source IDs
- it respects privacy controls
- it can compress repetitive history into useful summaries
- only later, optionally, it can generate grounded scenario/report outputs

