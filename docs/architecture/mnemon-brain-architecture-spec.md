# Mnemon Brain Architecture Spec

## Purpose

This document defines the target architecture for Mnemon as a brain-like memory daemon for LLMs.

The goal is not to imitate the human brain literally. The goal is to reproduce the **useful cognitive properties** of a brain-like system:
- selective attention
- multi-store memory
- consolidation
- forgetting
- contradiction handling
- self/world/task modeling
- action feedback
- transparent reasoning and provenance

Mnemon should feel less like “an app with many memory features” and more like one coherent cognitive system.

---

## Product Principle

Mnemon should be built as:

**A persistent cognitive runtime for agents and operators**

not as:
- a flat note store
- a generic RAG dashboard
- a feature pile of unrelated panels
- a full social-simulation engine by default

The interface and data model must reflect a single cognitive loop:

**Observe -> Prioritize -> Store -> Consolidate -> Relate -> Retrieve -> Act -> Reflect -> Forget**

---

# 1. Cognitive System Model

## 1.1 Core loop

Every meaningful input should move through the same internal pipeline:

1. **Observe**
   - ingest raw events from chat, files, tools, browser, channels, schedules
2. **Prioritize**
   - decide whether the event is noise, ephemeral, durable, private, urgent, or contradictory
3. **Store**
   - persist into the appropriate memory store(s)
4. **Consolidate**
   - compress repeated traces, extract durable facts, update summaries
5. **Relate**
   - link the event to scopes, goals, summaries, contradictions, and sources
6. **Retrieve**
   - query memory with scope, task, and recency awareness
7. **Act**
   - use the retrieved state to answer, plan, browse, patch, or report
8. **Reflect**
   - evaluate whether the action and retrieval were useful
9. **Forget**
   - decay or archive low-value traces and superseded facts

This loop should be visible both internally and in the UI.

---

## 1.2 Memory store roles

Mnemon should maintain five explicit stores.

### A. Working Memory
Purpose:
- active context for the current cycle
- current user request
- current goals and recently retrieved items

Properties:
- small, capacity-constrained
- summary-capable
- temporary

### B. Episodic Memory
Purpose:
- what happened
- interaction traces
- actions and outcomes
- high-fidelity provenance source

Properties:
- event-level
- timestamped
- scope-aware
- decays over time
- may be summarized or archived

### C. Semantic Memory
Purpose:
- what is currently believed to be true
- stable facts and relationships
- concept graph

Properties:
- structured
- contradiction-aware
- current-vs-historical distinction
- ideal for profile and world modeling

### D. Procedural Memory
Purpose:
- how to do things
- reusable action recipes and successful patterns

Properties:
- utility-scored
- updated from repeated successful action traces

### E. Profile / Self Model
Purpose:
- stable user preferences
- current work themes
- open questions
- assistant identity

Properties:
- structured JSON source of truth
- rendered to markdown for human readability
- backed by episodic provenance

---

# 2. Data Model

## 2.1 Observation

A new internal normalized event object should exist conceptually even if implemented incrementally.

```python
Observation {
  id: UUID
  source: "chat" | "browser" | "workspace" | "telegram" | "cron" | ...
  raw_text: str
  normalized_text: str
  timestamp: datetime
  scope_type: "personal" | "workspace" | "channel" | "global"
  scope_id: str
  workspace_path: str | None
  repo_name: str | None
  salience: float
  privacy_state: "public" | "redacted" | "excluded"
  tags: list[str]
}
```

This object is the right abstraction boundary between ingestion and memory storage.

---

## 2.2 Episodic record

Current `Episode` is close, but should continue evolving to include:

```python
Episode {
  id: UUID
  context: str
  action: str
  outcome: str
  timestamp: datetime
  importance: float
  scope_type: str
  scope_id: str
  workspace_path: str | None
  repo_name: str | None

  # provenance / graph fields
  source_episode_ids: list[UUID]
  summary_kind: str | None
  summary_of_count: int

  # contradiction / lifecycle
  current: bool
  supersedes: list[UUID]
  superseded_by: UUID | None
  lifecycle_state: "raw" | "consolidated" | "summary" | "archived" | "forgotten"
}
```

Notes:
- episodic memory remains the canonical provenance source
- summary episodes are still episodes, but flagged as summaries
- lifecycle state should become more explicit over time

---

## 2.3 Semantic fact

Semantic facts should explicitly support truth evolution.

```python
SemanticFact {
  id: UUID
  subject: EntityRef
  predicate: str
  object: EntityRef | str
  confidence: float

  current: bool
  valid_from: datetime | None
  valid_to: datetime | None
  supersedes: list[UUID]
  superseded_by: UUID | None
  contradiction_group: str | None

  source_episode_ids: list[UUID]
  scope_type: str
  scope_id: str
}
```

This allows Mnemon to distinguish:
- current truth
- historical truth
- contradictory claims
- low-confidence inferences

---

## 2.4 Profile fact

Profile facts are derived, not free-floating.

```python
ProfileFact {
  text: str
  section: str
  source_ids: list[str]
  updated_at: str
  citations: list[str]
  current: bool
}
```

Profile sections:
- Who Is My Master
- What Drives Them
- What They're Working On
- Patterns I've Noticed
- Questions I Want to Ask Them

---

# 3. Memory Lifecycle States

Every observation should move through explicit lifecycle states.

## 3.1 States

### Ingested
Observed, normalized, not yet classified.

### Ephemeral
Low-signal interaction; may influence the current turn but not persist long-term.

### Durable
Worth storing as episodic memory.

### Consolidated
Information has been extracted into semantic/profile/summary layers.

### Summarized
Represented by a higher-order compressed trace while source traces remain available.

### Historical
No longer current but still useful for traceability.

### Archived
Low-frequency, low-priority, still inspectable.

### Forgotten
No longer retrievable in normal workflows.

### Excluded
Never persisted due to privacy rules.

---

## 3.2 Lifecycle transitions

Typical flow:

`ingested -> durable -> consolidated -> summarized -> historical -> archived -> forgotten`

Privacy flow:

`ingested -> excluded`

Contradiction flow:

`durable/current -> historical when superseded by a newer current fact`

---

# 4. Graph Schema

Mnemon’s graph should explain memory, not just decorate it.

## 4.1 Node types

### Memory nodes
- raw episodic traces
- summary episodic traces

### Scope nodes
- personal
- workspace/repo
- channel/session clusters

### Goal nodes
- active goals
- blocked goals
- completed goals

### Fact nodes
- semantic truths / current beliefs

### Profile nodes
- profile facts or grouped profile sections

### Report nodes (optional)
- generated weekly/project reports

---

## 4.2 Edge types

Minimum useful edges:

- `stored_in` — memory -> scope
- `summarizes` — summary memory -> source memory
- `supports` — memory -> fact/profile/goal
- `contradicts` — fact -> fact
- `supersedes` — newer fact -> older fact
- `related_to` — memory <-> memory (shared tags/entity/topic)
- `goal_relevant_to` — memory/fact -> goal
- `derived_from` — report -> memory/fact/goal

The graph should answer:
- where does this memory live?
- what summary contains it?
- what current fact did it produce?
- what goal is it relevant to?
- what older fact did it replace?

---

## 4.3 Graph UI principles

The graph must not become a second product inside the product.

It should:
- default to a clean, bounded subgraph
- allow zoom/pan/reset
- open details on node click
- prioritize readability over visual density
- expose storage and relationship meaning clearly

It should not:
- show every possible node at once
- default to visual chaos
- require users to interpret raw graph theory to understand memory

---

# 5. Retrieval Architecture

## 5.1 Retrieval modes

Mnemon should support three explicit recall modes.

### Memory
- profile + episodic memory only

### Hybrid
- profile + episodic + goals + workspace snippets

### Graph-aware
- hybrid recall plus graph relationships / summaries / linked sources

---

## 5.2 Retrieval ranking inputs

Every recall path should consider:
- semantic similarity
- recency
- importance
- scope match
- goal relevance
- current-vs-historical status
- summary priority

### Default retrieval order
1. current summary / compressed memory
2. current directly relevant memory
3. current supporting fact
4. historical/superseded items

This mirrors useful human recall better than showing raw oldest/newest logs first.

---

## 5.3 Progressive disclosure

Default pattern:

1. compact result
2. relationship context
3. full detail

UI analogy:
- card list / graph node
- linked sources / nearby / storage
- full drawer

This should remain the interaction model across memory surfaces.

---

# 6. Action and Reflection Loop

## 6.1 Every action should emit memory

Every meaningful assistant/tool action should feed back into memory with:
- what was attempted
- what evidence was used
- what happened
- whether it worked

## 6.2 Reflection objects

Mnemon should accumulate reflective metadata such as:
- retrieval helped / did not help
- confidence was justified / not justified
- this result should become procedural knowledge
- this memory was stale or contradictory

Over time this lets Mnemon learn not just facts, but **how memory helps action**.

---

# 7. Forgetting Strategy

Human-like memory requires forgetting.

## 7.1 What should be forgotten first
- trivial acknowledgements
- stale workspace traces from dead scopes
- old superseded facts with no current relevance
- duplicated near-identical traces already summarized

## 7.2 What should almost never be forgotten
- strong user preferences
- identity facts
- current goals and durable project summaries
- provenance behind active beliefs

## 7.3 Forgetting modes
- decay ranking only
- archive from primary recall
- hard-delete excluded/private content

---

# 8. UI Architecture

## 8.1 Primary product surfaces

Mnemon should revolve around three surfaces:

### A. Memory
The home surface.
Must answer:
- what does Mnemon know?
- where is it stored?
- what changed recently?
- what has been summarized?

### B. Chat
The action surface.
Must answer:
- what did Jarvis use to answer?
- what was cited?
- what should be remembered?

### C. Graph
The relationship surface.
Must answer:
- how are memories connected?
- what summary compresses what?
- what scope contains what?
- what supports what?

These three surfaces should feel like one system, not separate apps.

---

## 8.2 Secondary surfaces

These are secondary/supporting, not equal peers:
- Goals
- Timeline
- Reports
- Inbox / Control
- Utilities (logs, thoughts, files, settings)

These should be visually quieter and mentally secondary.

---

## 8.3 UI design principles

### Principle 1 — reveal the memory lifecycle
The UI should make it clear whether a thing is:
- raw
- current
- summarized
- historical
- linked
- scoped

### Principle 2 — reduce panel competition
The app should not feel like 9 tabs fighting for attention.

### Principle 3 — every screen should explain provenance
Claims and summaries should always lead back to source memories.

### Principle 4 — graph should clarify, not overwhelm
The graph exists to simplify understanding of relationships.

### Principle 5 — one visual language
The app should feel editorial, calm, and human-readable.

---

# 9. Recommended Implementation Order

## Short-term
1. unify Memory / Chat / Graph as the main product path
2. make graph edges richer (`supports`, `contradicts`, `goal_relevant_to`)
3. expose memory lifecycle state in payloads and UI
4. add real forgetting/archival states

## Medium-term
5. semantic contradiction handling beyond profile heuristics
6. reflection scoring for retrieval usefulness
7. report tab / report history

## Long-term
8. embodied/environmental observation layer
9. richer world model and plan simulation
10. stronger procedural memory feedback loop

---

# 10. What is stopping us today

The core blockers are not lack of features. The blockers are:

1. **memory lifecycle visibility is incomplete**
2. **semantic contradiction handling is still partial**
3. **forgetting is still weak**
4. **retrieval usefulness is not yet evaluated as a first-class signal**
5. **UI still exposes implementation slices more than cognitive flow**
6. **graph semantics are still minimal**
7. **embodiment is still shallow (mostly text and workspace events)**

That means Mnemon is already partway to a brain-like runtime, but it still needs stronger integration across perception, memory, action, and reflection.

---

# 11. Definition of Success

Mnemon feels brain-like when:
- the user can see what it knows, where it came from, and what changed
- the assistant recalls current truths before stale ones
- repeated experience becomes compressed summaries
- goals, memory, and action feel integrated
- the graph clarifies relationships instead of confusing them
- privacy and forgetting are trusted
- the UI feels like one coherent cognitive product

That is the target.
