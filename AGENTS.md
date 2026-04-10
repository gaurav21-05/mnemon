# Mnemon — Agent Operating Manual

You are working on **Mnemon**, a brain-inspired cognitive memory framework for AI agents, plus an always-on Jarvis-style daemon. This file is your complete operating contract.

---

## 1. What Mnemon Is

Mnemon is two things in one repo:

| Layer | Entry point | Purpose |
|---|---|---|
| **Core framework** (`src/mnemon/`) | `mnemon` CLI / Python API | 6-subsystem cognitive memory engine for embedding into any AI agent |
| **Daemon layer** (`src/mnemon/daemon/`) | `mnemon-daemon` CLI | Always-on Jarvis process: idle thinking, goal persistence, IPC, autonomy control |

The daemon wraps the brain. The brain is the library. Changes to core modules must consider daemon integration.

---

## 2. Project Structure

```
src/mnemon/
├── core/           # Config, models, interfaces, event bus, exceptions, registry
├── memory/         # 6 subsystems: episodic, semantic, procedural, working, sensory, valence
├── control/        # Attention, goals, metacognition, orchestrator (cognitive cycle)
├── backends/       # Pluggable stores: memory, sqlite, qdrant, hnswlib, falkordb, igraph
├── learning/       # Consolidation, replay, reward, skill acquisition, bootstrap
├── providers/      # LLM abstraction (LiteLLM)
├── scheduling/     # APScheduler-backed job runner
├── services/       # MCP server bridge, memory service
├── evaluation/     # Benchmarks and evaluation suite
├── daemon/
│   ├── loop.py             # IdleThinkingLoop — Default Mode Network
│   ├── ipc.py              # Unix socket JSON-RPC server
│   ├── lifecycle.py        # OS daemon: fork, PID file, signals, auto-restart
│   ├── autonomy.py         # Permission gating: PASSIVE/SUGGEST/SEMI_AUTO/AUTONOMOUS
│   ├── goals/              # PersistentGoalStore — JSON-backed across restarts
│   ├── observers/          # FileSystem, Cron, Web observers → SensoryBuffer
│   ├── channels/           # Telegram integration
│   ├── tools/              # Workspace tools: file inspect, patch, git, worktrees
│   ├── cli/                # app.py (mnemon-daemon), client.py
│   ├── improve.py          # Supervised self-improvement workflow
│   ├── scenario.py         # Counterfactual simulation
│   ├── identity.py         # Agent identity and persona
│   ├── privacy.py          # Privacy controls
│   ├── reports.py          # Report generation
│   ├── state.py            # Daemon runtime state
│   └── webui.py            # Web dashboard
├── factory.py      # MnemonFactory / DaemonFactory — primary construction path
├── cli.py          # mnemon entrypoint
└── __main__.py
```

---

## 3. Core Vocabulary

Always use these names consistently. Never invent synonyms.

| Brain analog | Mnemon name | Model/class | Location |
|---|---|---|---|
| Sensory cortex | Sensory buffer | `PerceptUnit`, `SensoryBuffer` | `memory/sensory.py` |
| Hippocampus | Episodic memory | `Episode` | `memory/episodic.py` |
| Neocortex | Semantic memory | `SemanticTriple`, `Entity`, `Community` | `memory/semantic.py` |
| Basal ganglia / cerebellum | Procedural memory | `Skill` | `memory/procedural.py` |
| Prefrontal cortex | Working memory | `WorkingMemoryState`, `Goal` | `memory/working.py` |
| Amygdala | Valence memory | `ValenceAssociation` | `memory/valence.py` |
| Global Workspace | Cognitive bus | `CognitiveMessage` | `core/bus.py` |
| ACC / metacognition | Metacognition | `MetaEvaluation`, `Strategy` | `control/metacognition.py` |
| Default Mode Network | Idle thinking loop | `IdleThinkingLoop` | `daemon/loop.py` |

### Key enums (never use raw strings where these exist)
- `Modality` — text, image, audio, structured_data, tool_output
- `MessageType` — percept, retrieval_cue, retrieval_result, action_candidate, action_selected, reward_signal, consolidation_trigger, meta_signal, goal_update, attention_gate, broadcast
- `GoalStatus` — active, suspended, completed, failed, blocked
- `ConsolidationState` — raw, processing, failed, consolidated, archived
- `MemoryLifecycleState` — ingested, durable, consolidated, summary, historical, archived, forgotten, excluded
- `SkillType` — code, prompt_template, workflow_dag, tool_sequence
- `EvictionPolicy` — lru, lru_importance, custom

---

## 4. Architecture Rules

### Async-first
- All I/O uses `anyio` / `asyncio`. Never use `time.sleep()` — use `await anyio.sleep()`.
- All store operations and LLM calls are async. Keep sync wrappers out of core paths.

### Pydantic everywhere
- All data crossing module boundaries must be a Pydantic `BaseModel`. No raw dicts.
- Frozen models (`model_config = {"frozen": True}`) for immutable messages and results.
- Use `Field(ge=0.0, le=1.0)` for bounded floats (confidence, importance, score).

### Interface-based backends
- Every storage backend implements the interface in `core/interfaces.py`.
- Never import a concrete backend directly from core or memory modules — route through the registry or factory.
- Backend extras are optional. Guard imports with `try/except ImportError`.

### Event bus for inter-module communication
- Modules communicate via `CognitiveMessage` on the bus in `core/bus.py`.
- Direct module-to-module calls are only allowed within the same subsystem.
- Cross-subsystem effects go through `MessageType` events.

### Factory pattern
- Construct the brain via `MnemonFactory`, the daemon via `DaemonFactory`.
- Never instantiate subsystems directly in user-facing code.

---

## 5. The 6-Phase Cognitive Cycle

The orchestrator (`control/orchestrator.py`) runs every request through:

```
1. Perception   → SensoryBuffer normalises and embeds input → PerceptUnit
2. Attention    → AttentionGate scores salience → broadcast / queue / discard
3. Retrieval    → Query episodic, semantic, procedural stores → RetrievedItem[]
4. Deliberation → LLM reasons over working memory + retrieved context
5. Execution    → Selected action runs; tool results feed back as percepts
6. Learning     → Reward signal → RPE → update episodic importance + valence
```

Consolidation runs **out-of-band** (idle or scheduled), not inline. The `IdleThinkingLoop` triggers consolidation during daemon downtime.

---

## 6. The Daemon and Its Autonomy Levels

The daemon (`daemon/`) runs as an OS process. It is the user-facing application.

| Level | Constant | Behaviour |
|---|---|---|
| 0 | `PASSIVE` | Observe and record only. No unsolicited actions. |
| 1 | `SUGGEST` | Surface insights and contradictions. Ask before acting. |
| 2 | `SEMI_AUTO` | Auto-resolve low-confidence contradictions. Flag high-risk. |
| 3 | `AUTONOMOUS` | Full self-managing memory. Runs consolidation without asking. |

**Always gate destructive or irreversible actions on autonomy level.** Never take `AUTONOMOUS`-level actions when the daemon is in `PASSIVE` or `SUGGEST` mode.

The `IdleThinkingLoop` phases (Default Mode Network):
1. **Consolidation** — transfer raw episodes → semantic triples
2. **Reflection** — review recent cycles, detect errors
3. **Planning** — update goal stack based on new knowledge
4. **Exploration** — mine memories for novel patterns / insights

---

## 7. What Makes Mnemon Different (Build Toward This)

These are the strategic differentiators. Prioritise features that land here.

### 7.1 Confident Memory — Evidence Chains on Every Fact
Every `SemanticTriple` already has `confidence` and `source_episodes`. The next step: surface this to the user. When answering "why do you believe X?", trace `source_episodes → context/action/outcome` and present the chain. **No other memory system does this.**

### 7.2 Memory Decay — Ebbinghaus Forgetting Curve
`Episode` already has `decay_lambda` and `base_strength`. Wire this:
- `strength(t) = base_strength * exp(-decay_lambda * days_since_last_access)`
- Retrieval resets `last_accessed` and bumps strength (spaced repetition effect)
- Idle loop prunes episodes below a configurable threshold
- `SemanticTriple.confidence` degrades without `last_confirmed` updates

### 7.3 Proactive Insight Surfacing (The Jarvis Moment)
The `IdleThinkingLoop` should **push** insights, not just consolidate:
- Contradictions between triples → flag for resolution or user notification
- Repeated patterns in episodes → candidate for a new Skill
- Stalled goals → proactive suggestions
- Cross-session connections ("you solved this before in project X")

### 7.4 Goal-Anchored Memory Lifecycle
`Episode.goal_id` exists. Wire it:
- Episodes linked to a completed goal → lifecycle moves to `SUMMARY` faster
- Episodes linked to a dropped goal → decay accelerates
- `PersistentGoalStore` informs consolidation priority

### 7.5 Causal Memory Graph
Future: add `caused_by: UUID | None` and `led_to: list[UUID]` to `Episode`. Enable queries like "what chain of events led to this outcome?" The idle loop should mine causal chains from sequential episodes.

### 7.6 Universal Memory Protocol (MCP Bridge)
`services/mcp_contract.py` exists. Keep it compatible with Claude Code, Cursor, Continue.dev. Mnemon's memory should be queryable by any MCP-compatible agent.

---

## 8. Development Workflow

### Setup
```bash
cd ~/mnemon
uv sync --extra all          # install all optional deps
source .venv/bin/activate
```

### Run tests
```bash
pytest                       # all tests
pytest tests/unit/           # unit only
pytest tests/integration/    # integration (may need running backends)
pytest -x -q                 # stop on first failure, quiet
```

### Lint and typecheck
```bash
ruff check src tests         # lint
ruff format src tests        # format
mypy src                     # strict type checking
```

**Always run lint + typecheck before claiming a change is done.**

### Run the daemon locally
```bash
mnemon-daemon start          # start background daemon
mnemon-daemon chat           # interactive chat
mnemon-daemon goals          # list goals
mnemon-daemon thoughts       # show recent idle thoughts
mnemon-daemon stop           # stop daemon
```

### Run the core framework
```bash
mnemon                       # interactive CLI
python -m mnemon             # same
```

---

## 9. Coding Standards

- **Python 3.12+**. Use `from __future__ import annotations` in all files.
- **Line length**: 100 chars (ruff enforces this).
- **Type annotations**: strict mypy. All functions annotated. No `Any` except in Pydantic `payload` fields.
- **No `print()`** in library code — use the logger: `import logging; logger = logging.getLogger(__name__)`.
- **No new dependencies** without explicit request. Optional extras via `pyproject.toml [extras]`.
- **No backwards-compatibility shims**. Delete unused code.
- **No defensive validation** for internal invariants — trust Pydantic and internal contracts. Only validate at system boundaries (user input, IPC, MCP requests).
- **Prefer editing existing files** over creating new ones.
- **Three similar lines > premature abstraction.**

### Naming conventions
- Module names: `snake_case`
- Classes: `PascalCase`
- Async functions: same as sync — no `async_` prefix
- Private methods: `_single_underscore`
- Constants: `UPPER_SNAKE_CASE`

---

## 10. Commit Message Convention (Lore Protocol)

Every commit must communicate *why*, not just *what*. The diff already shows what.

```
<intent line: why this change was made>

<optional body: constraints, approach rationale>

Constraint: <external constraint that shaped the decision>
Rejected: <alternative considered> | <reason rejected>
Confidence: low | medium | high
Scope-risk: narrow | moderate | broad
Directive: <warning for future modifiers>
Tested: <what was verified>
Not-tested: <known gaps>
```

**Example:**
```
Wire decay_lambda into episodic retrieval scoring to enable memory fading

Episodes with high decay_lambda and old last_accessed dates now score lower
in retrieval. This allows the Ebbinghaus forgetting curve to reduce context
noise from stale, unreinforced memories.

Constraint: decay must not drop below min_score threshold to preserve identity memories
Rejected: time-based hard deletion | irreversible, breaks audit trail
Confidence: high
Scope-risk: moderate
Directive: do not increase default decay_lambda above 0.01 without benchmarking retrieval recall
Tested: unit tests for score calculation, manual retrieval smoke test
Not-tested: long-running decay with real Qdrant backend
```

---

## 11. Testing Philosophy

- **Unit tests** go in `tests/unit/`. Mock LLM calls. Use in-memory backends.
- **Integration tests** go in `tests/integration/`. Use real SQLite. Flag Qdrant/FalkorDB tests with `@pytest.mark.requires_qdrant`.
- **No mocking internal module boundaries** — only mock external I/O (LLM provider, disk).
- `pytest-asyncio` is configured with `asyncio_mode = "auto"` — all async test functions work without decorators.
- Use `hypothesis` for property-based tests on models and scoring functions.

---

## 12. Key Files to Know

| File | What it does |
|---|---|
| `src/mnemon/core/models.py` | Every data model. Read this first. |
| `src/mnemon/core/interfaces.py` | Backend contracts. All stores implement these. |
| `src/mnemon/factory.py` | How to build a brain or daemon. Construction entry point. |
| `src/mnemon/control/orchestrator.py` | The 6-phase cognitive cycle. |
| `src/mnemon/daemon/loop.py` | Default Mode Network idle thinking. |
| `src/mnemon/daemon/ipc.py` | Unix socket RPC — how CLI talks to daemon. |
| `src/mnemon/daemon/autonomy.py` | Autonomy level gating. Check before any autonomous action. |
| `src/mnemon/daemon/goals/persistent_store.py` | Goals survive daemon restarts here. |
| `src/mnemon/services/mcp_contract.py` | MCP tool definitions. Keep these agent-compatible. |
| `pyproject.toml` | All deps, extras, lint config, test config. |

---

## 13. What NOT to Build

- Do not add features the user hasn't asked for.
- Do not add error handling for internal invariants — Pydantic handles that.
- Do not add `print()` statements, temporary debug code, or placeholder TODOs.
- Do not create abstract base classes for one-off use.
- Do not add a new backend unless asked — use the existing pluggable interface.
- Do not break the `mcp_contract.py` MCP tool signatures — other agents depend on them.
- Do not touch `daemon/lifecycle.py` fork/signal logic unless you know what you're doing — breaking this breaks the daemon OS process.

---

## 14. Stopping Criteria

Before claiming any task complete:

1. `ruff check src tests` passes with zero errors
2. `mypy src` passes with zero errors
3. `pytest` passes (or failing tests are explicitly documented with reason)
4. The change does not regress the daemon IPC protocol (check `ipc.py` message formats)
5. Any new public API is covered by at least one test
6. Commit message follows the Lore Protocol in section 10
