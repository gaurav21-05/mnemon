# Mnemon

Mnemon is a brain-like memory framework for AI agents, plus an always-on daemon layer that turns those memory systems into a practical personal assistant. It combines episodic, semantic, procedural, working, sensory, and valence memory with goal management, consolidation, and daemon-side interfaces such as CLI, web UI, and MCP.

## What you get

### Core cognitive framework

- **6 memory subsystems** — episodic, semantic, procedural, working, sensory, and valence
- **6-phase cognitive cycle** — perception, attention, retrieval, deliberation, execution, learning
- **Learning pipeline** — consolidation, prioritized replay, reward prediction error, skill acquisition
- **Goal management** — hierarchical goals with LLM-assisted decomposition
- **Pluggable storage** — in-memory, Qdrant, FalkorDB, Neo4j, SQLite, PostgreSQL
- **LLM provider flexibility** — OpenAI, Anthropic, Ollama, Groq, Mistral, and other LiteLLM-supported models
- **Async-first design** — built around `anyio` / `asyncio`

### Daemon layer

- **Long-running Jarvis-style daemon** with persistent goals and memory-aware chat
- **Unix-socket IPC API** for CLI and external adapters
- **Workspace tools** for bounded file inspection, patching, verification, git diff/status, and worktrees
- **Supervised self-improvement workflow** with analyze → plan → worktree → patch → verify → approve/abort
- **Web UI dashboard** with live thoughts, goal management, chat, log tail, and memory search
- **MCP bridge examples** for both direct memory access and talking to a running daemon

## Installation

### Base install

```bash
pip install mnemon
```

### Install with optional extras

```bash
# Common development / demo extras
pip install "mnemon[mcp,sqlite,scheduler]"

# Everything
pip install "mnemon[all]"
```

## Quick start: core framework

```python
import anyio
from mnemon.factory import MnemonFactory


async def main() -> None:
    brain = await MnemonFactory().build()

    async with brain:
        result = await brain.run_cycle("Hello, I'm learning about AI")
        print(f"Cycle {result['cycle_number']}: {result['phases_completed']}")
        print(f"Retrieved {result['retrieved_count']} memories")

    await brain.close()


anyio.run(main)
```

## Interactive examples

### Agent demo

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python examples/agent.py

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python examples/agent.py --model claude-3-5-haiku-20241022

# Ollama
python examples/agent.py --model ollama/llama3
```

### Memory-focused chatbot demo

```bash
python examples/chatbot_with_memory.py
```

## Daemon quick start

The daemon layer exposes Mnemon as a long-running assistant with CLI, IPC, and web UI surfaces.

### Start the daemon

```bash
python -m mnemon.daemon.cli.app start --foreground
```

In another shell:

```bash
python -m mnemon.daemon.cli.app status
python -m mnemon.daemon.cli.app chat "Remember that I'm working on mnemon"
python -m mnemon.daemon.cli.app goals add "Ship the daemon bridge example"
python -m mnemon.daemon.cli.app thoughts --limit 5
```

### Goal management

```bash
python -m mnemon.daemon.cli.app goals add "Write release notes" --priority 0.8
python -m mnemon.daemon.cli.app goals list
```

### Workspace tooling through the daemon

```bash
python -m mnemon.daemon.cli.app ls src/mnemon/daemon
python -m mnemon.daemon.cli.app read src/mnemon/daemon/ipc.py
python -m mnemon.daemon.cli.app git-status
python -m mnemon.daemon.cli.app verify "python -m pytest tests/unit -q"
```

### Supervised self-improvement

Mnemon can run a guarded self-improvement workflow against its own repo:

1. Analyze current git/test state
2. Ask the LLM for a small structured patch plan
3. Create an isolated git worktree
4. Apply the planned patches
5. Re-run verification inside the worktree
6. Wait for human approval before merge

```bash
# Analysis only
python -m mnemon.daemon.cli.app improve --analyze

# Start a run
python -m mnemon.daemon.cli.app improve "improve code quality and fix any failing tests"

# Poll progress
python -m mnemon.daemon.cli.app improve --status

# Finish or discard once awaiting approval
python -m mnemon.daemon.cli.app improve --approve
python -m mnemon.daemon.cli.app improve --abort
```

## Web UI

Run the dashboard as a standalone process:

```bash
python -m mnemon.daemon.webui
```

Open <http://localhost:7777>.

The UI includes:

- daemon status and uptime
- live idle-thought stream
- active goals panel
- inline goal creation form
- chat composer for Jarvis
- live log tail
- memory search box in the top bar

## MCP integrations

### 1) Direct memory server

`examples/mcp_memory_server.py` exposes Mnemon memory tools directly.

```bash
pip install "mnemon[mcp]"

export MNEMON_MCP_MODEL=ollama/llama3.2
export MNEMON_MCP_EMBED_MODEL=ollama/nomic-embed-text
export MNEMON_MCP_EMBED_DIM=768
export MNEMON_MCP_NAMESPACE=mnemon

python examples/mcp_memory_server.py
```

Exposed tools:

- `mnemon.memory_write(...)`
- `mnemon.memory_retrieve(...)`
- `mnemon.memory_consolidate()`
- `mnemon.memory_state()`
- `mnemon.memory_resources_list()`
- `mnemon.memory_resources_read(uri)`

### 2) Running-daemon MCP bridge

`examples/mcp_daemon_server.py` exposes a *running daemon* as MCP tools over stdio.

Start the daemon first:

```bash
python -m mnemon.daemon.cli.app start --foreground
```

Then start the MCP bridge:

```bash
python examples/mcp_daemon_server.py
```

Exposed daemon tools:

- `daemon_chat(message)`
- `daemon_goals_list()`
- `daemon_goals_add(description, priority)`
- `daemon_memory_search(query, top_k)`

## Architecture

Mnemon implements a 6-phase cognitive cycle:

```text
Input
  -> 1. PERCEPTION   (sensory buffer -> percept)
  -> 2. ATTENTION    (salience scoring, gate: broadcast/queue/discard)
  -> 3. RETRIEVAL    (episodic + semantic + procedural fan-out)
  -> 4. DELIBERATION (assemble memory + goals + state)
  -> 5. EXECUTION    (agent/framework produces response or action)
  -> 6. LEARNING     (encode episode, update reward/valence/meta state)
```

### Module map

| Module | Role |
| --- | --- |
| `src/mnemon/core/` | models, config, interfaces, exceptions, bus, registry |
| `src/mnemon/memory/` | episodic, semantic, procedural, working, sensory, valence |
| `src/mnemon/learning/` | consolidation, replay, reward, skill acquisition |
| `src/mnemon/control/` | orchestrator, attention, goals, metacognition |
| `src/mnemon/backends/` | in-memory and external storage backends |
| `src/mnemon/daemon/` | daemon runtime, IPC, lifecycle, tools, web UI |
| `src/mnemon/services/` | external-facing service adapters |
| `examples/` | runnable demos and MCP examples |

## Configuration

Mnemon uses `MNEMON__` environment variables with double-underscore nesting:

```bash
MNEMON__LLM__DEFAULT_PROVIDER=openai
MNEMON__WORKING_MEMORY__TOKEN_BUDGET=16384
MNEMON__EPISODIC__CAPACITY__MAX_EPISODES=50000
MNEMON__ATTENTION__BROADCAST_THRESHOLD=0.6
```

You can also load a TOML file:

```python
from mnemon.core.config import load_config

config = load_config("/path/to/mnemon.toml")
```

## Development

```bash
# Install editable package + dev deps
pip install -e ".[all]"
pip install pytest pytest-asyncio pytest-benchmark hypothesis ruff mypy

# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Whole suite
pytest tests -v

# Lint + format
ruff check src tests examples
ruff format src tests examples

# Type-check
mypy src/mnemon
```

## Roadmap

**Bugs / correctness**
- [x] Consolidation engine: mark episodes as `FAILED` after N LLM extraction retries
- [x] Semantic store: atomic write between vector index and SQLite `_docs` table

**Self-improvement**
- [x] `daemon/improve.py` — 6-phase supervised workflow (analyze → plan → worktree → patch → verify → approve/abort)
- [x] IPC + CLI integration (`improve.analyze`, `improve.start`, `improve.status`, `improve.approve`, `improve.abort`)
- [ ] Structured planning memory: persist improvement plans across sessions in episodic store

**Interfaces**
- [x] Web UI: goal creation form
- [x] Web UI: memory search widget
- [x] MCP daemon bridge (`examples/mcp_daemon_server.py`)

**Distribution**
- [ ] PyPI release
- [ ] Docker image
- [ ] Homebrew formula (macOS)

## License

Apache-2.0
