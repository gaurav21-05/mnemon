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
- **Natural-language tool use** — Jarvis can browse the web and inspect/edit the workspace from normal chat, not only slash commands
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

### Simple interactive CLI

This is the easiest way to boot the Mnemon memory system and start chatting with it.

```bash
# Installed entrypoint
mnemon

# Start the daemon-oriented experience in one command
mnemon start

# First-time local setup (creates ~/.local/bin launchers)
./scripts/install-local-cli.sh

# Or module mode
python3 -m mnemon

# Local Ollama mode
mnemon --local
```

Behavior:

- `mnemon` checks your provider setup before booting
- if no cloud API key is available, it tries local Ollama mode automatically
- if Ollama is installed but not running, it attempts to start `ollama serve`
- if Ollama local models are missing, it attempts to pull them with visible progress
- if required local models are missing, it tells you exactly what to pull
- `mnemon --doctor` prints a setup diagnosis without starting the system

Useful startup details shown in the CLI:

- active chat model
- active embedding model
- Web UI command
- Web UI URL (`http://localhost:7777`)
- LAN URL for other devices on your network
- full daemon command
- doctor command
- quick-action footer inside the interactive session

Inside the CLI, use:

- `/memories`
- `/facts`
- `/skills`
- `/state`
- `/consolidate`
- `/goals`
- `/help`
- `/quit`

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
mnemon start

# or directly
mnemon-daemon start --foreground
```

In another shell:

```bash
mnemon-daemon status
mnemon-daemon chat "Remember that I'm working on mnemon"
mnemon-daemon goals add "Ship the daemon bridge example"
mnemon-daemon thoughts --limit 5
```

By default Jarvis now runs in `semi_auto` mode:

- low-risk tool actions like browse, list, and read execute automatically
- medium-risk workspace actions like write, patch, and worktree creation can also execute from normal chat
- high-risk shell execution still depends on the daemon autonomy level

Examples of plain-language tool use:

```bash
mnemon-daemon chat "research browser-use and summarize the current API"
mnemon-daemon chat "read src/mnemon/daemon/ipc.py"
mnemon-daemon chat "write a LICENSE file in Apache 2.0 format"
mnemon-daemon chat "patch README.md to add a quickstart section"
```

### Goal management

```bash
mnemon-daemon goals add "Write release notes" --priority 0.8
mnemon-daemon goals list
```

### Workspace tooling through the daemon

```bash
mnemon-daemon ls src/mnemon/daemon
mnemon-daemon read src/mnemon/daemon/ipc.py
mnemon-daemon git-status
mnemon-daemon verify "python -m pytest tests/unit -q"
```

### Memory profile + progressive recall

Inspired by tools like Supermemory and Claude-Mem, Mnemon now exposes a compact
profile-and-recall workflow through the daemon:

```bash
# Compact indexed search with IDs, timestamps, tags, and profile hints
mnemon-daemon memory search "what am I currently working on?"

# Expand one or more exact memory hits
mnemon-daemon memory get <episode-id>

# View the structured user profile Mnemon inferred over time
mnemon-daemon memory profile

# Inspect nearby memories around one anchor event
mnemon-daemon memory timeline <episode-id>
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
mnemon-daemon improve --analyze

# Start a run
mnemon-daemon improve "improve code quality and fix any failing tests"

# Poll progress
mnemon-daemon improve --status

# Finish or discard once awaiting approval
mnemon-daemon improve --approve
mnemon-daemon improve --abort
```

## Web UI

Run the dashboard as a standalone process:

```bash
mnemon-daemon webui
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
mnemon-daemon start --foreground
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
# Install project and dev tooling
uv sync --dev

# Unit tests
uv run pytest tests/unit -v

# Integration tests
uv run pytest tests/integration -v

# Whole suite
uv run pytest tests -v l

# Lint + format
uv run ruff check src tests examples
uv run ruff format src tests examples

# Type-check
uv run mypy src/mnemon
```

## License

[Apache-2.0](./LICENSE)
