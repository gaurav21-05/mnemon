# Mnemon

A brain-like cognitive memory framework for AI agents. Mnemon provides a modular,
bio-inspired memory architecture that mirrors real cognitive systems — episodic,
semantic, procedural, working, sensory, and valence memory — connected via an
event bus and driven by a 6-phase cognitive cycle.

## Features

- **6 Memory Subsystems** — episodic (hippocampus), semantic (neocortex), procedural (basal ganglia), working (dlPFC), sensory (thalamus), valence (amygdala)
- **Learning Pipeline** — consolidation (episodic-to-semantic), prioritized replay, reward prediction error, skill acquisition
- **Cognitive Control** — orchestrator, attention gating, hierarchical goals, metacognition
- **Pluggable Backends** — in-memory (zero-dep), Qdrant, FalkorDB, Neo4j, SQLite, PostgreSQL
- **Any LLM** — 100+ providers via LiteLLM (OpenAI, Anthropic, Ollama, Groq, Mistral, etc.)
- **Async-First** — full asyncio support via anyio

## Quick Start

### Installation

```bash
pip install mnemon

# Or with optional backends
pip install "mnemon[qdrant,sqlite,scheduler]"
```

### Basic Usage

```python
import anyio
from mnemon.factory import MnemonFactory

async def main():
    brain = await MnemonFactory().build()

    async with brain:
        result = await brain.run_cycle("Hello, I'm learning about AI")
        print(f"Cycle {result['cycle_number']}: {result['phases_completed']}")
        print(f"Retrieved {result['retrieved_count']} memories")

    await brain.close()

anyio.run(main)
```

### Interactive Agent Example

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python examples/agent.py

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python examples/agent.py --model claude-3-5-haiku-20241022

# Ollama (local, no API key needed)
python examples/agent.py --model ollama/llama3

# Custom settings
python examples/agent.py --model gpt-4o --temperature 0.5 --debug
```

The agent provides a REPL with commands: `/memories`, `/facts`, `/skills`,
`/state`, `/consolidate`, `/goals`, `/help`, `/quit`.

### MCP Memory Server (for other agents)

Expose Mnemon as an MCP tool server so any MCP-compatible agent can store and
retrieve long-term memory.

```bash
# Install optional MCP dependency
pip install "mnemon[mcp]"

# Optional: set model for consolidation (if omitted, consolidation is disabled)
export MNEMON_MCP_MODEL=ollama/llama3.2
export MNEMON_MCP_EMBED_MODEL=ollama/nomic-embed-text
export MNEMON_MCP_EMBED_DIM=768
export MNEMON_MCP_NAMESPACE=mnemon

# Run server
python examples/mcp_memory_server.py
```

Exposed MCP tools:
- `mnemon.memory_write(content, agent_id, session_id, tags, importance)`
- `mnemon.memory_retrieve(query, top_k, min_score)`
- `mnemon.memory_consolidate()`
- `mnemon.memory_state()`
- `mnemon.memory_resources_list()`
- `mnemon.memory_resources_read(uri)`

MCP resources (when client/server supports resource APIs):
- `memory://mnemon/state`
- `memory://mnemon/episodes/recent`

## Architecture

Mnemon implements a **6-phase cognitive cycle**:

```
Input ─> 1. PERCEPTION  (sensory buffer → PerceptUnit)
      ─> 2. ATTENTION   (salience scoring, gate: broadcast/queue/discard)
      ─> 3. RETRIEVAL   (cue-driven fan-out → episodic + semantic + procedural)
      ─> 4. DELIBERATION (assemble context + goals for action selection)
      ─> 5. EXECUTION   (agent framework consumes context, generates response)
      ─> 6. LEARNING    (encode episode, compute RPE, update valence, meta-eval)
```

### Module Map

| Module | Brain Analog | Role |
|--------|-------------|------|
| SensoryBuffer | Thalamus | Pre-attentive input processing with TTL |
| WorkingMemory | dlPFC | Token-budget-constrained active context |
| EpisodicMemory | Hippocampus | One-shot episode encoding, pattern completion |
| SemanticMemory | Neocortex | Knowledge graph with spreading activation |
| ProceduralMemory | Basal ganglia | Skill library with utility learning |
| ValenceMemory | Amygdala | Emotional associations, rapid appraisal |
| ConsolidationEngine | Sleep replay | Episodic → semantic fact extraction |
| AttentionController | Basal forebrain | Selective gating, adaptive thresholds |
| GoalManager | Anterior PFC | Hierarchical goals, LLM decomposition |
| MetaCognition | ACC | Error monitoring, strategy selection |
| CognitiveBus | Thalamic relay | Inter-module event routing |
| Orchestrator | Lateral PFC | Central executive, cycle coordination |

## Configuration

### Environment Variables

All settings use the `MNEMON__` prefix with double-underscore nesting:

```bash
MNEMON__LLM__DEFAULT_PROVIDER=openai
MNEMON__WORKING_MEMORY__TOKEN_BUDGET=16384
MNEMON__EPISODIC__CAPACITY__MAX_EPISODES=50000
MNEMON__ATTENTION__BROADCAST_THRESHOLD=0.6
```

### TOML File

```python
from mnemon.core.config import load_config

config = load_config("/path/to/mnemon.toml")
brain = await MnemonFactory(config).build()
```

### Programmatic

```python
from mnemon.core.config import MnemonConfig

config = MnemonConfig()
config.working_memory.token_budget = 16384
config.llm.default_provider = "anthropic"
config.llm.providers["anthropic"] = {
    "model": "claude-3-5-haiku-20241022",
    "embedding_model": "text-embedding-3-small",
}
```

## LLM Providers

Mnemon uses [LiteLLM](https://docs.litellm.ai/) for provider abstraction:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Groq
export GROQ_API_KEY=gsk_...

# Ollama (local)
# No key needed, just run: ollama serve
```

## Development

```bash
# Install dev dependencies
pip install -e ".[all]"
pip install pytest pytest-asyncio pytest-benchmark hypothesis ruff mypy

# Run tests
pytest tests/unit/ -v           # 199 unit tests
pytest tests/integration/ -v    # 17 integration tests
pytest tests/ -v                # All tests

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/mnemon/
```

## Project Structure

```
src/mnemon/
    core/           # Models, config, interfaces, exceptions, event bus, registry
    memory/         # Episodic, semantic, procedural, working, sensory, valence
    backends/       # In-memory, Qdrant, SQLite, FalkorDB storage implementations
    learning/       # Consolidation, replay, reward, skill acquisition
    control/        # Orchestrator, attention, goals, metacognition
    evaluation/     # Benchmark suite (retrieval, forgetting, calibration, latency)
    providers/      # LiteLLM wrapper for LLM and embedding providers
    scheduling/     # APScheduler-based consolidation scheduling
    factory.py      # One-shot builder that wires the entire framework
```

## License

Apache-2.0
