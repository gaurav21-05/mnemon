#!/usr/bin/env python3
"""
Mnemon live demo — conversational agent with brain-like memory.

This example runs a REPL chatbot that demonstrates:
  - Episodic memory: remembers past conversations
  - Semantic memory: extracts and stores facts via consolidation
  - Working memory: maintains context within a conversation
  - Valence memory: tracks emotional associations
  - Consolidation: periodically distills episodes into knowledge

Usage
-----
Option A — OpenAI (easiest):

    export OPENAI_API_KEY=sk-...
    python examples/chatbot_with_memory.py

Option B — Anthropic:

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/chatbot_with_memory.py --model anthropic/claude-3-haiku-20240307

Option C — Ollama (free, local):

    ollama pull llama3.2
    ollama pull nomic-embed-text
    python examples/chatbot_with_memory.py \
        --model ollama/llama3.2 \
        --embedding-model ollama/nomic-embed-text \
        --embedding-dim 768

Option D — Groq (fast, free tier):

    export GROQ_API_KEY=gsk_...
    python examples/chatbot_with_memory.py --model groq/llama3-8b-8192

Commands inside the REPL:
    /memories   — show what's in episodic memory
    /facts      — show extracted semantic triples
    /consolidate — run consolidation now (episode → facts)
    /state      — show orchestrator state
    /quit       — exit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

import anyio

# ---------------------------------------------------------------------------
# Mnemon imports
# ---------------------------------------------------------------------------

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.bus import CognitiveBus
from mnemon.core.config import MnemonConfig
from mnemon.core.models import (
    ContextBlock,
    ContextSource,
    Episode,
    RetrievalQuery,
)
from mnemon.control.attention import AttentionController
from mnemon.control.goals import GoalManager
from mnemon.control.metacognition import MetaCognitionController
from mnemon.control.orchestrator import Orchestrator
from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.learning.reward import RewardProcessor
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.procedural import ProceduralMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.sensory import SensoryBuffer
from mnemon.memory.valence import ValenceMemoryStore
from mnemon.memory.working import WorkingMemoryManager
from mnemon.providers.litellm_provider import (
    LiteLLMEmbeddingProvider,
    LiteLLMProvider,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("mnemon.demo")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Build the brain manually (more control than MnemonFactory)
# ---------------------------------------------------------------------------


async def build_brain(
    model: str,
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, Any]:
    """Assemble all Mnemon modules with real LLM + in-memory backends."""

    config = MnemonConfig()

    # -- Providers (real LLM calls via litellm) --
    llm = LiteLLMProvider(model=model, temperature=0.3, max_tokens=1024)
    embedder = LiteLLMEmbeddingProvider(
        model=embedding_model, dimensions=embedding_dim
    )

    # -- In-memory backends (no infrastructure needed) --
    episodic_vs = InMemoryVectorStore(config)
    episodic_ds = InMemoryDocumentStore(config)
    semantic_vs = InMemoryVectorStore(config)
    semantic_ds = InMemoryDocumentStore(config)
    semantic_gs = InMemoryGraphStore(config)
    procedural_vs = InMemoryVectorStore(config)
    procedural_ds = InMemoryDocumentStore(config)

    # -- Memory stores --
    sensory = SensoryBuffer(config=config.sensory)
    working = WorkingMemoryManager(config=config.working_memory, llm=llm)
    episodic = EpisodicMemoryStore(
        config=config.episodic,
        vector_store=episodic_vs,
        document_store=episodic_ds,
        embedding_provider=embedder,
    )
    semantic = SemanticMemoryStore(
        config=config.semantic,
        graph_store=semantic_gs,
        vector_store=semantic_vs,
        document_store=semantic_ds,
        embedding_provider=embedder,
        llm_provider=llm,
    )
    procedural = ProceduralMemoryStore(
        config=config.procedural,
        vector_store=procedural_vs,
        document_store=procedural_ds,
        embedding_provider=embedder,
    )
    valence = ValenceMemoryStore(
        config=config.valence,
        embedding_provider=embedder,
    )

    # -- Learning modules --
    replay = PrioritizedReplayBuffer(capacity=10_000)
    reward = RewardProcessor(config=config.reward)
    consolidation = ConsolidationEngine(
        config=config.consolidation,
        episodic_memory=episodic,
        semantic_memory=semantic,
        llm=llm,
        embedding_provider=embedder,
        replay_buffer=replay,
    )

    # -- Control modules --
    attention = AttentionController(config=config.attention, valence=valence)
    goals = GoalManager(llm=llm)
    meta = MetaCognitionController(config=config.meta_cognition, llm=llm)

    # -- Bus + Orchestrator --
    bus = CognitiveBus()
    orchestrator = Orchestrator(
        config=config,
        sensory=sensory,
        working_memory=working,
        episodic=episodic,
        semantic=semantic,
        procedural=procedural,
        valence=valence,
        attention=attention,
        goal_manager=goals,
        meta_cognition=meta,
        reward_processor=reward,
        embedding_provider=embedder,
        bus=bus,
    )

    return {
        "orchestrator": orchestrator,
        "episodic": episodic,
        "semantic": semantic,
        "consolidation": consolidation,
        "replay": replay,
        "working": working,
        "bus": bus,
        "llm": llm,
        "embedder": embedder,
    }


# ---------------------------------------------------------------------------
# REPL commands
# ---------------------------------------------------------------------------


async def cmd_memories(brain: dict[str, Any]) -> None:
    """Show recent episodic memories."""
    episodic: EpisodicMemoryStore = brain["episodic"]
    ds = episodic._document_store
    docs = await ds.query(filters={}, limit=20)
    if not docs:
        print("  (no episodes stored yet)")
        return
    for doc in docs[-10:]:  # show last 10
        ctx = doc.get("context", "")[:60]
        action = doc.get("action", "")[:60]
        outcome = doc.get("outcome", "")[:40]
        imp = doc.get("importance", 0)
        print(f"  [{imp:.2f}] {ctx} -> {action} -> {outcome}")


async def cmd_facts(brain: dict[str, Any]) -> None:
    """Show extracted semantic triples."""
    semantic: SemanticMemoryStore = brain["semantic"]
    docs = await semantic._docs.query(filters={"_type": "triple"}, limit=50)
    if not docs:
        print("  (no facts extracted yet — try /consolidate)")
        return
    for doc in docs:
        subj = doc.get("subject", {})
        pred = doc.get("predicate", "?")
        obj = doc.get("object", {})
        subj_name = subj.get("name", "?") if isinstance(subj, dict) else str(subj)
        obj_name = obj.get("name", "?") if isinstance(obj, dict) else str(obj)
        conf = doc.get("confidence", 0)
        print(f"  ({subj_name}) --[{pred}]--> ({obj_name})  conf={conf:.2f}")


async def cmd_consolidate(brain: dict[str, Any]) -> None:
    """Trigger a consolidation cycle (episodes -> semantic facts)."""
    consolidation: ConsolidationEngine = brain["consolidation"]
    print("  Running consolidation cycle...")
    try:
        result = await consolidation.run_cycle()
        print(
            f"  Done: {result.episodes_processed} episodes processed, "
            f"{result.triples_extracted} facts extracted, "
            f"{result.entities_resolved} entities resolved"
        )
    except Exception as exc:
        print(f"  Consolidation failed: {exc}")


async def cmd_state(brain: dict[str, Any]) -> None:
    """Show orchestrator state."""
    orchestrator: Orchestrator = brain["orchestrator"]
    state = orchestrator.get_state()
    wm = state["working_memory"]
    print(f"  Cycles completed: {state['cycle_count']}")
    print(f"  Working memory: {wm['token_used']}/{wm['token_budget']} tokens")
    print(f"  Active goals: {len(state['active_goals'])}")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


async def chat_loop(brain: dict[str, Any]) -> None:
    """Interactive chat loop with the Mnemon-powered agent."""
    orchestrator: Orchestrator = brain["orchestrator"]
    llm: LiteLLMProvider = brain["llm"]
    replay: PrioritizedReplayBuffer = brain["replay"]

    print("\n--- Mnemon Cognitive Agent ---")
    print("Type a message to chat. The agent remembers across turns.")
    print("Commands: /memories /facts /consolidate /state /quit\n")

    cycle_count = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # -- Handle commands --
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit":
                print("Goodbye!")
                break
            elif cmd == "/memories":
                await cmd_memories(brain)
            elif cmd == "/facts":
                await cmd_facts(brain)
            elif cmd == "/consolidate":
                await cmd_consolidate(brain)
            elif cmd == "/state":
                await cmd_state(brain)
            else:
                print(f"  Unknown command: {cmd}")
            continue

        # -- Run cognitive cycle --
        try:
            cycle_result = await orchestrator.run_cycle(raw_input=user_input)
            cycle_count += 1

            # Build a response using the LLM with retrieved context
            context = cycle_result.get("deliberation", {}).get("context", "")
            retrieved_count = cycle_result.get("retrieved_count", 0)

            prompt = (
                f"You are a helpful assistant with memory. "
                f"You remember past conversations and facts.\n\n"
            )
            if context:
                prompt += f"Relevant context from your memory:\n{context}\n\n"
            prompt += f"User says: {user_input}\n\nRespond naturally:"

            response = await llm.generate(prompt)
            print(f"\nAgent: {response}")

            if retrieved_count > 0:
                print(f"  (recalled {retrieved_count} memories)")

            # Push the episode into replay buffer for consolidation
            meta = cycle_result.get("meta_evaluation")
            if meta and meta.get("prediction_error") is not None:
                # Use |RPE| as priority — surprising events consolidate first
                priority = abs(meta["prediction_error"]) + 0.1
            else:
                priority = 0.5

            # The episode was auto-encoded in the learning phase; push its ID
            # to the replay buffer for consolidation
            episodic: EpisodicMemoryStore = brain["episodic"]
            docs = await episodic._document_store.query(filters={}, limit=1)
            if docs:
                from uuid import UUID
                last_doc = docs[-1]
                ep_id = UUID(last_doc["id"]) if "id" in last_doc else None
                if ep_id:
                    try:
                        replay.push(ep_id, priority=priority)
                    except Exception:
                        pass

            # Auto-consolidate every 5 cycles
            if cycle_count % 5 == 0 and cycle_count > 0:
                logger.info("Auto-consolidating after %d cycles...", cycle_count)
                try:
                    result = await brain["consolidation"].run_cycle()
                    if result.triples_extracted > 0:
                        print(
                            f"  (consolidated: {result.triples_extracted} new facts learned)"
                        )
                except Exception:
                    pass  # Non-fatal

        except Exception as exc:
            print(f"\nError during cognitive cycle: {exc}")
            logger.exception("Cycle failed")
        print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Mnemon chatbot demo")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="litellm model string (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="litellm embedding model (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1536,
        help="Embedding dimensions (default: 1536)",
    )
    args = parser.parse_args()

    print(f"Building brain with model={args.model}, embedder={args.embedding_model}...")
    brain = await build_brain(
        model=args.model,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
    )

    # Start the bus in a task group
    async with anyio.create_task_group() as tg:
        await brain["bus"].start(tg)
        try:
            await chat_loop(brain)
        finally:
            await brain["bus"].stop()
            tg.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(main)
