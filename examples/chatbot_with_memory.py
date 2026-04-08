#!/usr/bin/env python3
"""
Mnemon live demo — conversational agent with brain-like memory.

This example uses Mnemon's memory stores directly (bypassing the
Orchestrator's full cognitive cycle) to build a practical chatbot
that demonstrates:
  - Episodic memory: remembers past conversations verbatim
  - Semantic memory: extracts and stores facts via consolidation
  - Working memory: maintains sliding window context
  - Valence memory: tracks emotional associations
  - Consolidation: distills episodes into reusable knowledge

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
    python examples/chatbot_with_memory.py \\
        --model ollama/llama3.2 \\
        --embedding-model ollama/nomic-embed-text \\
        --embedding-dim 768

Option D — Groq (fast, free tier):

    export GROQ_API_KEY=gsk_...
    python examples/chatbot_with_memory.py --model groq/llama3-8b-8192

Commands inside the REPL:
    /memories     — show stored episodic memories
    /facts        — show extracted semantic triples
    /consolidate  — run consolidation now (episodes → facts)
    /state        — show memory stats
    /quit         — exit
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from typing import Any
from uuid import UUID, uuid4

import anyio

from mnemon.backends.memory_store import (
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mnemon.core.config import MnemonConfig
from mnemon.core.models import (
    Episode,
    RetrievalQuery,
)
from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer
from mnemon.memory.episodic import EpisodicMemoryStore
from mnemon.memory.semantic import SemanticMemoryStore
from mnemon.memory.valence import ValenceMemoryStore
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
# Brain assembly
# ---------------------------------------------------------------------------


class MnemonChatbot:
    """Chatbot that uses Mnemon memory stores for long-term recall."""

    def __init__(
        self,
        llm: LiteLLMProvider,
        embedder: LiteLLMEmbeddingProvider,
        episodic: EpisodicMemoryStore,
        semantic: SemanticMemoryStore,
        valence: ValenceMemoryStore,
        consolidation: ConsolidationEngine,
        replay: PrioritizedReplayBuffer,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.episodic = episodic
        self.semantic = semantic
        self.valence = valence
        self.consolidation = consolidation
        self.replay = replay

        # Conversation history (last N turns for LLM context window)
        self.history: deque[dict[str, str]] = deque(maxlen=20)
        self.session_id = uuid4()
        self.turn_count = 0

    async def respond(self, user_input: str) -> str:
        """Process user input: retrieve memories, generate response, encode episode."""

        self.turn_count += 1

        # --------------------------------------------------
        # 1. RETRIEVE — search episodic + semantic memory
        # --------------------------------------------------
        retrieved_context: list[str] = []

        try:
            query = RetrievalQuery(query_text=user_input, top_k=5, min_score=0.01)
            result = await self.episodic.retrieve(query)
            for item in result.items:
                retrieved_context.append(
                    f"[memory] {item.content}"
                )
        except Exception as exc:
            logger.debug("Episodic retrieval failed: %s", exc)

        try:
            query_emb = await self.embedder.embed(user_input)
            triples = await self.semantic.retrieve_by_similarity(query_emb, top_k=5)
            for t in triples:
                obj_name = t.object.name if hasattr(t.object, "name") else str(t.object)
                retrieved_context.append(
                    f"[fact] {t.subject.name} {t.predicate} {obj_name}"
                )
        except Exception as exc:
            logger.debug("Semantic retrieval failed: %s", exc)

        # --------------------------------------------------
        # 2. GENERATE — LLM call with history + retrieved context
        # --------------------------------------------------
        system_prompt = (
            "You are a helpful assistant with long-term memory. "
            "You remember things the user told you in previous messages. "
            "When you recall information from memory, use it naturally — "
            "don't say 'according to my memory' unless asked. "
            "Be concise and conversational."
        )

        if retrieved_context:
            system_prompt += (
                "\n\nRelevant information from your memory:\n"
                + "\n".join(retrieved_context)
            )

        # Build conversation for the LLM
        messages_text = f"System: {system_prompt}\n\n"
        for turn in self.history:
            messages_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        messages_text += f"User: {user_input}\nAssistant:"

        response = await self.llm.generate(messages_text)
        response = response.strip()

        # --------------------------------------------------
        # 3. ENCODE — store this exchange as an episodic memory
        # --------------------------------------------------
        episode = Episode(
            agent_id="chatbot",
            session_id=self.session_id,
            context=f"User said: {user_input}",
            action=f"Agent responded: {response[:200]}",
            outcome=f"Turn {self.turn_count} of conversation",
            importance=self._estimate_importance(user_input),
        )
        try:
            ep_id = await self.episodic.encode(episode)
            # Push to replay buffer for consolidation
            self.replay.add(ep_id, priority=episode.importance)
            logger.info("Encoded episode %s (importance=%.2f)", ep_id, episode.importance)
        except Exception as exc:
            logger.warning("Failed to encode episode: %s", exc)

        # Update valence for entities mentioned
        try:
            # Simple entity extraction: capitalised words as proxies
            words = user_input.split()
            entities = [w.strip(".,!?") for w in words if w[0:1].isupper() and len(w) > 1]
            if entities:
                # Positive valence for normal conversation
                await self.valence.update(entities, valence=0.1)
        except Exception:
            pass

        # --------------------------------------------------
        # 4. REMEMBER — add to conversation history
        # --------------------------------------------------
        self.history.append({"user": user_input, "assistant": response})

        # Auto-consolidate every 5 turns
        consolidation_note = ""
        if self.turn_count % 5 == 0:
            try:
                result = await self.consolidation.run_cycle()
                if result.triples_extracted > 0:
                    consolidation_note = (
                        f"  (learned {result.triples_extracted} new facts from conversation)"
                    )
            except Exception:
                pass

        recall_note = ""
        if retrieved_context:
            recall_note = f"  (recalled {len(retrieved_context)} memories)"

        return response + "\n" + recall_note + consolidation_note

    @staticmethod
    def _estimate_importance(text: str) -> float:
        """Heuristic importance scoring for an utterance."""
        importance = 0.3  # baseline

        # Personal information is important
        personal_markers = [
            "my name", "i am", "i'm", "i work", "i live", "i like",
            "i love", "i hate", "my job", "my team", "my project",
            "remember", "don't forget", "important",
        ]
        text_lower = text.lower()
        for marker in personal_markers:
            if marker in text_lower:
                importance += 0.15

        # Questions are moderately important
        if "?" in text:
            importance += 0.1

        # Longer messages tend to be more informative
        if len(text) > 100:
            importance += 0.1

        return min(importance, 1.0)


# ---------------------------------------------------------------------------
# REPL commands
# ---------------------------------------------------------------------------


async def cmd_memories(bot: MnemonChatbot) -> None:
    """Show recent episodic memories."""
    ds = bot.episodic._document_store
    docs = await ds.query(filters={}, limit=50)
    if not docs:
        print("  (no episodes stored yet)")
        return
    for doc in docs[-10:]:
        ctx = doc.get("context", "")[:80]
        action = doc.get("action", "")[:80]
        imp = doc.get("importance", 0)
        print(f"  [{imp:.2f}] {ctx}")
        print(f"         {action}")


async def cmd_facts(bot: MnemonChatbot) -> None:
    """Show extracted semantic triples."""
    docs = await bot.semantic._docs.query(filters={"_type": "triple"}, limit=50)
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


async def cmd_consolidate(bot: MnemonChatbot) -> None:
    """Trigger a consolidation cycle (episodes -> semantic facts)."""
    print("  Running consolidation cycle...")
    try:
        result = await bot.consolidation.run_cycle()
        print(
            f"  Done: {result.episodes_processed} episodes processed, "
            f"{result.triples_extracted} facts extracted, "
            f"{result.entities_resolved} entities resolved"
        )
    except Exception as exc:
        print(f"  Consolidation failed: {exc}")


async def cmd_state(bot: MnemonChatbot) -> None:
    """Show memory stats."""
    ep_docs = await bot.episodic._document_store.query(filters={}, limit=100_000)
    triple_docs = await bot.semantic._docs.query(filters={"_type": "triple"}, limit=100_000)
    print(f"  Turns completed:    {bot.turn_count}")
    print(f"  Episodic memories:  {len(ep_docs)}")
    print(f"  Semantic facts:     {len(triple_docs)}")
    print(f"  Replay buffer size: {bot.replay.size}")
    print(f"  Conversation history: {len(bot.history)} turns (max {bot.history.maxlen})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Mnemon chatbot demo")
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="litellm model string (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model", default="text-embedding-3-small",
        help="litellm embedding model (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=1536,
        help="Embedding dimensions (default: 1536)",
    )
    args = parser.parse_args()

    print(f"Building brain: model={args.model}, embedder={args.embedding_model}...")

    config = MnemonConfig()
    llm = LiteLLMProvider(model=args.model, temperature=0.3, max_tokens=1024)
    embedder = LiteLLMEmbeddingProvider(
        model=args.embedding_model, dimensions=args.embedding_dim,
    )

    # In-memory backends — no infrastructure needed
    episodic_vs = InMemoryVectorStore(config)
    episodic_ds = InMemoryDocumentStore(config)
    semantic_vs = InMemoryVectorStore(config)
    semantic_ds = InMemoryDocumentStore(config)
    semantic_gs = InMemoryGraphStore(config)

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
    valence = ValenceMemoryStore(
        config=config.valence,
        embedding_provider=embedder,
    )
    replay = PrioritizedReplayBuffer(capacity=10_000)
    consolidation = ConsolidationEngine(
        config=config.consolidation,
        episodic_memory=episodic,
        semantic_memory=semantic,
        llm=llm,
        embedding_provider=embedder,
        replay_buffer=replay,
    )

    bot = MnemonChatbot(
        llm=llm,
        embedder=embedder,
        episodic=episodic,
        semantic=semantic,
        valence=valence,
        consolidation=consolidation,
        replay=replay,
    )

    # Quick connectivity test
    print("Testing LLM connection...")
    try:
        test = await llm.generate("Say 'ok' in one word.")
        print(f"  LLM response: {test.strip()}")
    except Exception as exc:
        print(f"  ERROR: LLM connection failed: {exc}")
        print("  Check your API key and model name.")
        return

    print("Testing embedding connection...")
    try:
        test_emb = await embedder.embed("test")
        print(f"  Embedding dim: {len(test_emb)}")
    except Exception as exc:
        print(f"  ERROR: Embedding connection failed: {exc}")
        print("  Check your embedding model name.")
        return

    print("\n--- Mnemon Cognitive Agent ---")
    print("Type a message to chat. The agent remembers across turns.")
    print("Commands: /memories /facts /consolidate /state /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd == "/quit":
                print("Goodbye!")
                break
            elif cmd == "/memories":
                await cmd_memories(bot)
            elif cmd == "/facts":
                await cmd_facts(bot)
            elif cmd == "/consolidate":
                await cmd_consolidate(bot)
            elif cmd == "/state":
                await cmd_state(bot)
            else:
                print(f"  Unknown command: {cmd}")
            continue

        response = await bot.respond(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    anyio.run(main)
