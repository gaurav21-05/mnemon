"""
Interactive cognitive agent powered by the Mnemon memory framework.

This example demonstrates the full Mnemon pipeline: the MnemonFactory builds
all six memory subsystems, learning modules, and cognitive controls.  User
input flows through the 6-phase cognitive cycle (perception, attention,
retrieval, deliberation, execution, learning), and the agent generates
LLM responses augmented with retrieved memories.

Usage
-----
    # OpenAI (default)
    export OPENAI_API_KEY=sk-...
    python examples/agent.py

    # Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/agent.py --model claude-3-5-haiku-20241022

    # Ollama (local)
    python examples/agent.py --model ollama/llama3

    # Custom embedding model
    python examples/agent.py --embedding-model text-embedding-3-large --embedding-dim 3072

REPL commands
-------------
    /memories   Show recent episodic memories
    /facts      Show semantic knowledge triples
    /skills     Show learned procedural skills
    /state      Show cognitive state (working memory, goals, bus)
    /consolidate  Run memory consolidation (episodic -> semantic)
    /goals      Show active goals
    /help       Show available commands
    /quit       Exit the agent
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import deque
from typing import Any

from mnemon.core.config import MnemonConfig
from mnemon.core.models import RetrievalQuery
from mnemon.factory import MnemonFactory
from mnemon.providers.litellm_provider import LiteLLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful assistant with a cognitive memory system. You remember past
conversations and facts you have learned. When relevant memories are available,
use them to provide more informed and personalised responses.

If the user refers to something discussed earlier, draw on your memories.
Be concise and natural.\
"""


class CognitiveAgent:
    """Interactive agent that wraps the Mnemon cognitive cycle with LLM response generation."""

    def __init__(self, brain: Any, llm: LiteLLMProvider) -> None:
        self.brain = brain
        self.llm = llm
        self.history: deque[dict[str, str]] = deque(maxlen=20)
        self.turn_count = 0
        self._last_cycle: dict[str, Any] | None = None

    async def chat(self, user_input: str) -> str:
        """Run a full cognitive cycle and generate a response."""
        # Phase 1-6: Full cognitive cycle
        cycle_result = await self.brain.run_cycle(user_input)
        self._last_cycle = cycle_result

        # Generate LLM response from deliberation context
        response = await self._generate_response(cycle_result, user_input)

        self.history.append({"user": user_input, "assistant": response})
        self.turn_count += 1
        return response

    async def _generate_response(
        self, cycle_result: dict[str, Any], user_input: str
    ) -> str:
        """Build a memory-augmented prompt and generate a response."""
        deliberation = cycle_result.get("deliberation", {})
        memory_context = deliberation.get("context", "")
        goal_text = deliberation.get("goal", "No specific goal")
        meta = cycle_result.get("meta_evaluation")

        # Build prompt
        parts: list[str] = [f"System: {SYSTEM_PROMPT}"]

        if memory_context.strip():
            parts.append(f"\n--- Retrieved Memories ---\n{memory_context}\n--- End Memories ---")

        if goal_text and goal_text != "No specific goal":
            parts.append(f"\nCurrent goal: {goal_text}")

        if meta and meta.get("strategy_recommended"):
            parts.append(f"\n[Meta-cognitive note: consider {meta['strategy_recommended']}]")

        parts.append("")

        # Conversation history (last 10 turns)
        for turn in list(self.history)[-10:]:
            parts.append(f"User: {turn['user']}")
            parts.append(f"Assistant: {turn['assistant']}")
            parts.append("")

        parts.append(f"User: {user_input}")
        parts.append("Assistant:")

        prompt = "\n".join(parts)
        return await self.llm.generate(prompt)

    async def show_memories(self) -> str:
        """Query and format recent episodic memories."""
        query = RetrievalQuery(query_text="recent conversation", top_k=10, min_score=0.0)
        result = await self.brain.memory.episodic.retrieve(query)

        if not result.items:
            return "No episodic memories stored yet."

        lines = [f"Episodic memories ({len(result.items)} found):"]
        for i, item in enumerate(result.items, 1):
            content = item.content[:120].replace("\n", " ")
            lines.append(f"  {i}. [{item.source_store}] (score={item.score:.3f}) {content}")
        return "\n".join(lines)

    async def show_facts(self) -> str:
        """Show semantic knowledge triples."""
        # Retrieve facts by querying with a broad embedding
        embedder = self.brain.memory.semantic._embedding_provider
        query_emb = await embedder.embed("knowledge facts information")
        triples = await self.brain.memory.semantic.retrieve_by_similarity(query_emb, top_k=20)

        if not triples:
            return "No semantic facts stored yet. Run /consolidate to extract facts from memories."

        lines = [f"Semantic facts ({len(triples)} found):"]
        for i, t in enumerate(triples, 1):
            obj_name = t.object.name if hasattr(t.object, "name") else str(t.object)
            lines.append(
                f"  {i}. {t.subject.name} --[{t.predicate}]--> {obj_name}"
                f"  (conf={t.confidence:.2f})"
            )
        return "\n".join(lines)

    async def show_skills(self) -> str:
        """Show learned procedural skills."""
        embedder = self.brain.memory.procedural._embedding_provider
        query_emb = await embedder.embed("skills abilities procedures")
        skills = await self.brain.memory.procedural.retrieve(query_emb, top_k=10)

        if not skills:
            return "No procedural skills learned yet."

        lines = [f"Procedural skills ({len(skills)} found):"]
        for i, s in enumerate(skills, 1):
            lines.append(
                f"  {i}. {s.name} — {s.description[:80]} "
                f"(utility={s.utility:.2f}, status={s.status})"
            )
        return "\n".join(lines)

    def show_state(self) -> str:
        """Format the current cognitive state."""
        state = self.brain.get_state()
        wm = state["working_memory"]
        goals = state["active_goals"]
        bus = state["bus"]

        lines = [
            "Cognitive State:",
            f"  Cycle count:     {state['cycle_count']}",
            f"  Conversation:    {self.turn_count} turns",
            f"  Working memory:  {wm['token_used']}/{wm['token_budget']} tokens "
            f"({wm['token_available']} available)",
            f"  Active goals:    {len(goals)}",
            f"  Bus running:     {bus['running']}",
            f"  Subscriptions:   {bus['subscriptions']}",
        ]

        if goals:
            lines.append("  Goals:")
            for g in goals:
                lines.append(
                    f"    - {g['description'][:60]} "
                    f"(priority={g['priority']:.1f}, progress={g['progress']:.0%})"
                )

        if self._last_cycle:
            meta = self._last_cycle.get("meta_evaluation")
            if meta:
                lines.append(
                    f"  Last cycle meta: confidence={meta['confidence']:.2f}, "
                    f"prediction_error={meta['prediction_error']:.3f}"
                )

        return "\n".join(lines)

    async def run_consolidation(self) -> str:
        """Trigger memory consolidation (episodic -> semantic)."""
        try:
            result = await self.brain.learning.consolidation.run_cycle()
            return (
                f"Consolidation complete:\n"
                f"  Episodes processed: {result.episodes_processed}\n"
                f"  Triples extracted:  {result.triples_extracted}\n"
                f"  Entities resolved:  {result.entities_resolved}\n"
                f"  Duration:           {result.duration_ms:.0f}ms"
            )
        except Exception as e:
            return f"Consolidation failed: {e}"

    def show_goals(self) -> str:
        """Show active goals."""
        goals = self.brain.control.goals.get_active_goals()
        if not goals:
            return "No active goals."

        lines = [f"Active goals ({len(goals)}):"]
        for g in goals:
            lines.append(
                f"  - [{g.status.value}] {g.description} "
                f"(priority={g.priority:.1f}, attempts={g.attempts}/{g.max_attempts})"
            )
        return "\n".join(lines)


HELP_TEXT = """
Available commands:
  /memories     Show recent episodic memories
  /facts        Show semantic knowledge triples
  /skills       Show learned procedural skills
  /state        Show cognitive state (working memory, goals, bus)
  /consolidate  Run memory consolidation (episodic -> semantic)
  /goals        Show active goals
  /help         Show this help message
  /quit         Exit the agent
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive cognitive agent powered by Mnemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or other provider keys as env vars.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LiteLLM model identifier (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1536,
        help="Embedding dimensions (default: 1536)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Response generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Configure Mnemon with the user's model choices
    config = MnemonConfig()
    provider_name = config.llm.default_provider
    config.llm.providers[provider_name] = {
        **config.llm.providers.get(provider_name, {}),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": args.embedding_dim,
    }

    print(f"Building cognitive framework (model={args.model})...")
    factory = MnemonFactory(config)
    brain = await factory.build()

    # Separate LLM for response generation (higher temperature for natural conversation)
    response_llm = LiteLLMProvider(
        model=args.model,
        temperature=args.temperature,
        max_tokens=2048,
    )

    agent = CognitiveAgent(brain, response_llm)

    async with brain:
        print(f"Mnemon cognitive agent ready. Model: {args.model}")
        print("Type /help for commands, /quit to exit.\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd == "/quit":
                    print("Goodbye!")
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/memories":
                    print(await agent.show_memories())
                elif cmd == "/facts":
                    print(await agent.show_facts())
                elif cmd == "/skills":
                    print(await agent.show_skills())
                elif cmd == "/state":
                    print(agent.show_state())
                elif cmd == "/consolidate":
                    print("Running consolidation...")
                    print(await agent.run_consolidation())
                elif cmd == "/goals":
                    print(agent.show_goals())
                else:
                    print(f"Unknown command: {cmd}. Type /help for available commands.")
                print()
                continue

            # Normal conversation
            try:
                response = await agent.chat(user_input)
                print(f"\nAssistant: {response}\n")
            except Exception as e:
                logger.exception("Error during chat")
                print(f"\nError: {e}\n")

    await brain.close()


if __name__ == "__main__":
    asyncio.run(main())
