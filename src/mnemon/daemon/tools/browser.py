"""
JarvisBrowser — web browsing capability for the Jarvis daemon.

Uses browser-use (Playwright-backed LLM agent) to let Jarvis actually
browse the internet rather than hallucinating research results.

Integration points:
  1. Chat: user asks "research X" → Jarvis calls browse() → stores result →
     summarizes back to user
  2. Idle loop (_help_master resource mode): Jarvis identifies "look up X
     would help" → actually does it → stores in episodic memory
  3. IPC: `browse` RPC method for CLI/Telegram to trigger directly

Architecture:
  - browser-use Agent handles the browse-think-act loop
  - Results are plain text summaries (not raw HTML)
  - Results stored as episodes in Mnemon episodic memory so Jarvis
    remembers what it researched across sessions
  - Headless Chromium, no GPU, runs fine on a server
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Hard cap: browsing tasks shouldn't run forever
_BROWSE_TIMEOUT_S = 120


class JarvisBrowser:
    """Wraps browser-use Agent with Mnemon memory integration.

    Jarvis uses this to browse the web in response to user requests or
    during idle thinking when it decides research is needed.

    Results are automatically stored in episodic memory so future
    conversations and idle thinking have access to what was learned.
    """

    def __init__(self, brain: Any, llm_provider: Any | None = None) -> None:
        self._brain = brain
        self._llm = llm_provider  # LangChain-compatible LLM (required by browser-use)

    async def browse(self, task: str, store_in_memory: bool = True) -> str:
        """Run a browsing task and return a text summary of the result.

        Parameters
        ----------
        task:
            Natural language description of what to find/do.
            e.g. "Find the top profit-making strategies used by traders on Polymarket"
        store_in_memory:
            If True, store the result as an episodic memory so Jarvis
            remembers what it researched.

        Returns
        -------
        str
            Plain text summary of what was found. Empty string on failure.
        """
        logger.info("Browser task started: %s", task[:100])

        try:
            result = await asyncio.wait_for(
                self._run_browser_task(task),
                timeout=_BROWSE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("Browser task timed out after %ds: %s", _BROWSE_TIMEOUT_S, task[:80])
            return f"(browsing timed out after {_BROWSE_TIMEOUT_S}s — task: {task[:80]})"
        except Exception as exc:
            logger.exception("Browser task failed: %s", task[:80])
            return f"(browsing failed: {exc})"

        if result and store_in_memory:
            await self._store_result(task, result)

        return result

    async def _run_browser_task(self, task: str) -> str:
        """Execute the browser-use agent and return its final result."""
        from browser_use import Agent, Browser, BrowserProfile

        browser = Browser(
            browser_profile=BrowserProfile(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
        )

        llm = self._get_llm()
        if llm is None:
            return "(no LLM configured for browser-use)"

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            max_actions_per_step=5,
        )

        try:
            history = await agent.run(max_steps=15)
            # browser-use returns an AgentHistoryList — extract final result
            result = history.final_result()
            if not result:
                # Fall back to last extracted content
                for item in reversed(history.history):
                    if hasattr(item, "result") and item.result:
                        for r in reversed(item.result):
                            if hasattr(r, "extracted_content") and r.extracted_content:
                                result = r.extracted_content
                                break
                    if result:
                        break
            return str(result).strip() if result else "(no result extracted)"
        finally:
            try:
                await browser.close()
            except Exception:
                pass

    def _get_llm(self):
        """Build a LangChain-compatible LLM from the Mnemon provider config.

        browser-use requires a LangChain LLM. We bridge from Mnemon's
        litellm provider to langchain-openai or langchain-ollama depending
        on what's configured.
        """
        try:
            # Try to detect which backend Mnemon is using
            provider_cfg = {}
            try:
                cfg = self._brain.control.goals._llm
                # Introspect the litellm provider for model name
                if hasattr(cfg, '_model'):
                    model_name = cfg._model
                elif hasattr(cfg, 'model'):
                    model_name = cfg.model
                else:
                    model_name = None
            except Exception:
                model_name = None

            if model_name and model_name.startswith("ollama/"):
                # Use langchain-ollama
                from langchain_ollama import ChatOllama
                ollama_model = model_name.replace("ollama/", "")
                return ChatOllama(model=ollama_model, temperature=0)

            # Try langchain-openai as fallback
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # Last resort — try ollama with default model
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(model="qwen2.5:7b", temperature=0)
            except Exception:
                pass

            logger.error("No LLM available for browser-use — install langchain-ollama or set OPENAI_API_KEY")
            return None

        except Exception as exc:
            logger.error("Failed to build LLM for browser-use: %s", exc)
            return None

    async def _store_result(self, task: str, result: str) -> None:
        """Store a browsing result as an episodic memory."""
        try:
            from mnemon.core.models import Episode
            episode = Episode(
                agent_id="jarvis",
                session_id=uuid4(),
                context=f"[web research] {task}",
                action=f"Browsed the web for: {task}",
                outcome=result[:2000],  # truncate very long results
                tags=["web_research", "browsing"],
                importance=0.7,
            )
            await self._brain.memory.episodic.store(episode)
            logger.info("Stored browsing result as episode: %s", task[:60])
        except Exception as exc:
            logger.warning("Failed to store browsing result: %s", exc)
