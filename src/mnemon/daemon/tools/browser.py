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
import re
from contextlib import suppress
from typing import Any
from uuid import uuid4

import httpx

from mnemon.daemon.capture_policy import classify_interaction
from mnemon.daemon.config import DaemonConfig
from mnemon.daemon.privacy import apply_redactions, load_privacy_rules, should_exclude_text

logger = logging.getLogger(__name__)

# Hard cap: browsing tasks shouldn't run forever
_BROWSE_TIMEOUT_S = 120
_HTTP_RESEARCH_TIMEOUT_S = 25


def _strip_html(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    return re.sub(r"\s+", " ", text).strip()


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
        except TimeoutError:
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
        try:
            from browser_use import Agent, Browser, BrowserProfile
        except ImportError:
            return await self._run_http_research_task(task)

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
            with suppress(Exception):
                await browser.close()

    async def _run_http_research_task(self, task: str) -> str:
        """Dependency-light research fallback when browser-use is unavailable."""
        query = task.strip()
        if not query:
            return ""

        async with httpx.AsyncClient(
            timeout=_HTTP_RESEARCH_TIMEOUT_S,
            headers={"User-Agent": "Mnemon-Jarvis/0.1"},
            follow_redirects=True,
        ) as client:
            response = await client.get("https://duckduckgo.com/html/", params={"q": query})
            response.raise_for_status()
            search_text = _strip_html(response.text)

        prompt = (
            "Summarize these web search results for Jarvis memory. "
            "Keep only concrete findings relevant to the task, and state that these are "
            "search-result snippets if full pages were not opened.\n\n"
            f"Task: {task}\n\nSearch text:\n{search_text[:6000]}"
        )
        llm = getattr(getattr(self._brain.control, "goals", None), "_llm", None)
        if llm is not None:
            with suppress(Exception):
                summary = str(await llm.generate(prompt, max_tokens=700) or "").strip()
                if summary:
                    return summary
        return search_text[:2000]

    def _get_llm(self):
        """Build a LangChain-compatible LLM from the Mnemon provider config.

        browser-use requires a LangChain LLM. We bridge from Mnemon's
        litellm provider to langchain-openai or langchain-ollama depending
        on what's configured.
        """
        try:
            # Try to detect which backend Mnemon is using
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

            logger.error(
                "No LLM available for browser-use — install langchain-ollama or set OPENAI_API_KEY"
            )
            return None

        except Exception as exc:
            logger.error("Failed to build LLM for browser-use: %s", exc)
            return None

    async def _store_result(self, task: str, result: str) -> None:
        """Store a browsing result as an episodic memory."""
        try:
            from mnemon.core.models import Episode
            privacy_rules = load_privacy_rules(DaemonConfig().state_path)
            if (
                should_exclude_text(task, privacy_rules)
                or should_exclude_text(result, privacy_rules)
            ):
                logger.info("Skipping browse result storage due to privacy rules")
                return
            decision = classify_interaction(
                user_message=task,
                assistant_reply=result,
                source="browse",
                excluded_phrases=privacy_rules.excluded_phrases,
            )
            if not decision.store_memory:
                logger.info("Skipping browse result storage due to capture policy")
                return
            redacted_task = apply_redactions(task, privacy_rules)
            redacted_result = apply_redactions(result, privacy_rules)
            episode = Episode(
                agent_id="jarvis",
                session_id=uuid4(),
                context=f"[web research] {redacted_task}",
                action=f"Browsed the web for: {redacted_task}",
                outcome=redacted_result[:2000],  # truncate very long results
                tags=["web_research", "browsing", *decision.tags],
                importance=max(0.7, decision.importance),
            )
            await self._brain.memory.episodic.encode(episode)
            logger.info("Stored browsing result as episode: %s", task[:60])
        except Exception as exc:
            logger.warning("Failed to store browsing result: %s", exc)
