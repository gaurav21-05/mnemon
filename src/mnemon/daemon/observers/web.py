"""
WebLearningObserver — the daemon's reading sense.

Brain analog: The ventral visual stream (the "what" pathway) combined with
the parahippocampal place area — processes structured content from the
environment, recognises meaningful patterns, and feeds them into the
hippocampus for episodic encoding. Just as a human's eyes scan a page and
the hippocampus encodes what was read, this observer fetches web content
and injects it into Mnemon's sensory pipeline.

Learning flow:
    URL / RSS feed
        ↓ fetch (httpx async)
        ↓ parse (RSS/Atom XML or raw HTML)
        ↓ extract text (strip markup)
        ↓ sensory buffer (SensoryBuffer.process)
        ↓ cognitive cycle (encode → episodic memory)
        ↓ consolidation queue (replay buffer)
        → semantic KG grows over time via idle consolidation

The observer runs each source on an independent interval, checking
_last_fetched to skip sources that were recently read.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx

from mnemon.daemon.observers import ObserverPlugin

logger = logging.getLogger(__name__)

# RSS / Atom XML namespaces
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}

# Characters to split long articles into chunks (rough word count)
_CHUNK_WORDS = 300

# Maximum articles to ingest per fetch cycle (avoid flooding episodic memory)
_MAX_ARTICLES_PER_FETCH = 5


@dataclass
class WebSource:
    """A single web learning source (RSS feed or raw URL)."""

    url: str
    name: str = ""
    kind: str = "rss"          # "rss", "atom", "url"
    interval_s: int = 3600     # How often to re-fetch (seconds)
    last_fetched: float = 0.0  # monotonic timestamp of last successful fetch


def _strip_html(text: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_words: int = _CHUNK_WORDS) -> list[str]:
    """Split text into ~chunk_words word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i : i + chunk_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def _parse_rss(xml_text: str) -> list[dict[str, str]]:
    """Parse RSS 2.0 feed, return list of {title, summary} dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else root.findall(".//item")
        for item in items[:_MAX_ARTICLES_PER_FETCH]:
            title = (item.findtext("title") or "").strip()
            desc = (
                item.findtext("description")
                or item.findtext(f"{{{_NS['content']}}}encoded")
                or ""
            )
            summary = _strip_html(desc)[:600]
            if title:
                articles.append({"title": title, "summary": summary})
    except ET.ParseError as exc:
        logger.debug("RSS parse error: %s", exc)
    return articles


def _parse_atom(xml_text: str) -> list[dict[str, str]]:
    """Parse Atom feed, return list of {title, summary} dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        ns = _NS["atom"]
        entries = root.findall(f"{{{ns}}}entry")
        for entry in entries[:_MAX_ARTICLES_PER_FETCH]:
            title_el = entry.find(f"{{{ns}}}title")
            title = (title_el.text or "") if title_el is not None else ""
            summary_el = entry.find(f"{{{ns}}}summary") or entry.find(f"{{{ns}}}content")
            summary_text = (summary_el.text or "") if summary_el is not None else ""
            summary = _strip_html(summary_text)[:600]
            if title:
                articles.append({"title": title.strip(), "summary": summary})
    except ET.ParseError as exc:
        logger.debug("Atom parse error: %s", exc)
    return articles


class WebLearningObserver(ObserverPlugin):
    """Periodically fetches web sources and feeds content into Mnemon.

    Each source is fetched independently on its own interval.  Content is
    processed through the sensory buffer and encoded as episodic memories,
    which are later consolidated into semantic knowledge during idle ticks.

    Brain analog: The reading sense — the hippocampus encodes what was read,
    and slow-wave sleep (consolidation) distills it into neocortical knowledge.
    """

    def __init__(
        self,
        sources: list[WebSource] | None = None,
        poll_interval_s: float = 60.0,
    ) -> None:
        self._sources: list[WebSource] = sources if sources is not None else _default_sources()
        self._poll_interval_s = poll_interval_s
        self._brain: Any = None
        self._running = False

    @property
    def name(self) -> str:
        return "web_learning"

    async def start(self, brain: Any) -> None:
        self._brain = brain
        self._running = True
        logger.info(
            "WebLearningObserver started — %d source(s), poll=%.0fs",
            len(self._sources),
            self._poll_interval_s,
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("WebLearningObserver stopped.")

    def is_running(self) -> bool:
        return self._running

    async def run(self) -> None:
        """Main loop: check each source, fetch if due, inject content."""
        while self._running:
            now = time.monotonic()
            for source in self._sources:
                if now - source.last_fetched >= source.interval_s:
                    try:
                        await self._fetch_source(source)
                        source.last_fetched = time.monotonic()
                    except Exception:
                        logger.exception("WebLearningObserver: fetch failed for %s", source.url)
            await anyio.sleep(self._poll_interval_s)

    # ------------------------------------------------------------------
    # Internal fetching
    # ------------------------------------------------------------------

    async def _fetch_source(self, source: WebSource) -> None:
        """Fetch a source and inject articles into the brain."""
        logger.info("WebLearningObserver: fetching %s (%s)", source.name or source.url, source.kind)

        async with httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "Mnemon-WebLearner/1.0"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(source.url)
            resp.raise_for_status()
            raw = resp.text

        if source.kind in ("rss", "feed"):
            articles = _parse_rss(raw)
            if not articles:
                articles = _parse_atom(raw)  # try Atom if RSS parse yields nothing
        elif source.kind == "atom":
            articles = _parse_atom(raw)
        else:
            # Raw HTML / plain text — chunk it
            articles = []
            text = _strip_html(raw)
            for i, chunk in enumerate(_chunk_text(text)):
                articles.append({
                    "title": f"{source.name or source.url} (part {i + 1})",
                    "summary": chunk,
                })
            articles = articles[:_MAX_ARTICLES_PER_FETCH]

        if not articles:
            logger.debug("WebLearningObserver: no articles extracted from %s", source.url)
            return

        ingested = 0
        for article in articles:
            content = f"{article['title']}. {article['summary']}".strip()
            if len(content) < 20:
                continue
            try:
                await self._ingest_text(content, source_name=source.name or source.url)
                ingested += 1
            except Exception:
                logger.exception("WebLearningObserver: ingest failed for article '%s'", article["title"][:60])

        logger.info(
            "WebLearningObserver: ingested %d/%d articles from %s",
            ingested, len(articles), source.name or source.url,
        )

    async def _ingest_text(self, text: str, source_name: str) -> None:
        """Feed text into the sensory buffer and run a lightweight cognitive cycle."""
        # Label text with source so consolidation can tag origin
        labeled = f"[web:{source_name}] {text}"

        # Process through sensory buffer (generates embedding, NER, etc.)
        percept = await self._brain.memory.sensory.process(labeled)

        # Encode directly as an episode — importance=0.4 (moderate, not user-level)
        from mnemon.core.models import Episode
        import uuid

        episode = Episode(
            agent_id="web_learner",
            session_id=uuid.uuid4(),
            context=labeled[:800],
            action="read",
            outcome="content ingested for consolidation",
            importance=0.4,
        )
        ep_id = await self._brain.memory.episodic.encode(episode)

        # Push to replay buffer so idle consolidation picks it up
        try:
            self._brain.learning.replay_buffer.add(ep_id, priority=episode.importance)
        except Exception:
            pass  # replay buffer not critical

        logger.debug("Ingested episode %s from %s", ep_id, source_name)


def _default_sources() -> list[WebSource]:
    """Built-in starter sources — practical, broad, low-noise."""
    return [
        WebSource(
            url="https://feeds.feedburner.com/oreilly/radar/atom",
            name="O'Reilly Radar",
            kind="atom",
            interval_s=7200,
        ),
        WebSource(
            url="https://news.ycombinator.com/rss",
            name="Hacker News",
            kind="rss",
            interval_s=3600,
        ),
        WebSource(
            url="https://feeds.arstechnica.com/arstechnica/technology-lab",
            name="Ars Technica Tech",
            kind="rss",
            interval_s=7200,
        ),
        WebSource(
            url="https://rss.arxiv.org/rss/cs.AI",
            name="arXiv AI Papers",
            kind="rss",
            interval_s=86400,  # daily
        ),
    ]
