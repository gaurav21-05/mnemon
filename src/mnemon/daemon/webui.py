"""
Mnemon Web UI — real-time dashboard for the Jarvis daemon.

Serves an aiohttp web server on localhost:7777 that connects to the running
daemon via Unix socket IPC and streams live updates to the browser.

Features:
  - Status panel: uptime, cycles, autonomy level
  - Live thoughts feed: idle thinking activity (reflection/consolidation/planning/exploration)
  - Goal tree: active goals and their subgoals
  - Chat interface: send messages to the brain and see responses
  - Log tail: last N lines of daemon.log streamed via SSE

Usage:
    python -m mnemon.daemon.webui
    # Open http://localhost:7777 in your browser
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
from pathlib import Path
from typing import Any

from aiohttp import web

from mnemon.daemon.cli.client import DaemonClient
from mnemon.daemon.config import DaemonConfig
from mnemon.daemon.identity import JarvisIdentity
from mnemon.daemon.privacy import load_privacy_rules

logger = logging.getLogger(__name__)

SOCKET_PATH = Path("~/.mnemon/daemon.sock").expanduser()
LOG_PATH = Path("~/.mnemon/daemon.log").expanduser()
PORT = 7777
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

_MANAGED_FILES: dict[str, dict[str, str]] = {
    "master.md": {
        "label": "Master Profile",
        "description": "What Jarvis knows about you.",
        "format": "markdown",
    },
    "soul.md": {
        "label": "Jarvis Identity",
        "description": "Jarvis self-model and values.",
        "format": "markdown",
    },
    "learnings.md": {
        "label": "Learnings",
        "description": "Accumulated knowledge and insights.",
        "format": "markdown",
    },
    "goals.json": {
        "label": "Goals Store",
        "description": "Raw persisted goal graph.",
        "format": "json",
    },
    "daemon_state.json": {
        "label": "Daemon State",
        "description": "Raw persisted daemon runtime state.",
        "format": "json",
    },
    "privacy_rules.json": {
        "label": "Privacy Rules",
        "description": "Exclusion and redaction controls for persistent memory.",
        "format": "json",
    },
}

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mnemon — Jarvis Control Room</title>
<script>
  (() => {
    const saved = localStorage.getItem('mnemon-theme');
    const preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.dataset.theme = saved || preferred;
  })();
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link
  href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap"
  rel="stylesheet"
>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {
    color-scheme: light;
    --paper: #f2f1ed;
    --surface: #f7f7f4;
    --surface-muted: #ebeae5;
    --surface-soft: #e6e5e0;
    --surface-sand: #e1e0db;
    --border: oklab(0.263084 -0.00230259 0.0124794 / 0.1);
    --border-strong: oklab(0.263084 -0.00230259 0.0124794 / 0.24);
    --border-solid: #26251e;
    --ink: rgba(38,37,30,0.78);
    --ink-strong: #26251e;
    --muted: rgba(38,37,30,0.58);
    --muted-dim: rgba(38,37,30,0.42);
    --accent: #f54e00;
    --accent-strong: #cf2d56;
    --accent-soft: rgba(245,78,0,0.12);
    --accent-wash: rgba(245,78,0,0.06);
    --success: #1f8a65;
    --danger: #cf2d56;
    --focus: rgba(38,37,30,0.16);
    --topbar-bg: rgba(242,241,237,0.88);
    --text-on-accent: #ffffff;
    --shadow-card:
      rgba(0,0,0,0.045) 0 8px 22px,
      oklab(0.263084 -0.00230259 0.0124794 / 0.1) 0 0 0 1px;
    --shadow-soft: rgba(0,0,0,0.035) 0 6px 16px;
    --shadow-inset: transparent 0 0 0;
    --font-display: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    --font-sans: Georgia, 'Iowan Old Style', 'Times New Roman', serif;
    --font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
    /* Memory type colors */
    --mem-episodic: #3b82f6;
    --mem-semantic: #a855f7;
    --mem-procedural: #10b981;
    --mem-valence: #f43f5e;
    --mem-working: #f59e0b;
    --mem-sensory: #06b6d4;
    --mem-scope: #6366f1;
    --mem-summary: #8b5cf6;
    --mem-topic: #c08532;
  }

  :root[data-theme="dark"] {
    color-scheme: dark;
    --paper: #14120f;
    --surface: #1f1d18;
    --surface-muted: #29261f;
    --surface-soft: #343126;
    --surface-sand: #403b30;
    --border: rgba(242,241,237,0.12);
    --border-strong: rgba(242,241,237,0.26);
    --border-solid: #f2f1ed;
    --ink: rgba(242,241,237,0.82);
    --ink-strong: #fffaf0;
    --muted: rgba(242,241,237,0.62);
    --muted-dim: rgba(242,241,237,0.42);
    --accent: #ff7a1a;
    --accent-strong: #ffb071;
    --accent-soft: rgba(255,122,26,0.18);
    --accent-wash: rgba(255,122,26,0.08);
    --success: #5fd29b;
    --danger: #ff6f91;
    --focus: rgba(255,176,113,0.35);
    --topbar-bg: rgba(20,18,15,0.9);
    --text-on-accent: #1f130a;
    --shadow-card:
      rgba(0,0,0,0.28) 0 10px 26px,
      rgba(242,241,237,0.09) 0 0 0 1px;
    --shadow-soft: rgba(0,0,0,0.22) 0 8px 18px;
    --shadow-inset: transparent 0 0 0;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  html {
    scroll-behavior: smooth;
  }

  body {
    min-height: 100vh;
    background: var(--paper);
    color: var(--ink);
    font-family: var(--font-sans);
    font-size: 16px;
    line-height: 1.5;
    overflow: hidden;
  }

  body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
      radial-gradient(circle at 20% 0%, rgba(208,138,60,0.08), transparent 34%),
      radial-gradient(circle at 82% 18%, rgba(67,209,125,0.06), transparent 30%);
  }

  a {
    color: inherit;
    text-decoration: none;
  }

  button,
  input {
    font: inherit;
  }

  .topbar {
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--topbar-bg);
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }

  .topbar-inner {
    width: min(1380px, calc(100% - 32px));
    margin: 0 auto;
    padding: 14px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }

  .brand {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface);
    box-shadow: var(--shadow-card);
  }

  .brand-copy {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .brand-title {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.05em;
    line-height: 1;
    font-family: var(--font-display);
    color: var(--ink-strong);
  }

  .brand-subtitle {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 600;
  }

  .topbar-meta {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    flex: 1;
    min-width: 0;
  }

  .memory-search {
    position: relative;
    flex: 0 0 280px;
    min-width: 220px;
  }

  .memory-search-hint {
    margin-top: 6px;
    padding-left: 2px;
    color: var(--muted);
    font-size: 11px;
  }

  .memory-search-input {
    min-height: 40px;
    padding-right: 42px;
    font-size: 13px;
  }

  .memory-search-results {
    position: absolute;
    top: calc(100% + 8px);
    left: 0;
    right: 0;
    z-index: 20;
    display: none;
    max-height: 260px;
    overflow-y: auto;
    border: 1px solid var(--border-strong);
    border-radius: 18px;
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
    backdrop-filter: none;
  }

  .memory-search-results.show {
    display: block;
  }

  .nav-stats {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
    min-width: 0;
  }

  .nav-stat {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 82px;
    padding: 7px 10px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
  }

  .nav-stat-label {
    color: var(--muted);
    font-size: 10px;
    line-height: 1.1;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 600;
  }

  .nav-stat-value {
    color: var(--ink-strong);
    font-size: 16px;
    line-height: 1.15;
    letter-spacing: -0.03em;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
    white-space: nowrap;
  }

  .theme-toggle,
  .voice-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 38px;
    padding: 0 12px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface-muted);
    color: var(--ink-strong);
    cursor: pointer;
    box-shadow: var(--shadow-card);
    transition:
      color 0.15s ease,
      border-color 0.15s ease,
      background 0.15s ease,
      transform 0.18s ease;
  }

  .theme-toggle:hover {
    color: var(--accent-strong);
    border-color: var(--border-strong);
    transform: translateY(-1px);
  }

  .voice-toggle.enabled {
    border-color: var(--accent);
    background: var(--accent-soft);
    color: var(--accent-strong);
  }

  .theme-toggle:focus-visible,
  .voice-toggle:focus-visible {
    outline: none;
    box-shadow: 0 0 0 4px var(--focus), var(--shadow-card);
  }

  .theme-toggle-label,
  .voice-toggle-label {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }

  .status-pill.offline {
    border-color: rgba(255, 107, 107, 0.42);
    background: rgba(255, 107, 107, 0.09);
  }

  .status-pill.online {
    border-color: rgba(67, 209, 125, 0.34);
    box-shadow: var(--shadow-card);
  }

  .status-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    flex-shrink: 0;
    background: var(--danger);
    transition: background 0.3s ease, box-shadow 0.3s ease;
  }

  .status-dot.online {
    background: var(--success);
    box-shadow: 0 0 0 4px rgba(67, 209, 125, 0.16);
  }

  .status-copy {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .status-title {
    color: var(--ink-strong);
    font-size: 13px;
    line-height: 1.15;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
  }

  .status-label {
    color: var(--success);
    font-size: 11px;
    line-height: 1.1;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
  }

  .status-pill.offline .status-label {
    color: var(--danger);
  }

  .page {
    width: min(1380px, calc(100% - 32px));
    margin: 0 auto;
    padding: 14px 0 20px;
    height: calc(100vh - 82px);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: 12px;
  }

  .panel-shell {
    position: relative;
    overflow: hidden;
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-card);
  }

  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.08px;
    text-transform: uppercase;
  }

  .eyebrow::before {
    content: "";
    width: 28px;
    height: 1px;
    background: currentColor;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    padding: 6px 12px;
    border: 1px solid var(--border);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
    line-height: 1;
    box-shadow: var(--shadow-soft);
  }

  .badge.green {
    background: rgba(67, 209, 125, 0.16);
    color: var(--success);
  }

  .badge.orange {
    background: var(--accent-soft);
    color: var(--accent-strong);
  }

  .badge {
    background: rgba(232, 238, 248, 0.08);
    color: var(--ink-strong);
  }

  .offline-banner {
    display: none;
    width: 100%;
    padding: 12px 18px;
    border: 1px dashed var(--border-strong);
    border-radius: 999px;
    background: rgba(255, 107, 107, 0.12);
    color: #ffd6d6;
    text-align: center;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: var(--shadow-soft);
  }

  .offline-banner.show {
    display: block;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: minmax(280px, 0.92fr) minmax(360px, 1.15fr) minmax(320px, 1fr);
    grid-template-rows: minmax(0, 0.9fr) minmax(0, 0.78fr) minmax(0, 0.92fr);
    grid-template-areas:
      "log thoughts chat"
      "manage thoughts chat"
      "goals thoughts chat";
    gap: 14px;
    align-items: stretch;
    min-height: 0;
  }

  .workspace-shell {
    display: grid;
    grid-template-columns: 248px minmax(0, 1fr);
    gap: 14px;
    min-height: 0;
  }

  .sidebar-shell {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--shadow-card);
    min-height: 0;
    overflow: hidden;
  }

  .sidebar-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.04em;
    font-family: var(--font-display);
    color: var(--ink-strong);
  }

  .sidebar-copy {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
  }

  .sidebar-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .sidebar-group-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0 2px;
  }

  .tab-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.1) transparent;
  }

  .tab-list::-webkit-scrollbar {
    width: 7px;
  }

  .tab-list::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
  }

  .tab-list::-webkit-scrollbar-track {
    background: transparent;
  }

  .tab-btn {
    width: 100%;
    text-align: left;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-muted);
    color: var(--ink);
    padding: 10px 11px;
    cursor: pointer;
    box-shadow: var(--shadow-card);
    transition:
      border-color 0.18s ease,
      background 0.18s ease,
      box-shadow 0.18s ease,
      color 0.18s ease;
  }

  .tab-btn.active {
    background: linear-gradient(135deg, var(--surface-sand), var(--surface-muted));
    border-color: var(--border-strong);
    box-shadow: var(--shadow-card);
    color: var(--ink-strong);
  }

  .tab-btn:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-soft);
    background: var(--surface-sand);
  }

  .tab-btn-title {
    display: block;
    font-size: 13px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 2px;
    font-family: var(--font-display);
  }

  .tab-btn-copy {
    display: block;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.3;
  }

  .tab-btn.primary {
    padding: 11px 12px;
  }

  .tab-btn.primary .tab-btn-title {
    font-size: 14px;
  }

  .tab-btn.primary.active {
    box-shadow: var(--shadow-card);
  }

  .tab-btn.compact {
    padding: 8px 11px;
    border-radius: 10px;
    box-shadow: none;
    background: var(--surface);
  }

  .tab-btn.compact .tab-btn-copy {
    display: none;
  }

  .tab-btn[data-tab="thoughts"],
  .tab-btn[data-tab="log"],
  .tab-btn[data-tab="settings"],
  .tab-btn[data-tab="files"] {
    opacity: 0.88;
  }

  .content-shell {
    min-height: 0;
  }

  .files-shell {
    background: var(--surface-soft);
  }

  .memory-shell,
  .timeline-shell,
  .inbox-shell,
  .settings-shell {
    background: var(--surface-soft);
  }

  .tab-panel {
    display: none;
    height: 100%;
  }

  .tab-panel.active {
    display: flex;
  }

  .overview-shell {
    background: var(--surface-soft);
  }

  .overview-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }

  .overview-hero {
    display: grid;
    grid-template-columns: 1.35fr 0.95fr;
    gap: 10px;
    margin-bottom: 10px;
  }

  .overview-hero-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: linear-gradient(135deg, var(--surface-soft), var(--surface-muted));
    padding: 14px;
    box-shadow: var(--shadow-card);
  }

  .overview-kicker {
    color: var(--accent-strong);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .overview-headline {
    font-family: var(--font-display);
    font-size: clamp(28px, 3vw, 42px);
    line-height: 1.02;
    letter-spacing: -0.06em;
    color: var(--ink-strong);
    margin-bottom: 10px;
  }

  .overview-summary {
    color: var(--ink);
    font-size: 14px;
    line-height: 1.6;
    max-width: 54ch;
  }

  .overview-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
  }

  .overview-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 7px 11px;
    background: var(--surface);
    color: var(--ink-strong);
    font-size: 12px;
    font-weight: 600;
    box-shadow: var(--shadow-soft);
  }

  .overview-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface-muted);
    padding: 12px;
    box-shadow: var(--shadow-card);
  }

  .overview-card h3 {
    font-size: 14px;
    margin-bottom: 8px;
    line-height: 1.2;
    font-family: var(--font-display);
    letter-spacing: -0.02em;
  }

  .overview-stat {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.05em;
    margin-bottom: 6px;
  }

  .overview-note {
    color: var(--ink);
    font-size: 13px;
    line-height: 1.45;
  }

  .panel-shell {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .goals-shell {
    grid-area: goals;
    background: var(--surface-soft);
  }

  .manage-shell {
    grid-area: manage;
    background: var(--surface-soft);
  }

  .thoughts-shell {
    grid-area: thoughts;
    background: var(--surface-soft);
  }

  .log-shell {
    grid-area: log;
    background: var(--surface);
    border-color: rgba(255, 255, 255, 0.08);
  }

  .chat-shell {
    grid-area: chat;
    background: var(--surface-soft);
  }

  .section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px 0;
    flex-shrink: 0;
    position: relative;
  }

  .section-head::after {
    content: "";
    position: absolute;
    left: 14px;
    right: 14px;
    bottom: -6px;
    height: 1px;
    background: linear-gradient(90deg, var(--border-strong), transparent 82%);
    pointer-events: none;
  }

  .section-title {
    font-size: clamp(1.2rem, 2vw, 1.55rem);
    line-height: 1.12;
    letter-spacing: -0.05em;
    font-weight: 600;
    font-family: var(--font-display);
    color: var(--ink);
  }

  .log-shell .section-title {
    color: #ffffff;
  }

  .panel-body,
  .chat-messages,
  .log-panel-body {
    padding: 12px 14px 14px;
    overflow: auto;
    min-height: 0;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.12) transparent;
  }

  .panel-body::-webkit-scrollbar,
  .chat-messages::-webkit-scrollbar,
  .log-panel-body::-webkit-scrollbar {
    width: 8px;
  }

  .panel-body::-webkit-scrollbar-thumb,
  .chat-messages::-webkit-scrollbar-thumb,
  .log-panel-body::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.12);
    border-radius: 999px;
  }

  #thoughtsPanel,
  #goalsPanel {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .thought,
  .goal {
    border-radius: 16px;
    padding: 14px;
    background: var(--surface);
    box-shadow: var(--shadow-card);
    animation: float-in 0.25s ease;
  }

  @keyframes float-in {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: none;
    }
  }

  .thought {
    border: 1px solid var(--border);
  }

  .thought.reflection {
    background: var(--surface);
  }

  .thought.consolidation {
    background: var(--surface);
  }

  .thought.planning {
    background: var(--surface);
  }

  .thought.exploration {
    background: var(--surface);
  }

  .thought-header,
  .goal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
  }

  .thought-type {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 6px 10px;
    background: rgba(255,255,255,0.06);
    color: var(--ink-strong);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
  }

  .thought-time,
  .goal-progress-label {
    color: var(--muted);
    font-size: 12px;
    font-family: var(--font-mono);
  }

  .thought-summary,
  .goal-desc {
    font-size: 14px;
    line-height: 1.45;
    color: var(--ink-strong);
  }

  .goal {
    border: 1px solid var(--border);
  }

  .goal.has-subgoals {
    border-style: solid;
  }

  .goal-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
  }

  .goal-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
  }

  .goal-edit-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 120px;
    gap: 8px;
    margin-top: 10px;
  }

  .goal-edit-grid.expanded {
    grid-template-columns: 1fr;
  }

  .goal-edit-input {
    min-height: 38px;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    padding: 10px 12px;
    color: var(--ink);
    box-shadow: var(--shadow-card);
  }

  .progress-bar {
    flex: 1;
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,0.08);
  }

  .progress-fill {
    height: 100%;
    border-radius: inherit;
    background: var(--accent);
    transition: width 0.4s ease;
  }

  .subgoals {
    margin-top: 12px;
    padding-left: 12px;
    border-left: 1px solid var(--border);
  }

  .subgoal {
    font-size: 13px;
    line-height: 1.5;
    color: var(--muted);
    padding: 4px 0;
  }

  .log-panel-body {
    margin: 14px 18px 18px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.02);
  }

  .log-line {
    padding: 8px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.78);
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1.6;
    word-break: break-word;
  }

  .log-line:last-child {
    border-bottom: none;
  }

  .log-line.info {
    color: rgba(255, 255, 255, 0.82);
  }

  .log-line.warn {
    color: #f5d26a;
  }

  .log-line.error {
    color: #ffd4d7;
  }

  .log-line.debug {
    color: rgba(255, 255, 255, 0.46);
  }

  .log-line .log-mod {
    color: #9ec5ff;
    font-weight: 700;
  }

  .chat-shell {
    min-height: 0;
  }

  .chat-messages {
    display: flex;
    flex-direction: column;
    gap: 10px;
    flex: 1;
  }

  .msg {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 88%;
  }

  .msg.user {
    align-self: flex-end;
  }

  .msg.brain {
    align-self: flex-start;
  }

  .thinking-dots {
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }
  .thinking-dots span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--ink);
    opacity: 0.3;
    animation: thinking-blink 1.2s infinite;
  }
  .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes thinking-blink {
    0%, 80%, 100% { opacity: 0.3; transform: scale(1); }
    40% { opacity: 1; transform: scale(1.2); }
  }

  .msg-role {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 600;
  }

  .msg-text {
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.55;
  }

  .msg.user .msg-text {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }

  .msg.user .msg-role {
    text-align: right;
    color: var(--muted);
  }

  .msg-meta {
    align-self: flex-end;
    color: rgba(140, 125, 112, 0.92);
    font-family: var(--font-mono);
    font-size: 11px;
  }

  .chat-input-row {
    display: flex;
    gap: 10px;
    padding: 0 14px 16px;
    align-items: stretch;
    flex-shrink: 0;
  }

  .chat-input {
    flex: 1;
    min-height: 42px;
    border-radius: 12px;
    border: 1px solid rgba(118, 118, 118, 0.8);
    background: var(--surface-soft);
    padding: 12px 14px;
    color: var(--ink);
    outline: none;
    box-shadow: var(--shadow-card);
  }

  .chat-input::placeholder {
    color: var(--muted);
  }

  .chat-input:focus {
    border-color: #3b82f6;
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }

  .cta-button,
  .send-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 40px;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: #ffffff;
    color: var(--ink-strong);
    font-size: 16px;
    font-weight: 500;
    letter-spacing: -0.01em;
    cursor: pointer;
    box-shadow: var(--shadow-card);
    transition:
      background-color 0.18s ease,
      color 0.18s ease,
      border-color 0.18s ease,
      box-shadow 0.18s ease;
  }

  .cta-button:hover,
  .send-btn:hover {
    box-shadow: var(--shadow-soft);
  }

  .cta-button:focus-visible,
  .send-btn:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 3px;
  }

  .cta-button.primary,
  .send-btn {
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
    box-shadow: var(--shadow-card), var(--shadow-inset);
  }

  .cta-button.primary:hover,
  .send-btn:hover {
    background: var(--accent-strong);
    border-color: var(--accent-strong);
    color: #ffffff;
  }

  .cta-button.secondary {
    border-color: var(--border-strong);
    background: var(--surface-sand);
    color: var(--ink);
  }

  .cta-button.secondary:hover {
    background: var(--surface-muted);
    color: var(--ink-strong);
  }

  .send-btn:disabled {
    cursor: not-allowed;
    opacity: 0.4;
    transform: none;
    box-shadow: var(--shadow-card);
    background: var(--surface-muted);
    color: var(--muted);
  }

  .empty {
    padding: 18px;
    border-radius: 18px;
    border: 1px dashed var(--border-strong);
    background: var(--surface-soft);
    color: var(--ink);
    text-align: center;
    font-size: 14px;
    line-height: 1.6;
  }

  .log-panel-body .empty {
    background: transparent;
    color: rgba(255, 255, 255, 0.72);
    border-color: rgba(255, 255, 255, 0.16);
  }

  .goal-form {
    display: flex;
    gap: 10px;
    padding: 0 18px 18px;
    align-items: stretch;
    flex-shrink: 0;
  }

  .goal-form .chat-input {
    min-height: 40px;
  }

  .search-result {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    line-height: 1.45;
  }

  .search-result:last-child {
    border-bottom: none;
  }

  .search-result.selected {
    background: var(--surface-muted);
    border-color: var(--border-strong);
  }

  .search-result .search-text {
    color: var(--ink);
  }

  .search-profile {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-muted);
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .search-profile-label,
  .search-meta {
    color: var(--muted);
    font-size: 11px;
  }

  .search-meta {
    display: block;
    margin-top: 4px;
  }

  .search-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }

  .search-score {
    display: inline-block;
    min-width: 40px;
    margin-right: 6px;
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
    font-family: var(--font-mono);
  }

  .manage-stack {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .manage-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface-soft);
    padding: 12px;
    box-shadow: var(--shadow-card);
  }

  .manage-card h3 {
    font-size: 14px;
    line-height: 1.2;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
  }

  .manage-meta,
  .manage-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .manage-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 13px;
    color: var(--muted);
  }

  .manage-row strong {
    color: var(--ink);
  }

  .approval-item,
  .inbox-item {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 9px 10px;
    background: var(--surface-soft);
  }

  .approval-actions,
  .manage-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
  }

  .mini-btn {
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface-sand);
    color: var(--ink-strong);
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: var(--shadow-card);
    transition:
      border-color 0.16s ease,
      box-shadow 0.16s ease,
      background 0.16s ease,
      color 0.16s ease;
  }

  .mini-btn:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-soft);
    background: var(--surface-soft);
  }

  .mini-btn.approve {
    border-color: var(--accent);
    background: var(--accent);
    color: #faf9f5;
  }

  .mini-btn.deny {
    border-color: rgba(239, 68, 68, 0.3);
    color: #fca5a5;
  }

  .mini-btn.danger {
    border-color: rgba(239, 68, 68, 0.4);
    background: rgba(239, 68, 68, 0.1);
    color: #fca5a5;
  }

  .mini-btn.danger:hover {
    background: rgba(239, 68, 68, 0.2);
    border-color: rgba(239, 68, 68, 0.6);
  }

  .mini-btn.warn {
    border-color: rgba(0, 0, 0, 0.12);
    color: var(--ink-strong);
  }

  .manage-note {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.4;
  }

  .files-layout {
    display: grid;
    grid-template-columns: 260px minmax(0, 1fr);
    gap: 10px;
    min-height: 0;
  }

  .file-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .file-btn {
    width: 100%;
    text-align: left;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-soft);
    color: var(--ink);
    padding: 10px;
    cursor: pointer;
    box-shadow: var(--shadow-card);
    transition:
      border-color 0.16s ease,
      box-shadow 0.16s ease,
      background 0.16s ease,
      color 0.16s ease;
  }

  .file-btn.active {
    background: var(--surface-muted);
    border-color: var(--border-strong);
    box-shadow: var(--shadow-card);
  }

  .file-btn:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-soft);
    background: var(--surface);
  }

  .file-btn-title {
    display: block;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .file-btn-copy {
    display: block;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.4;
  }

  .file-editor-shell {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 0;
  }

  .file-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    padding: 9px 10px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
  }

  .file-meta {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.4;
  }

  .file-editor {
    width: 100%;
    min-height: 420px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--surface);
    padding: 12px;
    font: 13px/1.55 var(--font-mono);
    color: var(--ink);
    resize: vertical;
    box-shadow: var(--shadow-card);
  }

  .diff-shell {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 12px;
    box-shadow: var(--shadow-card);
  }

  .diff-pre {
    margin: 0;
    max-height: 240px;
    overflow: auto;
    font: 12px/1.5 var(--font-mono);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .memory-list,
  .timeline-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .memory-toolbar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }

  .memory-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .memory-group-title {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0 2px;
    margin-top: 4px;
  }

  .memory-item,
  .timeline-item {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 10px 12px;
    box-shadow: var(--shadow-card);
  }

  .memory-item h3,
  .timeline-item h3 {
    font-size: 14px;
    margin-bottom: 6px;
    line-height: 1.2;
  }

  .memory-item.clickable,
  .timeline-item.clickable,
  .search-result.clickable {
    cursor: pointer;
    transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
  }

  .memory-item.clickable:hover,
  .timeline-item.clickable:hover,
  .search-result.clickable:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-soft);
    background: var(--surface);
  }

  .memory-profile-card {
    margin-bottom: 10px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .memory-atlas-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(260px, 0.85fr);
    gap: 10px;
    margin-bottom: 10px;
  }

  .memory-atlas-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-soft);
    padding: 12px;
    box-shadow: var(--shadow-card);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .memory-atlas-card h3 {
    font-size: 13px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .memory-graph-shell {
    margin-bottom: 10px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface-soft);
    box-shadow: var(--shadow-card);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .memory-graph-shell.fullscreen {
    position: fixed;
    inset: 18px;
    z-index: 80;
    margin: 0;
    background: var(--surface);
    box-shadow: var(--shadow-card);
  }

  .memory-graph-shell.fullscreen .memory-graph-canvas {
    height: calc(100vh - 190px);
  }

  .memory-graph-shell.fullscreen::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background: rgba(0, 0, 0, 0.38);
  }

  .memory-graph-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  .memory-graph-controls {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .graph-filter-bar {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    padding: 8px 0 4px;
  }

  .graph-filter-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface-muted);
    color: var(--ink);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    letter-spacing: 0.04em;
  }

  .graph-filter-chip::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--chip-color, var(--muted));
    flex-shrink: 0;
  }

  .graph-filter-chip.active {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--ink-strong);
  }

  .graph-filter-chip[data-type="episodic"] { --chip-color: var(--mem-episodic); }
  .graph-filter-chip[data-type="semantic"]  { --chip-color: var(--mem-semantic); }
  .graph-filter-chip[data-type="procedural"]{ --chip-color: var(--mem-procedural); }
  .graph-filter-chip[data-type="valence"]   { --chip-color: var(--mem-valence); }
  .graph-filter-chip[data-type="working"]   { --chip-color: var(--mem-working); }
  .graph-filter-chip[data-type="sensory"]   { --chip-color: var(--mem-sensory); }
  .graph-filter-chip[data-type="scope"]     { --chip-color: var(--mem-scope); }
  .graph-filter-chip[data-type="summary"]   { --chip-color: var(--mem-summary); }

  .memory-graph-canvas {
    width: 100%;
    height: min(54vh, 430px);
    border-radius: 10px;
    border: 1px solid var(--border);
    background:
      radial-gradient(circle at 50% 50%, var(--accent-wash), transparent 42%),
      var(--surface);
    overflow: hidden;
    position: relative;
  }

  #d3GraphSvg {
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
  }

  #d3GraphSvg:active {
    cursor: grabbing;
  }

  .graph-node-label {
    font: 10px/1.2 var(--font-sans);
    fill: var(--ink-strong);
    pointer-events: none;
    text-shadow: none;
  }

  .graph-node.cluster circle {
    stroke-width: 2.5px;
  }

  .d3-tooltip {
    position: absolute;
    background: var(--surface-soft);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--ink-strong);
    pointer-events: none;
    max-width: 260px;
    box-shadow: var(--shadow-soft);
    z-index: 50;
    display: none;
  }

  .graph-legend {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    padding-top: 4px;
    font-size: 11px;
    color: var(--muted);
  }

  .graph-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .graph-legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
  }

  .memory-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
  }

  .memory-stat {
    padding: 9px 10px;
    border-radius: 10px;
    background: var(--surface-muted);
    border: 1px solid var(--border);
  }

  .memory-stat strong {
    display: block;
    font-size: 20px;
    line-height: 1.1;
    color: var(--ink-strong);
    margin-bottom: 3px;
  }

  .memory-stat span {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .memory-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .memory-profile-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
  }

  .memory-profile-block h3 {
    font-size: 12px;
    margin-bottom: 6px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .memory-profile-block ul {
    padding-left: 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
  }

  .fact-sources {
    display: block;
    margin-top: 2px;
    color: var(--muted);
    font-size: 11px;
    font-family: var(--font-mono);
  }

  .memory-tag-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .memory-tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface);
    font-size: 12px;
  }

  .memory-card-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }

  .memory-card-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 8px;
  }

  .memory-storage-line {
    margin-bottom: 8px;
    color: var(--muted);
    font-size: 12px;
  }

  .memory-card-meta .badge {
    box-shadow: none;
  }

  .memory-inline-actions {
    margin-top: 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .drawer-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .drawer-minimap {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
  }

  .drawer-minimap-item {
    padding: 10px 12px;
    border-radius: 16px;
    background: var(--surface-muted);
    border: 1px solid var(--border);
  }

  .drawer-minimap-item strong {
    display: block;
    font-size: 15px;
    line-height: 1.1;
    margin-bottom: 4px;
    color: var(--ink-strong);
  }

  .drawer-minimap-item span {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .drawer-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .drawer-memory-link {
    width: 100%;
    text-align: left;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface-soft);
    padding: 10px 12px;
    cursor: pointer;
    transition:
      transform 0.16s ease,
      border-color 0.16s ease,
      box-shadow 0.16s ease,
      background 0.16s ease;
  }

  .drawer-memory-link.active {
    border-color: var(--border-strong);
    background: var(--surface-muted);
    box-shadow: var(--shadow-card);
  }

  .drawer-memory-link:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-soft);
    background: var(--surface);
  }

  .drawer-memory-link strong {
    display: block;
    font-size: 13px;
    margin-bottom: 4px;
  }

  .timeline-kind {
    display: inline-flex;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.04);
    color: var(--ink-strong);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .settings-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }

  .settings-code {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.04);
    font: 12px/1.4 var(--font-mono);
    color: var(--ink-strong);
    word-break: break-all;
  }

  .settings-link {
    color: var(--accent);
    text-decoration: underline;
    text-underline-offset: 3px;
  }

  .drawer-backdrop {
    position: fixed;
    inset: 0;
    display: none;
    background: rgba(17, 17, 17, 0.18);
    z-index: 40;
  }

  .drawer-backdrop.show {
    display: block;
  }

  .detail-drawer {
    position: fixed;
    top: 0;
    right: 0;
    width: min(460px, 100vw);
    height: 100vh;
    display: none;
    flex-direction: column;
    gap: 12px;
    padding: 20px;
    background: var(--surface);
    border-left: 1px solid var(--border);
    box-shadow: var(--shadow-card);
    z-index: 50;
  }

  .detail-drawer.show {
    display: flex;
  }

  .drawer-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
  }

  .drawer-title {
    font-family: var(--font-display);
    font-size: 22px;
    line-height: 1.05;
    letter-spacing: -0.05em;
  }

  .drawer-body {
    overflow: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding-right: 4px;
  }

  .drawer-pre {
    margin: 0;
    border-radius: 16px;
    border: 1px solid var(--border);
    background: var(--surface-soft);
    padding: 12px;
    font: 12px/1.5 var(--font-mono);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .toast {
    position: fixed;
    left: 50%;
    bottom: 24px;
    transform: translateX(-50%);
    z-index: 80;
    padding: 10px 14px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(17, 17, 17, 0.92);
    color: white;
    box-shadow: var(--shadow-card);
    opacity: 0;
    pointer-events: none;
    transition: opacity 140ms ease, transform 140ms ease;
  }

  .toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(-2px);
  }

  @media (max-width: 1180px) {
    body {
      overflow: auto;
    }

    .page {
      height: auto;
      min-height: calc(100vh - 86px);
    }

    .dashboard-grid {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      grid-template-rows: minmax(0, 0.7fr) minmax(0, 0.7fr) minmax(0, 0.95fr) minmax(0, 0.95fr);
      grid-template-areas:
        "log thoughts"
        "manage thoughts"
        "goals thoughts"
        "chat chat";
    }

  }

  @media (max-width: 900px) {
    .workspace-shell {
      grid-template-columns: 1fr;
      gap: 12px;
    }

    .sidebar-shell {
      border-radius: 14px;
    }

    .tab-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .page {
      height: auto;
      min-height: auto;
    }
  }

  @media (max-width: 767px) {
    body {
      overflow: auto;
    }

    .topbar-inner,
    .page,
    .offline-banner {
      width: calc(100% - 20px);
    }

    .topbar-inner {
      padding: 14px 0;
      flex-direction: column;
      align-items: stretch;
    }

    .brand,
    .status-pill {
      width: 100%;
      justify-content: center;
    }

    .topbar-meta,
    .nav-stats {
      justify-content: stretch;
    }

    .topbar-meta {
      flex-direction: column;
      align-items: stretch;
    }

    .memory-search {
      flex: 1 1 auto;
      width: 100%;
      min-width: 0;
    }

    .tab-list {
      grid-template-columns: 1fr;
    }

    .overview-grid {
      grid-template-columns: 1fr;
    }

    .overview-hero {
      grid-template-columns: 1fr;
    }

    .nav-stats {
      width: 100%;
    }

    .nav-stat {
      flex: 1 1 120px;
    }

    .panel-shell {
      border-radius: 14px;
    }

    .chat-input-row {
      flex-direction: column;
    }

    .goal-form {
      flex-direction: column;
    }

    .cta-button,
    .send-btn {
      width: 100%;
    }

    .section-head,
    .panel-body,
    .chat-messages,
    .chat-input-row {
      padding-left: 18px;
      padding-right: 18px;
    }

    .log-panel-body {
      margin-left: 18px;
      margin-right: 18px;
    }

    .msg {
      max-width: 100%;
    }
  }
</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <div class="brand-copy">
        <div class="brand-title">Mnemon</div>
        <div class="brand-subtitle">Jarvis · Control Room</div>
      </div>
    </div>
    <div class="topbar-meta">
      <div class="memory-search">
        <input
          class="chat-input memory-search-input"
          id="memSearchInput"
          type="text"
          placeholder="Search memory…"
          autocomplete="off"
        />
        <div class="memory-search-results" id="memSearchResults" tabindex="0"></div>
        <div class="memory-search-hint">
          ↑/↓ move • Enter open • j/k when results focused • Esc clear
        </div>
      </div>
      <div class="nav-stats">
        <div class="nav-stat">
          <span class="nav-stat-label">Uptime</span>
          <span class="nav-stat-value" id="navUptime">--</span>
        </div>
        <div class="nav-stat">
          <span class="nav-stat-label">Cycles</span>
          <span class="nav-stat-value" id="navCycles">--</span>
        </div>
        <div class="nav-stat">
          <span class="nav-stat-label">Idle ticks</span>
          <span class="nav-stat-value" id="navTicks">--</span>
        </div>
        <div class="nav-stat">
          <span class="nav-stat-label">Autonomy</span>
          <span class="nav-stat-value" id="navAutonomy">--</span>
        </div>
      </div>
      <button class="theme-toggle" id="themeToggle" type="button" aria-label="Switch theme">
        <span id="themeIcon">☼</span>
        <span class="theme-toggle-label" id="themeLabel">Light</span>
      </button>
      <button class="voice-toggle" id="voiceToggle" type="button" aria-label="Speak new thoughts">
        <span id="voiceIcon">♪</span>
        <span class="voice-toggle-label" id="voiceLabel">Voice off</span>
      </button>
      <div class="status-pill offline" id="statusPill">
        <div class="status-dot" id="statusDotNav"></div>
        <div class="status-copy">
          <span class="status-title">Daemon</span>
          <span class="status-label" id="statusLabel">connecting</span>
        </div>
      </div>
    </div>
  </div>
</header>

<main class="page">
  <div class="offline-banner" id="offlineBanner">
    Daemon offline. The dashboard is retrying the event stream.
  </div>

  <section class="workspace-shell">
    <aside class="sidebar-shell">
      <div>
        <div class="sidebar-title">Jarvis workspace</div>
        <div class="sidebar-copy">
          Switch areas instead of cramming everything into one screen.
        </div>
      </div>

      <div class="tab-list" id="tabList">
        <div class="sidebar-group primary">
          <div class="sidebar-group-label">Primary</div>
          <button
            class="tab-btn primary active"
            data-tab="memory"
            type="button"
            aria-selected="true"
            onclick="setActiveTab('memory')"
          >
            <span class="tab-btn-title">Memory</span>
            <span class="tab-btn-copy">Where memories live, connect, and compress</span>
          </button>
          <button
            class="tab-btn primary"
            data-tab="graph"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('graph')"
          >
            <span class="tab-btn-title">Graph</span>
            <span class="tab-btn-copy">Relationships between summaries, sources, and storage</span>
          </button>
          <button
            class="tab-btn primary"
            data-tab="chat"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('chat')"
          >
            <span class="tab-btn-title">Chat</span>
            <span class="tab-btn-copy">Talk to Jarvis with memory in context</span>
          </button>
          <button
            class="tab-btn primary"
            data-tab="goals"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('goals')"
          >
            <span class="tab-btn-title">Goals</span>
            <span class="tab-btn-copy">Current work, priorities, and progress</span>
          </button>
        </div>

        <div class="sidebar-group">
          <div class="sidebar-group-label">Workspace</div>
          <button
            class="tab-btn"
            data-tab="timeline"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('timeline')"
          >
            <span class="tab-btn-title">Timeline</span>
            <span class="tab-btn-copy">Recent thoughts, inbox, approvals, memory</span>
          </button>
          <button
            class="tab-btn"
            data-tab="goals"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('goals')"
          >
            <span class="tab-btn-title">Goals</span>
            <span class="tab-btn-copy">Current work, priorities, and progress</span>
          </button>
          <button
            class="tab-btn"
            data-tab="inbox"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('inbox')"
          >
            <span class="tab-btn-title">Inbox & History</span>
            <span class="tab-btn-copy">Inbox plus recent chat</span>
          </button>
          <button
            class="tab-btn"
            data-tab="manage"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('manage')"
          >
            <span class="tab-btn-title">Control Center</span>
            <span class="tab-btn-copy">Approvals, inbox, bot actions</span>
          </button>
        </div>

        <div class="sidebar-group">
          <div class="sidebar-group-label">Utilities</div>
          <button
            class="tab-btn compact"
            data-tab="overview"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('overview')"
          >
            <span class="tab-btn-title">Overview</span>
          </button>
          <button
            class="tab-btn compact"
            data-tab="thoughts"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('thoughts')"
          >
            <span class="tab-btn-title">Thoughts</span>
          </button>
          <button
            class="tab-btn compact"
            data-tab="log"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('log')"
          >
            <span class="tab-btn-title">Logs</span>
          </button>
          <button
            class="tab-btn compact"
            data-tab="files"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('files')"
          >
            <span class="tab-btn-title">Files</span>
          </button>
          <button
            class="tab-btn compact"
            data-tab="settings"
            type="button"
            aria-selected="false"
            onclick="setActiveTab('settings')"
          >
            <span class="tab-btn-title">Settings</span>
          </button>
        </div>
      </div>
    </aside>

    <section class="content-shell">
    <article
      class="panel-shell overview-shell tab-panel"
      data-panel="overview"
      hidden
      aria-hidden="true"
    >
      <div class="section-head">
        <h2 class="section-title">Overview</h2>
        <span class="badge green" id="overviewState">live</span>
      </div>
      <div class="panel-body" id="overviewPanel">
        <div class="empty">Waiting for daemon overview…</div>
      </div>
    </article>

    <article
      class="panel-shell thoughts-shell tab-panel"
      data-panel="thoughts"
      id="thoughtsSection"
      hidden
      aria-hidden="true"
    >
      <div class="section-head">
        <h2 class="section-title">Idle Thoughts</h2>
        <span id="tickCount" class="badge green">0 ticks</span>
      </div>
      <div class="panel-body" id="thoughtsPanel">
        <div class="empty">Waiting for the first thought loop…</div>
      </div>
    </article>

    <article class="panel-shell goals-shell tab-panel" data-panel="goals" hidden aria-hidden="true">
      <div class="section-head">
        <h2 class="section-title">Goals</h2>
        <span id="goalCount" class="badge orange">0 goals</span>
      </div>
      <div class="panel-body" id="goalsPanel">
        <div class="empty">No active goals.</div>
      </div>
      <form class="goal-form" id="goalForm">
        <input
          class="chat-input"
          id="goalInput"
          type="text"
          placeholder="Add a new goal…"
          autocomplete="off"
        />
        <button class="send-btn" id="goalSubmitBtn" type="submit" disabled>Add</button>
      </form>
    </article>

    <article
      class="panel-shell manage-shell tab-panel"
      data-panel="manage"
      hidden
      aria-hidden="true"
    >
      <div class="section-head">
        <h2 class="section-title">Control Center</h2>
        <span id="pendingCount" class="badge">0 pending</span>
      </div>
      <div class="panel-body" id="managePanel">
        <div class="empty">Waiting for daemon management state…</div>
      </div>
    </article>

    <article class="panel-shell log-shell tab-panel" data-panel="log" hidden aria-hidden="true">
      <div class="section-head">
        <h2 class="section-title">Live Log</h2>
        <span class="badge green">tail</span>
      </div>
      <div class="log-panel-body" id="logPanel">
        <div class="empty">Loading log tail…</div>
      </div>
    </article>

    <article class="panel-shell chat-shell tab-panel" data-panel="chat" hidden aria-hidden="true">
      <div class="section-head">
        <h2 class="section-title">Chat with Jarvis</h2>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="empty">Say something to Jarvis when the daemon comes online.</div>
      </div>
      <div class="chat-input-row" id="chatComposer">
        <input
          class="chat-input"
          id="chatInput"
          type="text"
          placeholder="Ask Jarvis anything…"
          autocomplete="off"
        />
        <button class="send-btn" id="sendBtn" disabled>Send message</button>
      </div>
    </article>

    <article class="panel-shell files-shell tab-panel" data-panel="files" hidden aria-hidden="true">
      <div class="section-head">
        <h2 class="section-title">Identity & State Files</h2>
        <span id="fileDirtyBadge" class="badge">saved</span>
      </div>
      <div class="panel-body" id="filesPanel">
        <div class="files-layout">
          <div class="file-list" id="fileList">
            <div class="empty">Loading editable files…</div>
          </div>
          <div class="file-editor-shell">
            <div class="file-toolbar">
              <div class="file-meta" id="fileMeta">Choose a file to inspect and edit.</div>
              <div class="manage-actions">
                <button class="mini-btn" id="fileReloadBtn" type="button">Reload</button>
                <button class="mini-btn" id="fileDiffBtn" type="button">Preview diff</button>
                <button class="mini-btn approve" id="fileSaveBtn" type="button">Save</button>
              </div>
            </div>
            <textarea
              class="file-editor"
              id="fileEditor"
              spellcheck="false"
              placeholder="Select a managed file…"
              disabled
            ></textarea>
            <div class="diff-shell">
              <div class="file-meta">Diff preview</div>
              <pre class="diff-pre" id="fileDiffPreview">No diff yet.</pre>
            </div>
          </div>
        </div>
      </div>
    </article>
    <article
      class="panel-shell memory-shell tab-panel active"
      data-panel="memory"
      aria-hidden="false"
    >
      <div class="section-head">
        <h2 class="section-title">Memory Browser</h2>
        <span class="badge" id="memoryCountBadge">0 items</span>
      </div>
      <div class="panel-body" id="memoryPanel">
        <div class="empty">Loading recent memories…</div>
      </div>
    </article>
    <article
      class="panel-shell memory-shell tab-panel"
      data-panel="graph"
      hidden
      aria-hidden="true"
    >
      <div class="section-head">
        <h2 class="section-title">Memory Graph</h2>
        <span class="badge" id="graphCountBadge">0 nodes</span>
      </div>
      <div class="panel-body" id="graphPanel" style="padding:12px;overflow:hidden">
        <div class="empty">Loading memory graph…</div>
      </div>
    </article>
    <article
      class="panel-shell timeline-shell tab-panel"
      data-panel="timeline"
      hidden
      aria-hidden="true"
    >
      <div class="section-head">
        <h2 class="section-title">Timeline</h2>
        <span class="badge" id="timelineCountBadge">0 events</span>
      </div>
      <div class="panel-body" id="timelinePanel">
        <div class="empty">Loading recent timeline…</div>
      </div>
    </article>
    <article
      class="panel-shell settings-shell tab-panel"
      data-panel="settings"
      hidden
      aria-hidden="true"
    >
      <div class="section-head">
        <h2 class="section-title">Settings & Channels</h2>
        <span class="badge" id="settingsBadge">ready</span>
      </div>
      <div class="panel-body" id="settingsPanel">
        <div class="empty">Loading settings…</div>
      </div>
    </article>
    <article
      class="panel-shell inbox-shell tab-panel"
      data-panel="inbox"
      hidden
      aria-hidden="true"
    >
      <div class="section-head">
        <h2 class="section-title">Inbox & History</h2>
        <span class="badge" id="inboxCountBadge">0 items</span>
      </div>
      <div class="panel-body" id="inboxPanel">
        <div class="empty">Loading inbox and history…</div>
      </div>
    </article>
    </section>
  </section>
</main>

<div class="drawer-backdrop" id="drawerBackdrop"></div>
<aside class="detail-drawer" id="approvalDrawer" aria-hidden="true">
  <div class="drawer-header">
    <div>
      <div class="drawer-title">Approval details</div>
      <div class="manage-note" id="approvalDrawerMeta">Select a pending action.</div>
    </div>
    <button class="mini-btn" id="approvalDrawerClose" type="button">Close</button>
  </div>
  <div class="drawer-body">
    <div class="manage-card">
      <h3 id="approvalDrawerTitle">No action selected</h3>
      <div class="manage-row"><span>Risk</span><strong id="approvalDrawerRisk">—</strong></div>
      <div class="manage-row"><span>Source</span><strong id="approvalDrawerSource">—</strong></div>
      <div class="manage-row"><span>Proposed</span><strong id="approvalDrawerTime">—</strong></div>
      <div class="approval-actions">
        <button class="mini-btn approve" id="approvalDrawerApprove" type="button">Approve</button>
        <button class="mini-btn deny" id="approvalDrawerDeny" type="button">Deny</button>
      </div>
    </div>
    <div class="manage-card">
      <h3>Context</h3>
      <pre class="drawer-pre" id="approvalDrawerContext">No context available.</pre>
    </div>
  </div>
</aside>
<div class="drawer-backdrop" id="memoryDrawerBackdrop"></div>
<aside class="detail-drawer" id="memoryDrawer" aria-hidden="true">
  <div class="drawer-header">
    <div>
      <div class="drawer-title">Memory details</div>
      <div class="brand-subtitle" id="memoryDrawerMeta">Select a memory to inspect.</div>
    </div>
    <button class="mini-btn" id="memoryDrawerClose" type="button">Close</button>
  </div>
  <div class="drawer-body" id="memoryDrawerBody">
    <div class="empty">Select a memory from search, the browser, or the timeline.</div>
  </div>
</aside>
<div class="toast" id="toast" aria-live="polite" aria-atomic="true"></div>

<script>
const $ = id => document.getElementById(id);

let source;
let online = false;
let activeTab = 'memory';
let currentManagedFile = '';
let currentManagedFileOriginal = '';
let selectedApproval = null;
let selectedMemoryId = null;
let memoryFilter = 'all';
let memoryGroup = 'day';
let timelineGroup = 'day';
let telegramInfo = null;
let currentMemoryGraph = null;
let lastStatusAt = 0;
let reconnectTimer = null;
let voiceEnabled = localStorage.getItem('mnemon-voice') === 'on';
let voiceHydrated = false;
let lastSpokenThoughtKey = '';
// graphTransform and graphDrag are now handled by D3 zoom internally
const renderCache = {
  overview: '',
  thoughts: '',
  goals: '',
  management: '',
  memory: '',
  graph: '',
  timeline: '',
  inbox: '',
  settings: '',
};

function currentTheme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

function applyTheme(theme) {
  const normalized = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = normalized;
  localStorage.setItem('mnemon-theme', normalized);
  const label = $('themeLabel');
  const icon = $('themeIcon');
  if (label) label.textContent = normalized === 'dark' ? 'Dark' : 'Light';
  if (icon) icon.textContent = normalized === 'dark' ? '☾' : '☼';
  const toggle = $('themeToggle');
  if (toggle) {
    const nextTheme = normalized === 'dark' ? 'light' : 'dark';
    toggle.setAttribute('aria-label', `Switch to ${nextTheme} theme`);
  }
}

function toggleTheme() {
  applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
}

function applyVoice(enabled) {
  voiceEnabled = Boolean(enabled);
  localStorage.setItem('mnemon-voice', voiceEnabled ? 'on' : 'off');
  const toggle = $('voiceToggle');
  const label = $('voiceLabel');
  const icon = $('voiceIcon');
  if (toggle) {
    toggle.classList.toggle('enabled', voiceEnabled);
    toggle.setAttribute(
      'aria-label',
      voiceEnabled ? 'Stop speaking new thoughts' : 'Speak new thoughts',
    );
  }
  if (label) label.textContent = voiceEnabled ? 'Voice on' : 'Voice off';
  if (icon) icon.textContent = voiceEnabled ? '♫' : '♪';
  if (!voiceEnabled && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
}

function toggleVoice() {
  applyVoice(!voiceEnabled);
}

function speakThought(text, activity = 'thought') {
  if (!voiceEnabled || !text || !('speechSynthesis' in window)) return;
  const utterance = new SpeechSynthesisUtterance(
    `Jarvis ${formatActivity(activity)}. ${text}`,
  );
  utterance.rate = 0.96;
  utterance.pitch = 0.92;
  utterance.volume = 0.85;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function maybeSpeakNewestThought(thoughts) {
  if (!thoughts.length) return;
  const newest = thoughts[thoughts.length - 1];
  const key = `${newest.timestamp || ''}:${newest.activity || ''}:${newest.summary || ''}`;
  if (!voiceHydrated) {
    lastSpokenThoughtKey = key;
    voiceHydrated = true;
    return;
  }
  if (key === lastSpokenThoughtKey) return;
  lastSpokenThoughtKey = key;
  speakThought(String(newest.summary || ''), newest.activity || 'thought');
}

function setActiveTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(button => {
    const isActive = button.dataset.tab === tab;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });
  document.querySelectorAll('.tab-panel').forEach(panel => {
    const isActive = panel.dataset.panel === tab;
    panel.classList.toggle('active', isActive);
    panel.hidden = !isActive;
    panel.setAttribute('aria-hidden', isActive ? 'false' : 'true');
    panel.style.display = isActive ? 'flex' : 'none';
  });
  if (tab === 'files') loadManagedFileList();
  if (tab === 'memory') refreshMemoryBrowser();
  if (tab === 'graph') refreshMemoryGraph();
  if (tab === 'timeline') refreshTimeline();
  if (tab === 'inbox') renderInboxHistory(window.__lastStatusPayload || {});
  if (tab === 'settings') refreshTelegramInfo();
}

window.setActiveTab = setActiveTab;

const tabList = $('tabList');
if (tabList) {
  tabList.addEventListener('click', event => {
    const button = event.target.closest('.tab-btn');
    if (!button) return;
    const tab = button.dataset.tab;
    if (!tab) return;
    setActiveTab(tab);
  });
}

setActiveTab(activeTab);
applyTheme(currentTheme());
applyVoice(voiceEnabled);

const themeToggle = $('themeToggle');
if (themeToggle) {
  themeToggle.addEventListener('click', toggleTheme);
}

const voiceToggle = $('voiceToggle');
if (voiceToggle) {
  voiceToggle.addEventListener('click', toggleVoice);
}

async function fetchJson(url, options = undefined) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || `Request failed for ${url}`);
  }
  return data;
}

function connect() {
  if (source) source.close();
  source = new EventSource('/events');
  scheduleConnectionFallback();

  source.addEventListener('status', e => {
    const data = JSON.parse(e.data);
    const daemonOnline = !(data.error || data.connection_error);
    setOnline(daemonOnline, data.error || data.connection_error || '');
    renderStatus(data);
  });

  source.addEventListener('thoughts', e => {
    const thoughts = JSON.parse(e.data);
    renderThoughts(thoughts);
  });

  source.addEventListener('goals', e => {
    const goals = JSON.parse(e.data);
    renderGoals(goals);
  });

  source.addEventListener('log', e => {
    appendLog(e.data);
  });

  source.addEventListener('error', () => {
    setOnline(false, 'Event stream disconnected. Retrying…');
    source.close();
    setTimeout(connect, 3000);
  });
}

connect();

function scheduleConnectionFallback() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    if (!lastStatusAt) {
      setOnline(false, 'Waiting for daemon status. Retrying event stream…');
      renderOfflinePanels('Waiting for daemon status. The browser is connected to the web UI.');
    }
  }, 2500);
}

function setOnline(v, detail = '') {
  online = v;
  if (v) {
    lastStatusAt = Date.now();
    clearTimeout(reconnectTimer);
  }
  $('statusDotNav').className = 'status-dot' + (v ? ' online' : '');
  $('statusPill').className = 'status-pill ' + (v ? 'online' : 'offline');
  $('statusLabel').textContent = v ? 'online and streaming' : 'reconnecting';
  $('offlineBanner').className = 'offline-banner' + (v ? '' : ' show');
  $('offlineBanner').textContent = detail
    ? `Daemon offline. ${detail}`
    : 'Daemon offline. The dashboard is retrying the event stream.';
  $('sendBtn').disabled = !v;
  if ($('goalSubmitBtn')) $('goalSubmitBtn').disabled = !v;
}

function renderOfflinePanels(detail = '') {
  const reason = detail || 'The daemon socket is not responding yet.';
  const offlineHtml = `<div class="overview-hero">
      <div class="overview-hero-card">
        <div class="overview-kicker">Daemon reconnecting</div>
        <div class="overview-headline">Web UI is up. Jarvis is not reachable yet.</div>
        <div class="overview-summary">
          ${escHtml(reason)} Start or restart the daemon, then leave this page open; it retries
          automatically every few seconds.
        </div>
        <div class="overview-actions">
          <span class="overview-pill">Expected socket: ~/.mnemon/daemon.sock</span>
          <span class="overview-pill">Retrying event stream</span>
        </div>
      </div>
      <div class="overview-hero-card">
        <div class="overview-kicker">Recovery</div>
        <div class="overview-summary">
          Run <strong>mnemon-daemon start</strong> or check the daemon process/logs if it should
          already be online.
        </div>
      </div>
    </div>`;
  setPanelHTML($('overviewPanel'), offlineHtml);

  const loadingPanels = [
    ['memoryPanel', 'Recent memories will appear when the daemon reconnects.'],
    ['graphPanel', 'The memory graph needs daemon access before it can render.'],
    ['timelinePanel', 'Timeline events are waiting on daemon status.'],
    ['settingsPanel', 'Settings and channel health load after reconnection.'],
    ['inboxPanel', 'Inbox and chat history load after reconnection.'],
    ['managePanel', 'Approvals and controls load after reconnection.'],
  ];
  loadingPanels.forEach(([id, message]) => {
    const panel = $(id);
    if (!panel) return;
    if (!panel.textContent.toLowerCase().includes('loading')
        && !panel.textContent.toLowerCase().includes('waiting')) {
      return;
    }
    setPanelHTML(panel, `<div class="empty">${message}</div>`);
  });
}

function renderStatus(s) {
  window.__lastStatusPayload = s;
  const d = s.daemon || {};
  if (s.error || s.connection_error) {
    renderOfflinePanels(s.error || s.connection_error);
  }
  const started = d.started_at ? new Date(d.started_at) : null;
  const uptime = started ? formatDuration((Date.now() - started) / 1000) : '?';
  $('navUptime').textContent = uptime;
  $('navCycles').textContent = d.total_cycles ?? '?';
  $('navTicks').textContent = d.total_idle_ticks ?? '?';
  $('navAutonomy').textContent = d.autonomy_level ?? '?';

  const idleTicks = d.total_idle_ticks ?? 0;
  $('tickCount').textContent = `${idleTicks} tick${idleTicks === 1 ? '' : 's'}`;
  $('overviewState').textContent = online ? 'live' : 'syncing';
  renderOverview(s);
  renderManagement(s);
  renderSettings(s);
  renderInboxHistory(s);
}

function renderOverview(status) {
  const d = status.daemon || {};
  if (status.error || status.connection_error) {
    renderOfflinePanels(status.error || status.connection_error);
    renderCache.overview = JSON.stringify({ offline: status.error || status.connection_error });
    return;
  }
  const pending = status.pending_approvals || [];
  const goals = (status.brain && status.brain.active_goals) || [];
  const telegram = (status.channels && status.channels.telegram) || {};
  const unreadInbox = (status.proactive_inbox || []).filter(item => !item.read).length;
  const overviewState = online ? 'Daemon online' : 'Reconnecting';
  const summaryLine = [
    `${pending.length} approvals waiting`,
    `${goals.length} active goals`,
    `${unreadInbox} unread inbox`,
  ].join(' · ');

  const payload = JSON.stringify({ d, pending, goals, telegram });
  if (payload === renderCache.overview) return;
  renderCache.overview = payload;

  const html = `<div class="overview-hero">
      <div class="overview-hero-card">
        <div class="overview-kicker">${overviewState}</div>
        <div class="overview-headline">Operate Jarvis from one calm control surface.</div>
        <div class="overview-summary">
          Review state, approve actions, manage memory, inspect history, and edit identity files
          without leaving the workspace.
        </div>
        <div class="overview-actions">
          <span class="overview-pill">${summaryLine}</span>
          <span class="overview-pill">Web UI live at http://localhost:7777</span>
        </div>
      </div>
      <div class="overview-hero-card">
        <div class="overview-kicker">Operator checklist</div>
        <div class="overview-summary">
          Start with Control Center for approvals, then review Goals, Inbox & History, and
          Settings for channel/runtime health.
        </div>
      </div>
    </div>
    <div class="overview-grid">
      <div class="overview-card">
        <h3>Daemon</h3>
        <div class="overview-stat">${escHtml(String(d.total_cycles ?? 0))}</div>
        <div class="overview-note">
          cycles completed · autonomy ${escHtml(String(d.autonomy_level || '?'))}
        </div>
      </div>
      <div class="overview-card">
        <h3>Approvals</h3>
        <div class="overview-stat">${pending.length}</div>
        <div class="overview-note">items waiting for review in the control center</div>
      </div>
      <div class="overview-card">
        <h3>Goals</h3>
        <div class="overview-stat">${goals.length}</div>
        <div class="overview-note">active goals currently shaping Jarvis behavior</div>
      </div>
      <div class="overview-card">
        <h3>Telegram</h3>
        <div class="overview-stat">${telegram.paired ? 'Paired' : 'Idle'}</div>
        <div class="overview-note">
          ${telegram.configured ? 'bot configured' : 'token missing'} ·
          ${telegram.paired ? 'mobile channel ready' : 'send /start to pair'}
        </div>
      </div>
    </div>`;

  setPanelHTML($('overviewPanel'), html);
}

function renderSettings(status) {
  const daemon = status.daemon || {};
  const telegram = (status.channels && status.channels.telegram) || {};
  const config = status.config || {};
  const configured = telegram.configured ? 'Yes' : 'No';
  const paired = telegram.paired ? 'Yes' : 'No';
  const chatId = telegram.chat_id ? escHtml(String(telegram.chat_id)) : '—';
  const pollInterval = `${telegram.poll_interval_s || '?'}s`;
  const localWebUi = `http://localhost:${config.webui_port || 7777}`;
  const lanWebUi = `http://${window.location.hostname}:${config.webui_port || 7777}`;
  const settingsState = telegram.configured && telegram.paired ? 'ready' : 'attention';
  const webUiEnabled = config.webui_enabled ? 'Yes' : 'No';
  const socketPath = escHtml(String(config.socket_path || '—'));
  const logPath = escHtml(String(config.log_path || '—'));
  const statePath = escHtml(String(config.state_path || '—'));
  const gitJournal = config.git_journal_enabled ? 'Enabled' : 'Disabled';
  const lastInteraction = (
    daemon.last_user_interaction && daemon.last_user_interaction !== 'None'
      ? timeAgo(daemon.last_user_interaction)
      : 'never'
  );
  const lastConsolidation = (
    daemon.last_consolidation && daemon.last_consolidation !== 'None'
      ? timeAgo(daemon.last_consolidation)
      : 'never'
  );
  const autonomyButtons = ['passive', 'suggest', 'semi_auto', 'autonomous']
    .map(level => {
      const tone = daemon.autonomy_level === level ? 'approve' : '';
      return `<button class="mini-btn ${tone}"
        data-action="autonomy"
        data-level="${level}">
        ${escHtml(level)}
      </button>`;
    })
    .join('');
  const payload = JSON.stringify({ daemon, telegram, config });
  if (payload === renderCache.settings) return;
  renderCache.settings = payload;
  $('settingsBadge').textContent = settingsState;
  const telegramActions = [];
  const telegramInfoRows = [];

  if (telegramInfo && telegramInfo.ok) {
    const botName = escHtml(telegramInfo.display_name || telegramInfo.username || 'Unknown bot');
    telegramInfoRows.push(
      `<div class="manage-row"><span>Bot</span><strong>${botName}</strong></div>`
    );
    telegramInfoRows.push(
      `<div class="manage-row">
        <span>Username</span>
        <strong>${escHtml(telegramInfo.username || '—')}</strong>
      </div>`
    );
    if (telegramInfo.bot_url) {
      telegramActions.push(
        `<a class="mini-btn"
          href="${telegramInfo.bot_url}"
          target="_blank"
          rel="noreferrer">
          Open bot
        </a>`
      );
    }
  } else if (telegramInfo && telegramInfo.error) {
    telegramInfoRows.push(
      `<div class="manage-note">${escHtml(telegramInfo.error)}</div>`
    );
  }

  const surfacesCard = `<div class="manage-card">
      <h3>Surfaces</h3>
      <div class="manage-row">
        <span>Web UI</span>
        <a
          class="settings-link"
          href="${localWebUi}"
          target="_blank"
          rel="noreferrer"
        >
          ${localWebUi}
        </a>
      </div>
      <div class="manage-row">
        <span>LAN URL</span>
        <a
          class="settings-link"
          href="${lanWebUi}"
          target="_blank"
          rel="noreferrer"
        >
          ${lanWebUi}
        </a>
      </div>
      <div class="manage-row"><span>Enabled</span><strong>${webUiEnabled}</strong></div>
    </div>`;

  const storageCard = `<div class="manage-card">
      <h3>Runtime storage</h3>
      <div class="manage-meta">
        <div>
          <span class="manage-note">Socket</span><br />
          <span class="settings-code">
            ${socketPath}
          </span>
        </div>
        <div>
          <span class="manage-note">Log file</span><br />
          <span class="settings-code">
            ${logPath}
          </span>
        </div>
        <div>
          <span class="manage-note">State dir</span><br />
          <span class="settings-code">
            ${statePath}
          </span>
        </div>
        <div>
          <span class="manage-note">Git journal</span><br />
          <strong>${gitJournal}</strong>
        </div>
      </div>
    </div>`;

  const healthCard = `<div class="manage-card">
      <h3>Runtime health</h3>
      <div class="manage-row"><span>Cycles</span><strong>${daemon.total_cycles ?? 0}</strong></div>
      <div class="manage-row">
        <span>Idle ticks</span>
        <strong>${daemon.total_idle_ticks ?? 0}</strong>
      </div>
      <div class="manage-row">
        <span>Last interaction</span>
        <strong>${lastInteraction}</strong>
      </div>
      <div class="manage-row">
        <span>Last consolidation</span>
        <strong>${lastConsolidation}</strong>
      </div>
    </div>`;

  const html = `<div class="settings-grid">
      <div class="manage-card">
        <h3>Autonomy</h3>
        <div class="manage-note">Change how much Jarvis can do without approval.</div>
        <div class="manage-actions">${autonomyButtons}</div>
      </div>
      <div class="manage-card">
        <h3>Telegram</h3>
        <div class="manage-row"><span>Configured</span><strong>${configured}</strong></div>
        <div class="manage-row"><span>Paired</span><strong>${paired}</strong></div>
        <div class="manage-row"><span>Chat ID</span><strong>${chatId}</strong></div>
        <div class="manage-row"><span>Poll interval</span><strong>${pollInterval}</strong></div>
        ${telegramInfoRows.join('')}
        <div class="manage-note">
          ${telegram.paired
            ? 'Telegram is paired and ready for proactive messages.'
            : 'Pair by opening the bot and sending /start.'}
        </div>
        <div class="manage-actions">
          ${telegramActions.join('')}
          <button class="mini-btn" data-action="telegram-refresh">Refresh bot info</button>
          <button class="mini-btn" data-action="telegram-test">Send test message</button>
          <button class="mini-btn deny" data-action="telegram-unpair">Unpair chat</button>
        </div>
      </div>
      ${surfacesCard}
      ${storageCard}
      ${healthCard}
    </div>`;

  setPanelHTML($('settingsPanel'), html);
}

async function refreshTelegramInfo() {
  try {
    telegramInfo = await fetchJson('/api/telegram/info');
  } catch (error) {
    telegramInfo = { ok: false, error: String(error) };
  }
  if (window.__lastStatusPayload) {
    renderSettings(window.__lastStatusPayload);
  }
}

function renderInboxHistory(status) {
  const inbox = status.proactive_inbox || [];
  const history = status.chat_history || [];
  const payload = JSON.stringify({ inbox, history });
  if (payload === renderCache.inbox) return;
  renderCache.inbox = payload;

  $('inboxCountBadge').textContent = `${inbox.length} items`;

  const inboxItems = inbox.length
    ? inbox.map(item => {
        const buttonTone = item.read ? '' : 'warn';
        const buttonLabel = item.read ? 'Read' : 'Mark read';
        const messageId = escHtml(item.id);
        return `<div class="inbox-item">
          <div class="manage-row">
            <strong>${escHtml(formatActivity(item.source_activity))}</strong>
            <span>${timeAgo(item.timestamp)}</span>
          </div>
          <div class="manage-note">${escHtml(item.content)}</div>
          <div class="manage-actions">
            <button
              class="mini-btn ${buttonTone}"
              data-action="inbox-read"
              data-message-id="${messageId}"
            >
              ${buttonLabel}
            </button>
          </div>
        </div>`;
      }).join('')
    : '<div class="empty">No proactive inbox items.</div>';

  const historyItems = history.length
    ? history.map(item => `<div class="timeline-item">
        <div class="manage-row">
          <span class="timeline-kind">${escHtml(item.role || 'message')}</span>
        </div>
        <div class="manage-note">${escHtml(String(item.content || ''))}</div>
      </div>`).join('')
    : '<div class="empty">No recent chat history.</div>';

  const html = `<div class="manage-stack">
      <div class="manage-card">
        <h3>Proactive inbox</h3>
        <div class="manage-actions">
          <button class="mini-btn warn" data-action="mark-read">Mark all read</button>
        </div>
        <div class="manage-list">${inboxItems}</div>
      </div>
      <div class="manage-card">
        <h3>Recent chat history</h3>
        <div class="manage-list">${historyItems}</div>
      </div>
    </div>`;

  setPanelHTML($('inboxPanel'), html);
}

// ── Memory type config ───────────────────────────────────────────────────────
const MEM_TYPE_CONFIG = {
  episodic:   { color: '#3b82f6', label: 'Episodic',   radius: 10, icon: 'E' },
  semantic:   { color: '#a855f7', label: 'Semantic',   radius: 9,  icon: 'S' },
  procedural: { color: '#10b981', label: 'Procedural', radius: 9,  icon: 'P' },
  valence:    { color: '#f43f5e', label: 'Valence',    radius: 8,  icon: 'V' },
  working:    { color: '#f59e0b', label: 'Working',    radius: 8,  icon: 'W' },
  sensory:    { color: '#06b6d4', label: 'Sensory',    radius: 8,  icon: 'Sn'},
  scope:      { color: '#6366f1', label: 'Scope',      radius: 18, icon: '⬡' },
  summary:    { color: '#8b5cf6', label: 'Summary',    radius: 12, icon: 'Σ' },
  topic:      { color: '#c08532', label: 'Topic group', radius: 18, icon: 'T' },
  memory:     { color: '#3b82f6', label: 'Episodic',   radius: 10, icon: 'E' },
};

function getMemTypeConfig(node) {
  const mt = node.memory_type || node.kind || 'memory';
  return MEM_TYPE_CONFIG[mt] || MEM_TYPE_CONFIG.memory;
}

let d3Simulation = null;
let graphForceExpanded = false;
let graphIsFullscreen = false;
let graphCurrentZoom = 1;
let graphActiveFilters = new Set([
  'episodic', 'semantic', 'scope', 'summary', 'topic',
  'memory', 'procedural', 'valence', 'working', 'sensory',
]);

function getEdgeColor(kind) {
  return {
    summarizes: 'rgba(168,85,247,0.5)',
    extracted_from: 'rgba(99,102,241,0.4)',
    caused_by: 'rgba(239,68,68,0.4)',
    led_to: 'rgba(16,185,129,0.4)',
    grouped: 'rgba(192,133,50,0.35)',
    stored_in: 'rgba(100,116,139,0.25)',
  }[kind] || 'rgba(255,255,255,0.15)';
}

function graphTopicKey(node) {
  const label = String(node.label || '');
  const webMatch = label.match(/^\\[web:([^\\]]+)\\]/);
  if (webMatch) return `researched:${webMatch[1]}`;
  if (node.memory_type === 'semantic' && node.predicate) return `semantic:${node.predicate}`;
  if (node.scope_id) return `${node.memory_type || node.kind}:${node.scope_id}`;
  return `${node.memory_type || node.kind}:general`;
}

function graphTopicLabel(key, children) {
  if (key.startsWith('researched:')) {
    const source = key.replace('researched:', '') || 'web';
    return `Researched ${children.length} ${source} topics`;
  }
  if (key.startsWith('semantic:')) {
    const predicate = key.replace('semantic:', '') || 'facts';
    return `${children.length} facts about ${predicate}`;
  }
  return `${children.length} related memories`;
}

function graphViewData(allNodes, allEdges, collapsed) {
  if (!collapsed) {
    return {
      nodes: allNodes.map(n => ({ ...n })),
      edges: allEdges.map(e => ({ ...e })),
    };
  }

  const groups = new Map();
  const passthrough = [];
  for (const node of allNodes) {
    const type = node.memory_type || node.kind || 'memory';
    if (type === 'scope' || node.kind === 'summary') {
      passthrough.push(node);
      continue;
    }
    const key = graphTopicKey(node);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(node);
  }

  const nodeMap = new Map();
  const mappedNodes = [];
  passthrough.forEach(node => {
    nodeMap.set(node.id, node.id);
    mappedNodes.push({ ...node });
  });

  groups.forEach((children, key) => {
    if (children.length < 4) {
      children.forEach(node => {
        nodeMap.set(node.id, node.id);
        mappedNodes.push({ ...node });
      });
      return;
    }
    const id = `topic:${key}`;
    children.forEach(node => nodeMap.set(node.id, id));
    mappedNodes.push({
      id,
      label: graphTopicLabel(key, children),
      kind: 'topic',
      memory_type: 'topic',
      count: children.length,
      child_ids: children.map(node => node.id),
      importance: Math.max(...children.map(node => Number(node.importance || 0))),
    });
  });

  const edgeMap = new Map();
  allEdges.forEach(edge => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target || source === target) return;
    const key = `${source}->${target}:${edge.kind || 'grouped'}`;
    if (!edgeMap.has(key)) {
      edgeMap.set(key, {
        id: `edge:${key}`,
        source,
        target,
        kind: source.startsWith('topic:') || target.startsWith('topic:')
          ? 'grouped'
          : edge.kind,
      });
    }
  });

  return { nodes: mappedNodes, edges: [...edgeMap.values()] };
}

function renderD3Graph(graphData) {
  if (d3Simulation) { d3Simulation.stop(); d3Simulation = null; }

  const container = document.getElementById('graphPanel');
  if (!container) return;

  const allNodes = (graphData.nodes || []);
  const allEdges = (graphData.edges || []);

  if (!allNodes.length) {
    container.innerHTML =
      '<div class="empty">No memory graph data yet. Memories appear as the daemon runs.</div>';
    return;
  }

  // Build filter bar + graph shell
  const typeSet = new Set(allNodes.map(n => n.memory_type || n.kind || 'memory'));
  const filterChips = [...typeSet].map(t => {
    const cfg = MEM_TYPE_CONFIG[t] || MEM_TYPE_CONFIG.memory;
    const active = graphActiveFilters.has(t) ? 'active' : '';
    return `<button class="graph-filter-chip ${active}"` +
      ` data-type="${t}" style="--chip-color:${cfg.color}">${cfg.label}</button>`;
  }).join('');

  const legendItems = [...typeSet].map(t => {
    const cfg = MEM_TYPE_CONFIG[t] || MEM_TYPE_CONFIG.memory;
    return `<span class="graph-legend-item">` +
      `<span class="graph-legend-dot" style="background:${cfg.color}"></span>` +
      `${cfg.label}</span>`;
  }).join('');

  container.innerHTML = `
    <div class="memory-graph-shell">
      <div class="memory-graph-toolbar">
        <div class="manage-row">
          <h3>Memory Graph</h3>
          <span class="badge">${allNodes.length} nodes · ${allEdges.length} links</span>
        </div>
        <div class="memory-graph-controls">
          <button class="mini-btn" id="graphZoomOutBtn" type="button">−</button>
          <button class="mini-btn" id="graphZoomInBtn" type="button">+</button>
          <button class="mini-btn" id="graphResetBtn" type="button">Reset view</button>
          <button class="mini-btn" id="graphExpandBtn" type="button">Expand</button>
          <button class="mini-btn" id="graphFullscreenBtn" type="button">Full page</button>
          <button class="mini-btn" id="graphRefreshBtn" type="button">Refresh</button>
        </div>
      </div>
      <div class="graph-filter-bar" id="graphFilterBar">${filterChips}</div>
      <div class="memory-graph-canvas" id="graphCanvasWrap">
        <svg id="d3GraphSvg"></svg>
        <div class="d3-tooltip" id="graphTooltip"></div>
      </div>
      <div class="graph-legend">${legendItems}</div>
    </div>`;

  // Wire filter chips
  document.querySelectorAll('#graphFilterBar .graph-filter-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const t = btn.dataset.type;
      if (graphActiveFilters.has(t)) {
        graphActiveFilters.delete(t); btn.classList.remove('active');
      } else {
        graphActiveFilters.add(t); btn.classList.add('active');
      }
      rebuildD3Sim(allNodes, allEdges);
    });
  });

  document.getElementById('graphResetBtn')?.addEventListener('click', () => {
    const svgEl = document.getElementById('d3GraphSvg');
    if (svgEl && svgEl.__zoom) d3.select(svgEl).call(svgEl.__zoom.transform, d3.zoomIdentity);
  });

  document.getElementById('graphZoomInBtn')?.addEventListener('click', () => {
    const svgEl = document.getElementById('d3GraphSvg');
    if (svgEl && svgEl.__zoom) d3.select(svgEl).transition().call(svgEl.__zoom.scaleBy, 1.35);
  });

  document.getElementById('graphZoomOutBtn')?.addEventListener('click', () => {
    const svgEl = document.getElementById('d3GraphSvg');
    if (svgEl && svgEl.__zoom) d3.select(svgEl).transition().call(svgEl.__zoom.scaleBy, 0.74);
  });

  document.getElementById('graphExpandBtn')?.addEventListener('click', () => {
    graphForceExpanded = !graphForceExpanded;
    const btn = document.getElementById('graphExpandBtn');
    if (btn) btn.textContent = graphForceExpanded ? 'Group' : 'Expand';
    rebuildD3Sim(allNodes, allEdges);
  });

  document.getElementById('graphFullscreenBtn')?.addEventListener('click', () => {
    graphIsFullscreen = !graphIsFullscreen;
    const shell = document.querySelector('.memory-graph-shell');
    shell?.classList.toggle('fullscreen', graphIsFullscreen);
    const btn = document.getElementById('graphFullscreenBtn');
    if (btn) btn.textContent = graphIsFullscreen ? 'Exit full page' : 'Full page';
    setTimeout(() => rebuildD3Sim(allNodes, allEdges), 80);
  });

  document.getElementById('graphRefreshBtn')?.addEventListener('click', refreshMemoryGraph);

  rebuildD3Sim(allNodes, allEdges);
}

function rebuildD3Sim(allNodes, allEdges) {
  if (d3Simulation) { d3Simulation.stop(); d3Simulation = null; }
  const svgEl = document.getElementById('d3GraphSvg');
  const wrap = document.getElementById('graphCanvasWrap');
  const tooltip = document.getElementById('graphTooltip');
  if (!svgEl || !wrap) return;

  const W = wrap.clientWidth || 860;
  const H = wrap.clientHeight || 480;

  const collapsed = !graphForceExpanded && graphCurrentZoom < 1.15;
  const view = graphViewData(allNodes, allEdges, collapsed);

  // Filter nodes/edges by active types
  const visibleIds = new Set(
    view.nodes
      .filter(n => graphActiveFilters.has(n.memory_type || n.kind || 'memory'))
      .map(n => n.id)
  );
  const nodes = view.nodes
    .filter(n => visibleIds.has(n.id))
    .map(n => ({ ...n }));
  const edges = view.edges
    .filter(e => visibleIds.has(e.source) && visibleIds.has(e.target))
    .map(e => ({ ...e }));

  const svg = d3.select(svgEl)
    .attr('width', W)
    .attr('height', H);
  svg.selectAll('*').remove();

  // Zoom
  const zoomGroup = svg.append('g').attr('class', 'zoom-layer');
  const zoom = d3.zoom()
    .scaleExtent([0.15, 4])
    .on('zoom', e => {
      graphCurrentZoom = e.transform.k;
      zoomGroup.attr('transform', e.transform);
    })
    .on('end', e => {
      const shouldExpand = e.transform.k >= 1.15;
      const currentlyCollapsed = !graphForceExpanded && collapsed;
      if (shouldExpand === currentlyCollapsed) rebuildD3Sim(allNodes, allEdges);
    });
  svg.call(zoom);
  svgEl.__zoom = zoom;

  // Arrow markers
  const defs = svg.append('defs');
  const edgeKinds = [...new Set(edges.map(e => e.kind || 'default'))];
  edgeKinds.forEach(kind => {
    defs.append('marker')
      .attr('id', `arrow-${kind}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', getEdgeColor(kind));
  });

  d3Simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id).distance(d => {
      const sk = d.source.kind || ''; const tk = d.target.kind || '';
      return sk === 'scope' || tk === 'scope' ? 120 : 70;
    }).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-220))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide().radius(d => getMemTypeConfig(d).radius + 14))
    .alphaDecay(0.02);

  // Edges
  const link = zoomGroup.append('g').selectAll('line')
    .data(edges).join('line')
    .attr('stroke', d => getEdgeColor(d.kind))
    .attr('stroke-width', d => d.kind === 'stored_in' ? 1 : 1.5)
    .attr('stroke-dasharray', d => {
      if (d.kind === 'summarizes') return '5 4';
      if (d.kind === 'extracted_from') return '3 3';
      return 'none';
    })
    .attr('marker-end', d => `url(#arrow-${d.kind || 'default'})`);

  // Node groups
  const node = zoomGroup.append('g').selectAll('g')
    .data(nodes).join('g')
    .attr('class', d => `graph-node${d.kind === 'topic' ? ' cluster' : ''}`)
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (e, d) => {
        if (!e.active) d3Simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => {
        if (!e.active) d3Simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      })
    );

  node.each(function(d) {
    const g = d3.select(this);
    const cfg = getMemTypeConfig(d);
    const r = cfg.radius;
    // Glow ring for important nodes
    if ((d.importance || 0) > 0.7 || d.kind === 'scope') {
      g.append('circle')
        .attr('r', r + 3)
        .attr('fill', 'none')
        .attr('stroke', cfg.color)
        .attr('stroke-width', 1)
        .attr('opacity', 0.25);
    }
    g.append('circle')
      .attr('r', r)
      .attr('fill', d.kind === 'scope' ? 'rgba(99,102,241,0.15)' : cfg.color)
      .attr('stroke', cfg.color)
      .attr('stroke-width', d.kind === 'scope' ? 2 : 1.5)
      .attr('opacity', d.kind === 'scope' ? 1 : 0.9);
    // Icon text
    if (d.kind !== 'scope') {
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('font-size', r > 11 ? 9 : 7)
        .attr('font-weight', '700')
        .attr('fill', '#fff')
        .attr('pointer-events', 'none')
        .text(d.kind === 'topic' ? String(d.count || '').slice(0, 2) : cfg.icon.slice(0, 1));
    }
    // Label below
    const shortLabel = String(d.label || d.id || '').slice(0, 22);
    g.append('text')
      .attr('class', 'graph-node-label')
      .attr('text-anchor', 'middle')
      .attr('y', r + 12)
      .attr('font-size', 9)
      .attr('fill', d.kind === 'scope' ? cfg.color : 'currentColor')
      .attr('pointer-events', 'none')
      .text(shortLabel);
  });

  // Hover tooltip
  node.on('mouseenter', function(e, d) {
    if (!tooltip) return;
    const cfg = getMemTypeConfig(d);
    tooltip.style.display = 'block';
    const muted = 'color:var(--muted)';
    const lcState = d.lifecycle_state
      ? `<br><span style="${muted}">state: ${d.lifecycle_state}</span>` : '';
    const confPct = d.confidence != null
      ? (d.confidence * 100).toFixed(0) : null;
    const conf = confPct != null
      ? `<br><span style="${muted}">confidence: ${confPct}%</span>` : '';
    const impPct = d.importance != null
      ? (d.importance * 100).toFixed(0) : null;
    const imp = impPct != null
      ? `<br><span style="${muted}">importance: ${impPct}%</span>` : '';
    const lbl = escHtml(String(d.label || d.id || '').slice(0, 80));
    tooltip.innerHTML =
      `<strong style="color:${cfg.color}">${cfg.label}</strong>` +
      `<br>${lbl}${imp}${conf}${lcState}`;
  }).on('mousemove', function(e) {
    if (!tooltip || !wrap) return;
    const rect = wrap.getBoundingClientRect();
    tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
    tooltip.style.top = (e.clientY - rect.top + 12) + 'px';
  }).on('mouseleave', function() {
    if (tooltip) tooltip.style.display = 'none';
  });

  // Click memory node → open drawer
  node.filter(d => d.kind === 'topic').on('click', () => {
    graphForceExpanded = true;
    const btn = document.getElementById('graphExpandBtn');
    if (btn) btn.textContent = 'Group';
    rebuildD3Sim(allNodes, allEdges);
  });

  node.filter(d => d.memory_id).on('click', (e, d) => {
    openMemoryDrawer(d.memory_id);
  });

  d3Simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

async function refreshMemoryGraph() {
  const graphPanel = $('graphPanel');
  if (!graphPanel) return;
  try {
    const graph = await fetchJson('/api/memory/graph?limit=60');
    currentMemoryGraph = graph;
    $('graphCountBadge').textContent = `${(graph.nodes || []).length} nodes`;
    renderD3Graph(graph);
  } catch (error) {
    if (graphPanel) graphPanel.innerHTML = `<div class="empty">${escHtml(String(error))}</div>`;
  }
}

async function refreshMemoryBrowser() {
  try {
    const [data, profile] = await Promise.all([
      fetchJson('/api/memory/recent?limit=20'),
      fetchJson('/api/memory/profile'),
    ]);
    const allItems = data.items || [];
    const filtered = allItems.filter(item => {
      if (memoryFilter === 'important') return Number(item.importance || 0) >= 0.6;
      if (memoryFilter === 'tagged') return Array.isArray(item.tags) && item.tags.length > 0;
      return true;
    });
    const payload = JSON.stringify({ items: filtered, memoryFilter, memoryGroup, profile });
    if (payload === renderCache.memory) return;
    renderCache.memory = payload;
    $('memoryCountBadge').textContent = `${filtered.length} items`;
    const classifyMemoryKind = item => {
      if (Number(item.summary_of_count || 0) > 0) return `summary of ${item.summary_of_count}`;
      if (Array.isArray(item.tags) && item.tags.includes('profile_static')) return 'profile fact';
      if (Array.isArray(item.tags) && item.tags.includes('profile_dynamic')) {
        return 'active context';
      }
      if (Array.isArray(item.tags) && item.tags.includes('project_context')) return 'project trace';
      return 'raw memory';
    };
    const storageLabel = item => {
      if (item.scope_type === 'workspace') {
        return item.repo_name
          ? `workspace · ${item.repo_name}`
          : `workspace · ${item.scope_id || 'current'}`;
      }
      return 'personal memory';
    };
    const renderMemoryItem = item => {
      const title = escHtml((item.context || item.preview || 'memory').slice(0, 80));
      const importance = (Number(item.importance || 0) * 100).toFixed(0);
      const action = escHtml((item.action || '').slice(0, 180));
      const relationCount = Array.isArray(item.source_episode_ids)
        ? item.source_episode_ids.length
        : 0;
      const memoryId = escHtml(String(item.id || ''));
      const kind = escHtml(classifyMemoryKind(item));
      const storage = escHtml(storageLabel(item));
      const relation = relationCount
        ? `${relationCount} linked memories`
        : 'direct trace';
      return `<div class="memory-item clickable" data-memory-id="${memoryId}">
        <div class="memory-card-header">
          <h3>${title}</h3>
          <span class="badge">${importance}%</span>
        </div>
        <div class="memory-card-meta">
          <span class="memory-tag-chip">${storage}</span>
          <span class="memory-tag-chip">${kind}</span>
          <span class="memory-tag-chip">${escHtml(relation)}</span>
        </div>
        <div class="memory-storage-line">
          ${timeAgo(item.timestamp)} · ${escHtml(item.citation || '')}
        </div>
        <div class="manage-note">${action}</div>
        <div class="memory-inline-actions">
          <button
            class="mini-btn"
            type="button"
            data-action="memory-open"
            data-memory-id="${memoryId}">
            Inspect
          </button>
        </div>
      </div>`;
    };
    const groups = new Map();
    for (const item of filtered) {
      const date = new Date(item.timestamp);
      const key = memoryGroup === 'importance'
        ? `${Math.round(Number(item.importance || 0) * 100)}% importance`
        : date.toDateString();
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    const groupHtml = Array.from(groups.entries())
      .map(([label, items]) => `<div class="memory-group">
          <div class="memory-group-title">${escHtml(label)}</div>
          <div class="memory-list">${items.map(renderMemoryItem).join('')}</div>
        </div>`)
      .join('');
    const renderMemoryToggle = (kind, value, label, active) => (
      `<button class="mini-btn ${active ? 'approve' : ''}"
        data-action="${kind}"
        data-value="${value}">
        ${label}
      </button>`
    );
    const toolbar = `<div class="memory-toolbar">
        ${renderMemoryToggle('memory-filter', 'all', 'All', memoryFilter === 'all')}
        ${renderMemoryToggle(
          'memory-filter',
          'important',
          'Important',
          memoryFilter === 'important',
        )}
        ${renderMemoryToggle('memory-filter', 'tagged', 'Tagged', memoryFilter === 'tagged')}
        ${renderMemoryToggle('memory-group', 'day', 'Group by day', memoryGroup === 'day')}
        ${renderMemoryToggle(
          'memory-group',
          'importance',
          'Group by importance',
          memoryGroup === 'importance',
        )}
      </div>`;
    const renderFactList = facts => facts.length
      ? `<ul>${facts.map(fact => {
          const sources = Array.isArray(fact.source_ids) && fact.source_ids.length
            ? `<span class="fact-sources">sources: ${escHtml(fact.source_ids.join(', '))}</span>`
            : '';
          return `<li>${escHtml(String(fact.text || ''))}${sources}</li>`;
        }).join('')}</ul>`
      : '<div class="manage-note">None yet.</div>';
    const renderList = items => items.length
      ? `<ul>${items.map(item => `<li>${escHtml(String(item))}</li>`).join('')}</ul>`
      : '<div class="manage-note">None yet.</div>';
    const topTags = Array.isArray(profile.top_tags) && profile.top_tags.length
      ? `<div class="memory-tag-list">
          ${profile.top_tags.map(item => `<span class="memory-tag-chip">
            ${escHtml(String(item.tag || 'tag'))}
            <strong>${escHtml(String(item.count || 0))}</strong>
          </span>`).join('')}
        </div>`
      : '<div class="manage-note">No prominent tags yet.</div>';
    const summaryCount = filtered.filter(item => Number(item.summary_of_count || 0) > 0).length;
    const workspaceCount = filtered.filter(item => item.scope_type === 'workspace').length;
    const personalCount = filtered.filter(item => item.scope_type !== 'workspace').length;
    const linkedCount = filtered.filter(
      item => Array.isArray(item.source_episode_ids) && item.source_episode_ids.length
    ).length;
    const activeScope = profile.active_scope || {};
    const atlasCard = `<div class="memory-atlas-grid">
        <div class="memory-atlas-card">
          <h3>Memory atlas</h3>
          <div class="memory-stat-grid">
            <div class="memory-stat"><strong>${filtered.length}</strong><span>visible</span></div>
            <div class="memory-stat"><strong>${summaryCount}</strong><span>summaries</span></div>
            <div class="memory-stat"><strong>${workspaceCount}</strong><span>workspace</span></div>
            <div class="memory-stat"><strong>${linkedCount}</strong><span>linked</span></div>
          </div>
        </div>
        <div class="memory-atlas-card">
          <h3>Storage model</h3>
          <div class="manage-note">
            Active scope: ${escHtml(activeScope.repo_name || activeScope.scope_id || 'personal')}
          </div>
          <div class="memory-legend">
            <span class="memory-tag-chip">personal memory ${personalCount}</span>
            <span class="memory-tag-chip">workspace memory ${workspaceCount}</span>
            <span class="memory-tag-chip">summary nodes ${summaryCount}</span>
          </div>
        </div>
      </div>`;
    const profileCard = `<div class="memory-profile-card">
        <div class="manage-row">
          <h3>Profile snapshot</h3>
          <span class="badge">
            ${escHtml(String((profile.recent_memories || []).length || 0))} recent
          </span>
        </div>
        <div class="memory-profile-grid">
          <div class="memory-profile-block">
            <h3>Static</h3>
            ${renderFactList((profile.static_facts || []).slice(0, 3))}
          </div>
          <div class="memory-profile-block">
            <h3>Dynamic</h3>
            ${renderFactList((profile.dynamic_facts || []).slice(0, 4))}
          </div>
          <div class="memory-profile-block">
            <h3>Questions</h3>
            ${renderFactList((profile.question_facts || []).slice(0, 3))}
          </div>
        </div>
        <div class="memory-profile-block">
          <h3>Top tags</h3>
          ${topTags}
        </div>
        <div class="memory-profile-block">
          <h3>Recent changes</h3>
          ${renderFactList((profile.recent_changes || []).slice(0, 4))}
        </div>
      </div>`;
    const html = filtered.length
      ? `${atlasCard}${profileCard}${toolbar}${groupHtml}`
      : `${atlasCard}${profileCard}<div class="empty">No recent memories found.</div>`;
    setPanelHTML($('memoryPanel'), html);
  } catch (error) {
    setPanelHTML($('memoryPanel'), `<div class="empty">${escHtml(String(error))}</div>`);
  }
}

async function refreshTimeline() {
  try {
    const data = await fetchJson('/api/timeline?limit=40');
    const items = data.items || [];
    const payload = JSON.stringify({ items, timelineGroup });
    if (payload === renderCache.timeline) return;
    renderCache.timeline = payload;
    $('timelineCountBadge').textContent = `${items.length} events`;
    const renderTimelineItem = item => {
      const isMemory = item.kind === 'memory' && item.memory_id;
      const classes = `timeline-item${isMemory ? ' clickable' : ''}`;
      const attrs = isMemory
        ? ` data-memory-id="${escHtml(String(item.memory_id))}"`
        : '';
      const tags = Array.isArray(item.tags) && item.tags.length
        ? `<div class="manage-note">tags ${escHtml(item.tags.join(', '))}</div>`
        : '';
      return `<div class="${classes}"${attrs}>
          <div class="manage-row">
            <span class="timeline-kind">${escHtml(item.kind || 'event')}</span>
            <span>${timeAgo(item.timestamp)}</span>
          </div>
          <h3>${escHtml(String(item.title || 'event'))}</h3>
          <div class="manage-note">${escHtml(String(item.summary || ''))}</div>
          ${tags}
        </div>`;
    };
    const groups = new Map();
    for (const item of items) {
      const date = new Date(item.timestamp);
      const key = timelineGroup === 'kind'
        ? String(item.kind || 'event')
        : date.toDateString();
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    const groupedHtml = Array.from(groups.entries())
      .map(([label, bucket]) => `<div class="memory-group">
          <div class="memory-group-title">${escHtml(label)}</div>
          <div class="timeline-list">${bucket.map(renderTimelineItem).join('')}</div>
        </div>`)
      .join('');
    const renderTimelineToggle = (value, label) => (
      `<button class="mini-btn ${timelineGroup === value ? 'approve' : ''}"
        data-action="timeline-group"
        data-value="${value}">
        ${label}
      </button>`
    );
    const toolbar = `<div class="memory-toolbar">
        ${renderTimelineToggle('day', 'Group by day')}
        ${renderTimelineToggle('kind', 'Group by type')}
      </div>`;
    const html = items.length
      ? `${toolbar}${groupedHtml}`
      : '<div class="empty">No recent activity.</div>';
    setPanelHTML($('timelinePanel'), html);
  } catch (error) {
    setPanelHTML($('timelinePanel'), `<div class="empty">${escHtml(String(error))}</div>`);
  }
}

function renderManagement(status) {
  const pending = status.pending_approvals || [];
  const proactiveInbox = status.proactive_inbox || [];
  const telegram = (status.channels && status.channels.telegram) || {};
  const unread = proactiveInbox.filter(m => !m.read);

  const payload = JSON.stringify({ pending, telegram, unread });
  if (payload === renderCache.management) return;
  renderCache.management = payload;

  $('pendingCount').textContent = `${pending.length} pending`;

  const renderApprovalItem = item => {
    const description = escHtml(item.description);
    const risk = escHtml(item.risk);
    const source = escHtml(item.source || 'daemon');
    const proposedAt = timeAgo(item.proposed_at);
    const id = escHtml(item.id);

    return `<div class="approval-item">
      <div class="manage-row">
        <strong>${description}</strong>
        <span>${risk}</span>
      </div>
      <div class="manage-row">
        <span>${source}</span>
        <span>${proposedAt}</span>
      </div>
      <div class="approval-actions">
        <button class="mini-btn" data-action="approval-detail" data-id="${id}">Details</button>
        <button class="mini-btn approve" data-action="approve" data-id="${id}">Approve</button>
        <button class="mini-btn deny" data-action="deny" data-id="${id}">Deny</button>
      </div>
    </div>`;
  };

  const approvalsHtml = pending.length
    ? pending.map(renderApprovalItem).join('')
    : '<div class="empty">No pending approvals.</div>';

  const telegramConfigured = telegram.configured ? 'Yes' : 'No token set';
  const telegramPaired = telegram.paired ? 'Yes' : 'Waiting for /start';
  const telegramChatId = telegram.chat_id ? escHtml(String(telegram.chat_id)) : '—';
  const telegramPoll = `${telegram.poll_interval_s || '?'}s`;

  const telegramHtml = `<div class="manage-card">
      <h3>Telegram channel</h3>
      <div class="manage-meta">
        <div class="manage-row"><span>Configured</span><strong>${telegramConfigured}</strong></div>
        <div class="manage-row"><span>Paired</span><strong>${telegramPaired}</strong></div>
        <div class="manage-row"><span>Chat ID</span><strong>${telegramChatId}</strong></div>
        <div class="manage-row"><span>Push poll</span><strong>${telegramPoll}</strong></div>
      </div>
      <div class="manage-note">
        If Telegram is configured but not paired, send <code>/start</code> to your bot.
      </div>
    </div>`;

  const renderInboxItem = item => {
    const activity = escHtml(formatActivity(item.source_activity));
    const timestamp = timeAgo(item.timestamp);
    const content = escHtml(item.content);
    return `<div class="inbox-item">
      <div class="manage-row">
        <strong>${activity}</strong>
        <span>${timestamp}</span>
      </div>
      <div class="manage-note">${content}</div>
    </div>`;
  };

  const inboxHtml = `<div class="manage-card">
      <h3>Proactive inbox</h3>
      <div class="manage-row"><span>Unread items</span><strong>${unread.length}</strong></div>
      <div class="manage-actions">
        <button class="mini-btn" data-action="clear-pending">Clear pending</button>
        <button class="mini-btn warn" data-action="mark-read">Mark all read</button>
        <button class="mini-btn" data-action="clear-chat">Clear chat history</button>
        <button class="mini-btn deny" data-action="shutdown-daemon">Shutdown daemon</button>
      </div>
      <div class="manage-list">
        ${unread.length
          ? unread.slice(0, 3).map(renderInboxItem).join('')
          : '<div class="empty">No unread proactive messages.</div>'}
      </div>
    </div>`;

  const dangerHtml = `<div class="manage-card">
      <h3>Danger zone</h3>
      <div class="manage-note" style="margin-bottom:10px">
        Destructive operations. These cannot be undone.
      </div>
      <div class="manage-actions">
        <button class="mini-btn danger" data-action="clear-memory" id="clearMemoryBtn">
          Clear all episodic memory
        </button>
      </div>
      <div id="clearMemoryStatus" style="margin-top:8px;font-size:12px;color:var(--muted)"></div>
    </div>`;

  const html = `<div class="manage-stack">
      <div class="manage-card">
        <h3>Pending approvals</h3>
        <div class="manage-list">${approvalsHtml}</div>
      </div>
      ${telegramHtml}
      ${inboxHtml}
      ${dangerHtml}
    </div>`;

  setPanelHTML($('managePanel'), html);
}

function openApprovalDrawer(item) {
  selectedApproval = item;
  if (!approvalDrawer || !drawerBackdrop) return;
  approvalDrawerTitle.textContent = item.description || 'Pending action';
  approvalDrawerMeta.textContent = `Action ${item.id}`;
  approvalDrawerRisk.textContent = item.risk || 'unknown';
  approvalDrawerSource.textContent = item.source || 'daemon';
  approvalDrawerTime.textContent = timeAgo(item.proposed_at);
  const contextText = JSON.stringify(item.context || {}, null, 2) || 'No context available.';
  approvalDrawerContext.textContent = contextText;
  approvalDrawer.classList.add('show');
  drawerBackdrop.classList.add('show');
  approvalDrawer.setAttribute('aria-hidden', 'false');
}

function closeApprovalDrawer() {
  selectedApproval = null;
  if (!approvalDrawer || !drawerBackdrop) return;
  approvalDrawer.classList.remove('show');
  drawerBackdrop.classList.remove('show');
  approvalDrawer.setAttribute('aria-hidden', 'true');
}

async function openMemoryDrawer(memoryId) {
  selectedMemoryId = memoryId;
  if (!memoryDrawer || !memoryDrawerBackdrop || !memoryDrawerBody || !memoryDrawerMeta) return;

  memoryDrawer.classList.add('show');
  memoryDrawerBackdrop.classList.add('show');
  memoryDrawer.setAttribute('aria-hidden', 'false');
  memoryDrawerMeta.textContent = `Memory ${memoryId}`;
  memoryDrawerBody.innerHTML = '<div class="empty">Loading memory details…</div>';

  try {
    const [detailPayload, timelinePayload] = await Promise.all([
      fetchJson(`/api/memory/item?id=${encodeURIComponent(memoryId)}`),
      fetchJson(`/api/memory/timeline?id=${encodeURIComponent(memoryId)}&limit=8`),
    ]);

    const detail = (detailPayload.items || [])[0];
    if (!detail) {
      memoryDrawerBody.innerHTML = '<div class="empty">Memory not found.</div>';
      return;
    }

    const tags = Array.isArray(detail.tags) && detail.tags.length
      ? detail.tags
          .map(tag => `<span class="memory-tag-chip">${escHtml(String(tag))}</span>`)
          .join('')
      : '<span class="manage-note">No tags</span>';
    const citation = String(detail.citation || `[memory:${memoryId}]`);
    const sourceIds = Array.isArray(detail.source_episode_ids) ? detail.source_episode_ids : [];
    let linkedSourceHtml = '<div class="empty">No linked source memories.</div>';
    if (sourceIds.length) {
      const relatedUrl = `/api/memory/item?${
        sourceIds.map(id => `id=${encodeURIComponent(id)}`).join('&')
      }`;
      try {
        const relatedPayload = await fetchJson(relatedUrl);
        const relatedItems = relatedPayload.items || [];
        if (relatedItems.length) {
          linkedSourceHtml = relatedItems.map(item => `<button
              class="drawer-memory-link"
              type="button"
              data-action="drawer-memory-open"
              data-memory-id="${escHtml(String(item.id || ''))}">
              <strong>${escHtml(String(item.preview || item.context || 'memory'))}</strong>
              <span class="search-meta">${escHtml(String(item.citation || ''))}</span>
            </button>`).join('');
        }
      } catch (_error) {
        linkedSourceHtml = '<div class="empty">Linked sources unavailable.</div>';
      }
    }
    const nearby = (timelinePayload.items || []).map(item => {
      const itemId = String(item.id || '');
      const active = itemId === String(memoryId);
      return `<button
          class="drawer-memory-link ${active ? 'active' : ''}"
          type="button"
          data-action="drawer-memory-open"
          data-memory-id="${escHtml(itemId)}">
          <strong>${escHtml(String(item.preview || 'memory'))}</strong>
          <span class="search-meta">${escHtml(timeAgo(item.timestamp))}</span>
        </button>`;
    }).join('');

    memoryDrawerMeta.textContent = `${timeAgo(detail.timestamp)} · importance ${(
      Number(detail.importance || 0) * 100
    ).toFixed(0)}%`;
    const storageLabel = detail.scope_type === 'workspace'
      ? `workspace · ${detail.repo_name || detail.scope_id || 'current'}`
      : 'personal memory';
    const relationLabel = Number(detail.summary_of_count || 0) > 0
      ? `summary of ${detail.summary_of_count} memories`
      : sourceIds.length
        ? `${sourceIds.length} linked sources`
        : 'direct trace';
    memoryDrawerBody.innerHTML = `<div class="drawer-stack">
        <div class="drawer-minimap">
          <div class="drawer-minimap-item">
            <strong>${escHtml(storageLabel)}</strong>
            <span>storage</span>
          </div>
          <div class="drawer-minimap-item">
            <strong>${escHtml(String(detail.summary_kind || 'memory'))}</strong>
            <span>kind</span>
          </div>
          <div class="drawer-minimap-item">
            <strong>${escHtml(relationLabel)}</strong>
            <span>relationships</span>
          </div>
        </div>
        <div class="manage-card">
          <h3>Storage</h3>
          <div class="manage-row">
            <span>Stored in</span><strong>${escHtml(storageLabel)}</strong>
          </div>
          <div class="manage-row">
            <span>Kind</span><strong>${escHtml(String(detail.summary_kind || 'memory'))}</strong>
          </div>
          <div class="manage-row">
            <span>Relationships</span><strong>${escHtml(relationLabel)}</strong>
          </div>
        </div>
        <div class="manage-card">
          <h3>Context</h3>
          <div class="manage-note">${escHtml(String(detail.context || ''))}</div>
        </div>
        <div class="manage-card">
          <h3>Action</h3>
          <div class="manage-note">${escHtml(String(detail.action || ''))}</div>
        </div>
        <div class="manage-card">
          <h3>Outcome</h3>
          <div class="manage-note">${escHtml(String(detail.outcome || ''))}</div>
        </div>
        <div class="manage-card">
          <h3>Tags</h3>
          <div class="memory-tag-list">${tags}</div>
        </div>
        <div class="manage-card">
          <h3>Citation</h3>
          <div class="manage-row">
            <code>${escHtml(citation)}</code>
            <button
              class="mini-btn"
              type="button"
              data-action="drawer-copy-citation"
              data-citation="${escHtml(citation)}">
              Copy cite
            </button>
          </div>
        </div>
        <div class="manage-card">
          <h3>Linked sources</h3>
          <div class="drawer-list">
            ${linkedSourceHtml}
          </div>
        </div>
        <div class="manage-card">
          <h3>Nearby memories</h3>
          <div class="drawer-list">
            ${nearby || '<div class="empty">No nearby memories.</div>'}
          </div>
        </div>
      </div>`;
  } catch (error) {
    memoryDrawerBody.innerHTML = `<div class="empty">${escHtml(String(error))}</div>`;
  }
}

function closeMemoryDrawer() {
  selectedMemoryId = null;
  if (!memoryDrawer || !memoryDrawerBackdrop) return;
  memoryDrawer.classList.remove('show');
  memoryDrawerBackdrop.classList.remove('show');
  memoryDrawer.setAttribute('aria-hidden', 'true');
}

function renderThoughts(thoughts) {
  const serialized = JSON.stringify(thoughts);
  if (serialized === renderCache.thoughts) return;
  renderCache.thoughts = serialized;
  maybeSpeakNewestThought(thoughts);

  if (!thoughts.length) {
    setPanelHTML($('thoughtsPanel'), '<div class="empty">No thoughts yet.</div>');
    return;
  }

  const html = thoughts.map(t => {
    const activity = escHtml(formatActivity(t.activity));
    const time = t.timestamp ? timeAgo(t.timestamp) : 'just now';
    return `<article class="thought ${activityClass(t.activity)}">
      <div class="thought-header">
        <span class="thought-type">${activity}</span>
        <span class="thought-time">${time}</span>
      </div>
      <div class="thought-summary">${escHtml(t.summary)}</div>
    </article>`;
  }).join('');

  setPanelHTML($('thoughtsPanel'), html);
}

function renderGoals(goals) {
  const serialized = JSON.stringify(goals);
  if (serialized === renderCache.goals) return;
  renderCache.goals = serialized;

  $('goalCount').textContent = `${goals.length} goal${goals.length === 1 ? '' : 's'}`;

  if (!goals.length) {
    setPanelHTML($('goalsPanel'), '<div class="empty">No active goals.</div>');
    return;
  }

  const topLevel = goals.filter(g => !g.parent_id);

  function renderGoal(g) {
    const subs = goals.filter(s => g.subgoals && g.subgoals.includes(s.id));
    const progress = clampProgress(g.progress);
    const progressPct = Math.round(progress * 100);
    const hasSubs = subs.length > 0;
    const goalId = escHtml(g.id);
    const actions = [];
    const priorityPct = Math.round(Number(g.priority || 0) * 100);

    if (g.status === 'active') {
      actions.push(
        `<button class="mini-btn warn" data-action="goal-status" ` +
        `data-goal-id="${goalId}" data-status="suspended">Pause</button>`
      );
      actions.push(
        `<button class="mini-btn approve" data-action="goal-status" ` +
        `data-goal-id="${goalId}" data-status="completed">Complete</button>`
      );
      actions.push(
        `<button class="mini-btn deny" data-action="goal-status" ` +
        `data-goal-id="${goalId}" data-status="failed">Fail</button>`
      );
    } else if (g.status === 'suspended' || g.status === 'blocked') {
      actions.push(
        `<button class="mini-btn approve" data-action="goal-status" ` +
        `data-goal-id="${goalId}" data-status="active">Resume</button>`
      );
    }

    return `<article class="goal ${hasSubs ? 'has-subgoals' : ''}">
      <div class="goal-header">
        <span class="badge ${hasSubs ? 'orange' : ''}">${escHtml(g.status || 'unknown')}</span>
        <span class="goal-progress-label">${progressPct}%</span>
      </div>
      <div class="goal-desc">${escHtml(g.description)}</div>
      <div class="goal-meta">
        <div class="progress-bar">
          <div class="progress-fill" style="width:${progressPct}%"></div>
        </div>
      </div>
      <div class="goal-edit-grid expanded">
        <input
          class="goal-edit-input"
          data-goal-edit="description"
          data-goal-id="${goalId}"
          value="${escHtml(g.description)}"
        />
        <input
          class="goal-edit-input"
          data-goal-edit="success_criteria"
          data-goal-id="${goalId}"
          value="${escHtml(g.success_criteria || '')}"
          placeholder="Success criteria…"
        />
      </div>
      <div class="goal-edit-grid">
        <input
          class="goal-edit-input"
          data-goal-edit="priority"
          data-goal-id="${goalId}"
          type="number"
          min="0"
          max="100"
          value="${priorityPct}"
        />
      </div>
      ${actions.length ? `<div class="goal-actions">${actions.join('')}</div>` : ''}
      <div class="goal-actions">
        <button class="mini-btn" data-action="goal-save" data-goal-id="${goalId}">
          Save changes
        </button>
      </div>
      ${hasSubs ? `<div class="subgoals">${subs.map(s =>
        `<div class="subgoal">▸ ${escHtml(s.description)}</div>`).join('')}</div>` : ''}
    </article>`;
  }

  setPanelHTML($('goalsPanel'), topLevel.map(renderGoal).join(''));
}

const goalForm = $('goalForm');
const goalInput = $('goalInput');
const goalSubmitBtn = $('goalSubmitBtn');
const managePanel = $('managePanel');
const settingsPanel = $('settingsPanel');
const memoryPanel = $('memoryPanel');
const graphPanel = $('graphPanel');
const timelinePanel = $('timelinePanel');
const fileList = $('fileList');
const fileEditor = $('fileEditor');
const fileMeta = $('fileMeta');
const fileReloadBtn = $('fileReloadBtn');
const fileDiffBtn = $('fileDiffBtn');
const fileSaveBtn = $('fileSaveBtn');
const fileStatusBadge = $('fileStatusBadge');
const fileDiffPreview = $('fileDiffPreview');
const drawerBackdrop = $('drawerBackdrop');
const approvalDrawer = $('approvalDrawer');
const approvalDrawerClose = $('approvalDrawerClose');
const approvalDrawerTitle = $('approvalDrawerTitle');
const approvalDrawerMeta = $('approvalDrawerMeta');
const approvalDrawerRisk = $('approvalDrawerRisk');
const approvalDrawerSource = $('approvalDrawerSource');
const approvalDrawerTime = $('approvalDrawerTime');
const approvalDrawerContext = $('approvalDrawerContext');
const approvalDrawerApprove = $('approvalDrawerApprove');
const approvalDrawerDeny = $('approvalDrawerDeny');
const memoryDrawerBackdrop = $('memoryDrawerBackdrop');
const memoryDrawer = $('memoryDrawer');
const memoryDrawerBody = $('memoryDrawerBody');
const memoryDrawerMeta = $('memoryDrawerMeta');
const memoryDrawerClose = $('memoryDrawerClose');
const toast = $('toast');
let toastTimer = null;

if (goalForm && goalInput && goalSubmitBtn) {
  goalForm.addEventListener('submit', async event => {
    event.preventDefault();
    const description = goalInput.value.trim();
    if (!description || !online) return;

    goalInput.disabled = true;
    goalSubmitBtn.disabled = true;

    try {
      const response = await fetch('/api/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description, priority: 0.5 }),
      });
      const data = await response.json();
      if (response.ok && !data.error) {
        goalInput.value = '';
      }
    } catch (error) {
      console.error('goal create failed', error);
    } finally {
      goalInput.disabled = false;
      goalSubmitBtn.disabled = !online;
      goalInput.focus();
    }
  });
}

const goalsPanel = $('goalsPanel');
if (goalsPanel) {
  goalsPanel.addEventListener('click', async event => {
    const button = event.target.closest('button[data-action]');
    if (!button || !online) return;

    const action = button.dataset.action;
    const goalId = button.dataset.goalId;
    button.disabled = true;

    try {
      if (action === 'goal-save' && goalId) {
        const descSelector = `[data-goal-edit="description"][data-goal-id="${goalId}"]`;
        const prioritySelector = `[data-goal-edit="priority"][data-goal-id="${goalId}"]`;
        const criteriaSelector = `[data-goal-edit="success_criteria"][data-goal-id="${goalId}"]`;
        const descInput = goalsPanel.querySelector(descSelector);
        const priorityInput = goalsPanel.querySelector(prioritySelector);
        const criteriaInput = goalsPanel.querySelector(criteriaSelector);
        const description = descInput ? descInput.value.trim() : '';
        const rawPriority = priorityInput ? Number(priorityInput.value || 0) : 50;
        const priority = Math.max(0, Math.min(100, rawPriority)) / 100;
        const successCriteria = criteriaInput ? criteriaInput.value.trim() : '';
        await fetch('/api/goals/update', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            goal_id: goalId,
            description,
            priority,
            success_criteria: successCriteria,
          }),
        });
      }
    } catch (error) {
      console.error('goal edit failed', error);
    } finally {
      button.disabled = false;
    }
  });
}

async function loadManagedFileList() {
  if (!fileList) return;
  try {
    const response = await fetch('/api/files/list');
    const data = await response.json();
    const files = data.files || [];
    if (!files.length) {
      fileList.innerHTML = '<div class="empty">No managed files available.</div>';
      return;
    }

    fileList.innerHTML = files.map(file => `<button
        class="file-btn ${currentManagedFile === file.name ? 'active' : ''}"
        data-file-name="${escHtml(file.name)}"
        type="button">
        <span class="file-btn-title">${escHtml(file.label)}</span>
        <span class="file-btn-copy">${escHtml(file.description)}</span>
      </button>`).join('');

    if (!currentManagedFile) {
      await loadManagedFile(files[0].name);
    }
  } catch (error) {
    fileList.innerHTML = `<div class="empty">${escHtml(String(error))}</div>`;
  }
}

async function loadManagedFile(name) {
  if (!fileEditor || !fileMeta) return;
  try {
    const response = await fetch(`/api/files?name=${encodeURIComponent(name)}`);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Failed to load file');

    currentManagedFile = data.name;
    currentManagedFileOriginal = data.content || '';
    fileEditor.value = currentManagedFileOriginal;
    fileEditor.disabled = false;
    if (fileDiffPreview) fileDiffPreview.textContent = 'No diff yet.';

    const sizeText = `${data.bytes || 0} bytes`;
    const modified = data.modified_at
      ? new Date(data.modified_at * 1000).toLocaleString()
      : 'new file';
    fileMeta.textContent = `${data.label} · ${sizeText} · ${modified} · ${data.path}`;
    setFileDirty(false);
    highlightCurrentFileButton();
  } catch (error) {
    fileMeta.textContent = `Failed to load file: ${error}`;
  }
}

function highlightCurrentFileButton() {
  if (!fileList) return;
  fileList.querySelectorAll('.file-btn').forEach(button => {
    button.classList.toggle('active', button.dataset.fileName === currentManagedFile);
  });
}

function setFileDirty(isDirty) {
  if (!fileStatusBadge) return;
  fileStatusBadge.textContent = isDirty ? 'unsaved' : 'saved';
}

if (fileList) {
  fileList.addEventListener('click', async event => {
    const button = event.target.closest('.file-btn');
    if (!button) return;
    const name = button.dataset.fileName;
    if (!name) return;
    await loadManagedFile(name);
  });
}

if (fileEditor) {
  fileEditor.addEventListener('input', () => {
    setFileDirty(fileEditor.value !== currentManagedFileOriginal);
  });
}

if (fileReloadBtn) {
  fileReloadBtn.addEventListener('click', async () => {
    if (!currentManagedFile) return;
    await loadManagedFile(currentManagedFile);
  });
}

if (fileDiffBtn) {
  fileDiffBtn.addEventListener('click', async () => {
    if (!currentManagedFile || !fileEditor || !fileDiffPreview) return;
    fileDiffBtn.disabled = true;
    try {
      const data = await fetchJson('/api/files/diff', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: currentManagedFile, content: fileEditor.value }),
      });
      fileDiffPreview.textContent = data.diff || 'No changes.';
    } catch (error) {
      fileDiffPreview.textContent = `Diff failed: ${error}`;
    } finally {
      fileDiffBtn.disabled = false;
    }
  });
}

if (fileSaveBtn) {
  fileSaveBtn.addEventListener('click', async () => {
    if (!currentManagedFile || !fileEditor) return;
    fileSaveBtn.disabled = true;
    try {
      const response = await fetch('/api/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: currentManagedFile, content: fileEditor.value }),
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Failed to save file');
      currentManagedFileOriginal = data.content || '';
      fileEditor.value = currentManagedFileOriginal;
      setFileDirty(false);
      const modified = data.modified_at
        ? new Date(data.modified_at * 1000).toLocaleString()
        : 'saved';
      const sizeText = `${data.bytes || 0} bytes`;
      fileMeta.textContent = `${data.label} · ${sizeText} · ${modified} · ${data.path}`;
      await loadManagedFileList();
    } catch (error) {
      fileMeta.textContent = `Save failed: ${error}`;
    } finally {
      fileSaveBtn.disabled = false;
    }
  });
}

loadManagedFileList();
refreshMemoryBrowser();
refreshTimeline();

if (managePanel) {
  managePanel.addEventListener('click', async event => {
    const button = event.target.closest('button[data-action]');
    if (!button || !online) return;

    const action = button.dataset.action;
    const id = button.dataset.id;
    if (action === 'approval-detail' && id) {
      const payload = window.__lastStatusPayload || {};
      const pending = payload.pending_approvals || [];
      const item = pending.find(entry => entry.id === id);
      if (item) openApprovalDrawer(item);
      return;
    }
    button.disabled = true;

    try {
      if (action === 'approve' && id) {
        await fetch('/api/approve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action_id: id }),
        });
      } else if (action === 'deny' && id) {
        await fetch('/api/deny', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action_id: id }),
        });
      } else if (action === 'mark-read') {
        await fetch('/api/inbox/mark-read', { method: 'POST' });
      } else if (action === 'inbox-read') {
        await fetch('/api/inbox/mark-read', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message_id: button.dataset.messageId }),
        });
      } else if (action === 'clear-chat') {
        await fetch('/api/chat/clear', { method: 'POST' });
        $('chatMessages').innerHTML = '<div class="empty">Conversation cleared.</div>';
        chatEmpty = true;
      } else if (action === 'clear-pending') {
        await fetch('/api/pending/clear', { method: 'POST' });
      } else if (action === 'shutdown-daemon') {
        await fetch('/api/shutdown', { method: 'POST' });
      } else if (action === 'clear-memory') {
        const statusEl = $('clearMemoryStatus');
        const msg = 'Delete ALL episodic memories and reset the vector index?' +
          '\\nThis cannot be undone.';
        if (!confirm(msg)) return;
        if (statusEl) statusEl.textContent = 'Clearing…';
        try {
          const res = await fetch('/api/memory/clear', { method: 'POST' });
          const data = await res.json();
          if (statusEl)
            statusEl.textContent = data.ok ? 'Memory cleared.' : (data.error || 'Failed.');
          renderCache.memory = '';
          refreshMemoryGraph();
        } catch (e) {
          if (statusEl) statusEl.textContent = String(e);
        }
      } else if (action === 'goal-status') {
        const goalId = button.dataset.goalId;
        const status = button.dataset.status;
        await fetch('/api/goals/status', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal_id: goalId, status }),
        });
      }
    } catch (error) {
      console.error('management action failed', error);
    } finally {
      button.disabled = false;
    }
  });
}

if (approvalDrawerClose) {
  approvalDrawerClose.addEventListener('click', closeApprovalDrawer);
}
if (drawerBackdrop) {
  drawerBackdrop.addEventListener('click', closeApprovalDrawer);
}
if (memoryDrawerBackdrop) {
  memoryDrawerBackdrop.addEventListener('click', closeMemoryDrawer);
}
if (memoryDrawerClose) {
  memoryDrawerClose.addEventListener('click', closeMemoryDrawer);
}
if (memoryDrawerBody) {
  memoryDrawerBody.addEventListener('click', async event => {
    const copyButton = event.target.closest('button[data-action="drawer-copy-citation"]');
    if (copyButton) {
      const citation = copyButton.dataset.citation;
      if (citation) await copyText(citation, 'Copied citation');
      return;
    }
    const button = event.target.closest('button[data-action="drawer-memory-open"]');
    if (!button) return;
    const memoryId = button.dataset.memoryId;
    if (memoryId && memoryId !== selectedMemoryId) {
      await openMemoryDrawer(memoryId);
    }
  });
}
if (approvalDrawerApprove) {
  approvalDrawerApprove.addEventListener('click', async () => {
    if (!selectedApproval) return;
    await fetch('/api/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_id: selectedApproval.id }),
    });
    closeApprovalDrawer();
  });
}
if (approvalDrawerDeny) {
  approvalDrawerDeny.addEventListener('click', async () => {
    if (!selectedApproval) return;
    await fetch('/api/deny', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_id: selectedApproval.id }),
    });
    closeApprovalDrawer();
  });
}

if (settingsPanel) {
  settingsPanel.addEventListener('click', async event => {
    const button = event.target.closest('button[data-action]');
    if (!button || !online) return;

    const action = button.dataset.action;
    button.disabled = true;
    try {
      if (action === 'autonomy') {
        await fetch('/api/settings/autonomy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ level: button.dataset.level }),
        });
      } else if (action === 'telegram-refresh') {
        await refreshTelegramInfo();
      } else if (action === 'telegram-test') {
        await fetch('/api/telegram/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: 'Jarvis test message from web settings.' }),
        });
      } else if (action === 'telegram-unpair') {
        await fetch('/api/telegram/unpair', { method: 'POST' });
        telegramInfo = null;
        await refreshTelegramInfo();
      }
    } catch (error) {
      console.error('settings action failed', error);
    } finally {
      button.disabled = false;
    }
  });
}

if (memoryPanel) {
  memoryPanel.addEventListener('click', async event => {
    const memoryCard = event.target.closest('[data-memory-id]');
    const filterToggle = event.target.closest(
      'button[data-action="memory-filter"], button[data-action="memory-group"]',
    );
    if (memoryCard && !filterToggle) {
      const memoryId = memoryCard.dataset.memoryId;
      if (memoryId) {
        await openMemoryDrawer(memoryId);
        return;
      }
    }
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    if (button.dataset.action === 'memory-filter') {
      memoryFilter = button.dataset.value || 'all';
      await refreshMemoryBrowser();
      return;
    }
    if (button.dataset.action === 'memory-group') {
      memoryGroup = button.dataset.value || 'day';
      await refreshMemoryBrowser();
      return;
    }
    if (button.dataset.action === 'memory-open') {
      const memoryId = button.dataset.memoryId;
      if (memoryId) await openMemoryDrawer(memoryId);
    }
  });
}

if (graphPanel) {
  graphPanel.addEventListener('click', async event => {
    const memoryNode = event.target.closest('[data-memory-id]');
    if (memoryNode) {
      const memoryId = memoryNode.dataset.memoryId;
      if (memoryId) {
        await openMemoryDrawer(memoryId);
        return;
      }
    }
    // D3 zoom handles pan/zoom internally via SVG — no manual event handlers needed
  });
}

if (timelinePanel) {
  timelinePanel.addEventListener('click', async event => {
    const memoryCard = event.target.closest('[data-memory-id]');
    if (memoryCard) {
      const memoryId = memoryCard.dataset.memoryId;
      if (memoryId) {
        await openMemoryDrawer(memoryId);
        return;
      }
    }
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    if (button.dataset.action === 'timeline-group') {
      timelineGroup = button.dataset.value || 'day';
      await refreshTimeline();
    }
  });
}

const memSearchInput = $('memSearchInput');
const memSearchResults = $('memSearchResults');
let memSearchTimer = null;
let currentSearchItems = [];
let currentSearchSelectedIndex = -1;
let currentSearchProfile = {};

if (memSearchInput && memSearchResults) {
  memSearchInput.addEventListener('input', () => {
    clearTimeout(memSearchTimer);
    memSearchTimer = setTimeout(runMemorySearch, 250);
  });

  memSearchInput.addEventListener('keydown', async event => {
    if (event.key === 'Escape') {
      memSearchInput.value = '';
      setMemorySearchResults([]);
      return;
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      moveSearchSelection(1);
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      moveSearchSelection(-1);
      return;
    }
    if (event.key === 'Enter') {
      const item = currentSearchItems[currentSearchSelectedIndex];
      if (item && item.id) {
        event.preventDefault();
        await openSearchResult(item.id, true);
      }
    }
  });

  document.addEventListener('click', event => {
    if (!memSearchResults.contains(event.target) && event.target !== memSearchInput) {
      memSearchResults.classList.remove('show');
    }
  });

  memSearchResults.addEventListener('click', async event => {
    const actionButton = event.target.closest('button[data-action]');
    if (actionButton) {
      const memoryId = actionButton.dataset.memoryId;
      if (!memoryId) return;
      if (actionButton.dataset.action === 'search-open-memory-tab') {
        await openSearchResult(memoryId, false);
        return;
      }
      if (actionButton.dataset.action === 'search-copy-id') {
        await copyText(memoryId);
        return;
      }
      if (actionButton.dataset.action === 'search-copy-citation') {
        await copyText(`[memory:${memoryId}]`, 'Copied citation');
        return;
      }
    }
    const target = event.target.closest('[data-memory-id]');
    if (!target) return;
    const memoryId = target.dataset.memoryId;
    if (!memoryId) return;
    await openSearchResult(memoryId, true);
  });

  memSearchResults.addEventListener('keydown', async event => {
    if (event.key === 'j') {
      event.preventDefault();
      moveSearchSelection(1);
      return;
    }
    if (event.key === 'k') {
      event.preventDefault();
      moveSearchSelection(-1);
      return;
    }
    if (event.key === 'Enter') {
      const item = currentSearchItems[currentSearchSelectedIndex];
      if (item && item.id) {
        event.preventDefault();
        await openSearchResult(item.id, true);
      }
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      memSearchResults.classList.remove('show');
      memSearchInput.focus();
    }
  });
}

async function runMemorySearch() {
  if (!memSearchInput || !memSearchResults) return;

  const query = memSearchInput.value.trim();
  if (!query) {
    setMemorySearchResults([]);
    return;
  }

  try {
    const response = await fetch(`/api/memory/recall?q=${encodeURIComponent(query)}`);
    const data = await response.json();
    setMemorySearchResults(data.results || [], data.error || '', data.profile || {});
  } catch (_error) {
    setMemorySearchResults([], 'Search failed');
  }
}

function setMemorySearchResults(items, errorText = '', profile = {}) {
  if (!memSearchResults) return;
  currentSearchItems = Array.isArray(items) ? items : [];
  currentSearchProfile = profile || {};
  if (!currentSearchItems.length) {
    currentSearchSelectedIndex = -1;
  } else if (
    currentSearchSelectedIndex < 0 ||
    currentSearchSelectedIndex >= currentSearchItems.length
  ) {
    currentSearchSelectedIndex = 0;
  }

  if (!items.length && !errorText) {
    memSearchResults.innerHTML = '';
    memSearchResults.classList.remove('show');
    return;
  }

  if (errorText) {
    memSearchResults.innerHTML = `<div class="empty">${escHtml(errorText)}</div>`;
    memSearchResults.classList.add('show');
    return;
  }

  const profileLines = []
    .concat((currentSearchProfile.static || []).slice(0, 1))
    .concat((currentSearchProfile.dynamic || []).slice(0, 2));

  const profileHTML = profileLines.length
    ? `<div class="search-profile">
        <span class="search-profile-label">Profile context</span>
        ${profileLines
          .map(line => `<span class="search-text">${escHtml(String(line))}</span>`)
          .join('')}
      </div>`
    : '';

  memSearchResults.innerHTML = profileHTML + items.map((item, index) => {
    const score = Number(item.score || 0);
    const preview = String(item.preview || item.content || '').slice(0, 220);
    const timestamp = String(item.timestamp || '').replace('T', ' ').replace('+00:00', 'Z');
    const tags = Array.isArray(item.tags) && item.tags.length
      ? item.tags.slice(0, 3).join(', ')
      : '';
    const metaBits = [timestamp, tags].filter(Boolean);
    const memoryId = String(item.id || '');
    const selected = index === currentSearchSelectedIndex;
    const classes = `search-result${memoryId ? ' clickable' : ''}${selected ? ' selected' : ''}`;
    const attrs = memoryId ? ` data-memory-id="${escHtml(memoryId)}"` : '';
    return `<div class="${classes}"${attrs}>
      <span class="search-score">${(score * 100).toFixed(0)}%</span>
      <span class="search-text">${escHtml(preview)}</span>
      <span class="search-meta">${escHtml(metaBits.join(' • '))}</span>
      <div class="search-actions">
        <button
          class="mini-btn"
          type="button"
          data-action="search-open-memory-tab"
          data-memory-id="${escHtml(memoryId)}">
          Open in memory tab
        </button>
        <button
          class="mini-btn"
          type="button"
          data-action="search-copy-id"
          data-memory-id="${escHtml(memoryId)}">
          Copy id
        </button>
        <button
          class="mini-btn"
          type="button"
          data-action="search-copy-citation"
          data-memory-id="${escHtml(memoryId)}">
          Copy cite
        </button>
      </div>
    </div>`;
  }).join('');
  memSearchResults.classList.add('show');
}

function moveSearchSelection(direction) {
  if (!currentSearchItems.length) return;
  if (currentSearchSelectedIndex < 0) {
    currentSearchSelectedIndex = 0;
  } else {
    currentSearchSelectedIndex = (
      currentSearchSelectedIndex + direction + currentSearchItems.length
    ) % currentSearchItems.length;
  }
  setMemorySearchResults(currentSearchItems, '', currentSearchProfile);
  const selected = memSearchResults
    ? memSearchResults.querySelectorAll('.search-result')[currentSearchSelectedIndex]
    : null;
  if (selected) selected.scrollIntoView({ block: 'nearest' });
}

async function openSearchResult(memoryId, clearInput = true) {
  if (!memSearchResults || !memSearchInput) return;
  memSearchResults.classList.remove('show');
  if (clearInput) memSearchInput.value = '';
  setActiveTab('memory');
  await openMemoryDrawer(memoryId);
}

async function copyText(value, successMessage = 'Copied memory id') {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(value);
      showToast(successMessage);
      return;
    }
  } catch (_error) {
    // Fall through to legacy copy path.
  }

  const helper = document.createElement('textarea');
  helper.value = value;
  helper.setAttribute('readonly', 'true');
  helper.style.position = 'fixed';
  helper.style.opacity = '0';
  document.body.appendChild(helper);
  helper.select();
  document.execCommand('copy');
  document.body.removeChild(helper);
  showToast(successMessage);
}

function showToast(message) {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
  }, 1400);
}

const MAX_LOG_LINES = 200;
let logLines = [];

function appendLog(line) {
  logLines.push(line);
  if (logLines.length > MAX_LOG_LINES) logLines.shift();

  const panel = $('logPanel');
  const atBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 40;

  panel.innerHTML = logLines.length
    ? logLines.map(formatLogLine).join('')
    : '<div class="empty">No log lines yet.</div>';

  if (atBottom) panel.scrollTop = panel.scrollHeight;
}

function formatLogLine(line) {
  let cls = 'log-line';
  if (line.includes('[ERROR]')) cls += ' error';
  else if (line.includes('[WARNING]')) cls += ' warn';
  else if (line.includes('[INFO]')) cls += ' info';
  else if (line.includes('[DEBUG]')) cls += ' debug';

  const formatted = escHtml(line).replace(
    /(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2},\\d{3}) \\[(\\w+)\\] ([\\w.]+):/,
    (_, ts, level, mod) =>
      `<span style="color:rgba(255,255,255,0.48)">${ts}</span> ` +
      `[${level}] <span class="log-mod">${mod}</span>:`
  );
  return `<div class="${cls}">${formatted}</div>`;
}

function setPanelHTML(panel, html) {
  const previousTop = panel.scrollTop;
  const wasNearBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 18;
  panel.innerHTML = html;

  if (wasNearBottom) {
    panel.scrollTop = panel.scrollHeight;
    return;
  }

  panel.scrollTop = Math.min(previousTop, Math.max(0, panel.scrollHeight - panel.clientHeight));
}

let chatEmpty = true;

$('chatInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});

$('sendBtn').addEventListener('click', sendChat);

async function sendChat() {
  const input = $('chatInput');
  const text = input.value.trim();
  if (!text || !online) return;

  input.value = '';
  $('sendBtn').disabled = true;

  if (chatEmpty) {
    $('chatMessages').innerHTML = '';
    chatEmpty = false;
  }

  appendMsg('user', 'You', text);

  // Show thinking indicator
  const thinkingId = 'thinking-' + Date.now();
  appendThinking(thinkingId);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    removeThinking(thinkingId);
    const data = await res.json();
    if (data.error) {
      appendMsg('brain', 'Jarvis', `${data.error}`);
    } else {
      const reply = data.reply || '';
      const meta = data.meta || {};
      if (reply) {
        appendMsg('brain', 'Jarvis', reply);
      } else {
        appendMsg(
          'brain',
          'Jarvis',
          `[Cycle ${data.cycle}] Processed your message through all cognitive phases.`,
        );
      }
      const conf = meta.confidence;
      if (conf !== undefined) {
        appendMeta(`confidence ${(conf * 100).toFixed(0)}%  ·  cycle ${data.cycle}`);
      }
    }
  } catch (err) {
    removeThinking(thinkingId);
    appendMsg('brain', 'Jarvis', `Connection error: ${err.message}`);
  }

  $('sendBtn').disabled = !online;
  input.focus();
}

function appendThinking(id) {
  const panel = $('chatMessages');
  const div = document.createElement('div');
  div.id = id;
  div.className = 'msg brain thinking-msg';
  div.innerHTML =
    '<div class="msg-role">Jarvis</div>' +
    '<div class="thinking-dots"><span></span><span></span><span></span></div>';
  panel.appendChild(div);
  panel.scrollTop = panel.scrollHeight;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendMsg(role, label, text) {
  const panel = $('chatMessages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="msg-role">${escHtml(label)}</div>
    <div class="msg-text">${escHtml(text)}</div>`;
  panel.appendChild(div);
  panel.scrollTop = panel.scrollHeight;
}

function appendMeta(text) {
  const panel = $('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg-meta';
  div.textContent = text;
  panel.appendChild(div);
  panel.scrollTop = panel.scrollHeight;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function timeAgo(isoStr) {
  try {
    const d = new Date(isoStr);
    const secs = Math.floor((Date.now() - d) / 1000);
    if (secs < 5) return 'just now';
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
  } catch {
    return '';
  }
}

function formatDuration(secs) {
  secs = Math.floor(secs);
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatActivity(activity) {
  const value = String(activity || 'activity').replace(/_/g, ' ');
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function activityClass(activity) {
  return String(activity || '').toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
}

function clampProgress(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  if (num < 0) return 0;
  if (num > 1) return 1;
  return num;
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# SSE event stream
# ---------------------------------------------------------------------------


def _socket_path_for(request: web.Request | None = None) -> Path:
    if request is None:
        return SOCKET_PATH
    return Path(request.app.get("socket_path", SOCKET_PATH))


def _client_for(request: web.Request | None = None) -> DaemonClient:
    return DaemonClient(_socket_path_for(request))


def _state_dir() -> Path:
    return DaemonConfig().state_path


def _resolve_managed_file(name: str) -> tuple[Path, dict[str, str]]:
    meta = _MANAGED_FILES.get(name)
    if meta is None:
        raise ValueError(f"unknown managed file: {name}")

    state_dir = _state_dir()
    JarvisIdentity(state_dir)
    if name == "privacy_rules.json":
        load_privacy_rules(state_dir)
    return state_dir / name, meta


def _validate_managed_file_content(name: str, content: str) -> None:
    meta = _MANAGED_FILES.get(name)
    if meta is None:
        raise ValueError(f"unknown managed file: {name}")
    if meta["format"] == "json":
        json.loads(content)


def _managed_file_payload(name: str, path: Path, meta: dict[str, str]) -> dict[str, Any]:
    exists = path.exists()
    content = path.read_text(encoding="utf-8") if exists else ""
    stat = path.stat() if exists else None
    return {
        "name": name,
        "label": meta["label"],
        "description": meta["description"],
        "format": meta["format"],
        "path": str(path),
        "exists": exists,
        "bytes": stat.st_size if stat else 0,
        "modified_at": stat.st_mtime if stat else None,
        "content": content,
    }


def _unified_diff(before: str, after: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{name} (saved)",
            tofile=f"{name} (draft)",
        )
    )

async def sse_stream(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events endpoint — pushes status, thoughts, goals, and log lines."""
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)

    client = _client_for(request)
    log_offset = _get_log_tail_offset(200)

    async def send(event: str, data: object) -> None:
        payload = json.dumps(data) if not isinstance(data, str) else data
        await resp.write(f"event: {event}\ndata: {payload}\n\n".encode())

    try:
        while True:
            # --- Daemon IPC poll ---
            try:
                status = await asyncio.wait_for(client.status(), timeout=5.0)
                thoughts = await asyncio.wait_for(client.thoughts(limit=20), timeout=5.0)
                goals = await asyncio.wait_for(client.list_goals(), timeout=5.0)

                await send("status", status)
                await send("thoughts", thoughts)
                await send("goals", goals)

            except Exception as exc:
                logger.debug("SSE poll failed: %s", exc)
                await send(
                    "status",
                    {
                        "connection_error": str(exc),
                        "daemon": {
                            "online": False,
                            "autonomy_level": "?",
                            "total_cycles": 0,
                            "total_idle_ticks": 0,
                        },
                        "brain": {"active_goals": []},
                        "pending_approvals": [],
                        "proactive_inbox": [],
                        "channels": {"telegram": {}},
                    },
                )

            # --- Log tail ---
            new_lines, log_offset = _read_new_log_lines(log_offset)
            for line in new_lines[-50:]:  # max 50 lines per push
                await send("log", line)

            await asyncio.sleep(3)

    except (ConnectionResetError, asyncio.CancelledError):
        pass

    return resp


def _get_log_tail_offset(n_lines: int) -> int:
    """Return file offset that starts roughly n_lines from the end."""
    if not LOG_PATH.exists():
        return 0
    size = LOG_PATH.stat().st_size
    # Read last ~100KB and count lines
    chunk = min(size, 100_000)
    with open(LOG_PATH, "rb") as f:
        f.seek(max(0, size - chunk))
        data = f.read()
    lines = data.split(b"\n")
    if len(lines) <= n_lines:
        return 0
    # Find offset of the (len-n_lines)th line from the end
    target_lines = lines[-(n_lines + 1):]
    offset = size - len(b"\n".join(target_lines))
    return max(0, offset)


def _read_new_log_lines(offset: int) -> tuple[list[str], int]:
    """Read any new lines added to the log since offset."""
    if not LOG_PATH.exists():
        return [], offset
    size = LOG_PATH.stat().st_size
    if size <= offset:
        return [], offset
    with open(LOG_PATH, errors="replace") as f:
        f.seek(offset)
        data = f.read(size - offset)
    new_offset = offset + len(data.encode("utf-8", errors="replace"))
    lines = [line for line in data.splitlines() if line.strip()]
    return lines, new_offset


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

async def chat_handler(request: web.Request) -> web.Response:
    """POST /chat — proxy to daemon IPC."""
    try:
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            return web.json_response({"error": "message is required"}, status=400)

        client = _client_for(request)
        result = await asyncio.wait_for(client.chat(message), timeout=300.0)
        return web.json_response(result)

    except TimeoutError:
        return web.json_response(
            {"error": "Still thinking... (model is slow, try again in a moment)"},
            status=504,
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def goals_add_handler(request: web.Request) -> web.Response:
    """POST /api/goals — add a new goal via daemon IPC."""
    try:
        body = await request.json()
        description = body.get("description", "").strip()
        priority = float(body.get("priority", 0.5))
        if not description:
            return web.json_response({"error": "description is required"}, status=400)

        client = _client_for(request)
        result = await asyncio.wait_for(
            client.add_goal(description=description, priority=priority),
            timeout=10.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def approve_handler(request: web.Request) -> web.Response:
    """POST /api/approve — approve a pending daemon action."""
    try:
        body = await request.json()
        action_id = body.get("action_id", "").strip()
        if not action_id:
            return web.json_response({"error": "action_id is required"}, status=400)

        client = _client_for(request)
        result = await asyncio.wait_for(client.approve(action_id), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def goals_update_status_handler(request: web.Request) -> web.Response:
    """POST /api/goals/status — update goal lifecycle status."""
    try:
        body = await request.json()
        goal_id = body.get("goal_id", "").strip()
        status = body.get("status", "").strip()
        if not goal_id or not status:
            return web.json_response(
                {"error": "goal_id and status are required"},
                status=400,
            )

        client = _client_for(request)
        result = await asyncio.wait_for(
            client.update_goal_status(goal_id=goal_id, status=status),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def goals_update_handler(request: web.Request) -> web.Response:
    """POST /api/goals/update — edit a goal."""
    try:
        body = await request.json()
        goal_id = body.get("goal_id", "").strip()
        if not goal_id:
            return web.json_response({"error": "goal_id is required"}, status=400)

        payload: dict[str, Any] = {"goal_id": goal_id}
        if "description" in body:
            payload["description"] = str(body.get("description", ""))
        if "priority" in body:
            payload["priority"] = float(body.get("priority", 0.5))
        if "success_criteria" in body:
            payload["success_criteria"] = str(body.get("success_criteria", ""))

        client = _client_for(request)
        result = await asyncio.wait_for(client.update_goal(**payload), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def deny_handler(request: web.Request) -> web.Response:
    """POST /api/deny — deny a pending daemon action."""
    try:
        body = await request.json()
        action_id = body.get("action_id", "").strip()
        if not action_id:
            return web.json_response({"error": "action_id is required"}, status=400)

        client = _client_for(request)
        result = await asyncio.wait_for(client.deny(action_id), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def pending_clear_handler(request: web.Request) -> web.Response:
    """POST /api/pending/clear — clear pending approvals."""
    try:
        client = _client_for(request)
        result = await asyncio.wait_for(client.clear_pending(), timeout=10.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def chat_clear_handler(request: web.Request) -> web.Response:
    """POST /api/chat/clear — clear daemon chat history."""
    try:
        client = _client_for(request)
        result = await asyncio.wait_for(client.call("chat.clear"), timeout=10.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def shutdown_handler(request: web.Request) -> web.Response:
    """POST /api/shutdown — request daemon shutdown."""
    try:
        client = _client_for(request)
        result = await asyncio.wait_for(client.shutdown(), timeout=10.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def inbox_mark_read_handler(request: web.Request) -> web.Response:
    """POST /api/inbox/mark-read — mark proactive inbox messages as read."""
    try:
        client = _client_for(request)
        body = await request.json() if request.can_read_body else {}
        message_id = str(body.get("message_id", "")).strip() or None
        result = await asyncio.wait_for(client.mark_inbox_read(message_id), timeout=10.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def files_list_handler(request: web.Request) -> web.Response:
    """GET /api/files/list — list editable managed files."""
    try:
        entries = []
        for name, _meta in _MANAGED_FILES.items():
            path, resolved_meta = _resolve_managed_file(name)
            payload = _managed_file_payload(name, path, resolved_meta)
            payload.pop("content", None)
            entries.append(payload)
        return web.json_response({"files": entries})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def file_read_handler(request: web.Request) -> web.Response:
    """GET /api/files?name=... — read a managed file."""
    name = request.rel_url.query.get("name", "").strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    try:
        path, meta = _resolve_managed_file(name)
        return web.json_response(_managed_file_payload(name, path, meta))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def file_diff_handler(request: web.Request) -> web.Response:
    """POST /api/files/diff — preview diff before saving a managed file."""
    try:
        body = await request.json()
        name = body.get("name", "").strip()
        content = str(body.get("content", ""))
        if not name:
            return web.json_response({"error": "name is required"}, status=400)

        path, _meta = _resolve_managed_file(name)
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        diff = _unified_diff(before, content, name)
        return web.json_response({"name": name, "diff": diff})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def file_write_handler(request: web.Request) -> web.Response:
    """POST /api/files — write a managed file."""
    try:
        body = await request.json()
        name = body.get("name", "").strip()
        content = str(body.get("content", ""))
        if not name:
            return web.json_response({"error": "name is required"}, status=400)

        path, meta = _resolve_managed_file(name)
        _validate_managed_file_content(name, content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return web.json_response(_managed_file_payload(name, path, meta))
    except json.JSONDecodeError as exc:
        return web.json_response({"error": f"invalid JSON content: {exc}"}, status=400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def memory_recent_handler(request: web.Request) -> web.Response:
    """GET /api/memory/recent — recent episodic memories."""
    try:
        limit = int(request.rel_url.query.get("limit", "20"))
        client = _client_for(request)
        result = await asyncio.wait_for(client.memory_recent(limit=limit), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"items": [], "error": str(exc)}, status=502)


async def timeline_handler(request: web.Request) -> web.Response:
    """GET /api/timeline — recent daemon activity timeline."""
    try:
        limit = int(request.rel_url.query.get("limit", "40"))
        client = _client_for(request)
        result = await asyncio.wait_for(client.timeline_recent(limit=limit), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"items": [], "error": str(exc)}, status=502)


async def autonomy_set_handler(request: web.Request) -> web.Response:
    """POST /api/settings/autonomy — update daemon autonomy level."""
    try:
        body = await request.json()
        level = body.get("level", "").strip()
        if not level:
            return web.json_response({"error": "level is required"}, status=400)
        client = _client_for(request)
        result = await asyncio.wait_for(client.set_autonomy_level(level), timeout=10.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def telegram_test_handler(request: web.Request) -> web.Response:
    """POST /api/telegram/test — send a test message to the paired Telegram chat."""
    try:
        daemon_config = DaemonConfig()
        token = daemon_config.telegram_token or os.environ.get("JARVIS_TELEGRAM_TOKEN", "")
        pair_file = daemon_config.state_path / "telegram_chat_id.txt"
        if not token:
            return web.json_response({"error": "telegram token not configured"}, status=400)
        if not pair_file.exists():
            return web.json_response({"error": "telegram chat not paired"}, status=400)

        chat_id = pair_file.read_text(encoding="utf-8").strip()
        body = await request.json() if request.can_read_body else {}
        message = str(body.get("message", "Jarvis test message from the web UI."))

        from telegram import Bot

        bot = Bot(token=token)
        await bot.send_message(chat_id=int(chat_id), text=message)
        return web.json_response({"ok": True, "chat_id": chat_id, "message": message})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def telegram_info_handler(request: web.Request) -> web.Response:
    """GET /api/telegram/info — inspect configured Telegram bot identity."""
    try:
        daemon_config = DaemonConfig()
        token = daemon_config.telegram_token or os.environ.get("JARVIS_TELEGRAM_TOKEN", "")
        pair_file = daemon_config.state_path / "telegram_chat_id.txt"
        if not token:
            return web.json_response(
                {"ok": False, "configured": False, "paired": pair_file.exists()},
            )

        from telegram import Bot

        bot = Bot(token=token)
        me = await bot.get_me()
        username = me.username or ""
        display_name = " ".join(part for part in [me.first_name, me.last_name] if part).strip()
        return web.json_response(
            {
                "ok": True,
                "configured": True,
                "paired": pair_file.exists(),
                "username": username,
                "display_name": display_name,
                "bot_url": f"https://t.me/{username}" if username else "",
            }
        )
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)


async def telegram_unpair_handler(request: web.Request) -> web.Response:
    """POST /api/telegram/unpair — remove the stored paired chat id."""
    try:
        daemon_config = DaemonConfig()
        pair_file = daemon_config.state_path / "telegram_chat_id.txt"
        pair_file.unlink(missing_ok=True)
        return web.json_response({"ok": True, "paired": False})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def memory_search_handler(request: web.Request) -> web.Response:
    """GET /api/memory/search?q=... — search episodic memory."""
    query = request.rel_url.query.get("q", "").strip()
    if not query:
        return web.json_response({"results": []})

    try:
        scope = request.rel_url.query.get("scope", "all")
        scope_id = request.rel_url.query.get("scope_id", "").strip() or None
        client = _client_for(request)
        result = await asyncio.wait_for(
            client.memory_search(query=query, top_k=10, scope=scope, scope_id=scope_id),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"results": [], "error": str(exc)}, status=502)


async def memory_recall_handler(request: web.Request) -> web.Response:
    """GET /api/memory/recall?q=... — profile-aware memory recall."""
    query = request.rel_url.query.get("q", "").strip()
    if not query:
        return web.json_response({"results": [], "profile": {}})

    try:
        top_k = int(request.rel_url.query.get("top_k", "10"))
        scope = request.rel_url.query.get("scope", "all")
        scope_id = request.rel_url.query.get("scope_id", "").strip() or None
        client = _client_for(request)
        result = await asyncio.wait_for(
            client.memory_recall(query=query, top_k=top_k, scope=scope, scope_id=scope_id),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"results": [], "profile": {}, "error": str(exc)}, status=502)


async def memory_hybrid_handler(request: web.Request) -> web.Response:
    """GET /api/memory/hybrid?q=... — hybrid recall across memory/goals/workspace."""
    query = request.rel_url.query.get("q", "").strip()
    if not query:
        return web.json_response({"hybrid_results": [], "profile": {}})

    try:
        top_k = int(request.rel_url.query.get("top_k", "10"))
        scope = request.rel_url.query.get("scope", "all")
        scope_id = request.rel_url.query.get("scope_id", "").strip() or None
        client = _client_for(request)
        result = await asyncio.wait_for(
            client.memory_hybrid(query=query, top_k=top_k, scope=scope, scope_id=scope_id),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response(
            {"hybrid_results": [], "profile": {}, "error": str(exc)},
            status=502,
        )


async def memory_clear_handler(request: web.Request) -> web.Response:
    """POST /api/memory/clear — wipe all episodic memories and reset vector index."""
    try:
        client = _client_for(request)
        result = await asyncio.wait_for(client.memory_clear(confirm=True), timeout=30.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def debug_db_snapshot_handler(request: web.Request) -> web.Response:
    """GET /api/debug/dbs — transparent DB counts and sample records."""
    try:
        limit = int(request.rel_url.query.get("limit", "5"))
        client = _client_for(request)
        result = await asyncio.wait_for(client.debug_db_snapshot(limit=limit), timeout=20.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def debug_clear_all_handler(request: web.Request) -> web.Response:
    """POST /api/debug/clear-all — clear all DBs/state for testing."""
    try:
        body = await request.json() if request.can_read_body else {}
        confirm = bool(body.get("confirm"))
        client = _client_for(request)
        result = await asyncio.wait_for(client.debug_clear_all(confirm=confirm), timeout=60.0)
        status = 200 if result.get("ok") else 400
        return web.json_response(result, status=status)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def memory_graph_handler(request: web.Request) -> web.Response:
    """GET /api/memory/graph — relationship graph for visible memory."""
    try:
        limit = int(request.rel_url.query.get("limit", "40"))
        scope = request.rel_url.query.get("scope", "all")
        scope_id = request.rel_url.query.get("scope_id", "").strip() or None
        client = _client_for(request)
        result = await asyncio.wait_for(
            client.memory_graph(limit=limit, scope=scope, scope_id=scope_id),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"nodes": [], "edges": [], "error": str(exc)}, status=502)


async def scenario_handler(request: web.Request) -> web.Response:
    """GET /api/scenario?q=... — bounded scenario analysis."""
    scenario = request.rel_url.query.get("q", "").strip()
    if not scenario:
        return web.json_response({"error": "scenario is required"}, status=400)

    try:
        scope = request.rel_url.query.get("scope", "all")
        scope_id = request.rel_url.query.get("scope_id", "").strip() or None
        client = _client_for(request)
        result = await asyncio.wait_for(
            client.run_scenario(scenario=scenario, scope=scope, scope_id=scope_id),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def report_handler(request: web.Request) -> web.Response:
    """GET /api/report?type=... — bounded weekly/project report."""
    report_type = request.rel_url.query.get("type", "weekly").strip()
    focus = request.rel_url.query.get("focus", "").strip()

    try:
        scope = request.rel_url.query.get("scope", "all")
        scope_id = request.rel_url.query.get("scope_id", "").strip() or None
        client = _client_for(request)
        result = await asyncio.wait_for(
            client.run_report(report_type=report_type, focus=focus, scope=scope, scope_id=scope_id),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def memory_profile_handler(request: web.Request) -> web.Response:
    """GET /api/memory/profile — structured profile + recent context."""
    try:
        client = _client_for(request)
        result = await asyncio.wait_for(client.memory_profile(), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def memory_get_handler(request: web.Request) -> web.Response:
    """GET /api/memory/item?id=... — fetch one or more full memory records."""
    ids = [value for value in request.rel_url.query.getall("id", []) if value.strip()]
    if not ids:
        return web.json_response({"items": [], "missing": []})

    try:
        client = _client_for(request)
        result = await asyncio.wait_for(client.memory_get(ids), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"items": [], "missing": ids, "error": str(exc)}, status=502)


async def memory_timeline_handler(request: web.Request) -> web.Response:
    """GET /api/memory/timeline?id=... — fetch nearby memories around one anchor."""
    anchor_id = request.rel_url.query.get("id", "").strip()
    if not anchor_id:
        return web.json_response({"anchor_id": "", "items": []})

    try:
        limit = int(request.rel_url.query.get("limit", "6"))
        client = _client_for(request)
        result = await asyncio.wait_for(
            client.memory_timeline(anchor_id=anchor_id, limit=limit),
            timeout=15.0,
        )
        return web.json_response(result)
    except Exception as exc:
        return web.json_response(
            {"anchor_id": anchor_id, "items": [], "error": str(exc)},
            status=502,
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

async def index_handler(_request: web.Request) -> web.Response:
    """Serve the React web UI when built, falling back to the legacy inline shell."""
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return web.FileResponse(index_path)
    return web.Response(text=HTML, content_type="text/html")


def create_app(socket_path: Path | None = None) -> web.Application:
    app = web.Application()
    if socket_path:
        app["socket_path"] = socket_path
    app.router.add_get("/", index_handler)
    assets_path = FRONTEND_DIST / "assets"
    if assets_path.exists():
        app.router.add_static("/assets", assets_path, name="frontend_assets")
    app.router.add_get("/events", sse_stream)
    app.router.add_post("/chat", chat_handler)
    app.router.add_post("/api/goals", goals_add_handler)
    app.router.add_post("/api/goals/update", goals_update_handler)
    app.router.add_post("/api/goals/status", goals_update_status_handler)
    app.router.add_post("/api/approve", approve_handler)
    app.router.add_post("/api/deny", deny_handler)
    app.router.add_post("/api/pending/clear", pending_clear_handler)
    app.router.add_post("/api/chat/clear", chat_clear_handler)
    app.router.add_post("/api/shutdown", shutdown_handler)
    app.router.add_post("/api/inbox/mark-read", inbox_mark_read_handler)
    app.router.add_get("/api/files/list", files_list_handler)
    app.router.add_get("/api/files", file_read_handler)
    app.router.add_post("/api/files/diff", file_diff_handler)
    app.router.add_post("/api/files", file_write_handler)
    app.router.add_get("/api/memory/recent", memory_recent_handler)
    app.router.add_get("/api/timeline", timeline_handler)
    app.router.add_post("/api/settings/autonomy", autonomy_set_handler)
    app.router.add_get("/api/telegram/info", telegram_info_handler)
    app.router.add_post("/api/telegram/test", telegram_test_handler)
    app.router.add_post("/api/telegram/unpair", telegram_unpair_handler)
    app.router.add_get("/api/memory/search", memory_search_handler)
    app.router.add_get("/api/memory/recall", memory_recall_handler)
    app.router.add_get("/api/memory/hybrid", memory_hybrid_handler)
    app.router.add_get("/api/memory/graph", memory_graph_handler)
    app.router.add_post("/api/memory/clear", memory_clear_handler)
    app.router.add_get("/api/debug/dbs", debug_db_snapshot_handler)
    app.router.add_post("/api/debug/clear-all", debug_clear_all_handler)
    app.router.add_get("/api/scenario", scenario_handler)
    app.router.add_get("/api/report", report_handler)
    app.router.add_get("/api/memory/profile", memory_profile_handler)
    app.router.add_get("/api/memory/item", memory_get_handler)
    app.router.add_get("/api/memory/timeline", memory_timeline_handler)
    return app


async def run_webui(
    host: str = "0.0.0.0",
    port: int = PORT,
    socket_path: Path | None = None,
) -> None:
    """Start the web UI as an async task — for embedding inside the daemon."""
    import socket as _socket
    app = create_app(socket_path=socket_path)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    # Log all local IPs so the user knows how to connect from their phone
    try:
        hostname = _socket.gethostname()
        local_ip = _socket.gethostbyname(hostname)
    except Exception:
        local_ip = "your-machine-ip"

    logger.info(
        "Jarvis Web UI listening on http://%s:%d  (LAN: http://%s:%d)",
        host, port, local_ip, port,
    )
    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


def main(host: str = "0.0.0.0", port: int = PORT) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = create_app()

    try:
        import socket as _socket
        local_ip = _socket.gethostbyname(_socket.gethostname())
    except Exception:
        local_ip = "your-machine-ip"

    print("Jarvis Web UI:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}  ← open this on your phone")
    print(f"  Socket:  {SOCKET_PATH}")
    web.run_app(app, host=host, port=port, access_log=None)


if __name__ == "__main__":
    main()
