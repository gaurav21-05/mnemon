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
import json
import logging
from pathlib import Path

from aiohttp import web

from mnemon.daemon.cli.client import DaemonClient

logger = logging.getLogger(__name__)

SOCKET_PATH = Path("~/.mnemon/daemon.sock").expanduser()
LOG_PATH = Path("~/.mnemon/daemon.log").expanduser()
PORT = 7777

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mnemon — Jarvis Control Room</title>
<style>
  :root {
    --cream: #faf9f7;
    --paper: #ffffff;
    --oat: #dad4c8;
    --oat-light: #eee9df;
    --ink: #000000;
    --warm-silver: #9f9b93;
    --warm-charcoal: #55534e;
    --matcha-300: #84e7a5;
    --matcha-600: #078a52;
    --matcha-800: #02492a;
    --slushie-500: #3bd3fd;
    --slushie-800: #0089ad;
    --lemon-400: #f8cc65;
    --lemon-500: #fbbd41;
    --lemon-700: #d08a11;
    --ube-300: #c1b0ff;
    --ube-800: #43089f;
    --pomegranate-400: #fc7981;
    --blueberry-800: #01418d;
    --focus: rgb(20, 110, 245);
    --clay-shadow:
      rgba(0, 0, 0, 0.1) 0 1px 1px,
      rgba(0, 0, 0, 0.04) 0 -1px 1px inset,
      rgba(0, 0, 0, 0.05) 0 -0.5px 1px;
    --hard-shadow: rgb(0, 0, 0) -7px 7px 0;
    --font-sans: "Roobert", "Avenir Next", "Trebuchet MS", Arial, sans-serif;
    --font-mono: "Space Mono", "SFMono-Regular", Consolas, monospace;
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
    background: radial-gradient(circle at top left, rgba(251, 189, 65, 0.22), transparent 28%),
      radial-gradient(circle at top right, rgba(59, 211, 253, 0.2), transparent 30%),
      linear-gradient(180deg, #fffdf8 0%, var(--cream) 28%, #f7f2e7 100%);
    color: var(--ink);
    font-family: var(--font-sans);
    font-size: 16px;
    line-height: 1.5;
    font-feature-settings: "ss03" 1, "ss10" 1, "ss11" 1, "ss12" 1;
    overflow: hidden;
  }

  body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
      linear-gradient(rgba(218, 212, 200, 0.2) 1px, transparent 1px),
      linear-gradient(90deg, rgba(218, 212, 200, 0.2) 1px, transparent 1px);
    background-size: 36px 36px;
    mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.35), transparent 72%);
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
    backdrop-filter: blur(14px);
    background: rgba(250, 249, 247, 0.9);
    border-bottom: 1px solid var(--oat);
  }

  .topbar-inner {
    width: min(1380px, calc(100% - 32px));
    margin: 0 auto;
    padding: 18px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }

  .brand {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 10px 18px;
    border: 1px solid var(--oat);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.76);
    box-shadow: var(--clay-shadow);
  }

  .brand-copy {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .brand-title {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.04em;
    line-height: 1;
    font-feature-settings: "ss01" 1, "ss03" 1, "ss10" 1, "ss11" 1, "ss12" 1;
  }

  .brand-subtitle {
    color: var(--warm-silver);
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
    border: 1px solid var(--oat);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: var(--clay-shadow);
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
    min-width: 88px;
    padding: 8px 12px;
    border-radius: 18px;
    border: 1px solid var(--oat);
    background: rgba(255, 255, 255, 0.82);
    box-shadow: var(--clay-shadow);
  }

  .nav-stat-label {
    color: var(--warm-silver);
    font-size: 10px;
    line-height: 1.1;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 600;
  }

  .nav-stat-value {
    color: var(--ink);
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
    padding: 10px 16px;
    border: 1px solid rgba(7, 138, 82, 0.18);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.92);
    box-shadow: var(--clay-shadow);
    white-space: nowrap;
  }

  .status-pill.offline {
    border-color: rgba(252, 121, 129, 0.3);
    background: rgba(255, 255, 255, 0.9);
  }

  .status-pill.online {
    border-color: rgba(7, 138, 82, 0.28);
    box-shadow: 0 0 0 3px rgba(7, 138, 82, 0.08), var(--clay-shadow);
  }

  .status-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    flex-shrink: 0;
    background: var(--pomegranate-400);
    transition: background 0.3s ease, box-shadow 0.3s ease;
  }

  .status-dot.online {
    background: var(--matcha-600);
    box-shadow: 0 0 0 4px rgba(7, 138, 82, 0.16);
  }

  .status-copy {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .status-title {
    color: var(--ink);
    font-size: 13px;
    line-height: 1.15;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
  }

  .status-label {
    color: var(--matcha-600);
    font-size: 11px;
    line-height: 1.1;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
  }

  .status-pill.offline .status-label {
    color: #b33d47;
  }

  .page {
    width: min(1380px, calc(100% - 32px));
    margin: 0 auto;
    padding: 16px 0 20px;
    height: calc(100vh - 86px);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: 16px;
  }

  .panel-shell {
    position: relative;
    overflow: hidden;
    border-radius: 40px;
    border: 1px solid var(--oat);
    box-shadow: var(--clay-shadow);
  }

  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    color: var(--warm-silver);
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
    border: 1px solid rgba(0, 0, 0, 0.08);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
    line-height: 1;
    box-shadow: var(--clay-shadow);
  }

  .badge.green {
    background: var(--matcha-300);
    color: var(--matcha-800);
  }

  .badge.orange {
    background: rgba(251, 189, 65, 0.32);
    color: var(--lemon-700);
  }

  .badge {
    background: rgba(193, 176, 255, 0.32);
    color: var(--ube-800);
  }

  .offline-banner {
    display: none;
    width: 100%;
    padding: 12px 18px;
    border: 1px dashed rgba(0, 0, 0, 0.18);
    border-radius: 999px;
    background: rgba(252, 121, 129, 0.18);
    color: #7f1f2a;
    text-align: center;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: var(--clay-shadow);
  }

  .offline-banner.show {
    display: block;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: minmax(280px, 0.92fr) minmax(360px, 1.15fr) minmax(320px, 1fr);
    grid-template-rows: minmax(0, 1fr) minmax(0, 0.92fr);
    grid-template-areas:
      "log thoughts chat"
      "goals thoughts chat";
    gap: 24px;
    align-items: stretch;
    min-height: 0;
  }

  .panel-shell {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .goals-shell {
    grid-area: goals;
    border-style: dashed;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(250, 249, 247, 0.92));
  }

  .thoughts-shell {
    grid-area: thoughts;
    background: linear-gradient(155deg, rgba(59, 211, 253, 0.85), rgba(248, 249, 247, 0.96) 75%);
  }

  .log-shell {
    grid-area: log;
    background: linear-gradient(160deg, var(--matcha-800), #01311c 78%);
    border-color: rgba(255, 255, 255, 0.14);
  }

  .chat-shell {
    grid-area: chat;
    background: linear-gradient(145deg, rgba(193, 176, 255, 0.76), rgba(255, 255, 255, 0.94) 78%);
  }

  .section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 16px 18px 0;
    flex-shrink: 0;
  }

  .section-title {
    font-size: clamp(1.2rem, 2vw, 1.55rem);
    line-height: 1.12;
    letter-spacing: -0.04em;
    font-weight: 600;
    font-feature-settings: "ss01" 1, "ss03" 1, "ss10" 1, "ss11" 1, "ss12" 1;
  }

  .log-shell .section-title {
    color: #ffffff;
  }

  .panel-body,
  .chat-messages,
  .log-panel-body {
    padding: 14px 18px 18px;
    overflow: auto;
    min-height: 0;
    scrollbar-width: thin;
    scrollbar-color: rgba(0, 0, 0, 0.26) transparent;
  }

  .panel-body::-webkit-scrollbar,
  .chat-messages::-webkit-scrollbar,
  .log-panel-body::-webkit-scrollbar {
    width: 8px;
  }

  .panel-body::-webkit-scrollbar-thumb,
  .chat-messages::-webkit-scrollbar-thumb,
  .log-panel-body::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.16);
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
    border-radius: 20px;
    padding: 14px;
    background: rgba(255, 255, 255, 0.82);
    box-shadow: var(--clay-shadow);
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
    border: 1px dashed rgba(0, 0, 0, 0.16);
  }

  .thought.reflection {
    background: linear-gradient(180deg, rgba(193, 176, 255, 0.38), rgba(255, 255, 255, 0.92));
  }

  .thought.consolidation {
    background: linear-gradient(180deg, rgba(132, 231, 165, 0.42), rgba(255, 255, 255, 0.92));
  }

  .thought.planning {
    background: linear-gradient(180deg, rgba(251, 189, 65, 0.36), rgba(255, 255, 255, 0.92));
  }

  .thought.exploration {
    background: linear-gradient(180deg, rgba(59, 211, 253, 0.34), rgba(255, 255, 255, 0.92));
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
    background: rgba(255, 255, 255, 0.7);
    color: var(--ink);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
  }

  .thought-time,
  .goal-progress-label {
    color: var(--warm-charcoal);
    font-size: 12px;
    font-family: var(--font-mono);
  }

  .thought-summary,
  .goal-desc {
    font-size: 14px;
    line-height: 1.45;
    color: var(--ink);
  }

  .goal {
    border: 1px solid var(--oat);
  }

  .goal.has-subgoals {
    border-style: dashed;
  }

  .goal-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
  }

  .progress-bar {
    flex: 1;
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--oat-light);
  }

  .progress-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--ube-800), var(--pomegranate-400));
    transition: width 0.4s ease;
  }

  .subgoals {
    margin-top: 12px;
    padding-left: 12px;
    border-left: 2px dashed var(--oat);
  }

  .subgoal {
    font-size: 13px;
    line-height: 1.5;
    color: var(--warm-charcoal);
    padding: 4px 0;
  }

  .log-panel-body {
    margin: 14px 18px 18px;
    border-radius: 20px;
    border: 1px dashed rgba(255, 255, 255, 0.16);
    background: rgba(255, 255, 255, 0.06);
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
    color: var(--lemon-400);
  }

  .log-line.error {
    color: #ffd4d7;
  }

  .log-line.debug {
    color: rgba(255, 255, 255, 0.46);
  }

  .log-line .log-mod {
    color: var(--slushie-500);
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
    padding: 12px 18px;
    border-radius: 18px;
    border: 1px solid var(--oat);
    background: rgba(255,255,255,0.78);
    box-shadow: var(--clay-shadow);
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }
  .thinking-dots span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--warm-charcoal);
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
    color: var(--warm-charcoal);
    font-size: 12px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 600;
  }

  .msg-text {
    padding: 12px 14px;
    border-radius: 18px;
    border: 1px solid var(--oat);
    background: rgba(255, 255, 255, 0.78);
    box-shadow: var(--clay-shadow);
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.55;
  }

  .msg.user .msg-text {
    background: linear-gradient(135deg, var(--blueberry-800), #1d5fb8);
    border-color: transparent;
    color: #ffffff;
  }

  .msg.user .msg-role {
    text-align: right;
  }

  .msg-meta {
    align-self: flex-end;
    color: var(--warm-charcoal);
    font-family: var(--font-mono);
    font-size: 11px;
  }

  .chat-input-row {
    display: flex;
    gap: 12px;
    padding: 0 18px 18px;
    align-items: stretch;
    flex-shrink: 0;
  }

  .chat-input {
    flex: 1;
    min-height: 46px;
    border-radius: 16px;
    border: 1px solid #717989;
    background: rgba(255, 255, 255, 0.82);
    padding: 14px 16px;
    color: var(--ink);
    outline: none;
    box-shadow: var(--clay-shadow);
  }

  .chat-input::placeholder {
    color: var(--warm-silver);
  }

  .chat-input:focus {
    border-color: var(--focus);
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }

  .cta-button,
  .send-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 44px;
    padding: 6.4px 16px;
    border-radius: 999px;
    border: 1px solid transparent;
    background: #ffffff;
    color: var(--ink);
    font-size: 16px;
    font-weight: 500;
    letter-spacing: -0.01em;
    cursor: pointer;
    box-shadow: var(--clay-shadow);
    transition:
      transform 0.18s ease,
      box-shadow 0.18s ease,
      background-color 0.18s ease,
      color 0.18s ease,
      border-color 0.18s ease;
  }

  .cta-button:hover,
  .send-btn:hover {
    transform: translateY(-2px) rotate(-1.5deg);
    box-shadow: rgba(0, 0, 0, 0.18) -4px 4px 0;
  }

  .cta-button:focus-visible,
  .send-btn:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 3px;
  }

  .cta-button.primary,
  .send-btn {
    background: rgba(255, 255, 255, 0.92);
  }

  .cta-button.primary:hover,
  .send-btn:hover {
    background: var(--matcha-600);
    color: #ffffff;
  }

  .cta-button.secondary {
    border-color: #717989;
    background: transparent;
  }

  .cta-button.secondary:hover {
    background: var(--ube-800);
    color: #ffffff;
  }

  .send-btn:disabled {
    cursor: not-allowed;
    opacity: 0.55;
    transform: none;
    box-shadow: var(--clay-shadow);
    background: rgba(255, 255, 255, 0.72);
    color: var(--warm-silver);
  }

  .empty {
    padding: 18px;
    border-radius: 18px;
    border: 1px dashed var(--oat);
    background: rgba(255, 255, 255, 0.56);
    color: var(--warm-charcoal);
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
    border-bottom: 1px solid var(--oat-light);
    font-size: 13px;
    line-height: 1.45;
  }

  .search-result:last-child {
    border-bottom: none;
  }

  .search-result .search-text {
    color: var(--ink);
  }

  .search-score {
    display: inline-block;
    min-width: 40px;
    margin-right: 6px;
    color: var(--matcha-600);
    font-size: 11px;
    font-weight: 700;
    font-family: var(--font-mono);
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
      grid-template-rows: minmax(0, 0.75fr) minmax(0, 0.95fr) minmax(0, 0.95fr);
      grid-template-areas:
        "log thoughts"
        "goals thoughts"
        "chat chat";
    }

  }

  @media (max-width: 900px) {
    .dashboard-grid {
      grid-template-columns: 1fr;
      grid-template-rows: auto;
      grid-template-areas:
        "status"
        "thoughts"
        "goals"
        "log"
        "chat";
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

    .nav-stats {
      width: 100%;
    }

    .nav-stat {
      flex: 1 1 120px;
    }

    .panel-shell {
      border-radius: 28px;
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
        <div class="brand-title">Mnemon Jarvis</div>
        <div class="brand-subtitle">Warm control room for the daemon</div>
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
        <div class="memory-search-results" id="memSearchResults"></div>
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

  <section class="dashboard-grid">
    <article class="panel-shell thoughts-shell" id="thoughtsSection">
      <div class="section-head">
        <h2 class="section-title">Idle Thoughts</h2>
        <span id="tickCount" class="badge green">0 ticks</span>
      </div>
      <div class="panel-body" id="thoughtsPanel">
        <div class="empty">Waiting for the first thought loop…</div>
      </div>
    </article>

    <article class="panel-shell goals-shell">
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

    <article class="panel-shell log-shell">
      <div class="section-head">
        <h2 class="section-title">Live Log</h2>
        <span class="badge green">tail</span>
      </div>
      <div class="log-panel-body" id="logPanel">
        <div class="empty">Loading log tail…</div>
      </div>
    </article>

    <article class="panel-shell chat-shell">
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
  </section>
</main>

<script>
const $ = id => document.getElementById(id);

let source;
let online = false;
const renderCache = {
  thoughts: '',
  goals: '',
};

function connect() {
  source = new EventSource('/events');

  source.addEventListener('status', e => {
    const data = JSON.parse(e.data);
    renderStatus(data);
    setOnline(true);
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
    setOnline(false);
    source.close();
    setTimeout(connect, 3000);
  });
}

connect();

function setOnline(v) {
  online = v;
  $('statusDotNav').className = 'status-dot' + (v ? ' online' : '');
  $('statusPill').className = 'status-pill ' + (v ? 'online' : 'offline');
  $('statusLabel').textContent = v ? 'online and streaming' : 'reconnecting';
  $('offlineBanner').className = 'offline-banner' + (v ? '' : ' show');
  $('sendBtn').disabled = !v;
  if ($('goalSubmitBtn')) $('goalSubmitBtn').disabled = !v;
}

function renderStatus(s) {
  const d = s.daemon || {};
  const started = d.started_at ? new Date(d.started_at) : null;
  const uptime = started ? formatDuration((Date.now() - started) / 1000) : '?';
  $('navUptime').textContent = uptime;
  $('navCycles').textContent = d.total_cycles ?? '?';
  $('navTicks').textContent = d.total_idle_ticks ?? '?';
  $('navAutonomy').textContent = d.autonomy_level ?? '?';

  const idleTicks = d.total_idle_ticks ?? 0;
  $('tickCount').textContent = `${idleTicks} tick${idleTicks === 1 ? '' : 's'}`;
}

function renderThoughts(thoughts) {
  const serialized = JSON.stringify(thoughts);
  if (serialized === renderCache.thoughts) return;
  renderCache.thoughts = serialized;

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
      ${hasSubs ? `<div class="subgoals">${subs.map(s =>
        `<div class="subgoal">▸ ${escHtml(s.description)}</div>`).join('')}</div>` : ''}
    </article>`;
  }

  setPanelHTML($('goalsPanel'), topLevel.map(renderGoal).join(''));
}

const goalForm = $('goalForm');
const goalInput = $('goalInput');
const goalSubmitBtn = $('goalSubmitBtn');

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

const memSearchInput = $('memSearchInput');
const memSearchResults = $('memSearchResults');
let memSearchTimer = null;

if (memSearchInput && memSearchResults) {
  memSearchInput.addEventListener('input', () => {
    clearTimeout(memSearchTimer);
    memSearchTimer = setTimeout(runMemorySearch, 250);
  });

  memSearchInput.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      memSearchInput.value = '';
      setMemorySearchResults([]);
    }
  });

  document.addEventListener('click', event => {
    if (!memSearchResults.contains(event.target) && event.target !== memSearchInput) {
      memSearchResults.classList.remove('show');
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
    const response = await fetch(`/api/memory/search?q=${encodeURIComponent(query)}`);
    const data = await response.json();
    setMemorySearchResults(data.results || [], data.error || '');
  } catch (_error) {
    setMemorySearchResults([], 'Search failed');
  }
}

function setMemorySearchResults(items, errorText = '') {
  if (!memSearchResults) return;

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

  memSearchResults.innerHTML = items.map(item => {
    const score = Number(item.score || 0);
    const preview = String(item.content || '').slice(0, 220);
    return `<div class="search-result">
      <span class="search-score">${(score * 100).toFixed(0)}%</span>
      <span class="search-text">${escHtml(preview)}</span>
    </div>`;
  }).join('');
  memSearchResults.classList.add('show');
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

  $('sendBtn').disabled = false;
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


async def memory_search_handler(request: web.Request) -> web.Response:
    """GET /api/memory/search?q=... — search episodic memory."""
    query = request.rel_url.query.get("q", "").strip()
    if not query:
        return web.json_response({"results": []})

    try:
        client = _client_for(request)
        result = await asyncio.wait_for(client.memory_search(query=query, top_k=10), timeout=15.0)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"results": [], "error": str(exc)}, status=502)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(socket_path: Path | None = None) -> web.Application:
    app = web.Application()
    if socket_path:
        app["socket_path"] = socket_path
    app.router.add_get("/", lambda r: web.Response(text=HTML, content_type="text/html"))
    app.router.add_get("/events", sse_stream)
    app.router.add_post("/chat", chat_handler)
    app.router.add_post("/api/goals", goals_add_handler)
    app.router.add_get("/api/memory/search", memory_search_handler)
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
