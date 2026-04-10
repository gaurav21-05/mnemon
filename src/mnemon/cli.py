"""Simple interactive CLI for booting and chatting with the Mnemon memory system."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mnemon.core.config import MnemonConfig, load_config
from mnemon.core.models import RetrievalQuery
from mnemon.factory import MnemonFactory
from mnemon.providers.litellm_provider import LiteLLMProvider

logger = logging.getLogger(__name__)

_LOCAL_CHAT_MODEL = "ollama/llama3.2"
_LOCAL_EMBED_MODEL = "ollama/nomic-embed-text"
_LOCAL_EMBED_DIM = 768
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

SYSTEM_PROMPT = """\
You are a helpful assistant with a cognitive memory system. You remember past
conversations and facts you have learned. When relevant memories are available,
use them naturally to answer with better context.

Be concise, grounded, and conversational.
"""

HELP_TEXT = """
Available commands:
  /memories     Show recent episodic memories
  /facts        Show semantic knowledge triples
  /skills       Show learned procedural skills
  /state        Show cognitive state (working memory, goals, bus)
  /consolidate  Run memory consolidation (episodic -> semantic)
  /goals        Show active goals
  /help         Show this help message
  /quit         Exit the CLI
""".strip()

START_HELP_TEXT = """
Mnemon daemon launcher:
  mnemon start              Start the daemon in the background
  mnemon start --foreground Start the daemon in the foreground
  mnemon start --local      Start the daemon with local Ollama models

Once started:
  Web UI:       http://localhost:7777
  Daemon status: mnemon-daemon status
  Daemon chat:   mnemon-daemon chat
  Stop daemon:   mnemon-daemon stop
""".strip()


def _supports_color() -> bool:
    return (
        os.getenv("NO_COLOR") is None
        and hasattr(os.sys.stdout, "isatty")
        and os.sys.stdout.isatty()
    )


def _style(text: str, *, fg: str = "", bold: bool = False) -> str:
    if not _supports_color():
        return text
    codes: list[str] = []
    if bold:
        codes.append("1")
    palette = {
        "green": "32",
        "cyan": "36",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "red": "31",
        "dim": "2",
    }
    if fg:
        codes.append(palette[fg])
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _kv(label: str, value: str, *, tone: str = "cyan") -> str:
    return f"  {_style(label.ljust(16), fg='dim')}{_style(value, fg=tone, bold=True)}"


def _print_section(title: str) -> None:
    print(_style("=" * 64, fg="dim"))
    print(_style(title, fg="magenta", bold=True))
    print(_style("=" * 64, fg="dim"))


def _print_dashboard_footer() -> None:
    print(_style("-" * 64, fg="dim"))
    print(_style("Quick actions", fg="yellow", bold=True))
    print("  /help          show commands")
    print("  /state         show memory + goal state")
    print("  /memories      inspect episodic recall")
    print("  /consolidate   force learning pass")
    print("  Web UI         mnemon-daemon webui")
    print("  Daemon status  mnemon-daemon status")
    print(_style("-" * 64, fg="dim"))


def _run_with_spinner(command: list[str], label: str) -> bool:
    process = subprocess.Popen(command)  # noqa: S603
    if not hasattr(os.sys.stdout, "isatty") or not os.sys.stdout.isatty():
        return process.wait() == 0

    index = 0
    while True:
        status = process.poll()
        if status is not None:
            marker = (
                _style("✓", fg="green", bold=True)
                if status == 0
                else _style("✗", fg="red", bold=True)
            )
            print(f"\r{marker} {label}" + " " * 20)
            return status == 0
        frame = _style(_SPINNER_FRAMES[index % len(_SPINNER_FRAMES)], fg="blue", bold=True)
        print(f"\r{frame} {label}", end="", flush=True)
        index += 1
        time.sleep(0.1)


@dataclass(frozen=True)
class RuntimeDiagnosis:
    model: str
    embedding_model: str
    llm_env_var: str | None
    embedding_env_var: str | None
    missing_env_vars: tuple[str, ...]
    ollama_installed: bool
    ollama_running: bool
    local_runtime_required: bool
    missing_local_models: tuple[str, ...]

    @property
    def ok(self) -> bool:
        if self.missing_env_vars:
            return False
        if not self.local_runtime_required:
            return True
        return (
            self.ollama_installed
            and self.ollama_running
            and not self.missing_local_models
        )


def _uses_ollama(model: str) -> bool:
    return model.strip().lower().startswith("ollama/")


def _ollama_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _ollama_model_aliases(model: str) -> set[str]:
    aliases = {model}
    if ":" in model:
        aliases.add(model.split(":", 1)[0])
    else:
        aliases.add(f"{model}:latest")
    return aliases


def _provider_env_var(model: str) -> str | None:
    lowered = model.strip().lower()
    if lowered.startswith("ollama/"):
        return None
    if lowered.startswith("anthropic/") or lowered.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    if lowered.startswith("groq/"):
        return "GROQ_API_KEY"
    if lowered.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-")):
        return "OPENAI_API_KEY"
    return None


def _run_ollama_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ollama", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _ollama_running() -> bool:
    if shutil.which("ollama") is None:
        return False
    return _run_ollama_command("list").returncode == 0


def _list_ollama_models() -> set[str]:
    if not _ollama_running():
        return set()

    result = _run_ollama_command("list")
    if result.returncode != 0:
        return set()

    models: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        models.update(_ollama_model_aliases(stripped.split()[0]))
    return models


def _start_ollama_server(timeout_s: float = 8.0) -> bool:
    if shutil.which("ollama") is None:
        return False
    if _ollama_running():
        return True

    log_path = Path("~/.mnemon/ollama.log").expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.Popen(  # noqa: S603
            ["ollama", "serve"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.time() + timeout_s
    index = 0
    while time.time() < deadline:
        if _ollama_running():
            if hasattr(os.sys.stdout, "isatty") and os.sys.stdout.isatty():
                print(
                    "\r"
                    + _style("✓", fg="green", bold=True)
                    + " Started local Ollama server."
                    + " " * 20
                )
            return True
        if hasattr(os.sys.stdout, "isatty") and os.sys.stdout.isatty():
            frame = _style(_SPINNER_FRAMES[index % len(_SPINNER_FRAMES)], fg="blue", bold=True)
            print(f"\r{frame} Waiting for Ollama server", end="", flush=True)
        index += 1
        time.sleep(0.25)
    ok = _ollama_running()
    if hasattr(os.sys.stdout, "isatty") and os.sys.stdout.isatty():
        marker = _style("✓", fg="green", bold=True) if ok else _style("✗", fg="red", bold=True)
        print(f"\r{marker} Ollama server {'ready' if ok else 'not available'}" + " " * 20)
    return ok


def _pull_ollama_model(model_name: str) -> bool:
    if shutil.which("ollama") is None:
        return False
    return _run_with_spinner(["ollama", "pull", model_name], f"Pulling Ollama model {model_name}")


def _local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "your-machine-ip"


def diagnose_runtime(config: MnemonConfig) -> RuntimeDiagnosis:
    provider_name = config.llm.default_provider
    provider = config.llm.providers.get(provider_name, {})
    model = str(provider.get("model", "gpt-4o-mini"))
    embedding_model = str(provider.get("embedding_model", "text-embedding-3-small"))
    llm_env_var = _provider_env_var(model)
    embedding_env_var = _provider_env_var(embedding_model)

    missing_env_vars: list[str] = []
    for env_var in (llm_env_var, embedding_env_var):
        if env_var and not os.getenv(env_var) and env_var not in missing_env_vars:
            missing_env_vars.append(env_var)

    local_runtime_required = _uses_ollama(model) or _uses_ollama(embedding_model)
    ollama_installed = shutil.which("ollama") is not None
    ollama_running = _ollama_running() if ollama_installed else False
    available_models = _list_ollama_models() if local_runtime_required else set()

    required_local_models: list[str] = []
    for candidate in (model, embedding_model):
        if _uses_ollama(candidate):
            local_name = _ollama_model_name(candidate)
            if local_name not in required_local_models:
                required_local_models.append(local_name)

    missing_local_models = tuple(
        model_name
        for model_name in required_local_models
        if model_name not in available_models
    )

    return RuntimeDiagnosis(
        model=model,
        embedding_model=embedding_model,
        llm_env_var=llm_env_var,
        embedding_env_var=embedding_env_var,
        missing_env_vars=tuple(missing_env_vars),
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        local_runtime_required=local_runtime_required,
        missing_local_models=missing_local_models,
    )


def format_diagnosis(diag: RuntimeDiagnosis) -> str:
    lines = [
        "Mnemon setup check",
        f"  Chat model:      {diag.model}",
        f"  Embedding model: {diag.embedding_model}",
    ]

    if diag.llm_env_var:
        chat_state = "set" if os.getenv(diag.llm_env_var) else "missing"
        lines.append(f"  Chat auth:       {diag.llm_env_var}={chat_state}")
    else:
        lines.append("  Chat auth:       not required")

    if diag.embedding_env_var:
        embedding_state = "set" if os.getenv(diag.embedding_env_var) else "missing"
        lines.append(f"  Embedding auth:  {diag.embedding_env_var}={embedding_state}")
    else:
        lines.append("  Embedding auth:  not required")

    if diag.ollama_installed:
        ollama_state = "running" if diag.ollama_running else "installed, not running"
        lines.append(f"  Ollama:          {ollama_state}")
    else:
        lines.append("  Ollama:          not installed")

    if diag.local_runtime_required:
        if diag.missing_local_models:
            lines.append(
                "  Local models:    missing " + ", ".join(diag.missing_local_models)
            )
        else:
            lines.append("  Local models:    ready")

    return "\n".join(lines)


def format_setup_help(diag: RuntimeDiagnosis) -> str:
    lines = [format_diagnosis(diag)]
    if diag.ok:
        lines.append("\nSetup looks good.")
        return "\n".join(lines)

    lines.append("\nNo usable model setup was found for the current defaults.")
    lines.append("\nOption 1 — OpenAI")
    lines.append("  export OPENAI_API_KEY=your_key_here")
    lines.append("  mnemon")
    lines.append("\nOption 2 — Local Ollama")
    lines.append("  ollama serve")
    local_models = (
        _ollama_model_name(_LOCAL_CHAT_MODEL),
        _ollama_model_name(_LOCAL_EMBED_MODEL),
    )
    for model_name in local_models:
        lines.append(f"  ollama pull {model_name}")
    lines.append("  mnemon --local")
    lines.append("\nRun `mnemon --doctor` to re-check your setup.")
    return "\n".join(lines)


class InteractiveAgentProtocol(Protocol):
    async def show_memories(self) -> str: ...
    async def show_facts(self) -> str: ...
    async def show_skills(self) -> str: ...
    def show_state(self) -> str: ...
    async def run_consolidation(self) -> str: ...
    def show_goals(self) -> str: ...


@dataclass(frozen=True)
class LaunchPlan:
    mode: str
    prompt: str | None = None
    daemon_args: list[str] | None = None


_DAEMON_ALIASES = {
    "stop": ["stop"],
    "status": ["status"],
    "chat": ["chat"],
    "webui": ["webui"],
    "goals": ["goals"],
    "pending": ["pending"],
    "approve": ["approve"],
    "improve": ["improve"],
}


class MnemonInteractiveAgent:
    """Interactive agent wrapper around a live Mnemon brain."""

    def __init__(self, brain: Any, llm: LiteLLMProvider) -> None:
        self.brain = brain
        self.llm = llm
        self.history: list[dict[str, str]] = []
        self.turn_count = 0
        self._last_cycle: dict[str, Any] | None = None

    async def chat(self, user_input: str) -> str:
        cycle_result = await self.brain.run_cycle(user_input)
        self._last_cycle = cycle_result

        response = await self._generate_response(cycle_result, user_input)
        self.history.append({"user": user_input, "assistant": response})
        self.history = self.history[-20:]
        self.turn_count += 1
        return response

    async def _generate_response(self, cycle_result: dict[str, Any], user_input: str) -> str:
        deliberation = cycle_result.get("deliberation", {})
        memory_context = deliberation.get("context", "")
        goal_text = deliberation.get("goal", "No specific goal")
        meta = cycle_result.get("meta_evaluation")

        parts: list[str] = [f"System: {SYSTEM_PROMPT}"]

        if memory_context.strip():
            parts.append(f"\n--- Retrieved Memories ---\n{memory_context}\n--- End Memories ---")

        if goal_text and goal_text != "No specific goal":
            parts.append(f"\nCurrent goal: {goal_text}")

        if meta and meta.get("strategy_recommended"):
            parts.append(f"\n[Meta-cognitive note: consider {meta['strategy_recommended']}]")

        parts.append("")
        for turn in self.history[-10:]:
            parts.append(f"User: {turn['user']}")
            parts.append(f"Assistant: {turn['assistant']}")
            parts.append("")

        parts.append(f"User: {user_input}")
        parts.append("Assistant:")
        return (await self.llm.generate("\n".join(parts))).strip()

    async def show_memories(self) -> str:
        result = await self.brain.memory.episodic.retrieve(
            RetrievalQuery(query_text="recent conversation", top_k=10, min_score=0.0)
        )
        if not result.items:
            return "No episodic memories stored yet."

        lines = [f"Episodic memories ({len(result.items)} found):"]
        for index, item in enumerate(result.items, start=1):
            content = item.content[:120].replace("\n", " ")
            lines.append(
                f"  {index}. [{item.source_store}] (score={item.score:.3f}) {content}"
            )
        return "\n".join(lines)

    async def show_facts(self) -> str:
        embedder = self.brain.memory.semantic._embedding_provider
        query_embedding = await embedder.embed("knowledge facts information")
        triples = await self.brain.memory.semantic.retrieve_by_similarity(
            query_embedding,
            top_k=20,
        )
        if not triples:
            return "No semantic facts stored yet. Run /consolidate to extract facts from memories."

        lines = [f"Semantic facts ({len(triples)} found):"]
        for index, triple in enumerate(triples, start=1):
            obj_name = (
                triple.object.name if hasattr(triple.object, "name") else str(triple.object)
            )
            lines.append(
                f"  {index}. {triple.subject.name} --[{triple.predicate}]--> {obj_name} "
                f"(conf={triple.confidence:.2f})"
            )
        return "\n".join(lines)

    async def show_skills(self) -> str:
        embedder = self.brain.memory.procedural._embedding_provider
        query_embedding = await embedder.embed("skills abilities procedures")
        skills = await self.brain.memory.procedural.retrieve(query_embedding, top_k=10)
        if not skills:
            return "No procedural skills learned yet."

        lines = [f"Procedural skills ({len(skills)} found):"]
        for index, skill in enumerate(skills, start=1):
            lines.append(
                f"  {index}. {skill.name} — {skill.description[:80]} "
                f"(utility={skill.utility:.2f}, status={skill.status})"
            )
        return "\n".join(lines)

    def show_state(self) -> str:
        state = self.brain.get_state()
        wm = state["working_memory"]
        goals = state["active_goals"]
        bus = state["bus"]

        lines = [
            "Cognitive State:",
            f"  Cycle count:     {state['cycle_count']}",
            f"  Conversation:    {self.turn_count} turns",
            (
                "  Working memory:  "
                f"{wm['token_used']}/{wm['token_budget']} tokens "
                f"({wm['token_available']} available)"
            ),
            f"  Active goals:    {len(goals)}",
            f"  Bus running:     {bus['running']}",
            f"  Subscriptions:   {bus['subscriptions']}",
        ]

        if goals:
            lines.append("  Goals:")
            for goal in goals:
                lines.append(
                    f"    - {goal['description'][:60]} "
                    f"(priority={goal['priority']:.1f}, progress={goal['progress']:.0%})"
                )

        if self._last_cycle:
            meta = self._last_cycle.get("meta_evaluation")
            if meta:
                lines.append(
                    "  Last cycle meta: "
                    f"confidence={meta['confidence']:.2f}, "
                    f"prediction_error={meta['prediction_error']:.3f}"
                )

        return "\n".join(lines)

    async def run_consolidation(self) -> str:
        try:
            result = await self.brain.learning.consolidation.run_cycle()
        except Exception as exc:
            return f"Consolidation failed: {exc}"

        return (
            "Consolidation complete:\n"
            f"  Episodes processed: {result.episodes_processed}\n"
            f"  Triples extracted:  {result.triples_extracted}\n"
            f"  Entities resolved:  {result.entities_resolved}\n"
            f"  Duration:           {result.duration_ms:.0f}ms"
        )

    def show_goals(self) -> str:
        goals = self.brain.control.goals.get_active_goals()
        if not goals:
            return "No active goals."

        lines = [f"Active goals ({len(goals)}):"]
        for goal in goals:
            lines.append(
                f"  - [{goal.status.value}] {goal.description} "
                f"(priority={goal.priority:.1f}, attempts={goal.attempts}/{goal.max_attempts})"
            )
        return "\n".join(lines)


async def dispatch_command(
    command: str,
    agent: InteractiveAgentProtocol,
) -> tuple[bool, str]:
    normalized = command.lower().split()[0]

    if normalized == "/quit":
        return True, "Goodbye!"
    if normalized == "/help":
        return False, HELP_TEXT
    if normalized == "/memories":
        return False, await agent.show_memories()
    if normalized == "/facts":
        return False, await agent.show_facts()
    if normalized == "/skills":
        return False, await agent.show_skills()
    if normalized == "/state":
        return False, agent.show_state()
    if normalized == "/consolidate":
        return False, await agent.run_consolidation()
    if normalized == "/goals":
        return False, agent.show_goals()

    return False, f"Unknown command: {normalized}. Type /help for available commands."


def resolve_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    if args.entry == "start":
        return LaunchPlan(mode="start")
    if args.entry == "daemon":
        return LaunchPlan(mode="daemon", daemon_args=list(args.prompt_words))
    if args.entry in _DAEMON_ALIASES:
        return LaunchPlan(
            mode="daemon",
            daemon_args=[*_DAEMON_ALIASES[args.entry], *args.prompt_words],
        )

    prompt_parts = [part for part in [args.entry, *args.prompt_words] if part]
    if prompt_parts:
        return LaunchPlan(mode="prompt", prompt=" ".join(prompt_parts))

    return LaunchPlan(mode="interactive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemon",
        description="Simple interactive CLI for the Mnemon memory system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Set provider env vars (for example OPENAI_API_KEY), "
            "or use --doctor for setup help."
        ),
    )
    parser.add_argument(
        "entry",
        nargs="?",
        default=None,
        help="Use `start` to launch the daemon, or provide a one-shot prompt.",
    )
    parser.add_argument(
        "prompt_words",
        nargs="*",
        help="Additional words for the one-shot prompt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to mnemon.toml",
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Ollama defaults (llama3.2 + nomic-embed-text)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check model/provider setup and print guidance",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="When used with `mnemon start`, run the daemon in the foreground.",
    )
    return parser


def apply_local_defaults(config: MnemonConfig) -> MnemonConfig:
    provider_name = config.llm.default_provider
    config.llm.providers[provider_name] = {
        **config.llm.providers.get(provider_name, {}),
        "model": _LOCAL_CHAT_MODEL,
        "embedding_model": _LOCAL_EMBED_MODEL,
        "embedding_dimensions": _LOCAL_EMBED_DIM,
    }
    return config


def configure_runtime(args: argparse.Namespace) -> MnemonConfig:
    config = load_config(args.config) if args.config else MnemonConfig()
    if getattr(args, "local", False):
        return apply_local_defaults(config)

    provider_name = config.llm.default_provider
    config.llm.providers[provider_name] = {
        **config.llm.providers.get(provider_name, {}),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": args.embedding_dim,
    }
    return config


def ensure_runtime_ready(
    args: argparse.Namespace,
    config: MnemonConfig,
) -> tuple[MnemonConfig, RuntimeDiagnosis, list[str]]:
    notes: list[str] = []
    diagnosis = diagnose_runtime(config)

    wants_local = getattr(args, "local", False)
    should_try_local = wants_local or (
        not diagnosis.ok and not diagnosis.missing_env_vars and diagnosis.local_runtime_required
    ) or (
        not diagnosis.ok and diagnosis.ollama_installed
    )

    if should_try_local:
        if not wants_local and not diagnosis.ok:
            notes.append("No cloud API key detected; trying local Ollama mode.")
        config = apply_local_defaults(config)
        diagnosis = diagnose_runtime(config)
        if diagnosis.ollama_installed and not diagnosis.ollama_running:
            if _start_ollama_server():
                notes.append("Started local Ollama server automatically.")
            diagnosis = diagnose_runtime(config)
        if diagnosis.ollama_running and diagnosis.missing_local_models:
            for model_name in diagnosis.missing_local_models:
                print(f"Pulling missing Ollama model: {model_name}")
                if _pull_ollama_model(model_name):
                    notes.append(f"Pulled Ollama model: {model_name}")
                else:
                    notes.append(f"Failed to pull Ollama model: {model_name}")
            diagnosis = diagnose_runtime(config)

    return config, diagnosis, notes


def print_startup_banner(diag: RuntimeDiagnosis, notes: list[str]) -> None:
    for note in notes:
        print(_style(f"• {note}", fg="green", bold=True))
    if notes:
        print()

    _print_section("Mnemon interactive CLI ready")
    print(_kv("Chat model", diag.model))
    print(_kv("Embed model", diag.embedding_model))
    print(_kv("Web UI", "mnemon-daemon webui", tone="yellow"))
    print(_kv("Web UI URL", "http://localhost:7777", tone="green"))
    print(_kv("Web UI (LAN)", f"http://{_local_ip()}:7777", tone="green"))
    print(_kv("Daemon", "mnemon-daemon start --foreground", tone="yellow"))
    print(_kv("Doctor", "mnemon --doctor", tone="yellow"))
    print(_kv("Commands", "/help  /quit", tone="yellow"))
    print()
    _print_dashboard_footer()


def run_start_command(args: argparse.Namespace) -> None:
    from mnemon.daemon.config import DaemonConfig
    from mnemon.daemon.lifecycle import DaemonProcess

    config = configure_runtime(args)
    config, diagnosis, notes = ensure_runtime_ready(args, config)
    if not diagnosis.ok:
        print(format_setup_help(diagnosis))
        return

    daemon_config = DaemonConfig()
    process = DaemonProcess(daemon_config)

    local_url = f"http://localhost:{daemon_config.webui_port}"
    network_url = f"http://{_local_ip()}:{daemon_config.webui_port}"

    _print_section("Starting Mnemon daemon")
    for note in notes:
        print(_style(f"• {note}", fg="green", bold=True))
    if notes:
        print()
    print(_kv("Socket", str(daemon_config.socket_path)))
    print(_kv("Log file", str(daemon_config.log_path)))
    print(_kv("Web UI", local_url, tone="green"))
    print(_kv("Web UI (LAN)", network_url, tone="green"))
    print(_kv("Autonomy", str(daemon_config.autonomy_level), tone="yellow"))
    if diagnosis.model.startswith("ollama/"):
        print(_kv("Runtime", "Local Ollama mode", tone="green"))
    print(_kv("Chat model", diagnosis.model, tone="yellow"))
    print(_kv("Embed model", diagnosis.embedding_model, tone="yellow"))
    print()

    existing = process.status()
    if existing.get("running"):
        print(_style("Daemon already running.", fg="yellow", bold=True))
        print(_kv("PID", str(existing.get("pid", "?")), tone="yellow"))
        print(_kv("Open Web UI", local_url, tone="green"))
        print(_kv("On your phone", network_url, tone="green"))
        print(_kv("Status", "mnemon-daemon status", tone="yellow"))
        print(_kv("Chat", "mnemon-daemon chat", tone="yellow"))
        print(_kv("Stop", "mnemon daemon stop", tone="yellow"))
        print()
        return

    if args.foreground:
        process.start(foreground=True)
        return

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    devnull = os.fdopen(devnull_fd, "w")
    argv = [sys.executable, "-m", "mnemon.daemon.cli.app", "start", "--foreground"]
    if args.config:
        argv.extend(["--config", args.config])
    if getattr(args, "local", False) or diagnosis.model.startswith("ollama/"):
        argv.append("--local")
    if getattr(args, "model", None):
        argv.extend(["--model", args.model])
    if getattr(args, "embedding_model", None):
        argv.extend(["--embedding-model", args.embedding_model])
    if getattr(args, "embedding_dim", None) is not None:
        argv.extend(["--embedding-dim", str(args.embedding_dim)])

    subprocess.Popen(  # noqa: S603
        argv,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=devnull,
        stderr=devnull,
    )

    end = time.time() + 8.0
    while time.time() < end:
        current = process.status()
        if current.get("running"):
            print(_style("Daemon started.", fg="green", bold=True))
            print(_kv("PID", str(current.get("pid", "?")), tone="yellow"))
            print(_kv("Open Web UI", local_url, tone="green"))
            print(_kv("On your phone", network_url, tone="green"))
            print(_kv("Status", "mnemon-daemon status", tone="yellow"))
            print(_kv("Chat", "mnemon-daemon chat", tone="yellow"))
            print(_kv("Stop", "mnemon-daemon stop", tone="yellow"))
            print()
            return
        time.sleep(0.2)

    print(_style("Daemon launch did not become ready in time.", fg="red", bold=True))
    print(_kv("Log file", str(daemon_config.log_path), tone="yellow"))
    print(_kv("Try", "mnemon-daemon status", tone="yellow"))
    print()


def run_daemon_passthrough(daemon_args: list[str]) -> None:
    from mnemon.daemon.cli.app import main as daemon_main

    daemon_main(daemon_args)


async def run_cli(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    plan = resolve_launch_plan(args)
    if plan.mode == "start":
        run_start_command(args)
        return
    if plan.mode == "daemon":
        run_daemon_passthrough(plan.daemon_args or [])
        return

    config = configure_runtime(args)
    diagnosis = diagnose_runtime(config)
    if args.doctor:
        print(format_setup_help(diagnosis))
        return

    config, diagnosis, notes = ensure_runtime_ready(args, config)
    if not diagnosis.ok:
        print(format_setup_help(diagnosis))
        return

    print(f"Booting Mnemon memory system (model={diagnosis.model})...")
    brain = await MnemonFactory(config).build()
    response_llm = LiteLLMProvider(
        model=diagnosis.model,
        temperature=args.temperature,
        max_tokens=2048,
    )
    agent = MnemonInteractiveAgent(brain, response_llm)

    async with brain:
        if plan.prompt:
            response = await agent.chat(plan.prompt)
            print(response)
            return

        print_startup_banner(diagnosis, notes)

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                should_exit, output = await dispatch_command(user_input, agent)
                print(output)
                print()
                if should_exit:
                    break
                _print_dashboard_footer()
                continue

            try:
                print(f"\nAssistant: {await agent.chat(user_input)}\n")
            except Exception as exc:
                if args.debug:
                    logger.exception("Error during interactive chat")
                else:
                    logger.warning("Error during interactive chat: %s", exc)
                print(f"\nError: {exc}\n")
            _print_dashboard_footer()

    await brain.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv and raw_argv[0] == "daemon":
        run_daemon_passthrough(raw_argv[1:])
        return
    asyncio.run(run_cli(parse_args(raw_argv)))
