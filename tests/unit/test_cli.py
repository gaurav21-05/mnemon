from __future__ import annotations

from argparse import Namespace

import pytest

from mnemon.cli import (
    HELP_TEXT,
    START_HELP_TEXT,
    LaunchPlan,
    _ollama_model_aliases,
    apply_local_defaults,
    configure_runtime,
    diagnose_runtime,
    dispatch_command,
    format_setup_help,
    parse_args,
    resolve_launch_plan,
    run_start_command,
)


class _FakeAgent:
    async def show_memories(self) -> str:
        return "memories"

    async def show_facts(self) -> str:
        return "facts"

    async def show_skills(self) -> str:
        return "skills"

    def show_state(self) -> str:
        return "state"

    async def run_consolidation(self) -> str:
        return "consolidated"

    def show_goals(self) -> str:
        return "goals"


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.entry is None
    assert args.prompt_words == []
    assert args.model == "gpt-4o-mini"
    assert args.embedding_model == "text-embedding-3-small"
    assert args.embedding_dim == 1536
    assert args.temperature == 0.7
    assert args.debug is False
    assert args.local is False
    assert args.doctor is False
    assert args.foreground is False


def test_parse_args_local_flag() -> None:
    assert parse_args(["--local"]).local is True


def test_parse_args_doctor_flag() -> None:
    assert parse_args(["--doctor"]).doctor is True


def test_resolve_launch_plan_for_start() -> None:
    args = parse_args(["start"])
    assert resolve_launch_plan(args) == LaunchPlan(mode="start")


def test_resolve_launch_plan_for_prompt_words() -> None:
    args = parse_args(["remember", "my", "name"])
    assert resolve_launch_plan(args) == LaunchPlan(
        mode="prompt",
        prompt="remember my name",
    )


def test_resolve_launch_plan_for_daemon_passthrough() -> None:
    args = parse_args(["daemon", "status"])
    assert resolve_launch_plan(args) == LaunchPlan(
        mode="daemon",
        daemon_args=["status"],
    )


def test_resolve_launch_plan_for_stop_alias() -> None:
    args = parse_args(["stop"])
    assert resolve_launch_plan(args) == LaunchPlan(
        mode="daemon",
        daemon_args=["stop"],
    )


def test_resolve_launch_plan_for_chat_alias_with_message() -> None:
    args = parse_args(["chat", "hello"])
    assert resolve_launch_plan(args) == LaunchPlan(
        mode="daemon",
        daemon_args=["chat", "hello"],
    )


def test_parse_args_keeps_daemon_passthrough_tokens() -> None:
    args = parse_args(["daemon", "stop"])
    assert args.entry == "daemon"
    assert args.prompt_words == ["stop"]


def test_ollama_model_aliases_include_latest_and_base() -> None:
    assert _ollama_model_aliases("llama3.2") == {"llama3.2", "llama3.2:latest"}
    assert _ollama_model_aliases("llama3.2:latest") == {"llama3.2", "llama3.2:latest"}


def test_apply_local_defaults_sets_ollama_models() -> None:
    config = apply_local_defaults(configure_runtime(parse_args([])))
    provider_name = config.llm.default_provider
    provider = config.llm.providers[provider_name]
    assert provider["model"] == "ollama/llama3.2"
    assert provider["embedding_model"] == "ollama/nomic-embed-text"
    assert provider["embedding_dimensions"] == 768


def test_configure_runtime_applies_model_overrides() -> None:
    args = Namespace(
        config=None,
        model="ollama/llama3.2",
        embedding_model="ollama/nomic-embed-text",
        embedding_dim=768,
        local=False,
    )
    config = configure_runtime(args)
    provider_name = config.llm.default_provider
    provider = config.llm.providers[provider_name]
    assert provider["model"] == "ollama/llama3.2"
    assert provider["embedding_model"] == "ollama/nomic-embed-text"
    assert provider["embedding_dimensions"] == 768


def test_diagnose_runtime_reports_missing_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("mnemon.cli.shutil.which", lambda _name: None)
    monkeypatch.setattr("mnemon.cli._ollama_running", lambda: False)

    diagnosis = diagnose_runtime(configure_runtime(parse_args([])))

    assert diagnosis.ok is False
    assert diagnosis.missing_env_vars == ("OPENAI_API_KEY",)
    assert diagnosis.local_runtime_required is False
    help_text = format_setup_help(diagnosis)
    assert "OpenAI" in help_text
    assert "mnemon --local" in help_text


def test_diagnose_runtime_accepts_ready_ollama_local_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "mnemon.cli.shutil.which",
        lambda _name: "/usr/local/bin/ollama",
    )
    monkeypatch.setattr("mnemon.cli._ollama_running", lambda: True)
    monkeypatch.setattr(
        "mnemon.cli._list_ollama_models",
        lambda: {"llama3.2", "nomic-embed-text"},
    )

    diagnosis = diagnose_runtime(configure_runtime(parse_args(["--local"])))

    assert diagnosis.ok is True
    assert diagnosis.missing_env_vars == ()
    assert diagnosis.local_runtime_required is True
    assert diagnosis.missing_local_models == ()


def test_diagnose_runtime_reports_missing_ollama_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mnemon.cli.shutil.which",
        lambda _name: "/usr/local/bin/ollama",
    )
    monkeypatch.setattr("mnemon.cli._ollama_running", lambda: True)
    monkeypatch.setattr("mnemon.cli._list_ollama_models", lambda: set())

    diagnosis = diagnose_runtime(configure_runtime(parse_args(["--local"])))

    assert diagnosis.ok is False
    assert diagnosis.local_runtime_required is True
    assert diagnosis.missing_local_models == ("llama3.2", "nomic-embed-text")


@pytest.mark.asyncio
async def test_dispatch_command_routes_known_commands() -> None:
    agent = _FakeAgent()
    assert await dispatch_command("/help", agent) == (False, HELP_TEXT)
    assert await dispatch_command("/memories", agent) == (False, "memories")
    assert await dispatch_command("/facts", agent) == (False, "facts")
    assert await dispatch_command("/skills", agent) == (False, "skills")
    assert await dispatch_command("/state", agent) == (False, "state")
    assert await dispatch_command("/consolidate", agent) == (False, "consolidated")
    assert await dispatch_command("/goals", agent) == (False, "goals")
    assert await dispatch_command("/quit", agent) == (True, "Goodbye!")


@pytest.mark.asyncio
async def test_dispatch_command_handles_unknown_command() -> None:
    should_exit, output = await dispatch_command("/wat", _FakeAgent())
    assert should_exit is False
    assert "Unknown command" in output


def test_start_help_text_mentions_webui() -> None:
    assert "Web UI" in START_HELP_TEXT
    assert "mnemon-daemon status" in START_HELP_TEXT
    assert "mnemon start --local" in START_HELP_TEXT


def test_run_start_command_passes_local_flag_to_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: dict[str, object] = {}

    class _FakeProcess:
        def __init__(self) -> None:
            self._calls = 0

        def status(self) -> dict[str, object]:
            self._calls += 1
            if self._calls == 1:
                return {"running": False}
            return {"running": True, "pid": 123}

    monkeypatch.setattr("mnemon.cli.configure_runtime", lambda args: object())
    monkeypatch.setattr(
        "mnemon.cli.ensure_runtime_ready",
        lambda args, config: (
            config,
            Namespace(ok=True, model="ollama/llama3.2", embedding_model="ollama/nomic-embed-text"),
            ["Started local Ollama server automatically."],
        ),
    )
    monkeypatch.setattr("mnemon.cli._print_section", lambda _title: None)
    monkeypatch.setattr("mnemon.cli._kv", lambda _label, _value, tone="cyan": "")
    monkeypatch.setattr("mnemon.cli._style", lambda text, **_kwargs: text)
    monkeypatch.setattr("mnemon.cli._local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr("mnemon.cli.time.sleep", lambda _s: None)
    monkeypatch.setattr("mnemon.cli.os.open", lambda *_args, **_kwargs: 1)

    class _FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("mnemon.cli.os.fdopen", lambda *_args, **_kwargs: _FakeFile())
    monkeypatch.setattr(
        "mnemon.daemon.lifecycle.DaemonProcess",
        lambda *_args, **_kwargs: _FakeProcess(),
    )

    def fake_popen(argv, **kwargs):
        launched["argv"] = argv
        launched["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("mnemon.cli.subprocess.Popen", fake_popen)

    args = Namespace(
        config=None,
        foreground=False,
        local=False,
        model="ollama/llama3.2",
        embedding_model="ollama/nomic-embed-text",
        embedding_dim=768,
    )
    run_start_command(args)

    argv = launched["argv"]
    assert "--local" in argv
    assert "--model" in argv
    assert "--embedding-model" in argv
    assert "--embedding-dim" in argv
