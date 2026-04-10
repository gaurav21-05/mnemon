"""
Mnemon daemon CLI — command-line interface for controlling the Jarvis daemon.

Usage:
    mnemon-daemon start [--foreground]    Start the daemon
    mnemon-daemon stop                    Stop a running daemon
    mnemon-daemon status                  Check daemon status
    mnemon-daemon chat "message"          Send a message (one-shot)
    mnemon-daemon chat                    Interactive REPL
    mnemon-daemon improve [goal]          Run supervised self-improvement
    mnemon-daemon thoughts [--limit N]    View recent idle thinking
    mnemon-daemon goals add "desc"        Add a new goal
    mnemon-daemon goals list              List active goals
    mnemon-daemon pending                 List pending approvals
    mnemon-daemon approve <action-id>     Approve a pending action
"""

from __future__ import annotations

import argparse
import sys

import anyio

from mnemon.daemon.config import DaemonConfig


def _get_client():
    from mnemon.daemon.cli.client import DaemonClient
    config = DaemonConfig()
    return DaemonClient(config.socket_path)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> None:
    """Start the daemon process."""
    from mnemon.core.config import MnemonConfig, load_config
    from mnemon.daemon.lifecycle import DaemonProcess

    daemon_config = DaemonConfig()
    mnemon_config = load_config(args.config) if args.config else MnemonConfig()
    provider_name = mnemon_config.llm.default_provider
    provider = {**mnemon_config.llm.providers.get(provider_name, {})}
    if getattr(args, "local", False):
        provider.update(
            {
                "model": "ollama/llama3.2",
                "embedding_model": "ollama/nomic-embed-text",
                "embedding_dimensions": 768,
            }
        )
    if getattr(args, "model", None):
        provider["model"] = args.model
    if getattr(args, "embedding_model", None):
        provider["embedding_model"] = args.embedding_model
    if getattr(args, "embedding_dim", None) is not None:
        provider["embedding_dimensions"] = args.embedding_dim
    mnemon_config.llm.providers[provider_name] = provider

    process = DaemonProcess(daemon_config, mnemon_config)

    print("Starting Mnemon daemon...")
    print(f"  PID file: {daemon_config.pid_path}")
    print(f"  Log file: {daemon_config.log_path}")
    print(f"  Socket:   {daemon_config.socket_path}")
    print(f"  Autonomy: {daemon_config.autonomy_level}")
    print()

    process.start(foreground=args.foreground)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the running daemon."""
    from mnemon.daemon.lifecycle import DaemonProcess

    process = DaemonProcess(DaemonConfig())
    process.stop()


def cmd_status(args: argparse.Namespace) -> None:
    """Check daemon status."""
    from mnemon.daemon.lifecycle import DaemonProcess

    process = DaemonProcess(DaemonConfig())
    status = process.status()

    if status["running"]:
        print(f"Daemon is RUNNING (pid={status.get('pid', '?')})")
        # Also fetch runtime status via IPC
        try:
            client = _get_client()
            result = anyio.run(client.status)
            daemon_info = result.get("daemon", {})
            print(f"  Started:     {daemon_info.get('started_at', '?')}")
            print(f"  Cycles:      {daemon_info.get('total_cycles', '?')}")
            print(f"  Idle ticks:  {daemon_info.get('total_idle_ticks', '?')}")
            print(f"  Autonomy:    {daemon_info.get('autonomy_level', '?')}")
            print(f"  Last input:  {daemon_info.get('last_user_interaction', 'never')}")
        except Exception:
            print("  (could not connect to daemon for runtime status)")
    else:
        print(f"Daemon is NOT RUNNING ({status.get('reason', 'unknown')})")


def cmd_chat(args: argparse.Namespace) -> None:
    """Send a message or start interactive REPL."""
    client = _get_client()

    if args.message:
        # One-shot mode
        result = anyio.run(client.chat, args.message)
        _print_chat_result(result)
    else:
        # Interactive REPL
        print("Mnemon Daemon REPL (type 'quit' to exit)")
        print("=" * 50)
        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
                print("Goodbye.")
                break

            # Handle special commands
            if user_input.startswith("/"):
                _handle_repl_command(client, user_input)
                continue

            try:
                result = anyio.run(client.chat, user_input)
                _print_chat_result(result)
            except Exception as exc:
                print(f"Error: {exc}")


def cmd_thoughts(args: argparse.Namespace) -> None:
    """View recent idle thinking."""
    client = _get_client()
    thoughts = anyio.run(client.thoughts, args.limit)

    if not thoughts:
        print("No recorded thoughts yet. The daemon needs time to think.")
        return

    for t in thoughts:
        print(f"  [{t['timestamp']}] {t['activity']}: {t['summary']}")


def cmd_memory_search(args: argparse.Namespace) -> None:
    """Search episodic memory with compact indexed results."""
    client = _get_client()
    result = anyio.run(client.memory_recall, args.query, args.top_k, args.scope, args.scope_id)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    profile = result.get("profile") or {}
    static = profile.get("static") or []
    dynamic = profile.get("dynamic") or []
    if static or dynamic:
        print("Profile context:")
        for item in (profile.get("static_facts") or [])[:2]:
            citations = ", ".join(item.get("citations") or [])
            suffix = f" ({citations})" if citations else ""
            print(f"  static  • {item.get('text', '')}{suffix}")
        for item in (profile.get("dynamic_facts") or [])[:3]:
            citations = ", ".join(item.get("citations") or [])
            suffix = f" ({citations})" if citations else ""
            print(f"  dynamic • {item.get('text', '')}{suffix}")
        print()

    _print_memory_index_items(result.get("results", []))


def cmd_memory_get(args: argparse.Namespace) -> None:
    """Fetch full memory records by id."""
    client = _get_client()
    result = anyio.run(client.memory_get, args.ids)

    for item in result.get("items", []):
        print(f"[{item.get('id')}] {item.get('timestamp')}")
        if item.get("tags"):
            print(f"  tags: {', '.join(item['tags'])}")
        if item.get("citation"):
            print(f"  citation: {item['citation']}")
        print(f"  context: {item.get('context', '')}")
        print(f"  action:  {item.get('action', '')}")
        print(f"  outcome: {item.get('outcome', '')}")
        print()

    missing = result.get("missing", [])
    if missing:
        print(f"Missing: {', '.join(missing)}")


def cmd_memory_profile(args: argparse.Namespace) -> None:
    """Show the daemon's structured memory profile for the user."""
    client = _get_client()
    result = anyio.run(client.memory_profile)

    for label, key in (
        ("Static", "static"),
        ("Dynamic", "dynamic"),
        ("Questions", "questions"),
    ):
        items = result.get(key, [])
        print(f"{label}:")
        if items:
            fact_key = {
                "static": "static_facts",
                "dynamic": "dynamic_facts",
                "questions": "question_facts",
            }[key]
            fact_items = result.get(fact_key, [])
            if fact_items:
                for item in fact_items:
                    citations = ", ".join(item.get("citations") or [])
                    suffix = f" ({citations})" if citations else ""
                    print(f"  - {item.get('text', '')}{suffix}")
            else:
                for item in items:
                    print(f"  - {item}")
        else:
            print("  (none yet)")
        print()

    top_tags = result.get("top_tags", [])
    if top_tags:
        print("Top tags:")
        for item in top_tags:
            print(f"  - {item['tag']} ({item['count']})")


def cmd_memory_timeline(args: argparse.Namespace) -> None:
    """Show the local memory timeline around one anchor memory."""
    client = _get_client()
    result = anyio.run(client.memory_timeline, args.anchor_id, args.limit)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    for item in result.get("items", []):
        marker = "*" if item.get("anchor") else "-"
        tags = f" [{', '.join(item['tags'])}]" if item.get("tags") else ""
        print(f"{marker} {item.get('timestamp')} {item.get('id')}{tags}")
        print(f"    {item.get('preview', '')}")


def cmd_scenario(args: argparse.Namespace) -> None:
    """Run a bounded what-if analysis grounded in memory and goals."""
    client = _get_client()
    result = anyio.run(client.run_scenario, args.scenario, args.scope, args.scope_id)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    print(f"Scenario: {result.get('scenario', '')}")
    print(f"Summary: {result.get('summary', '')}")
    print(f"Uncertainty: {result.get('uncertainty', '')}")
    print()
    for label, key in (
        ("Assumptions", "assumptions"),
        ("Risks", "risks"),
        ("Recommendations", "recommendations"),
    ):
        print(f"{label}:")
        items = result.get(key, [])
        if items:
            for item in items:
                print(f"  - {item}")
        else:
            print("  (none)")
        print()
    citations = result.get("citations", [])
    if citations:
        print("Sources:")
        for item in citations:
            print(f"  - {item}")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a grounded weekly or project report."""
    client = _get_client()
    result = anyio.run(client.run_report, args.type, args.focus, args.scope, args.scope_id)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    print(f"Report: {result.get('title', '')}")
    print(f"Type: {result.get('type', '')}")
    if result.get("focus"):
        print(f"Focus: {result.get('focus')}")
    print(f"Summary: {result.get('summary', '')}")
    print()
    for label, key in (
        ("Highlights", "highlights"),
        ("Risks", "risks"),
        ("Next steps", "next_steps"),
    ):
        print(f"{label}:")
        items = result.get(key, [])
        if items:
            for item in items:
                print(f"  - {item}")
        else:
            print("  (none)")
        print()
    citations = result.get("citations", [])
    if citations:
        print("Sources:")
        for item in citations:
            print(f"  - {item}")


def cmd_goals_add(args: argparse.Namespace) -> None:
    """Add a new goal."""
    client = _get_client()
    result = anyio.run(client.add_goal, args.description, args.priority)
    print(f"Goal created: {result.get('id', '?')}")
    print(f"  Description: {result.get('description', '?')}")
    print(f"  Priority:    {result.get('priority', '?')}")


def cmd_goals_list(args: argparse.Namespace) -> None:
    """List active goals."""
    client = _get_client()
    goals = anyio.run(client.list_goals)

    if not goals:
        print("No active goals.")
        return

    for g in goals:
        subgoals = g.get("subgoals", [])
        sub_text = f" ({len(subgoals)} subgoals)" if subgoals else ""
        print(f"  [{g['priority']:.1f}] {g['description']}{sub_text}")
        print(f"        id={g['id']} status={g['status']} progress={g['progress']:.0%}")


def cmd_pending(args: argparse.Namespace) -> None:
    """List pending approvals."""
    client = _get_client()
    pending = anyio.run(client.pending)

    if not pending:
        print("No pending approvals.")
        return

    for a in pending:
        print(f"  [{a['risk']}] {a['description']}")
        print(f"        id={a['id']} source={a['source']}")


def cmd_improve(args: argparse.Namespace) -> None:
    """Run a supervised self-improvement cycle."""
    client = _get_client()

    if args.analyze:
        result = anyio.run(client.improve_analyze)
        print("\n=== Analysis ===")
        print(result.get("summary", "(no summary)"))
        print()
        print("Git status:", result.get("git_status", "(clean)") or "(clean)")
        return

    if args.abort:
        result = anyio.run(client.improve_abort)
        print("Aborted." if result.get("ok") else f"Error: {result.get('error')}")
        return

    if args.approve:
        result = anyio.run(client.improve_approve)
        if result.get("ok"):
            print(f"Merged branch '{result.get('branch')}' into HEAD.")
            print(f"Summary: {result.get('summary')}")
        else:
            print(f"Error: {result.get('error')}")
        return

    goal = args.goal or "improve code quality and fix any failing tests"
    result = anyio.run(client.improve_start, goal)
    if result.get("started"):
        print(f"Starting self-improvement session: {goal}")
        print("Run 'mnemon-daemon improve --status' to poll progress.")
    else:
        print(f"Error: {result.get('error', 'could not start session')}")


def cmd_improve_status(args: argparse.Namespace) -> None:
    """Show the current self-improvement session status."""
    client = _get_client()
    result = anyio.run(client.improve_status)
    phase = result.get("phase", "idle")
    print(f"Phase: {phase}")
    if result.get("analysis_summary"):
        print(f"Analysis: {result['analysis_summary']}")
    if result.get("plan_summary"):
        print(f"Plan: {result['plan_summary']}")
    steps_total = result.get("patch_steps_total", 0)
    steps_done = result.get("steps_applied", 0)
    if steps_total:
        print(f"Patches: {steps_done}/{steps_total} applied")
    if result.get("verify_output"):
        print(f"Verify: {'PASS' if result.get('verify_passed') else 'FAIL'}")
        print(result["verify_output"][:800])
    if result.get("error"):
        print(f"Error: {result['error']}")
    if phase == "awaiting_approval":
        approval_id = result.get("approval_id")
        print(f"\nReady for approval (id={approval_id})")
        print("  mnemon-daemon improve --approve    # merge and clean up")
        print("  mnemon-daemon improve --abort      # discard changes")


def cmd_browse(args: argparse.Namespace) -> None:
    """Run a browsing task through the daemon."""
    client = _get_client()
    result = anyio.run(client.browse, args.task)
    print(result.get("result", ""))


def cmd_ls(args: argparse.Namespace) -> None:
    """List a workspace directory via the daemon."""
    client = _get_client()
    result = anyio.run(client.list_dir, args.path)
    for entry in result.get("entries", []):
        print(f"{entry['type']:>4}  {entry['path']}")


def cmd_read(args: argparse.Namespace) -> None:
    """Read a workspace file via the daemon."""
    client = _get_client()
    result = anyio.run(client.read_file, args.path)
    print(result.get("content", ""))
    if result.get("truncated"):
        print("\n...<truncated>...")


def cmd_write(args: argparse.Namespace) -> None:
    """Write a workspace file via the daemon."""
    client = _get_client()
    content = args.content
    if content is None:
        content = sys.stdin.read()
    result = anyio.run(client.write_file, args.path, content, args.append)
    print(
        f"Wrote {result.get('bytes_written', 0)} bytes to {result.get('path', args.path)}"
    )


def cmd_exec(args: argparse.Namespace) -> None:
    """Run a bounded command via the daemon."""
    client = _get_client()
    result = anyio.run(client.exec_command, args.command, args.cwd, args.timeout)
    print(f"exit_code={result.get('exit_code')}")
    print(f"cwd={result.get('cwd')}")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if stdout:
        print("\nstdout:")
        print(stdout)
    if stderr:
        print("\nstderr:")
        print(stderr)
    if result.get("timed_out"):
        print("\ntimed_out=true")


def cmd_patch(args: argparse.Namespace) -> None:
    """Apply a targeted patch in the daemon workspace."""
    client = _get_client()
    result = anyio.run(
        client.patch_file,
        args.path,
        args.search,
        args.replace,
        args.cwd,
        args.replace_all,
    )
    print(result.get("diff", ""))


def cmd_verify(args: argparse.Namespace) -> None:
    """Run verification commands through the daemon."""
    client = _get_client()
    result = anyio.run(client.verify, args.command, args.cwd, args.timeout)
    print(f"passed={result.get('passed')}")
    for item in result.get("results", []):
        print(f"\n$ {item.get('command')}")
        print(f"exit_code={item.get('exit_code')} timed_out={item.get('timed_out')}")
        if item.get("stdout"):
            print(item["stdout"])
        if item.get("stderr"):
            print(item["stderr"])


def cmd_diff(args: argparse.Namespace) -> None:
    """Show git diff through the daemon."""
    client = _get_client()
    result = anyio.run(client.git_diff, args.cwd)
    print(result.get("stdout", ""))
    if result.get("stderr"):
        print(result["stderr"])


def cmd_git_status(args: argparse.Namespace) -> None:
    """Show git status through the daemon."""
    client = _get_client()
    result = anyio.run(client.git_status, args.cwd)
    print(result.get("stdout", ""))
    if result.get("stderr"):
        print(result["stderr"])


def cmd_worktree_create(args: argparse.Namespace) -> None:
    """Create a managed git worktree through the daemon."""
    client = _get_client()
    result = anyio.run(client.create_worktree, args.branch, args.base_ref, args.path)
    print(f"path={result.get('path')}")
    if result.get("stdout"):
        print(result["stdout"])
    if result.get("stderr"):
        print(result["stderr"])


def cmd_worktree_remove(args: argparse.Namespace) -> None:
    """Remove a managed git worktree through the daemon."""
    client = _get_client()
    result = anyio.run(client.remove_worktree, args.path, args.force)
    print(f"path={result.get('path')}")
    if result.get("stdout"):
        print(result["stdout"])
    if result.get("stderr"):
        print(result["stderr"])


def cmd_approve(args: argparse.Namespace) -> None:
    """Approve a pending action."""
    client = _get_client()
    result = anyio.run(client.approve, args.action_id)
    if result.get("approved"):
        print("Action approved.")
        if result.get("reply"):
            print(result["reply"])
    else:
        print("Action not found.")


def cmd_webui(args: argparse.Namespace) -> None:
    """Start the web dashboard (standalone mode)."""
    from mnemon.daemon.webui import main as webui_main
    webui_main(host=args.host, port=args.port)


def cmd_learn(args: argparse.Namespace) -> None:
    """Run knowledge bootstrap to seed Mnemon with foundational knowledge."""
    from mnemon.core.config import MnemonConfig
    from mnemon.factory import MnemonFactory
    from mnemon.learning.bootstrap import (
        FOUNDATION_CONCEPTS,
        FOUNDATION_TOPICS,
        KnowledgeBootstrap,
    )

    print("Mnemon Knowledge Bootstrap")
    print("=" * 50)

    phases = []
    if args.wikipedia:
        phases.append(1)
    if args.conceptnet:
        phases.append(2)
    if not phases:
        phases = [1, 2]  # default: run both

    topics = FOUNDATION_TOPICS
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]

    print(f"  Phases:    {phases}")
    print(f"  Wikipedia: {len(topics)} topics")
    print(f"  ConceptNet: {len(FOUNDATION_CONCEPTS)} concepts")
    print()

    async def _run() -> None:
        config = MnemonConfig()
        brain = await MnemonFactory(config).build()
        bootstrap = KnowledgeBootstrap(brain, topics=topics)

        async def _progress(msg: str) -> None:
            print(f"  {msg}")

        async with brain:
            results = await bootstrap.run(phases=phases, on_progress=_progress)

        print()
        print("Bootstrap complete:")
        print(f"  Wikipedia articles encoded: {results['wikipedia_articles']}")
        print(f"  ConceptNet triples written:  {results['conceptnet_triples']}")
        print()
        print("Run 'mnemon-daemon start' — idle consolidation will process these")
        print("into semantic facts automatically over the next few hours.")

    anyio.run(_run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_chat_result(result: dict) -> None:
    """Pretty-print a chat cycle result."""
    meta = result.get("meta") or {}
    delib = result.get("deliberation") or {}

    print(f"\nmnemon> [cycle #{result.get('cycle', '?')}]")
    if delib.get("context"):
        # Show a truncated context summary
        ctx = delib["context"]
        if len(ctx) > 300:
            ctx = ctx[:300] + "..."
        print(f"  Context: {ctx}")
    print(f"  Retrieved: {result.get('retrieved', 0)} memories")
    if meta.get("confidence"):
        print(f"  Confidence: {meta['confidence']:.2f}")
    if meta.get("lessons"):
        print(f"  Lessons: {', '.join(meta['lessons'])}")
    if result.get("reply"):
        print(f"\n{result['reply']}")


def _print_memory_index_items(items: list[dict[str, object]]) -> None:
    """Pretty-print compact search results with IDs for follow-up lookup."""
    if not items:
        print("No memory matches.")
        return

    print("Matches:")
    for item in items:
        score = float(item.get("score") or 0.0)
        timestamp = str(item.get("timestamp") or "?")
        memory_id = str(item.get("id") or "?")
        preview = str(item.get("preview") or item.get("content") or "")
        tags = item.get("tags") or []
        tag_text = f" [{', '.join(str(tag) for tag in tags)}]" if tags else ""
        print(f"  - {memory_id}  {(score * 100):.0f}%  {timestamp}{tag_text}")
        print(f"    {preview}")
        citation = str(item.get("citation") or "")
        if citation:
            print(f"    cite: {citation}")


def _handle_repl_command(client, cmd: str) -> None:
    """Handle slash commands in the REPL."""
    parts = cmd.split()
    command = parts[0].lower()

    try:
        if command == "/status":
            result = anyio.run(client.status)
            daemon = result.get("daemon", {})
            print(f"  Cycles: {daemon.get('total_cycles')}, Idle: {daemon.get('total_idle_ticks')}")
        elif command == "/thoughts":
            limit = int(parts[1]) if len(parts) > 1 else 5
            thoughts = anyio.run(client.thoughts, limit)
            for t in thoughts:
                print(f"  [{t['activity']}] {t['summary']}")
        elif command == "/goals":
            goals = anyio.run(client.list_goals)
            for g in goals:
                print(f"  [{g['priority']:.1f}] {g['description']} ({g['status']})")
        elif command == "/profile":
            result = anyio.run(client.memory_profile)
            for item in result.get("static", [])[:2]:
                print(f"  static  • {item}")
            for item in result.get("dynamic", [])[:3]:
                print(f"  dynamic • {item}")
        elif command == "/memories":
            query = cmd[len("/memories"):].strip()
            if not query:
                recent = anyio.run(client.memory_recent, 5)
                _print_memory_index_items(recent.get("items", []))
            else:
                result = anyio.run(client.memory_search, query, 5)
                _print_memory_index_items(result.get("results", []))
        elif command == "/pending":
            pending = anyio.run(client.pending)
            for a in pending:
                print(f"  [{a['risk']}] {a['description']} (id={a['id']})")
        elif command == "/browse":
            task = cmd[len("/browse"):].strip()
            result = anyio.run(client.browse, task)
            print(result.get("result", ""))
        elif command == "/ls":
            path = parts[1] if len(parts) > 1 else "."
            result = anyio.run(client.list_dir, path)
            for entry in result.get("entries", []):
                print(f"  {entry['type']:>4}  {entry['path']}")
        elif command == "/read":
            if len(parts) < 2:
                print("  Usage: /read <path>")
            else:
                result = anyio.run(client.read_file, parts[1])
                print(result.get("content", ""))
                if result.get("truncated"):
                    print("\n...<truncated>...")
        elif command == "/write":
            raw = cmd[len("/write"):].strip()
            write_parts = raw.split(maxsplit=1)
            if len(write_parts) < 2:
                print("  Usage: /write <path> <content>")
            else:
                result = anyio.run(client.write_file, write_parts[0], write_parts[1], False)
                print(
                    "  Wrote "
                    f"{result.get('bytes_written', 0)} bytes to "
                    f"{result.get('path', write_parts[0])}"
                )
        elif command == "/exec":
            raw_command = cmd[len("/exec"):].strip()
            result = anyio.run(client.exec_command, raw_command, None, 30.0)
            print(f"  exit_code={result.get('exit_code')}")
            if result.get("stdout"):
                print(result["stdout"])
            if result.get("stderr"):
                print(result["stderr"])
        elif command == "/verify":
            raw_command = cmd[len("/verify"):].strip()
            result = anyio.run(client.verify, [raw_command], None, 120.0)
            print(f"  passed={result.get('passed')}")
        elif command == "/diff":
            result = anyio.run(client.git_diff, None)
            print(result.get("stdout", ""))
        elif command == "/status-git":
            result = anyio.run(client.git_status, None)
            print(result.get("stdout", ""))
        elif command == "/help":
            print("  /status    - Daemon status")
            print("  /thoughts  - Recent idle thinking")
            print("  /goals     - Active goals")
            print("  /profile   - Structured memory profile")
            print("  /memories [query] - Recent memories or search memories")
            print("  /pending   - Pending approvals")
            print("  /browse    - Browse the web")
            print("  /ls        - List workspace files")
            print("  /read      - Read a file")
            print("  /write     - Write a file")
            print("  /exec      - Run a command")
            print("  /verify    - Run a verification command")
            print("  /diff      - Show git diff")
            print("  /status-git - Show git status")
            print("  /help      - This help")
        else:
            print(f"  Unknown command: {command}. Type /help for commands.")
    except Exception as exc:
        print(f"  Error: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mnemon-daemon",
        description="Mnemon Daemon — always-on cognitive AI assistant",
    )
    subparsers = parser.add_subparsers(dest="command")

    # daemon start
    p_start = subparsers.add_parser("start", help="Start the daemon")
    p_start.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    p_start.add_argument("--config", "-c", type=str, help="Path to mnemon.toml config file")
    p_start.add_argument("--local", action="store_true", help="Use local Ollama defaults")
    p_start.add_argument("--model", default=None, help="Override chat model")
    p_start.add_argument("--embedding-model", default=None, help="Override embedding model")
    p_start.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override embedding dimensions",
    )
    p_start.set_defaults(func=cmd_start)

    # daemon stop
    p_stop = subparsers.add_parser("stop", help="Stop the daemon")
    p_stop.set_defaults(func=cmd_stop)

    # daemon status
    p_status = subparsers.add_parser("status", help="Check daemon status")
    p_status.set_defaults(func=cmd_status)

    # chat
    p_chat = subparsers.add_parser("chat", help="Chat with the daemon")
    p_chat.add_argument("message", nargs="?", default=None, help="Message (omit for REPL)")
    p_chat.set_defaults(func=cmd_chat)

    # browse
    p_browse = subparsers.add_parser("browse", help="Browse the web through the daemon")
    p_browse.add_argument("task", help="Browsing task")
    p_browse.set_defaults(func=cmd_browse)

    # ls
    p_ls = subparsers.add_parser("ls", help="List files in the daemon workspace")
    p_ls.add_argument("path", nargs="?", default=".", help="Directory path")
    p_ls.set_defaults(func=cmd_ls)

    # read
    p_read = subparsers.add_parser("read", help="Read a file from the daemon workspace")
    p_read.add_argument("path", help="File path")
    p_read.set_defaults(func=cmd_read)

    # write
    p_write = subparsers.add_parser("write", help="Write a file in the daemon workspace")
    p_write.add_argument("path", help="File path")
    p_write.add_argument(
        "content",
        nargs="?",
        default=None,
        help="Content to write (omit to read from stdin)",
    )
    p_write.add_argument("--append", action="store_true", help="Append instead of overwrite")
    p_write.set_defaults(func=cmd_write)

    # exec
    p_exec = subparsers.add_parser("exec", help="Run a command in the daemon workspace")
    p_exec.add_argument("command", help="Command string")
    p_exec.add_argument("--cwd", default=None, help="Working directory relative to workspace root")
    p_exec.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds")
    p_exec.set_defaults(func=cmd_exec)

    # patch
    p_patch = subparsers.add_parser("patch", help="Apply a targeted patch in the daemon workspace")
    p_patch.add_argument("path", help="File path")
    p_patch.add_argument("search", help="Search text")
    p_patch.add_argument("replace", help="Replacement text")
    p_patch.add_argument("--cwd", default=None, help="Working directory relative to workspace root")
    p_patch.add_argument("--replace-all", action="store_true", help="Replace all matches")
    p_patch.set_defaults(func=cmd_patch)

    # verify
    p_verify = subparsers.add_parser(
        "verify",
        help="Run verification commands in the daemon workspace",
    )
    p_verify.add_argument("command", nargs="+", help="One or more commands to run sequentially")
    p_verify.add_argument(
        "--cwd",
        default=None,
        help="Working directory relative to workspace root",
    )
    p_verify.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout per command in seconds",
    )
    p_verify.set_defaults(func=cmd_verify)

    # diff
    p_diff = subparsers.add_parser("diff", help="Show git diff in the daemon workspace")
    p_diff.add_argument("--cwd", default=None, help="Working directory relative to workspace root")
    p_diff.set_defaults(func=cmd_diff)

    # git-status
    p_git_status = subparsers.add_parser(
        "git-status",
        help="Show git status in the daemon workspace",
    )
    p_git_status.add_argument(
        "--cwd",
        default=None,
        help="Working directory relative to workspace root",
    )
    p_git_status.set_defaults(func=cmd_git_status)

    # worktree
    p_worktree = subparsers.add_parser("worktree", help="Managed git worktree operations")
    worktree_sub = p_worktree.add_subparsers(dest="worktree_command")

    p_worktree_create = worktree_sub.add_parser("create", help="Create a managed worktree")
    p_worktree_create.add_argument("branch", help="Branch name")
    p_worktree_create.add_argument("--base-ref", default="HEAD", help="Base ref (default: HEAD)")
    p_worktree_create.add_argument("--path", default=None, help="Optional managed worktree path")
    p_worktree_create.set_defaults(func=cmd_worktree_create)

    p_worktree_remove = worktree_sub.add_parser("remove", help="Remove a managed worktree")
    p_worktree_remove.add_argument("path", help="Managed worktree path")
    p_worktree_remove.add_argument("--force", action="store_true", help="Force removal")
    p_worktree_remove.set_defaults(func=cmd_worktree_remove)

    # thoughts
    p_thoughts = subparsers.add_parser("thoughts", help="View recent idle thinking")
    p_thoughts.add_argument("--limit", "-n", type=int, default=10)
    p_thoughts.set_defaults(func=cmd_thoughts)

    # memory
    p_memory = subparsers.add_parser("memory", help="Structured memory inspection")
    memory_sub = p_memory.add_subparsers(dest="memory_command")

    p_memory_search = memory_sub.add_parser("search", help="Search episodic memory")
    p_memory_search.add_argument("query", help="Natural-language memory query")
    p_memory_search.add_argument("--top-k", type=int, default=5, help="Result count")
    p_memory_search.add_argument(
        "--scope",
        choices=["all", "personal", "workspace"],
        default="all",
        help="Limit recall to a scope",
    )
    p_memory_search.add_argument("--scope-id", default=None, help="Optional explicit scope id")
    p_memory_search.set_defaults(func=cmd_memory_search)

    p_memory_recall = memory_sub.add_parser(
        "recall",
        help="Profile-aware recall (profile + relevant memories)",
    )
    p_memory_recall.add_argument("query", help="Natural-language memory query")
    p_memory_recall.add_argument("--top-k", type=int, default=5, help="Result count")
    p_memory_recall.add_argument(
        "--scope",
        choices=["all", "personal", "workspace"],
        default="all",
        help="Limit recall to a scope",
    )
    p_memory_recall.add_argument("--scope-id", default=None, help="Optional explicit scope id")
    p_memory_recall.set_defaults(func=cmd_memory_search)

    p_scenario = subparsers.add_parser("scenario", help="Run a bounded what-if analysis")
    p_scenario.add_argument("scenario", help="Scenario question")
    p_scenario.add_argument(
        "--scope",
        choices=["all", "personal", "workspace"],
        default="all",
        help="Limit grounding to a scope",
    )
    p_scenario.add_argument("--scope-id", default=None, help="Optional explicit scope id")
    p_scenario.set_defaults(func=cmd_scenario)

    p_report = subparsers.add_parser("report", help="Generate a weekly or project report")
    p_report.add_argument(
        "--type",
        choices=["weekly", "project"],
        default="weekly",
        help="Report type",
    )
    p_report.add_argument("--focus", default="", help="Optional focus area or project name")
    p_report.add_argument(
        "--scope",
        choices=["all", "personal", "workspace"],
        default="all",
        help="Limit grounding to a scope",
    )
    p_report.add_argument("--scope-id", default=None, help="Optional explicit scope id")
    p_report.set_defaults(func=cmd_report)

    p_memory_get = memory_sub.add_parser("get", help="Fetch full memory records by id")
    p_memory_get.add_argument("ids", nargs="+", help="Episode IDs")
    p_memory_get.set_defaults(func=cmd_memory_get)

    p_memory_profile = memory_sub.add_parser("profile", help="Show structured user profile")
    p_memory_profile.set_defaults(func=cmd_memory_profile)

    p_memory_timeline = memory_sub.add_parser(
        "timeline",
        help="Show nearby memories around one anchor event",
    )
    p_memory_timeline.add_argument("anchor_id", help="Anchor episode ID")
    p_memory_timeline.add_argument("--limit", type=int, default=6, help="Timeline size")
    p_memory_timeline.set_defaults(func=cmd_memory_timeline)

    # goals
    p_goals = subparsers.add_parser("goals", help="Goal management")
    goals_sub = p_goals.add_subparsers(dest="goals_command")

    p_goals_add = goals_sub.add_parser("add", help="Add a new goal")
    p_goals_add.add_argument("description", help="Goal description")
    p_goals_add.add_argument("--priority", "-p", type=float, default=0.5)
    p_goals_add.set_defaults(func=cmd_goals_add)

    p_goals_list = goals_sub.add_parser("list", help="List active goals")
    p_goals_list.set_defaults(func=cmd_goals_list)

    # pending
    p_pending = subparsers.add_parser("pending", help="List pending approvals")
    p_pending.set_defaults(func=cmd_pending)

    # approve
    p_approve = subparsers.add_parser("approve", help="Approve a pending action")
    p_approve.add_argument("action_id", help="Action UUID to approve")
    p_approve.set_defaults(func=cmd_approve)

    # improve
    p_improve = subparsers.add_parser(
        "improve",
        help="Run a supervised self-improvement cycle",
    )
    p_improve.add_argument("goal", nargs="?", help="Improvement goal description")
    p_improve.add_argument("--analyze", action="store_true", help="Analyze only, don't start")
    p_improve.add_argument("--status", action="store_true", help="Show current session status")
    p_improve.add_argument("--approve", action="store_true", help="Approve pending improvement")
    p_improve.add_argument("--abort", action="store_true", help="Abort current session")
    p_improve.set_defaults(func=lambda a: cmd_improve_status(a) if a.status else cmd_improve(a))

    # webui
    p_webui = subparsers.add_parser("webui", help="Open the web dashboard (standalone)")
    p_webui.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_webui.add_argument("--port", type=int, default=7777, help="Port (default: 7777)")
    p_webui.set_defaults(func=cmd_webui)

    # learn (knowledge bootstrap)
    p_learn = subparsers.add_parser(
        "learn",
        help="Bootstrap foundational knowledge from Wikipedia + ConceptNet",
    )
    p_learn.add_argument(
        "--wikipedia", action="store_true",
        help="Phase 1: fetch Wikipedia article summaries",
    )
    p_learn.add_argument(
        "--conceptnet", action="store_true",
        help="Phase 2: load ConceptNet entity relationships",
    )
    p_learn.add_argument(
        "--topics", type=str, default=None,
        help="Comma-separated Wikipedia topic list (overrides defaults)",
    )
    p_learn.set_defaults(func=cmd_learn)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
