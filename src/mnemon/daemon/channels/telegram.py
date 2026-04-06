"""
Jarvis Telegram Channel — chat with Jarvis and receive proactive thoughts via Telegram.

This is the primary mobile interface. Unlike the web UI (which requires you to open
a browser), Telegram:
  - Pushes proactive thoughts to your phone automatically
  - Lets you chat from anywhere with a native keyboard
  - Supports slash commands for stats (/thoughts, /goals, /status)

Architecture:
  - Long-polling bot (no webhook needed — works behind NAT)
  - Forwards messages to Jarvis via the Unix socket IPC
  - Polls the proactive inbox every N seconds and pushes any new messages
  - Pairing model: only responds to the authorized chat_id (set on first /start)

Setup:
  1. Create a bot via @BotFather on Telegram → get a bot token
  2. Set JARVIS_TELEGRAM_TOKEN=<token> in your environment (or .env file)
  3. Start: `mnemon-daemon start` — Telegram runs as a daemon subtask
  4. Open the bot in Telegram and send /start to pair it

Config via environment:
  JARVIS_TELEGRAM_TOKEN   Bot token from BotFather (required)
  JARVIS_TELEGRAM_CHAT_ID Your chat ID (auto-saved on /start, or pre-set)
  JARVIS_TELEGRAM_POLL_S  Proactive push poll interval in seconds (default: 30)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slash command responses
# ---------------------------------------------------------------------------

HELP_TEXT = """\
*Jarvis Commands*

/thoughts — What Jarvis has been thinking about
/goals — Active goals
/status — Daemon stats (cycles, uptime, last think)
/soul — Read Jarvis's soul file
/master — What Jarvis knows about you
/help — This message

Or just send any message to chat.
"""


class JarvisTelegramBot:
    """Telegram bot that bridges your phone to the Jarvis daemon.

    Runs as a background asyncio task inside the daemon process.
    Handles incoming messages, slash commands, and pushes proactive
    messages from the daemon's inbox to your phone.
    """

    def __init__(
        self,
        token: str,
        socket_path: Path,
        state_dir: Path,
        authorized_chat_id: int | None = None,
        poll_interval_s: int = 30,
    ) -> None:
        self._token = token
        self._socket_path = socket_path
        self._state_dir = state_dir
        self._chat_id: int | None = authorized_chat_id
        self._poll_interval_s = poll_interval_s
        self._chat_id_file = state_dir / "telegram_chat_id.txt"
        self._running = False

        # Load persisted chat_id if not provided
        if self._chat_id is None and self._chat_id_file.exists():
            try:
                self._chat_id = int(self._chat_id_file.read_text().strip())
                logger.info("Loaded Telegram chat_id: %d", self._chat_id)
            except Exception:
                pass

    async def run(self) -> None:
        """Main entry point — runs polling loop and proactive push loop concurrently."""
        from telegram.ext import Application, CommandHandler, MessageHandler, filters

        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("thoughts", self._cmd_thoughts))
        self._app.add_handler(CommandHandler("goals", self._cmd_goals))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("soul", self._cmd_soul))
        self._app.add_handler(CommandHandler("master", self._cmd_master))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        self._running = True
        logger.info("Jarvis Telegram bot starting (chat_id=%s)", self._chat_id or "unpaired")

        async with self._app:
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)

            # Run proactive push alongside polling
            push_task = asyncio.create_task(self._proactive_push_loop())

            try:
                while self._running:
                    await asyncio.sleep(1)
            finally:
                push_task.cancel()
                await self._app.updater.stop()
                await self._app.stop()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Proactive push loop
    # ------------------------------------------------------------------

    async def _proactive_push_loop(self) -> None:
        """Poll the daemon's proactive inbox and push new messages to phone."""
        while self._running:
            await asyncio.sleep(self._poll_interval_s)
            if self._chat_id is None:
                continue
            try:
                await self._push_proactive_messages()
            except Exception:
                logger.debug("Proactive push failed.", exc_info=True)

    async def _push_proactive_messages(self) -> None:
        """Fetch unread proactive messages from daemon and send to Telegram."""
        from mnemon.daemon.cli.client import DaemonClient
        client = DaemonClient(self._socket_path)

        try:
            result = await client.status()
        except Exception:
            return  # Daemon not running

        inbox = result.get("proactive_inbox", [])
        unread = [m for m in inbox if not m.get("read")]

        for msg in unread:
            content = msg.get("content", "").strip()
            activity = msg.get("source_activity", "")
            if not content:
                continue

            label = {
                "help_master": "💡",
                "know_master": "🔍",
                "grow": "🌱",
            }.get(activity, "🤔")

            try:
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=f"{label} {content}",
                    parse_mode="Markdown",
                )
                logger.info("Pushed proactive message [%s]: %s", activity, content[:60])
            except Exception:
                logger.debug("Failed to send proactive message.", exc_info=True)

        # Mark all as read via IPC
        if unread:
            try:
                await client._rpc("inbox.mark_read", {})
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update, context) -> None:
        """Pair this chat with Jarvis."""
        chat_id = update.effective_chat.id

        if self._chat_id is not None and self._chat_id != chat_id:
            await update.message.reply_text(
                "⚠️ Jarvis is already paired with another chat. "
                "Clear JARVIS_TELEGRAM_CHAT_ID to re-pair."
            )
            return

        self._chat_id = chat_id
        self._chat_id_file.write_text(str(chat_id))
        logger.info("Telegram paired with chat_id: %d", chat_id)

        await update.message.reply_text(
            "✅ *Jarvis paired.* I'm always thinking. You'll hear from me.\n\n"
            + HELP_TEXT,
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update, context) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")

    async def _cmd_thoughts(self, update, context) -> None:
        if not self._is_authorized(update):
            return
        try:
            from mnemon.daemon.cli.client import DaemonClient
            client = DaemonClient(self._socket_path)
            thoughts = await client.thoughts(limit=5)
            if not thoughts:
                await update.message.reply_text("No thoughts recorded yet.")
                return
            lines = []
            for t in thoughts:
                ts = t.get("timestamp", "")[:16].replace("T", " ")
                activity = t.get("activity", "?")
                summary = t.get("summary", "")[:120]
                icon = {"help_master": "💡", "know_master": "🔍", "grow": "🌱",
                        "consolidation": "🧠", "exploration": "🗺️"}.get(activity, "💭")
                lines.append(f"{icon} *{activity}* `{ts}`\n{summary}")
            await update.message.reply_text("\n\n".join(lines), parse_mode="Markdown")
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")

    async def _cmd_goals(self, update, context) -> None:
        if not self._is_authorized(update):
            return
        try:
            from mnemon.daemon.cli.client import DaemonClient
            client = DaemonClient(self._socket_path)
            goals = await client.list_goals()
            if not goals:
                await update.message.reply_text("No active goals.")
                return
            lines = []
            for g in goals:
                pct = f"{g.get('progress', 0):.0%}"
                lines.append(f"• [{g.get('priority', 0):.1f}] {g.get('description', '')} ({pct})")
            await update.message.reply_text("\n".join(lines))
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")

    async def _cmd_status(self, update, context) -> None:
        if not self._is_authorized(update):
            return
        try:
            from mnemon.daemon.cli.client import DaemonClient
            client = DaemonClient(self._socket_path)
            result = await client.status()
            d = result.get("daemon", {})
            last = d.get("last_user_interaction") or "never"
            if last != "never":
                last = last[:16].replace("T", " ")
            text = (
                f"*Jarvis Status*\n"
                f"Cycles: {d.get('total_cycles', 0)}\n"
                f"Idle ticks: {d.get('total_idle_ticks', 0)}\n"
                f"Autonomy: {d.get('autonomy_level', '?')}\n"
                f"Last chat: {last}"
            )
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")

    async def _cmd_soul(self, update, context) -> None:
        if not self._is_authorized(update):
            return
        soul_path = self._state_dir / "soul.md"
        if not soul_path.exists():
            await update.message.reply_text("soul.md not found yet.")
            return
        content = soul_path.read_text()[:3000]
        await update.message.reply_text(f"```\n{content}\n```", parse_mode="Markdown")

    async def _cmd_master(self, update, context) -> None:
        if not self._is_authorized(update):
            return
        master_path = self._state_dir / "master.md"
        if not master_path.exists():
            await update.message.reply_text("master.md not found yet.")
            return
        content = master_path.read_text()[:3000]
        await update.message.reply_text(f"```\n{content}\n```", parse_mode="Markdown")

    # ------------------------------------------------------------------
    # Chat handler
    # ------------------------------------------------------------------

    async def _handle_message(self, update, context) -> None:
        """Forward a plain message to Jarvis and reply with the response."""
        if not self._is_authorized(update):
            return

        message = update.message.text.strip()
        if not message:
            return

        # Show typing indicator
        await update.message.chat.send_action("typing")

        try:
            from mnemon.daemon.cli.client import DaemonClient
            client = DaemonClient(self._socket_path)
            result = await client.chat(message)
            reply = result.get("reply", "").strip()

            if not reply:
                reply = "(no reply)"

            await update.message.reply_text(reply)

        except Exception as exc:
            logger.exception("Telegram chat handler error.")
            await update.message.reply_text(
                f"⚠️ Couldn't reach Jarvis daemon: {exc}\n"
                "Is it running? Try `mnemon-daemon status`."
            )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _is_authorized(self, update) -> bool:
        """Only respond to the paired chat_id."""
        if self._chat_id is None:
            return False
        if update.effective_chat.id != self._chat_id:
            logger.warning(
                "Rejected message from unpaired chat_id: %d",
                update.effective_chat.id,
            )
            return False
        return True
