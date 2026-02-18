from __future__ import annotations

"""
Minimal Telegram bot integration for alerts + kill switch control.

Environment variables:
  TELEGRAM_BOT_TOKEN   — Bot token from BotFather
  TELEGRAM_CHAT_ID     — Chat ID (user or group) to send messages to
  TELEGRAM_POLL_SEC    — Optional, poll interval for commands (default: 2.0)
  TELEGRAM_LOG_LEVEL   — Optional, min level to send logs (default: WARNING).
                         Use INFO to get more messages, WARNING to reduce spam.

Commands handled (from TELEGRAM_CHAT_ID only):
  /kill    — create the kill-switch file (bots will stop shortly)
  /resume  — remove the kill-switch file
  /status  — report kill-switch and basic health status
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from safety import KILL_SWITCH_FILE

logger = logging.getLogger(__name__)

# Telegram message size limit (UTF-8)
TELEGRAM_TEXT_MAX_LEN = 4000

# Reply keyboard with three command buttons shown at the bottom of the chat
REPLY_KEYBOARD = {
    "keyboard": [
        [{"text": "/kill"}, {"text": "/status"}, {"text": "/resume"}],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": False,
}


class TelegramLogHandler(logging.Handler):
    """
    Sends log records to Telegram via a send(text) callable.
    Use TELEGRAM_LOG_LEVEL (default WARNING) to control verbosity.
    """

    def __init__(self, send_fn, level=logging.NOTSET):
        super().__init__(level=level)
        self._send = send_fn

    def emit(self, record: logging.LogRecord) -> None:
        if not self._send:
            return
        try:
            text = self.format(record)
            if len(text) > TELEGRAM_TEXT_MAX_LEN:
                text = text[: TELEGRAM_TEXT_MAX_LEN - 3] + "..."
            self._send(text)
        except Exception as exc:
            logger.warning("TelegramLogHandler emit failed: %s", exc)


def add_telegram_log_handler(
    bot: "TelegramBot",
    root_logger: Optional[logging.Logger] = None,
    level: Optional[str] = None,
) -> None:
    """
    Add a handler that forwards log records to the Telegram bot.
    If level is None, uses TELEGRAM_LOG_LEVEL env (default WARNING).
    """
    if not bot.config.enabled:
        return
    level_name = (level or os.getenv("TELEGRAM_LOG_LEVEL", "WARNING")).upper()
    log_level = getattr(logging, level_name, logging.WARNING)
    handler = TelegramLogHandler(bot.send, level=log_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    (root_logger or logging.getLogger()).addHandler(handler)
    logger.info("Telegram log handler added (level=%s)", level_name)


@dataclass
class TelegramBotConfig:
    token: Optional[str]
    chat_id: Optional[str]
    poll_interval: float = 2.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "TelegramBotConfig":
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        enabled = bool(token and chat_id)
        interval = float(os.getenv("TELEGRAM_POLL_SEC", "2.0") or "2.0")
        return cls(
            token=token, chat_id=chat_id, poll_interval=interval, enabled=enabled
        )


class TelegramBot:
    """Lightweight Telegram bot client with polling for kill-switch commands."""

    def __init__(self, config: TelegramBotConfig):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update_id: Optional[int] = None

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        if not self.config.enabled:
            logger.info("Telegram bot disabled (missing TELEGRAM_BOT_TOKEN or CHAT_ID)")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="telegram-bot"
        )
        self._thread.start()
        logger.info("Telegram bot polling started for chat_id=%s", self.config.chat_id)
        # Show /kill, /status, /resume buttons at the bottom of the chat
        self.send_with_command_buttons(
            "Arb_Bot ready. Use the buttons below or type /kill, /status, /resume."
        )

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ── public API ───────────────────────────────────────────────

    def send(self, text: str, reply_markup: Optional[dict] = None) -> None:
        """Send a best-effort message to the configured chat.
        If reply_markup is provided (e.g. REPLY_KEYBOARD), buttons appear at the bottom.
        """
        if not self.config.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.config.token}/sendMessage"
            payload = {"chat_id": self.config.chat_id, "text": text}
            if reply_markup:
                payload["reply_markup"] = reply_markup
            resp = requests.post(url, json=payload, timeout=5.0)
            if resp.status_code >= 400:
                logger.warning(
                    "Telegram send failed (%d): %s", resp.status_code, resp.text
                )
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Telegram send error: %s", exc)

    def send_with_command_buttons(self, text: str) -> None:
        """Send a message and show the /kill, /status, /resume buttons at the bottom."""
        self.send(text, reply_markup=REPLY_KEYBOARD)

    # ── polling loop ─────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Telegram poll error: %s", exc)
            time.sleep(self.config.poll_interval)

    def _poll_once(self) -> None:
        url = f"https://api.telegram.org/bot{self.config.token}/getUpdates"
        params = {"timeout": 0}
        if self._last_update_id is not None:
            params["offset"] = self._last_update_id + 1
        resp = requests.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            return

        for update in data.get("result", []):
            self._last_update_id = int(update.get("update_id", 0))
            msg = update.get("message") or {}
            chat = msg.get("chat") or {}
            text = msg.get("text") or ""
            chat_id = str(chat.get("id"))

            # Only accept commands from the configured chat_id
            if chat_id != str(self.config.chat_id):
                continue

            text = text.strip()
            if not text.startswith("/"):
                continue

            if text.startswith("/kill"):
                Path(KILL_SWITCH_FILE).touch()
                self.send("Kill switch ACTIVATED. Bots will stop shortly.")
                logger.warning("Telegram /kill received — kill switch file created.")
            elif text.startswith("/resume"):
                try:
                    Path(KILL_SWITCH_FILE).unlink(missing_ok=True)
                except TypeError:  # Python <3.8
                    if Path(KILL_SWITCH_FILE).exists():
                        Path(KILL_SWITCH_FILE).unlink()
                self.send("Kill switch CLEARED. You may restart bots.")
                logger.info("Telegram /resume received — kill switch file removed.")
            elif text.startswith("/status"):
                active = Path(KILL_SWITCH_FILE).exists()
                status = "ACTIVE" if active else "inactive"
                self.send(f"Kill switch status: {status}")
