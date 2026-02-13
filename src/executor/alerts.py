"""
Webhook Alerts — fire HTTP notifications on critical events.

Supported triggers:
  * Circuit breaker trip (OPEN)
  * Circuit breaker half-open (probe allowed)
  * Circuit breaker reset (CLOSED)
  * Execution failure with unwind
  * Drawdown threshold hit

Each alert is a JSON POST to one or more webhook URLs.
Failed deliveries are retried with exponential back-off in a background
thread so they never block the hot path.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Empty, Queue
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    CIRCUIT_BREAKER_TRIP = auto()
    CIRCUIT_BREAKER_HALF_OPEN = auto()
    CIRCUIT_BREAKER_RESET = auto()
    EXECUTION_FAILURE = auto()
    UNWIND_TRIGGERED = auto()
    DRAWDOWN_ALERT = auto()
    CUSTOM = auto()


@dataclass
class Alert:
    """One alert payload."""

    alert_type: AlertType
    level: AlertLevel
    pair: Optional[str]
    message: str
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_payload(self) -> dict:
        return {
            "type": self.alert_type.name,
            "level": self.level.value,
            "pair": self.pair,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class WebhookConfig:
    """Configuration for the webhook alerter."""

    urls: list[str] = field(default_factory=list)
    timeout_seconds: float = 5.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    enabled: bool = True
    # Rate-limit: at most 1 alert of same type+pair per N seconds
    cooldown_seconds: float = 60.0
    # Optional secret for HMAC signing (future)
    secret: Optional[str] = None

    @classmethod
    def from_env(cls) -> "WebhookConfig":
        """Build config from environment variables."""
        raw = os.getenv("WEBHOOK_URLS", "")
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        return cls(
            urls=urls,
            timeout_seconds=float(os.getenv("WEBHOOK_TIMEOUT", "5")),
            max_retries=int(os.getenv("WEBHOOK_MAX_RETRIES", "3")),
            enabled=bool(urls),
            cooldown_seconds=float(os.getenv("WEBHOOK_COOLDOWN", "60")),
            secret=os.getenv("WEBHOOK_SECRET"),
        )


class WebhookAlerter:
    """
    Async-safe webhook alerter with background delivery.

    Usage::

        alerter = WebhookAlerter(WebhookConfig.from_env())
        alerter.start()

        # Fire-and-forget — never blocks
        alerter.send(Alert(
            alert_type=AlertType.CIRCUIT_BREAKER_TRIP,
            level=AlertLevel.CRITICAL,
            pair="ETH/USDT",
            message="Circuit breaker tripped",
            details={"failures": 3, "pnl": -12.5},
        ))

        alerter.stop()
    """

    def __init__(self, config: Optional[WebhookConfig] = None):
        self.config = config or WebhookConfig()
        self._queue: Queue[Alert] = Queue(maxsize=500)
        self._worker: Optional[threading.Thread] = None
        self._running = False
        # Cooldown tracking: (alert_type, pair) → last_sent_ts
        self._cooldowns: dict[tuple[str, Optional[str]], float] = {}
        self._sent_count = 0
        self._failed_count = 0
        self._history: list[dict] = []
        self._history_max = 100

    # ── lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the background delivery thread."""
        if not self.config.enabled or not self.config.urls:
            logger.info("Webhook alerter disabled (no URLs configured)")
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._delivery_loop, daemon=True, name="webhook-alerter"
        )
        self._worker.start()
        logger.info("Webhook alerter started — %d endpoint(s)", len(self.config.urls))

    def stop(self) -> None:
        """Gracefully stop the delivery thread."""
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5.0)

    # ── public API ─────────────────────────────────────────────

    def send(self, alert: Alert) -> bool:
        """
        Enqueue an alert for delivery.

        Returns True if queued, False if dropped (disabled / cooldown / full).
        """
        if not self.config.enabled or not self.config.urls:
            return False

        # Cooldown check
        key = (alert.alert_type.name, alert.pair)
        now = time.time()
        last = self._cooldowns.get(key, 0.0)
        if now - last < self.config.cooldown_seconds:
            logger.debug(
                "Alert %s/%s throttled (cooldown)", alert.alert_type.name, alert.pair
            )
            return False

        try:
            self._queue.put_nowait(alert)
            self._cooldowns[key] = now
            return True
        except Exception:
            logger.warning("Alert queue full — dropping %s", alert.alert_type.name)
            return False

    # ── convenience senders ────────────────────────────────────

    def on_circuit_breaker_trip(self, pair: Optional[str], snapshot: dict) -> None:
        """Fire alert when circuit breaker trips to OPEN."""
        self.send(
            Alert(
                alert_type=AlertType.CIRCUIT_BREAKER_TRIP,
                level=AlertLevel.CRITICAL,
                pair=pair,
                message=f"Circuit breaker TRIPPED{f' for {pair}' if pair else ''}",
                details=snapshot,
            )
        )

    def on_circuit_breaker_half_open(self, pair: Optional[str], snapshot: dict) -> None:
        """Fire alert when circuit breaker enters HALF_OPEN."""
        self.send(
            Alert(
                alert_type=AlertType.CIRCUIT_BREAKER_HALF_OPEN,
                level=AlertLevel.WARNING,
                pair=pair,
                message=f"Circuit breaker HALF-OPEN{f' for {pair}' if pair else ''}",
                details=snapshot,
            )
        )

    def on_circuit_breaker_reset(self, pair: Optional[str], snapshot: dict) -> None:
        """Fire alert when circuit breaker resets to CLOSED."""
        self.send(
            Alert(
                alert_type=AlertType.CIRCUIT_BREAKER_RESET,
                level=AlertLevel.INFO,
                pair=pair,
                message=f"Circuit breaker RESET{f' for {pair}' if pair else ''}",
                details=snapshot,
            )
        )

    def on_execution_failure(self, pair: str, error: str, unwound: bool) -> None:
        """Fire alert on execution failure."""
        level = AlertLevel.CRITICAL if not unwound else AlertLevel.WARNING
        self.send(
            Alert(
                alert_type=AlertType.EXECUTION_FAILURE,
                level=level,
                pair=pair,
                message=f"Execution failed: {error}",
                details={"unwound": unwound},
            )
        )

    def on_drawdown(self, current_pnl: float, threshold: float) -> None:
        """Fire alert when cumulative PnL breaches drawdown threshold."""
        self.send(
            Alert(
                alert_type=AlertType.DRAWDOWN_ALERT,
                level=AlertLevel.CRITICAL,
                pair=None,
                message=f"Drawdown alert: PnL ${current_pnl:.2f} < -${threshold:.2f}",
                details={
                    "current_pnl": current_pnl,
                    "threshold": threshold,
                },
            )
        )

    # ── stats ──────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "endpoints": len(self.config.urls),
            "sent": self._sent_count,
            "failed": self._failed_count,
            "queued": self._queue.qsize(),
            "recent": self._history[-10:],
        }

    # ── delivery loop (background thread) ──────────────────────

    def _delivery_loop(self) -> None:
        while self._running:
            try:
                alert = self._queue.get(timeout=1.0)
            except Empty:
                continue
            self._deliver(alert)

        # Drain remaining alerts on shutdown
        while not self._queue.empty():
            try:
                alert = self._queue.get_nowait()
                self._deliver(alert)
            except Empty:
                break

    def _deliver(self, alert: Alert) -> None:
        payload = alert.to_payload()
        for url in self.config.urls:
            success = self._post_with_retry(url, payload)
            record = {
                "type": alert.alert_type.name,
                "url": url,
                "success": success,
                "ts": time.time(),
            }
            self._history.append(record)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max :]

    def _post_with_retry(self, url: str, payload: dict) -> bool:
        for attempt in range(1 + self.config.max_retries):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code < 400:
                    self._sent_count += 1
                    logger.debug("Webhook delivered to %s", url)
                    return True
                logger.warning("Webhook %s returned %d", url, resp.status_code)
            except requests.RequestException as exc:
                logger.warning(
                    "Webhook delivery attempt %d failed: %s", attempt + 1, exc
                )

            if attempt < self.config.max_retries:
                delay = self.config.retry_base_delay * (2**attempt)
                time.sleep(delay)

        self._failed_count += 1
        logger.error("Webhook delivery to %s failed after retries", url)
        return False
