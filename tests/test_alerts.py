"""Tests for executor.alerts — WebhookAlerter."""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from executor.alerts import Alert, AlertLevel, AlertType, WebhookAlerter, WebhookConfig

# ── helpers ────────────────────────────────────────────────────────


class _RecordingHandler(BaseHTTPRequestHandler):
    """HTTP handler that records received payloads."""

    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.received.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def _start_test_server():
    """Spin up a tiny HTTP server and return (url, server)."""
    _RecordingHandler.received = []
    srv = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{port}/hook", srv


def _make_alert(**overrides):
    defaults = dict(
        alert_type=AlertType.CIRCUIT_BREAKER_TRIP,
        level=AlertLevel.CRITICAL,
        pair="ETH/USDT",
        message="test alert",
    )
    defaults.update(overrides)
    return Alert(**defaults)


# ── Alert dataclass ───────────────────────────────────────────────


class TestAlertPayload:
    def test_to_payload_contains_all_fields(self):
        a = _make_alert(details={"k": "v"})
        p = a.to_payload()
        assert p["type"] == "CIRCUIT_BREAKER_TRIP"
        assert p["level"] == "critical"
        assert p["pair"] == "ETH/USDT"
        assert p["message"] == "test alert"
        assert p["details"] == {"k": "v"}
        assert isinstance(p["timestamp"], float)

    def test_default_details_empty(self):
        a = _make_alert()
        assert a.details == {}


# ── WebhookConfig ─────────────────────────────────────────────────


class TestWebhookConfig:
    def test_from_env_no_urls(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = WebhookConfig.from_env()
        assert cfg.urls == []
        assert cfg.enabled is False

    def test_from_env_with_urls(self):
        env = {
            "WEBHOOK_URLS": "http://a.com/hook, http://b.com/hook",
            "WEBHOOK_TIMEOUT": "3",
            "WEBHOOK_MAX_RETRIES": "2",
            "WEBHOOK_COOLDOWN": "30",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = WebhookConfig.from_env()
        assert len(cfg.urls) == 2
        assert cfg.enabled is True
        assert cfg.timeout_seconds == 3.0
        assert cfg.max_retries == 2
        assert cfg.cooldown_seconds == 30.0


# ── WebhookAlerter ────────────────────────────────────────────────


class TestAlerterDisabled:
    def test_send_returns_false_when_disabled(self):
        alerter = WebhookAlerter(WebhookConfig(urls=[], enabled=False))
        assert alerter.send(_make_alert()) is False

    def test_start_does_nothing_when_disabled(self):
        alerter = WebhookAlerter(WebhookConfig(urls=[], enabled=False))
        alerter.start()
        assert alerter._worker is None
        alerter.stop()


class TestAlerterCooldown:
    def test_duplicate_alert_throttled(self):
        cfg = WebhookConfig(urls=["http://fake"], enabled=True, cooldown_seconds=60)
        alerter = WebhookAlerter(cfg)
        # Don't start the worker — we just test queueing
        assert alerter.send(_make_alert()) is True
        # Same type+pair within cooldown → throttled
        assert alerter.send(_make_alert()) is False

    def test_different_pair_not_throttled(self):
        cfg = WebhookConfig(urls=["http://fake"], enabled=True, cooldown_seconds=60)
        alerter = WebhookAlerter(cfg)
        assert alerter.send(_make_alert(pair="ETH/USDT")) is True
        assert alerter.send(_make_alert(pair="BTC/USDT")) is True

    def test_different_type_not_throttled(self):
        cfg = WebhookConfig(urls=["http://fake"], enabled=True, cooldown_seconds=60)
        alerter = WebhookAlerter(cfg)
        assert (
            alerter.send(_make_alert(alert_type=AlertType.CIRCUIT_BREAKER_TRIP)) is True
        )
        assert alerter.send(_make_alert(alert_type=AlertType.EXECUTION_FAILURE)) is True


class TestAlerterDelivery:
    def test_delivers_to_real_server(self):
        url, srv = _start_test_server()
        try:
            cfg = WebhookConfig(
                urls=[url],
                enabled=True,
                cooldown_seconds=0,
                timeout_seconds=2,
                max_retries=0,
            )
            alerter = WebhookAlerter(cfg)
            alerter.start()
            alerter.send(_make_alert())
            # Wait for background delivery
            time.sleep(1.0)
            alerter.stop()
            assert len(_RecordingHandler.received) == 1
            assert alerter._sent_count == 1
        finally:
            srv.shutdown()

    def test_retry_on_failure(self):
        """If server returns 500 first, then 200, delivery retries."""
        call_count = {"n": 0}

        class _FlakeyHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    self.send_response(500)
                else:
                    self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), _FlakeyHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        try:
            cfg = WebhookConfig(
                urls=[f"http://127.0.0.1:{port}/hook"],
                enabled=True,
                cooldown_seconds=0,
                timeout_seconds=2,
                max_retries=2,
                retry_base_delay=0.1,
            )
            alerter = WebhookAlerter(cfg)
            alerter.start()
            alerter.send(_make_alert())
            time.sleep(2.0)
            alerter.stop()
            assert call_count["n"] >= 2
            assert alerter._sent_count == 1
        finally:
            srv.shutdown()

    def test_all_retries_fail(self):
        """If server always returns 500, delivery fails after retries."""

        class _FailHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), _FailHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        try:
            cfg = WebhookConfig(
                urls=[f"http://127.0.0.1:{port}/hook"],
                enabled=True,
                cooldown_seconds=0,
                timeout_seconds=1,
                max_retries=1,
                retry_base_delay=0.1,
            )
            alerter = WebhookAlerter(cfg)
            alerter.start()
            alerter.send(_make_alert())
            time.sleep(2.0)
            alerter.stop()
            assert alerter._failed_count == 1
        finally:
            srv.shutdown()


class TestConvenienceSenders:
    def setup_method(self):
        self.cfg = WebhookConfig(urls=["http://fake"], enabled=True, cooldown_seconds=0)
        self.alerter = WebhookAlerter(self.cfg)

    def test_on_circuit_breaker_trip(self):
        self.alerter.on_circuit_breaker_trip("ETH/USDT", {"state": "open"})
        assert self.alerter._queue.qsize() == 1

    def test_on_circuit_breaker_half_open(self):
        self.alerter.on_circuit_breaker_half_open("ETH/USDT", {})
        assert self.alerter._queue.qsize() == 1

    def test_on_circuit_breaker_reset(self):
        self.alerter.on_circuit_breaker_reset("ETH/USDT", {})
        assert self.alerter._queue.qsize() == 1

    def test_on_execution_failure(self):
        self.alerter.on_execution_failure("ETH/USDT", "timeout", True)
        assert self.alerter._queue.qsize() == 1

    def test_on_drawdown(self):
        self.alerter.on_drawdown(-100.0, 50.0)
        assert self.alerter._queue.qsize() == 1


class TestAlerterStats:
    def test_stats_structure(self):
        alerter = WebhookAlerter(WebhookConfig(urls=["http://x"], enabled=True))
        s = alerter.stats
        assert "enabled" in s
        assert "endpoints" in s
        assert "sent" in s
        assert "failed" in s
        assert "queued" in s
        assert "recent" in s
