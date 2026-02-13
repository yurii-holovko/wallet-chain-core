"""
Prometheus Metrics Export — expose bot telemetry via ``/metrics``.

Runs a lightweight HTTP server on a background thread.  All counters
and gauges are plain Python objects — no ``prometheus_client`` dependency
required (we emit the text exposition format directly).

Metrics exposed:

  Counters
  ~~~~~~~~
  * ``arb_signals_total{pair,direction}``       — signals generated
  * ``arb_executions_total{pair,state}``         — execution outcomes
  * ``arb_unwinds_total{pair,success}``          — unwind attempts
  * ``arb_circuit_breaker_trips_total{pair}``    — CB trip events
  * ``arb_webhook_sent_total``                   — webhook deliveries

  Gauges
  ~~~~~~
  * ``arb_spread_bps{pair}``                     — last observed spread
  * ``arb_score{pair}``                          — last signal score
  * ``arb_pnl_total_usd``                        — cumulative PnL
  * ``arb_inventory_skew_pct{pair,venue}``        — inventory deviation
  * ``arb_circuit_breaker_state{pair}``           — 0=closed, 1=open, 2=half
  * ``arb_queue_depth``                           — priority queue size

  Histograms (approximated with summary-style quantiles)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  * ``arb_execution_latency_ms{pair,leg}``        — leg latency
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

logger = logging.getLogger(__name__)


# ── Metric Primitives ────────────────────────────────────────────


class Counter:
    """Monotonically increasing counter with optional labels."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_str = self._format_labels(key)
                lines.append(f"{self.name}{label_str} {value}")
        return lines

    @staticmethod
    def _format_labels(key: tuple) -> str:
        if not key:
            return ""
        parts = [f'{k}="{v}"' for k, v in key]
        return "{" + ",".join(parts) + "}"


class Gauge:
    """Point-in-time value with optional labels."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_str = Counter._format_labels(key)
                lines.append(f"{self.name}{label_str} {value}")
        return lines


class Histogram:
    """
    Simplified histogram that tracks count, sum, and a few buckets.

    Not a full Prometheus histogram — just enough for useful percentiles.
    """

    DEFAULT_BUCKETS = (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ):
        self.name = name
        self.help = help_text
        self._buckets = sorted(buckets)
        self._data: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            if key not in self._data:
                self._data[key] = {
                    "count": 0,
                    "sum": 0.0,
                    "buckets": {b: 0 for b in self._buckets},
                }
            d = self._data[key]
            d["count"] += 1
            d["sum"] += value
            for b in self._buckets:
                if value <= b:
                    d["buckets"][b] += 1

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key, data in sorted(self._data.items()):
                label_str = Counter._format_labels(key)
                base = f"{self.name}"
                for b in self._buckets:
                    le_label = self._merge_labels(key, ("le", str(b)))
                    lines.append(f"{base}_bucket{le_label} {data['buckets'][b]}")
                inf_label = self._merge_labels(key, ("le", "+Inf"))
                lines.append(f"{base}_bucket{inf_label} {data['count']}")
                lines.append(f"{base}_sum{label_str} {data['sum']}")
                lines.append(f"{base}_count{label_str} {data['count']}")
        return lines

    @staticmethod
    def _merge_labels(key: tuple, extra: tuple) -> str:
        parts = list(key) + [extra]
        label_parts = [f'{k}="{v}"' for k, v in parts]
        return "{" + ",".join(label_parts) + "}"


# ── Metrics Registry ─────────────────────────────────────────────


class MetricsRegistry:
    """
    Central registry holding all bot metrics.

    Usage::

        metrics = MetricsRegistry()
        metrics.signals_total.inc(pair="ETH/USDT", direction="BUY_CEX_SELL_DEX")
        metrics.pnl_total.set(12.5)
    """

    def __init__(self):
        # Counters
        self.signals_total = Counter(
            "arb_signals_total", "Total arbitrage signals generated"
        )
        self.executions_total = Counter(
            "arb_executions_total", "Total execution attempts"
        )
        self.unwinds_total = Counter("arb_unwinds_total", "Total unwind attempts")
        self.cb_trips_total = Counter(
            "arb_circuit_breaker_trips_total",
            "Total circuit breaker trip events",
        )
        self.webhook_sent_total = Counter(
            "arb_webhook_sent_total", "Total webhook alerts sent"
        )

        # Gauges
        self.spread_bps = Gauge(
            "arb_spread_bps", "Last observed spread in basis points"
        )
        self.score = Gauge("arb_score", "Last signal score")
        self.pnl_total = Gauge("arb_pnl_total_usd", "Cumulative PnL in USD")
        self.inventory_skew = Gauge(
            "arb_inventory_skew_pct", "Inventory deviation percentage"
        )
        self.cb_state = Gauge(
            "arb_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
        )
        self.queue_depth = Gauge("arb_queue_depth", "Current priority queue size")

        # Histograms
        self.execution_latency = Histogram(
            "arb_execution_latency_ms", "Execution leg latency in ms"
        )

        self._all = [
            self.signals_total,
            self.executions_total,
            self.unwinds_total,
            self.cb_trips_total,
            self.webhook_sent_total,
            self.spread_bps,
            self.score,
            self.pnl_total,
            self.inventory_skew,
            self.cb_state,
            self.queue_depth,
            self.execution_latency,
        ]

    def collect_all(self) -> str:
        """Return all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        for metric in self._all:
            lines.extend(metric.collect())
            lines.append("")  # blank line between metrics
        return "\n".join(lines) + "\n"


# ── HTTP Server ──────────────────────────────────────────────────


class _MetricsHandler(BaseHTTPRequestHandler):
    """Minimal handler that serves ``/metrics``."""

    registry: Optional[MetricsRegistry] = None

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = (self.registry or MetricsRegistry()).collect_all()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/plain; version=0.0.4; charset=utf-8",
            )
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:  # noqa: A002
        # Suppress default stderr logging
        pass


class MetricsServer:
    """
    Background HTTP server exposing ``/metrics`` for Prometheus scraping.

    Usage::

        registry = MetricsRegistry()
        server = MetricsServer(registry, port=9090)
        server.start()

        # ... bot runs ...

        server.stop()
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        host: str = "0.0.0.0",
        port: int = 9090,
    ):
        self.registry = registry
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler = type(
            "_Handler",
            (_MetricsHandler,),
            {"registry": self.registry},
        )

        class _ReusableHTTPServer(HTTPServer):
            allow_reuse_address = True

        self._server = _ReusableHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="metrics-server",
        )
        self._thread.start()
        logger.info(
            "Prometheus metrics server started on %s:%d",
            self.host,
            self.port,
        )

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Metrics server stopped")
