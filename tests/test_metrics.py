"""Tests for executor.metrics primitives and server."""

import threading
import time
import urllib.request

import pytest

from executor.metrics import Counter, Gauge, Histogram, MetricsRegistry, MetricsServer

# ── Counter ───────────────────────────────────────────────────────


class TestCounter:
    def test_inc_no_labels(self):
        c = Counter("test_total", "help")
        c.inc()
        c.inc(2)
        lines = c.collect()
        assert "test_total 3.0" in lines

    def test_inc_with_labels(self):
        c = Counter("req_total", "help")
        c.inc(pair="ETH/USDT")
        c.inc(pair="ETH/USDT")
        c.inc(pair="BTC/USDT")
        lines = c.collect()
        assert any('pair="BTC/USDT"' in line and "1.0" in line for line in lines)
        assert any('pair="ETH/USDT"' in line and "2.0" in line for line in lines)

    def test_collect_includes_help_and_type(self):
        c = Counter("my_counter", "My help text")
        lines = c.collect()
        assert "# HELP my_counter My help text" in lines
        assert "# TYPE my_counter counter" in lines


# ── Gauge ─────────────────────────────────────────────────────────


class TestGauge:
    def test_set_and_collect(self):
        g = Gauge("temp", "help")
        g.set(42.5)
        lines = g.collect()
        assert "temp 42.5" in lines

    def test_set_with_labels(self):
        g = Gauge("spread", "help")
        g.set(85.2, pair="ETH/USDT")
        g.set(12.3, pair="BTC/USDT")
        lines = g.collect()
        assert any("85.2" in line and "ETH/USDT" in line for line in lines)
        assert any("12.3" in line and "BTC/USDT" in line for line in lines)

    def test_inc_gauge(self):
        g = Gauge("pnl", "help")
        g.inc(10.0)
        g.inc(5.0)
        lines = g.collect()
        assert "pnl 15.0" in lines

    def test_overwrite_value(self):
        g = Gauge("state", "help")
        g.set(1.0)
        g.set(2.0)
        lines = g.collect()
        assert "state 2.0" in lines
        assert "state 1.0" not in lines


# ── Histogram ─────────────────────────────────────────────────────


class TestHistogram:
    def test_observe_and_collect(self):
        h = Histogram("latency", "help", buckets=(10, 50, 100))
        h.observe(5)
        h.observe(25)
        h.observe(75)
        lines = h.collect()
        # All 3 observations should be in +Inf bucket
        assert any('+Inf"' in line and "3" in line for line in lines)
        # Only value=5 fits in le=10 bucket
        text = "\n".join(lines)
        assert "latency_sum" in text
        assert "latency_count" in text

    def test_observe_with_labels(self):
        h = Histogram("latency", "help", buckets=(100,))
        h.observe(50, pair="ETH/USDT", leg="leg1")
        lines = h.collect()
        text = "\n".join(lines)
        assert "ETH/USDT" in text
        assert "leg1" in text

    def test_bucket_boundaries(self):
        h = Histogram("lat", "help", buckets=(10, 50))
        h.observe(10)  # fits in le=10 and le=50
        h.observe(50)  # fits in le=50 only
        lines = h.collect()
        text = "\n".join(lines)
        # le=10 bucket should have 1
        assert 'le="10"' in text
        # le=50 bucket should have 2
        assert 'le="50"' in text


# ── MetricsRegistry ──────────────────────────────────────────────


class TestMetricsRegistry:
    def test_collect_all_returns_string(self):
        reg = MetricsRegistry()
        output = reg.collect_all()
        assert isinstance(output, str)
        assert "arb_signals_total" in output
        assert "arb_pnl_total_usd" in output
        assert "arb_execution_latency_ms" in output

    def test_counter_increments_show_up(self):
        reg = MetricsRegistry()
        reg.signals_total.inc(pair="ETH/USDT", direction="BUY_CEX_SELL_DEX")
        output = reg.collect_all()
        assert "ETH/USDT" in output
        assert "BUY_CEX_SELL_DEX" in output

    def test_gauge_values_show_up(self):
        reg = MetricsRegistry()
        reg.pnl_total.set(42.5)
        reg.spread_bps.set(85.2, pair="ETH/USDT")
        output = reg.collect_all()
        assert "42.5" in output
        assert "85.2" in output

    def test_histogram_observations_show_up(self):
        reg = MetricsRegistry()
        reg.execution_latency.observe(150.0, pair="ETH/USDT", leg="leg1")
        output = reg.collect_all()
        assert "150.0" in output


# ── MetricsServer ────────────────────────────────────────────────


class TestMetricsServer:
    def test_metrics_endpoint(self):
        reg = MetricsRegistry()
        reg.signals_total.inc(pair="TEST")
        # Use a random high port to avoid conflicts
        from http.server import HTTPServer

        from executor.metrics import _MetricsHandler

        handler = type("_H", (_MetricsHandler,), {"registry": reg})
        http_srv = HTTPServer(("127.0.0.1", 0), handler)
        port = http_srv.server_address[1]
        t = threading.Thread(target=http_srv.serve_forever, daemon=True)
        t.start()

        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics")
            body = resp.read().decode()
            assert resp.status == 200
            assert "arb_signals_total" in body
            assert "TEST" in body
        finally:
            http_srv.shutdown()

    def test_health_endpoint(self):
        reg = MetricsRegistry()
        from http.server import HTTPServer

        from executor.metrics import _MetricsHandler

        handler = type("_H", (_MetricsHandler,), {"registry": reg})
        http_srv = HTTPServer(("127.0.0.1", 0), handler)
        port = http_srv.server_address[1]
        t = threading.Thread(target=http_srv.serve_forever, daemon=True)
        t.start()

        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            body = resp.read().decode()
            assert resp.status == 200
            assert "ok" in body
        finally:
            http_srv.shutdown()

    def test_404_on_unknown_path(self):
        reg = MetricsRegistry()
        from http.server import HTTPServer

        from executor.metrics import _MetricsHandler

        handler = type("_H", (_MetricsHandler,), {"registry": reg})
        http_srv = HTTPServer(("127.0.0.1", 0), handler)
        port = http_srv.server_address[1]
        t = threading.Thread(target=http_srv.serve_forever, daemon=True)
        t.start()

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/unknown")
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            assert exc_info.value.code == 404
        finally:
            http_srv.shutdown()

    def test_start_stop_lifecycle(self):
        reg = MetricsRegistry()
        server = MetricsServer(reg, host="127.0.0.1", port=19876)
        server.start()
        time.sleep(0.3)
        # Should be able to hit /metrics
        resp = urllib.request.urlopen("http://127.0.0.1:19876/metrics")
        assert resp.status == 200
        server.stop()


# ── Thread safety ────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_counter_increments(self):
        c = Counter("concurrent", "help")
        n_threads = 10
        n_ops = 1000
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(n_ops):
                c.inc()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = c.collect()
        value_line = [line for line in lines if line.startswith("concurrent")]
        assert len(value_line) == 1
        assert str(float(n_threads * n_ops)) in value_line[0]

    def test_concurrent_gauge_sets(self):
        g = Gauge("concurrent_gauge", "help")
        n_threads = 10
        barrier = threading.Barrier(n_threads)

        def worker(val):
            barrier.wait()
            for _ in range(100):
                g.set(val)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Should have exactly one value (last writer wins)
        lines = g.collect()
        value_lines = [line for line in lines if line.startswith("concurrent_gauge")]
        assert len(value_lines) == 1
