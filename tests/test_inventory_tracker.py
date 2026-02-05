from decimal import Decimal

from inventory.tracker import InventoryTracker, Venue


def _build_tracker() -> InventoryTracker:
    tracker = InventoryTracker([Venue.BINANCE, Venue.WALLET])
    tracker.update_from_cex(
        Venue.BINANCE,
        {
            "ETH": {"free": Decimal("1.5"), "locked": Decimal("0.5")},
            "USDT": {"free": Decimal("1000"), "locked": Decimal("0")},
        },
    )
    tracker.update_from_wallet(
        Venue.WALLET,
        {
            "ETH": Decimal("2.0"),
            "USDT": Decimal("100"),
        },
    )
    return tracker


def test_snapshot_aggregates_across_venues():
    tracker = _build_tracker()
    snapshot = tracker.snapshot()
    assert snapshot["totals"]["ETH"] == Decimal("4.0")
    assert snapshot["totals"]["USDT"] == Decimal("1100")


def test_can_execute_passes_when_sufficient():
    tracker = _build_tracker()
    result = tracker.can_execute(
        buy_venue=Venue.BINANCE,
        buy_asset="USDT",
        buy_amount=Decimal("500"),
        sell_venue=Venue.WALLET,
        sell_asset="ETH",
        sell_amount=Decimal("1"),
    )
    assert result["can_execute"] is True
    assert result["reason"] is None


def test_can_execute_fails_insufficient_buy():
    tracker = _build_tracker()
    result = tracker.can_execute(
        buy_venue=Venue.BINANCE,
        buy_asset="USDT",
        buy_amount=Decimal("5000"),
        sell_venue=Venue.WALLET,
        sell_asset="ETH",
        sell_amount=Decimal("1"),
    )
    assert result["can_execute"] is False
    assert "Insufficient USDT" in result["reason"]


def test_can_execute_fails_insufficient_sell():
    tracker = _build_tracker()
    result = tracker.can_execute(
        buy_venue=Venue.BINANCE,
        buy_asset="USDT",
        buy_amount=Decimal("100"),
        sell_venue=Venue.WALLET,
        sell_asset="ETH",
        sell_amount=Decimal("10"),
    )
    assert result["can_execute"] is False
    assert "Insufficient ETH" in result["reason"]


def test_record_trade_updates_balances():
    tracker = _build_tracker()
    tracker.record_trade(
        venue=Venue.BINANCE,
        side="buy",
        base_asset="ETH",
        quote_asset="USDT",
        base_amount=Decimal("1"),
        quote_amount=Decimal("200"),
        fee=Decimal("1"),
        fee_asset="USDT",
    )
    snapshot = tracker.snapshot()
    binance = snapshot["venues"]["binance"]
    assert binance["ETH"]["free"] == Decimal("2.5")
    assert binance["USDT"]["free"] == Decimal("799")


def test_skew_detects_imbalance():
    tracker = InventoryTracker([Venue.BINANCE, Venue.WALLET])
    tracker.update_from_cex(
        Venue.BINANCE, {"ETH": {"free": Decimal("8"), "locked": Decimal("0")}}
    )
    tracker.update_from_wallet(Venue.WALLET, {"ETH": Decimal("2")})
    result = tracker.skew("ETH")
    assert result["needs_rebalance"] is True
    assert result["max_deviation_pct"] > 30.0


def test_skew_balanced():
    tracker = InventoryTracker([Venue.BINANCE, Venue.WALLET])
    tracker.update_from_cex(
        Venue.BINANCE, {"ETH": {"free": Decimal("5"), "locked": Decimal("0")}}
    )
    tracker.update_from_wallet(Venue.WALLET, {"ETH": Decimal("5")})
    result = tracker.skew("ETH")
    assert result["needs_rebalance"] is False
    assert abs(result["max_deviation_pct"]) < 0.01
