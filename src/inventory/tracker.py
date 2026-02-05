# inventory/tracker.py

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum


class Venue(str, Enum):
    BINANCE = "binance"
    WALLET = "wallet"  # On-chain wallet (DEX venue)


@dataclass
class Balance:
    venue: Venue
    asset: str
    free: Decimal
    locked: Decimal = Decimal("0")

    @property
    def total(self) -> Decimal:
        return self.free + self.locked


class InventoryTracker:
    """
    Tracks positions across CEX and DEX venues.
    Single source of truth for where your money is.
    """

    def __init__(self, venues: list[Venue]):
        """Initialize tracker for given venues."""
        if not venues:
            raise ValueError("venues must be provided")
        self._venues = list(venues)
        self._balances: dict[Venue, dict[str, Balance]] = {
            venue: {} for venue in venues
        }

    @staticmethod
    def _to_decimal(value: Decimal | int | float | str) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _ensure_venue(self, venue: Venue) -> None:
        if venue not in self._balances:
            raise ValueError(f"Venue {venue} is not tracked")

    def _get_balance(self, venue: Venue, asset: str) -> Balance:
        self._ensure_venue(venue)
        asset_key = asset.upper()
        if asset_key not in self._balances[venue]:
            self._balances[venue][asset_key] = Balance(
                venue=venue, asset=asset_key, free=Decimal("0"), locked=Decimal("0")
            )
        return self._balances[venue][asset_key]

    def update_from_cex(self, venue: Venue, balances: dict):
        """
        Update balances from ExchangeClient.fetch_balance().
        Replaces previous snapshot for this venue.

        Args:
            venue: Which CEX venue
            balances: {asset: {free, locked, total}} from ExchangeClient
        """
        self._ensure_venue(venue)
        snapshot: dict[str, Balance] = {}
        for asset, values in (balances or {}).items():
            asset_key = str(asset).upper()
            free = self._to_decimal(values.get("free", "0"))
            locked = self._to_decimal(values.get("locked", "0"))
            snapshot[asset_key] = Balance(
                venue=venue, asset=asset_key, free=free, locked=locked
            )
        self._balances[venue] = snapshot

    def update_from_wallet(self, venue: Venue, balances: dict):
        """
        Update balances from on-chain wallet query.

        Args:
            venue: Wallet venue
            balances: {asset: amount} from chain/ module
        """
        self._ensure_venue(venue)
        snapshot: dict[str, Balance] = {}
        for asset, amount in (balances or {}).items():
            asset_key = str(asset).upper()
            free = self._to_decimal(amount)
            snapshot[asset_key] = Balance(
                venue=venue, asset=asset_key, free=free, locked=Decimal("0")
            )
        self._balances[venue] = snapshot

    def snapshot(self) -> dict:
        """
        Full portfolio snapshot at current time.

        Returns:
        {
            'timestamp': datetime,
            'venues': {
                'binance': {'ETH': {'free': ..., 'locked': ..., 'total': ...}, ...},
                'wallet':  {'ETH': {'free': ..., 'locked': ..., 'total': ...}, ...},
            },
            'totals': {
                'ETH':  Decimal('20.0'),
                'USDT': Decimal('40000.0'),
            },
            'total_usd': Decimal('80200.0'),  # requires price feed
        }
        """
        venues_snapshot: dict[str, dict] = {}
        totals: dict[str, Decimal] = {}
        for venue, assets in self._balances.items():
            venue_key = venue.value
            venues_snapshot[venue_key] = {}
            for asset, balance in assets.items():
                venues_snapshot[venue_key][asset] = {
                    "free": balance.free,
                    "locked": balance.locked,
                    "total": balance.total,
                }
                totals[asset] = totals.get(asset, Decimal("0")) + balance.total
        return {
            "timestamp": datetime.now(timezone.utc),
            "venues": venues_snapshot,
            "totals": totals,
            "total_usd": Decimal("0"),
        }

    def get_available(self, venue: Venue, asset: str) -> Decimal:
        """
        How much of `asset` is available to trade at `venue`.
        Returns free balance only (not locked in orders).
        """
        balance = self._get_balance(venue, asset)
        return balance.free

    def can_execute(
        self,
        buy_venue: Venue,
        buy_asset: str,  # What you're spending (e.g., "USDT")
        buy_amount: Decimal,  # How much you're spending
        sell_venue: Venue,
        sell_asset: str,  # What you're selling (e.g., "ETH")
        sell_amount: Decimal,  # How much you're selling
    ) -> dict:
        """
        Pre-flight check: can we execute both legs of an arb?

        Returns:
        {
            'can_execute': bool,
            'buy_venue_available': Decimal,
            'buy_venue_needed': Decimal,
            'sell_venue_available': Decimal,
            'sell_venue_needed': Decimal,
            'reason': str or None,  # Why not, if can_execute=False
        }
        """
        buy_available = self.get_available(buy_venue, buy_asset)
        sell_available = self.get_available(sell_venue, sell_asset)
        can_buy = buy_available >= buy_amount
        can_sell = sell_available >= sell_amount
        reason = None
        if not can_buy:
            reason = f"Insufficient {buy_asset} on {buy_venue.value}"
        elif not can_sell:
            reason = f"Insufficient {sell_asset} on {sell_venue.value}"
        return {
            "can_execute": can_buy and can_sell,
            "buy_venue_available": buy_available,
            "buy_venue_needed": buy_amount,
            "sell_venue_available": sell_available,
            "sell_venue_needed": sell_amount,
            "reason": reason,
        }

    def record_trade(
        self,
        venue: Venue,
        side: str,  # "buy" or "sell"
        base_asset: str,
        quote_asset: str,
        base_amount: Decimal,
        quote_amount: Decimal,
        fee: Decimal,
        fee_asset: str,
    ):
        """
        Update internal balances after a trade executes.
        Must handle: buy increases base / decreases quote,
                     sell decreases base / increases quote,
                     fee deducted from fee_asset.
        """
        side_value = side.lower()
        if side_value not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        base_balance = self._get_balance(venue, base_asset)
        quote_balance = self._get_balance(venue, quote_asset)
        fee_balance = self._get_balance(venue, fee_asset)

        base_amount_dec = self._to_decimal(base_amount)
        quote_amount_dec = self._to_decimal(quote_amount)
        fee_dec = self._to_decimal(fee)

        if side_value == "buy":
            if quote_balance.free < quote_amount_dec:
                raise ValueError("Insufficient quote asset for buy")
            quote_balance.free -= quote_amount_dec
            base_balance.free += base_amount_dec
        else:
            if base_balance.free < base_amount_dec:
                raise ValueError("Insufficient base asset for sell")
            base_balance.free -= base_amount_dec
            quote_balance.free += quote_amount_dec

        if fee_dec > 0:
            if fee_balance.free < fee_dec:
                raise ValueError("Insufficient balance to cover fee")
            fee_balance.free -= fee_dec

    def skew(self, asset: str) -> dict:
        """
        Calculate distribution skew for an asset across venues.

        Returns:
        {
            'asset': str,
            'total': Decimal,
            'venues': {
                'binance': {'amount': Decimal, 'pct': float, 'deviation_pct': float},
                'wallet':  {'amount': Decimal, 'pct': float, 'deviation_pct': float},
            },
            'max_deviation_pct': float,
            'needs_rebalance': bool,  # True if max_deviation > 30%
        }
        """
        asset_key = asset.upper()
        totals: dict[Venue, Decimal] = {}
        total_amount = Decimal("0")
        for venue in self._venues:
            amount = (
                self._balances.get(venue, {})
                .get(asset_key, Balance(venue, asset_key, Decimal("0"), Decimal("0")))
                .total
            )
            totals[venue] = amount
            total_amount += amount

        venues_result: dict[str, dict] = {}
        target_pct = 1.0 / len(self._venues)
        max_deviation_pct = 0.0

        for venue, amount in totals.items():
            if total_amount > 0:
                pct = float(amount / total_amount)
            else:
                pct = 0.0
            deviation_pct = (pct - target_pct) * 100.0
            max_deviation_pct = max(max_deviation_pct, abs(deviation_pct))
            venues_result[venue.value] = {
                "amount": amount,
                "pct": pct,
                "deviation_pct": deviation_pct,
            }

        return {
            "asset": asset_key,
            "total": total_amount,
            "venues": venues_result,
            "max_deviation_pct": max_deviation_pct,
            "needs_rebalance": max_deviation_pct > 30.0,
        }
