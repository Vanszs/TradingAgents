"""
Legacy decision schema for the stock margin backtester.

Compatibility shim. The canonical config/type family (BacktestConfig,
ExecutionConfig, MarginConfig, InstrumentSpec, DataConfig, AgentConfig,
LeakageGuardConfig, OutputConfig, SnapshotMetadata, MarketPoint,
PortfolioSnapshot, MarginEvent, FillRule, parse_date, ensure_dir) lives in
``position.py`` and is re-exported below — one definition each. Only the
genuinely legacy-only remnants remain here: the old-schema enums (AssetClass,
Rating, Action, OrderSide, OpenClose, OrderStatus) used by the
``_generate_from_old_schema`` path, markdown_parser, agent_runner and
broker/portfolio, plus the first-generation record shapes ParsedDecision /
Order / Trade / Position (see ponytail notes on each class).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from .position import (  # noqa: F401  (re-exports)
    AgentConfig,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    FillRule,
    InstrumentSpec,
    LeakageGuardConfig,
    MarginConfig,
    MarginEvent,
    MarketPoint,
    OutputConfig,
    PortfolioSnapshot,
    SnapshotMetadata,
    ensure_dir,
    parse_date,
)

# ---------------------------------------------------------------------------
# Legacy-only enums (old decision schema)
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    STOCK = "stock"


class Rating(str, Enum):
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"
    WNS = "WNS"
    UNDERWEIGHT = "Underweight"
    OVERWEIGHT = "Overweight"
    INVALID = "Invalid"


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    OPEN_SHORT = "OPEN_SHORT"
    COVER_SHORT = "COVER_SHORT"
    ADD = "ADD"
    REDUCE = "REDUCE"
    HOLD = "HOLD"
    SELL_ALL = "SELL_ALL"
    INVALID = "INVALID"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OpenClose(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    AUTO = "AUTO"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Legacy-only record shapes
# ---------------------------------------------------------------------------

@dataclass
class ParsedDecision:
    # ponytail: first-generation schema (.action/.rating enums, trim zones);
    # position.ParsedDecision is the PRD §9 rewrite with different fields, so
    # unifying would change parser output and the old-schema order path.
    decision_id: str
    ticker: str
    trade_date: str
    report_generated_at: str
    last_data_date: str
    decision_valid_from: str
    rating: Rating
    action: Action

    confidence: Optional[float] = None
    initial_trim_pct: Optional[float] = None
    second_trim_pct: Optional[float] = None
    second_trim_zone_low: Optional[float] = None
    second_trim_zone_high: Optional[float] = None
    stop_type: Optional[str] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon_days: Optional[int] = None
    target_position_pct: Optional[float] = None
    allow_new_position: bool = True
    reduce_only: bool = False
    source_report_path: Optional[str] = None
    raw_text_excerpt: Optional[str] = None
    valid: bool = True
    invalid_reason: Optional[str] = None
    next_review_date: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rating"] = self.rating.value
        data["action"] = self.action.value
        return data

    @classmethod
    def invalid(
        cls,
        ticker: str,
        trade_date: str,
        reason: str,
        source_report_path: Optional[str] = None,
    ) -> ParsedDecision:
        return cls(
            decision_id=f"{ticker}-{trade_date}-INVALID",
            ticker=ticker,
            trade_date=trade_date,
            report_generated_at=f"{trade_date} 16:30:00",
            last_data_date=trade_date,
            decision_valid_from=trade_date,
            rating=Rating.INVALID,
            action=Action.INVALID,
            allow_new_position=False,
            source_report_path=source_report_path,
            valid=False,
            invalid_reason=reason,
        )


@dataclass
class Order:
    # ponytail: legacy side/open_close enum order shape; position.Order is the
    # PRD §7.2 OrderType rewrite consumed by the new broker path.
    order_id: str
    decision_id: str
    ticker: str
    side: OrderSide
    quantity: int
    execution_date: str
    reason: str
    open_close: OpenClose = OpenClose.AUTO
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = self.side.value
        data["open_close"] = self.open_close.value
        data["status"] = self.status.value
        return data


@dataclass
class Trade:
    # ponytail: legacy fill record (gross_amount/net_amount/tick_size/
    # slippage_ticks) built by broker/portfolio; position.Trade lacks these
    # fields, so collapsing would change every trade-log row.
    date: str
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    gross_amount: float
    fee: float
    net_amount: float
    multiplier: float
    notional: float
    tick_size: float
    slippage_ticks: int
    realized_pnl_delta: float
    margin_delta: float
    open_close: OpenClose
    reason: str
    decision_id: str
    order_id: str
    mark_price: Optional[float] = None
    order_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = self.side.value if hasattr(self.side, 'value') else str(self.side)
        data["open_close"] = self.open_close.value if hasattr(self.open_close, 'value') else str(self.open_close)
        return data


@dataclass
class Position:
    """Signed-position state. quantity positive = long, negative = short."""
    # ponytail: legacy simple position (avg_price/tick_size/lifetime_fees)
    # backing Portfolio; position.Position is the PRD §7.3 state class behind
    # PortfolioV2 — merging would change portfolio accounting fields.
    ticker: str
    quantity: int = 0
    avg_price: float = 0.0
    multiplier: float = 1.0
    tick_size: float = 0.01
    realized_pnl: float = 0.0
    lifetime_fees: float = 0.0

    def is_long(self) -> bool:
        return self.quantity > 0

    def is_short(self) -> bool:
        return self.quantity < 0

    def is_flat(self) -> bool:
        return self.quantity == 0

    def abs_qty(self) -> int:
        return abs(self.quantity)

    def notional(self, price: float) -> float:
        return abs(self.quantity) * float(price) * self.multiplier

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return (mark_price - self.avg_price) * self.quantity * self.multiplier

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
