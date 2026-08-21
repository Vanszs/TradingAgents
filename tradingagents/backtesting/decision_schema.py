"""
Core data structures for the stock margin backtester.

DEPRECATED TYPES: The following classes are LEGACY definitions kept for
backward compatibility. New code should import from ``position.py`` instead:

- ``BacktestConfig``      → use ``position.BacktestConfig``
- ``ExecutionConfig``     → use ``position.ExecutionConfig``
- ``MarginConfig``        → use ``position.MarginConfig``
- ``AgentConfig``         → use ``position.AgentConfig``
- ``LeakageGuardConfig``  → use ``position.LeakageGuardConfig``
- ``OutputConfig``        → use ``position.OutputConfig``
- ``DataConfig``          → use ``position.DataConfig``
- ``Order``               → use ``position.Order``
- ``Trade``               → use ``position.Trade``
- ``InstrumentSpec``      → use ``position.InstrumentSpec``
- ``MarketPoint``         → use ``position.MarketPoint``
- ``SnapshotMetadata``    → use ``position.SnapshotMetadata``
- ``PortfolioSnapshot``   → use ``position.PortfolioSnapshot``
- ``MarginEvent``         → use ``position.MarginEvent``

The ``__init__.py`` re-exports both sets with V2 aliases for the new types.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union


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


class FillRule(str, Enum):
    NEXT_SESSION_OPEN = "next_session_open"
    NEXT_BAR_OPEN = "next_bar_open"
    STOP_LEVEL = "stop_level"


@dataclass
class InstrumentSpec:
    """
    Single-instrument specification for stocks.
    For stocks: multiplier=1, tick_size=0.01 (auto-detected from OHLCV).
    """
    ticker: str
    multiplier: float = 1.0
    tick_size: float = 0.01
    currency: str = "USD"

    @classmethod
    def auto_from_ohlcv(cls, ticker: str, ohlcv_df: Any = None) -> InstrumentSpec:
        """Auto-generate spec from OHLCV data precision."""
        tick_size = 0.01
        if ohlcv_df is not None and not ohlcv_df.empty:
            closes = ohlcv_df["close"].dropna()
            if not closes.empty:
                decimals = closes.apply(
                    lambda x: len(str(x).split(".")[-1]) if "." in str(x) else 0
                ).max()
                tick_size = float(10) ** (-decimals) if decimals > 0 else 0.01
        return cls(ticker=ticker, multiplier=1.0, tick_size=tick_size)


@dataclass
class MarginConfig:
    """Margin configuration."""
    initial_margin_pct: float = 0.50
    maintenance_margin_pct: float = 0.30
    margin_call_buffer_pct: float = 0.0
    borrow_fee_apr: float = 0.0
    auto_liquidate_on_breach: bool = True
    max_leverage: float = 2.0
    max_entry_pct: float = 0.10

    def __post_init__(self) -> None:
        if self.initial_margin_pct <= 0:
            raise ValueError(
                f"initial_margin_pct must be > 0, got {self.initial_margin_pct}"
            )
        if self.maintenance_margin_pct <= 0:
            raise ValueError(
                f"maintenance_margin_pct must be > 0, got {self.maintenance_margin_pct}"
            )
        if self.maintenance_margin_pct > self.initial_margin_pct:
            raise ValueError(
                "maintenance_margin_pct must be <= initial_margin_pct. "
                f"Got maintenance={self.maintenance_margin_pct}, "
                f"initial={self.initial_margin_pct}"
            )
        if self.margin_call_buffer_pct < 0:
            raise ValueError(
                f"margin_call_buffer_pct must be >= 0, got {self.margin_call_buffer_pct}"
            )
        if self.borrow_fee_apr < 0:
            raise ValueError(f"borrow_fee_apr must be >= 0, got {self.borrow_fee_apr}")


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    rule: str = "next_session_open"
    fill_at: str = "next_session_open"
    tick_slippage: int = 1
    fee_per_contract: float = 0.0
    intraday_margin_check: bool = True
    liquidation_fill_at: str = "next_session_open"
    liquidation_extra_slippage_ticks: int = 1
    lot_size: int = 100
    contract_multiplier: float = 1.0
    buy_fee: float = 0.0015
    sell_fee: float = 0.0025
    slippage: float = 0.001
    spread_bps: float = 0.0
    conservative_intraday_rule: bool = True

    def __post_init__(self) -> None:
        try:
            FillRule(self.fill_at)
        except ValueError as exc:
            raise ValueError(
                f"fill_at must be one of {[f.value for f in FillRule]}, got {self.fill_at!r}"
            ) from exc
        if self.tick_slippage < 0:
            raise ValueError(f"tick_slippage must be >= 0, got {self.tick_slippage}")
        if self.fee_per_contract < 0:
            raise ValueError(f"fee_per_contract must be >= 0, got {self.fee_per_contract}")


@dataclass
class DataConfig:
    provider: str = "snapshot"
    data_root: str = "data"
    snapshot_root: str = "snapshots"
    disable_live_news: bool = True
    disable_live_web_search: bool = True
    disable_live_fundamentals: bool = True
    require_fundamental_available_date: bool = True
    fetch_from_api: bool = False
    api_cache_dir: str = "api_cache"


@dataclass
class AgentConfig:
    run_frequency: str = "daily"
    report_language: str = "English"
    cache_reports: bool = True
    memory_enabled: bool = False
    backtest_mode: bool = True
    web_search_enabled: bool = False
    news_provider: str = "snapshot"


@dataclass
class LeakageGuardConfig:
    validate_cutoff: bool = True
    require_snapshot_metadata: bool = True
    require_next_bar_execution: bool = True
    fail_on_future_data: bool = True
    news_cutoff_strategy: str = "previous_day"
    sentiment_cutoff_strategy: str = "previous_day"
    broker_activity_cutoff_strategy: str = "previous_day"
    fundamental_buffer_days: int = 3


@dataclass
class OutputConfig:
    output_root: str = "backtest_results"
    reports_root: str = "reports"
    save_reports: bool = True
    save_decisions: bool = True
    save_trade_log: bool = True
    save_equity_curve: bool = True
    save_leakage_audit: bool = True
    save_account_state: bool = True
    save_margin_events: bool = True
    save_position_log: bool = True
    save_margin_log: bool = True


@dataclass
class BacktestConfig:
    asset_class: str = "stock"
    ticker: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_cash: float = 100_000.0
    initial_position_qty: int = 0
    initial_position_price: float = 0.0
    initial_report_path: Optional[str] = None
    benchmark_ticker: Optional[str] = None
    mode: str = "walk_forward_agent"
    backtest_mode: bool = True

    max_risk_per_trade_pct: float = 0.01
    default_reduce_pct: float = 0.5

    # Per-run lookback window in trading days. None = full history.
    # Allowed: None, 5, 10, 20, 40, 60, 80, 100, 120, 240.
    lookback_days: Optional[int] = 240

    # Decision mapping & trigger evaluator (filled from yaml).
    # Imported lazily to avoid a circular import at module load time.
    decision_mapping: Any = None  # DecisionMappingConfig
    trigger: Any = None           # TriggerConfig

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    leakage_guard: LeakageGuardConfig = field(default_factory=LeakageGuardConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    risk: Any = None  # RiskConfig; resolved in __post_init__ to avoid circular import

    def __post_init__(self) -> None:
        # Default decision_mapping to a fresh DecisionMappingConfig() if None.
        if self.decision_mapping is None:
            from .position import DecisionMappingConfig
            self.decision_mapping = DecisionMappingConfig()
        # Default risk to a fresh RiskConfig() if None.
        if self.risk is None:
            from .position import RiskConfig
            self.risk = RiskConfig()

    def validate(self) -> None:
        if not self.backtest_mode:
            raise ValueError(
                "BacktestConfig.backtest_mode must be True for historical simulation."
            )
        if AssetClass(self.asset_class) != AssetClass.STOCK:
            raise ValueError(
                f"asset_class must be 'stock', got {self.asset_class!r}"
            )
        if not self.ticker:
            raise ValueError("BacktestConfig.ticker is required.")
        if not self.start_date or not self.end_date:
            raise ValueError("BacktestConfig.start_date and end_date are required.")
        if self.data.provider != "snapshot":
            raise ValueError("Historical backtest must use snapshot data provider.")
        if not self.data.disable_live_news:
            raise ValueError("Live news must be disabled in backtest mode.")
        if not self.data.disable_live_web_search:
            raise ValueError("Live web search must be disabled in backtest mode.")
        if not self.data.disable_live_fundamentals:
            raise ValueError("Live fundamentals must be disabled in backtest mode.")
        if self.agent.memory_enabled:
            raise ValueError("MVP backtest requires memory_enabled=False to avoid leakage.")
        if self.agent.web_search_enabled:
            raise ValueError("Agent web search must be disabled during backtest.")
        if self.execution.fill_at == FillRule.STOP_LEVEL.value:
            raise ValueError(
                "execution.fill_at == 'stop_level' is reserved for stop/target orders."
            )

        # Lookback window check (lazy import to avoid circular import).
        if self.lookback_days is not None:
            from .data_window import ALLOWED_LOOKBACKS
            if self.lookback_days not in ALLOWED_LOOKBACKS:
                raise ValueError(
                    f"lookback_days must be one of {ALLOWED_LOOKBACKS}, "
                    f"got {self.lookback_days!r}"
                )

        # Decision mapping mode check.
        valid_modes = ("strict_5tier", "legacy_prd", "conservative", "aggressive")
        if self.decision_mapping.mode not in valid_modes:
            raise ValueError(
                f"decision_mapping.mode must be one of {valid_modes}, "
                f"got {self.decision_mapping.mode!r}"
            )


@dataclass
class SnapshotMetadata:
    ticker: str
    trade_date: str
    snapshot_created_at: str
    provider_mode: str = "snapshot"
    max_ohlcv_date: Optional[str] = None
    max_news_time: Optional[str] = None
    max_fundamental_available_date: Optional[str] = None
    max_sentiment_time: Optional[str] = None
    max_broker_activity_time: Optional[str] = None
    path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketPoint:
    """Generic OHLCV bar."""
    date: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class ParsedDecision:
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
    order_id: str
    decision_id: str
    ticker: str
    side: OrderSide
    quantity: int
    execution_date: str
    reason: str
    open_close: OpenClose = OpenClose.AUTO
    reduce_only: bool = False
    reference_price: Optional[float] = None
    tick_slippage: Optional[int] = None
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


@dataclass
class PortfolioSnapshot:
    date: str
    cash: float
    position_qty: int
    position_avg_price: float
    position_value: float
    mark_price: float
    total_equity: float
    drawdown: float
    margin_used: float
    margin_available: float
    excess_margin: float
    maintenance_margin_required: float
    initial_margin_required: float
    leverage: float
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    lifetime_realized_pnl: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarginEvent:
    date: str
    kind: str
    deficit: float
    mark_price: float
    maintenance_required: float
    account_equity: float
    action: str
    liquidation_price: Optional[float] = None
    liquidation_quantity: Optional[int] = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_date(value: Union[str, date, datetime]) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
