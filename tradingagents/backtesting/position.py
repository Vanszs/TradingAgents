"""
Position model for stock margin backtests.

Defines:
- PositionSide enum (LONG/SHORT/FLAT)
- Position dataclass (notional, margin, mark-to-market, PnL, stop/tp)
- Fill dataclass (execution record for a single leg)
- Order dataclass (intent to trade, PRD §7.2 OrderType)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(str, Enum):
    """PRD §7.2 — explicit order types replacing the old Action enum."""
    BUY_TO_OPEN = "BUY_TO_OPEN"
    BUY_TO_ADD = "BUY_TO_ADD"
    SELL_TO_REDUCE = "SELL_TO_REDUCE"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    SELL_TO_ADD = "SELL_TO_ADD"
    BUY_TO_REDUCE = "BUY_TO_REDUCE"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    REVERSE_TO_LONG = "REVERSE_TO_LONG"
    REVERSE_TO_SHORT = "REVERSE_TO_SHORT"
    NO_ORDER = "NO_ORDER"


class MarketMode(str, Enum):
    FUTURES_STYLE_SIMULATION = "FUTURES_STYLE_SIMULATION"
    SPOT_LONG_ONLY = "SPOT_LONG_ONLY"


class PositionIntent(str, Enum):
    OPEN = "open"
    INCREASE = "increase"
    HOLD = "hold"
    REDUCE = "reduce"
    CLOSE = "close"
    REVERSE = "reverse"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """PRD §13 execution defaults — percentage-based fees/slippage."""
    lot_size: int = 1
    contract_multiplier: float = 1.0
    buy_fee: float = 0.0015
    sell_fee: float = 0.0025
    slippage: float = 0.001
    spread_bps: float = 5.0
    financing_rate: float = 0.0001
    borrow_fee: float = 0.0002
    conservative_intraday_rule: bool = True
    initial_position_side: PositionSide = PositionSide.FLAT

    def __post_init__(self) -> None:
        if self.lot_size < 1:
            raise ValueError(f"lot_size must be >= 1, got {self.lot_size}")
        if self.contract_multiplier <= 0:
            raise ValueError(f"contract_multiplier must be > 0, got {self.contract_multiplier}")
        if self.buy_fee < 0:
            raise ValueError(f"buy_fee must be >= 0, got {self.buy_fee}")
        if self.sell_fee < 0:
            raise ValueError(f"sell_fee must be >= 0, got {self.sell_fee}")
        if self.slippage < 0:
            raise ValueError(f"slippage must be >= 0, got {self.slippage}")


@dataclass
class MarginConfig:
    """PRD §15 margin configuration."""
    initial_margin_pct: float = 0.50
    maintenance_margin_pct: float = 0.35
    max_leverage: float = 2.0
    margin_call_threshold: float = 0.40
    liquidation_equity_pct: float = 0.25
    max_entry_pct: float = 0.10  # max % of equity per single-day entry (avoid all-in)

    def __post_init__(self) -> None:
        if self.initial_margin_pct <= 0:
            raise ValueError(f"initial_margin_pct must be > 0, got {self.initial_margin_pct}")
        if self.maintenance_margin_pct <= 0:
            raise ValueError(f"maintenance_margin_pct must be > 0, got {self.maintenance_margin_pct}")
        if self.maintenance_margin_pct > self.initial_margin_pct:
            raise ValueError(
                "maintenance_margin_pct must be <= initial_margin_pct. "
                f"Got maintenance={self.maintenance_margin_pct}, "
                f"initial={self.initial_margin_pct}"
            )
        if self.max_leverage < 1.0:
            raise ValueError(f"max_leverage must be >= 1.0, got {self.max_leverage}")
        if self.max_entry_pct <= 0 or self.max_entry_pct > 1.0:
            raise ValueError(
                f"max_entry_pct must be in (0, 1], got {self.max_entry_pct}"
            )


@dataclass
class RiskConfig:
    """PRD §11 risk parameters."""
    default_stop_pct: float = 0.08
    default_take_profit_pct: float = 0.20
    trailing_stop_pct: float = 0.05
    max_position_pct: float = 0.30
    max_intraday_loss_pct: float = 0.05
    # ATR-based stop/take_profit
    use_atr_based_stops: bool = True
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0


@dataclass
class DecisionMappingConfig:
    """Decision mapping configuration.

    Two ``mode`` values are supported:

    * ``strict_5tier`` (recommended) — preserves the full 5-tier agent
      rating and applies the spec table. Conservative: never auto-reverses
      LONG<->SHORT; respects the spec literal mapping.
    * ``legacy_prd`` — original 3-bucket behaviour with the old
      ``conservative`` / ``aggressive`` sub-modes. ``conservative`` is the
      default for backward compatibility.
    """
    mode: str = "strict_5tier"
    legacy_submode: str = "conservative"  # only used when mode == "legacy_prd"
    allow_reverse_on_buy_sell: bool = False
    allow_short_on_underweight: bool = False
    allow_short_on_sell: bool = True
    allow_explicit_reverse: bool = False  # strict_5tier: never reverse unless True
    overweight_size_multiplier: float = 0.5
    underweight_size_multiplier: float = 0.5
    default_reduce_pct: float = 0.5
    base_allocation_pct: float = 0.20  # Deprecated: unused, use initial_entry_pct instead
    # Entry config (FLAT → Buy/Sell)
    initial_entry_pct: float = 0.30
    # Pyramiding config (position exists + rating aligned)
    pyramid_pct: float = 0.10
    # Gradual exit config (position exists + rating opposite)
    reduce_step_pct: float = 0.10

    def __post_init__(self) -> None:
        valid_modes = ("strict_5tier", "legacy_prd", "conservative", "aggressive")
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got {self.mode!r}"
            )
        if self.legacy_submode not in ("conservative", "aggressive"):
            raise ValueError(
                f"legacy_submode must be 'conservative' or 'aggressive', got {self.legacy_submode!r}"
            )


@dataclass
class DataConfig:
    """Data source configuration."""
    root: str = "data"
    layout: str = "flat"
    news_provider: str = "snapshot"
    fundamentals_provider: str = "snapshot"
    benchmark: str = "SPY"


@dataclass
class AgentConfig:
    """Agent runtime configuration."""
    provider: str = ""
    model: str = ""
    temperature: float = 0.2
    memory_enabled: bool = False
    web_search_enabled: bool = False
    news_provider: str = "snapshot"
    max_thesis_chars: int = 2000
    deterministic_seed: int = 42
    backtest_mode: bool = True
    run_frequency: str = "daily"
    report_language: str = "English"
    cache_reports: bool = True


@dataclass
class LeakageGuardConfig:
    """Leakage guard configuration."""
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
    """Output file configuration."""
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
    trade_log: str = "trade_log.csv"
    decision_log: str = "decision_log.csv"
    position_log: str = "position_log.csv"
    margin_log: str = "margin_log.csv"
    equity_curve: str = "equity_curve.csv"
    leakage_audit: str = "leakage_audit.json"
    summary: str = "summary.json"


# ---------------------------------------------------------------------------
# Instrument specification
# ---------------------------------------------------------------------------

@dataclass
class InstrumentSpec:
    """
    Single-instrument specification for stocks.
    For stocks: multiplier=1, tick_size=0.01.
    """
    ticker: str
    multiplier: float = 1.0
    tick_size: float = 0.01
    currency: str = "USD"
    lot_size: int = 100

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
        return cls(ticker=ticker, multiplier=1.0, tick_size=tick_size, lot_size=100)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    PRD §7.3 — position state with margin, mark, PnL, stop/tp.

    Convention: quantity > 0 = LONG, quantity < 0 = SHORT, quantity == 0 = FLAT.
    """
    ticker: str
    quantity: int = 0
    avg_entry_price: float = 0.0
    multiplier: float = 1.0
    mark_price: float = 0.0
    notional_value: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 0.0
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: Optional[str] = None
    last_updated_at: Optional[str] = None
    status: str = "open"

    def is_long(self) -> bool:
        return self.quantity > 0

    def is_short(self) -> bool:
        return self.quantity < 0

    def is_flat(self) -> bool:
        return self.quantity == 0

    def abs_qty(self) -> int:
        return abs(self.quantity)

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        if self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    def notional(self, price: float) -> float:
        return abs(self.quantity) * float(price) * self.multiplier

    def unrealized_pnl_calc(self, mark_price: float) -> float:
        """Recalculate unrealized PnL from current mark."""
        if self.quantity == 0:
            return 0.0
        return (mark_price - self.avg_entry_price) * self.quantity * self.multiplier

    def mark_to_market(self, mark_price: float) -> None:
        """Update mark price and recalculate derived fields."""
        self.mark_price = float(mark_price)
        self.notional_value = self.notional(mark_price)
        self.unrealized_pnl = self.unrealized_pnl_calc(mark_price)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = self.side.value
        return data


# ---------------------------------------------------------------------------
# Fill (execution record)
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """Single execution fill record — one leg of a (possibly two-leg) order."""
    fill_id: str
    order_id: str
    decision_id: str
    date: str
    ticker: str
    side: str
    quantity: int
    price: float
    fee: float
    slippage_amount: float
    order_type: str
    open_close: str
    realized_pnl_delta: float = 0.0
    margin_delta: float = 0.0
    mark_price: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Order (PRD §7.2)
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """
    PRD §7.2 — order with explicit OrderType.

    For reverse orders, the broker executes two fills (close leg + open leg).
    """
    order_id: str
    decision_id: str
    ticker: str
    order_type: OrderType
    quantity: int
    execution_date: str
    price: float = 0.0
    status: str = "PENDING"
    rejection_reason: Optional[str] = None
    reason: str = ""
    created_at: Optional[str] = None
    is_reverse: bool = False

    @property
    def side(self) -> str:
        """Backward-compat: derive side from order_type."""
        buy_types = {
            OrderType.BUY_TO_OPEN, OrderType.BUY_TO_ADD,
            OrderType.BUY_TO_REDUCE, OrderType.BUY_TO_CLOSE,
        }
        return "BUY" if self.order_type in buy_types else "SELL"

    @property
    def open_close(self) -> str:
        """Backward-compat: derive open/close from order_type."""
        close_types = {
            OrderType.SELL_TO_CLOSE, OrderType.BUY_TO_CLOSE,
            OrderType.SELL_TO_REDUCE, OrderType.BUY_TO_REDUCE,
        }
        return "CLOSE" if self.order_type in close_types else "OPEN"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["order_type"] = self.order_type.value
        data["status"] = self.status
        return data


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Top-level backtest configuration matching PRD §22."""
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
    asset_class: str = "stock"

    # Window for the daily review snapshot. None = full history.
    # Allowed values: None, 5, 10, 20, 40, 60, 80, 100, 120, 240.
    lookback_days: Optional[int] = 240

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    decision_mapping: DecisionMappingConfig = field(default_factory=DecisionMappingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    leakage_guard: LeakageGuardConfig = field(default_factory=LeakageGuardConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    trigger: Any = None  # TriggerConfig; resolved by the runner to avoid circular import

    def validate(self) -> None:
        """PRD §22 — validate configuration at startup."""
        if not self.backtest_mode:
            raise ValueError("BacktestConfig.backtest_mode must be True for historical simulation.")
        if self.asset_class != "stock":
            raise ValueError(f"asset_class must be 'stock', got {self.asset_class!r}")
        if not self.ticker:
            raise ValueError("BacktestConfig.ticker is required.")
        if not self.start_date or not self.end_date:
            raise ValueError("BacktestConfig.start_date and end_date are required.")
        if self.data.news_provider != "snapshot":
            raise ValueError("Historical backtest must use snapshot data provider.")
        if self.agent.memory_enabled:
            raise ValueError("Backtest requires memory_enabled=False to avoid leakage.")
        if self.agent.web_search_enabled:
            raise ValueError("Agent web search must be disabled during backtest.")
        if self.agent.news_provider != "snapshot":
            raise ValueError("Agent news_provider must be 'snapshot' in backtest.")
        if self.margin.max_leverage < 1.0:
            raise ValueError(f"max_leverage must be >= 1.0, got {self.margin.max_leverage}")

        valid_modes = ("strict_5tier", "legacy_prd", "conservative", "aggressive")
        if self.decision_mapping.mode not in valid_modes:
            raise ValueError(
                f"decision_mapping.mode must be one of {valid_modes}, "
                f"got {self.decision_mapping.mode!r}"
            )

        # Lazy import to avoid a circular import at module load time.
        from .data_window import ALLOWED_LOOKBACKS
        if self.lookback_days is not None and self.lookback_days not in ALLOWED_LOOKBACKS:
            raise ValueError(
                f"lookback_days must be one of {ALLOWED_LOOKBACKS}, "
                f"got {self.lookback_days!r}"
            )


# ---------------------------------------------------------------------------
# ParsedDecision (parser output, no action inference per PRD §9)
# ---------------------------------------------------------------------------

@dataclass
class ParsedDecision:
    """
    PRD §9 — parser output fields only.

    The parser extracts raw decision fields. The DecisionStateManager
    (PRD §10.5) maps these to position_intent and futures_action.
    """
    decision_id: str
    ticker: str
    trade_date: str
    agent_rating: str = ""
    report_generated_at: str = ""
    last_data_date: str = ""
    decision_valid_from: str = ""
    normalized_rating: str = ""
    allocation_pct: Optional[float] = None
    reduce_pct: Optional[float] = None
    leverage: float = 1.0
    confidence: Optional[float] = None
    short_allowed: bool = False
    allow_new_position: bool = True
    market_mode: str = "FUTURES_STYLE_SIMULATION"
    allowed_position_sides: str = "LONG,SHORT"
    position_intent: str = "hold"
    current_position_side: str = "FLAT"
    target_position_side: str = "FLAT"
    futures_action: str = "NO_ORDER"
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon_days: Optional[int] = None
    time_horizon_label: Optional[str] = None
    thesis_summary: str = ""
    source_report_path: Optional[str] = None
    raw_text_excerpt: Optional[str] = None
    valid: bool = True
    invalid_reason: Optional[str] = None
    next_review_date: Optional[str] = None

    # Trigger-Based Execution Agent fields. Filled at runtime by the
    # WalkForwardRunner; never set by the markdown parser.
    prev_rating: Optional[str] = None
    rating_changed: bool = False
    triggered: bool = True
    trigger_reasons: list[str] = field(default_factory=list)
    trigger_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def invalid(
        cls,
        ticker: str,
        trade_date: str,
        reason: str,
        source_report_path: Optional[str] = None,
    ) -> ExtendedDecision:
        return cls(
            decision_id=f"{ticker}-{trade_date}-INVALID",
            ticker=ticker,
            trade_date=trade_date,
            report_generated_at=f"{trade_date} 16:30:00",
            last_data_date=trade_date,
            decision_valid_from=trade_date,
            agent_rating="",
            valid=False,
            invalid_reason=reason,
            source_report_path=source_report_path,
        )


# ExtendedDecision is an alias for ParsedDecision (PRD §9.1)
ExtendedDecision = ParsedDecision


# ---------------------------------------------------------------------------
# SnapshotMetadata / MarketPoint / PortfolioSnapshot / MarginEvent
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trade (PRD §19.2)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """
    PRD §19.2 — trade log entry.
    Each fill produces one Trade record. Reverse orders produce two.
    """
    date: str
    ticker: str
    side: str
    quantity: int
    price: float
    fee: float
    slippage_amount: float
    realized_pnl_delta: float
    order_type: str
    reason: str
    decision_id: str
    order_id: str
    fill_id: str = ""
    is_reverse: bool = False
    mark_price: float = 0.0
    multiplier: float = 1.0
    notional: float = 0.0
    margin_delta: float = 0.0
    open_close: str = "OPEN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def ensure_dir(path: Any) -> Any:
    from pathlib import Path as _Path
    p = _Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
