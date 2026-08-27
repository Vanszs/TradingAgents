"""
Backtesting module for TradingAgents -- Stock margin backtester.

Design goals:
- Spot long-only positions with static BUY/WNS decisions.
- Daily mark-to-market settlement at close.
- Cash-bounded entries and static risk exits.
- Snapshot-only historical data, next-session-open execution.
- Single-instrument flat data layout (data/TICKER/ohlcv.csv).

New PRD-compliant types (PositionSide, OrderType, ExtendedDecision, etc.)
are defined in position.py and re-exported here.
"""
from .agent_runner import TradingAgentsRunner
from .broker import SimulatedBroker
from .decision_schema import (
    Action,
    AgentConfig,
    AssetClass,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    InstrumentSpec,
    LeakageGuardConfig,
    MarginConfig,
    MarginEvent,
    MarketPoint,
    OpenClose,
    Order,
    OrderSide,
    OrderStatus,
    OutputConfig,
    ParsedDecision,
    PortfolioSnapshot,
    Position,
    Rating,
    SnapshotMetadata,
    Trade,
)
from .decision_store import DecisionStore
from .engine import BacktestEngine
from .margin import MarginAccount
from .margin_engine import (
    excess_margin,
    initial_margin,
    is_intraday_margin_breach,
    is_margin_call,
    leverage,
    maintenance_margin,
    margin_utilization,
    max_contracts_by_margin,
    notional_value,
)
from .markdown_parser import MarkdownDecisionParser
from .metrics import MetricsCalculator
from .order_generator import OrderGenerator
from .portfolio import Portfolio, PortfolioV2

# New PRD-compliant BacktestConfig and configs from position.py
# (use these for new code; old BacktestConfig from decision_schema is kept for compat)
from .position import (
    BacktestConfig as BacktestConfigV2,
)

# New PRD-compliant types from position.py
# (only types that don't conflict with legacy types above)
from .position import (
    DecisionMappingConfig,
    ExtendedDecision,
    Fill,
    MarketMode,
    OrderType,
    PositionIntent,
    PositionSide,
    RiskConfig,
)
from .position import (
    ExecutionConfig as ExecutionConfigV2,
)
from .position import (
    MarginConfig as MarginConfigV2,
)
from .reports import BacktestReportGenerator
from .risk import RiskEngine
from .snapshot_provider import DataSnapshot, SnapshotDataProvider
from .walk_forward_runner import WalkForwardBacktestRunner

__all__ = [
    # Entry points
    "BacktestEngine",
    "WalkForwardBacktestRunner",
    "SimulatedBroker",
    "SnapshotDataProvider",
    "TradingAgentsRunner",
    "MarkdownDecisionParser",
    "OrderGenerator",
    "Portfolio",
    "PortfolioV2",
    "BacktestReportGenerator",
    "MetricsCalculator",
    "DecisionStore",
    "MarginAccount",
    "RiskEngine",
    # Core configs (legacy)
    "BacktestConfig",
    "ExecutionConfig",
    "DataConfig",
    "AgentConfig",
    "LeakageGuardConfig",
    "OutputConfig",
    "MarginConfig",
    "AssetClass",
    # Instrument
    "InstrumentSpec",
    # Decisions and orders (legacy)
    "ParsedDecision",
    "Order",
    "Trade",
    "Position",
    "OpenClose",
    "OrderSide",
    "OrderStatus",
    "Action",
    "Rating",
    # Data
    "MarketPoint",
    "SnapshotMetadata",
    "DataSnapshot",
    "PortfolioSnapshot",
    "MarginEvent",
    # Margin helpers
    "notional_value",
    "initial_margin",
    "maintenance_margin",
    "max_contracts_by_margin",
    "excess_margin",
    "is_margin_call",
    "is_intraday_margin_breach",
    "leverage",
    "margin_utilization",
    # New PRD-compliant types
    "PositionSide",
    "OrderType",
    "MarketMode",
    "PositionIntent",
    "ExtendedDecision",
    "DecisionMappingConfig",
    "RiskConfig",
    "Fill",
    # New PRD-compliant configs (V2)
    "BacktestConfigV2",
    "ExecutionConfigV2",
    "MarginConfigV2",
]
