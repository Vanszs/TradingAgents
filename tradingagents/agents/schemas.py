"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class EntryMode(str, Enum):
    """Execution intent for an actionable signal."""

    ASSUMED_AI_ENTRY = "ASSUMED_AI_ENTRY"
    T1_OPEN = "T1_OPEN"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )

    @field_validator("recommendation", mode="before")
    @classmethod
    def _normalize_recommendation(cls, v: Any) -> Any:
        if isinstance(v, PortfolioRating):
            return v
        if isinstance(v, str):
            clean = v.strip().title()
            for member in PortfolioRating:
                if member.value.lower() == clean.lower() or member.name.lower() == clean.lower():
                    return member
        return v


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the Research Manager's "
            "investment plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description=(
            "Entry price target in quote currency. "
            "Provide a specific number. "
            "Omit or set to null ONLY if no clear entry level."
        ),
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description=(
            "Stop-loss price in quote currency. "
            "Provide a specific number. "
            "Omit or set to null ONLY if no clear stop level."
        ),
    )
    take_profit: Optional[float] = Field(
        default=None,
        description=(
            "Take-profit target price in quote currency. "
            "Provide a specific number. "
            "Omit or set to null ONLY if no clear target."
        ),
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, v: Any) -> Any:
        if isinstance(v, TraderAction):
            return v
        if isinstance(v, str):
            clean = v.strip().title()
            for member in TraderAction:
                if member.value.lower() == clean.lower() or member.name.lower() == clean.lower():
                    return member
        return v

    @field_validator("entry_price", "stop_loss", "take_profit", mode="before")
    @classmethod
    def _coerce_none_strings(cls, v):
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("none", "null", "n/a", "", "undefined"):
                return None
            import re
            num_clean = re.sub(r"[^\d.-]", "", v.strip())
            if not num_clean:
                return None
            try:
                v = float(num_clean)
            except ValueError:
                return None
        if isinstance(v, (int, float)):
            import math
            if not math.isfinite(v) or v <= 0:
                return None
            return float(v)
        return v

    @model_validator(mode="after")
    def _validate_risk_reward_expectancy(self):
        if self.action in (TraderAction.BUY, TraderAction.HOLD):
            if self.entry_price and self.stop_loss and self.take_profit:
                risk = self.entry_price - self.stop_loss
                reward = self.take_profit - self.entry_price
                if risk > 0 and reward > 0:
                    rr = reward / risk
                    if rr < 1.95:
                        logger.warning("Trader proposal R:R ratio (%.2f) is below standard 2.0 desk threshold", rr)
        elif self.action == TraderAction.SELL:
            if self.entry_price and self.stop_loss and self.take_profit:
                risk = self.stop_loss - self.entry_price
                reward = self.entry_price - self.take_profit
                if risk > 0 and reward > 0:
                    rr = reward / risk
                    if rr < 1.95:
                        logger.warning("Trader proposal R:R ratio (%.2f) is below standard 2.0 desk threshold", rr)
        return self


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.take_profit is not None:
        parts.extend(["", f"**Take Profit**: {proposal.take_profit}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Numeric stop-loss price. Required for Buy, Overweight, Underweight, and Sell.",
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Numeric take-profit price. Required for Buy, Overweight, Underweight, and Sell.",
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Legacy alias for take-profit price.",
    )
    planned_entry_price: Optional[float] = Field(
        default=None,
        description="Numeric planned limit accumulation entry price if staging conditional order on support.",
    )
    time_horizon_days: int = Field(
        ge=1,
        le=252,
        description=(
            "Numeric target evaluation horizon in trading days; the agent must choose it. "
            "When stating a range such as 6-12 months, use the upper bound (252 days)."
        ),
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Numeric conviction score from 0.0 to 1.0.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Legacy display field for the recommended holding period.",
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description="Optional review date YYYY-MM-DD. If omitted, calculated deterministically in Python.",
    )

    @model_validator(mode="before")
    @classmethod
    def _use_upper_bound_for_horizon_range(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        label = data.get("time_horizon")
        if not isinstance(label, str):
            return data
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*"
            r"(day|days|week|weeks|month|months|year|years|hari|minggu|bulan|tahun)",
            label,
            re.IGNORECASE,
        )
        if not match:
            return data
        upper = float(match.group(2))
        unit = match.group(3).lower()
        multiplier = {
            "day": 1, "days": 1,
            "week": 5, "weeks": 5,
            "month": 21, "months": 21,
            "year": 252, "years": 252,
            "hari": 1, "minggu": 5, "bulan": 21, "tahun": 252,
        }[unit]
        normalized = dict(data)
        clamped_days = max(1, min(252, round(upper * multiplier)))
        normalized["time_horizon_days"] = clamped_days
        return normalized

    @field_validator("rating", mode="before")
    @classmethod
    def _normalize_rating(cls, v: Any) -> Any:
        if isinstance(v, PortfolioRating):
            return v
        if isinstance(v, str):
            clean = v.strip().title()
            for member in PortfolioRating:
                if member.value.lower() == clean.lower() or member.name.lower() == clean.lower():
                    return member
        return v

    @field_validator("take_profit", "price_target", "stop_loss", "planned_entry_price", mode="before")
    @classmethod
    def _coerce_optional_prices(cls, v):
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("none", "null", "n/a", "", "undefined"):
                return None
            num_clean = re.sub(r"[^\d.-]", "", v.strip())
            if not num_clean:
                return None
            try:
                v = float(num_clean)
            except ValueError:
                return None
        if isinstance(v, (int, float)):
            import math
            if not math.isfinite(v) or v <= 0:
                return None
            return float(v)
        return v

    @field_validator("next_review_date", mode="before")
    @classmethod
    def _validate_next_review_date(cls, v):
        if v is None or v == "" or str(v).lower() in ("none", "null", "n/a"):
            return None
        if not isinstance(v, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v.strip()):
            return None
        try:
            datetime.strptime(v.strip(), "%Y-%m-%d")
            return v.strip()
        except ValueError:
            return None


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {decision.stop_loss}"])
    take_profit = decision.take_profit if decision.take_profit is not None else decision.price_target
    if take_profit is not None:
        parts.extend(["", f"**Price Target**: {take_profit}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    parts.extend(["", f"**Time Horizon Days**: {decision.time_horizon_days}"])
    parts.extend(["", f"**Next Review Date**: {decision.next_review_date}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so that downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release.

    The ``narrative`` field preserves the rich source-by-source analysis that
    the analyst already produces; ``render_sentiment_report`` formats the
    structured fields as a header and appends the narrative, keeping the
    saved report content backwards-compatible.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "As a guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "These are prompt-level guidelines; only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points. Use 'medium' when data is present but sparse. "
            "Use 'high' when most of the six sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts) across news, StockTwits, Reddit, "
            "Bluesky, Mastodon, and the Fear & Greed index; "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence."
        ),
    )

    @field_validator("overall_band", mode="before")
    @classmethod
    def _normalize_overall_band(cls, v: Any) -> Any:
        if isinstance(v, SentimentBand):
            return v
        if isinstance(v, str):
            clean = v.strip().replace("_", " ").title()
            for member in SentimentBand:
                if member.value.lower() == clean.lower() or member.name.lower() == clean.lower():
                    return member
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: Any) -> Any:
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("low", "medium", "high"):
                return clean
        return v

    @field_validator("overall_score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> Any:
        if isinstance(v, str):
            num_match = re.search(r"(\d+(?:\.\d+)?)", v)
            if num_match:
                return float(num_match.group(1))
        return v


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report file is both human-readable and
    machine-parseable without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Signal Contract (Single-Shot Horizon Backtest Contract)
# ---------------------------------------------------------------------------


class SignalContract(BaseModel):
    """Validated identity and execution fields for a single-shot signal."""
    ticker: str = Field(min_length=1)
    signal_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    rating: PortfolioRating
    action: Literal["BUY", "SELL", "HOLD"]

    @field_validator("rating", mode="before")
    @classmethod
    def _normalize_rating(cls, v: Any) -> Any:
        if isinstance(v, PortfolioRating):
            return v
        if isinstance(v, str):
            clean = v.strip().title()
            for member in PortfolioRating:
                if member.value.lower() == clean.lower() or member.name.lower() == clean.lower():
                    return member
        return v
    entry_mode: Optional[EntryMode] = None
    planned_entry_price: Optional[float] = Field(
        default=None,
        description="Agent-planned entry level used as the assumed fill in ASSUMED_AI_ENTRY mode.",
    )
    signal_timestamp: Optional[str] = None
    reference_price_at_signal: Optional[float] = Field(default=None, gt=0)
    reference_price_timestamp: Optional[str] = None
    reference_timezone: Optional[str] = None
    take_profit: Optional[float] = Field(default=None, description="Take profit price target")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss protective boundary")
    time_horizon_days: int = Field(
        ge=1,
        le=252,
        description=(
            "Target evaluation horizon in trading days; the agent must choose it. "
            "For a stated range, use its upper bound."
        ),
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Conviction score")
    time_horizon_label: Optional[str] = Field(
        default=None,
        description="Original agent wording for the holding-period range, preserved for audit display.",
    )
    thesis_summary: str = Field(default="", description="Core narrative thesis")

    @model_validator(mode="after")
    def _validate_entry_contract(self):
        expected_action = {
            PortfolioRating.BUY: "BUY",
            PortfolioRating.OVERWEIGHT: "BUY",
            PortfolioRating.HOLD: "HOLD",
            PortfolioRating.UNDERWEIGHT: "SELL",
            PortfolioRating.SELL: "SELL",
        }[self.rating]
        if self.action != expected_action:
            raise ValueError("rating and action must use the same direction")

        for name in ("signal_timestamp", "reference_price_timestamp"):
            timestamp = getattr(self, name)
            if timestamp is not None:
                try:
                    parsed = datetime.fromisoformat(timestamp)
                    if parsed.tzinfo is None or parsed.utcoffset() is None:
                        raise ValueError("timestamp must include timezone")
                except ValueError as exc:
                    raise ValueError(f"{name} must be a timezone-aware ISO timestamp") from exc

        prices = {
            "planned_entry_price": self.planned_entry_price,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
        }
        for name, price in prices.items():
            if price is not None and (not math.isfinite(price) or price <= 0):
                raise ValueError(f"{name} must be finite and positive")

        if self.action == "HOLD":
            if self.planned_entry_price is not None and self.take_profit is not None and self.stop_loss is not None:
                if not (self.stop_loss < self.planned_entry_price < self.take_profit):
                    raise ValueError("HOLD limit accumulation: stop_loss < planned_entry_price < take_profit required")
                if self.entry_mode is None:
                    object.__setattr__(self, "entry_mode", EntryMode.ASSUMED_AI_ENTRY)
            return self
        if self.take_profit is None or self.stop_loss is None:
            raise ValueError("actionable signals require take_profit and stop_loss")
        if self.action == "BUY" and self.take_profit <= self.stop_loss:
            raise ValueError("BUY take_profit must be above stop_loss")
        if self.action == "SELL" and self.take_profit >= self.stop_loss:
            raise ValueError("SELL take_profit must be below stop_loss")
        if self.planned_entry_price is None:
            if self.entry_mode is None:
                object.__setattr__(self, "entry_mode", EntryMode.T1_OPEN)
            return self
        if self.action == "BUY" and not (self.stop_loss < self.planned_entry_price < self.take_profit):
            raise ValueError("BUY take_profit and stop_loss must surround planned_entry_price")
        if self.action == "SELL" and not (self.take_profit < self.planned_entry_price < self.stop_loss):
            raise ValueError("SELL take_profit and stop_loss must surround planned_entry_price")
        if self.entry_mode is None:
            object.__setattr__(self, "entry_mode", EntryMode.ASSUMED_AI_ENTRY)
        return self


def portfolio_decision_to_signal_contract(
    decision: PortfolioDecision,
    ticker: str,
    signal_date: str,
    planned_entry_price: Optional[float] = None,
) -> SignalContract:
    """Convert only typed PM fields into the canonical signal contract."""
    action = {
        PortfolioRating.BUY: "BUY",
        PortfolioRating.OVERWEIGHT: "BUY",
        PortfolioRating.HOLD: "HOLD",
        PortfolioRating.UNDERWEIGHT: "SELL",
        PortfolioRating.SELL: "SELL",
    }[decision.rating]
    take_profit = decision.take_profit if decision.take_profit is not None else decision.price_target
    if action != "HOLD" and (take_profit is None or decision.stop_loss is None):
        raise ValueError("actionable ratings require take_profit and stop_loss")

    # Validate whether planned_entry_price is coherent with stop_loss and take_profit; fallback to T1_OPEN if bounded violated
    valid_planned_entry = None
    if planned_entry_price is not None and decision.stop_loss is not None and take_profit is not None:
        if action in ("BUY", "HOLD") and decision.stop_loss < planned_entry_price < take_profit:
            valid_planned_entry = planned_entry_price
        elif action == "SELL" and take_profit < planned_entry_price < decision.stop_loss:
            valid_planned_entry = planned_entry_price

    entry_mode = None
    if valid_planned_entry is not None:
        entry_mode = EntryMode.ASSUMED_AI_ENTRY
    elif action != "HOLD":
        entry_mode = EntryMode.T1_OPEN

    return SignalContract(
        ticker=ticker,
        signal_date=signal_date,
        rating=decision.rating,
        action=action,
        entry_mode=entry_mode,
        planned_entry_price=valid_planned_entry,
        take_profit=take_profit,
        stop_loss=decision.stop_loss,
        time_horizon_days=decision.time_horizon_days,
        confidence=decision.confidence,
        time_horizon_label=decision.time_horizon,
        thesis_summary=decision.investment_thesis,
    )
