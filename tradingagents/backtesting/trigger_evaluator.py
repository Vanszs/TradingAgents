"""
Spot-long trigger evaluation.

BUY entries may be gated by setup and price conditions. SELL_TO_CLOSE orders
come only from static risk controls; agent ratings never create exits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .position import (
    ExtendedDecision,
    MarketPoint,
    Position,
)

# Regex set for "setup invalid" detection. Indonesian + English phrases.
# Conservative — every phrase must be a literal substring.
SETUP_INVALID_PATTERNS: tuple[str, ...] = (
    r"\bsetup\s+invalid\b",
    r"\bsetup\s+is\s+invalid\b",
    r"\bthesis\s+(?:broken|invalid|ruined|gone|failed)\b",
    r"\btesis\s+(?:rusak|tidak\s+valid|gagal|hilang)\b",
    r"\bskenario\s+(?:gagal|tidak\s+valid|rusak)\b",
    r"\bpositioning\s+no\s+longer\s+valid\b",
    r"\bno\s+longer\s+valid\b",
    r"\bsetup\s+(?:sudah\s+)?tidak\s+valid\b",
    r"\bbatal(?:kan)?\s+posisi\b",
    r"\bclose\s+thesis\b",
    r"\binvalidation\b",
)

@dataclass
class TriggerConfig:
    """Configuration for the trigger evaluator. All fields are optional."""
    # Master switch — disable trigger gate (every non-Hold rating fires).
    enabled: bool = True

    # Condition toggles
    trigger_on_tp_sl_hit: bool = True
    trigger_on_setup_invalid: bool = True
    trigger_on_rr_deteriorated: bool = True
    trigger_on_strong_exit_signal: bool = True
    trigger_on_better_candidate: bool = True
    trigger_on_entry_condition_changed: bool = True
    trigger_on_rating_confirmed: bool = True
    # Thresholds
    rr_deterioration_pct: float = 0.30  # 30% R:R drop fires trigger
    better_candidate_confidence_ratio: float = 1.5  # 1.5x yesterday
    better_candidate_min_confidence: float = 0.6
    setup_invalid_keyword_threshold: int = 2  # >= 2 keyword hits

    # If True, prefer the real-R:R (band-width) signal when both days carry
    # explicit stop/target legs; otherwise always use the confidence-drop
    # proxy. Default True so a documented setup with narrowing band triggers
    # even if the agent's confidence didn't move much.
    use_real_rr_when_available: bool = True

    # Hold streak length that counts as a "flat" baseline for entry-condition trigger.
    entry_condition_min_hold_streak: int = 1


@dataclass
class TriggerResult:
    triggered: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


class TriggerEvaluator:
    """
    Decide whether the agent's rating should produce an order today.

    A trigger fires if **any** enabled condition is true. If no trigger
    fires, the resulting ``triggered=False`` overrides the rating: even
    a ``Buy`` rating is treated as ``NO_ORDER`` for the day (logged as
    ``orders_skipped_no_trigger`` downstream).
    """

    SETUP_INVALID_RE = re.compile("|".join(SETUP_INVALID_PATTERNS), re.IGNORECASE)

    def __init__(self, config: Optional[TriggerConfig] = None):
        self.config = config or TriggerConfig()

    def evaluate(
        self,
        today_decision: ExtendedDecision,
        prev_decision: Optional[ExtendedDecision],
        current_position: Position,
        risk_order: Optional[Any] = None,
        current_equity: Optional[float] = None,
        current_bar: Optional[Any] = None,
    ) -> TriggerResult:
        if not self.config.enabled:
            return TriggerResult(
                triggered=True,
                reasons=["triggers_disabled"],
                details={"note": "Trigger gate is disabled; all ratings fire."},
            )

        reasons: list[str] = []
        details: dict[str, Any] = {}

        if self.config.trigger_on_tp_sl_hit and self._tp_sl_hit(risk_order):
            reasons.append("tp_sl_hit")
            details["tp_sl_hit"] = True

        if self.config.trigger_on_setup_invalid and self._setup_invalid(today_decision):
            reasons.append("setup_invalid")
            details["setup_invalid"] = True

        if self.config.trigger_on_rr_deteriorated and self._rr_deteriorated(
            today_decision, prev_decision, current_position, details
        ):
            reasons.append("rr_deteriorated")

        if self.config.trigger_on_strong_exit_signal and self._strong_exit_signal(
            today_decision, current_position
        ):
            reasons.append("strong_exit_signal")
            details["strong_exit_signal"] = True

        if self.config.trigger_on_better_candidate and self._better_candidate(
            today_decision, prev_decision, current_position, details
        ):
            reasons.append("better_candidate")

        if self.config.trigger_on_entry_condition_changed and self._entry_condition_changed(
            today_decision, prev_decision, current_position
        ):
            reasons.append("entry_condition_changed")
            details["entry_condition_changed"] = True

        if self.config.trigger_on_rating_confirmed and self._rating_confirmed(
            today_decision, current_position
        ):
            reasons.append("rating_confirmed")
            details["rating_confirmed"] = True

        if current_position.is_flat():
            price_touched = self._price_level_touched(today_decision, current_bar)
            if price_touched:
                reasons.append("price_level_touched")
                details["price_level_touched"] = True
            is_entry_signal = (
                "entry_condition_changed" in reasons
                or "rating_confirmed" in reasons
                or "better_candidate" in reasons
                or price_touched
            )
            is_invalid = (
                "setup_invalid" in reasons
                or "rr_deteriorated" in reasons
            )
            triggered = bool(is_entry_signal and not is_invalid)
        else:
            triggered = len(reasons) > 0

        return TriggerResult(
            triggered=triggered,
            reasons=reasons,
            details=details,
        )

    # ------------------------------------------------------------------
    # Individual conditions
    # ------------------------------------------------------------------
    def _price_level_touched(
        self,
        decision: ExtendedDecision,
        current_bar: Optional[Union[MarketPoint, dict, Any]] = None,
    ) -> bool:
        planned = getattr(decision, "planned_entry_price", None)
        if planned is None or current_bar is None:
            return False

        low = current_bar.get("low") if isinstance(current_bar, dict) else getattr(current_bar, "low", None)
        high = current_bar.get("high") if isinstance(current_bar, dict) else getattr(current_bar, "high", None)
        if low is None or high is None:
            return False

        planned = float(planned)

        # The only agent-created limit order is a long BUY entry.
        return float(low) <= planned
    def _tp_sl_hit(self, risk_order: Optional[Any]) -> bool:
        if risk_order is None:
            return False
        reason = getattr(risk_order, "reason", "")
        if not reason:
            return False
        reason_l = str(reason).lower()
        return "stop" in reason_l or "take_profit" in reason_l

    def _setup_invalid(self, decision: ExtendedDecision) -> bool:
        text_pieces: list[str] = []
        if decision.raw_text_excerpt:
            text_pieces.append(decision.raw_text_excerpt)
        if decision.thesis_summary:
            text_pieces.append(decision.thesis_summary)
        if not text_pieces:
            return False
        text = "\n".join(text_pieces)
        matches = self.SETUP_INVALID_RE.findall(text)
        return len(matches) >= self.config.setup_invalid_keyword_threshold

    def _rr_deteriorated(
        self,
        today: ExtendedDecision,
        prev: Optional[ExtendedDecision],
        position: Position,
        details: dict[str, Any],
    ) -> bool:
        # First-day and post-`reset` paths may not have a previous decision;
        # in that case there's nothing to deteriorate from — return False.
        if prev is None:
            return False
        # We use confidence as a proxy for R:R health when the explicit R:R
        # is not recorded on the decision. This is the same fallback the
        # order generator uses. If both decisions carry an explicit
        # entry/stop/target we can also compute a real R:R, in which case
        # a material R:R drop is a stronger signal than confidence drop.
        if today.confidence is None or prev.confidence is None:
            return False
        if prev.confidence <= 0:
            return False

        # Real R:R check: only meaningful when both days carry the
        # entry/stop/target legs.
        if self.config.use_real_rr_when_available:
            real_rr_drop = self._compute_real_rr_drop(today, prev)
            if real_rr_drop is not None:
                details["rr_drop_pct"] = real_rr_drop * 100
                details["rr_basis"] = "explicit_rr"
                if real_rr_drop >= self.config.rr_deterioration_pct:
                    return True

        # Fallback: confidence drop.
        drop = (prev.confidence - today.confidence) / prev.confidence
        details["confidence_drop_pct"] = drop * 100
        details["rr_basis"] = "confidence_proxy"
        return drop >= self.config.rr_deterioration_pct

    @staticmethod
    def _compute_real_rr_drop(
        today: ExtendedDecision,
        prev: ExtendedDecision,
    ) -> Optional[float]:
        """Compute true R:R drop based on distance from reference price."""
        if (
            today.stop_price is None
            or today.take_profit is None
            or prev.stop_price is None
            or prev.take_profit is None
        ):
            return None

        ref_today = getattr(today, "planned_entry_price", None)
        ref_prev = getattr(prev, "planned_entry_price", None)

        if ref_today is None:
            ref_today = getattr(today, "reference_price", None) or (today.stop_price * 1.05 if today.stop_price else None)
        if ref_prev is None:
            ref_prev = getattr(prev, "reference_price", None) or (prev.stop_price * 1.05 if prev.stop_price else None)

        if ref_today is None or ref_prev is None:
            return None

        prev_risk = abs(ref_prev - prev.stop_price)
        prev_reward = abs(prev.take_profit - ref_prev)
        today_risk = abs(ref_today - today.stop_price)
        today_reward = abs(today.take_profit - ref_today)

        if prev_risk <= 0 or today_risk <= 0:
            return None
        prev_rr = prev_reward / prev_risk
        today_rr = today_reward / today_risk
        if prev_rr <= 0:
            return None
        drop = (prev_rr - today_rr) / prev_rr
        return drop

    def _strong_exit_signal(
        self,
        decision: ExtendedDecision,
        position: Position,
    ) -> bool:
        # Agent outputs have no exit action. Static TP/SL and risk controls
        # own liquidation; WNS never creates an agent-driven sell.
        return False

    def _better_candidate(
        self,
        today: ExtendedDecision,
        prev: Optional[ExtendedDecision],
        position: Position,
        details: dict[str, Any],
    ) -> bool:
        if prev is None:
            return False
        # Guard against partially-populated prior decisions.
        if not hasattr(prev, "agent_rating") or not hasattr(today, "agent_rating"):
            return False
        if today.confidence is None or prev.confidence is None:
            return False
        if today.confidence < self.config.better_candidate_min_confidence:
            return False
        if prev.confidence <= 0:
            return False
        ratio = today.confidence / prev.confidence
        details["confidence_ratio"] = ratio
        return ratio >= self.config.better_candidate_confidence_ratio

    def _entry_condition_changed(
        self,
        today: ExtendedDecision,
        prev: Optional[ExtendedDecision],
        position: Position,
    ) -> bool:
        if not position.is_flat():
            return False
        rating = (today.agent_rating or "").strip().lower()
        return rating in {"buy", "overweight", "strong_buy", "strong buy"}

    def _rating_confirmed(
        self,
        today: ExtendedDecision,
        position: Position,
    ) -> bool:
        if position.is_flat():
            return False
        rating = (today.normalized_rating or "").strip().upper()
        return position.is_long() and rating == "BUY"
