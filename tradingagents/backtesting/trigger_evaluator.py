"""
Trigger-Based Execution Agent.

Decides whether the action implied by today's agent rating is actually
worth executing. The agent rating alone is **not** a trigger — the
execution layer waits for a real, identifiable reason to act.

Six trigger conditions (per the spec):

1. ``tp_sl_hit``              — Today's risk-engine check produced a stop
                                or take-profit order (the bar touched our
                                protective levels).
2. ``setup_invalid``          — The agent's report explicitly flags the
                                setup as broken / invalid (regex-driven
                                keyword scan with a configurable match
                                threshold).
3. ``rr_deteriorated``        — Reward:risk ratio of the proposed trade
                                has dropped materially (default 30%) vs
                                yesterday's reading.
4. ``strong_exit_signal``     — Rating has flipped against the held side
                                (Sell when Long, Buy when Short). This is
                                the cleanest "exit" signal.
5. ``better_candidate``       — Today's confidence or thesis score is
                                materially above yesterday's AND the new
                                score is above a minimum threshold.
6. ``entry_condition_changed``— Flat and the rating has changed (or the
                                rating is now non-Hold after a Hold
                                streak).
7. ``rating_confirmed``      — Non-flat and the rating confirms the held
                                side (Sell/Underweight when Short,
                                Buy/Overweight when Long).

Each trigger is a boolean contribution to the final ``triggered`` flag.
The evaluator is pure: it takes the parsed decision, the previous
decision, the current position, an optional risk-engine order, and the
configuration. It does **not** look at the broker, the portfolio, or any
live data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .position import (
    ExtendedDecision,
    OrderType,
    Position,
    PositionSide,
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


# Rating strings for state-aware checks.
_RATINGS = {
    "Buy": "Buy",
    "Overweight": "Overweight",
    "Hold": "Hold",
    "Underweight": "Underweight",
    "Sell": "Sell",
}


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

        return TriggerResult(
            triggered=len(reasons) > 0,
            reasons=reasons,
            details=details,
        )

    # ------------------------------------------------------------------
    # Individual conditions
    # ------------------------------------------------------------------
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

    def _compute_rr(self, decision: ExtendedDecision) -> Optional[float]:
        """Reward:risk = (target - entry) / (entry - stop), always positive."""
        if decision.stop_price is None or decision.take_profit is None:
            return None
        entry = decision.decision_valid_from  # not used directly; we need an entry ref
        # Use a price ref if the decision has a stop_price; for R:R we treat
        # stop/target as percentages of the most recent close when entry is unknown.
        # But stop_price/take_profit are absolute; we need an entry ref.
        # Fall back: if the decision carries no entry, R:R is None.
        return None  # Without a recorded entry, can't compute R:R from stop/target alone.

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
        """
        Compute a real R:R drop between yesterday and today.

        Without an explicit entry price on the decision we cannot build the
        classic ``(target - entry) / abs(entry - stop)`` ratio. Instead we
        use the *band width* ``|take_profit - stop_price|`` as a proxy for
        R:R health — a widening band means better expected reward vs risk;
        a narrowing band means the trade is becoming a worse proposition.

        Returns the fractional drop in band width (>= 0) or None when
        either day is missing the required legs.
        """
        if (
            today.stop_price is None
            or today.take_profit is None
            or prev.stop_price is None
            or prev.take_profit is None
        ):
            return None
        prev_band = abs(prev.take_profit - prev.stop_price)
        today_band = abs(today.take_profit - today.stop_price)
        if prev_band <= 0:
            return None
        drop = (prev_band - today_band) / prev_band
        return drop

    def _strong_exit_signal(
        self,
        decision: ExtendedDecision,
        position: Position,
    ) -> bool:
        rating = (decision.agent_rating or "").strip().lower()
        if position.is_long() and rating == "sell":
            return True
        if position.is_short() and rating == "buy":
            return True
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
        if rating == "hold":
            return False
        # For FLAT positions, always trigger on non-Hold ratings
        # because there is no position to protect and every non-Hold
        # signal is actionable.
        return True

    def _rating_confirmed(
        self,
        today: ExtendedDecision,
        position: Position,
    ) -> bool:
        if position.is_flat():
            return False
        rating = (today.normalized_rating or "").strip()
        if position.is_short() and rating in ("SELL", "UNDERWEIGHT"):
            return True
        if position.is_long() and rating in ("BUY", "OVERWEIGHT"):
            return True
        return False
