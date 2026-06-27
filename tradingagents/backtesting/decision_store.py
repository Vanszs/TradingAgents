from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .decision_schema import ParsedDecision, ensure_dir


class DecisionStore:
    def __init__(self, root: str):
        self.root = Path(root)
        ensure_dir(self.root)
        self.decisions: list[ParsedDecision] = []

    def save(self, decision: ParsedDecision) -> None:
        self.decisions.append(decision)

        ticker_dir = ensure_dir(self.root / decision.ticker)
        json_path = ticker_dir / f"{decision.trade_date}.json"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(decision.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        jsonl_path = ticker_dir / "decision_log.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict(), ensure_ascii=False, default=str) + "\n")

    def all(self) -> list[ParsedDecision]:
        return list(self.decisions)

    def latest_before(self, ticker: str, trade_date: str) -> ParsedDecision | None:
        candidates = [
            d for d in self.decisions
            if d.ticker == ticker and d.trade_date < trade_date
        ]

        if not candidates:
            return None

        return sorted(candidates, key=lambda d: d.trade_date)[-1]