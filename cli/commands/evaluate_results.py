"""Persist a point-in-time single-shot evaluation bundle."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from tradingagents.dataflows.utils import safe_ticker_component

_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|token|secret|password|credential|authorization|headers|auth)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(r"(?:bearer\s+|sk-[A-Za-z0-9][A-Za-z0-9_-]{8,})\S*", re.IGNORECASE)
_SOURCE_DATE_FIELDS = {
    "news": ("published_at", "publishedAt", "pub_date", "date"),
    "fundamentals": (
        "available_date", "availableDate", "reportedDate", "filingDate",
        "publicationDate", "publishedAt", "date",
    ),
    "sentiment": ("timestamp", "published_at", "publishedAt", "date"),
    "broker_activity": ("timestamp", "published_at", "publishedAt", "date"),
}


def _parse_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _validated_records(
    name: str, records: Any, trade_date: str
) -> tuple[list[dict[str, Any]], str | None]:
    if records is None:
        return [], None
    if not isinstance(records, list):
        raise ValueError(f"snapshot {name} must be a list")
    fields = _SOURCE_DATE_FIELDS[name]
    validated: list[dict[str, Any]] = []
    dates: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"snapshot {name}[{index}] must be an object")
        observed = next((_parse_date(record.get(field)) for field in fields if record.get(field)), None)
        if observed is None:
            raise ValueError(f"snapshot {name}[{index}] has no usable publication date")
        if observed > trade_date:
            raise ValueError(f"snapshot {name}[{index}] contains data after trade_date")
        validated.append(record)
        dates.append(observed)
    return validated, max(dates) if dates else None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="python"))
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _sanitize_url(value: str) -> str:
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value

    username = parts.username
    password = parts.password
    userinfo = ""
    if username is not None:
        userinfo = "[REDACTED]"
        if password is not None:
            userinfo += ":[REDACTED]"
        userinfo += "@"
    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{userinfo}{hostname}"
    if parts.port is not None:
        netloc += f":{parts.port}"
    query = [
        (key, "[REDACTED]" if _SECRET_KEY.search(key) else item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), parts.fragment))


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove credential-bearing keys and values before writing config.json."""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if not _SECRET_KEY.search(str(key))
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, str):
            return _SECRET_VALUE.sub("[REDACTED]", _sanitize_url(value))
        return value

    return clean(config)


def _strict_trade_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid trade date: {value!r}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def write_result_bundle(
    *,
    root: str | Path,
    ticker: str,
    trade_date: str,
    snapshot_data: dict[str, Any],
    agent_report: str,
    signal: Any,
    evaluation: Any,
    config: dict[str, Any],
    execution_data: dict[str, Any] | None = None,
) -> Path:
    """Write one immutable T0 snapshot plus its forward evaluation."""
    safe_ticker = safe_ticker_component(ticker)
    trade_date = _strict_trade_date(trade_date)

    ohlcv = pd.DataFrame(snapshot_data.get("ohlcv", []))
    ohlcv_cutoff_passed = True
    max_ohlcv_date = None
    if not ohlcv.empty:
        if "date" not in ohlcv.columns:
            raise ValueError("snapshot OHLCV has no date column")
        normalized_dates = ohlcv["date"].map(_parse_date)
        if normalized_dates.isna().any():
            raise ValueError("snapshot OHLCV contains an invalid date")
        ohlcv["date"] = normalized_dates
        max_ohlcv_date = ohlcv["date"].max()
        ohlcv_cutoff_passed = bool((ohlcv["date"] <= trade_date).all())
        if not ohlcv_cutoff_passed:
            raise ValueError("snapshot OHLCV contains data after trade_date")

    source_records: dict[str, list[dict[str, Any]]] = {}
    source_max_dates: dict[str, str | None] = {}
    source_cutoffs: dict[str, bool] = {}
    for name in ("news", "fundamentals", "sentiment", "broker_activity"):
        records, max_date = _validated_records(name, snapshot_data.get(name, []), trade_date)
        source_records[name] = records
        source_max_dates[name] = max_date
        source_cutoffs[name] = True

    base_dir = Path(root) / safe_ticker / trade_date
    base_dir.mkdir(parents=True, exist_ok=True)
    lock_path = base_dir / ".bundle.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            version = 1
            while (base_dir / f"v{version}.0").exists():
                version += 1
            output_dir = base_dir / f"v{version}.0"
            temp_dir = Path(tempfile.mkdtemp(prefix=f".v{version}.0-", dir=base_dir))
            try:
                snapshot_dir = temp_dir / "snapshot"
                snapshot_dir.mkdir()
                ohlcv.to_csv(snapshot_dir / "ohlcv.csv", index=False)

                for name, records in source_records.items():
                    _write_json(snapshot_dir / f"{name}.json", records)

                (temp_dir / "agent_report.md").write_text(agent_report or "", encoding="utf-8")
                _write_json(temp_dir / "signal.json", signal)
                _write_json(temp_dir / "evaluation.json", evaluation)
                execution_data = execution_data or {}
                _write_json(temp_dir / "execution_data.json", execution_data)
                _write_json(
                    temp_dir / "leakage_audit.json",
                    {
                        "status": "PASSED" if all(source_cutoffs.values()) and ohlcv_cutoff_passed else "FAILED",
                        "trade_date": trade_date,
                        "point_in_time_mode": True,
                        "model_snapshot_only": True,
                        "forward_execution_data_persisted": True,
                        "selected_interval": execution_data.get("selected_interval"),
                        "max_ohlcv_date": max_ohlcv_date,
                        "ohlcv_cutoff_passed": ohlcv_cutoff_passed,
                        "source_max_dates": source_max_dates,
                        "source_cutoffs": source_cutoffs,
                        "execution_policy": _field(evaluation, "entry_policy"),
                        "fill_status": _field(_field(evaluation, "outcome"), "value", _field(evaluation, "outcome")),
                        "actual_entry_price": _field(evaluation, "actual_entry_price"),
                        "planned_entry_price": _field(evaluation, "planned_entry_price"),
                        "signal_timestamp": _field(evaluation, "signal_timestamp") or _field(evaluation, "signal_date"),
                    },
                )
                _write_json(temp_dir / "config.json", sanitize_config(config))
                _write_json(
                    temp_dir / "summary.json",
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "signal": signal,
                        "evaluation": evaluation,
                    },
                )
                os.replace(temp_dir, output_dir)
            except BaseException:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return output_dir
