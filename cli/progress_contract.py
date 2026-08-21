"""Shared progress metadata and message formatting for CLI displays."""
from __future__ import annotations

import ast
from typing import Any

FIXED_AGENTS = {
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}

ANALYST_MAPPING = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

REPORT_SECTIONS = {
    "market_report": ("market", "Market Analyst"),
    "sentiment_report": ("social", "Sentiment Analyst"),
    "news_report": ("news", "News Analyst"),
    "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
    "investment_plan": (None, "Research Manager"),
    "trader_investment_plan": (None, "Trader"),
    "final_trade_decision": (None, "Portfolio Manager"),
}

ALL_TEAMS = {
    "Analyst": ["Market Analyst", "Sentiment Analyst", "News Analyst", "Fundamentals Analyst"],
    "Research": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading": ["Trader"],
    "Risk": ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"],
    "Portfolio": ["Portfolio Manager"],
}

CANONICAL_ANALYST_ORDER = ("market", "social", "news", "fundamentals")
ANALYST_ORDER = list(CANONICAL_ANALYST_ORDER)
ANALYST_AGENT_NAMES = ANALYST_MAPPING
ANALYST_REPORT_MAP = {
    key: section for section, (key, _) in REPORT_SECTIONS.items() if key is not None
}
def short_agent_label(agent: str) -> str:
    return agent.replace(" Analyst", "").replace(" Researcher", "").replace(" Manager", " Mgr")


def extract_content_string(content: Any) -> str | None:
    """Extract meaningful text from LangChain message content."""
    def is_empty(value: Any) -> bool:
        if value is None or value == "":
            return True
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return True
            if stripped in ("[]", "{}", "()", "set()"):
                return True
            return False
        return not bool(value)

    if is_empty(content):
        return None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if not is_empty(text) else None
    if isinstance(content, list):
        parts = [
            item.get("text", "").strip()
            if isinstance(item, dict) and item.get("type") == "text"
            else item.strip() if isinstance(item, str) else ""
            for item in content
        ]
        result = " ".join(part for part in parts if part and not is_empty(part))
        return result or None
    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message: Any) -> tuple[str, str | None]:
    """Return the main CLI display type and text for a LangChain message."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, "content", None))
    if isinstance(message, HumanMessage):
        return ("Control", content) if content and content.strip() == "Continue" else ("User", content)
    if isinstance(message, ToolMessage):
        return "Data", content
    if isinstance(message, AIMessage):
        return "Agent", content
    return "System", content


def format_tool_args(args: Any, max_length: int = 80) -> str:
    result = str(args)
    return result if len(result) <= max_length else result[: max_length - 3] + "..."
