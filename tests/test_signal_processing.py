"""Canonical BUY/WNS rating parser tests."""
import pytest

from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating


@pytest.mark.unit
def test_explicit_buy_normalizes_to_buy():
    assert parse_rating("Rating: Buy\nReasoning here.") == "BUY"
    assert parse_rating("Rating: Overweight\nDetails.") == "BUY"


@pytest.mark.unit
def test_legacy_non_buy_labels_normalize_to_wns():
    for label in ("Sell", "Underweight", "Hold", "WNS"):
        assert parse_rating(f"Rating: {label}") == "WNS"


@pytest.mark.unit
def test_explicit_label_wins_over_prose():
    text = "The buy thesis is weakened.\nRating: **Sell**\nExit before earnings."
    assert parse_rating(text) == "WNS"


@pytest.mark.unit
def test_defaults_are_canonical():
    assert parse_rating("No recommendation") == "WNS"
    assert parse_rating("Plain prose", default="Buy") == "BUY"
    assert parse_rating("Plain prose", default="Underweight") == "WNS"


@pytest.mark.unit
def test_all_legacy_labels_have_canonical_output():
    for label in RATINGS_5_TIER:
        assert parse_rating(f"Rating: {label}") in {"BUY", "WNS"}
