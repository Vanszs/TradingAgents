from typing import Optional

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.technical_indicators_tools import get_indicators


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str, asset_type: str = "stock", trade_date: Optional[str] = None) -> str:
    """Describe the exact instrument and inject quantitative structural levels."""
    instrument_label = "asset" if asset_type == "crypto" else "instrument"
    extra_hint = (
        " Treat it as a crypto asset rather than a company, and do not assume company fundamentals are available."
        if asset_type == "crypto"
        else ""
    )
    structural_info = ""
    if trade_date:
        from tradingagents.dataflows.structural_levels import get_market_structural_summary
        summary = get_market_structural_summary(ticker, trade_date)
        if summary:
            structural_info = f"\n\n{summary}"

    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
        + structural_info
    )


def build_exchange_filing_context(ticker: str, asset_type: str = "stock") -> str:
    """Return filing and regulatory disclosure guidance based on ticker and asset type."""
    ticker_clean = ticker.strip()
    ticker_upper = ticker_clean.upper()

    if asset_type == "crypto" or ticker_upper.endswith("-USD"):
        return (
            "Exchange & Regulatory Context: On-chain metrics, tokenomics, protocol whitepapers, "
            "developer GitHub activity, and analytics from platforms like CoinGecko and DeFiLlama."
        )

    if ticker_upper.endswith(".JK"):
        return (
            "Exchange & Regulatory Context: Indonesia Stock Exchange (IDX / Bursa Efek Indonesia - BEI) "
            "regulated by OJK (Otoritas Jasa Keuangan). Key disclosures include Quarterly Financial Statements "
            "(Q1, Semester I / Q2, Q3), Audited Annual Financial Statements, Keterbukaan Informasi BEI, "
            "and domestic financial media (CNBC Indonesia, Bisnis.com, Kontan)."
        )

    if ticker_upper.endswith(".TO"):
        return (
            "Exchange & Regulatory Context: Toronto Stock Exchange (TSX / TSX Venture), regulated by CIRO/CSA. "
            "Disclosures on SEDAR+ (Annual Information Forms, Quarterly MD&A, Financial Statements)."
        )

    if ticker_upper.endswith(".L"):
        return (
            "Exchange & Regulatory Context: London Stock Exchange (LSE), regulated by the FCA. "
            "Regulatory News Service (RNS) announcements, Annual Reports, and Half-Yearly Results."
        )

    if ticker_upper.endswith(".HK"):
        return (
            "Exchange & Regulatory Context: Hong Kong Exchanges and Clearing (HKEX), regulated by SFC. "
            "HKEXnews announcements, Interim and Annual Reports."
        )

    if ticker_upper.endswith(".T"):
        return (
            "Exchange & Regulatory Context: Tokyo Stock Exchange (TSE / JPX), regulated by the FSA/SESC. "
            "TDnet disclosures, Yuho (Yukashoken Hokokusho) Annual Securities Reports, and Tanshin quarterly summaries."
        )

    # US stocks or standard ticker without non-US dot suffix
    return (
        "Exchange & Regulatory Context: US Securities and Exchange Commission (SEC) EDGAR filings "
        "(Form 10-K Annual Reports, Form 10-Q Quarterly Reports, Form 8-K Material Events) "
        "and Investor Relations disclosures."
    )


