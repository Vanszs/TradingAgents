import datetime
from pathlib import Path
from typing import Optional

from rich.align import Align
from rich.panel import Panel

from cli.announcements import display_announcements, fetch_announcements
from cli.utils import (
    ANALYST_ORDER as UTILS_ANALYST_ORDER,
)
from cli.utils import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    console,
    detect_asset_type,
    ensure_api_key,
    get_analysis_date,
    get_ticker,
    normalize_ticker_symbol,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)
from tradingagents.default_config import DEFAULT_CONFIG


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", "r", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents (Vanszs Edition): Institutional Multi-Agent Trading Framework[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Maintained by Vanszs | Hardened fork of [Tauric Research](https://github.com/TauricResearch) (Apache-2.0)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents (Vanszs Edition)",
        subtitle="Institutional Multi-Agent LLM Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the exact ticker symbol to analyze, including exchange suffix when needed (examples: SPY, CNC.TO, 7203.T, 0700.HK)",
            "SPY",
        )
    )
    selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    console.print(
        f"[green]Detected asset type:[/green] {asset_type.value}"
    )

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Output language
    console.print(
        create_question_box(
            "Step 3: Output Language",
            "Select the language for analyst reports and final decision"
        )
    )
    output_language = ask_output_language()

    # Step 4: Select analysts
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts(asset_type)
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth
    console.print(
        create_question_box(
            "Step 5: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 6: LLM Provider
    console.print(
        create_question_box(
            "Step 6: LLM Provider", "Select your LLM provider"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()

    # Providers with regional endpoints prompt for the region as a secondary
    # step so the main dropdown stays clean (mainland China and international
    # accounts cannot share API keys).
    if selected_llm_provider == "qwen":
        selected_llm_provider, backend_url = ask_qwen_region()
    elif selected_llm_provider == "minimax":
        selected_llm_provider, backend_url = ask_minimax_region()
    elif selected_llm_provider == "glm":
        selected_llm_provider, backend_url = ask_glm_region()

    # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
    # before model selection so it's obvious where we're connecting.
    if selected_llm_provider == "ollama":
        confirm_ollama_endpoint(backend_url)

    # Confirm the provider's API key is present; prompt the user to paste
    # one and persist it to .env if it's missing, so the analysis run
    # doesn't fail later at the first API call.
    ensure_api_key(selected_llm_provider)

    # Step 7: Thinking agents
    console.print(
        create_question_box(
            "Step 7: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 8: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box(
                "Step 8: Thinking Mode",
                "Configure Gemini thinking mode"
            )
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "Step 8: Reasoning Effort",
                "Configure OpenAI reasoning effort level"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()
    elif provider_lower == "anthropic":
        console.print(
            create_question_box(
                "Step 8: Effort Level",
                "Configure Claude effort level"
            )
        )
        anthropic_effort = ask_anthropic_effort()

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def build_headless_selections(
    ticker: str,
    analysis_date: Optional[str] = None,
    provider: Optional[str] = None,
    research_depth: Optional[int] = None,
    language: Optional[str] = None,
) -> dict:
    """Build selections dictionary for headless / automated runs without Questionary prompts."""
    normalized_ticker = normalize_ticker_symbol(ticker)
    asset_type = detect_asset_type(normalized_ticker)
    trade_date = analysis_date or datetime.datetime.now().strftime("%Y-%m-%d")

    llm_provider = (provider or DEFAULT_CONFIG.get("llm_provider", "openai")).lower()

    # Provider-aware model defaults
    provider_quick_defaults = {
        "openai": "gpt-5.4-mini",
        "anthropic": "claude-haiku-4-5",
        "google": "gemini-2.5-flash",
        "deepseek": "deepseek-chat",
        "ollama": "llama3.2",
    }
    provider_deep_defaults = {
        "openai": "gpt-5.4",
        "anthropic": "claude-sonnet-4-6",
        "google": "gemini-2.5-pro",
        "deepseek": "deepseek-reasoner",
        "ollama": "llama3.3",
    }

    if provider:
        shallow_thinker = provider_quick_defaults.get(llm_provider, DEFAULT_CONFIG.get("quick_think_llm", "gpt-5.4-mini"))
        deep_thinker = provider_deep_defaults.get(llm_provider, DEFAULT_CONFIG.get("deep_think_llm", "gpt-5.4"))
    else:
        shallow_thinker = DEFAULT_CONFIG.get("quick_think_llm", "gpt-5.4-mini")
        deep_thinker = DEFAULT_CONFIG.get("deep_think_llm", "gpt-5.4")

    backend_url = DEFAULT_CONFIG.get("backend_url")
    depth = research_depth or int(DEFAULT_CONFIG.get("max_debate_rounds", 3))
    out_lang = language or DEFAULT_CONFIG.get("output_language", "English")

    analysts = [value for _, value in UTILS_ANALYST_ORDER]

    return {
        "ticker": normalized_ticker,
        "asset_type": asset_type.value,
        "analysis_date": trade_date,
        "analysts": analysts,
        "research_depth": depth,
        "llm_provider": llm_provider,
        "backend_url": backend_url,
        "shallow_thinker": shallow_thinker,
        "deep_thinker": deep_thinker,
        "google_thinking_level": DEFAULT_CONFIG.get("google_thinking_level"),
        "openai_reasoning_effort": DEFAULT_CONFIG.get("openai_reasoning_effort"),
        "anthropic_effort": DEFAULT_CONFIG.get("anthropic_effort"),
        "output_language": out_lang,
    }
