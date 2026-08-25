"""Pin the CLI module split: extracted modules stay importable and keep
working through the original entry points (cli.main / cli.commands.backtest)."""
import inspect

from cli.message_buffer import MessageBuffer


def test_message_buffer_init_builds_agent_status():
    buffer = MessageBuffer()
    buffer.init_for_analysis(["market"])
    assert "Market Analyst" in buffer.agent_status
    assert "Bull Researcher" in buffer.agent_status
    assert buffer.report_sections["market_report"] is None


def test_update_display_has_no_spinner_text_param():
    from cli.display import update_display

    assert list(inspect.signature(update_display).parameters) == [
        "layout",
        "stats_handler",
        "start_time",
    ]


def test_format_tokens():
    from cli.display import format_tokens

    assert format_tokens(999) == "999"
    assert format_tokens(1500) == "1.5k"


def test_main_reexports_moved_selection_and_report_functions():
    import cli.main as main
    import cli.report_io as report_io
    import cli.selections as selections

    assert main.build_headless_selections is selections.build_headless_selections
    assert main.get_user_selections is selections.get_user_selections
    assert main.save_report_to_disk is report_io.save_report_to_disk
    assert main.display_complete_report is report_io.display_complete_report
    assert main.MessageBuffer is MessageBuffer


def test_backtest_reexports_moved_ui_report_prompt_units():
    import cli.commands.backtest as backtest_mod
    import cli.commands.backtest_prompts as prompts_mod
    import cli.commands.backtest_report as report_mod
    import cli.commands.backtest_tui as tui_mod

    assert backtest_mod._BacktestUI is tui_mod._BacktestUI
    assert backtest_mod.PHASE_ICONS is tui_mod.PHASE_ICONS
    assert backtest_mod._render_all is tui_mod._render_all
    assert backtest_mod._print_window_explainer is report_mod._print_window_explainer
    assert backtest_mod._print_results_table is report_mod._print_results_table
    assert backtest_mod._prompt_for_missing_config is prompts_mod._prompt_for_missing_config
    assert backtest_mod._build_llm_agent_config is prompts_mod._build_llm_agent_config


def test_is_valid_date():
    from cli.commands.backtest_prompts import _is_valid_date

    assert _is_valid_date("2026-01-15") is True
    assert _is_valid_date("15-01-2026") is False
