from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

import cli.main as main


def test_main_menu_passes_console_to_announcement_renderer():
    data = {"announcements": [], "require_attention": False}
    with (
        patch.object(main, "fetch_announcements", return_value=data),
        patch.object(main, "display_announcements") as display,
        patch.object(main.questionary, "select") as select,
    ):
        select.return_value.ask.return_value = "exit"
        with pytest.raises(typer.Exit) as exc_info:
            main.main_menu(SimpleNamespace(invoked_subcommand=None))

    assert exc_info.value.exit_code == 0
    display.assert_called_once_with(main.console, data)


def test_main_uses_progress_analyst_contract_and_report_renderer():
    assert main.PROGRESS_ANALYST_ORDER == ["market", "social", "news", "fundamentals"]
    assert all(isinstance(name, str) for name in main.PROGRESS_ANALYST_ORDER)

    buffer = main.MessageBuffer()
    buffer.init_for_analysis(["market"])
    assert "Market Analyst" in buffer.agent_status

    main.display_complete_report({})
