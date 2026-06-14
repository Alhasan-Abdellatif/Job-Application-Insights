"""Smoke tests — verify the package imports cleanly and the CLI runs."""

from __future__ import annotations

import job_application_insights
from job_application_insights.cli import main
from job_application_insights.config import load_settings


def test_version_is_string() -> None:
    assert isinstance(job_application_insights.__version__, str)
    assert job_application_insights.__version__.count(".") >= 1


def test_cli_version_flag(capsys) -> None:
    rc = main(["--version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert job_application_insights.__version__ in captured.out


def test_cli_help_runs() -> None:
    rc = main([])
    assert rc == 0


def test_settings_load() -> None:
    s = load_settings()
    assert s.data_dir.name == "data"
    assert s.raw_dir.parts[-2:] == ("data", "raw")
