from unittest.mock import patch, MagicMock
import os
import subprocess
from datetime import datetime
import pytz
from src import trigger_git_push, run_all


class MockDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore
        ny_tz = pytz.timezone("America/New_York")
        dt = datetime(2026, 7, 17, 9, 0, 0)
        return ny_tz.localize(dt).astimezone(tz) if tz else ny_tz.localize(dt)


@patch("subprocess.run")
def test_trigger_git_push_disabled(mock_run):
    with patch.dict(os.environ, {"ENABLE_GIT_PUSH": "false"}):
        res = trigger_git_push("dummy.md", "feat: test commit")
        assert res is False
        mock_run.assert_not_called()


@patch("os.path.exists")
@patch("subprocess.run")
def test_trigger_git_push_enabled_success(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.return_value = MagicMock(returncode=0)
    with patch.dict(os.environ, {"ENABLE_GIT_PUSH": "true"}):
        res = trigger_git_push("dummy.md", "feat: test commit")
        assert res is True
        assert mock_run.call_count >= 3  # git add, commit, push


@patch("os.path.exists")
@patch("subprocess.run")
def test_trigger_git_push_data_sub_repository(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.return_value = MagicMock(returncode=0)
    with patch.dict(os.environ, {"ENABLE_GIT_PUSH": "true"}):
        res = trigger_git_push("data/report/alpha_signal.md", "feat: publish report")
        assert res is True
        assert mock_run.call_count >= 3

        add_call_args, add_call_kwargs = mock_run.call_args_list[0]
        assert "add" in add_call_args[0]
        assert add_call_args[0][-1] == "report/alpha_signal.md"
        assert add_call_kwargs["cwd"].endswith("data")

        commit_call_args, commit_call_kwargs = mock_run.call_args_list[1]
        assert "commit" in commit_call_args[0]
        assert commit_call_kwargs["cwd"].endswith("data")


@patch("os.path.exists")
@patch("subprocess.run")
def test_trigger_git_push_case_insensitive_toggle(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.return_value = MagicMock(returncode=0)
    with patch.dict(os.environ, {"ENABLE_GIT_PUSH": "TrUe"}):
        res = trigger_git_push("dummy.md", "feat: test commit")
        assert res is True
        assert mock_run.call_count >= 3


@patch("os.path.exists")
@patch("subprocess.run")
def test_trigger_git_push_nothing_to_commit(mock_run, mock_exists):
    mock_exists.return_value = True

    def run_side_effect(args, **kwargs):
        if "commit" in args:
            err = subprocess.CalledProcessError(returncode=1, cmd=args)
            err.stderr = b"nothing to commit, working tree clean"
            raise err
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect

    with patch.dict(os.environ, {"ENABLE_GIT_PUSH": "true"}):
        res = trigger_git_push("dummy.md", "feat: test commit")
        assert res is True
        assert mock_run.call_count == 5


@patch("os.path.exists")
@patch("subprocess.run")
@patch("src.logger")
def test_trigger_git_push_subprocess_error_logging(mock_logger, mock_run, mock_exists):
    mock_exists.return_value = True

    def run_side_effect(args, **kwargs):
        if "push" in args:
            err = subprocess.CalledProcessError(returncode=1, cmd=args)
            err.stderr = b"Permission denied (publickey)."
            raise err
        return MagicMock(returncode=0)

    mock_run.side_effect = run_side_effect

    with patch.dict(os.environ, {"ENABLE_GIT_PUSH": "true"}):
        res = trigger_git_push("dummy.md", "feat: test commit")
        assert res is False
        mock_logger.error.assert_called_once()
        log_msg = mock_logger.error.call_args[0][0]
        assert "Permission denied" in log_msg


@patch("src.is_us_trading_day")
@patch("src.pull_data_from_cloud")
@patch("src.run_sorter")
@patch("src.run_extractor")
@patch("src.run_cio")
@patch("src.run_formatter")
@patch("src.run_translator")
@patch("src.build_structured_report")
@patch("src.trigger_git_push")
@patch("src.os.path.exists")
@patch("src.os.remove")
@patch("src.os.makedirs")
@patch("src.datetime", MockDatetime)
def test_run_all_git_push_integration_full(
    mock_makedirs,
    mock_remove,
    mock_exists,
    mock_trigger_git_push,
    mock_build_structured,
    mock_translator,
    mock_formatter,
    mock_cio,
    mock_extractor,
    mock_sorter,
    mock_pull,
    mock_is_trading_day,
):
    # Setup mocks
    mock_is_trading_day.return_value = True
    mock_formatter.return_value = True
    mock_build_structured.return_value = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "structured",
        "signal_20260717.json",
    )

    # We want os.path.exists to return:
    # False for as.lock
    # True for Korean report file so we trigger the second push
    def side_effect_exists(path):
        if "as.lock" in path:
            return False
        if ("alpha_signal" in path and "_ko.md" in path) or "structured" in path:
            return True
        return False

    mock_exists.side_effect = side_effect_exists

    mock_formatter.return_value = True

    # Run pipeline
    run_all(report_type="full")

    # Assert trigger_git_push was called twice (once for EN, once for KO + Structured)
    assert mock_trigger_git_push.call_count == 2

    # Check calls
    calls = mock_trigger_git_push.call_args_list
    assert "alpha_signal_" in calls[0][0][0]
    assert "publish alpha signal (EN)" in calls[0][0][1]
    assert any("alpha_signal_" in f and "_ko.md" in f for f in calls[1][0][0])
    assert any("structured" in f for f in calls[1][0][0])
    assert "publish alpha signal (KO)" in calls[1][0][1]


@patch("src.is_us_trading_day")
@patch("src.pull_data_from_cloud")
@patch("src.run_sorter")
@patch("src.run_extractor")
@patch("src.run_cio")
@patch("src.run_formatter")
@patch("src.run_translator")
@patch("src.trigger_git_push")
@patch("src.os.path.exists")
@patch("src.os.remove")
@patch("src.os.makedirs")
@patch("src.datetime", MockDatetime)
def test_run_all_git_push_integration_premarket(
    mock_makedirs,
    mock_remove,
    mock_exists,
    mock_trigger_git_push,
    mock_translator,
    mock_formatter,
    mock_cio,
    mock_extractor,
    mock_sorter,
    mock_pull,
    mock_is_trading_day,
):
    # Setup mocks
    mock_is_trading_day.return_value = True

    # We want os.path.exists to return:
    # False for as.lock
    # True for Korean report file so we trigger the second push
    def side_effect_exists(path):
        if "as.lock" in path:
            return False
        if "alpha_signal_premarket" in path and "_ko.md" in path:
            return True
        return False

    mock_exists.side_effect = side_effect_exists

    mock_formatter.return_value = True

    # Run pipeline
    run_all(report_type="premarket")

    # Assert trigger_git_push was called twice (once for EN, once for KO)
    assert mock_trigger_git_push.call_count == 2

    # Check calls
    calls = mock_trigger_git_push.call_args_list
    assert "alpha_signal_premarket_" in calls[0][0][0]
    assert "publish premarket signal (EN)" in calls[0][0][1]
    assert any("alpha_signal_premarket_" in f and "_ko.md" in f for f in calls[1][0][0])
    assert "publish premarket signal (KO)" in calls[1][0][1]
