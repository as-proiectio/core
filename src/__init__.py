"""
Orchestration and Main Execution Loop for the Alpha Signals Core Pipeline.

This module acts as the pipeline driver, pulling news data via SCP from the Sieve
scraping server, managing active lock states to prevent duplicate executions,
and triggering sequential pipeline runs (incremental, full, and premarket).
"""

import argparse
import glob
import json
import os
import pytz
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

# Add project root to sys.path to allow importing from 'shared' and local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import holidays
from cio import run_cio
from extractor import run_extractor
from formatter import run_formatter
from shared.env_utils import load_env_file
from shared.shared_logger import setup_logger
from sorter import run_sorter
from translator import run_translator

# Populate os.environ with local config prior to bootstrapping dependencies.
load_env_file()

logger = setup_logger(logger_name=__name__)


def trigger_git_push(file_path: str, commit_msg: str) -> bool:
    """
    Automatically commits and pushes the formatted report to the git data repository.
    Safe exception wrapping ensures network/SSH failures do not interrupt core pipeline.
    """
    enable_push = os.getenv("ENABLE_GIT_PUSH", "false")
    if enable_push.lower() != "true":
        logger.info("Git push skipped: ENABLE_GIT_PUSH is not set to true.")
        return False

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if the target file is inside the 'data' directory which is a separate repository
    abs_file_path = os.path.abspath(os.path.join(project_root, file_path))
    data_dir = os.path.join(project_root, "data")

    if abs_file_path.startswith(os.path.abspath(data_dir)) and os.path.exists(
        os.path.join(data_dir, ".git")
    ):
        git_cwd = data_dir
        git_file_path = os.path.relpath(abs_file_path, data_dir)
        git_dir = os.path.join(data_dir, ".git")
        logger.info(
            f"Target file is inside sub-repository 'data'. Operating git from 'data' directory (path: {git_file_path})."
        )
    else:
        git_cwd = project_root
        git_file_path = file_path
        git_dir = os.path.join(project_root, ".git")

    try:
        # Check for git repo existence
        if not os.path.exists(git_dir):
            logger.warning(f"Git push skipped: .git directory not found at {git_dir}.")
            return False

        # Stage all modified and new files in git_cwd to ensure a clean worktree before rebase
        subprocess.run(
            ["git", "add", "-A"], check=True, capture_output=True, cwd=git_cwd
        )

        try:
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                check=True,
                capture_output=True,
                cwd=git_cwd,
            )
        except subprocess.CalledProcessError as commit_err:
            stderr_str = commit_err.stderr.decode("utf-8") if commit_err.stderr else ""
            stdout_str = commit_err.stdout.decode("utf-8") if commit_err.stdout else ""
            if (
                "nothing to commit" in stderr_str.lower()
                or "nothing to commit" in stdout_str.lower()
                or "working tree clean" in stderr_str.lower()
                or "working tree clean" in stdout_str.lower()
            ):
                logger.info(
                    "Git commit skipped: nothing to commit, working tree clean."
                )
            else:
                raise

        # Determine the current branch name
        branch_res = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            cwd=git_cwd,
            text=True,
        )
        current_branch = branch_res.stdout.strip()
        if not current_branch:
            current_branch = "main"

        # Pull before pushing to avoid conflicts with remote changes (like signals.json)
        logger.info(
            f"Pulling remote changes on branch '{current_branch}' with rebase..."
        )
        subprocess.run(
            ["git", "pull", "--rebase", "origin", current_branch],
            check=True,
            capture_output=True,
            cwd=git_cwd,
        )

        # Push to origin specifying the branch to handle missing tracking configuration
        logger.info(f"Pushing committed changes to origin '{current_branch}'...")
        subprocess.run(
            ["git", "push", "origin", current_branch],
            check=True,
            capture_output=True,
            cwd=git_cwd,
        )
        logger.info(f"Successfully pushed {file_path} with message: {commit_msg}")
        return True
    except subprocess.CalledProcessError as e:
        stderr_str = e.stderr.decode("utf-8") if e.stderr else ""
        logger.error(
            f"Git push failed for {file_path} due to subprocess error: {e}. Stderr: {stderr_str}"
        )
        return False
    except Exception as e:
        logger.error(f"Git push failed for {file_path}: {e}")
        return False


def is_us_trading_day(dt: datetime | None = None) -> bool:
    """
    Checks if the given datetime corresponds to a US trading day.
    US trading days are Monday-Friday excluding NYSE holidays.
    """
    us_tz = pytz.timezone("America/New_York")
    if dt is None:
        us_now = datetime.now(us_tz).date()
    else:
        us_now = dt.astimezone(us_tz).date()

    if us_now.weekday() >= 5:
        return False

    nyse_holidays = holidays.financial_holidays("US", years=us_now.year)
    if us_now in nyse_holidays:
        return False

    return True


def get_next_us_trading_day(dt: datetime) -> datetime:
    """
    Returns dt if it is a US trading day,
    otherwise rolls forward 1 day at a time until a US trading day is found.
    """
    current = dt
    while not is_us_trading_day(current):
        current += timedelta(days=1)
    return current


def pull_data_from_cloud(report_type: str = "full", target_date: str | None = None):
    """
    Pulls sieve scraped news feed data from the Oracle Cloud instance.
    Utilizes secure SCP transfer to download raw JSON data daily.
    """
    if target_date:
        today = target_date
    else:
        today = datetime.now(pytz.timezone("America/New_York")).strftime("%Y%m%d")

    # Fetch Oracle instance credentials from environment variables to protect infrastructure configuration.
    # We resolve the local user home directory dynamically rather than hardcoding.
    ORACLE_IP_ADDRESS = os.environ.get("ORACLE_IP_ADDRESS", "159.13.60.28")
    ORACLE_SSH_KEY = os.environ.get(
        "ORACLE_SSH_KEY", os.path.expanduser("~/.ssh/oracle-cloud-ssh.key")
    )

    # Resolve local data directory dynamically relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(project_root, "data")

    if report_type == "premarket":
        remote_file = f"/home/ubuntu/sieve/data/premarket_news_{today}.json"
        local_file = f"{local_dir}/premarket_news_{today}.json"
    else:
        # Both 'full' and 'incremental' use the daily_news file
        remote_file = f"/home/ubuntu/sieve/data/daily_news_{today}.json"
        local_file = f"{local_dir}/daily_news_{today}.json"

    os.makedirs(local_dir, exist_ok=True)
    scp_command = f"scp -i {ORACLE_SSH_KEY} -o StrictHostKeyChecking=no ubuntu@{ORACLE_IP_ADDRESS}:{remote_file} {local_file}"

    max_retries = 10 if report_type == "full" else 5
    retry_delay = 60  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Pulling data from oracle cloud storage... (Attempt {attempt}/{max_retries})"
            )

            subprocess.run(
                scp_command, shell=True, check=True, capture_output=True, text=True
            )

            if os.path.exists(local_file):
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if report_type == "full":
                    market_map = data.get("market_map")
                    is_empty = False
                    if not market_map:
                        is_empty = True
                    elif isinstance(market_map, dict):
                        indices = market_map.get("Indices")
                        sectors = market_map.get("Sectors")
                        if not indices and not sectors:
                            is_empty = True

                    if is_empty:
                        raise ValueError(
                            f"market_map is empty or missing in {local_file}"
                        )

                logger.info(f"Successfully pulled data: {local_file}")
                return  # Success, exit function
            else:
                raise FileNotFoundError(f"File not found after scp: {local_file}")

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            stderr_msg = ""
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                stderr_msg = f" | Details: {e.stderr.strip()}"

            logger.warning(f"Attempt {attempt} failed: {e}{stderr_msg}")

            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All retries failed.")
                raise


def run_all(report_type: str = "full"):
    """
    Run the entire data processing pipeline sequentially:
    Sorter -> Extractor -> CIO -> Formatter

    After formatter finishes successfully, cleans up all files in the /data directory
    EXCEPT for the final alpha_signal_*.md file.
    """
    # 1. New York Timezone Check
    us_tz = pytz.timezone("America/New_York")
    ny_now = datetime.now(us_tz)
    target_dt = get_next_us_trading_day(ny_now)
    today_str = target_dt.strftime("%Y%m%d")

    if report_type == "full":
        if ny_now.time() < datetime.strptime("06:00", "%H:%M").time():
            logger.info(
                f"Current NY time ({ny_now.strftime('%H:%M')}) is before 06:00. Skipping execution."
            )
            return
    elif report_type == "premarket":
        if ny_now.time() < datetime.strptime("08:30", "%H:%M").time():
            logger.info(
                f"Current NY time ({ny_now.strftime('%H:%M')}) is before 08:30. Skipping execution."
            )
            return
    elif report_type == "incremental":
        # Incremental runs at 00:00 and 04:00, no time lock strictly needed,
        # but we can let it pass anytime.
        pass

    # 2. Duplicate Execution Prevention & Stale Process Recovery
    # Uses a local PID file to prevent concurrent pipeline runs. If a lock file exists,
    # it inspects whether the active PID is alive using a liveness signal test (os.kill(pid, 0)).
    # If the process is dead, it self-heals by removing the stale lock.
    # If the process is alive but has been running for > 2 hours, it acts as a watchdog, sending SIGKILL
    # to terminate the hung process and unblock subsequent scheduler runs.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock_file = os.path.join(project_root, "logs", "as.lock")

    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as lf:
                lock_pid = int(lf.read().strip())
        except Exception:
            lock_pid = None

        if lock_pid:
            # Check process liveness by sending a dummy signal (0).
            # If the process does not exist, an OSError is caught and the process is known to be dead.
            is_running = False
            try:
                os.kill(lock_pid, 0)
                is_running = True
            except OSError:
                is_running = False

            if not is_running:
                logger.info(
                    f"Detected stale lock file (PID {lock_pid} is not running). Cleaning up lock and proceeding."
                )
                try:
                    os.remove(lock_file)
                except Exception:
                    pass
            else:
                # If running, check how long it has been running (stale threshold: 2 hours)
                try:
                    lock_mtime = os.path.getmtime(lock_file)
                    elapsed_hours = (time.time() - lock_mtime) / 3600.0
                except Exception:
                    elapsed_hours = 0.0

                if elapsed_hours >= 2.0:
                    logger.warning(
                        f"Detected hung Alpha Signals Core process (PID {lock_pid}, running for {elapsed_hours:.1f} hours). Forcefully terminating the hung process to clear bottleneck."
                    )
                    try:
                        os.kill(lock_pid, signal.SIGKILL)
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"Failed to kill hung process {lock_pid}: {e}")

                    try:
                        os.remove(lock_file)
                        logger.info(
                            "Hung process lock cleared. Proceeding with new pipeline."
                        )
                    except Exception:
                        pass
                else:
                    logger.warning(
                        f"Already running (Active process PID {lock_pid} detected, elapsed {elapsed_hours:.1f} hours). Terminating pipeline."
                    )
                    return
        else:
            try:
                os.remove(lock_file)
                logger.info("Removed corrupted lock file.")
            except Exception:
                pass

    try:
        # Create lock file
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))

        logger.info(f"Starting Alpha Signals Core Pipeline (Type: {report_type})...")

        # Bypass weekend execution skip only for incremental runs
        if report_type != "incremental" and not is_us_trading_day():
            logger.info(
                "US Market is closed (Weekend/Holiday). Skipping pipeline execution today."
            )
            return

        # Start processing pipeline
        logger.info(f"[1/5] Pulling Sieve data for {report_type}...")
        pull_data_from_cloud(report_type, target_date=today_str)
        logger.info("-----------------------------------------------------")

        logger.info("[2/5] Running Sorter...")
        run_sorter(report_type, target_date=today_str)
        logger.info("-----------------------------------------------------")

        logger.info("[3/5] Running Extractor...")
        run_extractor(report_type=report_type, target_date=today_str)
        logger.info("-----------------------------------------------------")

        # Release as.lock early after extraction completes
        if os.path.exists(lock_file):
            os.remove(lock_file)
            logger.info("Main as.lock released early after successful extraction.")

        if report_type == "incremental":
            logger.info("Running Translator for Incremental...")
            run_translator(report_type, target_date=today_str)
            logger.info(
                "Incremental pipeline completed successfully up to Translator. Exiting."
            )
            return

        logger.info("[4/5] Running CIO...")
        run_cio(report_type, target_date=today_str)
        logger.info("-----------------------------------------------------")

        logger.info("[5/5] Running Formatter...")
        logger.info("Running Formatter for English report...")
        data_dir = os.path.join(project_root, "data")
        cio_file = os.path.join(
            data_dir,
            (
                f"premarket_report_{today_str}.txt"
                if report_type == "premarket"
                else f"final_report_{today_str}.txt"
            ),
        )

        if report_type == "premarket":
            premarket_dir = os.path.join(data_dir, "premarket")
            os.makedirs(premarket_dir, exist_ok=True)
            out_file = os.path.join(
                premarket_dir, f"alpha_signal_premarket_{today_str}.md"
            )
        else:
            report_dir = os.path.join(data_dir, "report")
            os.makedirs(report_dir, exist_ok=True)
            out_file = os.path.join(report_dir, f"alpha_signal_{today_str}.md")

        success = run_formatter(cio_file, out_file, lang="en")

        if success:
            # Trigger Git push for the English report
            rel_out_file = os.path.relpath(out_file, project_root)
            if report_type == "premarket":
                commit_msg_en = f"feat(data): publish premarket signal (EN) {today_str}"
            else:
                commit_msg_en = f"feat(data): publish alpha signal (EN) {today_str}"
            trigger_git_push(rel_out_file, commit_msg_en)

            logger.info("Running Translator for Korean report...")
            try:
                ko_pushed = False

                def on_ko_ready(ko_file_path: str):
                    nonlocal ko_pushed
                    if os.path.exists(ko_file_path):
                        rel_out_file_ko = os.path.relpath(ko_file_path, project_root)
                        if report_type == "premarket":
                            commit_msg_ko = (
                                f"feat(data): publish premarket signal (KO) {today_str}"
                            )
                        else:
                            commit_msg_ko = (
                                f"feat(data): publish alpha signal (KO) {today_str}"
                            )
                        trigger_git_push(rel_out_file_ko, commit_msg_ko)
                        ko_pushed = True
                    else:
                        logger.warning(
                            f"Korean report file not found at {ko_file_path}, skipping immediate Git push."
                        )

                run_translator(
                    report_type, target_date=today_str, on_ko_report_ready=on_ko_ready
                )

                # Fallback push check if callback wasn't executed
                if not ko_pushed:
                    if report_type == "premarket":
                        out_file_ko = os.path.join(
                            data_dir,
                            "premarket",
                            f"alpha_signal_premarket_{today_str}_ko.md",
                        )
                        commit_msg_ko = (
                            f"feat(data): publish premarket signal (KO) {today_str}"
                        )
                    else:
                        out_file_ko = os.path.join(
                            data_dir, "report", f"alpha_signal_{today_str}_ko.md"
                        )
                        commit_msg_ko = (
                            f"feat(data): publish alpha signal (KO) {today_str}"
                        )

                    if os.path.exists(out_file_ko):
                        rel_out_file_ko = os.path.relpath(out_file_ko, project_root)
                        trigger_git_push(rel_out_file_ko, commit_msg_ko)
                    else:
                        logger.warning(
                            f"Korean report file not found at {out_file_ko}, skipping Git push."
                        )
            except Exception as e:
                logger.error(f"Translator failed: {e}")

            if report_type == "premarket":
                logger.info(
                    "Premarket pipeline completed successfully. Cleaning up intermediate data files..."
                )
                _cleanup_data_files(data_dir)
                logger.info("Cleanup complete.")
            else:
                logger.info(
                    "Full pipeline completed successfully. Deferring cleanup until premarket run."
                )

            logger.info("Successfully completed Alpha Signals Core Pipeline!")
        else:
            logger.warning(
                "Formatter did not return success. Skipping translation and cleanup."
            )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Cleanup (Remove lock file)
        if os.path.exists(lock_file):
            os.remove(lock_file)
            logger.info("Lock file removed.")


def _cleanup_data_files(data_dir: str):
    """Deletes all files in data_dir except *_state_pre.json and final reports"""
    if not os.path.exists(data_dir):
        return

    all_files = glob.glob(os.path.join(data_dir, "*"))
    for file_path in all_files:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            # Keep configuration files, source scripts, state files, and hidden files
            if filename in (
                "generate_index.py",
                "signals.json",
                "signals_meta.json",
                "extracted_state_pre.json",
                "translated_state_pre.json",
                "macro_translation_cache.json",
            ) or filename.startswith("."):
                continue

            # Keep the final alpha signal reports
            if filename.startswith("alpha_signal_") and filename.endswith(".md"):
                continue

            try:
                os.remove(file_path)
                logger.debug(f"Deleted {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Signals Core Pipeline")
    parser.add_argument(
        "--type",
        choices=["full", "premarket", "incremental"],
        default="full",
        help="Type of report to generate",
    )
    args = parser.parse_args()
    run_all(report_type=args.type)
