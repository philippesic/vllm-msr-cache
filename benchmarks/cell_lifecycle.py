# SPDX-License-Identifier: Apache-2.0
"""Durable per-cell lifecycle markers for benchmark subprocesses."""

import csv
import json
import os
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any

from benchmarks.csv_utils import CSV_FIELD_SIZE_LIMIT  # noqa: F401

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

CELL_MARKER_SCHEMA = "semantic-cache-cell-end-v1"
_VALID_SIGNALS = {int(value) for value in signal.valid_signals()}

# Preservation-event JSON is intentionally kept in the per-cell CSV for
# exact lifecycle/merge binding; one event array can exceed csv's platform
# default field limit. Set this once at the shared lifecycle boundary so
# count_csv_rows, watchdog cleanup, and grid merge use the same limit.


def cell_marker_path(
    output_dir: str,
    policy: str,
    seed: int | None,
    attempt_id: str | None = None,
) -> str:
    suffix = f"_attempt{attempt_id}" if attempt_id is not None else ""
    return os.path.join(output_dir, f"cell_{policy}_seed{seed}{suffix}.end.json")


def cell_result_path(output_dir: str, attempt_id: str | None = None) -> str:
    if attempt_id is None:
        return os.path.join(output_dir, "results.csv")
    return os.path.join(output_dir, f"results_attempt{attempt_id}.csv")


def process_birth_identity(pid: int | None) -> str | None:
    """Return Linux /proc start ticks for PID-reuse-safe cleanup."""
    if pid is None or pid <= 0 or sys.platform != "linux":
        return None
    try:
        with open(f"/proc/{pid}/stat") as handle:
            raw = handle.read()
        fields = raw[raw.rfind(")") + 2 :].split()
        return fields[19]
    except (FileNotFoundError, IndexError, OSError):
        return None


@contextmanager
def _marker_lock(path: str):
    lock_path = f"{path}.lock"
    with open(lock_path, "a") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_cell_marker_unlocked(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    record = {
        "schema": CELL_MARKER_SCHEMA,
        "updated_at_unix_s": time.time(),
        **payload,
    }
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_cell_marker(path: str, payload: dict[str, Any]) -> None:
    """Atomically and durably replace a cell marker under its lock."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with _marker_lock(path):
        _write_cell_marker_unlocked(path, payload)


def read_cell_marker(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(record, dict) or record.get("schema") != CELL_MARKER_SCHEMA:
        return None
    return record


def update_cell_marker(
    path: str,
    updates: dict[str, Any],
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with _marker_lock(path):
        current = read_cell_marker(path)
        if expected:
            if current is None:
                raise ValueError(f"cannot update missing cell marker: {path}")
            mismatches = {
                key: (current.get(key), value)
                for key, value in expected.items()
                if current.get(key) != value
            }
            if mismatches:
                raise ValueError(f"cell marker identity mismatch: {mismatches}")
        current = current or {}
        current.update(updates)
        _write_cell_marker_unlocked(path, current)
        return current


def count_csv_rows(path: str) -> int:
    try:
        with open(path, newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (FileNotFoundError, OSError):
        return 0


def termination_details(
    returncode: int | None, *, timed_out: bool = False
) -> dict[str, Any]:
    """Normalize exit status into an auditable reason and signal."""
    if timed_out:
        termination_signal = None
        if returncode is not None and returncode < 0 and -returncode in _VALID_SIGNALS:
            termination_signal = -returncode
        elif (
            returncode is not None
            and 128 < returncode < 256
            and returncode - 128 in _VALID_SIGNALS
        ):
            termination_signal = returncode - 128
        return {
            "termination_reason": "timeout",
            "termination_signal": termination_signal,
        }
    if returncode is None:
        return {
            "termination_reason": "parent_disappeared",
            "termination_signal": None,
        }
    if returncode < 0:
        if -returncode not in _VALID_SIGNALS:
            return {
                "termination_reason": "nonzero_exit",
                "termination_signal": None,
            }
        return {
            "termination_reason": "signal",
            "termination_signal": -returncode,
        }
    # run_latency_suite's SIGTERM handler exits with 128 + SIGTERM so that
    # its cleanup path runs before the wrapper observes the process exit.
    if 128 < returncode < 256 and returncode - 128 in _VALID_SIGNALS:
        return {
            "termination_reason": "signal",
            "termination_signal": returncode - 128,
        }
    return {
        "termination_reason": "completed" if returncode == 0 else "nonzero_exit",
        "termination_signal": None,
    }


def _fsync_directory(directory: str) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_csv_atomically(
    path: str, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    directory = os.path.dirname(path) or "."
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
