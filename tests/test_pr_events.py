from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmarking.resources import process_stat, sample_process_tree
from wait_hint_offload.evidence_events import EventJournal, read_journal


def test_event_journal_detects_missing_record_clock_regression_and_truncation(
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path, "fixture")
    journal.emit("transfer", {"bytes": 256, "success": False})
    journal.close()
    journal.close()
    records = read_journal(journal.path)
    assert [record["event"] for record in records] == ["open", "transfer", "close"]
    assert records[1]["data"] == {"bytes": 256, "success": False}
    missing = tmp_path / "missing.jsonl"
    missing.write_text("".join(json.dumps(record) + "\n" for record in records[::2]))
    with pytest.raises(ValueError, match="ordering"):
        read_journal(missing)
    records[1]["monotonic_ns"] = 0
    missing.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="clock"):
        read_journal(missing)
    missing.write_text(journal.path.read_text().rstrip())
    with pytest.raises(ValueError, match="truncated"):
        read_journal(missing)


def test_process_tree_measurement_includes_shared_pss_faults_swap_and_outside_activity(
    tmp_path: Path,
) -> None:
    values = ["S"] + [str(value) for value in range(1, 40)]
    text = "10 (a tricky (process) name) " + " ".join(values)
    assert process_stat(text) == {
        "parent": 1,
        "minor_faults": 7,
        "major_faults": 9,
        "user_ticks": 11,
        "system_ticks": 12,
        "start_ticks": 19,
    }
    for pid, parent, pss in ((10, 1, 50), (11, 10, 30), (12, 11, 20), (13, 1, 70)):
        directory = tmp_path / str(pid)
        directory.mkdir()
        values[1] = str(parent)
        (directory / "stat").write_text(f"{pid} (fixture) " + " ".join(values))
        (directory / "smaps_rollup").write_text(
            f"Pss: {pss} kB\nRss: 100 kB\nSwap: 2 kB\nSwapPss: 1 kB\n"
        )
    (tmp_path / "meminfo").write_text(
        "MemAvailable: 1000 kB\nSwapTotal: 400 kB\nSwapFree: 300 kB\n"
    )
    (tmp_path / "vmstat").write_text("pswpin 3\npswpout 4\npgmajfault 5\n")
    measured = sample_process_tree(10, tmp_path)
    assert set(measured["processes"]) == {"10", "11", "12"}
    assert measured["pss_bytes"] == 100 * 1024
    assert measured["swap_bytes"] == 6 * 1024
    assert set(measured["outside_process_ticks"]) == {"13"}
    assert measured["host_swap_used_bytes"] == 100 * 1024
    assert measured["host_swap_in_pages"] == 3 and measured["host_swap_out_pages"] == 4
