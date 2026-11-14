from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def process_memory(session: int) -> dict[str, int]:
    retained: dict[str, int] = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) != session:
                continue
            status = (directory / "status").read_text()
            resident = next(
                line for line in status.splitlines() if line.startswith("VmRSS:")
            )
            retained[directory.name] = int(resident.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
    return retained


def sample_resources(session: int) -> dict[str, Any]:
    processes = process_memory(session)
    error = None
    devices: list[list[str]] = []
    try:
        device = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        devices = [line.strip().split(", ") for line in device.stdout.splitlines()]
    except (OSError, subprocess.SubprocessError) as failure:
        error = str(failure)
    return {
        "monotonic_ns": time.monotonic_ns(),
        "session": session,
        "process_rss_bytes": processes,
        "sum_rss_bytes": sum(processes.values()),
        "devices": devices,
        "error": error,
        "complete": bool(processes) and bool(devices) and error is None,
        "limits": "summed RSS double-counts shared pages; device memory includes all processes; these are not live tensor or metadata counts",
    }


def verify_resources(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text())
    if (
        type(record["monotonic_ns"]) is not int
        or record["monotonic_ns"] <= 0
        or record["complete"]
        != (
            bool(record["process_rss_bytes"])
            and bool(record["devices"])
            and record["error"] is None
        )
        or any(
            type(value) is not int or value < 0
            for value in record["process_rss_bytes"].values()
        )
        or record["sum_rss_bytes"] != sum(record["process_rss_bytes"].values())
    ):
        raise ValueError("invalid resource observations")
    return record


def _kilobytes(path: Path) -> dict[str, int]:
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == "kB":
            values[fields[0].rstrip(":")] = int(fields[1]) * 1024
    return values


def process_stat(text: str) -> dict[str, int]:
    fields = text[text.rindex(")") + 2 :].split()
    return {
        "parent": int(fields[1]),
        "minor_faults": int(fields[7]),
        "major_faults": int(fields[9]),
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "start_ticks": int(fields[19]),
    }


def sample_process_tree(root_pid: int, proc: Path = Path("/proc")) -> dict[str, Any]:
    started = time.monotonic_ns()
    stats: dict[int, dict[str, int]] = {}
    for directory in proc.iterdir():
        if directory.name.isdecimal():
            try:
                stats[int(directory.name)] = process_stat(
                    (directory / "stat").read_text()
                )
            except (FileNotFoundError, ProcessLookupError):
                continue
    descendants = {root_pid}
    while True:
        expanded = descendants | {
            pid for pid, stat in stats.items() if stat["parent"] in descendants
        }
        if expanded == descendants:
            break
        descendants = expanded
    processes: dict[str, dict[str, int]] = {}
    vanished = []
    for pid in sorted(descendants & stats.keys()):
        try:
            memory = _kilobytes(proc / str(pid) / "smaps_rollup")
            processes[str(pid)] = stats[pid] | {
                "pss_bytes": memory["Pss"],
                "rss_bytes": memory["Rss"],
                "swap_bytes": memory["Swap"],
                "swap_pss_bytes": memory["SwapPss"],
            }
        except (FileNotFoundError, ProcessLookupError):
            vanished.append(pid)
    memory = _kilobytes(proc / "meminfo")
    vm = {
        parts[0]: int(parts[1])
        for line in (proc / "vmstat").read_text().splitlines()
        if len(parts := line.split()) == 2
    }
    return {
        "schema": "pr-resources-v1",
        "monotonic_ns": started,
        "sample_duration_ns": time.monotonic_ns() - started,
        "root_pid": root_pid,
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
        "processes": processes,
        "vanished_during_sample": vanished,
        "outside_process_ticks": {
            str(pid): {
                "start_ticks": stat["start_ticks"],
                "cpu_ticks": stat["user_ticks"] + stat["system_ticks"],
            }
            for pid, stat in stats.items()
            if pid not in descendants
        },
        "pss_bytes": sum(value["pss_bytes"] for value in processes.values()),
        "swap_bytes": sum(value["swap_bytes"] for value in processes.values()),
        "available_bytes": memory["MemAvailable"],
        "host_swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
        "host_swap_in_pages": vm["pswpin"],
        "host_swap_out_pages": vm["pswpout"],
        "host_major_faults": vm["pgmajfault"],
    }
