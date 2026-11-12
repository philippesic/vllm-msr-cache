from __future__ import annotations

import json
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
