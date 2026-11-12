from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import digest_file, verify_probe
from tools.process import stop_session
from tools.runtime_verify import write_record


def owner() -> dict[str, str]:
    process = Path("/proc/self").resolve()
    return {
        "pid": process.name,
        "start_ticks": (process / "stat").read_text().rsplit(")", 1)[1].split()[19],
    }


def owner_alive(identity: dict[str, str]) -> bool:
    try:
        fields = (
            (Path("/proc") / identity["pid"] / "stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()
        )
        return fields[19] == identity["start_ticks"] and fields[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def start_invocation(session: Path, remaining: float) -> None:
    write_record(
        session / "invocation.json",
        {
            "owner": owner(),
            "started_unix": time.time(),
            "remaining_seconds": remaining,
        },
    )


def recover_sessions(output: Path) -> None:
    for session in sorted(
        (output / "sessions").glob("*"), key=lambda path: int(path.name)
    ):
        history = output / f"terminal-{int(session.name):03d}.json"
        if history.exists():
            continue
        invocation = json.loads((session / "invocation.json").read_text())
        if owner_alive(invocation["owner"]):
            raise ValueError("campaign invocation is still running; resume refused")
        panels = []
        interruptions = []
        charged = 0.0
        for launch_path in sorted(session.glob("launch-*.json")):
            launch = json.loads(launch_path.read_text())
            attempt = output / launch["directory"]
            if not attempt.resolve().is_relative_to(session.resolve()):
                raise ValueError("interrupted panel escapes its invocation")
            if (attempt / "terminal.json").exists():
                receipt = verify_probe(attempt, allow_failed_panel=True)
                charged += receipt["elapsed_seconds"]
                panels.append(
                    {
                        "directory": launch["directory"],
                        "identities": launch["identities"],
                        "exit_code": None,
                    }
                )
            else:
                charged += launch["budget_seconds"]
                if (attempt / "runtime-owner.json").exists():
                    runtime = json.loads((attempt / "runtime-owner.json").read_text())
                    if runtime["identity"] is not None:
                        boot, born, group, process_session = runtime["identity"]
                        stop_session(
                            runtime["pid"], (boot, born, group, process_session)
                        )
                interruptions.append(
                    {
                        "launch": str(launch_path.relative_to(output)),
                        "launch_sha256": digest_file(launch_path),
                        "identities": launch["identities"],
                        "reason": "owner exited before a complete receipt; all admitted cells remain interrupted, never retried",
                    }
                )
        elapsed = min(
            invocation["remaining_seconds"],
            max(charged, time.time() - invocation["started_unix"], 0),
        )
        write_record(
            history,
            {
                "schema": "benchmark-campaign-terminal-v3",
                "admission_sha256": digest_file(output / "admission.json"),
                "elapsed_seconds": elapsed,
                "panels": panels,
                "interruptions": interruptions,
                "recovery": {
                    "invocation_sha256": digest_file(session / "invocation.json"),
                    "budget_policy": "interrupted panels charge their complete admitted budgets; total also includes wall time, bounded by remaining campaign allowance",
                },
            },
        )


def interrupted_results(output: Path, terminal: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for interruption in terminal.get("interruptions", []):
        launch = output / interruption["launch"]
        if (
            not launch.resolve().is_relative_to(output.resolve())
            or digest_file(launch) != interruption["launch_sha256"]
        ):
            raise ValueError("interrupted launch receipt drift")
        if json.loads(launch.read_text())["identities"] != interruption["identities"]:
            raise ValueError("interrupted denominators differ")
        results.extend(
            {
                "identity": identity,
                "status": "failed",
                "quality": "ungraded",
                "reason": interruption["reason"],
            }
            for identity in interruption["identities"]
        )
    return results
