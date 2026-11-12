from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.config import benchmark_environment
from tools.process import run_owned
from tools.runtime_verify import verify


def prepare_functional(root: Path, python: Path, model: Path, output: Path) -> None:
    destination = output / "qualification"
    destination.mkdir()
    for mode, seconds in (("runtime-cpu", 180), ("hardware", 600)):
        command = [
            str(python.absolute()),
            "-m",
            "tools.runtime_verify",
            "--out",
            str(destination / mode),
            "--timeout",
            str(seconds),
        ]
        if mode == "hardware":
            command.extend(["--gpu", "--model", str(model)])
        with (destination / (mode + ".log")).open("x") as log:
            status = run_owned(
                command,
                benchmark_environment(root, model, True),
                root,
                seconds + 60,
                log,
            )
        if status != 0:
            raise ValueError(
                "release functional preparation failed; retained proof/log explains the failure"
            )
        verify(destination / mode)


def verify_functional(output: Path, admission: dict[str, Any]) -> dict[str, Any]:
    if not admission.get("functional_preparation"):
        return {
            "status": "unresolved",
            "reason": "no mandatory functional/runtime proofs admitted",
        }
    elapsed = 0.0
    for mode in ("runtime-cpu", "hardware"):
        directory = output / "qualification" / mode
        result = verify(directory)
        child = json.loads((directory / "admission.json").read_text())
        if (
            child["source_sha256"] != admission["source_sha256"]
            or child["mode"] != mode
        ):
            raise ValueError("functional proof differs from campaign product snapshot")
        elapsed += result["elapsed_seconds"]
    return {
        "status": "passed",
        "elapsed_seconds": elapsed,
        "reason": "pinned CPU integration and mandatory real GPU functional/fault assertions passed; finite tests do not prove zero future failures",
    }
