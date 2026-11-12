from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import probe, verify_probe


def prepare_load(
    root: Path,
    python: Path,
    harness: Path,
    models: dict[str, Path],
    jobs: list[dict[str, Any]],
    output: Path,
) -> dict[str, dict[str, float]]:
    required = sorted(
        {job["model"] for job in jobs if job["case"] in ("T3", "R1", "A1", "S1")}
    )
    rates: dict[str, dict[str, float]] = {}
    for model in required:
        destination = output / "preparation" / model
        job = {
            "identity": "calibration/" + model,
            "kind": "trace",
            "case": "calibration",
            "seconds": 120,
        }
        status = probe(
            root,
            python,
            models[model],
            destination,
            600,
            "native-arc",
            panel=(harness, [job]),
        )
        if status != 0:
            raise ValueError(
                "native-only load calibration failed; preserved preparation cannot qualify a campaign"
            )
        verify_probe(destination)
        record = json.loads((destination / "panel/0/terminal.json").read_text())
        rates[model] = record["load_intervals"]
    return rates


def verify_load(output: Path, admission: dict[str, Any]) -> float:
    elapsed = 0.0
    for model, rates in admission.get("load_intervals", {}).items():
        destination = output / "preparation" / model
        result = verify_probe(destination)
        child = json.loads((destination / "admission.json").read_text())
        command = child["command"]
        actual = json.loads((destination / "panel/0/terminal.json").read_text())
        if (
            child["system"] != "native-arc"
            or child["source_sha256"] != admission["source_sha256"]
            or command[command.index("--model") + 1] != admission["models"][model]
            or rates != actual["load_intervals"]
        ):
            raise ValueError("load calibration differs from campaign admission")
        elapsed += result["elapsed_seconds"]
    for job in admission["jobs"]:
        if (
            job["case"] in ("T3", "R1", "A1", "S1")
            and job["load_intervals"] != admission["load_intervals"][job["model"]]
        ):
            raise ValueError("comparator received a different offered load")
    return elapsed
