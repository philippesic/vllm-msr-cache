from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.benchmarking.harness import pin_swebench_image
from tools.benchmarking.plan import cells, describe
from tools.benchmarking.probe import digest_file, probe, verify_probe
from tools.repo_policy import git_paths
from tools.runtime_verify import snapshot, write_record


def run_campaign(
    root: Path,
    profile: str,
    models: dict[str, Path],
    python: Path,
    lmcache_python: Path,
    harness_python: Path,
    dataset: Path,
    output: Path,
    selected_cases: list[str],
) -> int:
    roster = cells(profile)
    if selected_cases:
        unknown = set(selected_cases) - {cell.case for cell in roster}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        roster = tuple(cell for cell in roster if cell.case in selected_cases)
    definitions = json.loads((root / "benchmarks/workloads.json").read_text())
    available = {case["id"]: case for case in definitions["cases"]}
    unsupported = sorted(
        {
            cell.case
            for cell in roster
            if cell.kind != "application" or cell.case not in available
        }
    )
    if unsupported:
        raise ValueError(f"campaign cases not implemented/admitted: {unsupported}")
    if output.resolve().is_relative_to(root.resolve()) or output.exists():
        raise ValueError("campaign requires a new external output")
    if any(
        cell.model not in models or not models[cell.model].is_dir() for cell in roster
    ):
        raise ValueError("campaign requires each selected local model")
    if not all(
        path.is_file() for path in (python, lmcache_python, harness_python, dataset)
    ):
        raise ValueError("campaign requires existing runtimes and dataset")
    if digest_file(dataset) != definitions["dataset_sha256"]:
        raise ValueError("campaign dataset differs from admitted source")
    rows = json.loads(dataset.read_text())
    for cell in roster:
        if (
            sum(row["instance_id"] == available[cell.case]["instance"] for row in rows)
            != 1
        ):
            raise ValueError("campaign instance missing or duplicated")
    selected = {cell.case: available[cell.case] for cell in roster}
    images = {
        case["instance"]: pin_swebench_image(case["instance"], case.get("image_id"))
        for case in selected.values()
    }
    output.mkdir(parents=True)
    sources = snapshot(root, output / "source")
    budget = describe(profile)["budget_seconds"]
    admission = {
        "schema": "benchmark-campaign-v2",
        "profile": profile,
        "partial": bool(selected_cases),
        "selected_cases": selected_cases,
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "budget_seconds": budget,
        "source_sha256": sources,
        "dataset_sha256": digest_file(dataset),
        "images": images,
        "cells": [cell.identity for cell in roster],
    }
    write_record(output / "admission.json", admission)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    try:
        for index, cell in enumerate(roster):
            remaining = budget - (time.monotonic() - started)
            if remaining < 420:
                results.append(
                    {
                        "cell": cell.identity,
                        "status": "not_run",
                        "reason": "campaign budget",
                    }
                )
                continue
            attempt = output / "cells" / str(index)
            attempt.parent.mkdir(exist_ok=True)
            current = {
                str(path): digest_file(root / path)
                for path in git_paths(root)
                if (root / path).is_file()
            }
            if current != sources:
                raise ValueError("campaign source changed during acquisition")
            instance = available[cell.case]["instance"]
            pin_swebench_image(instance, images[instance]["id"])
            status = probe(
                root,
                lmcache_python if cell.system == "lmcache" else python,
                models[cell.model],
                attempt,
                min(900, int(remaining)),
                cell.system,
                coding=(harness_python, dataset, instance),
            )
            child_admission = json.loads((attempt / "admission.json").read_text())
            if child_admission["source_sha256"] != sources:
                raise ValueError("campaign cell used different source")
            results.append(
                {
                    "cell": cell.identity,
                    "status": "complete" if status == 0 else "failed",
                    "attempt": str(attempt.relative_to(output)),
                }
            )
            write_record(
                output / f"progress-{index}.json",
                {"schema": "benchmark-campaign-progress-v1", "results": results},
            )
    finally:
        write_record(
            output / "terminal.json",
            {
                "schema": "benchmark-campaign-terminal-v2",
                "admission_sha256": digest_file(output / "admission.json"),
                "elapsed_seconds": time.monotonic() - started,
                "results": results,
                "status": "partial"
                if selected_cases
                else "complete"
                if len(results) == len(roster)
                and all(result["status"] == "complete" for result in results)
                else "incomplete",
            },
        )
    return (
        0
        if len(results) == len(roster)
        and all(result["status"] == "complete" for result in results)
        else 1
    )


def verify_campaign(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if admission["schema"] not in (
        "benchmark-campaign-v1",
        "benchmark-campaign-v2",
    ) or terminal["schema"] != admission["schema"].replace(
        "campaign-", "campaign-terminal-"
    ):
        raise ValueError("unsupported campaign receipt")
    if terminal["admission_sha256"] != digest_file(output / "admission.json"):
        raise ValueError("campaign admission drift")
    for relative, digest in admission["source_sha256"].items():
        path = (output / "source" / relative).resolve()
        if (
            not path.is_relative_to((output / "source").resolve())
            or digest_file(path) != digest
        ):
            raise ValueError("campaign source drift")
    expected = [
        cell.identity
        for cell in cells(admission["profile"])
        if not admission["selected_cases"] or cell.case in admission["selected_cases"]
    ]
    if admission["cells"] != expected or admission["partial"] != bool(
        admission["selected_cases"]
    ):
        raise ValueError("campaign does not match requested schedule")
    if [result["cell"] for result in terminal["results"]] != admission["cells"]:
        raise ValueError("campaign coverage incomplete")
    if terminal["elapsed_seconds"] > admission["budget_seconds"] + 60:
        raise ValueError("campaign exceeded budget")
    for result in terminal["results"]:
        if result["status"] != "complete":
            raise ValueError(f"campaign cell incomplete: {result['cell']}")
        attempt = (output / result["attempt"]).resolve()
        if not attempt.is_relative_to(output.resolve()):
            raise ValueError("campaign cell path escapes output")
        observed = verify_probe(attempt)
        child = json.loads((attempt / "admission.json").read_text())
        _, model, case, system, _ = result["cell"].split("/")
        definitions = json.loads(
            (output / "source/benchmarks/workloads.json").read_text()
        )
        instance = next(
            item["instance"] for item in definitions["cases"] if item["id"] == case
        )
        if (
            child["system"] != system
            or child["coding"]["instance"] != instance
            or child["coding"]["dataset_sha256"] != admission["dataset_sha256"]
        ):
            raise ValueError("campaign cell identity differs")
        if (
            admission["schema"] == "benchmark-campaign-v2"
            and observed["coding"]["image"] != admission["images"][instance]
        ):
            raise ValueError("campaign cell task image differs")
        model_path = admission["models"][model]
        command = child["command"]
        if command[command.index("--model") + 1] != model_path:
            raise ValueError("campaign model differs")
    if bool(admission["partial"]) != (terminal["status"] == "partial"):
        raise ValueError("campaign partial/full status differs")
    return terminal
