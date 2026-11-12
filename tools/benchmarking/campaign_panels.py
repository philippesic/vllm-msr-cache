from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.benchmarking.harness import pin_swebench_image
from tools.benchmarking.plan import cells, describe
from tools.benchmarking.probe import digest_file, probe, verify_probe
from tools.runtime_verify import snapshot, write_record


def admitted_jobs(root: Path, profile: str, assets: Path) -> list[dict[str, Any]]:
    definitions = json.loads((root / "benchmarks/workloads.json").read_text())
    cases = {case["id"]: case for case in definitions["cases"]}
    result = []
    for cell in cells(profile):
        job = {**asdict(cell), "identity": cell.identity}
        if cell.kind == "application":
            if cell.case not in cases:
                raise ValueError(f"campaign case not implemented/admitted: {cell.case}")
            definition = cases[cell.case]
            source = definition.get("dataset", "swebench-c1")
            dataset = assets / source / "test.json"
            expected = (
                definitions["dataset_sha256"]
                if source == "swebench-c1"
                else definitions["additional_datasets"][source]["dataset_sha256"]
            )
            if not dataset.is_file() or digest_file(dataset) != expected:
                raise ValueError(f"missing or changed admitted dataset: {source}")
            rows = [
                row
                for row in json.loads(dataset.read_text())
                if row["instance_id"] == definition["instance"]
            ]
            if len(rows) != 1:
                raise ValueError("campaign source task missing or duplicated")
            kind = definition.get("task_kind", "repair")
            image = (
                {"id": rows[0]["image_name"]}
                if kind in ("terminal", "navigation", "grounded")
                else pin_swebench_image(definition["instance"], definition["image_id"])
            )
            if image["id"] != definition["image_id"]:
                raise ValueError("campaign image differs from frozen task")
            job.update(
                dataset=str(dataset.resolve()),
                dataset_sha256=expected,
                instance=definition["instance"],
                task_kind=kind,
                image=image,
                exposure=definition["exposure"],
            )
            if cell.case == "V3":
                job["diagnostic_seconds"] = 180
                job["pressure_prefixes"] = 48
        elif cell.kind in ("trace", "soak"):
            if cell.kind == "soak":
                job["kind"] = "trace"
            job["pressure_prefixes"] = 24 if "gemma" in cell.model else 48
            job["reference"] = (
                "frozen advice-disabled native ARC; no previous-release claim"
                if cell.system == "previous-or-base"
                else "current implementation"
            )
        else:
            raise ValueError("unsupported panel campaign kind")
        result.append(job)
    return result


def serving_system(system: str) -> str:
    if system in (
        "previous-or-base",
        "no-registration",
        "registration-only",
        "injected-delay",
    ):
        return "native-arc"
    return system


def run_panels(
    root: Path,
    profile: str,
    models: dict[str, Path],
    python: Path,
    lmcache_python: Path,
    harness_python: Path,
    assets: Path,
    output: Path,
    selected_cases: list[str],
    resume: bool = False,
) -> int:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("campaign requires an external output")
    if not resume:
        jobs = admitted_jobs(root, profile, assets)
        if selected_cases:
            unknown = set(selected_cases) - {job["case"] for job in jobs}
            if unknown:
                raise ValueError(f"unknown campaign cases: {sorted(unknown)}")
            jobs = [job for job in jobs if job["case"] in selected_cases]
        required_models = {job["model"] for job in jobs}
        if any(
            name not in models or not models[name].is_dir() for name in required_models
        ):
            raise ValueError(
                "campaign requires each admitted local model, including the dense sentinel"
            )
        if not all(path.is_file() for path in (python, lmcache_python, harness_python)):
            raise ValueError("campaign requires all three existing runtimes")
        output.mkdir(parents=True, exist_ok=False)
        sources = snapshot(root, output / "source")
        write_record(
            output / "admission.json",
            {
                "schema": "benchmark-campaign-v3",
                "profile": profile,
                "partial": bool(selected_cases),
                "jobs": jobs,
                "models": {name: str(path.resolve()) for name, path in models.items()},
                "runtimes": {
                    "core": str(python.absolute()),
                    "lmcache": str(lmcache_python.absolute()),
                    "harness": str(harness_python.absolute()),
                },
                "source_sha256": sources,
                "budget_seconds": describe(profile)["budget_seconds"],
                "qualification": "engineering acquisition; exposure and missing operating gates block release claims",
            },
        )
    admission = json.loads((output / "admission.json").read_text())
    if (
        admission["schema"] != "benchmark-campaign-v3"
        or admission["profile"] != profile
    ):
        raise ValueError("resume requires a matching v3 profile")
    if any(
        digest_file(root / path) != value
        for path, value in admission["source_sha256"].items()
    ):
        raise ValueError(
            "resume source differs; retain this campaign and create a new one"
        )
    for key, path in admission["runtimes"].items():
        current = {
            "core": python,
            "lmcache": lmcache_python,
            "harness": harness_python,
        }[key]
        if str(current.absolute()) != path:
            raise ValueError("resume runtime path differs")
    if {name: str(path.resolve()) for name, path in models.items()} != admission[
        "models"
    ]:
        raise ValueError("resume model paths differ")
    jobs = admission["jobs"]
    histories = sorted(output.glob("terminal-*.json"))
    prior = [json.loads(path.read_text()) for path in histories]
    consumed = sum(record["elapsed_seconds"] for record in prior)
    finished: set[str] = set()
    for record in prior:
        for panel in record["panels"]:
            measured = verify_probe(
                output / panel["directory"], allow_failed_panel=True
            )
            finished.update(
                item["identity"]
                for item in measured["panel"]["results"]
                if item["status"] != "not_run"
            )
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        if job["identity"] not in finished:
            groups[(job["model"], serving_system(job["system"]), job["block"])].append(
                job
            )
    number = len(histories)
    session = output / "sessions" / str(number)
    session.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    panels: list[dict[str, Any]] = []
    try:
        for index, ((model, system, _block), pending) in enumerate(groups.items()):
            remaining = (
                admission["budget_seconds"] - consumed - (time.monotonic() - started)
            )
            if remaining < 420:
                break
            attempt = session / str(index)
            budget = min(int(remaining), sum(job["seconds"] for job in pending) + 600)
            if attempt.exists():
                raise ValueError(
                    "resume never overwrites or retries an existing attempt"
                )
            status = probe(
                root,
                lmcache_python if system == "lmcache" else python,
                models[model],
                attempt,
                budget,
                system,
                panel=(harness_python, pending),
            )
            panels.append(
                {
                    "directory": str(attempt.relative_to(output)),
                    "identities": [job["identity"] for job in pending],
                    "exit_code": status,
                }
            )
            write_record(session / f"progress-{index}.json", panels[-1])
    finally:
        write_record(
            output / f"terminal-{number:03d}.json",
            {
                "schema": "benchmark-campaign-terminal-v3",
                "admission_sha256": digest_file(output / "admission.json"),
                "elapsed_seconds": time.monotonic() - started,
                "panels": panels,
            },
        )
    result = verify_panels(output)
    write_record(session / "report.json", result)
    return int(result["measurement"] != "complete")


def verify_panels(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    if admission["schema"] != "benchmark-campaign-v3":
        raise ValueError("unsupported panel campaign")
    for name, expected in admission["source_sha256"].items():
        if digest_file(output / "source" / name) != expected:
            raise ValueError("campaign source snapshot drift")
    results: dict[str, Any] = {}
    elapsed = 0.0
    for history in sorted(output.glob("terminal-*.json")):
        terminal = json.loads(history.read_text())
        if terminal["admission_sha256"] != digest_file(output / "admission.json"):
            raise ValueError("campaign admission drift")
        elapsed += terminal["elapsed_seconds"]
        for panel in terminal["panels"]:
            directory = (output / panel["directory"]).resolve()
            if not directory.is_relative_to(output.resolve()):
                raise ValueError("campaign panel escapes its receipt")
            measured = verify_probe(directory, allow_failed_panel=True)
            child = json.loads((directory / "admission.json").read_text())
            if child["source_sha256"] != admission["source_sha256"]:
                raise ValueError("panel executed a different source")
            planned = {job["identity"]: job for job in admission["jobs"]}
            if any(
                job != planned.get(job["identity"]) for job in child["panel"]["jobs"]
            ):
                raise ValueError("panel job differs from campaign admission")
            for job in child["panel"]["jobs"]:
                if child["system"] != serving_system(job["system"]):
                    raise ValueError("panel comparator differs from campaign admission")
                command = child["command"]
                if (
                    command[command.index("--model") + 1]
                    != admission["models"][job["model"]]
                ):
                    raise ValueError("panel model differs from campaign admission")
            actual = measured["panel"]["results"]
            if [record["identity"] for record in actual] != panel["identities"]:
                raise ValueError("panel denominators differ")
            for record in actual:
                if (
                    record["identity"] in results
                    and results[record["identity"]]["status"] != "not_run"
                ):
                    raise ValueError("campaign acquired the same cell twice")
                results[record["identity"]] = record
    expected_ids = {job["identity"] for job in admission["jobs"]}
    if not results.keys() <= expected_ids:
        raise ValueError("campaign contains an unplanned cell")
    complete = results.keys() == expected_ids and all(
        result["status"] == "complete" for result in results.values()
    )
    return {
        "schema": "benchmark-campaign-report-v3",
        "profile": admission["profile"],
        "partial": admission["partial"],
        "measurement": "complete" if complete else "incomplete",
        "planned": len(expected_ids),
        "attempted": len(results),
        "completed": sum(result["status"] == "complete" for result in results.values()),
        "failed": sum(result["status"] == "failed" for result in results.values()),
        "not_run": sorted(expected_ids - results.keys())
        + [
            identity
            for identity, record in results.items()
            if record["status"] == "not_run"
        ],
        "elapsed_seconds": elapsed,
        "within_budget": elapsed <= admission["budget_seconds"],
        "release": "unresolved",
        "benefit": "unresolved",
        "limits": [
            "acquisition completeness does not qualify operating or benefit claims",
            "fresh confirmation, usefulness floors, paired quality and operating gates remain explicit prerequisites",
            "failed and censored tasks remain in their offered-work denominators",
        ],
        "results": results,
    }
