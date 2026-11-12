from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.benchmarking.campaign_recovery import (
    interrupted_results,
    recover_sessions,
    start_invocation,
)
from tools.benchmarking.exposure import admit_exposure
from tools.benchmarking.harness import pin_swebench_image
from tools.benchmarking.load_calibration import prepare_load, verify_load
from tools.benchmarking.panel_report import cell_observations, paired_observations
from tools.benchmarking.plan import cells, describe
from tools.benchmarking.probe import digest_file, probe, verify_probe
from tools.benchmarking.qualification import prepare_functional, verify_functional
from tools.benchmarking.release_report import read_policy, release_gates
from tools.config import output_overwrites_source
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
            if cell.case == "V3" or (cell.system == "lmcache" and cell.case == "B1-1"):
                job["diagnostic_seconds"] = 180
                job["pressure_prefixes"] = 24 if "gemma" in cell.model else 48
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


def panel_results(
    child: dict[str, Any], measured: dict[str, Any]
) -> list[dict[str, Any]]:
    if "panel" in measured:
        records: list[dict[str, Any]] = measured["panel"]["results"]
        return records
    return [
        {
            "identity": job["identity"],
            "status": "failed",
            "quality": "ungraded",
            "reason": "serving/restore qualification failed before application admission: "
            + str(measured.get("reason")),
        }
        for job in child["panel"]["jobs"]
    ]


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
    if output_overwrites_source(root, output):
        raise ValueError("campaign requires an artifact output")
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
        claim_policy = read_policy(root)
        output.mkdir(parents=True, exist_ok=False)
        sources = snapshot(root, output / "source")
        exposure = (
            admit_exposure(root, assets, output, jobs)
            if (root / "benchmarks/workloads.json").is_file()
            else None
        )
        functional_preparation = profile == "release" and not selected_cases
        if functional_preparation:
            prepare_functional(root, python, models["dense-fixture"], output)
        load_intervals = prepare_load(
            root, python, harness_python, models, jobs, output
        )
        for job in jobs:
            if job["case"] in ("T3", "R1", "A1", "S1"):
                job["load_intervals"] = load_intervals[job["model"]]
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
                "load_intervals": load_intervals,
                "functional_preparation": functional_preparation,
                "exposure": exposure,
                "claim_policy": claim_policy,
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
    recover_sessions(output)
    histories = sorted(output.glob("terminal-*.json"))
    prior = [json.loads(path.read_text()) for path in histories]
    consumed = sum(record["elapsed_seconds"] for record in prior)
    finished: set[str] = set()
    for record in prior:
        finished.update(
            item["identity"] for item in interrupted_results(output, record)
        )
        for panel in record["panels"]:
            measured = verify_probe(
                output / panel["directory"], allow_failed_panel=True
            )
            child = json.loads(
                (output / panel["directory"] / "admission.json").read_text()
            )
            finished.update(
                item["identity"]
                for item in panel_results(child, measured)
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
    start_invocation(session, max(0, admission["budget_seconds"] - consumed))
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
            budget = min(
                int(remaining),
                sum(
                    job["seconds"]
                    + job.get("diagnostic_seconds", 0)
                    + (
                        360
                        if job.get("task_kind") == "branching"
                        else 180
                        if job.get("kind") == "application"
                        else 0
                    )
                    for job in pending
                )
                + 600,
            )
            if attempt.exists():
                raise ValueError(
                    "resume never overwrites or retries an existing attempt"
                )
            write_record(
                session / f"launch-{index:03d}.json",
                {
                    "directory": str(attempt.relative_to(output)),
                    "identities": [job["identity"] for job in pending],
                    "budget_seconds": budget,
                },
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
        recorded = {panel["directory"] for panel in panels}
        interruptions = []
        for launch_path in session.glob("launch-*.json"):
            launch = json.loads(launch_path.read_text())
            if launch["directory"] not in recorded:
                interruptions.append(
                    {
                        "launch": str(launch_path.relative_to(output)),
                        "launch_sha256": digest_file(launch_path),
                        "identities": launch["identities"],
                        "reason": "probe aborted before a complete campaign receipt; admitted cells are never retried",
                    }
                )
        write_record(
            output / f"terminal-{number:03d}.json",
            {
                "schema": "benchmark-campaign-terminal-v3",
                "admission_sha256": digest_file(output / "admission.json"),
                "elapsed_seconds": time.monotonic() - started,
                "panels": panels,
                "interruptions": interruptions,
            },
        )
    result = verify_panels(output)
    write_record(session / "report.json", result)
    return int(result["measurement"] != "complete")


def verify_panels(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    if admission["schema"] != "benchmark-campaign-v3":
        raise ValueError("unsupported panel campaign")
    preparation_seconds = verify_load(output, admission)
    if admission.get("claim_policy") != read_policy(output / "source"):
        raise ValueError("campaign claim policy differs from frozen source")
    if (
        admission.get("exposure")
        and digest_file(output / "exposure.json") != admission["exposure"]["sha256"]
    ):
        raise ValueError("campaign exposure receipt drift")
    for name, expected in admission["source_sha256"].items():
        if digest_file(output / "source" / name) != expected:
            raise ValueError("campaign source snapshot drift")
    results: dict[str, Any] = {}
    observations: dict[str, dict[str, Any]] = {}
    elapsed = 0.0
    for history in sorted(output.glob("terminal-*.json")):
        terminal = json.loads(history.read_text())
        if terminal["admission_sha256"] != digest_file(output / "admission.json"):
            raise ValueError("campaign admission drift")
        elapsed += terminal["elapsed_seconds"]
        for record in interrupted_results(output, terminal):
            if (
                record["identity"] in results
                and results[record["identity"]]["status"] != "not_run"
            ):
                raise ValueError("campaign repeated an interrupted cell")
            results[record["identity"]] = record
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
            actual = panel_results(child, measured)
            if [record["identity"] for record in actual] != panel["identities"]:
                raise ValueError("panel denominators differ")
            for record in actual:
                if (
                    record["identity"] in results
                    and results[record["identity"]]["status"] != "not_run"
                ):
                    raise ValueError("campaign acquired the same cell twice")
                results[record["identity"]] = record
                job = planned[record["identity"]]
                if "kind" in job:
                    observations[record["identity"]] = cell_observations(
                        directory / "panel", job, record
                    )
    expected_ids = {job["identity"] for job in admission["jobs"]}
    if not results.keys() <= expected_ids:
        raise ValueError("campaign contains an unplanned cell")
    complete = results.keys() == expected_ids and all(
        result["status"] == "complete" for result in results.values()
    )
    pairs = paired_observations(observations)
    functional = verify_functional(output, admission)
    verdict = release_gates(
        admission, observations, pairs, complete, elapsed, functional
    )
    preparation_seconds += functional.get("elapsed_seconds", 0)
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
        "preparation_seconds": preparation_seconds,
        "within_budget": elapsed <= admission["budget_seconds"],
        **verdict,
        "limits": [
            "acquisition completeness does not qualify operating or benefit claims",
            "fresh confirmation, usefulness floors, paired quality and operating gates remain explicit prerequisites",
            "failed and censored tasks remain in their offered-work denominators",
        ],
        "results": results,
        "observations": observations,
        "paired_observations": pairs,
    }
