from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.benchmarking.branching import run_branching, verify_branching
from tools.benchmarking.grounded import verify_grounded
from tools.benchmarking.harness import run_swebench, verify_coding_grade
from tools.benchmarking.navigation import verify_navigation
from tools.benchmarking.probe import digest_file, request
from tools.benchmarking.proxy import local_proxy
from tools.benchmarking.regression import verify_regression
from tools.benchmarking.terminal import run_terminal, verify_terminal
from tools.benchmarking.traces import acquire_trace, verify_trace
from tools.runtime_verify import write_record


def acquire_panel(
    root: Path,
    python: Path,
    jobs: list[dict[str, Any]],
    base: str,
    namespace: dict[str, str],
    system: str,
    output: Path,
    deadline: float,
    cpu_bytes: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-panel-v1",
            "jobs": jobs,
            "system": system,
            "state_policy": "full GPU/external reset before every application or diagnostic; persistent serving process",
            "deadline_monotonic": deadline,
            "cpu_bytes": cpu_bytes,
        },
    )
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    try:
        for index, job in enumerate(jobs):
            result: dict[str, Any] = {
                "identity": job["identity"],
                "status": "not_run",
                "quality": "ungraded",
                "reason": None,
            }
            results.append(result)
            destination = output / str(index)
            if deadline - time.monotonic() < job["seconds"] + 180:
                result["reason"] = (
                    "panel budget exhausted before complete task admission"
                )
                continue
            try:
                if not request(base + "/reset_prefix_cache?reset_external=true", {})[
                    "success"
                ]:
                    raise ValueError("panel reset refused")
                if job["kind"] == "trace":
                    measured = acquire_trace(
                        base,
                        namespace,
                        system,
                        job["case"],
                        destination,
                        job["seconds"],
                        cpu_bytes,
                        job.get("pressure_prefixes", 24),
                    )
                    result["status"] = "complete" if measured["complete"] else "failed"
                elif job["kind"] == "application":
                    dataset = Path(job["dataset"])
                    if digest_file(dataset) != job["dataset_sha256"]:
                        raise ValueError("panel application dataset drift")
                    destination.mkdir()
                    kind = job["task_kind"]
                    with local_proxy(
                        base,
                        system,
                        namespace,
                        destination / "requests",
                        "terminus-json" if kind == "terminal" else "mini-swe",
                    ) as endpoint:
                        if kind == "terminal":
                            measured = run_terminal(
                                root,
                                python,
                                dataset,
                                job["instance"],
                                endpoint,
                                destination / "task",
                                job["seconds"],
                            )
                        elif kind == "branching":
                            measured = run_branching(
                                root,
                                python,
                                dataset,
                                job["instance"],
                                endpoint,
                                destination / "task",
                                job["seconds"],
                            )
                        else:
                            measured = run_swebench(
                                root,
                                python,
                                dataset,
                                job["instance"],
                                endpoint,
                                destination / "task",
                                job["seconds"],
                                kind,
                            )
                    result["quality"] = measured["quality"]
                    result["status"] = (
                        "complete"
                        if measured["quality"] != "ungraded"
                        and measured["execution_exit"] == 0
                        else "failed"
                    )
                    if job.get("diagnostic_seconds"):
                        result["diagnostic"] = acquire_trace(
                            base,
                            namespace,
                            system,
                            "T2",
                            destination / "diagnostic",
                            job["diagnostic_seconds"],
                            cpu_bytes,
                            job["pressure_prefixes"],
                        )
                        if not result["diagnostic"]["complete"]:
                            result["status"] = "failed"
                else:
                    raise ValueError("panel job kind not implemented")
                result["measurement"] = measured
                result["directory"] = str(index)
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["status"] = "failed"
                result["reason"] = str(error)
            write_record(output / f"progress-{index}.json", result)
    finally:
        terminal = {
            "schema": "benchmark-panel-terminal-v1",
            "admission_sha256": digest_file(output / "admission.json"),
            "elapsed_seconds": time.monotonic() - started,
            "results": results,
            "complete": len(results) == len(jobs)
            and all(result["status"] == "complete" for result in results),
        }
        write_record(output / "terminal.json", terminal)
    return terminal


def verify_application(output: Path, job: dict[str, Any]) -> dict[str, Any]:
    acquisition: dict[str, Any] = json.loads(
        (output / "task/acquisition.json").read_text()
    )
    if (
        acquisition["instance"] != job["instance"]
        or acquisition["execution_exit"] != 0
        or acquisition["quality"] not in ("resolved", "unresolved")
    ):
        raise ValueError("application did not complete")
    kind = job["task_kind"]
    grading_directory = (
        output / "task" / ("repair-grade" if kind == "branching" else "grading")
    )
    if kind != "terminal":
        grading_admission = json.loads(
            (grading_directory / "admission.json").read_text()
        )
        if (
            grading_admission["image"]["id"] != job["image"]["id"]
            or grading_admission["dataset_sha256"] != job["dataset_sha256"]
        ):
            raise ValueError(
                "application grader differs from admitted image or dataset"
            )
    if kind in ("terminal", "branching"):
        reader = verify_terminal if kind == "terminal" else verify_branching
        if reader(output / "task") != acquisition:
            raise ValueError("application acquisition drift")
    else:
        if acquisition["task_kind"] != kind:
            raise ValueError("application task kind drift")
        if kind == "repair":
            grading = verify_coding_grade(output / "task/grading")
            quality = grading["quality"]
        else:
            reader = {
                "regression": verify_regression,
                "navigation": verify_navigation,
                "grounded": verify_grounded,
            }[kind]
            grading = reader(output / "task/grading")
            quality = grading["quality"]
        if grading != acquisition["grading"] or quality != acquisition["quality"]:
            raise ValueError("application verdict differs from observations")
    captures = sorted(
        path
        for path in (output / "requests").glob("request-*.json")
        if not path.name.endswith(".admission.json")
    )
    admissions = sorted((output / "requests").glob("request-*.admission.json"))
    if not captures or len(captures) != len(admissions):
        raise ValueError("incomplete application request capture")
    for index, path in enumerate(captures, 1):
        record = json.loads(path.read_text())
        original = json.loads(admissions[index - 1].read_text())
        if (
            original["generation"] != index
            or record["start_ns"] != original["start_ns"]
            or not record["start_ns"] <= record["received_ns"] <= record["forwarded_ns"]
        ):
            raise ValueError("application request identity or timing drift")
        usage = record["response"]["usage"]
        if (
            usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
            or not 0
            <= usage["prompt_tokens_details"]["cached_tokens"]
            <= usage["prompt_tokens"]
        ):
            raise ValueError("application token accounting drift")
    return acquisition


def verify_panel(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if (
        admission["schema"] != "benchmark-panel-v1"
        or terminal["schema"] != "benchmark-panel-terminal-v1"
        or digest_file(output / "admission.json") != terminal["admission_sha256"]
        or [job["identity"] for job in admission["jobs"]]
        != [result["identity"] for result in terminal["results"]]
    ):
        raise ValueError("panel identity or denominator drift")
    complete = True
    for job, result in zip(admission["jobs"], terminal["results"], strict=True):
        if result["status"] != "complete":
            complete = False
            continue
        destination = output / result["directory"]
        if not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError("panel cell escapes its receipt")
        observed = (
            verify_trace(destination)
            if job["kind"] == "trace"
            else verify_application(destination, job)
        )
        if observed != result["measurement"]:
            raise ValueError("panel differs from independent observations")
        if (
            "diagnostic" in result
            and verify_trace(destination / "diagnostic") != result["diagnostic"]
        ):
            raise ValueError("validation diagnostic differs from observations")
    if terminal["complete"] != complete:
        raise ValueError("panel completion differs from coverage")
    return terminal
