from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from tools.config import check_environment
from tools.process import run_owned
from tools.runtime_verify import write_record


def pin_swebench_image(
    instance: str, expected_identity: str | None = None
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", instance):
        raise ValueError("unsupported SWE-bench instance identity")
    repository = (
        "docker.io/swebench/sweb.eval.x86_64."
        + instance.replace("__", "_1776_").lower()
    )
    inspected = subprocess.run(
        [
            "sg",
            "docker",
            "-c",
            shlex.join(["docker", "image", "inspect", repository + ":latest"]),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    records = json.loads(inspected.stdout)
    if len(records) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", records[0]["Id"]):
        raise ValueError("invalid local task image identity")
    identity = records[0]["Id"]
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("task image changed between acquisition and grading")
    tag = "benchmark-" + identity.removeprefix("sha256:")
    subprocess.run(
        [
            "sg",
            "docker",
            "-c",
            shlex.join(["docker", "tag", identity, repository + ":" + tag]),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return {
        "id": identity,
        "reference": repository + ":" + tag,
        "tag": tag,
        "repository_digests": records[0].get("RepoDigests", []),
    }


def coding_config(endpoint: str, seconds: int, label: str) -> dict[str, Any]:
    return {
        "agent": {
            "step_limit": 32,
            "wall_time_limit_seconds": seconds,
            "cost_limit": 0,
            "instance_template": "Solve this issue in /testbed: {{task}}\nRead the relevant code, fix it, and run a focused check. Keep explanations concise. Do not change existing tests. Use exactly one mswea_bash_command fenced command per response. To finish, run: printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; git diff --no-ext-diff",
        },
        "environment": {
            "environment_class": "docker",
            "cwd": "/testbed",
            "timeout": 30,
            "interpreter": ["bash", "-c"],
            "env": {
                "BASH_ENV": "/root/.bashrc",
                "PAGER": "cat",
                "PIP_PROGRESS_BAR": "off",
            },
            "container_timeout": f"{seconds + 60}s",
            "run_args": [
                "--rm",
                "--network=none",
                "--cpus=1",
                "--memory=2g",
                "--label",
                label,
            ],
        },
        "model": {
            "model_class": "litellm_textbased",
            "model_name": "openai/qualification",
            "cost_tracking": "ignore_errors",
            "model_kwargs": {
                "api_base": endpoint + "/v1",
                "api_key": "local-benchmark",
                "temperature": 0,
                "max_tokens": 512,
                "timeout": 120,
                "num_retries": 0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        },
    }


def run_swebench(
    root: Path,
    python: Path,
    dataset: Path,
    instance: str,
    endpoint: str,
    output: Path,
    seconds: int,
    task_kind: str = "repair",
) -> dict[str, Any]:
    if task_kind not in ("repair", "regression", "navigation"):
        raise ValueError("unsupported coding task kind")
    output.mkdir(parents=True, exist_ok=False)
    selected = [
        row for row in json.loads(dataset.read_text()) if row["instance_id"] == instance
    ]
    if len(selected) != 1:
        raise ValueError("coding requires exactly one source instance")
    if task_kind == "navigation":
        image = {"id": selected[0]["image_name"]}
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image["id"]):
            raise ValueError("navigation requires a content-addressed image")
    else:
        image = pin_swebench_image(instance)
    selected[0]["image_name"] = image["id"]
    pinned_dataset = output / "dataset"
    pinned_dataset.mkdir()
    with (pinned_dataset / "test.json").open("x") as destination:
        json.dump(selected, destination)
    write_record(output / "image.json", image)
    label = "wait-hint-attempt=" + uuid.uuid4().hex
    configuration = coding_config(endpoint, seconds, label)
    if task_kind == "regression":
        configuration["agent"]["instance_template"] = (
            "Write a regression test in /testbed for this issue: {{task}}\n"
            "Only add benchmark_generated_test.py. Do not fix production code or change existing tests. "
            "Exercise the actual API without network access. The test must fail on this checkout "
            "and pass after the hidden reference fix. Inspect code and run the test. "
            "Use exactly one mswea_bash_command fenced command per response. "
            "To submit, run git add -N benchmark_generated_test.py, then "
            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; git diff --no-ext-diff"
        )
    elif task_kind == "navigation":
        configuration["agent"]["instance_template"] = (
            "Investigate this repository navigation/change-impact task in /testbed: {{task}}\n"
            "Read the actual implementation and run affected tests. Do not change source or tests. "
            "Only add benchmark_navigation.json with the requested answer. "
            "Use exactly one mswea_bash_command fenced command per response. "
            "To submit, run git add -N benchmark_navigation.json, then "
            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; git diff --no-ext-diff"
        )
    configuration["agent"]["output_path"] = str(output / "trajectory-live.json")
    write_record(output / "agent.yaml", configuration)
    command = [
        str(python),
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        str(pinned_dataset),
        "--split",
        "test",
        "--filter",
        "^" + instance + "$",
        "--workers",
        "1",
        "--output",
        str(output / "agent"),
        "--config",
        "mini_textbased.yaml",
        "--config",
        str(output / "agent.yaml"),
    ]
    started = time.monotonic()
    status = 1
    with (output / "agent.log").open("x") as log:
        try:
            status = run_owned(
                ["sg", "docker", "-c", shlex.join(command)],
                check_environment(root),
                output,
                seconds,
                log,
            )
        finally:
            listing = subprocess.run(
                [
                    "sg",
                    "docker",
                    "-c",
                    shlex.join(["docker", "ps", "-aq", "--filter", "label=" + label]),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            containers = listing.stdout.split()
            if containers:
                if status == 124 and len(containers) == 1:
                    saved = subprocess.run(
                        [
                            "sg",
                            "docker",
                            "-c",
                            shlex.join(
                                [
                                    "docker",
                                    "exec",
                                    "-w",
                                    "/testbed",
                                    containers[0],
                                    "git",
                                    "diff",
                                    "--no-ext-diff",
                                ]
                            ),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if saved.returncode == 0:
                        write_record(
                            output / "deadline-preds.json",
                            {
                                instance: {
                                    "model_name_or_path": "openai/qualification",
                                    "instance_id": instance,
                                    "model_patch": saved.stdout,
                                }
                            },
                        )
                subprocess.run(
                    [
                        "sg",
                        "docker",
                        "-c",
                        shlex.join(["docker", "rm", "-f", *containers]),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
    result: dict[str, Any] = {
        "execution_exit": status,
        "elapsed_seconds": time.monotonic() - started,
        "instance": instance,
        "quality": "ungraded",
        "image": image,
        "task_kind": task_kind,
    }
    trajectory = output / "agent" / instance / (instance + ".traj.json")
    result["agent_exit"] = (
        json.loads(trajectory.read_text())["info"]["exit_status"]
        if trajectory.is_file()
        else "trajectory_missing"
    )
    predictions = output / "agent/preds.json"
    result["prediction_origin"] = "agent-submission"
    if not predictions.is_file() and (output / "deadline-preds.json").is_file():
        predictions = output / "deadline-preds.json"
        result["prediction_origin"] = "deadline-snapshot"
    if predictions.is_file():
        result["prediction"] = json.loads(predictions.read_text()).get(instance)
        if task_kind in ("regression", "navigation"):
            from tools.benchmarking.navigation import grade_navigation
            from tools.benchmarking.regression import grade_regression

            patch = output / "submitted.patch"
            patch.write_text((result["prediction"] or {}).get("model_patch") or "")
            grader = grade_regression if task_kind == "regression" else grade_navigation
            result["grading"] = grader(
                root, dataset, patch, instance, output / "grading"
            )
        else:
            result["grading"] = grade_swebench(
                root,
                python,
                dataset,
                predictions,
                instance,
                output / "grading",
                image["id"],
            )
        result["quality"] = result["grading"]["quality"]
    write_record(output / "acquisition.json", result)
    return result


def grade_swebench(
    root: Path,
    python: Path,
    dataset: Path,
    predictions: Path,
    instance: str,
    output: Path,
    image_identity: str | None = None,
) -> dict[str, Any]:
    import hashlib

    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("grading requires a new external output directory")
    if not python.is_file() or not dataset.is_file() or not predictions.is_file():
        raise ValueError("grading requires existing runtime, dataset and predictions")
    rows = json.loads(dataset.read_text())
    if sum(row.get("instance_id") == instance for row in rows) != 1:
        raise ValueError("grading requires exactly one matching dataset instance")
    output.mkdir(parents=True, exist_ok=False)
    image = pin_swebench_image(instance, image_identity)
    run_id = "wh-" + uuid.uuid4().hex
    command = [
        str(python),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset),
        "--predictions_path",
        str(predictions),
        "--instance_ids",
        instance,
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--timeout",
        "120",
        "--instance_image_tag",
        image["tag"],
        "--cache_level",
        "instance",
        "--clean",
        "False",
    ]
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-coding-grade-v1",
            "command": command,
            "predictions": str(predictions),
            "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "dataset": str(dataset),
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "instance": instance,
            "image": image,
        },
    )
    with (output / "grading.log").open("x") as log:
        try:
            status = run_owned(
                ["sg", "docker", "-c", shlex.join(command)],
                check_environment(root),
                output,
                180,
                log,
            )
        finally:
            subprocess.run(
                [
                    "sg",
                    "docker",
                    "-c",
                    shlex.join(
                        ["docker", "rm", "-f", f"sweb.eval.{instance.lower()}.{run_id}"]
                    ),
                ],
                capture_output=True,
                timeout=30,
            )
    result: dict[str, Any] = {"exit_code": status, "quality": "ungraded"}
    reports = list(
        (output / "logs/run_evaluation" / run_id).glob("*/" + instance + "/report.json")
    )
    if status == 0 and len(reports) == 1:
        result["report"] = json.loads(reports[0].read_text())
        result["quality"] = (
            "resolved" if result["report"][instance]["resolved"] else "unresolved"
        )
    write_record(output / "terminal.json", result)
    return result
