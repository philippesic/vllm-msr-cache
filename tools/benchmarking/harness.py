from __future__ import annotations

import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from tools.config import check_environment
from tools.process import run_owned
from tools.runtime_verify import write_record


def coding_config(endpoint: str, seconds: int, label: str) -> dict[str, Any]:
    return {
        "agent": {
            "step_limit": 16,
            "wall_time_limit_seconds": seconds,
            "cost_limit": 0,
            "instance_template": "Solve this issue in /testbed: {{task}}\nRead the relevant code, fix it, and run a focused check. Keep explanations concise. Do not change existing tests. Use exactly one mswea_bash_command fenced command per response. To finish, run: printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; git diff --no-ext-diff",
        },
        "environment": {
            "environment_class": "docker",
            "cwd": "/testbed",
            "timeout": 30,
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
                "timeout": 60,
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
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    label = "wait-hint-attempt=" + uuid.uuid4().hex
    configuration = coding_config(endpoint, seconds, label)
    write_record(output / "agent.yaml", configuration)
    command = [
        str(python),
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        str(dataset.parent),
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
    }
    predictions = output / "agent/preds.json"
    if predictions.is_file():
        result["prediction"] = json.loads(predictions.read_text()).get(instance)
        grading = [
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
            "pilot",
            "--timeout",
            "120",
        ]
        with (output / "grading.log").open("x") as log:
            result["grading_exit"] = run_owned(
                ["sg", "docker", "-c", shlex.join(grading)],
                check_environment(root),
                output,
                180,
                log,
            )
        reports = list(
            (output / "logs/run_evaluation/pilot").glob(
                "*/" + instance + "/report.json"
            )
        )
        if len(reports) == 1:
            result["grading_report"] = json.loads(reports[0].read_text())
            result["quality"] = (
                "resolved"
                if result["grading_report"][instance]["resolved"]
                else "unresolved"
            )
    write_record(output / "acquisition.json", result)
    return result
