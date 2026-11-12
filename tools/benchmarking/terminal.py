from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import digest_file
from tools.config import check_environment
from tools.process import run_owned
from tools.runtime_verify import snapshot, write_record


def task_identity(source: Path) -> dict[str, str]:
    return {
        str(path.relative_to(source)): digest_file(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def trial_config(
    task: Path, output: Path, name: str, endpoint: str, agent: str, seconds: int
) -> dict[str, Any]:
    agent_config: dict[str, Any] = {
        "name": agent,
        "override_timeout_sec": seconds,
        "override_setup_timeout_sec": 30,
    }
    if agent == "terminus-2":
        agent_config.update(
            model_name="openai/qualification",
            base_url=endpoint + "/v1",
            env={"OPENAI_API_KEY": "local-benchmark"},
            kwargs={
                "max_turns": 20,
                "parser_name": "json",
                "temperature": 0,
                "record_terminal_session": False,
                "enable_summarize": False,
                "model_info": {
                    "max_input_tokens": 8192,
                    "max_output_tokens": 512,
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
                "llm_kwargs": {"timeout": 120, "num_retries": 0, "max_tokens": 512},
                "llm_call_kwargs": {
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
                },
            },
        )
    return {
        "task": {"path": str(task)},
        "trial_name": name,
        "trials_dir": str(output / "trials"),
        "agent": agent_config,
        "environment": {
            "type": "docker",
            "delete": False,
            "override_cpus": 1,
            "override_memory_mb": 2048,
        },
        "verifier": {"override_timeout_sec": 90, "env": {"UV_OFFLINE": "1"}},
    }


def terminal_observation(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    trial = output / "trials" / admission["trial_name"]
    result = json.loads((trial / "result.json").read_text())
    if (
        result["trial_name"] != admission["trial_name"]
        or result.get("exception_info") is not None
    ):
        raise ValueError("terminal trial did not complete normally")
    reward = float((trial / "verifier/reward.txt").read_text().strip())
    if reward not in (0, 1) or result["verifier_result"]["rewards"] != {
        "reward": reward
    }:
        raise ValueError("terminal reward disagrees with original verifier")
    observations = json.loads((trial / "verifier/ctrf.json").read_text())
    tests = observations["results"]["tests"]
    identities = [item["name"] for item in tests]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("terminal validator has no unique test observations")
    if bool(reward) != all(item["status"] == "passed" for item in tests):
        raise ValueError("terminal reward disagrees with executable observations")
    return {
        "quality": "resolved" if reward else "unresolved",
        "reward": reward,
        "tests": identities,
    }


def run_terminal(
    root: Path,
    python: Path,
    dataset: Path,
    instance: str,
    endpoint: str,
    output: Path,
    seconds: int,
    agent: str = "terminus-2",
) -> dict[str, Any]:
    if output.resolve().is_relative_to(root.resolve()) or output.exists():
        raise ValueError("terminal acquisition requires a new external output")
    if agent not in ("terminus-2", "oracle", "nop") or not 0 < seconds <= 240:
        raise ValueError("invalid terminal agent or acquisition budget")
    if agent == "terminus-2" and not re.fullmatch(
        r"http://127\.0\.0\.1:[0-9]+", endpoint
    ):
        raise ValueError("terminal model must use the owned localhost proxy")
    rows = [
        row for row in json.loads(dataset.read_text()) if row["instance_id"] == instance
    ]
    if len(rows) != 1:
        raise ValueError("terminal acquisition requires one admitted source task")
    row = rows[0]
    source = (dataset.parent / row["source_path"]).resolve()
    if task_identity(source) != row["source_sha256"]:
        raise ValueError("terminal source differs from admitted task")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", row["image_name"]):
        raise ValueError("terminal task requires a content-addressed image")
    output.mkdir(parents=True)
    driver = snapshot(root, output / "driver-source")
    (output / "dataset.json").write_bytes(dataset.read_bytes())
    shutil.copytree(source, output / "source-task")
    shutil.copytree(source, output / "task")
    config_path = output / "task/task.toml"
    original = config_path.read_text()
    old_image = tomllib.loads(original)["environment"]["docker_image"]
    if original.count('docker_image = "' + old_image + '"') != 1:
        raise ValueError("terminal source image field is ambiguous")
    config_path.write_text(
        original.replace(
            'docker_image = "' + old_image + '"',
            'docker_image = "' + row["image_name"] + '"',
        )
    )
    (output / "task/environment/docker-compose.yaml").write_text(
        "services:\n  main:\n    network_mode: none\n"
    )
    name = "wh-" + uuid.uuid4().hex
    configuration = trial_config(
        output / "task", output, name, endpoint, agent, seconds
    )
    write_record(output / "trial.json", configuration)
    command = [
        str(python),
        "-m",
        "harbor.cli.main",
        "trials",
        "start",
        "--config",
        str(output / "trial.json"),
    ]
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-terminal-v2",
            "driver_sha256": driver,
            "instance": instance,
            "dataset_sha256": digest_file(output / "dataset.json"),
            "source_sha256": row["source_sha256"],
            "task_sha256": task_identity(output / "task"),
            "image": {"id": row["image_name"]},
            "agent": agent,
            "trial_name": name,
            "command": command,
            "budget_seconds": seconds + 180,
            "scope": "Terminal-Bench task in Harbor 0.25.0 with original validators; bounded agent and cached tool dependencies",
        },
    )
    started = time.monotonic()
    terminal: dict[str, Any] = {
        "quality": "ungraded",
        "error": None,
        "execution_exit": 1,
        "instance": instance,
        "image": {"id": row["image_name"]},
        "task_kind": "terminal",
        "agent": agent,
    }
    with (output / "execution.log").open("x") as log:
        try:
            terminal["execution_exit"] = run_owned(
                ["sg", "docker", "-c", shlex.join(command)],
                check_environment(root),
                output,
                seconds + 165,
                log,
            )
            terminal.update(terminal_observation(output))
        except (OSError, ValueError, KeyError, TypeError) as error:
            terminal["error"] = str(error)
        finally:
            project = name + "__env"
            try:
                listed = subprocess.run(
                    [
                        "sg",
                        "docker",
                        "-c",
                        shlex.join(
                            [
                                "docker",
                                "ps",
                                "-aq",
                                "--filter",
                                "label=com.docker.compose.project=" + project,
                            ]
                        ),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                if listed.stdout.split():
                    subprocess.run(
                        [
                            "sg",
                            "docker",
                            "-c",
                            shlex.join(["docker", "rm", "-f", *listed.stdout.split()]),
                        ],
                        capture_output=True,
                        check=True,
                        timeout=10,
                    )
            except (OSError, subprocess.SubprocessError) as error:
                terminal.update(
                    quality="ungraded", error="owned cleanup failed: " + str(error)
                )
    terminal["elapsed_seconds"] = time.monotonic() - started
    terminal["admission_sha256"] = digest_file(output / "admission.json")
    terminal["artifacts_sha256"] = task_identity(output)
    write_record(output / "acquisition.json", terminal)
    return terminal


def verify_terminal(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "acquisition.json").read_text())
    if admission["schema"] not in (
        "benchmark-terminal-v1",
        "benchmark-terminal-v2",
    ) or terminal["admission_sha256"] != digest_file(output / "admission.json"):
        raise ValueError("invalid terminal admission")
    actual = task_identity(output)
    actual.pop("acquisition.json")
    if actual != terminal["artifacts_sha256"]:
        raise ValueError("terminal evidence drift")
    if (
        digest_file(output / "dataset.json") != admission["dataset_sha256"]
        or task_identity(output / "source-task") != admission["source_sha256"]
        or task_identity(output / "task") != admission["task_sha256"]
    ):
        raise ValueError("terminal source or task drift")
    if admission["schema"] == "benchmark-terminal-v2" and any(
        digest_file(output / "driver-source" / relative) != expected
        for relative, expected in admission["driver_sha256"].items()
    ):
        raise ValueError("terminal driver source drift")
    rows = [
        row
        for row in json.loads((output / "dataset.json").read_text())
        if row["instance_id"] == admission["instance"]
    ]
    if (
        len(rows) != 1
        or rows[0]["source_sha256"] != admission["source_sha256"]
        or rows[0]["image_name"] != admission["image"]["id"]
    ):
        raise ValueError("terminal task identity differs from source")
    if not 0 < terminal["elapsed_seconds"] <= admission["budget_seconds"] + 1:
        raise ValueError("terminal execution exceeded its budget")
    if terminal["execution_exit"] != 0 or terminal["quality"] == "ungraded":
        raise ValueError("terminal acquisition incomplete")
    derived = terminal_observation(output)
    if any(terminal[key] != value for key, value in derived.items()):
        raise ValueError("terminal verdict differs from original observations")
    return terminal
