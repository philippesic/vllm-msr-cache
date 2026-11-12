from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from tools.benchmarking.harness import grade_swebench, pin_swebench_image
from tools.benchmarking.probe import digest_file, request
from tools.benchmarking.regression import grade_regression, verify_regression
from tools.runtime_verify import write_record


def transcript_identity(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def run_branching(
    root: Path,
    python: Path,
    dataset: Path,
    instance: str,
    endpoint: str,
    output: Path,
    seconds: int,
) -> dict[str, Any]:
    if (
        output.resolve().is_relative_to(root.resolve())
        or output.exists()
        or not 0 < seconds <= 240
    ):
        raise ValueError(
            "branching requires a new external output and bounded acquisition"
        )
    rows = [
        row for row in json.loads(dataset.read_text()) if row["instance_id"] == instance
    ]
    if len(rows) != 1:
        raise ValueError("branching requires one admitted real issue")
    row = rows[0]
    image = pin_swebench_image(instance)
    output.mkdir(parents=True)
    (output / "dataset.json").write_bytes(dataset.read_bytes())
    write_record(output / "image.json", image)
    names = {key: "wait-hint-fork-" + uuid.uuid4().hex for key in ("A", "B")}
    started = time.monotonic()
    events: list[dict[str, Any]] = []
    complete = {"A": False, "B": False}
    messages: dict[str, list[dict[str, str]]] = {}
    state: dict[str, Any] = {
        "execution_exit": 1,
        "quality": "ungraded",
        "instance": instance,
        "image": image,
        "task_kind": "branching",
        "error": None,
    }

    def docker(
        arguments: list[str], timeout: int = 20
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sg", "docker", "-c", shlex.join(["docker", *arguments])],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def execute(
        branch: str, command: str, *, retain_full: bool = False
    ) -> dict[str, Any]:
        result = docker(
            [
                "exec",
                "-w",
                "/testbed",
                "-e",
                "PAGER=cat",
                "-e",
                "GIT_PAGER=cat",
                names[branch],
                "timeout",
                "15",
                "bash",
                "-c",
                command,
            ]
        )
        combined = result.stdout + result.stderr
        return {
            "exit_code": result.returncode,
            "output": combined if retain_full else combined[-24000:],
        }

    def turns(branch: str, count: int, stage: str) -> None:
        for _ in range(count):
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 2:
                raise TimeoutError("branching acquisition deadline")
            response = request(
                endpoint + "/v1/chat/completions",
                {
                    "model": "qualification",
                    "messages": messages[branch],
                    "temperature": 0,
                    "max_tokens": 512,
                    "benchmark_session": branch,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=min(90, int(remaining)),
            )
            content = response["choices"][0]["message"].get("content") or ""
            messages[branch].append({"role": "assistant", "content": content})
            actions = re.findall(
                r"```mswea_bash_command\s*\n(.*?)\n```", content, re.DOTALL
            )
            if len(actions) != 1:
                observation: dict[str, Any] = {
                    "exit_code": None,
                    "output": "Use exactly one mswea_bash_command fenced command.",
                }
            else:
                observation = execute(branch, actions[0])
                if observation["output"].startswith(
                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                ):
                    complete[branch] = True
            messages[branch].append(
                {"role": "user", "content": json.dumps(observation)}
            )
            events.append(
                {
                    "branch": branch,
                    "stage": stage,
                    "observation": observation,
                    "transcript_sha256": transcript_identity(messages[branch]),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            write_record(
                output / f"event-{len(events):03d}.json",
                {"event": events[-1], "messages": messages[branch]},
            )
            if complete[branch]:
                break

    try:
        for branch, name in names.items():
            created = docker(
                [
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    name,
                    "--network=none",
                    "--cpus=1",
                    "--memory=2g",
                    image["id"],
                    "sleep",
                    str(seconds + 60),
                ]
            )
            if created.returncode:
                raise ValueError("branch container failed to start")
        command = "sed -n '1,240p' requests/models.py; sed -n '1,220p' HISTORY.rst"
        first = execute("A", command)
        if first != execute("B", command) or first["exit_code"]:
            raise ValueError("branch source contexts differ")
        shared = [
            {
                "role": "system",
                "content": "Work in /testbed on this real issue. Use exactly one mswea_bash_command fenced shell command per response. Keep prose brief. Do not use the network or change existing tests. To finish, print COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. The harness retains your actual Git diff.",
            },
            {"role": "user", "content": row["problem_statement"]},
            {
                "role": "assistant",
                "content": "```mswea_bash_command\n" + command + "\n```",
            },
            {"role": "user", "content": json.dumps(first)},
        ]
        write_record(output / "shared-transcript.json", {"messages": shared})
        for branch in names:
            messages[branch] = copy.deepcopy(shared)
        messages["A"].append(
            {
                "role": "user",
                "content": "Branch A: fix the production implementation for the issue. Preserve unrelated behavior. Run a focused check.",
            }
        )
        messages["B"].append(
            {
                "role": "user",
                "content": "Branch B: independently write only benchmark_generated_test.py as a regression test. Do not fix production code. Test the public API offline. It must fail here and pass after the hidden real fix. Run it, then git add -N benchmark_generated_test.py before finishing.",
            }
        )
        write_record(
            output / "admission.json",
            {
                "schema": "benchmark-branching-v1",
                "instance": instance,
                "dataset_sha256": digest_file(output / "dataset.json"),
                "image": image,
                "shared_transcript_sha256": transcript_identity(shared),
                "budget_seconds": seconds,
                "schedule": ["A:3", "B:4", "A:5"],
                "scope": "project-authored transcript fork, independent real patches, return to original branch and original/fail-before-pass-after grading",
            },
        )
        turns("A", 3, "initial-fix")
        original_a = execute("A", "git diff --no-ext-diff", retain_full=True)
        write_record(output / "A-before-B.json", original_a)
        turns("B", 4, "competing-test")
        after_b = execute("A", "git diff --no-ext-diff", retain_full=True)
        write_record(output / "A-after-B.json", after_b)
        if after_b != original_a:
            raise ValueError("competing branch modified original branch state")
        state["branch_isolation"] = True
        complete["A"] = False
        diagnostic = execute("A", "git status --short; git diff --stat")
        messages["A"].append(
            {
                "role": "user",
                "content": "Return to branch A. The alternative test-authoring branch has run in a separate checkout. Review only your original fix, run a focused check and finish. Your current branch state: "
                + json.dumps(diagnostic),
            }
        )
        turns("A", 5, "return-and-revise")
        state["execution_exit"] = 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        state["error"] = str(error)
    finally:
        for branch in names:
            try:
                if branch == "B":
                    execute(
                        branch,
                        "git add -N benchmark_generated_test.py 2>/dev/null || true",
                    )
                saved = execute(branch, "git diff --no-ext-diff", retain_full=True)
                if saved["exit_code"] == 0:
                    (output / (branch + ".patch")).write_text(saved["output"])
            except (OSError, subprocess.SubprocessError) as error:
                state["error"] = "patch preservation failed: " + str(error)
                state["execution_exit"] = 1
            finally:
                try:
                    docker(["rm", "-f", names[branch]], 10)
                except (OSError, subprocess.SubprocessError) as error:
                    state["error"] = "owned cleanup failed: " + str(error)
                    state["execution_exit"] = 1
    state["elapsed_seconds"] = time.monotonic() - started
    state["completed_branches"] = complete
    state["events"] = events
    if all((output / (branch + ".patch")).is_file() for branch in names):
        predictions = output / "predictions.json"
        write_record(
            predictions,
            {
                instance: {
                    "instance_id": instance,
                    "model_name_or_path": "openai/qualification",
                    "model_patch": (output / "A.patch").read_text(),
                }
            },
        )
        state["repair_grade"] = grade_swebench(
            root,
            python,
            dataset,
            predictions,
            instance,
            output / "repair-grade",
            image["id"],
        )
        state["regression_grade"] = grade_regression(
            root, dataset, output / "B.patch", instance, output / "regression-grade"
        )
        state["quality"] = (
            "resolved"
            if all(complete.values())
            and state["repair_grade"]["quality"]
            == state["regression_grade"]["quality"]
            == "resolved"
            else "unresolved"
        )
        if (
            state["repair_grade"]["quality"] == "ungraded"
            or state["regression_grade"]["quality"] == "ungraded"
        ):
            state["quality"] = "ungraded"
    state["artifacts_sha256"] = {
        str(path.relative_to(output)): digest_file(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    write_record(output / "acquisition.json", state)
    return state


def verify_branching(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    state: dict[str, Any] = json.loads((output / "acquisition.json").read_text())
    if admission["schema"] != "benchmark-branching-v1":
        raise ValueError("unsupported branching receipt")
    actual = {
        str(path.relative_to(output)): digest_file(path)
        for path in output.rglob("*")
        if path.is_file() and path != output / "acquisition.json"
    }
    if (
        actual != state["artifacts_sha256"]
        or digest_file(output / "dataset.json") != admission["dataset_sha256"]
    ):
        raise ValueError("branching evidence drift")
    shared = json.loads((output / "shared-transcript.json").read_text())["messages"]
    if transcript_identity(shared) != admission[
        "shared_transcript_sha256"
    ] or not state.get("branch_isolation"):
        raise ValueError("missing shared prefix or branch isolation")
    if (output / "A-before-B.json").read_bytes() != (
        output / "A-after-B.json"
    ).read_bytes():
        raise ValueError("competing branch changed the original branch")
    sequence: list[str] = []
    for index, event in enumerate(state["events"], 1):
        record = json.loads((output / f"event-{index:03d}.json").read_text())
        if (
            record["event"] != event
            or record["messages"][: len(shared)] != shared
            or transcript_identity(record["messages"]) != event["transcript_sha256"]
        ):
            raise ValueError("branch transcript differs from its execution")
        if not sequence or event["branch"] != sequence[-1]:
            sequence.append(event["branch"])
    if (
        sequence != ["A", "B", "A"]
        or state["execution_exit"] != 0
        or state["elapsed_seconds"] > admission["budget_seconds"] + 60
    ):
        raise ValueError("branch schedule incomplete or over budget")
    regression = verify_regression(output / "regression-grade")
    regression_admission = json.loads(
        (output / "regression-grade/admission.json").read_text()
    )
    repair_admission = json.loads((output / "repair-grade/admission.json").read_text())
    prediction = json.loads((output / "predictions.json").read_text())
    for grading in (regression_admission, repair_admission):
        if (
            grading["instance"] != admission["instance"]
            or grading["dataset_sha256"] != admission["dataset_sha256"]
            or grading["image"]["id"] != admission["image"]["id"]
        ):
            raise ValueError("branch graders used different inputs")
    if (
        regression_admission["submitted_sha256"] != digest_file(output / "B.patch")
        or repair_admission["predictions_sha256"]
        != digest_file(output / "predictions.json")
        or prediction[admission["instance"]]["model_patch"]
        != (output / "A.patch").read_text()
    ):
        raise ValueError("branch graders used different patches")
    repair = json.loads((output / "repair-grade/terminal.json").read_text())
    reports = list((output / "repair-grade/logs/run_evaluation").rglob("report.json"))
    if (
        len(reports) != 1
        or json.loads(reports[0].read_text()) != repair["report"]
        or repair["exit_code"] != 0
    ):
        raise ValueError("branch repair grade incomplete")
    if regression != state["regression_grade"] or repair != state["repair_grade"]:
        raise ValueError("branch grading differs from original observations")
    resolved = (
        all(state["completed_branches"].values())
        and repair["report"][admission["instance"]]["resolved"]
        and regression["quality"] == "resolved"
    )
    if state["quality"] != ("resolved" if resolved else "unresolved"):
        raise ValueError("branch quality differs from original observations")
    return state
