from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from tools.benchmarking.answers import added_json
from tools.benchmarking.probe import digest_file
from tools.benchmarking.regression import finish_grade
from tools.runtime_verify import write_record

ANSWER_FILE = "benchmark_navigation.json"


def navigation_answer(patch: Path) -> dict[str, Any]:
    return added_json(patch, ANSWER_FILE)


def check_answer(answer: dict[str, Any], rubric: dict[str, Any]) -> bool:
    return (
        set(answer) == {"symbols", "behavior", "tests"}
        and answer["symbols"] == rubric["symbols"]
        and json.dumps(answer["behavior"], sort_keys=True)
        == json.dumps(rubric["behavior"], sort_keys=True)
        and isinstance(answer["tests"], list)
        and bool(answer["tests"])
        and all(isinstance(item, str) for item in answer["tests"])
        and set(answer["tests"]) <= set(rubric["tests"])
    )


def navigation_observations(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    answer = navigation_answer(output / "submitted.patch")
    if not check_answer(answer, admission["rubric"]):
        return {"quality": "unresolved", "answer_matches": False}
    tree = ET.parse(output / "tests.xml")
    cases = list(tree.iter("testcase"))
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("missing or duplicate navigation check observations")
    passed = all(
        case.find("failure") is None
        and case.find("error") is None
        and case.find("skipped") is None
        for case in cases
    )
    return {
        "quality": "resolved" if passed else "unresolved",
        "answer_matches": True,
        "tests": [list(identity) for identity in identities],
    }


def grade_navigation(
    root: Path, dataset: Path, prediction: Path, instance: str, output: Path
) -> dict[str, Any]:
    if output.resolve().is_relative_to(root.resolve()) or output.exists():
        raise ValueError("navigation grading requires a new external output")
    rows = [
        row for row in json.loads(dataset.read_text()) if row["instance_id"] == instance
    ]
    if len(rows) != 1:
        raise ValueError("navigation grading requires one admitted instance")
    row = rows[0]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", row["image_name"]):
        raise ValueError("navigation requires a content-addressed image")
    output.mkdir(parents=True)
    (output / "dataset.json").write_bytes(dataset.read_bytes())
    (output / "submitted.patch").write_bytes(prediction.read_bytes())
    (output / "reference_check.py").write_text(row["reference_check"])
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-navigation-grade-v1",
            "instance": instance,
            "dataset_sha256": digest_file(output / "dataset.json"),
            "submitted_sha256": digest_file(output / "submitted.patch"),
            "rubric": row["rubric"],
            "image": {"id": row["image_name"]},
            "budget_seconds": 90,
            "scope": "project-authored navigation and executable change-impact rubric",
        },
    )
    started = time.monotonic()
    terminal: dict[str, Any] = {"quality": "ungraded", "error": None}
    try:
        answer = navigation_answer(output / "submitted.patch")
    except (ValueError, KeyError, TypeError) as error:
        terminal.update(quality="unresolved", rubric_refusal=str(error))
        (output / "execution.log").touch(exist_ok=False)
        return finish_grade(output, terminal, started)
    if not check_answer(answer, row["rubric"]):
        terminal.update(quality="unresolved", answer_matches=False)
        (output / "execution.log").touch(exist_ok=False)
        return finish_grade(output, terminal, started)
    name = "wait-hint-navigation-" + uuid.uuid4().hex
    with (output / "execution.log").open("x") as log:

        def docker(arguments: list[str], seconds: int = 15) -> int:
            if arguments[0] != "rm":
                seconds = min(seconds, int(75 - (time.monotonic() - started)))
                if seconds <= 0:
                    raise ValueError("navigation grading budget exhausted")
            return subprocess.run(
                ["sg", "docker", "-c", shlex.join(["docker", *arguments])],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds,
            ).returncode

        try:
            if docker(
                [
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    name,
                    "--network=none",
                    "--cpus=1",
                    "--memory=2g",
                    row["image_name"],
                    "sleep",
                    "75",
                ]
            ):
                raise ValueError("navigation container did not start")
            if docker(
                [
                    "cp",
                    str(output / "reference_check.py"),
                    name + ":/tmp/benchmark_reference.py",
                ]
            ):
                raise ValueError("navigation reference check copy failed")
            terminal["test_exit"] = docker(
                [
                    "exec",
                    "-w",
                    "/testbed",
                    name,
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=short",
                    *answer["tests"],
                    "/tmp/benchmark_reference.py",
                    "--junitxml=/tmp/tests.xml",
                ],
                50,
            )
            if docker(["cp", name + ":/tmp/tests.xml", str(output / "tests.xml")]):
                raise ValueError("navigation tests did not produce observations")
            terminal.update(navigation_observations(output))
            if terminal["test_exit"] not in (0, 1):
                raise ValueError("navigation tests failed outside ordinary assertions")
        except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as error:
            terminal.update(quality="ungraded", error=str(error))
        finally:
            try:
                docker(["rm", "-f", name])
            except (OSError, subprocess.TimeoutExpired) as error:
                terminal.update(
                    quality="ungraded", error="owned cleanup failed: " + str(error)
                )
    return finish_grade(output, terminal, started)


def verify_navigation(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if admission["schema"] != "benchmark-navigation-grade-v1":
        raise ValueError("unsupported navigation grading receipt")
    if not 0 < terminal["elapsed_seconds"] <= admission["budget_seconds"] + 1:
        raise ValueError("navigation grading exceeded its budget")
    if terminal["admission_sha256"] != digest_file(output / "admission.json"):
        raise ValueError("navigation admission drift")
    paths = {path.name for path in output.iterdir() if path.is_file()} - {
        "admission.json",
        "terminal.json",
    }
    if paths != set(terminal["artifacts_sha256"]):
        raise ValueError("navigation artifact set changed")
    for name, expected in terminal["artifacts_sha256"].items():
        if digest_file(output / name) != expected:
            raise ValueError("navigation evidence drift")
    if (
        digest_file(output / "dataset.json") != admission["dataset_sha256"]
        or digest_file(output / "submitted.patch") != admission["submitted_sha256"]
    ):
        raise ValueError("navigation input drift")
    rows = [
        row
        for row in json.loads((output / "dataset.json").read_text())
        if row["instance_id"] == admission["instance"]
    ]
    if (
        len(rows) != 1
        or rows[0]["rubric"] != admission["rubric"]
        or rows[0]["image_name"] != admission["image"]["id"]
    ):
        raise ValueError("navigation rubric differs from source")
    if (output / "reference_check.py").read_text() != rows[0]["reference_check"]:
        raise ValueError("navigation reference check differs from source")
    try:
        answer = navigation_answer(output / "submitted.patch")
    except (ValueError, KeyError, TypeError):
        if terminal["quality"] != "unresolved" or "rubric_refusal" not in terminal:
            raise ValueError("invalid navigation refusal")
        return terminal
    if terminal["quality"] == "ungraded":
        raise ValueError("navigation grading incomplete")
    observed = (
        navigation_observations(output)
        if check_answer(answer, admission["rubric"])
        else {"quality": "unresolved", "answer_matches": False}
    )
    if any(terminal.get(key) != value for key, value in observed.items()):
        raise ValueError("navigation verdict differs from executable observations")
    if terminal["quality"] == "resolved" and terminal["test_exit"] != 0:
        raise ValueError("navigation success without passing tests")
    return terminal
