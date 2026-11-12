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

from tools.benchmarking.harness import pin_swebench_image
from tools.benchmarking.probe import digest_file
from tools.config import output_overwrites_source
from tools.runtime_verify import write_record

TEST_FILE = "benchmark_generated_test.py"


def regression_outcome(before: Path, after: Path) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for path in (before, after):
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("regression test report exceeds size limit")
        tree = ET.parse(path)
        cases = list(tree.iter("testcase"))
        identities = [(case.get("classname"), case.get("name")) for case in cases]
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("empty or duplicate regression test observations")
        observations.append(
            {
                "identities": [list(identity) for identity in identities],
                "failures": sum(case.find("failure") is not None for case in cases),
                "errors": sum(case.find("error") is not None for case in cases),
                "skips": sum(case.find("skipped") is not None for case in cases),
            }
        )
    first, second = observations
    if {tuple(item) for item in first["identities"]} != {
        tuple(item) for item in second["identities"]
    }:
        raise ValueError("regression test identities changed between versions")
    resolved = (
        first["failures"] > 0
        and first["errors"] == second["errors"] == 0
        and first["skips"] == second["skips"] == 0
        and second["failures"] == 0
    )
    return {
        "quality": "resolved" if resolved else "unresolved",
        "before": first,
        "after": second,
    }


def validate_test_patch(patch: Path) -> None:
    result = subprocess.run(
        ["git", "apply", "--numstat", str(patch.resolve())],
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [line.split("\t") for line in result.stdout.splitlines()]
    if (
        result.returncode != 0
        or len(rows) != 1
        or len(rows[0]) != 3
        or rows[0][2] != TEST_FILE
        or not rows[0][0].isdigit()
        or int(rows[0][0]) <= 0
        or rows[0][1] != "0"
        or "new file mode 100644\n" not in patch.read_text()
    ):
        raise ValueError("submission must only add the designated regression test")


def finish_grade(
    output: Path, terminal: dict[str, Any], started: float
) -> dict[str, Any]:
    terminal["elapsed_seconds"] = time.monotonic() - started
    terminal["admission_sha256"] = digest_file(output / "admission.json")
    terminal["artifacts_sha256"] = {
        path.name: digest_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "admission.json"
    }
    write_record(output / "terminal.json", terminal)
    return terminal


def grade_regression(
    root: Path, dataset: Path, prediction: Path, instance: str, output: Path
) -> dict[str, Any]:
    if output_overwrites_source(root, output) or output.exists():
        raise ValueError("regression grading requires a new artifact output")
    dataset_bytes = dataset.read_bytes()
    selected = [
        row for row in json.loads(dataset_bytes) if row["instance_id"] == instance
    ]
    if len(selected) != 1:
        raise ValueError("regression grading requires one source issue")
    image = pin_swebench_image(instance)
    output.mkdir(parents=True)
    (output / "dataset.json").write_bytes(dataset_bytes)
    (output / "submitted.patch").write_bytes(prediction.read_bytes())
    (output / "reference.patch").write_text(selected[0]["patch"])
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-regression-grade-v1",
            "instance": instance,
            "dataset_sha256": digest_file(output / "dataset.json"),
            "retained_dataset": "dataset.json",
            "image": image,
            "submitted_sha256": digest_file(output / "submitted.patch"),
            "reference_sha256": digest_file(output / "reference.patch"),
            "test_file": TEST_FILE,
            "budget_seconds": 165,
            "scope": "project-authored fail-before/pass-after rubric; not an official benchmark score",
        },
    )
    name = "wait-hint-regression-" + uuid.uuid4().hex
    terminal: dict[str, Any] = {"quality": "ungraded", "error": None}
    started = time.monotonic()
    try:
        validate_test_patch(output / "submitted.patch")
    except ValueError as error:
        terminal.update(quality="unresolved", rubric_refusal=str(error))
        (output / "execution.log").touch(exist_ok=False)
        return finish_grade(output, terminal, started)
    with (output / "execution.log").open("x") as log:

        def docker(arguments: list[str], seconds: int = 15) -> int:
            if arguments[0] != "rm":
                seconds = min(seconds, int(150 - (time.monotonic() - started)))
                if seconds <= 0:
                    raise ValueError("regression grading budget exhausted")
            completed = subprocess.run(
                ["sg", "docker", "-c", shlex.join(["docker", *arguments])],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds,
            )
            return completed.returncode

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
                    image["id"],
                    "sleep",
                    "150",
                ],
                30,
            ):
                raise ValueError("regression container did not start")
            for filename in ("submitted.patch", "reference.patch"):
                if docker(["cp", str(output / filename), f"{name}:/tmp/{filename}"]):
                    raise ValueError("regression patch copy failed")
            if docker(
                ["exec", "-w", "/testbed", name, "git", "apply", "/tmp/submitted.patch"]
            ):
                raise ValueError("submitted regression test did not apply")
            exits = []
            for phase in ("before", "after"):
                if phase == "after" and docker(
                    [
                        "exec",
                        "-w",
                        "/testbed",
                        name,
                        "git",
                        "apply",
                        "/tmp/reference.patch",
                    ]
                ):
                    raise ValueError("reference fix did not apply")
                command = (
                    "source /opt/miniconda3/bin/activate && conda activate testbed && "
                    f"python -m pytest -q --tb=short {TEST_FILE} --junitxml=/tmp/{phase}.xml"
                )
                exits.append(
                    docker(
                        [
                            "exec",
                            "-w",
                            "/testbed",
                            name,
                            "timeout",
                            "45",
                            "bash",
                            "-c",
                            command,
                        ],
                        50,
                    )
                )
                if docker(
                    ["cp", f"{name}:/tmp/{phase}.xml", str(output / f"{phase}.xml")]
                ):
                    raise ValueError("regression test report missing")
            terminal.update(
                regression_outcome(output / "before.xml", output / "after.xml")
            )
            terminal["test_exits"] = exits
            if exits not in ([1, 0], [0, 0], [1, 1]):
                terminal["quality"] = "ungraded"
                terminal["error"] = (
                    "test execution failed outside an ordinary assertion"
                )
        except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as error:
            terminal["error"] = str(error)
        finally:
            try:
                docker(["rm", "-f", name])
            except (OSError, subprocess.TimeoutExpired) as error:
                terminal["quality"] = "ungraded"
                terminal["error"] = "owned cleanup failed: " + str(error)
    return finish_grade(output, terminal, started)


def verify_regression(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if admission["schema"] != "benchmark-regression-grade-v1":
        raise ValueError("unsupported regression grading receipt")
    if terminal["admission_sha256"] != digest_file(output / "admission.json"):
        raise ValueError("regression admission drift")
    required = {
        "submitted.patch",
        "reference.patch",
        "execution.log",
    }
    if "rubric_refusal" not in terminal:
        required.update(("before.xml", "after.xml"))
    if "retained_dataset" in admission:
        if admission["retained_dataset"] != "dataset.json":
            raise ValueError("unsupported retained regression dataset")
        required.add("dataset.json")
        rows = json.loads((output / "dataset.json").read_text())
        source = [row for row in rows if row["instance_id"] == admission["instance"]]
        if (
            digest_file(output / "dataset.json") != admission["dataset_sha256"]
            or len(source) != 1
            or source[0]["patch"] != (output / "reference.patch").read_text()
        ):
            raise ValueError("regression reference differs from admitted source")
    if (
        set(terminal["artifacts_sha256"]) != required
        or not 0 <= terminal["elapsed_seconds"] <= admission["budget_seconds"] + 5
        or digest_file(output / "submitted.patch") != admission["submitted_sha256"]
        or digest_file(output / "reference.patch") != admission["reference_sha256"]
    ):
        raise ValueError("incomplete regression grading receipt")
    for filename, digest in terminal["artifacts_sha256"].items():
        if (
            not re.fullmatch(r"[a-z.]+", filename)
            or digest_file(output / filename) != digest
        ):
            raise ValueError("regression artifact drift")
    if "rubric_refusal" in terminal:
        try:
            validate_test_patch(output / "submitted.patch")
        except ValueError:
            if terminal["quality"] != "unresolved":
                raise ValueError("invalid regression refusal verdict") from None
            return terminal
        raise ValueError("regression rubric refused a valid submission")
    validate_test_patch(output / "submitted.patch")
    derived = regression_outcome(output / "before.xml", output / "after.xml")
    if derived["quality"] == "resolved" and terminal["test_exits"] != [1, 0]:
        raise ValueError("regression test exit status disagrees with observations")
    if terminal["quality"] == "ungraded" or any(
        terminal[key] != value for key, value in derived.items()
    ):
        raise ValueError("regression verdict disagrees with observations")
    return terminal
