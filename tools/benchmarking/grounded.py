from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from tools.benchmarking.answers import added_json
from tools.benchmarking.probe import digest_file
from tools.benchmarking.regression import finish_grade
from tools.runtime_verify import write_record

ANSWER_FILE = "benchmark_grounded.json"


def grounded_observation(
    answer: dict[str, Any], rubric: dict[str, Any], sources: dict[str, str]
) -> dict[str, Any]:
    if set(answer) != {"answers", "citations"} or json.dumps(
        answer["answers"], sort_keys=True
    ) != json.dumps(rubric["answers"], sort_keys=True):
        return {
            "quality": "unresolved",
            "reason": "answer differs from the frozen fact rubric",
        }
    citations = answer["citations"]
    if not isinstance(citations, dict) or set(citations) != set(rubric["support"]):
        return {"quality": "unresolved", "reason": "missing or extra fact citations"}
    spans = {}
    for key, citation in citations.items():
        if not isinstance(citation, dict) or set(citation) != {"file", "start", "end"}:
            return {"quality": "unresolved", "reason": "invalid citation format"}
        filename, first, last = citation["file"], citation["start"], citation["end"]
        if (
            not isinstance(filename, str)
            or filename not in sources
            or type(first) is not int
            or type(last) is not int
            or not 1 <= first <= last <= len(sources[filename].splitlines())
            or last - first >= 12
        ):
            return {
                "quality": "unresolved",
                "reason": "citation exceeds its admitted source or span limit",
            }
        acceptable = rubric["support"][key]
        if not any(
            filename == span["file"] and first <= span["start"] <= span["end"] <= last
            for span in acceptable
        ):
            return {
                "quality": "unresolved",
                "reason": "citation does not support its associated fact",
            }
        spans[key] = {
            "citation": citation,
            "text": "\n".join(sources[filename].splitlines()[first - 1 : last]),
        }
    return {"quality": "resolved", "citations": spans}


def grade_grounded(
    root: Path, dataset: Path, prediction: Path, instance: str, output: Path
) -> dict[str, Any]:
    if output.resolve().is_relative_to(root.resolve()) or output.exists():
        raise ValueError("grounded grading requires a new external output")
    rows = [
        row for row in json.loads(dataset.read_text()) if row["instance_id"] == instance
    ]
    if len(rows) != 1:
        raise ValueError("grounded grading requires one admitted instance")
    row = rows[0]
    sources = {}
    for name, expected in row["sources_sha256"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid grounded source filename")
        path = dataset.parent / name
        if digest_file(path) != expected:
            raise ValueError("grounded document differs from admitted bytes")
        sources[name] = path.read_text()
    output.mkdir(parents=True)
    (output / "dataset.json").write_bytes(dataset.read_bytes())
    (output / "submitted.patch").write_bytes(prediction.read_bytes())
    (output / "sources").mkdir()
    for name in sources:
        (output / "sources" / name).write_bytes((dataset.parent / name).read_bytes())
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-grounded-grade-v1",
            "instance": instance,
            "dataset_sha256": digest_file(output / "dataset.json"),
            "submitted_sha256": digest_file(output / "submitted.patch"),
            "sources_sha256": row["sources_sha256"],
            "rubric": row["rubric"],
            "image": {"id": row["image_name"]},
            "scope": "project-authored fact and citation rubric on immutable public source documents",
        },
    )
    started = time.monotonic()
    terminal: dict[str, Any] = {"quality": "unresolved"}
    try:
        terminal.update(
            grounded_observation(
                added_json(output / "submitted.patch", ANSWER_FILE),
                row["rubric"],
                sources,
            )
        )
    except (ValueError, KeyError, TypeError) as error:
        terminal["rubric_refusal"] = str(error)
    return finish_grade(output, terminal, started)


def verify_grounded(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if admission["schema"] != "benchmark-grounded-grade-v1" or terminal[
        "admission_sha256"
    ] != digest_file(output / "admission.json"):
        raise ValueError("invalid grounded admission")
    actual = {
        path.name: digest_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name not in ("admission.json", "terminal.json")
    }
    if (
        actual != terminal["artifacts_sha256"]
        or digest_file(output / "dataset.json") != admission["dataset_sha256"]
        or digest_file(output / "submitted.patch") != admission["submitted_sha256"]
    ):
        raise ValueError("grounded evidence drift")
    sources = {}
    if {path.name for path in (output / "sources").iterdir()} != set(
        admission["sources_sha256"]
    ):
        raise ValueError("grounded source set changed")
    for name, expected in admission["sources_sha256"].items():
        path = output / "sources" / name
        if digest_file(path) != expected:
            raise ValueError("grounded source drift")
        sources[name] = path.read_text()
    rows = [
        row
        for row in json.loads((output / "dataset.json").read_text())
        if row["instance_id"] == admission["instance"]
    ]
    if (
        len(rows) != 1
        or rows[0]["rubric"] != admission["rubric"]
        or rows[0]["sources_sha256"] != admission["sources_sha256"]
        or rows[0]["image_name"] != admission["image"]["id"]
    ):
        raise ValueError("grounded rubric differs from source")
    try:
        derived = grounded_observation(
            added_json(output / "submitted.patch", ANSWER_FILE),
            admission["rubric"],
            sources,
        )
    except (ValueError, KeyError, TypeError):
        if terminal["quality"] != "unresolved" or "rubric_refusal" not in terminal:
            raise ValueError("invalid grounded refusal")
        return terminal
    if any(terminal.get(key) != value for key, value in derived.items()):
        raise ValueError("grounded verdict differs from retained facts and citations")
    return terminal
