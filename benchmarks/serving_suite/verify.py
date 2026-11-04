"""Offline integrity and data-completeness verification; no GPU or model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .contracts import SCHEMA, write_json
from .report import summarize_requests


def seal(directory):
    directory = Path(directory)
    files = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name != "artifacts.json":
            with p.open("rb") as f:
                files[str(p.relative_to(directory))] = hashlib.file_digest(
                    f, "sha256"
                ).hexdigest()
    write_json(directory / "artifacts.json", {"schema": SCHEMA, "sha256": files})


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "artifacts.json").read_text())
    for name, expected in manifest["sha256"].items():
        p = directory / name
        if not p.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Artifact escapes result directory")
        if not p.is_file():
            raise ValueError(f"Missing artifact: {name}")
        with p.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Changed artifact: {name}")
    result = json.loads((directory / "result.json").read_text())
    if result["schema"] != SCHEMA:
        raise ValueError("Unsupported result schema")
    count = 0
    for case in result["cases"]:
        rows = [
            json.loads(line)
            for line in (directory / case["name"] / "requests.jsonl")
            .read_text()
            .splitlines()
        ]
        recomputed = summarize_requests(rows, case["load_duration_s"])
        keys = [
            (r.get("wave"), r.get("source_id"), r.get("turn"))
            for r in rows
            if not r.get("cancel_probe")
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate request identity")
        if case["evaluation"]["status"] == "passed" and case.get("source_ids"):
            expected = {
                (w, s, t)
                for w in range(result["plan"]["waves"])
                for s in case["source_ids"]
                for t in range(result["plan"]["turns"])
            }
            if set(keys) != expected:
                raise ValueError("Incomplete planned request identities")
        for key in (
            "completed",
            "failed",
            "output_tokens",
            "finish_reasons",
            "prompt_tokens",
            "content_checks",
            "content_checks_failed",
        ):
            if recomputed[key] != case["summary"][key]:
                raise ValueError(f"Wrong request aggregate: {case['name']}/{key}")
        for key in ("output_tokens_per_s", "ttft_mean_s", "ttft_p95_s", "e2e_p95_s"):
            a, b = recomputed[key], case["summary"][key]
            if (a is None) != (b is None) or (
                a is not None and not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
            ):
                raise ValueError(f"Wrong timing aggregate: {case['name']}/{key}")
        if case["evaluation"]["status"] == "passed":
            if (
                recomputed["completed"] != case["expected_requests"]
                or recomputed["failed"]
            ):
                raise ValueError("Passed case has incomplete requests")
            if not case["cleanup"]["verified"]:
                raise ValueError("Passed case has unverified cleanup")
            for name in (
                "metrics-before.prom",
                "metrics-after.prom",
                "resources.jsonl",
            ):
                if not (directory / case["name"] / name).is_file():
                    raise ValueError(f"Missing required measurement: {name}")
        count += len(rows)
    return {
        "verified_files": len(manifest["sha256"]),
        "cases": len(result["cases"]),
        "request_records": count,
        "measurement_status": result["status"],
        "scope": "Artifact integrity and aggregate consistency; not a new GPU/quality qualification.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.directory), indent=2))


if __name__ == "__main__":
    main()
