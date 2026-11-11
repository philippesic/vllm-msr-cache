from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import results


def specimen(root: Path) -> Path:
    source = {
        "repository": "current",
        "revision": "a" * 40,
        "path": "report.md",
        "sha256": results.digest(b"measured result"),
    }
    data: dict[str, Any] = {
        "schema": 1,
        "title": "A bounded result",
        "date": "2026-11-10",
        "origin": "current",
        "verdict": "mixed",
        "summary": "Observed benefit with a cost.",
        "scope": "One fixed input.",
        "limitations": "No population inference.",
        "decision": "Retain the result.",
        "sources": [source],
        "correction": "",
        "chart": {
            "metric": "Throughput ratio",
            "caption": "One point, no interval.",
            "baseline": 1,
            "points": [
                {"label": "Candidate", "estimate": 1.1, "lower": None, "upper": None}
            ],
        },
    }
    path = root / "results/case/result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    archive = root / "results/sources" / (source["sha256"] + ".json")
    archive.parent.mkdir()
    archive.write_text(json.dumps({"source": source, "content": "measured result"}))
    return path


def test_render_is_reproducible_and_tampering_refuses(tmp_path: Path) -> None:
    path = specimen(tmp_path)
    results.render(tmp_path)
    first = (path.parent / "chart.svg").read_bytes()
    results.render(tmp_path)
    assert (path.parent / "chart.svg").read_bytes() == first
    assert results.check(tmp_path) == []
    (path.parent / "chart.svg").write_bytes(first + b"changed")
    assert "changed output" in results.check(tmp_path)[0]
    results.render(tmp_path)
    data = json.loads(path.read_text())
    data["chart"]["points"][0]["estimate"] = 1.2
    path.write_text(json.dumps(data))
    assert "stale rendering" in results.check(tmp_path)[0]
    results.render(tmp_path)
    assert (path.parent / "chart.svg").read_bytes() != first
    archive = next((tmp_path / "results/sources").glob("*.json"))
    payload = json.loads(archive.read_text())
    payload["content"] = "invented result"
    archive.write_text(json.dumps(payload))
    assert "source archive mismatch" in results.check(tmp_path)[0]


def test_unknown_schema_and_invalid_intervals_refuse(tmp_path: Path) -> None:
    path = specimen(tmp_path)
    original = json.loads(path.read_text())
    data = {**original, "schema": 99}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        results.read_result(path)
    original["chart"]["points"][0].update(lower=1.2, upper=1.3)
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="interval"):
        results.read_result(path)


def test_source_recovery_checks_committed_bytes(tmp_path: Path) -> None:
    path = specimen(tmp_path)
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "report.md").write_text("measured result")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Fixture",
        ],
        cwd=repository,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    data = json.loads(path.read_text())
    data["sources"][0]["revision"] = revision
    path.write_text(json.dumps(data))
    results.verify_sources(tmp_path, repository)
    data["sources"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        results.verify_sources(tmp_path, repository)
