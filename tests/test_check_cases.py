"""Versioned positive/negative controls, isolated outside the working repository."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    id: str
    component: str
    good: dict[str, str | None]
    bad: dict[str, str | None]
    diagnostic: str


def cases() -> list[Case]:
    raw = cast(
        list[dict[str, object]],
        json.loads((ROOT / "tools/check-cases.json").read_text()),
    )
    return [
        Case(
            str(c["id"]),
            str(c["component"]),
            cast(dict[str, str | None], c["good"]),
            cast(dict[str, str | None], c["bad"]),
            str(c["diagnostic"]),
        )
        for c in raw
    ]


def write(root: Path, files: dict[str, str | None]) -> None:
    for name, data in files.items():
        target = root / name
        if data is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if data.startswith("$SYMLINK:"):
                target.symlink_to(data.removeprefix("$SYMLINK:"))
                continue
            if data == "$TRACKED_MISSING":
                target.write_text("tracked specimen\n")
                subprocess.run(["git", "add", "-f", name], cwd=root, check=True)
                target.unlink()
                continue
            tracked = data.startswith("$TRACKED:")
            data = data.removeprefix("$TRACKED:")
            data = data.replace("$OVERSIZE", "x" * (256 * 1024 + 1))
            target.write_text(data.replace("$FAKE_TOKEN", "ghp_" + "A" * 36))
            if tracked:
                subprocess.run(["git", "add", "-f", name], cwd=root, check=True)


def prepare(root: Path) -> None:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    config = (
        (ROOT / "pyproject.toml")
        .read_text()
        .replace(
            'files = ["semantic_offload", "benchmarks", "tools", "tests"]',
            'files = ["probe.py"]',
        )
    )
    write(
        root,
        {
            "pyproject.toml": config,
            "AGENTS.md": "Research instructions.\n",
            "CLAUDE.md": "@AGENTS.md\n",
            "probe.py": '"""Valid control."""\n',
            "semantic_offload/__init__.py": "",
            "semantic_offload/contracts.py": "",
            "semantic_offload/vllm_adapter/__init__.py": "",
            "benchmarks/__init__.py": "",
            "vllm/__init__.py": "",
            "tools/native.json": (ROOT / "tools/native.json").read_text(),
            "test_probe.py": "def test_control() -> None:\n    assert 1 + 1 == 2\n",
            "cpu.json": '["test_probe.py::test_control"]\n',
        },
    )


def invoke(root: Path, component: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(ROOT)])
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["MYPY_CACHE_DIR"] = str(root.parent.parent / "mypy-cache")
    if component == "cpu":
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tools.pytest_gate",
            "--gate-manifest",
            "cpu.json",
            "test_probe.py",
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "tools/check.py"),
            "--component",
            component,
            "--root",
            str(root),
        ]
    return subprocess.run(
        command, cwd=root, env=env, text=True, capture_output=True, timeout=60
    )


def require_rejection(code: int, output: str, diagnostic: str) -> None:
    assert code != 0, f"Violation was accepted; expected {diagnostic}"
    assert diagnostic in output, f"Expected {diagnostic}; got:\n{output}"
    assert "Traceback (most recent call last)" not in output
    assert "tool.missing:" not in output


@pytest.fixture(scope="session")
def proof_workspace() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="vllm-cache-proof-") as directory:
        yield Path(directory)


@pytest.mark.parametrize("case", cases(), ids=lambda c: c.id)
def test_rule_rejects_specimen(case: Case, proof_workspace: Path) -> None:
    tmp_path = proof_workspace / case.id
    tmp_path.mkdir()
    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    for root, changes in ((positive, case.good), (negative, case.bad)):
        prepare(root)
        write(root, changes)
    good = invoke(positive, case.component)
    assert good.returncode == 0, good.stdout + good.stderr
    bad = invoke(negative, case.component)
    require_rejection(bad.returncode, bad.stdout + bad.stderr, case.diagnostic)
    print(
        f"REJECTED {case.id}: {case.diagnostic} (exit {bad.returncode}); positive control accepted"
    )


def require_manifest(required: set[str], actual: list[str]) -> None:
    assert len(actual) == len(set(actual)), "self.manifest: duplicate specimen"
    assert set(actual) == required, "self.manifest: missing or unregistered specimen"


def test_manifest_covers_adopted_rules() -> None:
    # Independent registry means deleting a specimen cannot quietly reduce coverage.
    required = set(json.loads((ROOT / "tools/rules.json").read_text()))
    actual = [c.id for c in cases()]
    require_manifest(required, actual)


@pytest.mark.parametrize("actual", [[], ["expected", "expected"], ["unexpected"]])
def test_manifest_rejects_missing_duplicate_extra(actual: list[str]) -> None:
    with pytest.raises(AssertionError, match="self.manifest:"):
        require_manifest({"expected"}, actual)


@pytest.mark.parametrize(
    "code,output",
    [(0, "expected"), (1, "unrelated failure"), (1, "tool.missing: expected")],
)
def test_rejection_proof_cannot_be_faked(code: int, output: str) -> None:
    with pytest.raises(AssertionError):
        require_rejection(code, output, "expected")
