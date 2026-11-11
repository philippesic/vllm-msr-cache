from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

from tools.config import check_environment
from tools.process import process_identity, run_owned, signal_owned
from tools.runtime_verify import snapshot, verify, write_record


def test_owned_timeout_and_failed_startup(tmp_path: Path) -> None:
    log = tmp_path / "child.log"
    pid_file = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        "import os,signal,pathlib; pathlib.Path('pid').write_text(str(os.getpid())); signal.pause()",
    ]
    with log.open("w") as output:
        assert (
            run_owned(command, check_environment(tmp_path), tmp_path, 0.2, output)
            == 124
        )
    assert process_identity(int(pid_file.read_text())) is None
    with pytest.raises(FileNotFoundError):
        run_owned([str(tmp_path / "missing")], {}, tmp_path, 0.1)
    with pytest.raises(ValueError):
        run_owned([sys.executable], {}, tmp_path, 0)


def test_stale_control_does_not_signal_current_process() -> None:
    import os

    identity = process_identity(os.getpid())
    assert identity is not None
    wrong = (identity[0], identity[1] + 1, identity[2], identity[3])
    with pytest.raises(RuntimeError, match="identity changed"):
        signal_owned(os.getpid(), wrong, signal.SIGTERM)


def test_snapshot_includes_dirty_and_untracked_inputs(tmp_path: Path) -> None:
    import subprocess

    source = tmp_path / "repo"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "tracked.py").write_text("before")
    subprocess.run(["git", "add", "tracked.py"], cwd=source, check=True)
    (source / "tracked.py").write_text("dirty")
    (source / "generated.py").write_text("generated")
    identities = snapshot(source, tmp_path / "snapshot")
    assert set(identities) == {"tracked.py", "generated.py"}
    assert (tmp_path / "snapshot/tracked.py").read_text() == "dirty"


def test_receipt_unknown_incomplete_and_tampered_evidence_refuse(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
    write_record(tmp_path / "admission.json", {"schema": "unknown"})
    write_record(tmp_path / "terminal.json", {"schema": "functional-terminal-v1"})
    with pytest.raises(ValueError, match="version"):
        verify(tmp_path)
    with pytest.raises(FileExistsError):
        write_record(tmp_path / "terminal.json", {})
    (tmp_path / "admission.json").write_text(
        json.dumps(
            {"schema": "functional-admission-v1", "source_sha256": {}, "gpu": False}
        )
    )
    (tmp_path / "terminal.json").write_text(
        json.dumps(
            {"schema": "functional-terminal-v1", "exit_code": 1, "artifacts_sha256": {}}
        )
    )
    with pytest.raises(ValueError, match="qualify"):
        verify(tmp_path)


def test_runtime_pin_matches_lock_files() -> None:
    from wait_hint_offload.runtime_pin import PYTHON_VERSION, VLLM_VERSION

    root = Path(__file__).resolve().parents[1]
    wheel = f"/vllm-{VLLM_VERSION.replace('+', '%2B')}-"
    for pattern in ("requirements.in", "requirements-*.txt"):
        pins = [
            line
            for path in sorted((root / "runtime").glob(pattern))
            for line in path.read_text().splitlines()
            if line.startswith("vllm @ ")
        ]
        assert len(pins) == 1 and wheel in pins[0], pattern
    python = ".".join(map(str, PYTHON_VERSION))
    assert (root / ".python-version").read_text().strip() == python
    assert f'requires-python = "=={python}"' in (root / "pyproject.toml").read_text()


def test_type_caches_separate_portable_and_runtime_environments(tmp_path: Path) -> None:
    from tools.config import type_cache

    portable = type_cache(tmp_path)
    runtime = type_cache(tmp_path, tmp_path / "runtime/bin/python")
    other = type_cache(tmp_path, tmp_path / "other/bin/python")
    assert len({portable, runtime, other}) == 3
    assert runtime == type_cache(tmp_path, tmp_path / "runtime/bin/python")
