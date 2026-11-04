"""Portable coordination and benchmark contracts; no CUDA/vLLM imports."""

import subprocess
import sys

import pytest

from benchmarks.research_checks.http import verify_http
from benchmarks.serving_suite.contracts import fixture_corpus, validate_corpus
from harness.gpu_lock import GPULock


def test_cross_process_reservation_and_release(tmp_path):
    path = tmp_path / "lock"
    command = [
        sys.executable,
        "-c",
        f"from harness.gpu_lock import GPULock;\nwith GPULock({str(path)!r}): pass",
    ]
    with GPULock(path):
        assert subprocess.run(command, capture_output=True).returncode != 0
    assert subprocess.run(command, capture_output=True).returncode == 0


def test_failed_second_lock_releases_first(tmp_path, monkeypatch):
    import harness.gpu_lock as module

    first, legacy = tmp_path / "first", tmp_path / "legacy"
    monkeypatch.setattr(module, "GPU_LOCK", first)
    monkeypatch.setattr(module, "LEGACY_LOCK", legacy)
    with GPULock(legacy):
        with pytest.raises(RuntimeError):
            with GPULock():
                pass
        with GPULock(first):
            pass


def test_supervisor_protects_gap_and_allows_legacy_child(tmp_path, monkeypatch):
    import harness.gpu_lock as module

    first, legacy = tmp_path / "first", tmp_path / "legacy"
    monkeypatch.setattr(module, "GPU_LOCK", first)
    monkeypatch.setattr(module, "LEGACY_LOCK", legacy)

    @module.gpu_reserved
    def run():
        with pytest.raises(RuntimeError):
            with GPULock(first):
                pass
        with GPULock(legacy):
            pass

    run()
    with GPULock():
        pass


def test_fixture_provenance_and_duplicate_rejection():
    corpus = fixture_corpus(2)
    assert corpus["synthetic"] and corpus["provenance"]
    corpus["sessions"].append(corpus["sessions"][0])
    with pytest.raises(ValueError):
        validate_corpus(corpus)


def test_http_verifier_fails_closed_without_records():
    assert not verify_http({})["qualified"]
