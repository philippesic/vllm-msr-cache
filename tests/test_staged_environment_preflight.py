# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import staged_environment_preflight as preflight


def _git_repo(path: Path) -> dict[str, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "marker").write_text("fixture")
    subprocess.run(["git", "-C", str(path), "add", "marker"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True
    )
    return {
        "path": str(path),
        "revision": subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tree": subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def test_actual_preflight_entrypoint_aggregates_independent_failures(tmp_path):
    valid_repo = _git_repo(tmp_path / "valid-repo")
    supervisor = tmp_path / "supervisor.json"
    supervisor.write_text("{}")
    missing_input = Path("/home/preflight-fixture/missing-input.json")
    contexts = [
        {
            "id": "missing-native",
            "source": valid_repo,
            "runtime": valid_repo,
            "interpreter": "/usr/bin/python3",
            "cwd": str(tmp_path),
            "pythonpath": ["/home/preflight-fixture/empty-pythonpath"],
            "environment": {},
            "base_binary_inventory": {
                "path": "/home/preflight-fixture/missing-inventory.json",
                "sha256": "0" * 64,
            },
            "native_modules": ["preflight_definitely_missing_native_module"],
        },
        {
            "id": "invalid-runtime",
            "source": valid_repo,
            "runtime": {
                "path": "/home/preflight-fixture/missing-runtime",
                "revision": "1" * 40,
                "tree": "2" * 40,
            },
            "interpreter": "/usr/bin/python3",
            "cwd": str(tmp_path),
            "pythonpath": ["/home/preflight-fixture/missing-runtime"],
            "environment": {},
            "base_binary_inventory": {
                "path": "/home/preflight-fixture/missing-inventory.json",
                "sha256": "0" * 64,
            },
            "native_modules": ["preflight_definitely_missing_native_module"],
        },
    ]
    manifest = {
        "schema": preflight.SCHEMA,
        "execution_route": "sc-gpu tmux fixture",
        "inputs": [{"runtime_path": str(missing_input), "sha256": "0" * 64}],
        "contexts": contexts,
        "validators": [
            {
                "id": "blocked-validator",
                "context_id": "missing-native",
                "argv": ["/usr/bin/python3", "-c", "print('validate')"],
            }
        ],
        "execution_contract": {
            "output_paths": ["/home/preflight-fixture/fresh-output/result.json"],
            "geometry": {
                "gpu_blocks": 640,
                "cpu_bytes": 256 * 1024 * 1024,
                "block_size_tokens": 16,
                "max_model_len": 8192,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 8192,
            },
            "maximum_prompt_tokens": 4828,
            "max_new_tokens": 32,
            "supervisor": {
                "path": str(supervisor),
                "sha256": preflight.file_digest(supervisor),
                "route": "sc-gpu",
                "first_failure_stop": True,
                "automatic_retries": 0,
            },
        },
    }
    manifest["manifest_digest"] = preflight.digest(manifest)
    manifest_path = tmp_path / "preflight.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest))

    assert (
        preflight.main(
            [
                f"--manifest={manifest_path}",
                f"--output={output_path}",
            ]
        )
        == 1
    )
    report = json.loads(output_path.read_text())

    assert report["valid"] is False
    assert "input_closure" in report["failed_checks"]
    assert "missing-native:native_imports" in report["failed_checks"]
    assert "invalid-runtime:identity" in report["failed_checks"]
    assert "invalid-runtime:native_imports" in report["blocked_checks"]
    assert "validators" in report["blocked_checks"]
    assert len(report["failed_checks"]) >= 3
    input_check = next(
        item for item in report["checks"] if item["id"] == "input_closure"
    )
    assert any(
        "missing runtime input" in error for error in input_check["evidence"]["errors"]
    )


def test_preflight_rejects_model_execution_as_validator(tmp_path):
    manifest = {
        "validators": [
            {
                "id": "unsafe",
                "context_id": "context",
                "argv": ["python", "execute-cell"],
            }
        ],
        "contexts": [
            {
                "id": "context",
                "cwd": str(tmp_path),
                "interpreter": "/usr/bin/python3",
                "pythonpath": ["/home/context"],
            }
        ],
    }

    try:
        preflight._check_validators(manifest)
    except preflight.PreflightError as exc:
        assert "unsafe validator argv" in str(exc)
    else:
        raise AssertionError("model execution was accepted as a preflight validator")


def test_validator_failure_preserves_every_command_result(tmp_path):
    context = {
        "id": "context",
        "cwd": str(tmp_path),
        "interpreter": "/usr/bin/python3",
        "pythonpath": ["/home/context"],
    }
    manifest = {
        "contexts": [context],
        "validators": [
            {
                "id": "fails",
                "context_id": "context",
                "argv": ["/usr/bin/python3", "-c", "raise SystemExit(7)"],
            },
            {
                "id": "passes",
                "context_id": "context",
                "argv": ["/usr/bin/python3", "-c", "print('ok')"],
            },
        ],
    }

    try:
        preflight._check_validators(manifest)
    except preflight.CheckFailure as exc:
        assert exc.evidence["errors"] == ["validator failed: fails"]
        assert [result["id"] for result in exc.evidence["results"]] == [
            "fails",
            "passes",
        ]
        assert [result["returncode"] for result in exc.evidence["results"]] == [
            7,
            0,
        ]
    else:
        raise AssertionError("validator failure did not preserve evidence")


def test_execution_contract_accepts_only_frozen_functional_or_baseline_geometry(
    tmp_path,
):
    supervisor = tmp_path / "supervisor.json"
    supervisor.write_text("{}")
    contract = {
        "output_paths": ["/home/preflight-functional/fresh-result.json"],
        "geometry": {
            "gpu_blocks": 320,
            "cpu_bytes": 256 * 1024 * 1024,
            "block_size_tokens": 16,
            "max_model_len": 5120,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
        },
        "maximum_prompt_tokens": 4828,
        "max_new_tokens": 32,
        "supervisor": {
            "path": str(supervisor),
            "sha256": preflight.file_digest(supervisor),
            "route": "sc-gpu",
            "first_failure_stop": True,
            "automatic_retries": 0,
        },
    }

    preflight._check_execution_contract({"execution_contract": contract})
    contract["geometry"]["gpu_blocks"] = 321
    try:
        preflight._check_execution_contract({"execution_contract": contract})
    except preflight.PreflightError as exc:
        assert "execution geometry" in str(exc)
    else:
        raise AssertionError("unfrozen geometry was accepted")


def test_validator_launch_and_shape_faults_continue_and_keep_raw_output(tmp_path):
    context = {
        "id": "c",
        "cwd": str(tmp_path),
        "interpreter": sys.executable,
        "pythonpath": ["/home/preflight-test"],
    }
    manifest = {
        "contexts": [None, context],
        "validators": [
            {
                "id": "fails",
                "context_id": "c",
                "argv": [
                    sys.executable,
                    "-c",
                    "print('raw-failure');raise SystemExit(7)",
                ],
            },
            {
                "id": "missing",
                "context_id": "c",
                "argv": [str(tmp_path / "absent-executable")],
            },
            None,
            {"id": "bad-context", "context_id": [], "argv": ["unused"]},
            {"id": "empty-argv", "context_id": "c", "argv": []},
            {
                "id": "later",
                "context_id": "c",
                "argv": [sys.executable, "-c", "print('later-pass')"],
            },
        ],
    }
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._check_validators(manifest)
    evidence = error.value.evidence
    assert evidence["counts"] == {
        "planned": 6,
        "pass": 1,
        "fail": 5,
        "blocked": 0,
        "not_run": 0,
    }
    assert evidence["results"][0]["stdout"] == "raw-failure\n"
    assert "FileNotFoundError" in evidence["results"][1]["error"]
    assert evidence["results"][-1]["stdout"] == "later-pass\n"


def test_timeout_retains_partial_output_and_runs_later_validator(tmp_path, monkeypatch):
    real_run = subprocess.run
    context = {
        "id": "c",
        "cwd": str(tmp_path),
        "interpreter": sys.executable,
        "pythonpath": ["/home/preflight-test"],
    }

    def bounded_run(argv, **kwargs):
        # Exercise a real timeout while keeping this regression inexpensive.
        kwargs["timeout"] = 0.15
        return real_run(argv, **kwargs)

    monkeypatch.setattr(preflight.subprocess, "run", bounded_run)
    command = "import time,sys;print('partial',flush=True);print('error-partial',file=sys.stderr,flush=True);time.sleep(5)"
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._check_validators(
            {
                "contexts": [context],
                "validators": [
                    {
                        "id": "timeout",
                        "context_id": "c",
                        "argv": [sys.executable, "-c", command],
                    },
                    {
                        "id": "later",
                        "context_id": "c",
                        "argv": [sys.executable, "-c", "print('ok')"],
                    },
                ],
            }
        )
    records = error.value.evidence["results"]
    assert records[0]["returncode"] is None
    assert records[0]["stdout"] == "partial\n"
    assert records[0]["stderr"] == "error-partial\n"
    assert records[1]["status"] == "pass"
    probe = preflight._run_probe(context, command)
    assert probe["stdout"] == "partial\n"
    assert probe["stderr"] == "error-partial\n"


def test_native_import_failure_preserves_all_module_results(tmp_path):
    context = {
        "id": "c",
        "cwd": str(tmp_path),
        "interpreter": sys.executable,
        "pythonpath": ["/home/preflight-test"],
        "native_modules": ["missing_native_one", "missing_native_two", "json"],
    }
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._check_native_imports(context)
    evidence = error.value.evidence
    assert set(evidence["origins"]["errors"]) == {
        "missing_native_one",
        "missing_native_two",
    }
    assert evidence["origins"]["modules"]["json"]
    assert evidence["probe"]["stdout"]


@pytest.mark.parametrize("stdout", ["", "not-json", "[]"])
def test_bad_probe_payload_keeps_raw_evidence(stdout):
    probe = {"returncode": 0, "stdout": stdout, "stderr": "raw-stderr"}
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._probe_payload(probe, "fixture")
    assert error.value.evidence["probe"] == probe


def test_malformed_contexts_do_not_escape_cli_aggregate(tmp_path):
    manifest = {
        "schema": preflight.SCHEMA,
        "contexts": [None, {"id": "duplicate"}, {"id": "duplicate"}, {"id": "later"}],
        "inputs": [],
        "validators": [],
    }
    manifest["manifest_digest"] = preflight.digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "report.json"
    assert preflight.main(["--manifest", str(path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    ids = [item["id"] for item in report["checks"]]
    assert len(ids) == len(set(ids)) == 23
    assert "later:identity" in report["failed_checks"]
    assert "later:native_imports" in report["blocked_checks"]
    assert report["counts"]["planned"] == sum(
        report["counts"][state] for state in ("pass", "fail", "blocked", "not_run")
    )
    assert ids[-1] == "execution_contract"


def test_failed_context_blocks_only_its_validator(tmp_path):
    context = {
        "id": "bad",
        "cwd": str(tmp_path),
        "pythonpath": ["/home/preflight-test"],
    }
    manifest = {
        "contexts": [context, dict(context, id="good")],
        "validators": [
            {"id": "blocked", "context_id": "bad", "argv": ["/must/not/run"]},
            {
                "id": "later",
                "context_id": "good",
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        ],
    }
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._check_validators(manifest, {"bad": "fail", "good": "pass"})
    assert [item["status"] for item in error.value.evidence["results"]] == [
        "blocked",
        "pass",
    ]


def test_manifest_integrity_stop_executes_no_checks(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("integrity-invalid manifest executed a check")

    monkeypatch.setattr(preflight, "_check_input_closure", forbidden)
    monkeypatch.setattr(preflight, "_check_context_identity", forbidden)
    manifest = {
        "schema": preflight.SCHEMA,
        "manifest_digest": "invalid",
        "contexts": [{"id": "c"}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "report.json"
    assert preflight.main(["--manifest", str(path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["failed_checks"] == ["manifest"]
    assert report["counts"] == {
        "planned": 1,
        "pass": 0,
        "fail": 1,
        "blocked": 0,
        "not_run": 0,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"modules": []},
        {"modules": None},
        {},
        {"modules": {"json": "origin"}, "errors": []},
    ],
)
def test_malformed_native_payload_keeps_probe(payload, monkeypatch):
    probe = {
        "returncode": 0,
        "stdout": json.dumps(payload),
        "stderr": "original-stderr",
    }
    monkeypatch.setattr(preflight, "_run_probe", lambda *args, **kwargs: probe)
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._check_native_imports({"native_modules": ["json"]})
    assert error.value.evidence == {"probe": probe, "origins": payload}


def _assert_exact_stream(record, stream, raw):
    saved = record[f"{stream}_raw"]
    assert base64.b64decode(saved["data"]) == raw
    assert saved["byte_count"] == len(raw)
    assert saved["sha256"] == hashlib.sha256(raw).hexdigest()


def test_binary_subprocess_bytes_and_known_exit_survive(tmp_path):
    context = {
        "id": "c",
        "cwd": str(tmp_path),
        "interpreter": sys.executable,
        "pythonpath": ["/home/preflight-test"],
    }
    command = "import os;os.write(1,bytes([255,0,65,13,10]));os.write(2,b'error-evidence\\r\\n')"
    probe = preflight._run_probe(context, command)
    assert probe["returncode"] == 0
    _assert_exact_stream(probe, "stdout", b"\xff\x00A\r\n")
    _assert_exact_stream(probe, "stderr", b"error-evidence\r\n")
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._probe_payload(probe, "binary")
    assert error.value.evidence["probe"] == probe
    with pytest.raises(preflight.CheckFailure) as error:
        preflight._check_validators(
            {
                "contexts": [context],
                "validators": [
                    {
                        "id": "binary",
                        "context_id": "c",
                        "argv": [sys.executable, "-c", command],
                    },
                    {
                        "id": "nonzero",
                        "context_id": "c",
                        "argv": [
                            sys.executable,
                            "-c",
                            command + ";raise SystemExit(7)",
                        ],
                    },
                    {
                        "id": "later",
                        "context_id": "c",
                        "argv": [sys.executable, "-c", "print('later-pass')"],
                    },
                ],
            }
        )
    records = error.value.evidence["results"]
    assert [row["returncode"] for row in records] == [0, 7, 0]
    assert [row["status"] for row in records] == ["fail", "fail", "pass"]
    for row in records[:2]:
        _assert_exact_stream(row, "stdout", b"\xff\x00A\r\n")
        _assert_exact_stream(row, "stderr", b"error-evidence\r\n")


def test_binary_timeout_retains_exact_partial_streams(tmp_path):
    context = {
        "id": "c",
        "cwd": str(tmp_path),
        "interpreter": sys.executable,
        "pythonpath": ["/home/preflight-test"],
    }
    probe = preflight._run_probe(
        context,
        "import os,time;os.write(1,bytes([255,0,13,10]));os.write(2,bytes([254,13,10]));time.sleep(5)",
        timeout=0.15,
    )
    assert probe["returncode"] is None
    assert "TimeoutExpired" in probe["error"]
    _assert_exact_stream(probe, "stdout", b"\xff\x00\r\n")
    _assert_exact_stream(probe, "stderr", b"\xfe\r\n")


def test_outer_aggregate_combines_malformed_probe_binary_failure_and_later_pass(
    tmp_path, monkeypatch
):
    context = {
        "id": "good",
        "cwd": str(tmp_path),
        "interpreter": sys.executable,
        "pythonpath": ["/home/preflight-test"],
        "native_modules": ["json"],
    }
    visited = []

    def identity(value):
        if value["id"] == "bad":
            raise preflight.PreflightError("bad identity")
        return {}

    def inventory(value):
        raise preflight.PreflightError("independent inventory failure")

    def later():
        visited.append("later-execution-contract")
        return {}

    monkeypatch.setattr(preflight, "_check_input_closure", lambda _: {})
    monkeypatch.setattr(preflight, "_check_context_identity", identity)
    monkeypatch.setattr(preflight, "_check_inventory", inventory)
    monkeypatch.setattr(
        preflight,
        "_run_probe",
        lambda *a, **k: {
            "returncode": 0,
            "stdout": '{"modules":null}',
            "stderr": "probe-raw",
        },
    )
    monkeypatch.setattr(preflight, "_check_native_constructor", lambda _: {})
    monkeypatch.setattr(preflight, "_check_execution_contract", lambda _: later())
    manifest = {
        "schema": preflight.SCHEMA,
        "contexts": [context, dict(context, id="bad")],
        "validators": [
            {
                "id": "binary",
                "context_id": "good",
                "argv": [
                    sys.executable,
                    "-c",
                    "import os;os.write(1,bytes([255]));os.write(2,b'raw-error')",
                ],
            },
            {"id": "blocked", "context_id": "bad", "argv": ["/never-executed"]},
            {
                "id": "later",
                "context_id": "good",
                "argv": [sys.executable, "-c", "print('later-pass')"],
            },
        ],
    }
    manifest["manifest_digest"] = preflight.digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "report.json"
    assert preflight.main(["--manifest", str(path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["counts"] == {
        "planned": 13,
        "pass": 5,
        "fail": 4,
        "blocked": 4,
        "not_run": 0,
    }
    assert visited == ["later-execution-contract"]
    checks = {check["id"]: check for check in report["checks"]}
    assert checks["good:native_imports"]["evidence"]["probe"]["stderr"] == "probe-raw"
    records = checks["validators"]["evidence"]["results"]
    assert [row["status"] for row in records] == ["fail", "blocked", "pass"]
    _assert_exact_stream(records[0], "stdout", b"\xff")
    _assert_exact_stream(records[0], "stderr", b"raw-error")
