from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

from tools.benchmarking.pr_jobs import (
    _stop,
    client_command,
    model_digest,
    screen_server_command,
    spent_seconds,
)
from tools.process import process_identity


def test_screen_commands_keep_platform_and_route_controls_matched(
    tmp_path: Path,
) -> None:
    for model in ("D8", "HM", "HS"):
        commands = {}
        for arm in ("A1", "A2", "A3", "A4t", "A3p", "A4p", "A5"):
            command = screen_server_command(
                Path("python"),
                tmp_path,
                model,
                arm,
                8000,
                2,
                tmp_path / "events",
                False,
            )
            assert "--num-gpu-blocks-override" not in command
            assert command[command.index("--gpu-memory-utilization") + 1] == "0.85"
            assert command[command.index("--max-num-seqs") + 1] == "16"
            assert command[command.index("--max-num-batched-tokens") + 1] == (
                "512" if model == "HS" else "2048"
            )
            config = json.loads(command[-1])["kv_connector_extra_config"]
            assert config["cpu_bytes_to_use"] == 2 * 1024**3
            assert config["evidence_byte_probe"] is False
            commands[arm] = config
        assert commands["A3"] == commands["A3p"]
        assert {
            key: value
            for key, value in commands["A4t"].items()
            if key != "eviction_policy"
        } == {
            key: value
            for key, value in commands["A4p"].items()
            if key != "eviction_policy"
        }


def test_gpu_budget_retains_failed_attempt_and_blocks_unfinished_job(
    tmp_path: Path,
) -> None:
    job = tmp_path / "jobs" / "failed"
    job.mkdir(parents=True)
    (job / "admission.json").write_text(
        json.dumps({"phase": "admission", "maximum_seconds": 120})
    )
    with pytest.raises(ValueError, match="unfinished"):
        spent_seconds(tmp_path)
    (job / "terminal.json").write_text(
        json.dumps({"gpu_charge_seconds": 71, "status": "failed"})
    )
    assert spent_seconds(tmp_path) == {"admission": 71, "screen": 0}
    (job / "terminal.json").write_text(json.dumps({"gpu_charge_seconds": -1}))
    with pytest.raises(ValueError, match="budget"):
        spent_seconds(tmp_path)


def test_interruption_cleans_owned_server_and_descendant() -> None:
    command = [
        sys.executable,
        "-c",
        "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(child.pid,flush=True); time.sleep(60)",
    ]
    server = subprocess.Popen(
        command, start_new_session=True, text=True, stdout=subprocess.PIPE
    )
    owner = process_identity(server.pid)
    descriptor = None
    try:
        assert server.stdout is not None
        assert select.select([server.stdout], [], [], 5)[0]
        descendant = int(server.stdout.readline())
        descriptor = os.pidfd_open(descendant)
        assert process_identity(descendant) is not None
    finally:
        _stop(server, owner)
        if server.stdout is not None:
            server.stdout.close()
    assert server.poll() is not None
    assert descriptor is not None
    try:
        assert select.select([descriptor], [], [], 5)[0]
    finally:
        os.close(descriptor)


def test_model_checksum_cache_release_preserves_exact_bytes(tmp_path: Path) -> None:
    import hashlib

    model = tmp_path / "fixture.safetensors"
    data = bytes(range(256)) * 1024
    model.write_bytes(data)
    assert model_digest(model) == hashlib.sha256(data).hexdigest()
    assert model.read_bytes() == data


def test_captured_client_command_passes_real_output_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.benchmark
    import tools.benchmarking.pr_admission

    captured = tmp_path / "source/artifacts/client"
    captured.parent.mkdir(parents=True)

    async def admission(url: str, arm: str, directory: Path) -> dict[str, bool]:
        assert url == "http://127.0.0.1:8000" and arm == "A2"
        directory.mkdir()
        (directory / "partial.json").write_text('{"retained":true}')
        return {"fixture": True}

    monkeypatch.setattr(
        tools.benchmarking.pr_admission, "admission_requests", admission
    )
    monkeypatch.setattr(
        tools.benchmark, "__file__", str(tmp_path / "source/tools/benchmark.py")
    )
    command = client_command(Path(sys.executable), tmp_path, "A2", 8000, None)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", *command[3:]])
    assert tools.benchmark.main() == 0
    captured.rename(tmp_path / "client")
    assert json.loads((tmp_path / "client/partial.json").read_text()) == {
        "retained": True
    }
