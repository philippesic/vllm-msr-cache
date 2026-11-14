from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.config import check_environment
from tools.process import run_owned
from tools.runtime_verify import snapshot, write_record

UPSTREAM = "e9281bd939d4ade90d6aacf4b2319aa89ea46c39"
SOURCE_PATHS = (
    "vllm/v1/kv_offload/base.py",
    "vllm/v1/kv_offload/cpu/manager.py",
    "vllm/v1/kv_offload/cpu/policies/base.py",
    "vllm/v1/kv_offload/cpu/policies/arc.py",
    "vllm/v1/kv_offload/cpu/policies/lru.py",
    "vllm/v1/kv_offload/cpu/policies/factory.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/state.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/store_triggers/on_compute.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(runtime_python: Path, upstream: Path, out: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    out = out.resolve()
    if not out.is_relative_to(root / "artifacts"):
        raise ValueError("attempt must be under artifacts")
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    ).strip()
    if revision != UPSTREAM or dirty:
        raise ValueError("requires clean frozen upstream checkout")
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    identity = snapshot(root, source)
    environment = check_environment(source)
    environment["VLLM_PLUGINS"] = ""
    environment["VLLM_LOGGING_LEVEL"] = "ERROR"
    installed = Path(
        subprocess.check_output(
            [str(runtime_python.absolute()), "-c", "import vllm; print(vllm.__file__)"],
            env=environment,
            text=True,
        ).strip()
    ).parent.parent
    upstream_hashes = {path: _sha(upstream / path) for path in SOURCE_PATHS}
    installed_hashes = {path: _sha(installed / path) for path in SOURCE_PATHS}
    write_record(
        out / "admission.json",
        {
            "schema": "pr-feasibility-admission-v1",
            "upstream_revision": revision,
            "upstream_sha256": upstream_hashes,
            "installed_sha256": installed_hashes,
            "source_sha256": identity,
            "runtime_python": str(runtime_python.absolute()),
            "gpu_exposure_seconds": 0,
            "scope": "CPU policy-interface reproduction, not full runtime qualification",
        },
    )
    if installed_hashes != upstream_hashes:
        write_record(out / "terminal.json", {"error": "installed source mismatch"})
        raise ValueError("installed source does not match upstream")
    captured = source / "artifacts/observed.json"
    captured.parent.mkdir()
    command = [
        str(runtime_python.absolute()),
        "-m",
        "tools.benchmark",
        "pr-feasibility-native",
        "--out",
        str(captured),
    ]
    start = time.monotonic()
    with (out / "execution.log").open("x") as log:
        code = run_owned(command, environment, source, 120, log)
    if captured.exists():
        (out / "observed.json").write_bytes(captured.read_bytes())
    write_record(
        out / "terminal.json",
        {
            "schema": "pr-feasibility-terminal-v1",
            "exit_code": code,
            "wall_seconds": time.monotonic() - start,
            "execution_sha256": _sha(out / "execution.log"),
            "observed_sha256": _sha(out / "observed.json")
            if (out / "observed.json").exists()
            else None,
        },
    )
    return verify(out)


def verify(directory: Path) -> dict[str, Any]:
    admission = json.loads((directory / "admission.json").read_text())
    terminal = json.loads((directory / "terminal.json").read_text())
    if set(admission) != {
        "schema",
        "upstream_revision",
        "upstream_sha256",
        "installed_sha256",
        "source_sha256",
        "runtime_python",
        "gpu_exposure_seconds",
        "scope",
    } or set(terminal) != {
        "schema",
        "exit_code",
        "wall_seconds",
        "execution_sha256",
        "observed_sha256",
    }:
        raise ValueError("unknown receipt fields")
    if (
        admission.get("schema") != "pr-feasibility-admission-v1"
        or terminal.get("schema") != "pr-feasibility-terminal-v1"
        or admission["upstream_revision"] != UPSTREAM
        or set(admission["upstream_sha256"]) != set(SOURCE_PATHS)
        or admission["upstream_sha256"] != admission["installed_sha256"]
        or admission["gpu_exposure_seconds"] != 0
        or terminal["exit_code"] != 0
    ):
        raise ValueError("unqualified feasibility attempt")
    if not admission["source_sha256"]:
        raise ValueError("missing source identity")
    for path, expected in admission["source_sha256"].items():
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("unsafe source path")
        if _sha(directory / "source" / path) != expected:
            raise ValueError("source snapshot drift")
    if _sha(directory / "execution.log") != terminal["execution_sha256"]:
        raise ValueError("execution drift")
    if _sha(directory / "observed.json") != terminal["observed_sha256"]:
        raise ValueError("observation drift")
    observations = json.loads((directory / "observed.json").read_text())
    if set(observations) != {"arc", "lru"}:
        raise ValueError("missing policy observations")
    for observed in observations.values():
        if set(observed) != {
            "callbacks_at_return",
            "gpu_hit_finish_groups",
            "shared_gpu_hit_finish_groups",
            "scheduler_gpu_hit_chunks_skipped",
            "request_level_alternative_start",
            "ready_cpu_keys_present",
            "positions_available_privately",
            "late_store_readiness_callback",
            "observer_identity_evictions",
            "stock_identity_evictions",
            "callback_trace",
        }:
            raise ValueError("unknown observation fields")
        if (
            observed["scheduler_gpu_hit_chunks_skipped"] != 2
            or observed["request_level_alternative_start"] != 0
            or len(observed["observer_identity_evictions"]) != 5
            or observed["observer_identity_evictions"]
            != observed["stock_identity_evictions"]
            or not observed["callback_trace"]
            or observed["callbacks_at_return"] != []
            or observed["gpu_hit_finish_groups"] != []
            or observed["shared_gpu_hit_finish_groups"] != []
            or not observed["late_store_readiness_callback"]
            or len(observed["ready_cpu_keys_present"]) != 2
            or observed["positions_available_privately"] != 2
        ):
            raise ValueError("interface observation changed")
    return {
        "schema": "pr-feasibility-verification-v1",
        "upstream_revision": UPSTREAM,
        "scope": "CPU manager/policy callbacks only; no GPU or efficacy qualification",
        "observations": observations,
    }
