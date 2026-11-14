from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any

from tools.benchmarking.feasibility import SOURCE_PATHS, UPSTREAM
from tools.benchmarking.pr_workload import ARMS
from tools.benchmarking.probe import digest_file, request
from tools.benchmarking.resources import sample_process_tree
from tools.config import pr_evidence_environment
from tools.process import process_identity, signal_owned, stop_session
from tools.runtime_verify import snapshot, write_record

POLICIES = {
    "A1": "lru",
    "A2": "arc",
    "A3": "AdviceOffARC",
    "A4t": "RequestTTLARC",
    "A3p": "AdviceOffARC",
    "A4p": "PostResponseARC",
    "A5": "RequestTTLLRU",
}
MODEL_REVISIONS = {
    "D8": "4da05a8edb55c6046cce958586c33b61da07bb79",
    "HM": "d3e78b50c727b1f34802f440ca84b25781625352",
    "HS": "7c781be42ce8d29d0975b857f6d0c0b2558d9b63",
}
CAPS = {"admission": 3600.0, "screen": 28800.0}


def screen_server_command(
    python: Path,
    model: Path,
    label: str,
    arm: str,
    port: int,
    pool_gib: int,
    events: Path,
    byte_probe: bool,
) -> list[str]:
    if label not in MODEL_REVISIONS or arm not in ARMS or pool_gib not in {1, 2, 4}:
        raise ValueError("job outside frozen screen models, arms or pools")
    extra = {
        "spec_name": "ObservedCPUOffloadingSpec",
        "spec_module_path": "wait_hint_offload.vllm_adapter.evidence_manager",
        "cpu_bytes_to_use": pool_gib * 1024**3,
        "blocks_per_chunk": 1,
        "eviction_policy": POLICIES[arm],
        "evidence_directory": str(events),
        "evidence_byte_probe": byte_probe,
    }
    if arm not in {"A1", "A2"}:
        extra["cache_policy_module_path"] = "wait_hint_offload.vllm_adapter.policy"
    return [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        "pr-evidence",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        "local-benchmark",
        "--dtype",
        "float16",
        "--kv-cache-dtype",
        "auto",
        "--mamba-cache-dtype",
        "float16",
        "--mamba-ssm-cache-dtype",
        "float16",
        "--enforce-eager",
        "--attention-backend",
        "TRITON_ATTN",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "6144",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "512" if label == "HS" else "2048",
        "--enable-prefix-caching",
        "--prefix-cache-retention-interval",
        "0",
        "--language-model-only",
        "--enable-per-request-metrics",
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector": "EvidenceConnector",
                "kv_connector_module_path": "wait_hint_offload.vllm_adapter.evidence_connector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": extra,
            },
            sort_keys=True,
        ),
    ]


def gpu_status() -> dict[str, Any]:
    captures = {}
    for name, query in (
        (
            "devices",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
        ),
        ("processes", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"),
    ):
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        captures[name] = result.stdout
    rows = [line for line in captures["devices"].splitlines() if line.strip()]
    if len(rows) != 1 or "RTX 2080 Ti" not in rows[0]:
        raise ValueError("authorized hardware is the single local RTX 2080 Ti")
    return {"monotonic_ns": time.monotonic_ns(), **captures}


def spent_seconds(campaign: Path) -> dict[str, float]:
    spent = dict.fromkeys(CAPS, 0.0)
    for path in sorted(campaign.glob("jobs/*/admission.json")):
        admission = json.loads(path.read_text())
        terminal = path.parent / "terminal.json"
        if not terminal.is_file():
            raise ValueError(f"unfinished attempt needs reconciliation: {path.parent}")
        record = json.loads(terminal.read_text())
        seconds = record["gpu_charge_seconds"]
        if (
            type(seconds) not in (int, float)
            or not 0 <= seconds <= admission["maximum_seconds"] + 60
        ):
            raise ValueError("invalid GPU budget receipt")
        spent[admission["phase"]] += seconds
    return spent


def runtime_identity(root: Path, python: Path, arm: str) -> dict[str, Any]:
    info = subprocess.run(
        [
            str(python),
            "-c",
            "import sysconfig,importlib.metadata,json; print(json.dumps({'site':sysconfig.get_path('purelib'),'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))}))",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    identity: dict[str, Any] = json.loads(info.stdout)
    if identity["packages"]["vllm"] != "0.34.0rc1.dev465+ge9281bd93":
        raise ValueError("runtime version differs from frozen main")
    expected_packages = {
        re.sub(r"[-_.]+", "-", name).lower(): version
        for name, version in (
            line.split("==", 1)
            for line in (root / "runtime/pr-evidence-requirements.txt")
            .read_text()
            .splitlines()
            if line.strip()
        )
    }
    expected_packages["wait-hint-kv"] = "0.0.0"
    actual_packages = {
        re.sub(r"[-_.]+", "-", name).lower(): version
        for name, version in identity["packages"].items()
    }
    if actual_packages != expected_packages:
        raise ValueError("runtime package set differs from CPU freeze")
    site = Path(identity["site"])
    upstream = root / "artifacts/checkouts/vllm-pr-evidence"
    if (
        subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
        != UPSTREAM
    ):
        raise ValueError("upstream checkout identity drift")
    paths = {path: digest_file(upstream / path) for path in SOURCE_PATHS}
    patches = json.loads(
        (root / "benchmarks/runtime-patches/manifest.json").read_text()
    )
    for patch in patches:
        if patch["name"] == "gemma-turing-main":
            paths[patch["target"]] = patch["patched_sha256"]
        elif patch["name"] == "pr-post-response":
            paths[patch["target"]] = (
                patch["patched_sha256"]
                if arm in {"A3p", "A4p"}
                else patch["original_sha256"]
            )
    installed = {path: digest_file(site / path) for path in paths}
    if paths != installed:
        raise ValueError("native sources or matched runtime patches differ")
    return identity | {"source_sha256": installed}


def _stop(
    child: subprocess.Popen[str], identity: tuple[str, int, int, int] | None
) -> None:
    if identity is not None:
        if child.poll() is None:
            signal_owned(child.pid, identity, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        stop_session(child.pid, identity)
    child.wait()


def verify_gpu_job(directory: Path) -> dict[str, Any]:
    admission = json.loads((directory / "admission.json").read_text())
    terminal = json.loads((directory / "terminal.json").read_text())
    if (
        admission.get("schema") != "pr-gpu-job-v1"
        or terminal.get("schema") != "pr-gpu-job-terminal-v1"
    ):
        raise ValueError("unknown GPU receipt schema")
    for relative, digest in admission["source_sha256"].items():
        if digest_file(directory / "source" / relative) != digest:
            raise ValueError("GPU job source drift")
    for relative, digest in admission["model_sha256"].items():
        if digest_file(Path(admission["model_path"]) / relative) != digest:
            raise ValueError("GPU model drift")
    for relative, digest in terminal["artifacts_sha256"].items():
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or digest_file(directory / relative) != digest
        ):
            raise ValueError("GPU job artifact drift")
    if (
        terminal["gpu_charge_seconds"]
        != (terminal["ended_ns"] - admission["started_ns"]) / 1e9
        or not 0 <= terminal["gpu_charge_seconds"] <= admission["maximum_seconds"] + 60
        or json.loads((directory / "idle.json").read_text())["processes"].strip()
    ):
        raise ValueError("GPU time or idle evidence differs")
    return {
        "schema": "pr-gpu-job-verification-v1",
        "receipt_valid": True,
        "terminal": terminal,
        "scope": "immutable execution receipt; model correctness and cell validity are separate verdicts",
    }


def _interrupted(number: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"termination signal {number}")


def run_gpu_job(
    campaign: Path,
    name: str,
    python: Path,
    model: Path,
    label: str,
    arm: str,
    phase: str,
    pool_gib: int,
    maximum_seconds: int,
    schedule: Path | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    campaign = campaign.resolve()
    python, model = python.absolute(), model.absolute()
    if (
        campaign != root / "artifacts/runs/pr-evidence-2080ti"
        or not name
        or Path(name).name != name
        or name in {".", ".."}
        or phase not in CAPS
        or not 30 <= maximum_seconds <= 1800
        or label not in MODEL_REVISIONS
        or model.name != MODEL_REVISIONS[label]
        or (phase == "screen") != (schedule is not None)
    ):
        raise ValueError("job arguments outside authorized protocol")
    freeze = root / "benchmarks/pr-evidence/stage1-freeze.json"
    if not freeze.is_file() or json.loads(freeze.read_text())["stage1_exit"] != "go":
        raise ValueError("committed Stage 1 exit required before GPU")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise ValueError("GPU job requires a clean committed source")
    spent = spent_seconds(campaign)
    if spent[phase] + maximum_seconds + 60 > CAPS[phase]:
        raise ValueError("complete job plus cleanup does not fit remaining GPU budget")
    runtime = runtime_identity(root, python, arm)
    model_hashes = {
        str(path.relative_to(model)): digest_file(path)
        for path in sorted(model.rglob("*"))
        if path.is_file()
    }
    output = campaign / "jobs" / name
    output.mkdir(parents=True, exist_ok=False)
    identities = snapshot(root, output / "source")
    if schedule is not None:
        (output / "schedule.json").write_bytes(schedule.read_bytes())
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = screen_server_command(
        python,
        model,
        label,
        arm,
        port,
        pool_gib,
        output / "events",
        phase == "admission",
    )
    idle = gpu_status()
    write_record(output / "idle.json", idle)
    if idle["processes"].strip():
        write_record(
            output / "deferred.json",
            {"reason": "GPU busy; no process started", "gpu_charge_seconds": 0},
        )
        raise ValueError("GPU busy; wait before creating a new attempt")
    started = time.monotonic_ns()
    admission = {
        "schema": "pr-gpu-job-v1",
        "phase": phase,
        "model": label,
        "arm": arm,
        "pool_gib": pool_gib,
        "maximum_seconds": maximum_seconds,
        "started_ns": started,
        "source_sha256": identities,
        "runtime": runtime,
        "model_path": str(model),
        "model_sha256": model_hashes,
        "command": command,
        "stage1_freeze_sha256": digest_file(freeze),
        "prior_spent_seconds": spent,
        "schedule_sha256": digest_file(output / "schedule.json") if schedule else None,
    }
    write_record(output / "admission.json", admission)
    environment = pr_evidence_environment(output / "source", model)
    owned: list[tuple[subprocess.Popen[str], tuple[str, int, int, int] | None]] = []
    client: subprocess.Popen[str] | None = None
    status, detail = "failed", "server did not become ready"
    last_gpu = 0
    first_sample: dict[str, Any] | None = None
    previous_term = signal.signal(signal.SIGTERM, _interrupted)
    try:
        with (
            (output / "server.log").open("x") as server_log,
            (output / "client.log").open("x") as client_log,
            (output / "resources.jsonl").open("x", buffering=1) as resources,
        ):
            server = subprocess.Popen(
                command,
                cwd=output / "source",
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            owner = process_identity(server.pid)
            owned.append((server, owner))
            write_record(
                output / "server-owner.json", {"pid": server.pid, "identity": owner}
            )
            while (time.monotonic_ns() - started) / 1e9 < maximum_seconds:
                sample = sample_process_tree(os.getpid())
                if first_sample is None:
                    first_sample = sample
                if sample["monotonic_ns"] - last_gpu >= 5_000_000_000:
                    sample["gpu"] = gpu_status()
                    last_gpu = sample["monotonic_ns"]
                resources.write(json.dumps(sample, separators=(",", ":")) + "\n")
                if (
                    sample["available_bytes"] < 2 * 1024**3
                    or sample["swap_bytes"] > first_sample["swap_bytes"]
                    or sample["host_swap_out_pages"]
                    > first_sample["host_swap_out_pages"]
                ):
                    status, detail = (
                        "resource_exhaustion",
                        "available RAM below 2 GiB or additional swap",
                    )
                    break
                if server.poll() is not None:
                    detail = f"server exited {server.returncode}"
                    break
                if client is None:
                    try:
                        request(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                    except (urllib.error.URLError, TimeoutError, ConnectionError):
                        time.sleep(0.5)
                        continue
                    operation = (
                        "pr-admission-native"
                        if phase == "admission"
                        else "pr-client-native"
                    )
                    client_command = [
                        str(python),
                        "-m",
                        "tools.benchmark",
                        operation,
                        "--url",
                        f"http://127.0.0.1:{port}",
                        "--arm",
                        arm,
                        "--out",
                        str(output / "client"),
                    ]
                    if schedule:
                        client_command += ["--schedule", str(output / "schedule.json")]
                    client = subprocess.Popen(
                        client_command,
                        cwd=output / "source",
                        env=environment,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        text=True,
                    )
                    client_owner = process_identity(client.pid)
                    owned.append((client, client_owner))
                    write_record(
                        output / "client-owner.json",
                        {
                            "pid": client.pid,
                            "identity": client_owner,
                            "command": client_command,
                        },
                    )
                elif client.poll() is not None:
                    status = (
                        "client_complete" if client.returncode == 0 else "client_failed"
                    )
                    detail = f"client exited {client.returncode}"
                    break
                time.sleep(0.5)
            else:
                status, detail = "timeout", "bounded job deadline reached"
    except (Exception, KeyboardInterrupt) as error:
        status, detail = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            f"{type(error).__name__}: {error}",
        )
    finally:
        for child, owner in reversed(owned):
            _stop(child, owner)
        signal.signal(signal.SIGTERM, previous_term)
        ended = time.monotonic_ns()
        terminal = {
            "schema": "pr-gpu-job-terminal-v1",
            "status": status,
            "detail": detail,
            "ended_ns": ended,
            "gpu_charge_seconds": (ended - started) / 1e9,
            "artifacts_sha256": {
                str(path.relative_to(output)): digest_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and not path.is_relative_to(output / "source")
            },
        }
        write_record(output / "terminal.json", terminal)
    return terminal
