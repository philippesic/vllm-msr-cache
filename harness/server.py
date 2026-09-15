# SPDX-License-Identifier: Apache-2.0
"""Launch/wait-for-ready/teardown a real `vllm serve` subprocess for a
given policy config -- the piece every benchmark run needs, factored out
so run_latency_suite.py and run_accuracy_suite.py both use the exact same
launch/readiness/shutdown discipline (matching this project's own
real-server verification methodology throughout the issues log)."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from benchmarks.cell_lifecycle import process_birth_identity


def resolve_vllm_cli() -> str:
    """Resolve the vLLM CLI executable from the selected environment."""
    explicit = os.environ.get("VLLM_CLI")
    if explicit:
        return explicit
    sibling = os.path.join(os.path.dirname(sys.executable), "vllm")
    return (
        sibling if os.path.isfile(sibling) and os.access(sibling, os.X_OK) else "vllm"
    )


def resolve_vllm_command() -> list[str]:
    """Return the executable prefix used to start ``vllm serve``.

    A source-only vLLM checkout may be importable but not have its console
    script installed (for example, a Windows-hosted WSL experiment).  The
    module override keeps that setup reproducible without changing the
    normal installed-CLI path.
    """
    module = os.environ.get("VLLM_CLI_MODULE")
    if module:
        return [sys.executable, "-m", module]
    return [resolve_vllm_cli()]


class ServerHandle:
    def __init__(
        self,
        proc: subprocess.Popen,
        port: int,
        log_path: str,
        command: list[str] | None = None,
    ):
        self.proc = proc
        self.port = port
        self.log_path = log_path
        self.birth_id = process_birth_identity(proc.pid)
        self.command = list(command or [])
        if os.name == "nt":
            # Windows has no POSIX process groups.  The launcher uses
            # CREATE_NEW_PROCESS_GROUP below, but the group is managed by
            # taskkill rather than os.getpgid/killpg.
            self.pgid = None
        else:
            try:
                self.pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                self.pgid = None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def metrics_url(self) -> str:
        return f"{self.base_url()}/metrics"

    def is_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url()}/health", timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def shutdown(self, timeout_s: float = 20.0) -> None:
        if self.proc.poll() is not None:
            return
        if os.name == "nt":
            # terminate() is the least-forceful portable operation available
            # here.  If the server owns worker children, taskkill /T is used
            # only after the grace period so the benchmark's post-shutdown
            # child-process check remains meaningful.
            try:
                self.proc.terminate()
            except (ProcessLookupError, PermissionError):
                return
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    return
                time.sleep(0.5)
            if self.proc.poll() is None:
                try:
                    subprocess.run(
                        [
                            "taskkill",
                            "/PID",
                            str(self.proc.pid),
                            "/T",
                            "/F",
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    try:
                        self.proc.kill()
                    except (ProcessLookupError, PermissionError):
                        pass
            return
        if sys.platform == "linux":
            if (
                self.birth_id is None
                or process_birth_identity(self.proc.pid) != self.birth_id
            ):
                return
        try:
            if self.pgid is None or os.getpgid(self.proc.pid) != self.pgid:
                return
            if (
                sys.platform == "linux"
                and process_birth_identity(self.proc.pid) != self.birth_id
            ):
                return
            os.killpg(self.pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return
            time.sleep(0.5)
        try:
            if self.proc.poll() is not None:
                return
            if self.pgid is None or os.getpgid(self.proc.pid) != self.pgid:
                return
            if (
                sys.platform == "linux"
                and process_birth_identity(self.proc.pid) != self.birth_id
            ):
                return
            os.killpg(self.pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return


def launch_server(
    model: str,
    port: int,
    log_dir: str,
    *,
    gpu_memory_utilization: float = 0.5,
    max_model_len: int = 2048,
    num_gpu_blocks_override: int | None = None,
    kv_transfer_config: dict | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    ready_timeout_s: float = 180.0,
    log_label: str | None = None,
) -> ServerHandle:
    """Launch a real `vllm serve` subprocess and block until /health is 200
    (or raise TimeoutError). Caller must call .shutdown() when done -- no
    context-manager wrapper here so callers can decide whether a failed run
    should still tear the server down (yes, always) or leave it up for
    manual inspection (deliberately opt-in, not the default).

    `log_label` (e.g. a policy name) plus a wall-clock timestamp make the
    log filename unique per call -- a benchmark grid reuses the same port
    across many sequential launches, and a bare `server_{port}.log`
    opened in write mode silently overwrote every earlier run's log,
    losing the ability to debug any but the last one (found during real-
    server validation of this harness)."""
    os.makedirs(log_dir, exist_ok=True)
    label = f"{log_label}_" if log_label else ""
    log_path = os.path.join(
        log_dir, f"server_{label}{port}_{int(time.time() * 1000)}.log"
    )

    cmd = resolve_vllm_command() + [
        "serve",
        model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if num_gpu_blocks_override is not None:
        cmd += ["--num-gpu-blocks-override", str(num_gpu_blocks_override)]
    if kv_transfer_config is not None:
        cmd += ["--kv-transfer-config", json.dumps(kv_transfer_config)]
    if extra_args:
        cmd += extra_args

    # The descriptor must stay open for the lifetime of Popen; failure paths
    # below close it explicitly after the child exits.
    log_file = open(log_path, "w")  # noqa: SIM115
    popen_kwargs = {
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
    }
    if extra_env:
        popen_kwargs["env"] = {**os.environ, **extra_env}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        popen_kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(cmd, **popen_kwargs)
    handle = ServerHandle(proc, port, log_path, cmd)

    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            with open(log_path) as f:
                tail = f.read()[-4000:]
            raise RuntimeError(
                f"vllm serve exited early (code {proc.returncode}) before "
                f"becoming ready. Log tail:\n{tail}"
            )
        if handle.is_healthy():
            return handle
        time.sleep(2.0)

    handle.shutdown()
    log_file.close()
    with open(log_path) as f:
        tail = f.read()[-4000:]
    raise TimeoutError(
        f"vllm serve did not become healthy within {ready_timeout_s}s. "
        f"Log tail:\n{tail}"
    )
