"""Owned server lifecycle and external telemetry; no engine monkey patches."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

from harness.gpu_lock import GPULock as GPULock


def available_gpu():
    import pynvml as nv

    nv.nvmlInit()
    try:
        h = nv.nvmlDeviceGetHandleByIndex(0)
        processes = nv.nvmlDeviceGetComputeRunningProcesses(h)
        if processes:
            raise RuntimeError(
                "GPU has existing compute processes; preserve their owner and retry after release"
            )
        return dict(
            name=nv.nvmlDeviceGetName(h),
            uuid=nv.nvmlDeviceGetUUID(h),
            total_bytes=nv.nvmlDeviceGetMemoryInfo(h).total,
            driver=nv.nvmlSystemGetDriverVersion(),
        )
    finally:
        nv.nvmlShutdown()


def get_text(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OwnedServer:
    def __init__(self, command, directory, env, port):
        self.command = command
        self.directory = Path(directory)
        self.env = env
        self.port = port
        self.proc = None
        self.log = None
        # Adopt orphaned grandchildren, including workers that create a new session.
        # This suite is serialized and owns only children created after this point.
        import ctypes

        if (
            sys.platform == "linux"
            and ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0
        ):
            raise OSError("Cannot enable owned-child recovery")
        self.preexisting_children = {
            p.pid for p in psutil.Process().children(recursive=True)
        }
        self.cleanup = {"verified": False}

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout):
        # Refuse an existing listener before starting a process; never adopt it.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.port))
        self.log = (self.directory / "server.log").open("wb")
        try:
            self.proc = subprocess.Popen(
                self.command,
                env=self.env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.log.close()
            raise
        self.pgid = self.proc.pid
        start = time.monotonic()
        try:
            while time.monotonic() - start < timeout:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {self.proc.returncode}; see server.log"
                    )
                try:
                    get_text(self.url + "/health", timeout=min(1, timeout))
                    return time.monotonic() - start
                except (OSError, ValueError):
                    time.sleep(0.1)
            raise TimeoutError("Server readiness timed out; see server.log")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.proc is None:
            self.cleanup = {"verified": True, "spawned": False}
            return self.cleanup
        # The new session belongs to this launch. Kill its descendants even if
        # its leader exits early. Track birth times before signalling any stragglers.
        owned = []
        for p in psutil.process_iter(["pid", "status"]):
            try:
                if os.getpgid(p.pid) == self.pgid:
                    owned.append(p)
            except (ProcessLookupError, PermissionError):
                pass
        seen = {p.pid for p in owned}
        for p in psutil.Process().children(recursive=True):
            if p.pid not in self.preexisting_children and p.pid not in seen:
                owned.append(p)
        for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 3)):
            alive = [
                p
                for p in owned
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            ]
            if not alive:
                break
            for p in alive:
                try:
                    p.send_signal(sig)
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=wait)
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        survivors = []
        for p in owned:
            try:
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    survivors.append(p.pid)
            except psutil.NoSuchProcess:
                pass
        if self.log:
            self.log.close()
        self.cleanup = {
            "verified": not survivors and self.proc.poll() is not None,
            "survivors": survivors,
            "exit_code": self.proc.returncode,
        }
        return self.cleanup


class Sampler:
    def __init__(self, pid, path, interval=0.25):
        self.pid = pid
        self.path = Path(path)
        self.interval = interval
        self.stop_event = threading.Event()
        self.rows = []
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        nv = None
        try:
            import pynvml as nv

            nv.nvmlInit()
            gpu = nv.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            self.error = repr(e)
            nv = None
        try:
            with self.path.open("w") as output:
                while not self.stop_event.is_set():
                    row = {
                        "monotonic_s": time.monotonic(),
                        "rss_bytes": 0,
                        "process_cpu_s": {},
                        "gpu_used_bytes": None,
                    }
                    try:
                        root = psutil.Process(self.pid)
                        for p in [root, *root.children(recursive=True)]:
                            try:
                                # Stable Linux start ticks avoid WSL boot-time estimates.
                                ticks = (
                                    Path(f"/proc/{p.pid}/stat")
                                    .read_text()
                                    .rsplit(")", 1)[1]
                                    .split()[19]
                                )
                                t = p.cpu_times()
                                row["process_cpu_s"][f"{p.pid}:{ticks}"] = (
                                    t.user + t.system
                                )
                                row["rss_bytes"] += p.memory_info().rss
                            except (psutil.Error, OSError):
                                pass
                        if nv:
                            row.update(
                                gpu_used_bytes=nv.nvmlDeviceGetMemoryInfo(gpu).used,
                                gpu_utilization_pct=nv.nvmlDeviceGetUtilizationRates(
                                    gpu
                                ).gpu,
                                gpu_power_w=nv.nvmlDeviceGetPowerUsage(gpu) / 1000,
                            )
                    except Exception as e:
                        row["error"] = repr(e)
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    self.rows.append(row)
                    self.stop_event.wait(self.interval)
        except Exception as e:
            self.error = repr(e)
        finally:
            if nv:
                nv.nvmlShutdown()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        cpu = {}
        for r in self.rows:
            for k, v in r["process_cpu_s"].items():
                cpu.setdefault(k, []).append(v)
        mem = [
            r["gpu_used_bytes"] for r in self.rows if r["gpu_used_bytes"] is not None
        ]
        return dict(
            samples=len(self.rows),
            gpu_peak_bytes=max(mem) if mem else None,
            host_rss_peak_bytes=max((r["rss_bytes"] for r in self.rows), default=None),
            process_cpu_elapsed_s=sum(max(v) - min(v) for v in cpu.values()),
            sample_interval_s=self.interval,
            error=self.error
            or next((r["error"] for r in self.rows if r.get("error")), None)
            or ("sampler_thread_did_not_stop" if self.thread.is_alive() else None),
            scope="Load-window sampled whole-device GPU memory and summed server-tree RSS (shared pages may double count); CPU lifetime deltas exclude startup. Not allocator peaks.",
        )
