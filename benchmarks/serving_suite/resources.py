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
    def __init__(self, pid, path, interval=0.25, metrics_url=None):
        self.pid = pid
        self.path = Path(path)
        self.interval = interval
        self.stop_event = threading.Event()
        self.rows = []
        self.error = None
        self.metrics_url = metrics_url

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
                        memory = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        row.update(host_available_bytes=memory.available,
                            host_swap_used_bytes=swap.used, host_swap_in_bytes=swap.sin,
                            host_swap_out_bytes=swap.sout,
                            host_cpu_percent=psutil.cpu_percent())
                        root = psutil.Process(self.pid)
                        server_pids = set()
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
                                server_pids.add(p.pid)
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
                            # Optional WSL/driver capabilities stay unavailable,
                            # rather than invalidating otherwise useful samples.
                            for name, read in (
                                ('gpu_temperature_c', lambda:nv.nvmlDeviceGetTemperature(gpu,nv.NVML_TEMPERATURE_GPU)),
                                ('gpu_sm_clock_mhz', lambda:nv.nvmlDeviceGetClockInfo(gpu,nv.NVML_CLOCK_SM)),
                                ('gpu_memory_clock_mhz', lambda:nv.nvmlDeviceGetClockInfo(gpu,nv.NVML_CLOCK_MEM)),
                                ('gpu_pcie_rx_kib_s', lambda:nv.nvmlDeviceGetPcieThroughput(gpu,nv.NVML_PCIE_UTIL_RX_BYTES)),
                                ('gpu_pcie_tx_kib_s', lambda:nv.nvmlDeviceGetPcieThroughput(gpu,nv.NVML_PCIE_UTIL_TX_BYTES))):
                                try:
                                    row[name] = read()
                                except Exception:
                                    row[name] = None
                        if self.metrics_url:
                            row.update(client_tool_rss_bytes=0, client_tool_cpu_s={})
                            parent = psutil.Process(os.getpid())
                            for p in [parent, *parent.children(recursive=True)]:
                                if p.pid in server_pids:
                                    continue
                                try:
                                    ticks = Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                                    t = p.cpu_times()
                                    row['client_tool_cpu_s'][f'{p.pid}:{ticks}'] = t.user+t.system
                                    row['client_tool_rss_bytes'] += p.memory_info().rss
                                except (psutil.Error, OSError):
                                    pass
                            from harness.metrics import parse_metrics, parse_gauge_metrics
                            began = time.monotonic_ns()
                            raw = get_text(self.metrics_url, timeout=2)
                            row.update(metrics=parse_metrics(raw), gauges=parse_gauge_metrics(raw),
                                metrics_scrape_s=(time.monotonic_ns()-began)/1e9)
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
        client_cpu = {}
        for r in self.rows:
            for k, v in r["process_cpu_s"].items():
                cpu.setdefault(k, []).append(v)
            for k,v in r.get('client_tool_cpu_s',{}).items():
                client_cpu.setdefault(k,[]).append(v)
        mem = [
            r["gpu_used_bytes"] for r in self.rows if r["gpu_used_bytes"] is not None
        ]
        return dict(
            samples=len(self.rows),
            gpu_peak_bytes=max(mem) if mem else None,
            host_rss_peak_bytes=max((r["rss_bytes"] for r in self.rows), default=None),
            host_available_min_bytes=min((r['host_available_bytes'] for r in self.rows if 'host_available_bytes' in r),default=None),
            host_swap_in_delta_bytes=(self.rows[-1].get('host_swap_in_bytes',0)-self.rows[0].get('host_swap_in_bytes',0)) if self.rows else None,
            host_swap_out_delta_bytes=(self.rows[-1].get('host_swap_out_bytes',0)-self.rows[0].get('host_swap_out_bytes',0)) if self.rows else None,
            gpu_temperature_peak_c=max((r['gpu_temperature_c'] for r in self.rows if r.get('gpu_temperature_c') is not None),default=None),
            gpu_sm_clock_range_mhz=[f(r['gpu_sm_clock_mhz'] for r in self.rows if r.get('gpu_sm_clock_mhz') is not None)
                for f in (min,max)] if any(r.get('gpu_sm_clock_mhz') is not None for r in self.rows) else None,
            process_cpu_elapsed_s=sum(max(v) - min(v) for v in cpu.values()),
            client_tool_cpu_elapsed_s=sum(max(v)-min(v) for v in client_cpu.values()) if self.metrics_url else None,
            client_tool_rss_peak_bytes=max((r.get('client_tool_rss_bytes',0) for r in self.rows),default=None) if self.metrics_url else None,
            sample_interval_s=self.interval,
            error=self.error
            or next((r["error"] for r in self.rows if r.get("error")), None)
            or ("sampler_thread_did_not_stop" if self.thread.is_alive() else None),
            scope="Load-window sampled whole-device GPU memory and summed server-tree RSS (shared pages may double count); CPU lifetime deltas exclude startup. Not allocator peaks.",
        )
