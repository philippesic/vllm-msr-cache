"""Opt-in host/CPU and sampled, deferred CUDA timing; no per-step fence.

Set SEMANTIC_PROFILE_DIR to enable. Totals are inclusive and may overlap:
they must never be added to infer end-to-end latency. Files are per process.
"""

import atexit
from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time


class ComponentProfiler:
    def __init__(self, directory=None, cuda_stride=16):
        self.directory = directory
        self.cuda_stride = max(1, cuda_stride)
        self.pid = os.getpid()
        self.metrics = {}
        self.counters = Counter()
        self.pending = []
        self.lock = threading.RLock()
        self.calls = Counter()
        self.started_ns = time.monotonic_ns()

    def count(self, name, amount=1):
        if self.directory:
            with self.lock:
                self.counters[name] += amount

    def observe(self, name, value_ns):
        value_ns = max(0, int(value_ns))
        with self.lock:
            slot = self.metrics.setdefault(name, {
                "count": 0, "total_ns": 0, "max_ns": 0, "log2_ns_histogram": {}})
            slot["count"] += 1
            slot["total_ns"] += value_ns
            slot["max_ns"] = max(slot["max_ns"], value_ns)
            bucket = str(value_ns.bit_length())
            slot["log2_ns_histogram"][bucket] = slot["log2_ns_histogram"].get(bucket, 0) + 1

    def scope(self, name, *, cuda=False):
        if not self.directory:
            return nullcontext()
        return self._scope(name, cuda=cuda)

    @contextmanager
    def _scope(self, name, *, cuda=False):
        self.calls[name] += 1
        events = None
        if cuda and (self.calls[name] - 1) % self.cuda_stride == 0:
            import torch
            if len(self.pending) < 256 and torch.cuda.is_initialized():
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
            else:
                self.count(name + ".cuda_samples_dropped")
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            yield
        finally:
            end_cpu, end_wall = time.thread_time_ns(), time.perf_counter_ns()
            if events:
                events[1].record()
                with self.lock:
                    self.pending.append((name, *events))
            self.observe(name + ".host_wall", end_wall - start)
            self.observe(name + ".thread_cpu", end_cpu - cpu)

    def drain(self):
        """Read only completed samples, retaining unfinished events."""
        with self.lock:
            waiting = []
            for name, start, end in self.pending:
                if end.query():
                    self.observe(name + ".cuda_stream", start.elapsed_time(end) * 1_000_000)
                else:
                    waiting.append((name, start, end))
            self.pending = waiting

    def flush(self):
        if not self.directory:
            return
        self.drain()
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        with self.lock:
            report = {
                "schema": "semantic-component-profile-v1", "pid": self.pid,
                "clock": "host_boot_monotonic_ns", "started_ns": self.started_ns,
                "written_ns": time.monotonic_ns(), "cuda_stride": self.cuda_stride,
                "pending_cuda_samples": len(self.pending),
                "interpretation": "Inclusive wall/CPU and sampled CUDA stream intervals overlap; not additive. Histograms give power-of-two duration bounds, not exact quantiles. No device synchronization added.",
                "metrics": self.metrics, "counters": dict(self.counters),
            }
            target = directory / f"components-{self.pid}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            temporary.replace(target)


PROFILE = ComponentProfiler(os.environ.get("SEMANTIC_PROFILE_DIR"),
                            int(os.environ.get("SEMANTIC_PROFILE_CUDA_STRIDE", "16")))
atexit.register(PROFILE.flush)


def profiled(name, *, cuda=False):
    def decorate(function):
        if not PROFILE.directory:
            return function
        @wraps(function)
        def wrapper(*args, **kwargs):
            with PROFILE.scope(name, cuda=cuda):
                return function(*args, **kwargs)
        return wrapper
    return decorate
