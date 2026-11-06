"""Bound one worker invocation and reap only descendants it created."""
from __future__ import annotations

import signal
import subprocess

import psutil


def run_worker(command, *, cwd, log, timeout):
    process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                               close_fds=True, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        if returncode:
            raise RuntimeError(f'Worker exited {returncode}; see worker.log')
    except BaseException:
        # Capture birth identities before asking the worker to close OwnedServer.
        # The worker catches TERM and runs the normal server/tool cleanup path.
        parent = psutil.Process(process.pid) if process.poll() is None else None
        descendants = parent.children(recursive=True) if parent else []
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        for child in descendants:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(descendants, timeout=5)
        process.wait(timeout=5)
        raise
