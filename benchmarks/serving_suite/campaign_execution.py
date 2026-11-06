"""Bound one worker invocation and reap only descendants it created."""
from __future__ import annotations

import signal
import subprocess
import ctypes
import os
import sys
import time

import psutil


def run_worker(command, *, cwd, log, timeout):
    # Adopt workers that create their own session if their immediate parent dies.
    # This supervisor is serialized; preserve every child present before launch.
    if sys.platform == 'linux' and ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise OSError('Cannot enable campaign orphan recovery')
    supervisor = psutil.Process()
    preexisting = {p.pid for p in supervisor.children(recursive=True)}
    process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                               close_fds=True, start_new_session=True)
    owned = {}

    def collect():
        for child in supervisor.children(recursive=True):
            if child.pid not in preexisting and child.pid != process.pid:
                try:
                    owned.setdefault((child.pid, child.create_time()), child)
                except psutil.NoSuchProcess:
                    pass

    def alive():
        result = []
        for child in owned.values():
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    result.append(child)
            except psutil.NoSuchProcess:
                pass
        return result

    try:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            collect()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                process.wait(timeout=min(.5,remaining))
            except subprocess.TimeoutExpired:
                pass
        collect()
        returncode = process.returncode
        if returncode:
            raise RuntimeError(f'Worker exited {returncode}; see worker.log')
        if alive():
            raise RuntimeError('Worker exited with live owned descendants')
    except BaseException:
        # Capture birth identities before asking the worker to close OwnedServer.
        # The worker catches TERM and runs the normal server/tool cleanup path.
        collect()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        collect()
        for child in alive():
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(list(owned.values()), timeout=5)
        process.wait(timeout=5)
        if alive():
            raise RuntimeError('Owned campaign descendants survived cleanup')
        raise
    finally:
        for child in owned.values():
            try:
                os.waitpid(child.pid, os.WNOHANG)
            except ChildProcessError:
                pass
