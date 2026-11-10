from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import IO


def process_identity(pid: int) -> tuple[str, int, int, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return boot, int(fields[19]), int(fields[2]), int(fields[3])
    except (FileNotFoundError, ProcessLookupError):
        return None


def signal_owned(pid: int, expected: tuple[str, int, int, int], number: int) -> bool:
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        if process_identity(pid) != expected:
            raise RuntimeError("process identity changed; refusing control")
        signal.pidfd_send_signal(descriptor, number)
        return True
    finally:
        os.close(descriptor)


def stop_session(owner_pid: int, owner: tuple[str, int, int, int]) -> None:
    leader = process_identity(owner_pid)
    if leader is not None and leader != owner:
        raise RuntimeError("session owner identity changed")
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        identity = process_identity(pid)
        if identity is None:
            continue
        boot, born, group, session = identity
        if group == owner_pid and session == owner_pid:
            if boot != owner[0] or born < owner[1]:
                raise RuntimeError("foreign process in owned session")
            signal_owned(pid, identity, signal.SIGKILL)


def run_owned(
    command: list[str],
    environment: dict[str, str],
    cwd: Path,
    seconds: float,
    output: IO[str] | None = None,
) -> int:
    if seconds <= 0:
        raise ValueError("positive timeout required")
    child = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        start_new_session=True,
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    owner = process_identity(child.pid)
    if owner is None:
        return child.wait()
    try:
        try:
            return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            return 124
    finally:
        stop_session(child.pid, owner)
        child.wait()
