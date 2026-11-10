from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--runtime-capture", required=True)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    files: dict[str, str] = {}
    missing = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if path.as_posix() in files:
            continue
        if path.is_file():
            files[str(path)] = hashlib.file_digest(
                path.open("rb"), "sha256"
            ).hexdigest()
        else:
            missing.append(str(path))
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    modules = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(("wait_hint_offload", "vllm")):
            continue
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_file():
            modules[name] = {
                "path": str(filename),
                "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            }
    record = {
        "loaded_project_and_runtime_modules": modules,
        "schema": "runtime-observation-v1",
        "exit_code": int(exitstatus),
        "collected": session.testscollected,
        "failed": session.testsfailed,
        "python": sys.executable,
        "python_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "pid": os.getpid(),
        "birth_ticks": Path("/proc/self/stat")
        .read_text()
        .rsplit(")", 1)[1]
        .split()[19],
        "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cuda_initialized": bool(cuda and cuda.is_initialized()),
        "distributions": {
            str(d.metadata["Name"]): d.version
            for d in importlib.metadata.distributions()
        },
        "mapped_files_sha256": files,
        "unavailable_maps": missing,
    }
    with Path(session.config.getoption("runtime_capture")).open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
