from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import vllm


def require_binding() -> None:
    if sys.flags.optimize or sys.version_info[:3] != (3, 12, 3):
        raise RuntimeError("pinned unoptimized CPython 3.12.3 required")
    if importlib.metadata.version("vllm") != "0.26.1rc1.dev75+gdc1be7903":
        raise RuntimeError("wrong vLLM runtime")
    package = Path(vllm.__file__).parent
    marker = package / ".wait-hint-overlay.json"
    if not marker.is_file():
        raise RuntimeError(
            "launch through tools.run_runtime; attested RPC overlay missing"
        )
    record = json.loads(marker.read_text())
    if record.get("schema") != "wait-hint-overlay-v1":
        raise RuntimeError("unknown runtime overlay version")
    for name, digest in record["files"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"runtime overlay drift: {name}")
