from __future__ import annotations

import importlib.metadata
import sys

from wait_hint_offload.runtime_pin import PYTHON_VERSION, VLLM_VERSION


def require_runtime() -> None:
    if sys.flags.optimize:
        raise RuntimeError(
            "optimized Python is unsupported: pinned vLLM transfer safety relies on assertions"
        )
    if sys.version_info[:3] != PYTHON_VERSION:
        raise RuntimeError(
            f"CPython {'.'.join(map(str, PYTHON_VERSION))} required, found {sys.version.split()[0]}"
        )
    installed = importlib.metadata.version("vllm")
    if installed != VLLM_VERSION:
        raise RuntimeError(
            f"vLLM {VLLM_VERSION} required, found {installed}; install runtime/requirements.txt"
        )
