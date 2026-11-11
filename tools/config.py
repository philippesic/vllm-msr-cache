from __future__ import annotations

import hashlib
import os
from pathlib import Path


def check_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def proof_environment(root: Path, project: Path) -> dict[str, str]:
    env = check_environment(root)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(project)])
    env["MYPY_CACHE_DIR"] = str(root.parent.parent / "mypy-cache")
    return env


def attempts_root() -> Path:
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(data) / "vllm-semantic-cache-ext-next/attempts"


def hardware_environment(root: Path, model: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["WAIT_HINT_TEST_MODEL"] = str(model.resolve())
    env["VLLM_PLUGINS"] = "wait_hint"
    env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
    return env


def hardware_model() -> Path:
    return Path(os.environ["WAIT_HINT_TEST_MODEL"])


def server_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "wait_hint"
    return env


def type_cache(root: Path, runtime_python: Path | None = None) -> Path:
    base = Path(os.environ.get("MYPY_CACHE_DIR", str(root / ".mypy_cache")))
    lane = "portable"
    if runtime_python is not None:
        identity = str(runtime_python.absolute())
        lane = "runtime-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    return base / lane
