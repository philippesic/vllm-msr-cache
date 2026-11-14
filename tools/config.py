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
    env["MYPY_CACHE_DIR"] = str(project / "artifacts/cache/proofs")
    return env


def attempts_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts/runs"


def output_overwrites_source(root: Path, output: Path) -> bool:
    destination = output.resolve()
    project = root.resolve()
    return destination.is_relative_to(project) and not destination.is_relative_to(
        project / "artifacts"
    )


def hardware_environment(root: Path, model: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["WAIT_HINT_TEST_MODEL"] = str(model.resolve())
    env["VLLM_PLUGINS"] = "wait_hint"
    env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
    env["VLLM_SERVER_DEV_MODE"] = "1"
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


def benchmark_environment(
    root: Path, model: Path, endpoint: bool = False, lmcache: bool = False
) -> dict[str, str]:
    env = hardware_environment(root, model)
    env["VLLM_PLUGINS"] = "wait_hint" if endpoint else ""
    if lmcache:
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
        env["LMCACHE_WSL_VMM"] = "1"
    return env


def pr_evidence_environment(root: Path, model: Path) -> dict[str, str]:
    env = hardware_environment(root, model)
    env["VLLM_PLUGINS"] = "pr_evidence"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cache = attempts_root().parent / "cache/pr-evidence"
    for variable, name in (
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        env[variable] = str(cache / name)
    return env
