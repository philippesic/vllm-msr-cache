import subprocess
import sys
from pathlib import Path

from tools import native, repo_policy

ROOT = Path(__file__).resolve().parents[1]


def test_package_import_is_inert() -> None:
    code = (
        "import sys; import wait_hint_offload; import wait_hint_offload.contracts; "
        "assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0


def test_repository_policy_positive_control() -> None:
    assert repo_policy.check(ROOT) == []
    assert repo_policy.boundary_policy(ROOT) == []


def test_native_positive_control() -> None:
    assert native.check(ROOT, repo_policy.git_paths(ROOT)) == []


def test_native_block_names_are_not_reservations() -> None:
    from tools.repo_policy import LOCK_PATTERN

    for name in ("gpu_block_pool", "num_gpu_blocks_override", "vllm_block_size"):
        assert LOCK_PATTERN.search(name) is None
    assert LOCK_PATTERN.search("gpu_" + "reserved") is not None


def test_cpu_environment_hides_cuda() -> None:
    from pathlib import Path

    from tools.config import check_environment

    assert check_environment(Path.cwd())["CUDA_VISIBLE_DEVICES"] == ""
