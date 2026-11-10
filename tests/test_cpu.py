import subprocess
import sys
from pathlib import Path

from tools import native, repo_policy

ROOT = Path(__file__).resolve().parents[1]


def test_package_import_is_inert() -> None:
    code = (
        "import sys; import semantic_offload; import semantic_offload.contracts; "
        "assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0


def test_repository_policy_positive_control() -> None:
    assert repo_policy.check(ROOT) == []
    assert repo_policy.boundary_policy(ROOT) == []


def test_native_positive_control() -> None:
    assert native.check(ROOT, repo_policy.git_paths(ROOT)) == []
