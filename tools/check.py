from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import native, repo_policy  # noqa: I001


def run(command: list[str], root: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        return subprocess.run(command, cwd=root, env=env, check=False).returncode
    except FileNotFoundError:
        print(f"tool.missing: {command[0]}; run uv sync --locked", flush=True)
        return 1


def component(name: str, root: Path) -> int:
    if name in {"policy", "boundaries", "native"}:
        if name == "policy":
            errors = repo_policy.check(root)
        elif name == "boundaries":
            errors = repo_policy.boundary_policy(root)
        else:
            errors = native.check(root, repo_policy.git_paths(root))
        for error in errors:
            print(error)
        if not errors:
            print(
                f"{name}: OK"
                if name != "native"
                else "skipped: no native sources yet (sm_75 and sm_120; clang-format, clang-tidy, CMake)"
            )
        return int(bool(errors))
    if name == "format":
        return run(["ruff", "format", "--check", "."], root)
    if name == "lint":
        return run(["ruff", "check", "."], root)
    if name == "types":
        return run(["mypy", "--strict"], root)
    if name == "imports":
        return run(["lint-imports", "--config", "pyproject.toml", "--no-cache"], root)
    raise ValueError(f"Unknown component: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic CPU/tooling checks")
    parser.add_argument(
        "--component",
        choices=[
            "format",
            "lint",
            "types",
            "imports",
            "boundaries",
            "policy",
            "native",
        ],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--self-test", action="store_true", help="only deliberate-violation proofs"
    )
    args = parser.parse_args()
    if args.component:
        return component(args.component, args.root.resolve())
    if args.root.resolve() != ROOT:
        parser.error("--root is only for isolated component specimens")
    started = time.monotonic()
    failed = 0
    skipped = 0
    names = (
        []
        if args.self_test
        else ["format", "lint", "types", "imports", "boundaries", "policy", "native"]
    )
    for name in names:
        before = time.monotonic()
        print(f"\n=== {name} ===", flush=True)
        code = component(name, ROOT)
        failed += bool(code)
        is_skipped = name == "native" and code == 0
        skipped += is_skipped
        status = "FAIL" if code else ("SKIPPED" if is_skipped else "OK")
        print(
            f"{name}: {status} ({time.monotonic() - before:.2f}s)",
            flush=True,
        )
    print("\n=== deliberate-violation proofs / CPU pytest ===", flush=True)
    manifest = "tools/self-tests.json" if args.self_test else "tools/cpu-tests.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-s",
        "-p",
        "tools.pytest_gate",
        "--gate-manifest",
        manifest,
    ]
    if args.self_test:
        command.append("tests/test_check_cases.py")
    code = run(command, ROOT)
    failed += bool(code)
    groups = len(names) + 1
    print(
        f"\nRESULT: {groups - failed - skipped} passed, {skipped} skipped, {failed} failed ({time.monotonic() - started:.2f}s)",
        flush=True,
    )
    print(
        "Scope: deterministic CPU/tooling checks; no runtime, GPU, performance or native-build qualification."
    )
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
