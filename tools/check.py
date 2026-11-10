from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tools import config, native, repo_policy, runtime_policy, unused  # noqa: I001


def run(command: list[str], root: Path) -> int:
    env = config.check_environment(root)
    try:
        return subprocess.run(command, cwd=root, env=env, check=False).returncode
    except FileNotFoundError:
        print(f"tool.missing: {command[0]}; run uv sync --locked", flush=True)
        return 1


def component(name: str, root: Path) -> int:
    if name in {"policy", "boundaries", "native", "unused", "runtime"}:
        if name == "policy":
            errors = repo_policy.check(root)
        elif name == "boundaries":
            errors = repo_policy.boundary_policy(root)
        elif name == "unused":
            errors = unused.code_check(root)
        elif name == "runtime":
            errors = runtime_policy.runtime_stub_check(root)
        else:
            errors = native.check(root, repo_policy.git_paths(root))
        for error in errors:
            print(error)
        if not errors:
            if name == "native":
                print(
                    "skipped: no native sources yet (sm_75 and sm_120; clang-format, clang-tidy, CMake)"
                )
            elif name == "runtime":
                print(
                    "skipped: runtime verification is not implemented until source port (command exits 2)"
                )
            else:
                print(f"{name}: OK")
        return int(bool(errors))
    if name == "dependencies":
        return run(unused.dependency_command(root), root)
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
            "unused",
            "dependencies",
            "runtime",
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
        else [
            "format",
            "lint",
            "types",
            "imports",
            "boundaries",
            "policy",
            "unused",
            "dependencies",
            "native",
            "runtime",
        ]
    )
    for name in names:
        before = time.monotonic()
        print(f"\n=== {name} ===", flush=True)
        code = component(name, ROOT)
        failed += bool(code)
        is_skipped = name in {"native", "runtime"} and code == 0
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
