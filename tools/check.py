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
            errors = runtime_policy.runtime_interface_check(root)
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
                    "skipped: runtime qualification requires explicit pinned Python and external --out"
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
    parser.add_argument(
        "--test", help="Run selected CPU tests; partial, never full acceptance"
    )
    parser.add_argument("--update-tests", action="store_true")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.update_tests:
        import json

        if args.runtime_python:
            inventories = [
                ("tests/runtime", "runtime/cpu-tests.json"),
                ("tests/runtime/gpu_cases.py", "runtime/gpu-tests.json"),
            ]
        else:
            inventories = [("tests", "tools/cpu-tests.json")]
        for selected, manifest in inventories:
            command = [sys.executable, "-m", "pytest", "--collect-only", "-q", selected]
            if args.runtime_python:
                command = [
                    str(args.runtime_python),
                    "-m",
                    "tools.run_runtime",
                    str(args.runtime_python),
                    "-m",
                    "pytest",
                    "--collect-only",
                    "-q",
                    selected,
                ]
            else:
                command.append("--ignore=tests/runtime")
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=config.check_environment(ROOT),
                capture_output=True,
                text=True,
            )
            if result.returncode:
                print(result.stderr + result.stdout)
                return result.returncode
            manifest_path = ROOT / manifest
            previous = (
                set(json.loads(manifest_path.read_text()))
                if manifest_path.exists()
                else set()
            )
            collected = sorted(
                line
                for line in result.stdout.splitlines()
                if line.startswith("tests/") and "::" in line
            )
            print(manifest, "Added:", sorted(set(collected) - previous))
            print(manifest, "Removed:", sorted(previous - set(collected)))
            manifest_path.write_text(json.dumps(collected, indent=2) + "\n")
        return 0
    if args.runtime_python:
        if args.out is None:
            parser.error("--runtime-python requires a new external --out")
        selected_files = sorted(
            (ROOT / "wait_hint_offload/vllm_adapter").glob("*.py")
        ) + sorted((ROOT / "tests/runtime").glob("*.py"))
        command = [
            "mypy",
            "--strict",
            "--follow-imports=silent",
            "--python-executable",
            str(args.runtime_python),
            *[str(path) for path in selected_files],
        ]
        print(
            "Runtime lane: strict adapter/test types, then pinned CPU integration; GPU omitted",
            flush=True,
        )
        code = run(command, ROOT)
        if code:
            return code
        return run(
            [
                str(args.runtime_python),
                "-m",
                "tools.runtime_verify",
                "--out",
                str(args.out),
            ],
            ROOT,
        )
    if args.test:
        print("PARTIAL CPU tests; all other acceptance groups omitted")
        return run([sys.executable, "-m", "pytest", "-q", args.test], ROOT)
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
        "--ignore=tests/runtime",
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
