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

from tools import config, native, repo_policy, results, runtime_policy, unused  # noqa: I001

FIXES = "Fixes for common diagnostics: docs/checks.md, 'Common failures and fixes'."
PROOFS = "tests/test_check_cases.py"


def run(command: list[str], root: Path) -> int:
    env = config.check_environment(root)
    try:
        return subprocess.run(command, cwd=root, env=env, check=False).returncode
    except FileNotFoundError:
        print(f"tool.missing: {command[0]}; run uv sync --locked", flush=True)
        return 1


def component(name: str, root: Path) -> int:
    if name == "results":
        errors = results.check(root)
        for error in errors:
            print(error)
        return int(bool(errors))
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
        if errors:
            print(FIXES)
        else:
            if name == "native":
                print(
                    "skipped: no native sources yet (sm_75 and sm_120; clang-format, clang-tidy, CMake)"
                )
            elif name == "runtime":
                print(
                    "skipped: runtime lane needs --runtime-python; see docs/runtime.md"
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
        return run(
            ["mypy", "--strict", "--cache-dir", str(config.type_cache(root))], root
        )
    if name == "imports":
        return run(["lint-imports", "--config", "pyproject.toml", "--no-cache"], root)
    raise ValueError(f"Unknown component: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic CPU/tooling checks")
    parser.add_argument(
        "--component",
        choices=[
            "results",
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
        "--full",
        action="store_true",
        help="fast lane plus rule proofs; run before committing (the hook runs it)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="only the rule proofs",
    )
    parser.add_argument(
        "--test", help="Run selected CPU tests; partial, never full acceptance"
    )
    parser.add_argument("--update-tests", action="store_true")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--render-results", action="store_true")
    parser.add_argument("--verify-result-sources", type=Path)
    args = parser.parse_args()
    if args.render_results:
        results.render(ROOT)
        return 0
    if args.verify_result_sources:
        results.verify_sources(ROOT, args.verify_result_sources)
        return 0
    if args.update_tests:
        import json

        if args.runtime_python:
            inventories = [
                ("tests/runtime", "runtime/cpu-tests.json"),
                ("tests/runtime/gpu_cases.py", "runtime/gpu-tests.json"),
            ]
        else:
            inventories = [
                ("tests", "tools/cpu-tests.json"),
                (PROOFS, "tools/self-tests.json"),
            ]
        for selected, manifest in inventories:
            command = [sys.executable, "-m", "pytest", "--collect-only", "-q", selected]
            if args.runtime_python:
                command = [
                    str(args.runtime_python),
                    "-m",
                    "pytest",
                    "--collect-only",
                    "-q",
                    selected,
                ]
            elif selected == "tests":
                command.extend(["--ignore=tests/runtime", f"--ignore={PROOFS}"])
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
    if args.runtime_python and args.test:
        print(
            "PARTIAL runtime CPU tests; no types, record or full acceptance", flush=True
        )
        return run([str(args.runtime_python), "-m", "pytest", "-q", args.test], ROOT)
    if args.runtime_python:
        selected_files = sorted(
            (ROOT / "wait_hint_offload/vllm_adapter").glob("*.py")
        ) + sorted((ROOT / "tests/runtime").glob("*.py"))
        command = [
            "mypy",
            "--strict",
            "--cache-dir",
            str(config.type_cache(ROOT, args.runtime_python)),
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
        verify = [str(args.runtime_python), "-m", "tools.runtime_verify"]
        if args.out is not None:
            verify.extend(["--out", str(args.out)])
        return run(verify, ROOT)
    if args.test:
        print("PARTIAL CPU tests; all other acceptance groups omitted")
        return run([sys.executable, "-m", "pytest", "-q", args.test], ROOT)
    if args.component:
        return component(args.component, args.root.resolve())
    if args.root.resolve() != ROOT:
        parser.error("--root is only for isolated component specimens")
    if args.full:
        fast = lane(False)
        proofs = lane(True)
        print(
            f"\nFULL: {'FAILED' if fast or proofs else 'PASSED'} (fast lane {'failed' if fast else 'passed'}, rule proofs {'failed' if proofs else 'passed'})",
            flush=True,
        )
        return int(fast or proofs)
    return lane(args.self_test)


def lane(self_test: bool) -> int:
    started = time.monotonic()
    failed = 0
    skipped = 0
    names = (
        []
        if self_test
        else [
            "results",
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
    label = "rule proofs" if self_test else "CPU pytest"
    print(f"\n=== {label} ===", flush=True)
    manifest = "tools/self-tests.json" if self_test else "tools/cpu-tests.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q" if not self_test else "-v",
        "-p",
        "tools.pytest_gate",
        "--gate-manifest",
        manifest,
    ]
    if self_test:
        command.extend(["-s", PROOFS])
    else:
        command.extend(["--ignore=tests/runtime", f"--ignore={PROOFS}", "tests"])
    code = run(command, ROOT)
    if code:
        print(FIXES)
    failed += bool(code)
    groups = len(names) + 1
    print(
        f"\nRESULT: {groups - failed - skipped} passed, {skipped} skipped, {failed} failed ({time.monotonic() - started:.2f}s)",
        flush=True,
    )
    print(
        "Scope: rule proofs only."
        if self_test
        else "Scope: PARTIAL fast lane; run --full before committing. No runtime, GPU, performance or native-build qualification."
    )
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
