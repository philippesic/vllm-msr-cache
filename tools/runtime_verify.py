from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

UNAVAILABLE = "UNAVAILABLE: run with the runtime Python; install runtime/requirements.txt with --require-hashes in a separate environment."


def write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")


def snapshot(root: Path, destination: Path) -> dict[str, str]:
    from tools.repo_policy import git_paths

    identities = {}
    for relative in git_paths(root):
        source = root / relative
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"unsupported executable input: {relative}")
        data = source.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
        identities[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    return identities


def verify(directory: Path) -> dict[str, Any]:
    admission = json.loads((directory / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((directory / "terminal.json").read_text())
    if (
        admission.get("schema") != "functional-admission-v1"
        or terminal.get("schema") != "functional-terminal-v1"
    ):
        raise ValueError("unsupported receipt reader version")
    for filename, digest in admission.get("model_sha256", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("model drift or missing retained dependency")
    for relative, digest in admission["source_sha256"].items():
        if (
            hashlib.sha256((directory / "source" / relative).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError("snapshot drift")
    for name, digest in terminal["artifacts_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("receipt artifact drift")
    if terminal["exit_code"] != 0:
        raise ValueError("attempt did not qualify")
    if set(terminal["artifacts_sha256"]) != {"execution.log", "observed.json"}:
        raise ValueError("incomplete artifact identity set")
    if (
        terminal["attempts"] != 1
        or terminal["scope"] != admission["mode"]
        or type(admission["gpu"]) is not bool
    ):
        raise ValueError("admission/terminal disagreement")
    if (
        not 0 <= terminal["elapsed_seconds"] <= terminal["cumulative_seconds"]
        or terminal["cumulative_seconds"] > admission["budget_seconds"] + 60
    ):
        raise ValueError("execution budget disagreement")
    observed = json.loads((directory / "observed.json").read_text())
    if (
        observed.get("schema") != "runtime-observation-v1"
        or observed["exit_code"]
        or observed["failed"]
        or observed["collected"] <= 0
    ):
        raise ValueError("invalid runtime observations")
    manifest = "gpu-tests.json" if admission["gpu"] else "cpu-tests.json"
    expected = json.loads((directory / "source/runtime" / manifest).read_text())
    if observed["collected"] != len(expected):
        raise ValueError("execution count disagrees with selected manifest")
    if observed["cuda_initialized"] != admission["gpu"]:
        raise ValueError("execution scope disagrees with CUDA observation")
    for name, module in observed.get("loaded_project_and_runtime_modules", {}).items():
        if name.startswith("wait_hint_offload") and not Path(
            module["path"]
        ).is_relative_to(directory / "source"):
            raise ValueError("loaded product is outside attempt snapshot")
        if name.startswith("wait_hint_offload"):
            relative = str(Path(module["path"]).relative_to(directory / "source"))
            if admission["source_sha256"].get(relative) != module["sha256"]:
                raise ValueError("loaded product disagrees with frozen source")
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.verify:
        try:
            print(json.dumps(verify(args.verify.resolve()), sort_keys=True))
            return 0
        except (OSError, ValueError, KeyError) as error:
            print(f"INVALID: {error}")
            return 1
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        print(UNAVAILABLE)
        return 2
    from wait_hint_offload.runtime_pin import VLLM_VERSION

    if version != VLLM_VERSION:
        parser.error(f"vLLM {VLLM_VERSION} required, found {version}")
    from tools.config import attempts_root, check_environment
    from tools.process import run_owned

    root = Path(__file__).resolve().parents[1]
    mode = "hardware" if args.gpu else "runtime-cpu"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory = (args.out or attempts_root() / f"{mode}-{stamp}").resolve()
    if not directory.is_relative_to(root / "artifacts") or not 0 < args.timeout <= 1800:
        parser.error(
            "use a new artifacts/ output directory and timeout in 1..1800 seconds"
        )
    if args.gpu and (args.model is None or not args.model.is_dir()):
        parser.error(
            "hardware verification needs --model pointing to a local model directory"
        )
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "source"
    identity = snapshot(root, source)
    model_identity = (
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.model.resolve().rglob("*")
            if p.is_file()
        }
        if args.model
        else {}
    )
    admission = {
        "model_sha256": model_identity,
        "schema": "functional-admission-v1",
        "goal": "minimal wait-hint functional qualification",
        "mode": mode,
        "gpu": args.gpu,
        "source_sha256": identity,
        "mechanism": "CPU exact offload with reachable-hold advice; disabled control shares correctness guards",
        "limiting_risk": "one pinned dense FP16 TP1 runtime; performance and other hardware unqualified",
        "budget_seconds": args.timeout,
        "retry_limit": 0,
        "stop_action": "terminate owned process group",
        "authorization": "external task scope; this record grants no authority",
        "evidence_scope": "functional correctness only",
    }
    write_record(directory / "admission.json", admission)
    selected = "tests/runtime/gpu_cases.py" if args.gpu else "tests/runtime"
    manifest = "runtime/gpu-tests.json" if args.gpu else "runtime/cpu-tests.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        selected,
        "-p",
        "tools.pytest_gate",
        "--gate-manifest",
        manifest,
        "-p",
        "tools.runtime_capture",
        "--runtime-capture",
        str(directory / "observed.json"),
    ]
    environment = check_environment(source)
    if args.gpu:
        from tools.config import hardware_environment

        environment = hardware_environment(source, args.model)
    started = time.monotonic()
    code = 1
    failure_detail: str | None = None
    try:
        with (directory / "execution.log").open("x") as log:
            code = run_owned(command, environment, source, args.timeout, log)
    except OSError as failure:
        failure_detail = str(failure)
    artifacts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (directory / "execution.log", directory / "observed.json")
        if p.exists()
    }
    terminal = {
        "schema": "functional-terminal-v1",
        "exit_code": code,
        "elapsed_seconds": time.monotonic() - started,
        "error": failure_detail,
        "artifacts_sha256": artifacts,
        "attempts": 1,
        "cumulative_seconds": time.monotonic() - started,
        "scope": mode,
        "performance_evaluated": False,
        "cause": "confirmed exit status; detailed diagnosis in execution.log",
    }
    write_record(directory / "terminal.json", terminal)
    elapsed = terminal["elapsed_seconds"]
    if code != 0:
        reason = "timed out" if code == 124 else f"exit {code}"
        print(
            f"{mode}: FAILED ({reason}, {elapsed:.0f}s); log {directory / 'execution.log'}"
        )
        return 1
    try:
        verify(directory)
    except (OSError, ValueError, KeyError) as failure:
        print(f"{mode}: INVALID ({failure}); record {directory}")
        return 1
    observed = json.loads((directory / "observed.json").read_text())
    print(
        f"{mode}: PASSED ({observed['collected']} tests, {elapsed:.0f}s); record {directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
