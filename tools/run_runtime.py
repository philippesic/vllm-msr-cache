from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

from tools.config import runtime_environment
from tools.process import run_owned


def stage_runtime(root: Path, destination: Path) -> dict[str, str]:
    if sys.flags.optimize:
        raise RuntimeError("optimized execution is unsupported by the pinned runtime")
    if sys.version_info[:3] != (3, 12, 3):
        raise RuntimeError("CPython 3.12.3 required")
    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != "0.26.1rc1.dev75+gdc1be7903":
        raise RuntimeError("wrong vLLM wheel")
    source = Path(str(distribution.locate_file("vllm")))
    compatibility = json.loads((root / "runtime/compatibility.json").read_text())
    for name, digest in compatibility.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"private runtime seam drift: {name}")
    patched: dict[str, str] = {}
    records: list[dict[str, Any]] = json.loads((root / "runtime/rpc.json").read_text())
    for record in records:
        name = str(record["path"])
        original = (source / name).read_text()
        if hashlib.sha256(original.encode()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"runtime source drift: {name}")
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)
        insertions: dict[int, list[str]] = {}
        for method in record["methods"]:
            cls = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == method["class"]
            )
            if any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method["name"]
                for node in cls.body
            ):
                raise RuntimeError("RPC seam already exists")
            if cls.end_lineno is None:
                raise RuntimeError("missing class source extent")
            addition = (
                "\n"
                + "\n".join("    " + line for line in method["source"].splitlines())
                + "\n"
            )
            insertions.setdefault(cls.end_lineno, []).append(addition)
        for position in sorted(insertions, reverse=True):
            lines.insert(position, "".join(insertions[position]))
        patched[name] = "".join(lines)
    package = destination / "vllm"
    for file in distribution.files or ():
        parts = file.parts
        if parts[0] != "vllm" or "__pycache__" in parts or file.suffix == ".pyc":
            continue
        relative = Path(*parts[1:])
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() in patched:
            target.write_text(patched[relative.as_posix()])
        else:
            target.symlink_to(Path(str(distribution.locate_file(file))).absolute())
    patches = {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in patched.items()
    }
    (package / ".wait-hint-overlay.json").write_text(
        json.dumps(
            {"schema": "wait-hint-overlay-v1", "files": {**compatibility, **patches}},
            sort_keys=True,
        )
    )
    return patches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.timeout <= 0:
        parser.error("supply a command and positive timeout")
    if Path(args.command[0]).absolute() != Path(sys.executable).absolute():
        parser.error("child must use the launcher's exact Python executable")
    root = Path(__file__).resolve().parents[1]
    if any(option in {"-O", "-OO"} for option in args.command[1:]):
        parser.error("optimized execution is unsupported")
    signal.signal(signal.SIGTERM, lambda _signal, _frame: sys.exit(143))
    with tempfile.TemporaryDirectory(prefix="wait-hint-runtime-") as temporary:
        stage_runtime(root, Path(temporary))
        environment = runtime_environment(root, Path(temporary), args.gpu)
        return run_owned(args.command, environment, root, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
