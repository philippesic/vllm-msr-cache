from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.benchmarking.plan import describe
from tools.benchmarking.probe import probe, verify_probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded benchmark qualification")
    sub = parser.add_subparsers(dest="operation", required=True)
    command = sub.add_parser("probe-model")
    command.add_argument("--runtime-python", required=True, type=Path)
    command.add_argument("--model", required=True, type=Path)
    command.add_argument("--out", required=True, type=Path)
    command.add_argument("--timeout", type=int, default=600)
    command.add_argument("--coding-dataset", type=Path)
    command.add_argument("--harness-python", type=Path)
    command.add_argument("--instance")
    command.add_argument(
        "--backend", choices=("TRITON_ATTN", "FLEX_ATTENTION"), default="TRITON_ATTN"
    )
    command.add_argument(
        "--system",
        choices=("gpu-only", "native-arc", "wait-hint", "lmcache"),
        default="gpu-only",
    )
    plan = sub.add_parser("plan")
    plan.add_argument(
        "--profile", choices=("functional", "compare", "release"), required=True
    )
    sub.add_parser("models")
    verify = sub.add_parser("verify-probe")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.operation == "plan":
        print(json.dumps(describe(args.profile), indent=2))
        return 0
    if args.operation == "models":
        print((root / "benchmarks/models.json").read_text())
        return 0
    try:
        if args.operation == "verify-probe":
            print(json.dumps(verify_probe(args.directory.resolve()), indent=2))
            return 0
        coding = None
        if any((args.coding_dataset, args.harness_python, args.instance)):
            if not all((args.coding_dataset, args.harness_python, args.instance)):
                parser.error(
                    "coding-dataset, harness-python and instance must be supplied together"
                )
            coding = (
                args.harness_python.absolute(),
                args.coding_dataset.resolve(),
                args.instance,
            )
        status = probe(
            root,
            args.runtime_python.absolute(),
            args.model.resolve(),
            args.out.resolve(),
            args.timeout,
            args.system,
            args.backend,
            coding,
        )
        print(json.dumps({"exit_code": status, "attempt": str(args.out)}))
        return status
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
