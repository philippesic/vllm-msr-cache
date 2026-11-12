from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.benchmarking.harness import grade_swebench
from tools.benchmarking.plan import describe
from tools.benchmarking.probe import probe, verify_probe
from tools.benchmarking.report import summarize_probe
from tools.config import benchmark_environment
from tools.process import run_owned


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded benchmark qualification")
    sub = parser.add_subparsers(dest="operation", required=True)
    functional = sub.add_parser("functional")
    functional.add_argument("--runtime-python", type=Path, required=True)
    functional.add_argument("--model", type=Path, required=True)
    functional.add_argument("--out", type=Path, required=True)
    functional.add_argument("--timeout", type=int, default=600)
    command = sub.add_parser("probe-model")
    command.add_argument("--runtime-python", required=True, type=Path)
    command.add_argument("--model", required=True, type=Path)
    command.add_argument("--out", required=True, type=Path)
    command.add_argument("--timeout", type=int, default=600)
    command.add_argument("--coding-dataset", type=Path)
    command.add_argument("--harness-python", type=Path)
    command.add_argument("--instance")
    command.add_argument(
        "--coding-kind",
        choices=("repair", "regression", "navigation", "terminal"),
        default="repair",
    )
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
    report = sub.add_parser("report-probe")
    report.add_argument("directory", type=Path)
    grade = sub.add_parser("grade-coding")
    grade.add_argument("--harness-python", type=Path, required=True)
    grade.add_argument("--dataset", type=Path, required=True)
    grade.add_argument("--predictions", type=Path, required=True)
    grade.add_argument("--instance", required=True)
    grade.add_argument("--out", type=Path, required=True)
    patch = sub.add_parser("patch-runtime")
    patch.add_argument("--runtime-python", required=True, type=Path)
    patch.add_argument("names", nargs="+")
    campaign = sub.add_parser("run")
    campaign.add_argument("--profile", choices=("compare", "release"), required=True)
    campaign.add_argument("--runtime-python", type=Path, required=True)
    campaign.add_argument("--lmcache-python", type=Path, required=True)
    campaign.add_argument("--harness-python", type=Path, required=True)
    campaign.add_argument("--dataset", type=Path, required=True)
    campaign.add_argument("--model", action="append", required=True)
    campaign.add_argument("--case", action="append", default=[])
    campaign.add_argument("--out", type=Path, required=True)
    campaign_verify = sub.add_parser("verify-campaign")
    campaign_verify.add_argument("directory", type=Path)
    regression = sub.add_parser("grade-regression")
    regression.add_argument("--dataset", type=Path, required=True)
    regression.add_argument("--prediction", type=Path, required=True)
    regression.add_argument("--instance", required=True)
    regression.add_argument("--out", type=Path, required=True)
    regression_verify = sub.add_parser("verify-regression")
    regression_verify.add_argument("directory", type=Path)
    navigation = sub.add_parser("grade-navigation")
    navigation.add_argument("--dataset", type=Path, required=True)
    navigation.add_argument("--prediction", type=Path, required=True)
    navigation.add_argument("--instance", required=True)
    navigation.add_argument("--out", type=Path, required=True)
    navigation_verify = sub.add_parser("verify-navigation")
    navigation_verify.add_argument("directory", type=Path)
    terminal = sub.add_parser("terminal")
    terminal.add_argument("--harness-python", type=Path, required=True)
    terminal.add_argument("--dataset", type=Path, required=True)
    terminal.add_argument("--instance", required=True)
    terminal.add_argument("--endpoint", default="")
    terminal.add_argument(
        "--agent", choices=("terminus-2", "oracle", "nop"), default="terminus-2"
    )
    terminal.add_argument("--out", type=Path, required=True)
    terminal_verify = sub.add_parser("verify-terminal")
    terminal_verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.operation in ("terminal", "verify-terminal"):
        from tools.benchmarking.terminal import run_terminal, verify_terminal

        result = (
            verify_terminal(args.directory)
            if args.operation == "verify-terminal"
            else run_terminal(
                root,
                args.harness_python,
                args.dataset,
                args.instance,
                args.endpoint,
                args.out.resolve(),
                240,
                args.agent,
            )
        )
        print(json.dumps(result, indent=2))
        return int(result["quality"] == "ungraded")
    if args.operation in ("grade-navigation", "verify-navigation"):
        from tools.benchmarking.navigation import grade_navigation, verify_navigation

        result = (
            verify_navigation(args.directory)
            if args.operation == "verify-navigation"
            else grade_navigation(
                root, args.dataset, args.prediction, args.instance, args.out
            )
        )
        print(json.dumps(result, indent=2))
        return int(result["quality"] == "ungraded")
    if args.operation in ("grade-regression", "verify-regression"):
        from tools.benchmarking.regression import grade_regression, verify_regression

        if args.operation == "verify-regression":
            result = verify_regression(args.directory)
        else:
            result = grade_regression(
                root, args.dataset, args.prediction, args.instance, args.out
            )
        print(json.dumps(result, indent=2))
        return int(result["quality"] == "ungraded")
    if args.operation == "run":
        from tools.benchmarking.campaign import run_campaign

        models = {}
        for value in args.model:
            identity, separator, path = value.partition("=")
            if not separator or identity in models:
                raise ValueError("models require unique ID=PATH entries")
            models[identity] = Path(path).resolve()
        return run_campaign(
            root,
            args.profile,
            models,
            args.runtime_python,
            args.lmcache_python,
            args.harness_python,
            args.dataset,
            args.out,
            args.case,
        )
    if args.operation == "verify-campaign":
        from tools.benchmarking.campaign import verify_campaign

        result = verify_campaign(args.directory)
        print(json.dumps({"status": result["status"], "cells": len(result["results"])}))
        return 0
    if args.operation == "patch-runtime":
        from tools.benchmarking.patches import apply_runtime_patches

        print(json.dumps(apply_runtime_patches(root, args.runtime_python, args.names)))
        return 0
    if args.operation == "plan":
        print(json.dumps(describe(args.profile), indent=2))
        return 0
    if args.operation == "models":
        print((root / "benchmarks/models.json").read_text())
        return 0
    try:
        if args.operation == "functional":
            if not 0 < args.timeout <= 900:
                raise ValueError("functional budget must be in 1..900 seconds")
            if args.out.resolve().is_relative_to(root):
                raise ValueError("functional requires a new external output")
            return run_owned(
                [
                    str(args.runtime_python.absolute()),
                    "-m",
                    "tools.runtime_verify",
                    "--gpu",
                    "--model",
                    str(args.model.resolve()),
                    "--out",
                    str(args.out.resolve()),
                    "--timeout",
                    str(args.timeout),
                ],
                benchmark_environment(root, args.model.resolve(), True),
                root,
                args.timeout + 60,
            )
        if args.operation == "report-probe":
            print(json.dumps(summarize_probe(args.directory.resolve()), indent=2))
            return 0
        if args.operation == "grade-coding":
            result = grade_swebench(
                root,
                args.harness_python.absolute(),
                args.dataset.resolve(),
                args.predictions.resolve(),
                args.instance,
                args.out.resolve(),
            )
            print(json.dumps(result, indent=2))
            return 1 if result["quality"] == "ungraded" else 0
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
            args.coding_kind,
        )
        print(json.dumps({"exit_code": status, "attempt": str(args.out)}))
        return status
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
