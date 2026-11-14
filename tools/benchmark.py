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
from tools.runtime_verify import write_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded benchmark qualification")
    sub = parser.add_subparsers(dest="operation", required=True)
    pr_events = sub.add_parser("pr-reconcile-events")
    pr_events.add_argument("directory", type=Path)
    pr_events.add_argument("--out", type=Path, required=True)
    feasibility = sub.add_parser("pr-feasibility")
    feasibility.add_argument("--runtime-python", type=Path, required=True)
    feasibility.add_argument("--upstream", type=Path, required=True)
    feasibility.add_argument("--out", type=Path, required=True)
    feasibility.add_argument("--private-state", action="store_true")
    feasibility.add_argument("--hold-policy", action="store_true")
    feasibility.add_argument("--bookkeeping", action="store_true")
    feasibility.add_argument("--post-response", action="store_true")
    feasibility.add_argument("--instrument", action="store_true")
    feasibility_native = sub.add_parser("pr-feasibility-native")
    feasibility_native.add_argument("--out", type=Path, required=True)
    feasibility_native.add_argument("--private-state", action="store_true")
    feasibility_native.add_argument("--hold-policy", action="store_true")
    feasibility_native.add_argument("--bookkeeping", action="store_true")
    feasibility_native.add_argument("--post-response", action="store_true")
    feasibility_native.add_argument("--instrument", action="store_true")
    feasibility_verify = sub.add_parser("verify-pr-feasibility")
    feasibility_verify.add_argument("directory", type=Path)
    for operation in ("pr-admit-w1", "pr-w1-native"):
        workload = sub.add_parser(operation)
        if operation == "pr-admit-w1":
            workload.add_argument("--runtime-python", type=Path, required=True)
        workload.add_argument("--dataset", type=Path, required=True)
        workload.add_argument("--model", action="append", required=True)
        workload.add_argument("--out", type=Path, required=True)
    workload_freeze = sub.add_parser("pr-freeze-w1")
    workload_freeze.add_argument("directory", type=Path)
    workload_freeze.add_argument("--out", type=Path, required=True)
    workload_verify = sub.add_parser("verify-pr-w1")
    workload_verify.add_argument("directory", type=Path)
    pr_schedule = sub.add_parser("pr-plan-w1")
    pr_schedule.add_argument("--data", type=Path, required=True)
    pr_schedule.add_argument("--model", choices=("D8", "HM", "HS"), required=True)
    pr_schedule.add_argument(
        "--partition", choices=("calibration", "pilot"), required=True
    )
    pr_schedule.add_argument("--seed", type=int, required=True)
    pr_schedule.add_argument("--arrival-rate", type=float, required=True)
    pr_schedule.add_argument("--one-shot-rate", type=float, required=True)
    pr_schedule.add_argument("--horizon", type=float, required=True)
    pr_schedule.add_argument("--out", type=Path, required=True)
    pr_client = sub.add_parser("pr-client-native")
    pr_client.add_argument("--schedule", type=Path, required=True)
    pr_client.add_argument(
        "--arm", choices=("A1", "A2", "A3", "A4t", "A3p", "A4p", "A5"), required=True
    )
    pr_client.add_argument("--url", required=True)
    pr_client.add_argument("--out", type=Path, required=True)
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
    command.add_argument("--trace-case", choices=("T1", "T2", "I1"))
    command.add_argument("--trace-seconds", type=int, default=180)
    command.add_argument("--pressure-prefixes", type=int, default=24)
    command.add_argument(
        "--coding-kind",
        choices=(
            "repair",
            "regression",
            "navigation",
            "terminal",
            "grounded",
            "branching",
        ),
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
        "--profile",
        choices=(
            "functional",
            "compare",
            "release",
            "pressure-smoke",
            "pressure-advice",
            "pressure-full",
        ),
        required=True,
    )
    plan.add_argument("--calibration", type=Path, action="append", default=[])
    pressure = sub.add_parser("pressure")
    pressure.add_argument(
        "--profile", choices=("smoke", "advice", "full"), required=True
    )
    pressure.add_argument("--runtime-python", type=Path, required=True)
    pressure.add_argument("--lmcache-python", type=Path)
    pressure.add_argument("--model", action="append", required=True)
    pressure.add_argument("--arm", action="append", default=[])
    pressure.add_argument("--seed", type=int, action="append", default=[])
    pressure.add_argument("--out", type=Path, required=True)
    pressure_verify = sub.add_parser("verify-pressure")
    pressure_verify.add_argument("directory", type=Path, nargs="+")
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
    campaign.add_argument("--resume", action="store_true")
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
    grounded = sub.add_parser("grade-grounded")
    grounded.add_argument("--dataset", type=Path, required=True)
    grounded.add_argument("--prediction", type=Path, required=True)
    grounded.add_argument("--instance", required=True)
    grounded.add_argument("--out", type=Path, required=True)
    grounded_verify = sub.add_parser("verify-grounded")
    grounded_verify.add_argument("directory", type=Path)
    branching_verify = sub.add_parser("verify-branching")
    branching_verify.add_argument("directory", type=Path)
    for name in ("simulate", "simulate-native"):
        simulation = sub.add_parser(name)
        if name == "simulate":
            simulation.add_argument("--runtime-python", type=Path, required=True)
        simulation.add_argument(
            "--scenarios", type=Path, default=Path("benchmarks/replay/scenarios.json")
        )
        simulation.add_argument(
            "--shapes",
            type=Path,
            default=Path("benchmarks/replay/release-01-shapes.json"),
        )
        simulation.add_argument("--scenario", action="append", default=[])
        simulation.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if hasattr(args, "out") and not args.out.resolve().is_relative_to(
        root / "artifacts"
    ):
        parser.error("write new project output under artifacts/ in this repository")
    if args.operation == "pr-plan-w1":
        from tools.benchmarking.pr_workload import validate_schedule, w1_schedule

        result = w1_schedule(
            args.data,
            args.model,
            args.partition,
            args.seed,
            args.arrival_rate,
            args.one_shot_rate,
            args.horizon,
        )
        validate_schedule(result)
        write_record(args.out, result)
        print(
            json.dumps(
                {
                    "schedule": str(args.out),
                    "sessions": len(result["sessions"]),
                    "one_shots": len(result["one_shots"]),
                }
            )
        )
        return 0
    if args.operation == "pr-reconcile-events":
        from tools.benchmarking.pr_events import reconcile_events

        write_record(args.out, reconcile_events(args.directory))
        return 0
    if args.operation == "pr-client-native":
        import asyncio

        from tools.benchmarking.pr_client import run_schedule

        asyncio.run(
            run_schedule(
                args.url, json.loads(args.schedule.read_text()), args.arm, args.out
            )
        )
        return 0
    if args.operation in {
        "pr-admit-w1",
        "pr-w1-native",
        "verify-pr-w1",
        "pr-freeze-w1",
    }:
        from tools.benchmarking import w1

        if args.operation == "pr-w1-native":
            w1.admit_native(args.dataset, args.model, args.out)
            return 0
        result = (
            w1.admit(args.runtime_python, args.dataset, args.model, args.out)
            if args.operation == "pr-admit-w1"
            else w1.freeze(args.directory, args.out)
            if args.operation == "pr-freeze-w1"
            else w1.verify(args.directory)
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.operation == "pr-feasibility-native" and args.instrument:
        from tools.benchmarking.instrument import observe_instrument

        write_record(args.out, observe_instrument(args.out.parent / "instrument"))
        return 0
    if args.operation == "pr-feasibility-native" and args.post_response:
        from wait_hint_offload.vllm_adapter.policy_delivery_checks import (
            observe_delivery,
        )

        write_record(args.out, observe_delivery())
        return 0
    if args.operation == "pr-feasibility-native" and args.bookkeeping:
        from wait_hint_offload.vllm_adapter.policy_benchmark import observe_bookkeeping

        write_record(
            args.out,
            observe_bookkeeping(
                lambda cell: print(
                    cell["policy"],
                    cell["chunks"],
                    cell["held_sessions"],
                    cell["evict_p99_ns"],
                    cell["owned_metadata_bytes"],
                    flush=True,
                )
            ),
        )
        return 0
    if args.operation == "pr-feasibility-native" and args.hold_policy:
        from wait_hint_offload.vllm_adapter.policy_checks import observe_hold_policy

        write_record(args.out, observe_hold_policy())
        return 0
    if args.operation == "pr-feasibility-native":
        from wait_hint_offload.vllm_adapter.feasibility import (
            observe_interfaces,
            observe_private_context,
        )

        write_record(
            args.out,
            observe_private_context() if args.private_state else observe_interfaces(),
        )
        return 0
    if args.operation in {"pr-feasibility", "verify-pr-feasibility"}:
        from tools.benchmarking.feasibility import qualify
        from tools.benchmarking.feasibility import verify as verify_feasibility

        result = (
            qualify(
                args.runtime_python,
                args.upstream,
                args.out,
                private_state=args.private_state,
                hold_policy=args.hold_policy,
                bookkeeping=args.bookkeeping,
                post_response=args.post_response,
                instrument=args.instrument,
            )
            if args.operation == "pr-feasibility"
            else verify_feasibility(args.directory)
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.operation == "simulate":
        from tools.benchmarking.replay import simulate

        return simulate(
            root,
            args.runtime_python.absolute(),
            args.scenarios,
            args.shapes,
            args.scenario,
            args.out,
        )
    if args.operation == "simulate-native":
        from tools.benchmarking.replay import simulate_native

        print(
            json.dumps(
                simulate_native(
                    args.scenarios.resolve(),
                    args.shapes.resolve(),
                    args.scenario,
                    args.out.resolve(),
                ),
                indent=2,
            )
        )
        return 0
    if args.operation == "verify-branching":
        from tools.benchmarking.branching import verify_branching

        print(json.dumps(verify_branching(args.directory), indent=2))
        return 0
    if args.operation in ("grade-grounded", "verify-grounded"):
        from tools.benchmarking.grounded import grade_grounded, verify_grounded

        result = (
            verify_grounded(args.directory)
            if args.operation == "verify-grounded"
            else grade_grounded(
                root, args.dataset, args.prediction, args.instance, args.out
            )
        )
        print(json.dumps(result, indent=2))
        return 0
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
            args.resume,
        )
    if args.operation == "verify-campaign":
        from tools.benchmarking.campaign import verify_campaign

        result = verify_campaign(args.directory)
        print(json.dumps(result, indent=2))
        return 0
    if args.operation == "patch-runtime":
        from tools.benchmarking.patches import apply_runtime_patches

        print(json.dumps(apply_runtime_patches(root, args.runtime_python, args.names)))
        return 0
    if args.operation == "plan":
        if args.profile.startswith("pressure-"):
            from tools.benchmarking.pressure import projection

            print(
                json.dumps(
                    projection(
                        root, args.profile.removeprefix("pressure-"), args.calibration
                    ),
                    indent=2,
                )
            )
            return 0
        print(json.dumps(describe(args.profile), indent=2))
        return 0
    if args.operation == "pressure":
        from tools.benchmarking.pressure import run_pressure

        models = {}
        for value in args.model:
            identity, separator, path = value.partition("=")
            if not separator or identity in models:
                raise ValueError("models require unique ID=PATH entries")
            models[identity] = Path(path).resolve()
        summary = run_pressure(
            root,
            args.profile,
            models,
            args.runtime_python.absolute(),
            args.lmcache_python.absolute() if args.lmcache_python else None,
            args.out.resolve(),
            args.arm,
            args.seed,
        )
        write = args.out.resolve() / "report.json"
        write.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"report": str(write)}))
        return 0
    if args.operation == "verify-pressure":
        from tools.benchmarking.pressure import verify_pressure

        print(
            json.dumps(
                verify_pressure(*(path.resolve() for path in args.directory)), indent=2
            )
        )
        return 0
    if args.operation == "models":
        print((root / "benchmarks/models.json").read_text())
        return 0
    try:
        if args.operation == "functional":
            if not 0 < args.timeout <= 900:
                raise ValueError("functional budget must be in 1..900 seconds")
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
        panel = None
        if args.trace_case:
            if any((args.coding_dataset, args.harness_python, args.instance)):
                parser.error("trace and coding probes must be separate")
            if (
                not 60 <= args.trace_seconds <= 600
                or not 2 <= args.pressure_prefixes <= 64
            ):
                parser.error(
                    "trace requires 60..600 seconds and 2..64 pressure prefixes"
                )
            panel = (
                args.runtime_python.absolute(),
                [
                    {
                        "identity": f"trace/{args.trace_case}/{args.system}",
                        "kind": "trace",
                        "case": args.trace_case,
                        "seconds": args.trace_seconds,
                        "pressure_prefixes": args.pressure_prefixes,
                    }
                ],
            )
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
            panel,
        )
        print(json.dumps({"exit_code": status, "attempt": str(args.out)}))
        return status
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
