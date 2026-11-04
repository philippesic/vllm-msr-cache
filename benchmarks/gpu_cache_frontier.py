"""Reproduce the frozen live profile at different GPU KV cache capacities.

This explicitly invokes the retained research driver, rather than transferring
its qualification to the newer general serving suite. Run prepare, run, report
as separate commands. Every plan and attempted invocation remains immutable.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from harness.gpu_lock import gpu_reserved

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "output/prefetch-native-20261102"
EVIDENCE = ROOT / "evidence/live-optimization-2080ti-20261103"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def quantile(values, fraction):
    values = sorted(values)
    point = fraction * (len(values) - 1)
    index = int(point)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (point - index)


def prepare(args):
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    original = read(EVIDENCE / "priority-confirmation-frozen.json")
    profile = read(EVIDENCE / "qualified-live-profile.json")
    source = ROOT / profile["qualified_source_directory"]
    for name, expected in profile["source_pins"].items():
        if sha(source / name) != expected:
            raise ValueError("Qualified source changed: " + name)
    for name, expected in profile["runtime_pins"].items():
        if sha(CAMPAIGN / name) != expected:
            raise ValueError("Qualified runtime changed: " + name)
    templates = {arm: next(x for x in original["execution_order"] if x["arm"] == arm)["args"]
                 for arm in ("native8", "candidate")}
    runs = []
    # All sizes and outcomes are retained. The same offered workload and model
    # context remain fixed even if a smaller cache preempts or fails.
    for repeat in range(args.repeats):
        for seed in args.seeds:
            # Reverse the entire order on odd repeats; alternate within-size
            # arm order as well. Confirmation uses a new, separate plan.
            sizes = args.blocks if repeat % 2 == 0 else list(reversed(args.blocks))
            for index, blocks in enumerate(sizes):
                arms = ["native8", "candidate"] if (index + repeat) % 2 == 0 else ["candidate", "native8"]
                for arm in arms:
                    name = f"{args.prefix}-r{repeat:02d}-s{seed}-g{blocks}-{arm}"
                    command_args = list(templates[arm])
                    for flag, value in (("--name", name), ("--source-dir", source),
                                        ("--gpu-blocks", blocks), ("--seed", seed)):
                        command_args[command_args.index(flag) + 1] = str(value)
                    if (CAMPAIGN / "runs" / name).exists():
                        raise FileExistsError(name)
                    runs.append(dict(name=name, arm=arm, seed=seed, repeat=repeat,
                                     gpu_blocks=blocks, gate_enabled=arm == "candidate",
                                     priority_credit_ms=8000, args=command_args))
    files = {ROOT / name for name in original["file_pins"] if name.endswith(".py")}
    files.update(source.rglob("*.py"))
    files.update((ROOT / "benchmarks/research_checks").glob("*.py"))
    files.add(ROOT / "harness/gpu_lock.py")
    files.update([Path(__file__).resolve(), EVIDENCE / "qualified-live-profile.json",
                  CAMPAIGN / "model-selection.json",
                  CAMPAIGN / "fresh-agent-priority-nov3/session-corpus.json"])
    # Freeze the full installed Python runtime, not the concurrently evolving
    # vendored runtime; five executed replacements are separately pinned above.
    import sysconfig
    installed = Path(sysconfig.get_paths()["purelib"]) / "vllm"
    files.update(installed.rglob("*.py"))
    pins = {str(path): sha(path) for path in sorted(files) if path.is_file()}
    plan = dict(schema="gpu-cache-live-frontier-v1", created_ns=time.time_ns(),
                python=original["python"], runner=str(ROOT / original["runner"]),
                output=str(destination), mode=args.mode, runs=runs, file_pins=pins,
                qualified_profile=str(EVIDENCE / "qualified-live-profile.json"),
                workload="Previously measured public SWE repository triage inputs; actual fresh natural-EOS generations and histories. Reused corpus, not held-out task/population evidence.",
                constants=dict(cpu_blocks=2048, block_bytes=917504, context=7680,
                               active_sequences=8, batch_tokens=1024, return_credit_ms=8000),
                interpretation="Screening is advisory. Compare candidate and native at every smaller capacity, and both against native1536. Similarity must be judged from estimates and intervals, not absence of significance. No output caps, retries, replacements or truncation.",
                stop_conditions=["changed frozen source", "owned cleanup failure", "non-idle GPU before invocation", "device-side or KV corruption fault", "three consecutive incomplete invocations"],
                analysis_unit="Whole-workflow paired repeats conditional on fixed corpus traces; requests are not independent replications.")
    write(destination / "plan.json", plan)
    print(json.dumps(dict(plan=str(destination / "plan.json"), invocations=len(runs), pins=len(pins))))


def verify_pins(plan):
    changed = [name for name, expected in plan["file_pins"].items() if not Path(name).is_file() or sha(name) != expected]
    if changed:
        raise RuntimeError("Frozen files changed: " + repr(changed))


def idle_gpu():
    with open("/tmp/vllm-semantic-cache-gpu.lock", "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           check=True, text=True, capture_output=True).stdout.strip()
    if state:
        raise RuntimeError("GPU has active compute processes: " + state)


@gpu_reserved
def run(args):
    plan = read(args.plan)
    destination = Path(plan["output"])
    receipt = destination / "execution.json"
    if receipt.exists():
        raise FileExistsError("Never overwrite or silently resume an execution receipt")
    verify_pins(plan)
    idle_gpu()
    state = dict(status="running", plan_sha256=sha(args.plan), started_ns=time.time_ns(), runs=[])
    write(receipt, state)
    failures = 0
    try:
        for item in plan["runs"]:
            verify_pins(plan)
            idle_gpu()
            directory = CAMPAIGN / "runs" / item["name"]
            if directory.exists():
                raise FileExistsError(directory)
            console = destination / (item["name"] + ".log")
            print("START " + item["name"], flush=True)
            state["current"] = item["name"]
            write(receipt, state)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            for key in ("VLLM_CLI", "VLLM_CLI_MODULE", "PYTHONPATH"):
                env.pop(key, None)
            with console.open("w") as stream:
                completed = subprocess.run([plan["python"], plan["runner"], *item["args"]],
                                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            raw_path = directory / "raw.json.gz"
            raw = read(raw_path) if raw_path.exists() else {}
            # The existing arithmetic validator expects this exact archived
            # helper; archive the unchanged executed helper without substitution.
            if directory.exists():
                shutil.copyfile(CAMPAIGN / "return_priority_nov3.py", directory / "return_priority_helper_astra.py")
            row = dict(name=item["name"], exit_code=completed.returncode,
                       raw=str(raw_path), raw_sha256=sha(raw_path) if raw_path.exists() else None,
                       status=raw.get("status", "missing_raw"), error=raw.get("error"),
                       requests=len(raw.get("requests", [])),
                       natural_eos=sum(q.get("natural_eos", False) for q in raw.get("requests", [])),
                       owned_descendants_remaining=raw.get("owned_descendants_remaining"))
            state["runs"].append(row)
            state.pop("current", None)
            write(receipt, state)
            print("END " + json.dumps(row), flush=True)
            verify_pins(plan)
            idle_gpu()
            okay = completed.returncode == 0 and row["status"] == "complete" and row["requests"] == row["natural_eos"] == 72
            failures = 0 if okay else failures + 1
            fault = (raw.get("error", "") + raw.get("error_traceback", "")).lower()
            for log in directory.glob("server*.log.gz"):
                with gzip.open(log, "rt", errors="replace") as stream:
                    fault += stream.read().lower()
            if row["owned_descendants_remaining"] or any(term in fault for term in
                    ("actual kv transfer mismatch", "device-side assert triggered", "an illegal memory access")):
                raise RuntimeError("Correctness or cleanup fault")
            if failures >= 3:
                raise RuntimeError("Three consecutive incomplete invocations")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="stopped", error=repr(error))
        raise
    finally:
        state["completed_ns"] = time.time_ns()
        write(receipt, state)


def native_delta(raw, name, label=None):
    def value(text):
        values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                  if (line.startswith(name + "{") or line.startswith(name + " "))
                  and (label is None or label in line)]
        return sum(values) if values else None
    before, after = value(raw.get("metrics_initial", "")), value(raw.get("metrics_final", ""))
    return after - before if before is not None and after is not None and after >= before else None


def summarize(raw, item):
    from .research_checks.http import verify_http
    from .research_checks.priority import verify_priority
    http, priority = verify_http(raw), verify_priority(raw, item, CAMPAIGN)
    requests = raw["requests"]
    groups = {}
    for label, rows in (("all", requests), ("initial", [q for q in requests if not q["turn"]]),
                        ("return", [q for q in requests if q["turn"]])):
        groups[label] = {}
        for metric, endpoint in (("ttft", "first_token_ns"), ("e2e", "request_end_ns")):
            values = [(q[endpoint] - q["request_start_ns"]) / 1e6 for q in rows if endpoint in q]
            groups[label].update({f"mean_{metric}_ms": statistics.mean(values) if values else None,
                                 f"p95_{metric}_ms": quantile(values, .95) if values else None,
                                 f"max_{metric}_ms": max(values) if values else None})
    seconds = (raw["load_end_ns"] - raw["load_start_ns"]) / 1e9
    tokens = sum(len(q.get("output_token_ids", [])) for q in requests)
    peaks = {}
    for name, key in (("serving", "resource_samples"), ("startup", "startup_resource_samples")):
        samples = [s["gpu"]["memory_used_bytes"] / 2**20 for s in raw.get(key, []) if "gpu" in s]
        peaks[name + "_vram_mib"] = max(samples) if samples else None
    valid_eos = all(q.get("success") and q.get("natural_eos") and q.get("terminal_complete")
                    and q.get("finish_reasons") == ["stop"] for q in requests)
    by_index = {q["index"]: q for q in requests}
    contiguous = all(q["payload"]["prompt"][:len(past) // 16 * 16] == past[:len(past) // 16 * 16]
                     for q in requests if q["turn"]
                     for previous in [by_index[q["index"] - 1]]
                     for past in [previous["payload"]["prompt"] + previous["output_token_ids"][:-1]])
    cfg = raw["config"].get("kv_connector_extra_config", {})
    eligible = (raw["status"] == "complete" and raw["performance_eligible"] and len(requests) == 72
                and valid_eos and contiguous and http["qualified"] and priority["qualified"]
                and raw.get("owned_descendants_remaining") == []
                and raw["geometry"]["gpu_blocks"] == item["gpu_blocks"]
                and raw["geometry"]["cpu_blocks"] == 2048
                and (not item["gate_enabled"] or cfg.get("preservation_audit") is False))
    return dict(name=item["name"], arm=item["arm"], seed=item["seed"], repeat=item["repeat"],
                gpu_blocks=item["gpu_blocks"], eligible=eligible, requests=len(requests),
                natural_eos=sum(q.get("natural_eos", False) for q in requests),
                load_s=seconds, output_tokens=tokens, tokens_s=tokens / seconds,
                workflows_s=24 / seconds,
                tokens_s_gib=tokens / seconds / (peaks["serving_vram_mib"] / 1024),
                **peaks, groups=groups, contiguous_returns=contiguous,
                preemptions=native_delta(raw, "vllm:num_preemptions_total"),
                transfer_bytes={direction: native_delta(raw, "vllm:kv_offload_total_bytes_total", f'transfer_type="{direction}"')
                                for direction in ("GPU_to_CPU", "CPU_to_GPU")},
                gate_summary=priority["per_hint"],
                HTTP_qualified=http["qualified"], priority_gate_qualified=priority["qualified"],
                validation_issues=dict(http=http["issues"], priority=priority["issues"]),
                first_prompts={q["index"]: hashlib.sha256(json.dumps(q["payload"]["prompt"]).encode()).hexdigest()
                               for q in requests if not q["turn"]}, private_waits=raw["workload"]["private_wait_ms"])


def report(args):
    plan = read(args.plan)
    rows = []
    for item in plan["runs"]:
        path = CAMPAIGN / "runs" / item["name"] / "raw.json.gz"
        if not path.exists():
            rows.append(dict(name=item["name"], eligible=False, status="missing_raw"))
            continue
        raw = read(path)
        try:
            row = summarize(raw, item)
        except Exception as error:
            row = dict(name=item["name"], eligible=False, status=raw.get("status"), error=repr(error))
        rows.append(dict(row, raw=str(path), raw_sha256=sha(path)))
    contrasts = []
    for row in rows:
        if not row["eligible"]:
            continue
        for reference in rows:
            if not reference["eligible"] or reference["seed"] != row["seed"] or reference["repeat"] != row["repeat"]:
                continue
            if (reference["arm"] == "native8" and reference["gpu_blocks"] in (1536, row["gpu_blocks"])):
                if reference["name"] == row["name"]:
                    continue
                if row["first_prompts"] != reference["first_prompts"] or row["private_waits"] != reference["private_waits"]:
                    raise ValueError("Mismatched offered inputs")
                contrasts.append(dict(candidate=row["name"], reference=reference["name"],
                                      arm=row["arm"], gpu_blocks=row["gpu_blocks"], reference_gpu_blocks=reference["gpu_blocks"],
                                      seed=row["seed"], repeat=row["repeat"],
                                      tokens_percent=100 * (row["tokens_s"] / reference["tokens_s"] - 1),
                                      workflows_percent=100 * (row["workflows_s"] / reference["workflows_s"] - 1),
                                      tokens_per_gib_percent=100 * (row["tokens_s_gib"] / reference["tokens_s_gib"] - 1),
                                      vram_saved_mib=reference["serving_vram_mib"] - row["serving_vram_mib"],
                                      groups={g: {k: row["groups"][g][k] - reference["groups"][g][k]
                                                  for k in row["groups"][g]} for g in row["groups"]}))
    result = dict(schema="gpu-cache-live-frontier-report-v1", plan_sha256=sha(args.plan),
                  all_planned_eligible=all(r["eligible"] for r in rows), rows=rows,
                  contrasts=contrasts, interpretation=plan["interpretation"], confidence_intervals=None)
    # Nominal paired Student-t intervals are conditional on the fixed traces.
    # Withhold aggregate inference unless every predeclared invocation passed.
    if plan["mode"] == "confirm" and result["all_planned_eligible"]:
        from scipy.stats import t
        grouped = collections.defaultdict(list)
        for contrast in contrasts:
            grouped[(contrast["arm"], contrast["gpu_blocks"], contrast["reference_gpu_blocks"])].append(contrast)
        intervals = []
        for (arm, blocks, baseline), pairs in grouped.items():
            if len(pairs) < 2:
                continue
            summary = dict(arm=arm, gpu_blocks=blocks, reference_gpu_blocks=baseline, pairs=len(pairs))
            metrics = {key: [math.log1p(p[key] / 100) for p in pairs]
                       for key in ("tokens_percent", "workflows_percent", "tokens_per_gib_percent")}
            metrics.update({f"{group}.{key}": [p["groups"][group][key] for p in pairs]
                            for group in pairs[0]["groups"] for key in pairs[0]["groups"][group]})
            for key, values in metrics.items():
                average = statistics.mean(values)
                half = float(t.ppf(.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
                estimate = [average, average - half, average + half]
                if key.endswith("_percent"):
                    estimate = [100 * math.expm1(v) for v in estimate]
                summary[key] = dict(zip(("mean", "lower95", "upper95"), estimate))
            summary["vram_saved_mib"] = [p["vram_saved_mib"] for p in pairs]
            intervals.append(summary)
        result["confidence_intervals"] = intervals
    write(Path(plan["output"]) / "report.json", result)
    print(json.dumps(dict(all_planned_eligible=result["all_planned_eligible"],
                          rows=[{k: r.get(k) for k in ("name", "eligible", "tokens_s", "serving_vram_mib", "preemptions", "error")} for r in rows],
                          contrasts=contrasts, confidence_intervals=result["confidence_intervals"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--blocks", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[14009])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--mode", choices=("screen", "confirm"), default="screen")
    for name in ("run", "report"):
        sub.add_parser(name).add_argument("--plan", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
