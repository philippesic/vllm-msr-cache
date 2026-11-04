"""Supplement live GPU cache comparisons with tail, allocator and host evidence."""

from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path
import statistics

from scipy.stats import t

from .gpu_cache_frontier import CAMPAIGN, quantile, read, sha, write


def distribution(values):
    return dict(count=len(values), mean=statistics.mean(values), p95=quantile(values, .95), maximum=max(values)) if values else None


def interval(values, percentage=False):
    if percentage:
        values = [math.log1p(value / 100) for value in values]
    mean = statistics.mean(values)
    half = float(t.ppf(.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    limits = [mean, mean - half, mean + half]
    if percentage:
        limits = [100 * math.expm1(value) for value in limits]
    return dict(n=len(values), **dict(zip(("mean", "lower95", "upper95"), limits)))


def resources(raw):
    samples = raw["resource_samples"]
    processes = collections.defaultdict(list)
    for sample in samples:
        for process in sample["processes"]:
            processes[process["identity"]].append(process)
    host = [sample["host_memory"] for sample in samples]
    return dict(
        CUDA_workers=[{key: worker.get(key) for key in
                       ("gpu_allocated_bytes", "gpu_reserved_bytes", "load_peak_allocated_bytes", "load_peak_reserved_bytes")}
                      for worker in raw["postload_export"]["workers"]],
        serving_nvml=distribution([sample["gpu"]["memory_used_bytes"] for sample in samples if "gpu" in sample]),
        startup_nvml=distribution([sample["gpu"]["memory_used_bytes"] for sample in raw["startup_resource_samples"] if "gpu" in sample]),
        server_summed_RSS_peak_bytes=max(sum(process["rss_bytes"] for process in sample["processes"]) for sample in samples),
        driver_RSS_peak_bytes=max(sample["driver_rss_bytes"] for sample in samples),
        sampled_server_CPU_seconds=sum(rows[-1]["cpu_s"] - rows[0]["cpu_s"] for rows in processes.values()),
        sampled_driver_CPU_seconds=samples[-1]["driver_cpu_s"] - samples[0]["driver_cpu_s"],
        host_available_min_bytes=min(sample["available_bytes"] for sample in host),
        host_swap_peak_bytes=max(sample["swap_used_bytes"] for sample in host),
        host_swap_in_bytes=host[-1]["swap_in_bytes"] - host[0]["swap_in_bytes"],
        host_swap_out_bytes=host[-1]["swap_out_bytes"] - host[0]["swap_out_bytes"],
        configured_CPU_KV_bytes=raw["geometry"]["cpu_kv_bytes"],
        limits="NVML is sampled whole-device usage; RSS sums double-count shared mappings; swap is host-wide. CUDA load peaks were reset after warmup and exclude startup allocator peaks.",
    )


def tails(raw):
    classes = collections.defaultdict(list)
    for request in raw["requests"]:
        classes["all"].append(request)
        if request["turn"] == 0:
            classes["initial"].append(request)
        else:
            wait = raw["workload"]["private_wait_ms"][f"{request['index']//4}:{request['turn']-1}"]
            classes["zero_wait_return" if wait == 0 else "positive_wait_return"].append(request)
    return {name: dict(count=len(requests), **{
        f"{statistic}_{metric}_ms": value
        for metric, endpoint in (("ttft", "first_token_ns"), ("e2e", "request_end_ns"))
        for statistic, value in distribution([(r[endpoint] - r["request_start_ns"]) / 1e6 for r in requests]).items()
        if statistic != "count"
    }) for name, requests in classes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    report = read(Path(plan["output"]) / "report.json")
    if not report["all_planned_eligible"] or report["plan_sha256"] != sha(args.plan):
        raise ValueError("Complete, validated planned results required")
    rows = {row["name"]: dict(row) for row in report["rows"]}
    for item in plan["runs"]:
        raw = read(CAMPAIGN / "runs" / item["name"] / "raw.json.gz")
        row = rows[item["name"]]
        if row["raw_sha256"] != sha(row["raw"]):
            raise ValueError("Raw result changed")
        row.update(resources=resources(raw), tails=tails(raw))
    contrasts = []
    for original in report["contrasts"]:
        contrast = dict(original)
        candidate, reference = rows[contrast["candidate"]], rows[contrast["reference"]]
        contrast["tails"] = {group: {key: candidate["tails"][group][key] - reference["tails"][group][key]
                                     for key in candidate["tails"][group] if key != "count"}
                             for group in candidate["tails"]}
        contrasts.append(contrast)
    intervals = []
    if plan["mode"] == "confirm":
        grouped = collections.defaultdict(list)
        for contrast in contrasts:
            grouped[(contrast["arm"], contrast["gpu_blocks"], contrast["reference_gpu_blocks"])].append(contrast)
        for (arm, blocks, baseline), pairs in grouped.items():
            for seed, subset in [("all_fixed_traces", pairs)] + [
                    (str(seed), [p for p in pairs if p["seed"] == seed]) for seed in sorted({p["seed"] for p in pairs})]:
                if len(subset) < 2:
                    continue
                summary = dict(arm=arm, gpu_blocks=blocks, reference_gpu_blocks=baseline, trace=seed)
                for metric in ("tokens_percent", "workflows_percent", "tokens_per_gib_percent"):
                    summary[metric] = interval([p[metric] for p in subset], True)
                summary["tails"] = {group: {key: interval([p["tails"][group][key] for p in subset])
                                             for key in subset[0]["tails"][group]}
                                     for group in subset[0]["tails"]}
                intervals.append(summary)
    result = dict(schema="gpu-cache-frontier-resource-tail-v1", plan_sha256=sha(args.plan),
                  report_sha256=sha(Path(plan["output"]) / "report.json"),
                  analyzer_sha256=sha(__file__), rows=list(rows.values()), contrasts=contrasts,
                  confidence_intervals=intervals,
                  scope="Nominal 95% paired Student-t intervals on whole workflows, conditional on three fixed previously used traces. Per-trace intervals have four repeats; no independent request samples, general workload guarantee, multiplicity adjustment, or inferred equivalence from a nonsignificant difference.")
    write(Path(plan["output"]) / "resources-tails.json", result)
    print("Validated resource/tail rows:", len(rows), "conditional interval groups:", len(intervals))


if __name__ == "__main__":
    main()
