"""Measurement aggregation. Incomplete or inactive tests never silently pass."""

from __future__ import annotations

import math
import statistics
from collections import Counter


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_requests(rows, duration_s):
    good = [r for r in rows if r.get("success") and not r.get("cancel_probe")]
    failures = [r for r in rows if not r.get("success") and not r.get("cancel_probe")]
    ttft = [r["ttft_s"] for r in good if r.get("ttft_s") is not None]
    e2e = [r["latency_s"] for r in good]
    itl = [v for r in good for v in r.get("chunk_intervals_s", [])]
    return dict(
        planned_observed=len(rows),
        completed=len(good),
        failed=len(failures),
        content_checks=sum("content_check_passed" in r for r in good),
        content_checks_failed=sum(r.get("content_check_passed") is False for r in good),
        prompt_tokens=sum(len(r.get("payload", {}).get("prompt", [])) for r in good),
        output_tokens=sum(len(r["output_token_ids"]) for r in good),
        output_tokens_per_s=sum(len(r["output_token_ids"]) for r in good) / duration_s
        if duration_s > 0
        else None,
        completed_requests_per_s=len(good) / duration_s if duration_s > 0 else None,
        client_queue_p95_s=percentile([r.get("client_queue_s", 0) for r in good], 0.95),
        end_user_ttft_p95_s=percentile(
            [
                r["ttft_s"] + r.get("client_queue_s", 0)
                for r in good
                if r.get("ttft_s") is not None
            ],
            0.95,
        ),
        ttft_mean_s=statistics.mean(ttft) if ttft else None,
        ttft_p95_s=percentile(ttft, 0.95),
        e2e_mean_s=statistics.mean(e2e) if e2e else None,
        e2e_p95_s=percentile(e2e, 0.95),
        chunk_interval_p95_s=percentile(itl, 0.95),
        chunk_interval_scope="Client streaming chunk gaps; not individual GPU token timing.",
        finish_reasons=dict(
            Counter(r.get("finish_reason", "unavailable") for r in good)
        ),
        quality_assessment="supplied substring assertions only; exact outputs retained for review"
        if any("content_check_passed" in r for r in good)
        else "not_evaluated; exact outputs retained for review",
    )


def activity(metrics, hints):
    return {
        "scoring": metrics.get("vllm:semantic_relevance_update_batches_total"),
        "store": metrics.get("vllm:kv_offload_store_bytes_total"),
        "restore": metrics.get("vllm:kv_offload_load_bytes_total"),
        "hint_accepted": sum(
            h.get("response", {}).get("status") == "accepted" for h in hints
        ),
        "cpu_recency": sum(
            h["response"].get("ready_cpu_prefix_keys", 0)
            for h in hints
            if h.get("response", {}).get("reason") == "cpu_prefix_recency_refreshed"
        ),
    }


def evaluate_case(case, required):
    errors = []
    if case.get("error"):
        errors.append(case["error"])
    if case.get("summary", {}).get("content_checks_failed"):
        errors.append("content_assertions_failed")
    if case.get("summary", {}).get("failed"):
        errors.append("request_failures")
    if case.get("summary", {}).get("completed", 0) != case.get("expected_requests"):
        errors.append("incomplete_request_coverage")
    if not case.get("cleanup", {}).get("verified"):
        errors.append("cleanup_unverified")
    if not case.get("telemetry", {}).get("samples"):
        errors.append("no_resource_samples")
    if case.get("telemetry", {}).get("gpu_peak_bytes") is None:
        errors.append("gpu_memory_unavailable")
    if case.get("telemetry", {}).get("error"):
        errors.append("resource_sampler_error")
    if case.get("smoke_output_match") is False:
        errors.append("smoke_output_mismatch")
    if any(h.get("error") for h in case.get("hints", [])):
        errors.append("wait_hint_failure")
    if case.get("metrics_error"):
        errors.append("metrics_unavailable")
    checks = case.get("operational", {})
    if checks and not checks.get("passed"):
        errors.append("operational_checks_failed")
    missing = [key for key in required if case.get("activity", {}).get(key) is None]
    inactive = [
        key
        for key in required
        if case.get("activity", {}).get(key) is not None and case["activity"][key] <= 0
    ]
    return dict(
        status="failed"
        if errors
        else "inconclusive"
        if missing or inactive
        else "passed",
        errors=errors,
        unavailable_activity=missing,
        inactive_activity=inactive,
    )


def paired_comparison(cases):
    pairs = []
    grouped = {}
    for c in cases:
        grouped.setdefault(c["repeat"], {})[c["arm"]] = c
    for repeat, arms in sorted(grouped.items()):
        a, b = arms.get("arc"), arms.get("candidate")
        if not a or not b or any(c["evaluation"]["status"] != "passed" for c in (a, b)):
            pairs.append(
                dict(repeat=repeat, eligible=False, reason="missing_or_unqualified_arm")
            )
            continue
        x = a["summary"]["output_tokens_per_s"]
        y = b["summary"]["output_tokens_per_s"]
        left = {str(r["key"]): r for r in a.get("request_fingerprints", [])}
        right = {str(r["key"]): r for r in b.get("request_fingerprints", [])}
        same_inputs = (
            bool(left)
            and left.keys() == right.keys()
            and all(left[k]["prompt"] == right[k]["prompt"] for k in left)
        )
        same_outputs = (
            bool(left)
            and left.keys() == right.keys()
            and all(left[k]["output"] == right[k]["output"] for k in left)
        )
        pairs.append(
            dict(
                same_prompt_tokens=same_inputs,
                same_output_tokens=same_outputs,
                interpretation="Exact recorded prompt/output token agreement"
                if same_inputs and same_outputs
                else "Live workflows differ; rates are not identical-work causal attribution",
                repeat=repeat,
                eligible=True,
                throughput_ratio=y / x if x else None,
                ttft_mean_delta_s=b["summary"]["ttft_mean_s"]
                - a["summary"]["ttft_mean_s"],
                e2e_p95_delta_s=b["summary"]["e2e_p95_s"] - a["summary"]["e2e_p95_s"],
                device_peak_delta_bytes=b["telemetry"]["gpu_peak_bytes"]
                - a["telemetry"]["gpu_peak_bytes"],
                output_tokens_arc=a["summary"]["output_tokens"],
                output_tokens_candidate=b["summary"]["output_tokens"],
            )
        )
    ratios = [
        p["throughput_ratio"]
        for p in pairs
        if p.get("eligible") and p["throughput_ratio"] and p["throughput_ratio"] > 0
    ]
    # Paired log-ratio t interval; independent repetitions, not per-request pseudoreplication.
    interval = None
    if len(ratios) >= 3:
        # Conservative two-sided 95% Student-t critical values, df 1..30.
        criticals = [
            12.7063,
            4.3027,
            3.1825,
            2.7765,
            2.5706,
            2.4469,
            2.3647,
            2.3061,
            2.2622,
            2.2282,
            2.2010,
            2.1789,
            2.1604,
            2.1448,
            2.1315,
            2.1199,
            2.1099,
            2.1009,
            2.0931,
            2.0860,
            2.0797,
            2.0739,
            2.0687,
            2.0639,
            2.0596,
            2.0556,
            2.0519,
            2.0485,
            2.0453,
            2.0423,
        ]
        logs = [math.log(r) for r in ratios]
        mean = statistics.mean(logs)
        margin = (
            criticals[min(len(logs) - 2, 29)]
            * statistics.stdev(logs)
            / math.sqrt(len(logs))
        )
        interval = [math.exp(mean - margin), math.exp(mean + margin)]
    return dict(
        pairs=pairs,
        eligible_pairs=len(ratios),
        geometric_throughput_ratio=math.exp(statistics.mean(map(math.log, ratios)))
        if ratios
        else None,
        throughput_ratio_95ci=interval,
        interval_scope="Paired repetitions conditional on these inputs; not population-wide generalization.",
        decision="inconclusive"
        if len(ratios) != len(pairs) or interval is None
        else "measured_comparison",
        screen_policy="No automatic rejection: inactive, noisy, capped, or unrepresentative tests require targeted follow-up.",
    )


def render_report(result):
    lines = [
        "# Serving suite result",
        "",
        f"Tier: {result['plan']['tier']}. Status: {result['status']}.",
        "",
        "Live model generation with scripted traffic; not a real-user deployment. Output caps and failures are retained.",
        "",
        "| Case | Result | Requests | tok/s | TTFT p95 (s) | Device peak (MiB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def fmt(v):
        return "unavailable" if v is None else f"{v:.3f}"

    for c in result["cases"]:
        s = c.get("summary", {})
        mem = c.get("telemetry", {}).get("gpu_peak_bytes")
        lines.append(
            f"| {c['name']} | {c['evaluation']['status']} | {s.get('completed', 0)}/{c.get('expected_requests', 0)} | {fmt(s.get('output_tokens_per_s'))} | {fmt(s.get('ttft_p95_s'))} | {fmt(mem / 2**20 if mem is not None else None)} |"
        )
        if c["evaluation"]["errors"]:
            lines.append("")  # Details live in JSON, never hide failed cases.
    if result.get("error"):
        lines += ["", "Failure: " + result["error"]]
    lines += [
        "",
        "See result.json for all failures, activity checks, pair intervals, operational results, and provenance.",
        "Per-case requests.jsonl retains prompts, output tokens/text, timing, and termination reasons.",
        "Raw metrics and resource samples are retained; absent metrics are unavailable, never zero.",
        "Byte-identical KV transport and answer quality are not inferred from successful HTTP responses.",
    ]
    return "\n".join(lines) + "\n"
