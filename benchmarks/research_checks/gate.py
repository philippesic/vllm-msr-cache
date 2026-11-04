"""Validate actual CPU gating and preserve unavailable GPU coverage explicitly."""

from collections import Counter, defaultdict

from .cadence import verify_cadence


def verify_gpu_gate(raw, arm, cadence_arm=None):
    expected = arm == "gate"
    cadence = verify_cadence(
        raw, cadence_arm or ("native" if arm == "native" else "periodic")
    )
    issues = []
    rows = []
    replays = []
    cfg = raw.get("config", {}).get("kv_connector_extra_config", {})
    if (
        cfg.get("wait_hint_cpu_retention_gpu_gate", False) is not expected
        or raw.get("args", {}).get("gpu_gated_cpu_retention") is not expected
    ):
        issues.append("gate_configuration_mismatch")
    if any(
        cfg.get(k, False) is not False
        for k in [
            "wait_hint_gpu_prefix_observation",
            "wait_hint_cpu_retention_gpu_aware",
            "wait_hint_gpu_cache_recency",
        ]
    ):
        issues.append("unexpected_other_GPU_policy")
    for h in raw.get("hints", []):
        ack = h.get("response", {})
        if ack.get("reason") != "cpu_prefix_recency_refreshed":
            continue
        if not expected:
            if any(
                k.startswith(("cpu_retention_gpu_", "cpu_retention_skipped_"))
                for k in ack
            ):
                issues.append("control_gate_telemetry")
            continue
        if (
            ack.get("cpu_retention_gpu_gate_policy")
            != "skip_CPU_advice_while_full_GPU_prefix_cached"
        ):
            issues.append("wrong_gate_label")
        n = ack.get("cpu_retention_source_keys")
        before = ack.get("cpu_retention_ready_prefix_keys")
        actual = ack.get("ready_cpu_prefix_keys")
        skipped = ack.get("cpu_retention_skipped_ready_prefix_keys")
        available = ack.get("cpu_retention_gpu_index_available")
        gpu = ack.get("cpu_retention_gpu_cached_prefix_keys")
        skip = ack.get("cpu_retention_skipped_full_gpu")
        if (
            not all(type(v) is int and v >= 0 for v in [n, before, actual, skipped])
            or not 0 <= actual <= before <= n
            or skipped > before
            or n <= 0
        ):
            issues.append("invalid_CPU_counts")
            continue
        if n != h["hint"].get("prefix_boundary_tokens", -1) // 16:
            issues.append("source_count_mismatch")
        if type(available) is not bool or type(skip) is not bool:
            issues.append("invalid_boolean_telemetry")
            continue
        if available:
            if type(gpu) is not int or not 0 <= gpu <= n:
                issues.append("invalid_GPU_prefix_count")
                continue
        elif gpu is not None:
            issues.append("unavailable_GPU_is_not_null")
        if skip != (available and gpu == n):
            issues.append("incorrect_full_GPU_gate")
        if skipped != (before if skip else 0) or (skip and actual != 0):
            issues.append("incorrect_skip_or_actual_touch_count")
        if (
            ack.get("reserved_gpu_blocks") != 0
            or ack.get("speculative_transfer_bytes") != 0
        ):
            issues.append("unexpected_GPU_work")
        x = {
            "event_id": h["hint"].get("event_id"),
            "session_id": h["hint"].get("session_id"),
            "owner_id": h["hint"].get("owner_id"),
            "source_completed_at_ns": h.get("source_completed_at_ns"),
            "started_ns": h["started_ns"],
            "source_keys": n,
            "CPU_ready_before": before,
            "CPU_actual_refreshed": actual,
            "CPU_skipped_ready_keys": skipped,
            "GPU_index_available": available,
            "GPU_cached_prefix_keys": gpu,
            "skipped_full_GPU": skip,
        }
        (replays if ack.get("idempotent_replay") is True else rows).append(x)
    if expected and not rows:
        issues.append("no_fresh_gate_ACKs")
    groups = defaultdict(list)
    for x in sorted(rows, key=lambda x: x["started_ns"]):
        groups[(x["session_id"], x["owner_id"], x["source_completed_at_ns"])].append(x)

    def describe(xs):
        def cat(x):
            return (
                (
                    "unavailable"
                    if not x["GPU_index_available"]
                    else "complete"
                    if x["GPU_cached_prefix_keys"] == x["source_keys"]
                    else "zero"
                    if x["GPU_cached_prefix_keys"] == 0
                    else "partial"
                )
                + "/"
                + (
                    "complete"
                    if x["CPU_ready_before"] == x["source_keys"]
                    else "zero"
                    if x["CPU_ready_before"] == 0
                    else "partial"
                )
            )

        av = [x for x in xs if x["GPU_index_available"]]
        return {
            "observations": len(xs),
            "GPU_available": sum(x["GPU_index_available"] for x in xs),
            "GPU_unavailable": sum(not x["GPU_index_available"] for x in xs),
            "GPU_CPU_ready_before_crosstab": dict(Counter(cat(x) for x in xs)),
            "full_GPU_skips": sum(x["skipped_full_GPU"] for x in xs),
            "full_GPU_skips_with_positive_CPU_ready": sum(
                x["skipped_full_GPU"] and x["CPU_ready_before"] > 0 for x in xs
            ),
            "CPU_ready_key_observations": sum(x["CPU_ready_before"] for x in xs)
            if xs
            else None,
            "CPU_skipped_ready_key_observations": sum(
                x["CPU_skipped_ready_keys"] for x in xs
            )
            if xs
            else None,
            "CPU_actual_refresh_operations": sum(x["CPU_actual_refreshed"] for x in xs)
            if xs
            else None,
            "GPU_cached_key_observations_available_subset": sum(
                x["GPU_cached_prefix_keys"] for x in av
            )
            if av
            else None,
        }

    waits = []
    for key, xs in groups.items():
        declines = []
        max_prior_skipped_ready = None
        max_prior_skipped = None
        for x in xs:
            if (
                max_prior_skipped_ready is not None
                and x["CPU_ready_before"] < max_prior_skipped_ready
            ):
                declines.append(
                    {
                        "earlier_full_GPU_skip_event": max_prior_skipped["event_id"],
                        "earlier_CPU_ready": max_prior_skipped_ready,
                        "later_event_id": x["event_id"],
                        "later_CPU_ready": x["CPU_ready_before"],
                        "later_GPU_still_full": x["skipped_full_GPU"],
                        "later_CPU_actual_refresh": x["CPU_actual_refreshed"],
                    }
                )
            if x["skipped_full_GPU"] and (
                max_prior_skipped_ready is None
                or x["CPU_ready_before"] > max_prior_skipped_ready
            ):
                max_prior_skipped_ready = x["CPU_ready_before"]
                max_prior_skipped = x
        waits.append(
            {
                "key": list(key),
                "snapshots": len(xs),
                "first": xs[0],
                "last": xs[-1],
                "CPU_ready_prefix_declines_after_full_GPU_skip": declines,
            }
        )
    return {
        "qualified": cadence["qualified"] and not issues,
        "issues": issues,
        "cadence": cadence,
        "expected_gate_policy": expected,
        "replayed_ACKs_excluded": len(replays) if expected else None,
        "hint_response_reasons": dict(
            Counter(
                h.get("response", {}).get("reason", "missing_response")
                for h in raw.get("hints", [])
            )
        ),
        "per_hint": describe(rows),
        "per_wait_first": describe([x[0] for x in groups.values()]),
        "per_wait_last": describe([x[-1] for x in groups.values()]),
        "waits_with_CPU_ready_prefix_decrease_after_skip": sum(
            bool(x["CPU_ready_prefix_declines_after_full_GPU_skip"]) for x in waits
        )
        if expected
        else None,
        "waits_with_CPU_ready_decrease_while_GPU_still_full": sum(
            any(
                z["later_GPU_still_full"]
                for z in x["CPU_ready_prefix_declines_after_full_GPU_skip"]
            )
            for x in waits
        )
        if expected
        else None,
        "wait_groups": waits,
        "fresh_gate_rows": rows,
        "scope": "Ready-prefix snapshots, not complete CPU cache membership or causal evidence that gating evicted data. Repeated key observations are not unique keys. Unavailable GPU index remainsNone. Actual refresh is separate from ready-before and skipped ready counts; cached replay excluded.",
    }


def verify_gate_once(raw, arm, campaign):
    import hashlib
    from pathlib import Path

    H = Path(campaign)
    expected_once = arm == "once_positive"
    candidate = arm != "native"
    out = verify_gpu_gate(
        raw,
        "gate" if candidate else "native",
        cadence_arm=arm if expected_once else None,
    )
    issues = out["issues"]
    args = raw.get("args", {})
    cfg = raw.get("config", {}).get("kv_connector_extra_config", {})
    if args.get("stop_after_first_cpu_touch") is not expected_once:
        issues.append("once_flag_mismatch")
    if candidate and (
        cfg.get("preservation_audit") is not False or args.get("no_audit") is not True
    ):
        issues.append("bounded_candidate_mode_mismatch")
    selected = H / (
        "live_once_service_workload_nov3.py"
        if expected_once
        else "live_service_workload_nov3.py"
    )
    archive = H / "runs" / raw["name"] / "workload.py"
    expected_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
    actual_hash = (
        hashlib.sha256(archive.read_bytes()).hexdigest() if archive.is_file() else None
    )
    out["selected_workload_archive"] = {
        "source": str(selected),
        "source_sha256": expected_hash,
        "archive_sha256": actual_hash,
        "matches": actual_hash == expected_hash,
        "raw_pins_caveat": "Driver raw pins list periodic source even for once; this explicit archive check plus frozen manifest supplies selected-workload provenance.",
    }
    if actual_hash != expected_hash:
        issues.append("selected_workload_archive_mismatch")
    expected_policy = (
        "retry_while_pending_until_first_positive_cpu_recency_ack"
        if expected_once
        else None
    )
    if raw.get("workload", {}).get("hint_emission_policy") != expected_policy:
        issues.append("executed_producer_policy_mismatch")
    stops = []
    for h in raw.get("hints", []):
        if not h.get("producer_policy_stop"):
            continue
        ack = h.get("response", {})
        hint = h["hint"]
        if (
            not expected_once
            or ack.get("idempotent_replay")
            or ack.get("cpu_retention_skipped_full_gpu") is not False
            or ack.get("cpu_retention_gpu_index_available") is not True
            or not (
                type(ack.get("ready_cpu_prefix_keys")) is int
                and ack["ready_cpu_prefix_keys"] > 0
            )
        ):
            issues.append("stop_not_fresh_positive_partial_GPU_refresh")
        stops.append(
            {
                "owner_id": hint["owner_id"],
                "session_id": hint["session_id"],
                "observed_prefix_id": hint["observed_prefix_id"],
                "source_completed_at_ns": h["source_completed_at_ns"],
                "event_id": hint["event_id"],
                "CPU_actual_refreshed": ack.get("ready_cpu_prefix_keys"),
                "GPU_cached_prefix_keys": ack.get(
                    "cpu_retention_gpu_cached_prefix_keys"
                ),
                "source_keys": ack.get("cpu_retention_source_keys"),
            }
        )
    out["actual_positive_stop_receipts"] = stops
    out["request_contract_issues"] = []
    for r in raw.get("requests", []):
        body = r.get("payload", {})
        if (
            body.get("max_tokens")
            != raw["geometry"]["max_model_len"] - len(body.get("prompt", []))
            or body.get("ignore_eos", False) is not False
            or body.get("stop")
            or body.get("stop_token_ids")
        ):
            out["request_contract_issues"].append(r["index"])
    if out["request_contract_issues"]:
        issues.append("remaining_context_or_EOS_contract_mismatch")
    out["qualified"] = out["cadence"]["qualified"] and not issues
    return out
