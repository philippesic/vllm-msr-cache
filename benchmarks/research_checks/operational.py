"""Independent raw operational validation; descriptive statistics, no new CIs."""

import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def p95(xs):
    xs = sorted(xs)
    v = 0.95 * (len(xs) - 1)
    i = int(v)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (v - i)


def group(qs):
    t = [(q["first_token_ns"] - q["request_start_ns"]) / 1e6 for q in qs]
    z = [(q["request_end_ns"] - q["request_start_ns"]) / 1e6 for q in qs]
    return (
        dict(
            count=len(qs),
            mean_ttft_ms=statistics.mean(t),
            p95_ttft_ms=p95(t),
            max_ttft_ms=max(t),
            mean_e2e_ms=statistics.mean(z),
            p95_e2e_ms=p95(z),
            max_e2e_ms=max(z),
        )
        if qs
        else None
    )


def metric(text, name, label=None):
    xs = [
        float(s.rsplit(" ", 1)[1])
        for s in text.splitlines()
        if (s.startswith(name + "{") or s.startswith(name + " "))
        and (label is None or label in s)
    ]
    return sum(xs) if xs else None


def analyze(item, campaign, root=None):
    R = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    H = Path(campaign)
    p = H / "runs" / item["name"] / "raw.json.gz"
    raw = json.load(gzip.open(p, "rt"))
    assert (
        raw["status"] == "complete"
        and not raw["performance_eligible"]
        and raw["operational_qualification"]
    )
    assert (
        len(raw["requests"])
        == raw["planned_requests"]
        == item["normal_natural_eos_planned"]
    )
    assert not raw["owned_descendants_remaining"]
    origin = raw["load_start_ns"]
    contract = raw["operational_contract"]
    a = raw["args"]
    gate = item["arm"] == "candidate"
    assert (
        contract["gateway_origin_ns"] == origin
        and not contract["between_wave_cache_reset"]
        and not contract["between_wave_counter_reset"]
    )
    assert a["gpu_gated_cpu_retention"] == gate and a["return_priority_ms"] == 8000
    for label, original in [
        ("driver.py", "run_live_operational_v2_nov3.py"),
        ("workload.py", "live_operational_workload_v2_nov3.py"),
        ("hooks.py", "hooks-operational-v2/sitecustomize.py"),
        ("wave_driver.py", "operational_waves_v2_nov3.py"),
        ("scalar_telemetry.py", "operational_state_v2_nov3.py"),
        ("abort_observer.py", "operational_abort_v2_nov3.py"),
        ("cancel_probe.py", "operational_cancel_probe_v2_nov3.py"),
        ("failure_capture.py", "operational_failure_capture_v2_nov3.py"),
    ]:
        assert sha(p.parent / label) == sha(H / original)
    for key, pin in raw["source_pins"].items():
        assert sha(R / a["source_dir"] / key) == pin
    for key, pin in raw["runtime_pins"].items():
        assert sha(H / key) == pin
    namespace = raw["namespace"]
    ids = set()
    sessions = set()
    returns = contiguous = attempts = capacity = 0
    prefixes = []
    acks = []
    sidcounts = []
    wave_rows = []
    initial = []
    zero = []
    positive = []
    exports = []
    for wave in raw["waves"]:
        wi = wave["wave_index"]
        qs = wave["requests"]
        assert len(qs) == 72 and wave["status"] == "complete"
        local = {q["index"]: q for q in qs}
        sids = {q["session_id"] for q in qs}
        assert len(sids) == 24 and not sessions & sids
        sessions |= sids
        assert wave["application_namespace"] == raw["name"] + "-wave-" + str(wi)
        cc = collections.Counter(e["event"] for e in wave["http_connection_events"])
        n = 72 + len(wave["hints"])
        attempts += n
        assert cc == collections.Counter(
            dict.fromkeys(
                [
                    "request_start",
                    "connection_create_start",
                    "connection_create_end",
                    "request_headers_received",
                ],
                n,
            )
        )
        assert all(
            e["status"] == 200
            for e in wave["http_connection_events"]
            if e["event"] == "request_headers_received"
        )
        hc = wave["http_client_configuration"]
        assert (
            hc["force_close"]
            and hc["automatic_POST_retry"] is False
            and hc["application_retries"] == 0
        )
        for q in qs:
            b = q["payload"]
            out = q["output_token_ids"]
            sid = q["session_id"]
            rid = b["request_id"]
            assert rid not in ids
            ids.add(rid)
            assert (
                q["success"]
                and q["natural_eos"]
                and q["http_status"] == 200
                and q["done_seen"]
                and q["terminal_complete"]
                and q["finish_reasons"] == ["stop"]
            )
            assert (
                b["max_tokens"] + len(b["prompt"]) == 7680
                and 0 < len(out) < b["max_tokens"]
            )
            assert (
                not b.get("ignore_eos", False)
                and not b.get("stop")
                and not b.get("stop_token_ids")
            )
            assert (
                q["priority_origin_ns"] == origin
                and q["priority_clock_ns"] == q["request_start_ns"]
                and q["priority_credit_ms"] == 8000
            )
            assert b["priority"] == (q["priority_clock_ns"] - origin) // 1000000 - (
                8000 if q["turn"] else 0
            )
            assert (
                len(q["token_receipt_ns"]) == len(out)
                and q["token_receipt_ns"][0] == q["first_token_ns"]
            )
            assert (
                hashlib.sha256(
                    json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                == q["output_token_digest"]
            )
            if q["turn"]:
                prev = local[q["index"] - 1]
                assert prev["session_id"] == sid
                hist = prev["payload"]["prompt"] + prev["output_token_ids"][:-1]
                boundary = len(hist) // 16 * 16
                returns += 1
                contiguous += b["prompt"][:boundary] == hist[:boundary]
                (
                    zero
                    if wave["workload"]["private_wait_ms"][
                        f"{q['index'] // 4}:{q['turn'] - 1}"
                    ]
                    == 0
                    else positive
                ).append(q)
            else:
                initial.append(q)
        for hint in wave["hints"]:
            source = hint["hint"]
            ack = hint.get("response", {})
            capacity += ack.get("reason") == "session_registry_capacity"
            assert (
                source["run_id"] == wave["application_namespace"]
                and source["session_id"] in sids
            )
            idx = int(source["owner_id"].rsplit("sample-", 1)[1])
            prev = local[idx - 1]
            nxt = local[idx]
            hist = prev["payload"]["prompt"] + prev["output_token_ids"][:-1]
            tokens = hist[: len(hist) // 16 * 16]
            assert (
                source["prefix_boundary_tokens"] == len(tokens)
                and prev["request_end_ns"] == hint["source_completed_at_ns"]
            )
            expected = hashlib.sha256(
                json.dumps(
                    {"namespace": namespace, "token_ids": tokens},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            assert (
                source["observed_prefix_id"] == expected
                and prev["request_end_ns"]
                <= hint["started_ns"]
                <= nxt["request_start_ns"]
            )
            if ack.get("reason") == "cpu_prefix_recency_refreshed":
                assert (
                    gate
                    and ack["cpu_retention_gpu_gate_policy"]
                    == "skip_CPU_advice_while_full_GPU_prefix_cached"
                )
                assert ack["cpu_retention_source_keys"] == len(tokens) // 16
                assert (
                    0
                    <= ack["ready_cpu_prefix_keys"]
                    <= ack["cpu_retention_ready_prefix_keys"]
                    <= len(tokens) // 16
                )
                full = (
                    ack["cpu_retention_gpu_index_available"]
                    and ack["cpu_retention_gpu_cached_prefix_keys"] == len(tokens) // 16
                )
                assert ack["cpu_retention_skipped_full_gpu"] == full
                assert ack["cpu_retention_skipped_ready_prefix_keys"] == (
                    ack["cpu_retention_ready_prefix_keys"] if full else 0
                )
                if full:
                    assert ack["ready_cpu_prefix_keys"] == 0
                assert (
                    ack["reserved_gpu_blocks"] == ack["speculative_transfer_bytes"] == 0
                )
        seconds = (wave["load_end_ns"] - wave["load_start_ns"]) / 1e9
        nt = sum(len(q["output_token_ids"]) for q in qs)
        wave_rows.append(
            {
                "wave": wi,
                "seed": wave["seed"],
                "load_s": seconds,
                "tokens": nt,
                "tokens_s": nt / seconds,
                **group(qs),
            }
        )
        exports += [
            wave[k]
            for k in ["immediate_export", "settled_export"]
            if isinstance(wave.get(k), dict)
        ]
    exports += [
        raw[k]
        for k in ["overlap_settled_export", "postload_export"]
        if isinstance(raw.get(k), dict)
    ]
    for ex in exports:
        sc = ex["scheduler"]["operational_scalar_state"]
        assert (
            sc["native_request_count"]
            == sc["native_running_count"]
            == sc["native_waiting_count"]
            == 0
        )
        if gate:
            s = sc["scheduler_lengths"]
            m = sc["manager_lengths"]
            w = ex["workers"][0]["operational_worker_state"]["worker_lengths"]
            assert (
                0 <= s["_observed_prefix_generations"] <= a["registry_limit"]
                and 0 <= s["_observed_prefixes"] <= 1024
                and 0 <= s["_wait_hint_results"] <= 4096
            )
            assert all(
                v == 0
                for k, v in s.items()
                if k
                not in (
                    "_observed_prefix_generations",
                    "_observed_prefixes",
                    "_wait_hint_results",
                )
            )
            assert m["_score_residencies"] <= 2048 and all(
                v == 0 for k, v in m.items() if k != "_score_residencies"
            )
            assert all(v == 0 for v in w.values())
            prefixes.append(s["_observed_prefixes"])
            acks.append(s["_wait_hint_results"])
            sidcounts.append(s["_observed_prefix_generations"])
    probe = raw.get("cancellation_probe")
    health = []
    if item["deliberate_abort_planned"]:
        assert probe["qualified"]
        abort = probe["abort_request"]
        health = probe["health_requests"]
        assert (
            abort["disconnect_ns"]
            and not abort["terminal_before_close"]
            and len(health) == 4
        )
        records = probe["settled_after_health_export"]["scheduler"][
            "operational_abort_transitions"
        ]
        assert records["count"] == 1 and len(records["records"]) == 1
        record = records["records"][0]
        assert (
            record["status"] == "FINISHED_ABORTED"
            and record["external_mapping_available"]
            and record["external_request_id"] == abort["payload"]["request_id"]
        )
        for q in [abort, *health]:
            assert (
                q["priority_origin_ns"] == origin
                and q["priority_credit_ms"] == 8000
                and q["priority_turn"] == 0
            )
            assert (
                q["payload"]["priority"] == (q["priority_clock_ns"] - origin) // 1000000
                and q["request_start_ns"] >= q["priority_clock_ns"]
            )
        assert all(
            q["http_status"] == 200
            and q["natural_eos"]
            and q["terminal_complete"]
            and q["finish_reasons"] == ["stop"]
            for q in health
        )
    else:
        assert probe is None
    deltas = {}
    for key, (name, label) in {
        "H2D_bytes": (
            "vllm:kv_offload_total_bytes_total",
            'transfer_type="CPU_to_GPU"',
        ),
        "D2H_bytes": (
            "vllm:kv_offload_total_bytes_total",
            'transfer_type="GPU_to_CPU"',
        ),
        "preemptions": ("vllm:num_preemptions_total", None),
    }.items():
        b = metric(raw["metrics_initial"], name, label)
        f = metric(raw["metrics_final"], name, label)
        deltas[key] = f - b if b is not None and f is not None else None
    gpu = [
        s["gpu"]["memory_used_bytes"] / 2**20
        for s in raw["resource_samples"]
        if "gpu" in s
    ]
    if a["wave_release_interval_s"]:
        for wave in raw["waves"]:
            intended = origin + wave["wave_index"] * 20000000000
            assert (
                wave["intended_wave_release_ns"] == intended
                and wave["actual_wave_start_ns"] >= intended
                and wave["immediate_export"] is None
            )
    if a["registry_limit"] == 128:
        assert 128 in sidcounts and capacity > 0
    return {
        "name": raw["name"],
        "phase": item["phase"],
        "arm": item["arm"],
        "raw_sha256": sha(p),
        "natural_eos": len(raw["requests"]),
        "health_eos": len(health),
        "qualified_actual_aborts": int(bool(probe)),
        "distinct_session_ids": len(sessions),
        "returns": returns,
        "actual_contiguous_returns": contiguous,
        "normal_HTTP_attempts": attempts,
        "native_idle_and_candidate_pending_cleared_at_all_exports": True,
        "session_counts": sidcounts if gate else None,
        "prefix_counts": prefixes if gate else None,
        "ACK_counts": acks if gate else None,
        "capacity_no_action_ACKs": capacity,
        "counter_deltas": deltas,
        "NVML_mib_range": [min(gpu), max(gpu)],
        "groups": {
            "all": group(raw["requests"]),
            "initial": group(initial),
            "zero_wait_return": group(zero),
            "positive_wait_return": group(positive),
        },
        "generation_wave_load_s_sum": sum(w["load_s"] for w in wave_rows),
        "output_tokens": sum(w["tokens"] for w in wave_rows),
        "normal_wave_span_s": (
            max(w["load_end_ns"] for w in raw["waves"])
            - min(w["load_start_ns"] for w in raw["waves"])
        )
        / 1e9,
        "wave_rows": wave_rows,
        "overlap": bool(a["wave_release_interval_s"]),
        "load_start_ns": origin,
        "load_end_ns": raw["load_end_ns"],
        "first_prompts_by_wave": [
            {q["index"]: q["payload"]["prompt"] for q in w["requests"] if not q["turn"]}
            for w in raw["waves"]
        ],
    }
