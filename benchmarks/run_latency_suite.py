# SPDX-License-Identifier: Apache-2.0
"""Phase 1 latency/throughput suite (plan's Benchmarking section, Step
1.6). For each (policy, workload, request_rate) combination: launch a real
server with that policy, run the workload (`vllm bench serve` for
chat/rag/longdoc, the bespoke needle_workload for `needle`, concurrent
chat/rag/longdoc + needle sub-streams for `mixed` -- the plan's "headline
workload"), snapshot the offload-subsystem Prometheus counters before/
after, tear the server down, and append one combined result row (or, for
`mixed`, one row per sub-stream) to a CSV.

This deliberately wraps `vllm bench serve` rather than reimplementing a
load generator -- it already does open-loop Poisson arrivals, TTFT/TPOT/
ITL percentiles, and (via `prefix_repetition`) the `rag` workload's
shared-prefix structure, per the plan's own "wrap, don't rebuild" guidance.

Usage (small-dev-model validation run, matching this project's real
launch configs throughout the issues log):

  python benchmarks/run_latency_suite.py \\
      --model Qwen/Qwen2.5-1.5B-Instruct \\
      --policies lru,semantic-minmax \\
      --workloads chat \\
      --request-rates 2.0 \\
      --num-prompts 20 \\
      --scale 0.05 \\
      --cpu-bytes-to-use 268435456 \\
      --output-dir /tmp/latency_suite_smoke
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.cell_lifecycle import (
    cell_marker_path,
    cell_result_path,
    count_csv_rows,
    process_birth_identity,
    read_cell_marker,
    termination_details,
    update_cell_marker,
    write_cell_marker,
)
from benchmarks.csv_utils import CSV_FIELD_SIZE_LIMIT  # noqa: F401
from benchmarks.latency_validation import is_needle_row, validate_latency_row
from benchmarks.prefetch_attribution import (
    PREFETCH_ATTRIBUTION_SCHEMA,
    PrefetchAttributionError,
    validate_prefetch_episode,
)
from benchmarks.preservation import (
    AUTHORITATIVE_METADATA_MISSING_REASON,
    NEEDLE_ISOLATED_COUNTER_REASON,
    NEEDLE_METRICS_UNAVAILABLE_REASON,
    REQUEST_OUTCOME_FIELDS,
    is_canonical_request_id_prefix,
    map_preservation_events,
)
from harness import metrics as metrics_mod
from harness import needle_workload
from harness import policies as policies_mod
from harness import workloads as workloads_mod
from harness.server import launch_server, resolve_vllm_cli
from semantic_offload.decision_trace import (
    TRACE_MARKER,
    validate_decision_feature_record,
    validate_decision_feature_records,
)

RESULT_FIELDS = [
    "policy",
    "seed",
    "workload",
    "sub_workload",
    "request_rate",
    "parent_request_rate",
    "reference_count",
    "sample_index",
    "run_id",
    "cell_id",
    "request_id_prefix",
    "request_ids_json",
    "request_timings_json",
    "preservation_events_json",
    "preservation_event_count",
    "preservation_join_status",
    "decision_feature_trace_json",
    "decision_feature_trace_count",
    "decision_feature_trace_join_status",
    "preservation_outcome",
    "preservation_reason",
    "preservation_request_id",
    "preservation_request_id_supplied",
    "prefetch_attribution_json",
    "prefetch_attribution_count",
    "prefetch_attribution_join_status",
    # Chat/RAG request-event projection. These fields are populated only by
    # a complete preservation-event join; missing values remain unavailable,
    # and no field is inferred from workload-level load/store bytes.
    *REQUEST_OUTCOME_FIELDS,
    "num_prompts",
    "duration_s",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "ttft_p95_ms",
    "ttft_max_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "itl_p99_ms",
    "itl_p95_ms",
    "itl_max_ms",
    "throughput_tok_s",
    "needle_hit_rate",
    # needle-v2's preservation signal (see needle_workload.classify_needle_
    # outcome): the recall's isolated CPU-tier interaction, not a string
    # match. `needle_outcome` in {hit, partial, miss, not_pressured}; the two
    # byte columns are the recall-only counter deltas it was derived from.
    "needle_outcome",
    "recall_load_bytes",
    "recall_store_bytes",
    "load_bytes_delta",
    "store_bytes_delta",
    "preemptions_delta",
    "time_to_first_token_sum_s",
    "time_to_first_token_count",
    "inter_token_latency_sum_s",
    "inter_token_latency_count",
    "request_decode_time_sum_s",
    "request_decode_time_count",
    "request_prefill_kv_computed_tokens_sum",
    "request_prefill_kv_computed_tokens_count",
    "prefix_cache_queries_delta",
    "prefix_cache_hits_delta",
    "external_prefix_cache_queries_delta",
    "external_prefix_cache_hits_delta",
    # Monotonic vLLM request-histogram sum/count deltas for the same interval
    # as the offload counters. These separate queueing, model prefill, and
    # running/inference time from the end-to-end request total without
    # claiming per-request attribution.
    "request_e2e_time_sum_s",
    "request_e2e_time_count",
    "request_queue_time_sum_s",
    "request_queue_time_count",
    "request_prefill_time_sum_s",
    "request_prefill_time_count",
    "request_inference_time_sum_s",
    "request_inference_time_count",
    "error",
]


def check_existing_schema(csv_path: str) -> None:
    """A stale results.csv from an older RESULT_FIELDS schema (e.g. before
    needle-v2 added needle_outcome/recall_load_bytes/recall_store_bytes)
    silently appending under a shorter existing header produced rows wider
    than the file's own header line -- undetected until a later grid-sweep
    merge crashed reading it back. Fail fast on append instead: call this
    before appending to an existing results.csv, never on a fresh one."""
    with open(csv_path, newline="") as existing:
        existing_header = next(csv.reader(existing), [])
    if existing_header != RESULT_FIELDS:
        raise ValueError(
            f"{csv_path} already exists with a different schema than the "
            f"current RESULT_FIELDS -- refusing to append (would produce a "
            f"corrupt/unmergeable CSV). Existing header: {existing_header}. "
            f"Current RESULT_FIELDS: {RESULT_FIELDS}. Move or remove the "
            f"stale file, or use a fresh --output-dir."
        )


# `mixed`: the plan's "headline workload" (§Benchmarking: "all four
# interleaved") -- proportions chosen so chat/rag dominate (the realistic
# high-arrival-rate and shared-prefix-reuse cases) with longdoc present at
# lower weight (its own arrival is rarer and each request is much larger).
_MIXED_SUBWORKLOAD_WEIGHTS = {"chat": 0.40, "rag": 0.35, "longdoc": 0.25}


def _seed_tag(seed: int | None) -> str:
    """Filesystem-safe tag for raw-result filenames -- without this, two
    cells sharing a policy/workload/rate but different seeds (the plan's
    own >=3-seed protocol, not optional) silently overwrite each other's
    raw JSON, and concurrently-running cells (multi-GPU dispatch) could
    race on the same file mid-write."""
    return f"seed{seed}" if seed is not None else "seedNone"


def _request_id_prefix(
    run_id: str, cell_id: str, workload: str, sub_workload: str | None = None
) -> str:
    tag = sub_workload or workload
    return f"sc|{run_id}|{cell_id}|{tag}|sample-"


_CANONICAL_EXTERNAL_ID_RE = re.compile(r"^sc\|([^|]+)\|([^|]+)\|([^|]+)\|sample-(\d+)$")
_CANONICAL_INTERNAL_ID_RE = re.compile(
    r"^cmpl-(sc\|[^|]+\|[^|]+\|[^|]+\|sample-\d+)-\d+-([0-9a-f]{8})$"
)
_NEEDLE_EXTERNAL_ID_RE = re.compile(
    r"^needle-v2--?\d+-ref\d+-sample\d+-(?:needle|recall|probe\d+|distractor\d+)$"
)
_NEEDLE_INTERNAL_ID_RE = re.compile(
    r"^cmpl-(needle-v2--?\d+-ref\d+-sample\d+-(?:needle|recall|probe\d+|distractor\d+))-\d+-([0-9a-f]{8})$"
)
_KNOWN_APISERVER_METRICS_SUFFIX_RE = re.compile(
    r"\s*\(APIServer pid=\d+\) INFO:[ \t]+"
    r"\d{1,3}(?:\.\d{1,3}){3}:\d+ - "
    r'"GET /metrics HTTP/1\.1" 200 OK\s*'
)
_UNAVAILABLE_METADATA_REASONS = (
    ("request_identity_valid", "request_identity_unavailable"),
    ("lookup_metadata_complete", "lookup_metadata_unavailable"),
    ("transfer_metadata_complete", "transfer_metadata_unavailable"),
    ("admission_metadata_complete", "admission_metadata_unavailable"),
    ("physical_block_identity_complete", "physical_block_metadata_unavailable"),
)


def _terminal_success(event: dict) -> bool:
    status = str(event.get("terminal_status", "")).lower()
    return (
        event.get("terminal_reason") == "completed"
        and (
            status in {"finished", "completed", "success"}
            or status.startswith("finished_")
        )
        and not any(token in status for token in ("aborted", "error", "ignored"))
    )


def _explicit_unavailable_reason(event: dict) -> str | None:
    values = {}
    for field, reason in _UNAVAILABLE_METADATA_REASONS:
        value = event.get(field)
        if not isinstance(value, bool):
            return None
        values[field] = value
    tp_complete = event.get("tp_contributors_complete")
    if not isinstance(tp_complete, bool):
        return None
    if tp_complete != values["transfer_metadata_complete"]:
        return None
    for field, reason in _UNAVAILABLE_METADATA_REASONS:
        if values[field] is False:
            return reason
    return None


def _metric_delta_fields(delta: dict[str, float] | None) -> dict[str, float | None]:
    """Map one metrics interval onto explicit CSV fields.

    The request-time values are monotonic histogram sum/count deltas from
    vLLM's existing Prometheus endpoint. They describe the interval as a
    whole, not per-request spans; keeping sums and counts separate avoids
    inventing a percentile or pretending that a mixed interval is exact
    request attribution.
    """
    if delta is None:
        return {
            "load_bytes_delta": None,
            "store_bytes_delta": None,
            "preemptions_delta": None,
            "time_to_first_token_sum_s": None,
            "time_to_first_token_count": None,
            "inter_token_latency_sum_s": None,
            "inter_token_latency_count": None,
            "request_decode_time_sum_s": None,
            "request_decode_time_count": None,
            "request_prefill_kv_computed_tokens_sum": None,
            "request_prefill_kv_computed_tokens_count": None,
            "prefix_cache_queries_delta": None,
            "prefix_cache_hits_delta": None,
            "external_prefix_cache_queries_delta": None,
            "external_prefix_cache_hits_delta": None,
            "request_e2e_time_sum_s": None,
            "request_e2e_time_count": None,
            "request_queue_time_sum_s": None,
            "request_queue_time_count": None,
            "request_prefill_time_sum_s": None,
            "request_prefill_time_count": None,
            "request_inference_time_sum_s": None,
            "request_inference_time_count": None,
        }
    return {
        "load_bytes_delta": delta.get("vllm:kv_offload_load_bytes_total"),
        "store_bytes_delta": delta.get("vllm:kv_offload_store_bytes_total"),
        "preemptions_delta": delta.get("vllm:num_preemptions_total"),
        "time_to_first_token_sum_s": delta.get("vllm:time_to_first_token_seconds_sum"),
        "time_to_first_token_count": delta.get(
            "vllm:time_to_first_token_seconds_count"
        ),
        "inter_token_latency_sum_s": delta.get("vllm:inter_token_latency_seconds_sum"),
        "inter_token_latency_count": delta.get(
            "vllm:inter_token_latency_seconds_count"
        ),
        "request_decode_time_sum_s": delta.get("vllm:request_decode_time_seconds_sum"),
        "request_decode_time_count": delta.get(
            "vllm:request_decode_time_seconds_count"
        ),
        "request_prefill_kv_computed_tokens_sum": delta.get(
            "vllm:request_prefill_kv_computed_tokens_sum"
        ),
        "request_prefill_kv_computed_tokens_count": delta.get(
            "vllm:request_prefill_kv_computed_tokens_count"
        ),
        "prefix_cache_queries_delta": delta.get("vllm:prefix_cache_queries_total"),
        "prefix_cache_hits_delta": delta.get("vllm:prefix_cache_hits_total"),
        "external_prefix_cache_queries_delta": delta.get(
            "vllm:external_prefix_cache_queries_total"
        ),
        "external_prefix_cache_hits_delta": delta.get(
            "vllm:external_prefix_cache_hits_total"
        ),
        "request_e2e_time_sum_s": delta.get("vllm:e2e_request_latency_seconds_sum"),
        "request_e2e_time_count": delta.get("vllm:e2e_request_latency_seconds_count"),
        "request_queue_time_sum_s": delta.get("vllm:request_queue_time_seconds_sum"),
        "request_queue_time_count": delta.get("vllm:request_queue_time_seconds_count"),
        "request_prefill_time_sum_s": delta.get(
            "vllm:request_prefill_time_seconds_sum"
        ),
        "request_prefill_time_count": delta.get(
            "vllm:request_prefill_time_seconds_count"
        ),
        "request_inference_time_sum_s": delta.get(
            "vllm:request_inference_time_seconds_sum"
        ),
        "request_inference_time_count": delta.get(
            "vllm:request_inference_time_seconds_count"
        ),
    }


def _unavailable_preservation_fields(reason: str) -> dict[str, object]:
    return {
        "preservation_outcome": "unavailable",
        "preservation_reason": reason,
        "preservation_request_id": None,
        "preservation_request_id_supplied": "false",
        "preservation_events_json": "[]",
        "preservation_event_count": None,
        "preservation_join_status": "not_requested",
        "decision_feature_trace_json": "[]",
        "decision_feature_trace_count": None,
        "decision_feature_trace_join_status": "not_requested",
        "preservation_outcome_basis": None,
        "preservation_request_ids_json": "[]",
        "preservation_request_outcome_details_json": "[]",
        "preservation_authoritative_request_count": None,
        "preservation_required_tokens": None,
        "preservation_reused_tokens": None,
        "preservation_loaded_tokens": None,
        "preservation_recomputed_tokens": None,
        "preservation_required_blocks": None,
        "preservation_reused_blocks": None,
        "preservation_loaded_blocks": None,
        "preservation_recomputed_blocks": None,
        "prefetch_attribution_json": "[]",
        "prefetch_attribution_count": None,
        "prefetch_attribution_join_status": "not_requested",
    }


def _needle_preservation_fields(result: dict) -> dict[str, object]:
    outcome = result.get("needle_outcome")
    if outcome is None:
        return _unavailable_preservation_fields(NEEDLE_METRICS_UNAVAILABLE_REASON)
    fields = _unavailable_preservation_fields(NEEDLE_ISOLATED_COUNTER_REASON)
    fields.update(
        {
            "preservation_outcome": outcome,
            "preservation_reason": NEEDLE_ISOLATED_COUNTER_REASON,
            "preservation_request_id": result.get("preservation_request_id"),
            "preservation_request_id_supplied": (
                "true"
                if result.get("preservation_request_id_supplied", False)
                else "false"
            ),
        }
    )
    return fields


def _request_timing_rows(data: dict, request_ids: list[str] | None) -> list[dict]:
    """Preserve the benchmark's per-sample IDs and timing arrays."""
    arrays = [
        data.get("ttfts") or [],
        data.get("itls") or [],
        data.get("start_times") or [],
        data.get("errors") or [],
    ]
    count = max((len(values) for values in arrays), default=0)
    actual_request_ids = request_ids or []
    count = min(count, len(actual_request_ids))

    def _sample_index(request_id: str) -> int | None:
        match = re.search(r"(?:sample-)?(\d+)$", request_id)
        return int(match.group(1)) if match else None

    return [
        {
            "request_id": actual_request_ids[index],
            "sample_index": _sample_index(actual_request_ids[index]),
            "ttft_s": arrays[0][index] if index < len(arrays[0]) else None,
            "itl_s": arrays[1][index] if index < len(arrays[1]) else None,
            "start_time": arrays[2][index] if index < len(arrays[2]) else None,
            "error": arrays[3][index] if index < len(arrays[3]) else None,
        }
        for index in range(count)
    ]


def parse_vllm_bench_result(
    result_json_path: str,
    request_id_prefix: str | None = None,
    expected_count: int | None = None,
    require_detailed: bool = False,
) -> dict:
    """Pure, unit-testable: extract the fields we care about from a real
    `vllm bench serve --save-result` JSON file into our flat row shape.

    Includes `completed` (not part of RESULT_FIELDS -- callers must pop it
    before handing the row to the CSV writer) so `run_vllm_bench_serve` can
    detect a zero-completions run: `vllm`'s own `calculate_metrics`
    (`vllm/benchmarks/serve.py`) computes every percentile as
    `np.percentile(ttfts or 0, p)`, which is exactly `0.0` when `ttfts` is
    empty (no request both succeeded and reported a streamed first token).
    Silently returning that 0.0 is indistinguishable from a real,
    fast TTFT -- this is the mixed workload's rag/longdoc sub-streams
    reporting `ttft_p50=0.0` for every policy (issues log entry #76): under
    the concurrent chat+rag+longdoc+needle load `mixed` puts on one server,
    those two much-larger-prompt sub-streams are the ones most likely to
    see a degenerate 0-completion run, and the bare percentile number gives
    no way to tell that apart from a genuine (if surprising) fast TTFT."""
    with open(result_json_path) as f:
        data = json.load(f)
    if require_detailed:
        required_arrays = ("ttfts", "itls", "start_times", "errors")
        for field in required_arrays:
            values = data.get(field)
            if not isinstance(values, list) or (
                expected_count is not None and len(values) != expected_count
            ):
                raise ValueError(
                    f"detailed benchmark field {field!r} must contain exactly "
                    f"{expected_count} entries"
                )
    actual_request_ids = data.get("request_ids")
    if require_detailed and actual_request_ids is None:
        # Older `vllm bench serve` serializers accept --request-id-prefix and
        # pass those IDs to the server, but omit request_ids from the saved
        # JSON.  Its detailed arrays retain input order (the benchmark awaits
        # requests with asyncio.gather), so recover the IDs only when the
        # caller supplies the exact prefix used for that invocation.  Do not
        # synthesize IDs for an unbound payload or for partial arrays.
        if expected_count is None:
            raise ValueError(
                "detailed benchmark request_ids missing and no expected count "
                "was supplied for deterministic recovery"
            )
        if not isinstance(request_id_prefix, str) or not request_id_prefix:
            raise ValueError(
                "detailed benchmark request_ids missing and no request_id_prefix "
                "was supplied for deterministic recovery"
            )
        actual_request_ids = [
            f"{request_id_prefix}{index}" for index in range(expected_count)
        ]
    if require_detailed and (
        not isinstance(actual_request_ids, list)
        or not all(isinstance(request_id, str) for request_id in actual_request_ids)
        or expected_count is not None
        and len(actual_request_ids) != expected_count
    ):
        raise ValueError(
            "detailed benchmark request_ids must be an actual string array "
            f"with exactly {expected_count} entries"
        )
    timing_rows = _request_timing_rows(data, actual_request_ids)
    if expected_count is not None and len(timing_rows) != expected_count:
        raise ValueError(
            "detailed benchmark request rows do not match num_prompts: "
            f"expected {expected_count}, got {len(timing_rows)}"
        )

    def _finite_ms(values) -> list[float]:
        if not isinstance(values, list):
            return []
        result = []
        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                result.append(value * 1000.0)
        return result

    ttft_values_ms = _finite_ms(data.get("ttfts"))
    itl_values_ms = [
        value
        for values in (data.get("itls") or [])
        if isinstance(values, list)
        for value in _finite_ms(values)
    ]

    def _tail(values: list[float], fraction: float, fallback):
        if values:
            ordered = sorted(values)
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (
                position - lower
            )
        return fallback

    return {
        "duration_s": data.get("duration"),
        "completed": data.get("completed"),
        "ttft_p50_ms": data.get("p50_ttft_ms"),
        "ttft_p90_ms": data.get("p90_ttft_ms"),
        "ttft_p99_ms": data.get("p99_ttft_ms"),
        "ttft_p95_ms": _tail(ttft_values_ms, 0.95, data.get("p95_ttft_ms")),
        "ttft_max_ms": max(ttft_values_ms, default=data.get("max_ttft_ms")),
        "itl_p50_ms": data.get("p50_itl_ms"),
        "itl_p90_ms": data.get("p90_itl_ms"),
        "itl_p99_ms": data.get("p99_itl_ms"),
        "itl_p95_ms": _tail(itl_values_ms, 0.95, data.get("p95_itl_ms")),
        "itl_max_ms": max(itl_values_ms, default=data.get("max_itl_ms")),
        "throughput_tok_s": data.get("output_throughput"),
        "request_ids_json": json.dumps(
            [row["request_id"] for row in timing_rows], separators=(",", ":")
        ),
        "request_timings_json": json.dumps(
            timing_rows, separators=(",", ":"), sort_keys=True
        ),
    }


def resolve_num_prompts(
    request_rate: float,
    num_prompts: int | None,
    target_duration_s: float | None,
    min_num_prompts: int | None = None,
) -> int:
    """Pure, unit-testable: either an explicit --num-prompts, or a count
    derived from a target steady-state duration at the given request rate
    (--target-duration-s) -- a flat --num-prompts produces very different
    steady-state windows at different arrival rates, which the plan's own
    protocol (>=10 min steady-state per configuration) doesn't allow (see
    the Step 1.6 grid-trim proposal's follow-up #2).

    `min_num_prompts`: some datasets have their own hard floor (e.g. `rag`'s
    `prefix_repetition` needs num_requests >= num_prefixes,
    `workloads.RAG_NUM_PREFIXES`) that a derived count can silently fall
    below at a low rate/short duration -- raise here, before launching
    `vllm bench serve`, instead of letting that cell crash with a
    less legible error deep inside the benchmark subprocess."""
    if target_duration_s is not None:
        if request_rate == float("inf"):
            raise ValueError(
                "--target-duration-s requires a finite --request-rate "
                "(inf sends everything at t=0, duration is undefined)"
            )
        resolved = max(1, round(target_duration_s * request_rate))
    elif num_prompts is not None:
        resolved = num_prompts
    else:
        raise ValueError("either --num-prompts or --target-duration-s is required")
    if min_num_prompts is not None and resolved < min_num_prompts:
        raise ValueError(
            f"resolved num_prompts ({resolved}) is below this workload's "
            f"minimum ({min_num_prompts}) at request_rate={request_rate}, "
            f"num_prompts={num_prompts}, target_duration_s={target_duration_s} "
            "-- raise --target-duration-s or --num-prompts for this cell"
        )
    return resolved


def resolve_needle_num_distractors(
    needle_num_prompts: int, explicit: int | None
) -> int:
    """Pure, unit-testable: the needle-case distractor count, held CONSTANT
    across every `reference_count` in a run.

    Previously this was computed inline as `max(1, needle_num_prompts -
    reference_count)`, which silently confounds "was the needle referenced"
    with "how much total distractor traffic ran before recall" -- higher
    reference_count meant fewer distractors, by construction, since probes
    and distractors shared one fixed prompt budget. Issues log entry #77
    caught this via a clean negative control: `arc`, which has zero
    content-awareness, showed the SAME hit-at-reference_count>=1 pattern
    semantic did, which is only possible if less traffic (not "was
    referenced") is what's actually driving the result. Decoupling the two
    (this function never takes `reference_count` as an input) removes the
    confound structurally rather than by convention."""
    return explicit if explicit is not None else needle_num_prompts


def run_vllm_bench_serve(
    base_url: str,
    model: str,
    workload: str,
    num_prompts: int,
    request_rate: float,
    scale: float,
    result_path: str,
    seed: int | None = None,
    request_id_prefix: str | None = None,
    save_detailed: bool = False,
) -> dict:
    args = workloads_mod.get_args(
        workload,
        num_prompts=num_prompts,
        request_rate=request_rate,
        scale=scale,
        request_id_prefix=request_id_prefix,
        save_detailed=save_detailed,
    )
    cmd = [
        resolve_vllm_cli(),
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--model",
        model,
        "--result-filename",
        os.path.basename(result_path),
        "--result-dir",
        os.path.dirname(result_path),
        *args,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "error": (
                "vllm bench serve timed out after 1800s -- likely a stalled "
                "request (see vLLM issue #45388)"
            )
        }
    if proc.returncode != 0:
        return {"error": f"vllm bench serve failed: {proc.stderr[-2000:]}"}
    if not os.path.exists(result_path):
        return {"error": f"vllm bench serve produced no result file at {result_path}"}
    row = parse_vllm_bench_result(
        result_path,
        request_id_prefix=request_id_prefix,
        expected_count=num_prompts if save_detailed else None,
        require_detailed=save_detailed,
    )
    completed = row.pop("completed", None)
    if not completed:
        return {
            "error": (
                f"vllm bench serve completed {completed or 0}/{num_prompts} "
                "requests -- reported ttft/itl/throughput would be "
                "degenerate zeros, not a real measurement (vllm's own "
                "calculate_metrics reports 0.0 for every percentile when no "
                "request both succeeded and streamed a first token); see "
                f"{result_path}'s 'errors' field for why requests failed"
            )
        }
    # The request-time attribution fields are added by the caller after its
    # before/after Prometheus snapshots. Validate the benchmark JSON's
    # percentile contract here; the completed row is validated again after
    # attribution fields are attached.
    percentile_errors = validate_latency_row(row, require_percentiles=True)
    if percentile_errors:
        return {
            "error": (
                "vllm bench serve produced invalid latency metrics: "
                + "; ".join(percentile_errors)
            )
        }
    return row


def _read_preservation_events(
    log_path: str,
    request_id_prefix: str,
    expected_request_ids: set[str],
    run_id: str,
    cell_id: str,
    start_offset: int = 0,
    allowed_request_prefixes: tuple[str, ...] = (),
) -> list[dict]:
    """Read and strictly join one benchmark invocation's request events."""
    prefix = "SEMANTIC_PRESERVATION_EVENT "

    def _event_lines(log_file):
        decoder = json.JSONDecoder()
        for raw_line in log_file:
            marker_positions = [
                match.start() for match in re.finditer(re.escape(prefix), raw_line)
            ]
            if not marker_positions:
                continue
            original_prefix = raw_line[: marker_positions[0]]
            if original_prefix and not re.fullmatch(
                r"\s*\(EngineCore pid=\d+\)\s*", original_prefix
            ):
                raise ValueError(f"unknown preservation log prefix {original_prefix!r}")
            for index, marker_index in enumerate(marker_positions):
                end = (
                    marker_positions[index + 1]
                    if index + 1 < len(marker_positions)
                    else len(raw_line)
                )
                payload = raw_line[marker_index + len(prefix) : end].strip()
                if not payload:
                    raise ValueError("malformed preservation event: empty JSON segment")
                try:
                    event, next_offset = decoder.raw_decode(payload, 0)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed preservation event: {exc}") from exc
                if not isinstance(event, dict):
                    raise TypeError(
                        "malformed preservation event: expected JSON object"
                    )
                trailing = payload[next_offset:]
                if (
                    trailing.strip()
                    and not _KNOWN_APISERVER_METRICS_SUFFIX_RE.fullmatch(trailing)
                ):
                    raise ValueError(
                        "malformed preservation event: non-whitespace text "
                        "after JSON object"
                    )
                yield (
                    original_prefix
                    + prefix
                    + json.dumps(event, separators=(",", ":"))
                    + "\n"
                )

    def _canonicalize_event_identity(event: dict) -> dict:
        external_id = event.get("external_request_id")
        if external_id is None:
            request_id = event.get("request_id")
            if not isinstance(request_id, str):
                raise ValueError(
                    "preservation event missing canonical request identity"
                )
            match = _CANONICAL_INTERNAL_ID_RE.fullmatch(request_id)
            if match is None:
                raise ValueError("malformed canonical preservation request identity")
            external_id = match.group(1)
            event["external_request_id"] = external_id
        if not isinstance(external_id, str):
            raise TypeError("preservation event external_request_id is not a string")
        external_match = _CANONICAL_EXTERNAL_ID_RE.fullmatch(external_id)
        if external_match is None:
            raise ValueError(
                f"malformed canonical preservation identity {external_id!r}"
            )
        request_id = event.get("request_id")
        if not isinstance(request_id, str):
            raise TypeError(f"missing canonical request_id for {external_id!r}")
        internal_match = _CANONICAL_INTERNAL_ID_RE.fullmatch(request_id)
        if internal_match is None or internal_match.group(1) != external_id:
            raise ValueError(f"request/external identity mismatch for {external_id!r}")
        sample_index = int(external_match.group(4))
        if not event.get("run_id"):
            event["run_id"] = external_match.group(1)
        if not event.get("cell_id"):
            event["cell_id"] = external_match.group(2)
        if not event.get("sample_id"):
            event["sample_id"] = external_id
        if event.get("sample_index") is None:
            event["sample_index"] = sample_index
        return event

    deadline = time.monotonic() + 10.0
    while True:
        events: list[dict] = []
        with open(log_path) as log_file:
            log_file.seek(start_offset)
            for line in _event_lines(log_file):
                marker_index = line.find(prefix)
                if marker_index < 0:
                    continue
                log_prefix = line[:marker_index]
                if log_prefix and not re.fullmatch(
                    r"\s*\(EngineCore pid=\d+\)\s*", log_prefix
                ):
                    raise ValueError(f"unknown preservation log prefix {log_prefix!r}")
                try:
                    event = json.loads(line[marker_index + len(prefix) :])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed preservation event: {exc}") from exc
                allowed_prefixes = (request_id_prefix, *allowed_request_prefixes)
                raw_external_id = event.get("external_request_id")
                if isinstance(raw_external_id, str):
                    if _NEEDLE_EXTERNAL_ID_RE.fullmatch(raw_external_id):
                        if not any(
                            raw_external_id.startswith(request_prefix)
                            for request_prefix in allowed_request_prefixes
                        ):
                            raise ValueError(
                                f"foreign needle preservation request {raw_external_id}"
                            )
                        request_id = event.get("request_id")
                        needle_match = (
                            _NEEDLE_INTERNAL_ID_RE.fullmatch(request_id)
                            if isinstance(request_id, str)
                            else None
                        )
                        if (
                            needle_match is None
                            or needle_match.group(1) != raw_external_id
                        ):
                            raise ValueError(
                                "malformed needle preservation request "
                                f"{raw_external_id}"
                            )
                        continue
                    if not raw_external_id.startswith("sc|"):
                        raise ValueError(
                            f"foreign preservation event request {raw_external_id}"
                        )
                    event = _canonicalize_event_identity(event)
                    external_id = event["external_request_id"]
                    if not any(
                        external_id.startswith(request_prefix)
                        for request_prefix in allowed_prefixes
                    ):
                        raise ValueError(
                            f"foreign preservation event request {external_id}"
                        )
                    if not external_id.startswith(request_id_prefix):
                        continue
                else:
                    raw_request_id = event.get("request_id")
                    if not isinstance(raw_request_id, str):
                        raise TypeError(
                            "preservation event missing canonical request identity"
                        )
                    needle_match = _NEEDLE_INTERNAL_ID_RE.fullmatch(raw_request_id)
                    if needle_match is not None:
                        if not any(
                            needle_match.group(1).startswith(request_prefix)
                            for request_prefix in allowed_request_prefixes
                        ):
                            raise ValueError(
                                "foreign needle preservation request "
                                f"{needle_match.group(1)}"
                            )
                        continue
                    if not _CANONICAL_INTERNAL_ID_RE.fullmatch(raw_request_id):
                        raise ValueError(
                            "malformed canonical preservation request identity"
                        )
                    event = _canonicalize_event_identity(event)
                    external_id = event["external_request_id"]
                    if not any(
                        external_id.startswith(request_prefix)
                        for request_prefix in allowed_prefixes
                    ):
                        raise ValueError(
                            f"foreign preservation event request {external_id}"
                        )
                    if not external_id.startswith(request_id_prefix):
                        continue
                if event.get("schema") != "semantic-preservation-event-v1":
                    raise ValueError("preservation event schema mismatch")
                if (
                    event.get("event_instance") != 1
                    or event.get("expected_event_count") != 1
                ):
                    raise ValueError(
                        f"invalid preservation event cardinality for {external_id}"
                    )
                if event.get("run_id") != run_id:
                    raise ValueError(
                        f"foreign preservation event run_id for {external_id}"
                    )
                if event.get("cell_id") != cell_id:
                    raise ValueError(
                        f"foreign preservation event cell_id for {external_id}"
                    )
                if external_id not in expected_request_ids:
                    raise ValueError(
                        f"foreign preservation event request {external_id}"
                    )
                if event.get("sample_id") != external_id:
                    raise ValueError(
                        f"sample/request identity mismatch for {external_id}"
                    )
                if (
                    not isinstance(event.get("sample_index"), int)
                    or event.get("sample_index") < 0
                ):
                    raise ValueError(f"invalid sample identity for {external_id}")
                try:
                    expected_sample_index = int(external_id.rsplit("sample-", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"unparseable sample identity for {external_id}"
                    ) from exc
                if event["sample_index"] != expected_sample_index:
                    raise ValueError(f"sample index mismatch for {external_id}")
                if (
                    not event.get("authoritative")
                    or event.get("event_integrity") != "complete"
                    or not event.get("transfer_metadata_complete")
                    or not event.get("admission_metadata_complete")
                    or not event.get("physical_block_identity_complete")
                    or not _terminal_success(event)
                ):
                    explicit_unavailable_reason = _explicit_unavailable_reason(event)
                    explicit_unavailable = (
                        event.get("authoritative") is False
                        and event.get("event_integrity") == "unavailable"
                        and event.get("outcome") == "unavailable"
                        and _terminal_success(event)
                        and explicit_unavailable_reason is not None
                        and event.get("reason") == explicit_unavailable_reason
                    )
                    if not explicit_unavailable:
                        raise ValueError(
                            f"non-authoritative or non-success event for {external_id}"
                        )
                events.append(event)

        ids = [event.get("external_request_id") for event in events]
        events_by_id: dict[str, list[dict]] = {}
        for event in events:
            events_by_id.setdefault(event["external_request_id"], []).append(event)
        collapsed_events: list[dict] = []
        for external_id, event_history in events_by_id.items():
            if len(event_history) == 1:
                collapsed_events.append(event_history[0])
                continue
            authoritative_events = [
                event for event in event_history if event.get("authoritative") is True
            ]
            unavailable_events = [
                event
                for event in event_history
                if event.get("authoritative") is False
                and event.get("event_integrity") == "unavailable"
            ]
            if (
                len(event_history) == 2
                and len(authoritative_events) == 1
                and len(unavailable_events) == 1
            ):
                internal_ids = [event.get("request_id") for event in event_history]
                if len(set(internal_ids)) != len(internal_ids):
                    raise ValueError(
                        "duplicate preservation events: "
                        f"{external_id} has repeated internal request_id"
                    )
                unavailable_event = dict(unavailable_events[0])
                unavailable_event["reason"] = "duplicate_external_request_events"
                unavailable_event["duplicate_event_count"] = len(event_history)
                collapsed_events.append(unavailable_event)
                continue
            raise ValueError(
                "duplicate preservation events: "
                f"{external_id} (count={len(event_history)})"
            )
        events = collapsed_events
        ids = [event.get("external_request_id") for event in events]
        missing = expected_request_ids - set(ids)
        if not missing:
            return sorted(events, key=lambda event: str(event["external_request_id"]))
        if time.monotonic() >= deadline:
            raise ValueError(
                "missing preservation events: " + ",".join(sorted(missing)[:8])
            )
        time.sleep(0.2)


def _read_decision_feature_records(
    log_path: str,
    request_id_prefix: str,
    expected_request_ids: Sequence[str],
    run_id: str,
    cell_id: str,
    start_offset: int = 0,
    allowed_request_prefixes: tuple[str, ...] = (),
) -> list[dict]:
    """Read and validate the separate decision-time feature stream."""
    records: list[dict] = []
    expected = set(expected_request_ids)
    if len(expected) != len(expected_request_ids):
        raise ValueError("duplicate request IDs in decision feature join")
    decoder = json.JSONDecoder()
    with open(log_path) as log_file:
        log_file.seek(start_offset)
        for raw_line in log_file:
            marker_positions = [
                match.start()
                for match in re.finditer(re.escape(TRACE_MARKER), raw_line)
            ]
            if marker_positions:
                original_prefix = raw_line[: marker_positions[0]]
                if original_prefix and not re.fullmatch(
                    r"\s*\(EngineCore pid=\d+\)\s*", original_prefix
                ):
                    raise ValueError(
                        f"unknown decision feature log prefix {original_prefix!r}"
                    )
            for index, marker_index in enumerate(marker_positions):
                end = (
                    marker_positions[index + 1]
                    if index + 1 < len(marker_positions)
                    else len(raw_line)
                )
                payload = raw_line[marker_index + len(TRACE_MARKER) : end].strip()
                try:
                    record, next_offset = decoder.raw_decode(payload, 0)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed decision feature record: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise TypeError(
                        "malformed decision feature record: expected JSON object"
                    )
                external_id = record.get("external_request_id")
                if not isinstance(external_id, str):
                    raise TypeError(
                        "decision feature record lacks external request identity"
                    )
                if external_id.startswith(request_id_prefix):
                    if external_id not in expected:
                        raise ValueError(
                            f"unexpected decision feature request {external_id}"
                        )
                elif any(
                    external_id.startswith(prefix)
                    for prefix in allowed_request_prefixes
                ):
                    # A mixed/needle stream explicitly allowed by the caller
                    # belongs to another row and is intentionally ignored.
                    trailing = payload[next_offset:]
                    if trailing.strip() and not (
                        _KNOWN_APISERVER_METRICS_SUFFIX_RE.fullmatch(trailing)
                    ):
                        raise ValueError(
                            "malformed decision feature record: non-whitespace text "
                            "after JSON"
                        )
                    continue
                else:
                    raise ValueError(f"foreign decision feature request {external_id}")
                validate_decision_feature_record(record, run_id=run_id, cell_id=cell_id)
                records.append(record)
                trailing = payload[next_offset:]
                if trailing.strip() and not (
                    _KNOWN_APISERVER_METRICS_SUFFIX_RE.fullmatch(trailing)
                ):
                    raise ValueError(
                        "malformed decision feature record: non-whitespace text "
                        "after JSON"
                    )
    return validate_decision_feature_records(
        records,
        expected_request_ids=expected_request_ids,
        run_id=run_id,
        cell_id=cell_id,
    )


def _prefetch_status_event(
    *,
    run_id: str,
    cell_id: str,
    seed: int | None,
    request_id: str,
    status: str,
    reason: str,
    unavailable_fields: list[str] | None = None,
) -> dict[str, object]:
    """Create an explicit non-success attribution without fake metrics."""
    return {
        "schema": PREFETCH_ATTRIBUTION_SCHEMA,
        "event": "prefetch_attribution",
        "run_id": run_id,
        "cell_id": cell_id,
        "seed": seed,
        "episode_id": "episode-unavailable",
        "request_id": request_id,
        "status": status,
        "status_reason": reason,
        "unavailable_fields": unavailable_fields or [],
    }


def _join_prefetch_attribution(
    events: Sequence[dict],
    request_ids: Sequence[str],
    *,
    run_id: str,
    cell_id: str,
    seed: object,
) -> dict[str, object]:
    """Join the optional nested producer record to the request order.

    Preservation events remain the authoritative request join.  Prefetch
    attribution is a separate status-bearing projection: a missing nested
    record is an unavailable attribution, and a malformed or foreign record
    is an invalid attribution. Neither case is turned into zero bytes or a
    latency measurement.
    """
    bound_seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
    joined: list[dict[str, object]] = []
    missing = False
    invalid = False
    for request_id, preservation_event in zip(request_ids, events):
        raw = preservation_event.get("prefetch_attribution")
        if not isinstance(raw, dict):
            missing = True
            joined.append(
                _prefetch_status_event(
                    run_id=run_id,
                    cell_id=cell_id,
                    seed=bound_seed,
                    request_id=request_id,
                    status="unavailable",
                    reason="prefetch_attribution_event_missing",
                    unavailable_fields=["prefetch_attribution_event"],
                )
            )
            continue
        candidate = dict(raw)
        candidate_seed = candidate.get("seed")
        # The request protocol does not carry the benchmark seed. Binding the
        # immutable cell seed here is an explicit harness join, not inference;
        # a conflicting producer value is rejected below.
        if candidate_seed is None and bound_seed is not None:
            candidate["seed"] = bound_seed
            candidate["seed_source"] = "harness_cell"
            candidate_seed = bound_seed
        if (
            candidate.get("run_id") != run_id
            or candidate.get("cell_id") != cell_id
            or candidate.get("request_id") != request_id
            or candidate_seed != bound_seed
        ):
            invalid = True
            joined.append(
                _prefetch_status_event(
                    run_id=run_id,
                    cell_id=cell_id,
                    seed=bound_seed,
                    request_id=request_id,
                    status="invalid",
                    reason="prefetch_attribution_identity_mismatch",
                    unavailable_fields=["prefetch_attribution_identity"],
                )
            )
            continue
        try:
            validate_prefetch_episode(candidate)
        except PrefetchAttributionError as exc:
            invalid = True
            joined.append(
                _prefetch_status_event(
                    run_id=run_id,
                    cell_id=cell_id,
                    seed=bound_seed,
                    request_id=request_id,
                    status="invalid",
                    reason=str(exc),
                    unavailable_fields=["prefetch_attribution_contract"],
                )
            )
            continue
        joined.append(candidate)
    join_status = "invalid" if invalid else "unavailable" if missing else "complete"
    return {
        "prefetch_attribution_json": json.dumps(
            joined, separators=(",", ":"), sort_keys=True
        ),
        "prefetch_attribution_count": len(joined),
        "prefetch_attribution_join_status": join_status,
    }


def join_preservation_events(
    row: dict,
    *,
    log_path: str,
    request_id_prefix: str,
    run_id: str,
    cell_id: str,
    expected_count: int,
    start_offset: int = 0,
    allowed_request_prefixes: tuple[str, ...] = (),
    decision_feature_trace_enabled: bool = False,
) -> dict:
    if not is_canonical_request_id_prefix(
        request_id_prefix, run_id=run_id, cell_id=cell_id
    ):
        raise ValueError(
            "preservation event join requires a canonical request_id_prefix"
        )
    request_ids = json.loads(row.get("request_ids_json") or "[]")
    if len(request_ids) != expected_count:
        raise ValueError(
            f"latency request cardinality mismatch: expected {expected_count}, "
            f"got {len(request_ids)}"
        )
    expected_ids = set(request_ids)
    if len(expected_ids) != expected_count:
        raise ValueError("duplicate request IDs in latency result")
    events = _read_preservation_events(
        log_path,
        request_id_prefix,
        expected_ids,
        run_id,
        cell_id,
        start_offset,
        allowed_request_prefixes,
    )
    events_by_id = {event["external_request_id"]: event for event in events}
    ordered_events = [events_by_id[request_id] for request_id in request_ids]
    row.update(
        {
            "preservation_events_json": json.dumps(
                ordered_events, separators=(",", ":"), sort_keys=True
            ),
            "preservation_event_count": len(ordered_events),
            "preservation_join_status": "complete",
        }
    )
    if decision_feature_trace_enabled:
        feature_records = _read_decision_feature_records(
            log_path,
            request_id_prefix,
            request_ids,
            run_id,
            cell_id,
            start_offset,
            allowed_request_prefixes,
        )
        row.update(
            {
                "decision_feature_trace_json": json.dumps(
                    feature_records, separators=(",", ":"), sort_keys=True
                ),
                "decision_feature_trace_count": len(feature_records),
                "decision_feature_trace_join_status": "complete",
            }
        )
    row.update(
        map_preservation_events(
            ordered_events,
            expected_request_ids=request_ids,
            run_id=run_id,
            cell_id=cell_id,
        )
    )
    row.update(
        _join_prefetch_attribution(
            ordered_events,
            request_ids,
            run_id=run_id,
            cell_id=cell_id,
            seed=row.get("seed"),
        )
    )
    return row


def run_mixed_case(
    base_url: str,
    model: str,
    output_dir: str,
    policy: str,
    rate: float,
    resolved_num_prompts: int,
    scale: float,
    seed: int | None,
    ref_counts: list[int],
    metrics_url: str | None = None,
    run_id: str | None = None,
    cell_id: str | None = None,
    preservation_events_enabled: bool = False,
    decision_feature_trace_enabled: bool = False,
    server_log_path: str | None = None,
    event_start_offset: int = 0,
) -> list[dict]:
    """`mixed`: chat/rag/longdoc run concurrently (via `vllm bench serve`
    sub-streams in parallel threads, proportioned by
    `_MIXED_SUBWORKLOAD_WEIGHTS` of the total rate/prompt count) plus a
    periodic stream of needle cases cycling through `ref_counts`, all
    against the same server in the same wall-clock window. Returns one row
    per sub-stream/needle-case, tagged `sub_workload`, rather than one
    lossily-averaged row -- chat/rag/longdoc and needle have different
    metric shapes (TTFT/ITL percentiles vs. a hit/miss rate) that don't
    average meaningfully. Each sub-stream's own failure (e.g. a stalled
    request, see vLLM issue #45388) is caught independently so one bad
    sub-stream doesn't lose the others' real data.

    `metrics_url`, when given, is used to snapshot the offload-subsystem
    Prometheus counters (`harness.metrics`) around EACH sub-stream/needle-
    case individually (issues log entry #76: the previous design took one
    snapshot pair around the whole concurrent call and copied the same
    aggregate `load_bytes_delta` onto every row via `setdefault` -- in
    practice that produced no usable per-row signal at all). Per-row
    snapshots are still an approximation, not exact attribution: the four
    streams share one live server and one set of global counters, so bytes
    another still-running stream moves during a given row's own window can
    bleed into that row's delta. That's strictly better than one shared
    number on every row, but still coarser than the standalone workloads'
    single-stream-at-a-time deltas -- worth keeping in mind when reading
    `mixed`'s per-row cache-hit volume. Without `metrics_url` (e.g. unit
    tests that don't stand up a real server) the byte/preemption fields are
    left `None` rather than guessed at."""

    def _snap() -> dict[str, float] | None:
        return metrics_mod.snapshot(metrics_url) if metrics_url else None

    def _delta_fields(
        before: dict[str, float] | None, after: dict[str, float] | None
    ) -> dict[str, float | None]:
        if before is None or after is None:
            return _metric_delta_fields(None)
        return _metric_delta_fields(metrics_mod.diff(before, after))

    def _run_sub(sub_workload: str, weight: float) -> dict:
        sub_rate = rate * weight if rate != float("inf") else rate
        sub_num_prompts = max(1, round(resolved_num_prompts * weight))
        result_path = os.path.join(
            output_dir,
            f"raw_{policy}_mixed-{sub_workload}_{rate}_{_seed_tag(seed)}.json",
        )
        request_prefix = (
            _request_id_prefix(run_id, cell_id, "mixed", sub_workload)
            if run_id is not None and cell_id is not None
            else None
        )
        before = _snap()
        try:
            if (
                sub_workload == "rag"
                and sub_num_prompts < workloads_mod.RAG_NUM_PREFIXES
            ):
                raise ValueError(
                    f"mixed's rag sub-stream num_prompts ({sub_num_prompts}, "
                    f"{resolved_num_prompts}*{weight} weight) is below "
                    f"RAG_NUM_PREFIXES ({workloads_mod.RAG_NUM_PREFIXES}) -- "
                    "raise the overall --num-prompts/--target-duration-s"
                )
            bench_kwargs = {}
            if preservation_events_enabled:
                bench_kwargs = {
                    "request_id_prefix": request_prefix,
                    "save_detailed": True,
                }
            row = run_vllm_bench_serve(
                base_url,
                model,
                sub_workload,
                sub_num_prompts,
                sub_rate,
                scale,
                result_path,
                seed=seed,
                **bench_kwargs,
            )
        except Exception as e:  # noqa: BLE001 - preserve each failed benchmark row
            row = {"error": str(e)[:500]}
        after = _snap()
        row.update(_delta_fields(before, after))
        row.update(
            {
                "policy": policy,
                "seed": seed,
                "workload": "mixed",
                "sub_workload": sub_workload,
                "request_rate": sub_rate,
                "parent_request_rate": rate,
                "num_prompts": sub_num_prompts,
                "sample_index": 0,
                "run_id": run_id,
                "cell_id": cell_id,
                "request_id_prefix": request_prefix,
                **_unavailable_preservation_fields(
                    AUTHORITATIVE_METADATA_MISSING_REASON
                ),
            }
        )
        if (
            not row.get("error")
            and preservation_events_enabled
            and server_log_path is not None
            and request_prefix is not None
        ):
            row = join_preservation_events(
                row,
                log_path=server_log_path,
                request_id_prefix=request_prefix,
                run_id=run_id,
                cell_id=cell_id,
                expected_count=sub_num_prompts,
                start_offset=event_start_offset,
                decision_feature_trace_enabled=decision_feature_trace_enabled,
                allowed_request_prefixes=(
                    *(
                        _request_id_prefix(run_id, cell_id, "mixed", name)
                        for name in _MIXED_SUBWORKLOAD_WEIGHTS
                        if name != sub_workload
                    ),
                    f"needle-v2-{seed}-",
                ),
            )
        return row

    def _run_needle_stream() -> list[dict]:
        duration_estimate = (
            resolved_num_prompts / rate if rate not in (0, float("inf")) else 60.0
        )
        n_cases = max(1, round(duration_estimate / 120))  # ~1 case/2 min
        rows = []
        for i in range(n_cases):
            ref_count = ref_counts[i % len(ref_counts)]
            row = {
                "policy": policy,
                "seed": seed,
                "workload": "mixed",
                "sub_workload": "needle",
                "parent_request_rate": rate,
                "reference_count": ref_count,
                "sample_index": i,
                **_needle_preservation_fields({"needle_outcome": None}),
            }
            before = _snap()
            try:
                # needle-v2's preservation-aware classification (issues log
                # entry #58), not the old `run_needle_case`'s `expected_code
                # in recall_text` check -- that check is mathematically
                # pinned to 1.0 for every policy under vLLM's lossless
                # offload path (the recall always recomputes the right
                # answer whether or not the cache preserved it), which is
                # exactly why `mixed`'s needle sub-stream read a constant
                # 1.0 `needle_hit_rate` regardless of real eviction
                # behavior. `run_needle_v2_case` isolates the recall's own
                # CPU-tier load/store bytes and classifies hit/miss/
                # not_pressured from those, same as the standalone
                # `needle-v2` workload branch in `main()`.
                result = needle_workload.run_needle_v2_case(
                    base_url,
                    reference_count=ref_count,
                    num_distractors=max(1, 10 - ref_count),
                    model=model,
                    seed=(seed or 0) * 10_000_000 + 1_000_000 + i,
                    snapshot_metrics=_snap if metrics_url else None,
                    request_id_prefix=(f"needle-v2-{seed}-ref{ref_count}-sample{i}"),
                )
                row["needle_outcome"] = result["needle_outcome"]
                row["recall_load_bytes"] = result["recall_load_bytes"]
                row["recall_store_bytes"] = result["recall_store_bytes"]
                row.update(_needle_preservation_fields(result))
                if result["needle_outcome"] is not None:
                    row["needle_hit_rate"] = (
                        1.0
                        if result["needle_outcome"] == needle_workload.NEEDLE_HIT
                        else 0.0
                    )
            except Exception as e:  # noqa: BLE001 - preserve the failed needle row
                row["error"] = str(e)[:500]
            after = _snap()
            row.update(_delta_fields(before, after))
            rows.append(row)
        return rows

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        sub_futures = [
            ex.submit(_run_sub, name, weight)
            for name, weight in _MIXED_SUBWORKLOAD_WEIGHTS.items()
        ]
        needle_future = ex.submit(_run_needle_stream)
        for fut in concurrent.futures.as_completed(sub_futures):
            rows.append(fut.result())
        rows.extend(needle_future.result())
    return rows


_active_handle: list = []


def _handle_sigterm(signum, frame):
    """Best-effort graceful shutdown when killed from outside (e.g. by
    run_grid_sweep.py's per-cell timeout after a stalled request, see
    vLLM issue #45388) -- without this, the `finally: handle.shutdown()`
    block never runs on SIGTERM and the `vllm serve` child (its own
    process group via setsid) is orphaned, leaking GPU memory and the
    port for the next cell."""
    if _active_handle:
        try:
            _active_handle[0].shutdown()
        except Exception:  # noqa: BLE001, S110 - best-effort signal cleanup
            pass
    sys.exit(143)


def _exception_returncode(error: BaseException | None) -> int:
    if error is None:
        return 0
    if isinstance(error, SystemExit) and isinstance(error.code, int):
        return error.code
    return 1


def _config_identity(args, policy: str) -> str:
    payload = {
        "model": args.model,
        "policy": policy,
        "workloads": args.workloads,
        "request_rates": args.request_rates,
        "needle_reference_counts": args.needle_reference_counts,
        "needle_num_distractors": args.needle_num_distractors,
        "needle_settle_s": args.needle_settle_s,
        "needle_max_settle_polls": args.needle_max_settle_polls,
        "num_prompts": args.num_prompts,
        "target_duration_s": args.target_duration_s,
        "scale": args.scale,
        "cpu_bytes_to_use": args.cpu_bytes_to_use,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "num_gpu_blocks_override": args.num_gpu_blocks_override,
        "ready_timeout_s": args.ready_timeout_s,
        "extra_config": getattr(args, "extra_config", None),
        "seed": args.seed,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_manifest(args, *, run_id: str, output_dir: str) -> dict[str, object]:
    """Return reproducibility metadata for one latency-suite invocation."""
    repo_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_clean = not bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
        tracked_clean = None
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "schema": "semantic-latency-suite-manifest-v1",
        "run_id": run_id,
        "output_dir": output_dir,
        "argv": list(sys.argv),
        "python": sys.executable,
        "platform": platform.platform(),
        "repo": {
            "root": str(repo_root),
            "revision": revision or None,
            "tracked_worktree_clean": tracked_clean,
        },
        "result_fields": list(RESULT_FIELDS),
        "config": config,
    }


def _write_run_manifest(args, *, run_id: str) -> str:
    """Write an immutable run manifest and return its path."""
    output_dir = Path(args.output_dir)
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    manifest_path = output_dir / f"run_manifest_{safe_run_id}.json"
    payload = _run_manifest(args, run_id=run_id, output_dir=str(output_dir))
    try:
        with manifest_path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"run manifest already exists at {manifest_path}; use a fresh "
            "output directory"
        ) from exc
    return str(manifest_path)


def _start_cell_marker(
    args, policy: str, marker_path: str, rows_before: int
) -> dict[str, object]:
    attempt_id = args.cell_attempt_id or (f"standalone-{os.getpid()}-{time.time_ns()}")
    run_id = getattr(args, "run_id", None) or attempt_id
    cell_id = getattr(args, "cell_id", None) or f"{policy}/seed{args.seed}"
    payload = {
        "policy": policy,
        "seed": args.seed,
        "attempt_id": attempt_id,
        "status": "running",
        "phase": "runner_started",
        "termination_reason": "running",
        "termination_signal": None,
        "runner_pid": os.getpid(),
        "runner_ppid": os.getppid(),
        "runner_pgid": os.getpgid(os.getpid()),
        "runner_birth_id": process_birth_identity(os.getpid()),
        "grid_pid": args.cell_grid_pid,
        "grid_birth_id": getattr(args, "cell_grid_birth_id", None),
        "config_identity": _config_identity(args, policy),
        "run_id": run_id,
        "cell_id": cell_id,
        "preservation_events_enabled": bool(
            getattr(args, "extra_config", None)
            and json.loads(args.extra_config).get("preservation_events") is True
        ),
        "decision_feature_trace_enabled": bool(
            getattr(args, "extra_config", None)
            and json.loads(args.extra_config).get("decision_feature_trace") is True
        ),
        "result_path": cell_result_path(args.output_dir, args.cell_attempt_id),
        "rows_before": rows_before,
        "rows_written": 0,
    }
    if args.cell_attempt_id is not None:
        expected = {
            "attempt_id": attempt_id,
            "status": "running",
            "grid_pid": args.cell_grid_pid,
        }
        cell_grid_birth_id = getattr(args, "cell_grid_birth_id", None)
        if cell_grid_birth_id is not None:
            expected["grid_birth_id"] = cell_grid_birth_id
        return update_cell_marker(
            marker_path,
            payload,
            expected=expected,
        )
    write_cell_marker(marker_path, payload)
    return payload


def _finish_cell_marker(
    marker_path: str,
    start: dict[str, object],
    csv_path: str,
    error: BaseException | None,
    handle=None,
    csv_file=None,
) -> None:
    if csv_file is not None:
        csv_file.flush()
        os.fsync(csv_file.fileno())
    returncode = _exception_returncode(error)
    details = termination_details(returncode)
    rows_before = int(start.get("rows_before", 0) or 0)
    rows_after = count_csv_rows(csv_path)
    server_proc = getattr(handle, "proc", None)
    server_pid = getattr(server_proc, "pid", None)
    current = read_cell_marker(marker_path) or {}
    if current.get("status") == "failed" and current.get("ended_by") == "cell_watchdog":
        return
    if server_pid is None:
        server_pid = current.get("server_pid")
    server_returncode = server_proc.poll() if server_proc is not None else None
    if server_returncode is None:
        server_returncode = current.get("server_returncode")
    payload = {
        **start,
        **current,
        "status": "completed" if error is None else "failed",
        "phase": "runner_finalized",
        "termination_reason": details["termination_reason"],
        "termination_signal": details["termination_signal"],
        "runner_returncode": returncode,
        "server_pid": server_pid,
        "server_pgid": current.get("server_pgid"),
        "server_returncode": server_returncode,
        "rows_written": max(0, rows_after - rows_before),
        "ended_by": "run_latency_suite",
    }
    if error is not None:
        payload["error"] = str(error)[:1000]
    update_cell_marker(
        marker_path,
        payload,
        expected={
            "attempt_id": start["attempt_id"],
            "runner_pid": start["runner_pid"],
            "runner_ppid": start["runner_ppid"],
            "grid_pid": start.get("grid_pid"),
            "status": "running",
        },
    )


def _write_row_or_failure(
    writer,
    csv_file,
    row: dict,
    *,
    context: str,
    validation_failures: list[str],
    validate_latency: bool,
) -> None:
    """Write only validated data, or a quarantined error identity row."""
    row_error = (row.get("error") or "").strip()
    errors = [row_error] if row_error else []
    if not errors and validate_latency:
        errors = validate_latency_row(row, require_complete=True)
    if errors:
        message = f"{context}: " + "; ".join(errors)
        validation_failures.append(message)
        failure_row = {
            field: row.get(field)
            for field in (
                "policy",
                "seed",
                "workload",
                "sub_workload",
                "request_rate",
                "parent_request_rate",
                "reference_count",
                "sample_index",
            )
            if field in row
        }
        failure_row["error"] = message
        writer.writerow(failure_row)
    else:
        writer.writerow(row)
    csv_file.flush()


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policies", required=True, help="comma-separated")
    parser.add_argument("--workloads", required=True, help="comma-separated")
    parser.add_argument("--request-rates", default="inf", help="comma-separated")
    parser.add_argument(
        "--needle-reference-counts", default="0,1,2", help="comma-separated"
    )
    parser.add_argument(
        "--needle-num-distractors",
        type=int,
        default=None,
        help=(
            "fixed distractor count for every needle/needle-v2 case, held "
            "constant across all --needle-reference-counts values in this "
            "run. Defaults to needle_num_prompts (--num-prompts, or the "
            "target-duration-derived estimate) when unset. Deliberately "
            "does NOT scale with reference_count -- see issues log entry "
            "#77: subtracting reference_count from a shared prompt budget "
            "confounded 'was referenced' with 'less total distractor "
            "traffic before recall'."
        ),
    )
    parser.add_argument(
        "--needle-settle-s",
        type=float,
        default=1.0,
        help=(
            "needle-v2 only: seconds between polls while waiting for the "
            "distractor store counter to stop moving before snapshotting "
            "the recall's 'before' baseline (see needle_workload."
            "_drain_store_counter). Under heavy host contention (e.g. an "
            "8-GPU concurrent grid sharing host CPU/RAM/PCIe bandwidth for "
            "CPU-tier offload), the default may settle too early and "
            "contaminate the recall-isolated delta -- raise this and/or "
            "--needle-max-settle-polls if grid runs show settle-exhausted "
            "warnings that isolated single-GPU runs don't (entry #77)."
        ),
    )
    parser.add_argument(
        "--needle-max-settle-polls",
        type=int,
        default=5,
        help="needle-v2 only: max polls in the store-counter settle wait "
        "above, before giving up and using the last-read snapshot anyway.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="flat request count per run; mutually exclusive with --target-duration-s",
    )
    parser.add_argument(
        "--target-duration-s",
        type=float,
        default=None,
        help=(
            "derive num_prompts as target_duration_s * request_rate per "
            "(workload, rate) cell, so every cell gets the same steady-state "
            "window instead of a flat prompt count producing very different "
            "durations at different arrival rates (plan protocol: >=10 min)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "single seed for this invocation's `vllm bench serve` calls "
            "(passed through as-is); a multi-seed sweep loops this whole "
            "script once per seed, see benchmarks/run_grid_sweep.py"
        ),
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--cpu-bytes-to-use", type=int, default=2 * 1024**3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument(
        "--ready-timeout-s",
        type=float,
        default=180.0,
        help="seconds to wait for vllm serve /health before rejecting a cell",
    )
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell-attempt-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cell-grid-pid", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--cell-grid-birth-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cell-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--extra-config",
        default=None,
        help=(
            "JSON dict merged into each semantic policy's "
            "kv_connector_extra_config (ignored for lru/arc). Use to enable "
            "SemanticPolicy tuning knobs, e.g. "
            '\'{"session_aware": true, "session_bonus_half_life": 8}\'.'
        ),
    )
    args = parser.parse_args()

    extra_config = json.loads(args.extra_config) if args.extra_config else None
    preservation_events_enabled = bool(
        extra_config and extra_config.get("preservation_events") is True
    )
    decision_feature_trace_enabled = bool(
        extra_config and extra_config.get("decision_feature_trace") is True
    )
    if decision_feature_trace_enabled and not preservation_events_enabled:
        parser.error(
            "--extra-config decision_feature_trace requires preservation_events"
        )
    run_id = (
        args.run_id
        or args.cell_attempt_id
        or f"standalone-{os.getpid()}-{time.time_ns()}"
    )

    if args.num_prompts is None and args.target_duration_s is None:
        parser.error("one of --num-prompts or --target-duration-s is required")
    if args.num_prompts is not None and args.target_duration_s is not None:
        parser.error("--num-prompts and --target-duration-s are mutually exclusive")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = _write_run_manifest(args, run_id=run_id)
    print(f"wrote {manifest_path}", flush=True)
    csv_path = cell_result_path(args.output_dir, args.cell_attempt_id)
    write_header = not os.path.exists(csv_path)
    if not write_header:
        check_existing_schema(csv_path)

    policies = args.policies.split(",")
    workloads = args.workloads.split(",")
    request_rates = [float(r) for r in args.request_rates.split(",")]
    ref_counts = [int(r) for r in args.needle_reference_counts.split(",")]

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()

        for policy in policies:
            cell_id = args.cell_id or f"{policy}/seed{args.seed}"
            marker_path = cell_marker_path(
                args.output_dir, policy, args.seed, args.cell_attempt_id
            )
            rows_before = count_csv_rows(csv_path)
            marker_start = _start_cell_marker(args, policy, marker_path, rows_before)
            kv_config = policies_mod.kv_transfer_config(
                policy, args.cpu_bytes_to_use, extra_config=extra_config
            )
            print(f"=== policy={policy} : launching server ===", flush=True)
            try:
                handle = launch_server(
                    args.model,
                    args.port,
                    args.output_dir,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    num_gpu_blocks_override=args.num_gpu_blocks_override,
                    ready_timeout_s=args.ready_timeout_s,
                    kv_transfer_config=kv_config,
                    log_label=policy,
                )
            except (RuntimeError, TimeoutError) as e:
                print(f"policy={policy} FAILED TO LAUNCH: {e}", flush=True)
                writer.writerow({"policy": policy, "error": str(e)[:500]})
                csv_file.flush()
                _finish_cell_marker(
                    marker_path, marker_start, csv_path, e, csv_file=csv_file
                )
                continue

            _active_handle[:] = [handle]
            server_proc = getattr(handle, "proc", None)
            server_pid = getattr(server_proc, "pid", None)
            try:
                server_pgid = os.getpgid(server_pid) if server_pid is not None else None
            except ProcessLookupError:
                server_pgid = None
            update_cell_marker(
                marker_path,
                {
                    "phase": "server_started",
                    "server_pid": server_pid,
                    "server_pgid": server_pgid,
                    "server_birth_id": process_birth_identity(server_pid),
                    "server_log_path": getattr(handle, "log_path", None),
                },
                expected={
                    "attempt_id": marker_start["attempt_id"],
                    "runner_pid": marker_start["runner_pid"],
                    "runner_ppid": marker_start["runner_ppid"],
                    "grid_pid": marker_start.get("grid_pid"),
                },
            )

            # needle isn't rate-based, so --target-duration-s doesn't map
            # onto it the same way -- fall back to a rough ~15s/distractor
            # estimate when only a duration was given, matching this
            # project's own needle-case latency scale (issues log entries
            # #33/#35).
            needle_num_prompts = args.num_prompts or max(
                1, round(args.target_duration_s / 15)
            )
            needle_num_distractors = resolve_needle_num_distractors(
                needle_num_prompts, args.needle_num_distractors
            )
            # The grid's real per-cell seed (entry #84) -- must drive content
            # generation, not `reference_count` itself, or `--seeds` becomes a
            # no-op for needle/needle-v2. See needle_case_content_seed's
            # docstring for why.
            needle_case_seed_base = args.seed if args.seed is not None else 0

            def _needle_content_seed(
                reference_count: int, case_seed: int = needle_case_seed_base
            ) -> int:
                # The DGX audit runs each reference-count arm against a fresh
                # server and opts into identical content across arms. General
                # multi-arm invocations retain disjoint content to avoid
                # cross-case prefix reuse on their shared server.
                if os.environ.get("NEEDLE_SHARED_CONTENT") == "1":
                    return needle_workload.needle_case_content_seed(case_seed, 0)
                return needle_workload.needle_case_content_seed(
                    case_seed, reference_count
                )

            cell_error = None
            validation_failures: list[str] = []
            try:
                for workload in workloads:
                    if workload == "needle":
                        for ref_count in ref_counts:
                            try:
                                before = metrics_mod.snapshot(handle.metrics_url())
                                result = needle_workload.run_needle_case(
                                    handle.base_url(),
                                    reference_count=ref_count,
                                    num_distractors=needle_num_distractors,
                                    model=args.model,
                                    seed=_needle_content_seed(ref_count),
                                )
                                after = metrics_mod.snapshot(handle.metrics_url())
                                delta = metrics_mod.diff(before, after)
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "sample_index": 0,
                                    "needle_hit_rate": 1.0 if result["hit"] else 0.0,
                                    "duration_s": result["t_needle_s"]
                                    + result["t_recall_s"]
                                    + sum(result["probe_latencies_s"])
                                    + sum(result["distractor_latencies_s"]),
                                }
                                row.update(_metric_delta_fields(delta))
                                row.update(
                                    _unavailable_preservation_fields(
                                        "legacy_needle_has_no_isolated_preservation_label"
                                    )
                                )
                            except Exception as e:  # noqa: BLE001 - preserve failed cell
                                # A stalled/hung request (see vLLM issue
                                # #45388) or any other per-case failure must
                                # not abort every remaining cell in this
                                # (and every later) policy's run.
                                print(
                                    f"policy={policy} workload=needle "
                                    f"ref_count={ref_count} FAILED: {e}",
                                    flush=True,
                                )
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "sample_index": 0,
                                    "error": str(e)[:500],
                                }
                                row.update(
                                    _unavailable_preservation_fields(
                                        "legacy_needle_has_no_isolated_preservation_label"
                                    )
                                )
                            _write_row_or_failure(
                                writer,
                                csv_file,
                                row,
                                context=(
                                    f"policy={policy} workload={workload} "
                                    f"ref_count={ref_count}"
                                ),
                                validation_failures=validation_failures,
                                validate_latency=False,
                            )
                    elif workload == "needle-v2":
                        # Preservation-aware needle: the recall's cache-hit
                        # outcome (hit/miss/not_pressured), the only signal
                        # that distinguishes semantic-preserved from
                        # lru-evicted under lossless offload (issues log
                        # entry #58). REQUIRES a tightened GPU
                        # (--num-gpu-blocks-override) and a small
                        # --cpu-bytes-to-use, else every cell reads
                        # not_pressured (the built-in validity check).
                        def _snap(server_handle=handle):
                            return metrics_mod.snapshot(server_handle.metrics_url())

                        for ref_count in ref_counts:
                            try:
                                before = _snap()
                                result = needle_workload.run_needle_v2_case(
                                    handle.base_url(),
                                    reference_count=ref_count,
                                    num_distractors=needle_num_distractors,
                                    model=args.model,
                                    seed=_needle_content_seed(ref_count),
                                    snapshot_metrics=_snap,
                                    settle_s=args.needle_settle_s,
                                    max_settle_polls=args.needle_max_settle_polls,
                                    request_id_prefix=(
                                        f"needle-v2-{args.seed}-ref{ref_count}-sample0"
                                    ),
                                )
                                after = _snap()
                                delta = metrics_mod.diff(before, after)
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "sample_index": 0,
                                    "needle_outcome": result["needle_outcome"],
                                    "recall_load_bytes": result["recall_load_bytes"],
                                    "recall_store_bytes": result["recall_store_bytes"],
                                    "needle_hit_rate": (
                                        1.0
                                        if result["needle_outcome"]
                                        == needle_workload.NEEDLE_HIT
                                        else 0.0
                                    ),
                                    "duration_s": result["t_needle_s"]
                                    + result["t_recall_s"]
                                    + sum(result["probe_latencies_s"])
                                    + sum(result["distractor_latencies_s"]),
                                }
                                row.update(_metric_delta_fields(delta))
                                row.update(_needle_preservation_fields(result))
                            except Exception as e:  # noqa: BLE001 - preserve failed cell
                                print(
                                    f"policy={policy} workload=needle-v2 "
                                    f"ref_count={ref_count} FAILED: {e}",
                                    flush=True,
                                )
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "reference_count": ref_count,
                                    "sample_index": 0,
                                    "error": str(e)[:500],
                                }
                                row.update(
                                    _unavailable_preservation_fields(
                                        NEEDLE_METRICS_UNAVAILABLE_REASON
                                    )
                                )
                            _write_row_or_failure(
                                writer,
                                csv_file,
                                row,
                                context=(
                                    f"policy={policy} workload={workload} "
                                    f"ref_count={ref_count}"
                                ),
                                validation_failures=validation_failures,
                                validate_latency=False,
                            )
                    elif workload == "mixed":
                        for rate in request_rates:
                            resolved_num_prompts = resolve_num_prompts(
                                rate, args.num_prompts, args.target_duration_s
                            )
                            before = metrics_mod.snapshot(handle.metrics_url())
                            event_start_offset = (
                                os.path.getsize(handle.log_path)
                                if preservation_events_enabled
                                else 0
                            )
                            try:
                                mixed_rows = run_mixed_case(
                                    handle.base_url(),
                                    args.model,
                                    args.output_dir,
                                    policy,
                                    rate,
                                    resolved_num_prompts,
                                    args.scale,
                                    args.seed,
                                    ref_counts,
                                    metrics_url=handle.metrics_url(),
                                    run_id=run_id,
                                    cell_id=cell_id,
                                    preservation_events_enabled=(
                                        preservation_events_enabled
                                    ),
                                    decision_feature_trace_enabled=(
                                        decision_feature_trace_enabled
                                        and policy.startswith("semantic-")
                                    ),
                                    server_log_path=handle.log_path,
                                    event_start_offset=event_start_offset,
                                )
                            except Exception as e:  # noqa: BLE001 - preserve failed cell
                                print(
                                    f"policy={policy} workload=mixed "
                                    f"rate={rate} FAILED: {e}",
                                    flush=True,
                                )
                                mixed_rows = [
                                    {
                                        "policy": policy,
                                        "seed": args.seed,
                                        "workload": "mixed",
                                        "request_rate": rate,
                                        "num_prompts": resolved_num_prompts,
                                        "sample_index": 0,
                                        "run_id": run_id,
                                        "cell_id": cell_id,
                                        "request_id_prefix": None,
                                        "error": str(e)[:500],
                                        **_unavailable_preservation_fields(
                                            "mixed_case_failed_before_preservation_classification"
                                        ),
                                    }
                                ]
                            after = metrics_mod.snapshot(handle.metrics_url())
                            delta = metrics_mod.diff(before, after)
                            # run_mixed_case already attributes its own
                            # per-row byte/preemption deltas (issues log
                            # entry #76) -- these setdefaults are only a
                            # fallback for the single synthetic error row
                            # produced above when run_mixed_case raised
                            # before populating anything itself.
                            for row in mixed_rows:
                                for field, value in _metric_delta_fields(delta).items():
                                    row.setdefault(field, value)
                                _write_row_or_failure(
                                    writer,
                                    csv_file,
                                    row,
                                    context=(
                                        f"policy={policy} workload=mixed rate={rate}"
                                    ),
                                    validation_failures=validation_failures,
                                    validate_latency=not is_needle_row(row),
                                )
                            print(
                                f"policy={policy} workload=mixed rate={rate} "
                                f"sub_rows={len(mixed_rows)} metrics_delta={delta}",
                                flush=True,
                            )
                    else:
                        for rate in request_rates:
                            resolved_num_prompts = resolve_num_prompts(
                                rate,
                                args.num_prompts,
                                args.target_duration_s,
                                min_num_prompts=workloads_mod.RAG_NUM_PREFIXES
                                if workload == "rag"
                                else None,
                            )
                            try:
                                before = metrics_mod.snapshot(handle.metrics_url())
                                event_start_offset = (
                                    os.path.getsize(handle.log_path)
                                    if preservation_events_enabled
                                    else 0
                                )
                                result_path = os.path.join(
                                    args.output_dir,
                                    f"raw_{policy}_{workload}_{rate}_"
                                    f"{_seed_tag(args.seed)}.json",
                                )
                                request_prefix = _request_id_prefix(
                                    run_id, cell_id, workload
                                )
                                row = run_vllm_bench_serve(
                                    handle.base_url(),
                                    args.model,
                                    workload,
                                    resolved_num_prompts,
                                    rate,
                                    args.scale,
                                    result_path,
                                    seed=args.seed,
                                    request_id_prefix=request_prefix,
                                    save_detailed=preservation_events_enabled,
                                )
                                after = metrics_mod.snapshot(handle.metrics_url())
                                delta = metrics_mod.diff(before, after)
                                row.update(
                                    {
                                        "policy": policy,
                                        "seed": args.seed,
                                        "workload": workload,
                                        "request_rate": rate,
                                        "num_prompts": resolved_num_prompts,
                                        "sample_index": 0,
                                        "run_id": run_id,
                                        "cell_id": cell_id,
                                        "request_id_prefix": request_prefix,
                                    }
                                )
                                row.update(_metric_delta_fields(delta))
                                row.update(
                                    _unavailable_preservation_fields(
                                        AUTHORITATIVE_METADATA_MISSING_REASON
                                    )
                                )
                                if preservation_events_enabled:
                                    row = join_preservation_events(
                                        row,
                                        log_path=handle.log_path,
                                        request_id_prefix=request_prefix,
                                        run_id=run_id,
                                        cell_id=cell_id,
                                        expected_count=resolved_num_prompts,
                                        start_offset=event_start_offset,
                                        decision_feature_trace_enabled=(
                                            decision_feature_trace_enabled
                                            and policy.startswith("semantic-")
                                        ),
                                    )
                                print(
                                    f"policy={policy} workload={workload} "
                                    f"rate={rate} metrics_delta={delta}",
                                    flush=True,
                                )
                            except Exception as e:  # noqa: BLE001 - preserve failed cell
                                # Same rationale as the needle branch above:
                                # one stalled cell must not abort the rest
                                # of the grid.
                                print(
                                    f"policy={policy} workload={workload} "
                                    f"rate={rate} FAILED: {e}",
                                    flush=True,
                                )
                                row = {
                                    "policy": policy,
                                    "seed": args.seed,
                                    "workload": workload,
                                    "request_rate": rate,
                                    "num_prompts": resolved_num_prompts,
                                    "sample_index": 0,
                                    "run_id": run_id,
                                    "cell_id": cell_id,
                                    "request_id_prefix": _request_id_prefix(
                                        run_id, cell_id, workload
                                    ),
                                    "error": str(e)[:500],
                                    **_unavailable_preservation_fields(
                                        AUTHORITATIVE_METADATA_MISSING_REASON
                                    ),
                                }
                            _write_row_or_failure(
                                writer,
                                csv_file,
                                row,
                                context=(
                                    f"policy={policy} workload={workload} rate={rate}"
                                ),
                                validation_failures=validation_failures,
                                validate_latency=not is_needle_row(row),
                            )
                if validation_failures:
                    raise RuntimeError(
                        "completed row validation failed: "
                        + "; ".join(validation_failures)
                    )
            except BaseException as error:
                cell_error = error
                raise
            finally:
                print(f"=== policy={policy} : shutting down ===", flush=True)
                shutdown_error = None
                try:
                    handle.shutdown()
                except BaseException as error:
                    shutdown_error = error
                    raise
                finally:
                    _active_handle.clear()
                    time.sleep(2.0)
                    _finish_cell_marker(
                        marker_path,
                        marker_start,
                        csv_path,
                        cell_error or shutdown_error,
                        handle,
                        csv_file,
                    )

    print(f"Done. Results: {csv_path}")


if __name__ == "__main__":
    main()
