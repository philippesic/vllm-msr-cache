# SPDX-License-Identifier: Apache-2.0
"""Snapshot/diff vLLM's real Prometheus /metrics endpoint for the
offload-subsystem counters and request-time histogram aggregates this project
cares about. `vllm bench serve` reports TTFT/ITL/throughput itself; these
fields add offload traffic, preemptions, and the request's e2e/queue/prefill/
inference time decomposition without changing the load generator.

Deliberately a thin, dependency-free text-format parser (stdlib urllib +
regex) rather than pulling in prometheus_client -- the exposition format
is simple enough (`name{labels} value`) that a real parser dependency
would be more machinery than the problem needs."""

import re
import urllib.request
from math import isfinite

# Prefixes worth tracking for the latency suite's "index overhead" and
# "offload internals" metrics (plan's Benchmarking section). Summed across
# all label combinations for a given metric name -- per-run totals, not
# broken out per-label, since none of these currently vary by label in
# this project's single-model single-worker setup.
TRACKED_METRICS = (
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:num_preemptions_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    # vLLM exports these as monotonic Prometheus histogram sum/count series.
    # Keep the aggregates rather than reconstructing percentiles: interval
    # deltas give exact total time and completed-request count for the same
    # window as the existing offload counters.
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_inference_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:request_prefill_kv_computed_tokens_count",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)
GAUGE_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
    "vllm:kv_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_write_usage_perc",
    "vllm:kv_offload_cpu_cache_read_usage_perc",
    "vllm:semantic_async_scoring_pending",
)
REQUEST_HISTOGRAM_METRICS = frozenset(TRACKED_METRICS[3:])
_DEPRECATED_TOTAL_BYTES = "vllm:kv_offload_total_bytes_total"
_DEPRECATED_BYTE_DIRECTIONS = {
    "CPU_to_GPU": "vllm:kv_offload_load_bytes_total",
    "GPU_to_CPU": "vllm:kv_offload_store_bytes_total",
}

_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?P<labels>\{[^}]*\})?\s+(?P<value>[-\d.eE+]+)\s*$"
)
_TRANSFER_TYPE_RE = re.compile(r'(?:^|,)transfer_type="(?P<value>[^"]+)"(?:,|$)')


def fetch_metrics_text(metrics_url: str, timeout_s: float = 10.0) -> str:
    with urllib.request.urlopen(metrics_url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_selected_metrics(
    text: str, names: tuple[str, ...]
) -> dict[str, float | None]:
    return _parse_selected_metrics_evidence(text, names)[0]


def _parse_selected_metrics_evidence(
    text: str, names: tuple[str, ...]
) -> tuple[dict[str, float | None], dict[str, str], list[str]]:
    totals: dict[str, float | None] = {name: None for name in names}
    sources: dict[str, str] = {name: "unavailable" for name in names}
    deprecated_bytes: dict[str, float] = {}
    selected_lines: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name in totals:
            try:
                value = float(m.group("value"))
                totals[name] = (totals[name] or 0.0) + value
                sources[name] = "modern_flat_counter"
                selected_lines.append(line)
            except ValueError:
                continue
        elif name == _DEPRECATED_TOTAL_BYTES:
            labels = (m.group("labels") or "").strip("{}")
            transfer_match = _TRANSFER_TYPE_RE.search(labels)
            if transfer_match is None:
                continue
            canonical_name = _DEPRECATED_BYTE_DIRECTIONS.get(
                transfer_match.group("value")
            )
            if canonical_name not in totals:
                continue
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            deprecated_bytes[canonical_name] = (
                deprecated_bytes.get(canonical_name, 0.0) + value
            )
            selected_lines.append(line)
    for canonical_name, value in deprecated_bytes.items():
        if totals[canonical_name] is None:
            totals[canonical_name] = value
            sources[canonical_name] = "eager_directional_compat_counter"
    return totals, sources, selected_lines


def parse_metrics(text: str) -> dict[str, float | None]:
    """Sum tracked counter/histogram metrics, preserving absent as unavailable."""
    return _parse_selected_metrics(text, TRACKED_METRICS)


def parse_metrics_evidence(text: str) -> dict[str, object]:
    values, sources, selected_lines = _parse_selected_metrics_evidence(
        text, TRACKED_METRICS
    )
    return {
        "values": values,
        "sources": sources,
        "selected_exposition_lines": selected_lines,
    }


def parse_gauge_metrics(text: str) -> dict[str, float | None]:
    """Sum exposed scheduler gauges across label/engine combinations."""
    return _parse_selected_metrics(text, GAUGE_METRICS)


def snapshot(metrics_url: str) -> dict[str, float | None]:
    return parse_metrics(fetch_metrics_text(metrics_url))


def snapshot_evidence(metrics_url: str) -> dict[str, object]:
    return parse_metrics_evidence(fetch_metrics_text(metrics_url))


def snapshot_gauges(metrics_url: str) -> dict[str, float | None]:
    return parse_gauge_metrics(fetch_metrics_text(metrics_url))


def diff(
    before: dict[str, float | None], after: dict[str, float | None]
) -> dict[str, float | None]:
    return diff_with_provenance(before, after)[0]


def diff_with_provenance(
    before: dict[str, float | None], after: dict[str, float | None]
) -> tuple[dict[str, float | None], dict[str, str]]:
    result: dict[str, float | None] = {}
    provenance: dict[str, str] = {}
    for key, before_value in before.items():
        after_value = after.get(key)
        if before_value is None or after_value is None:
            result[key] = None
            provenance[key] = "unavailable_missing_endpoint_series"
        else:
            delta = after_value - before_value
            result[key] = delta if isfinite(delta) and delta >= 0 else None
            provenance[key] = (
                "observed_endpoint_delta"
                if result[key] is not None
                else "unavailable_counter_reset_or_nonfinite"
            )
    return result, provenance
