# SPDX-License-Identifier: Apache-2.0
"""Validation and conservative mapping for preservation labels.

The semantic connector emits one authoritative event per completed request.
This module is deliberately the only place that turns those event-level
facts into a row-level Chat/RAG outcome.  It never derives a label from the
aggregate load/store byte counters: those counters cover a whole workload
interval and cannot distinguish a reused block from unrelated traffic.
"""

import hashlib
import json
import math
import re

ALLOWED_PRESERVATION_OUTCOMES = frozenset(
    {"unavailable", "hit", "partial", "miss", "not_pressured"}
)
AUTHORITATIVE_METADATA_MISSING_REASON = "authoritative_metadata_missing"
NEEDLE_ISOLATED_COUNTER_REASON = "needle_v2_recall_isolated_counters"
NEEDLE_METRICS_UNAVAILABLE_REASON = "needle_v2_metrics_unavailable"
LEGACY_NEEDLE_UNAVAILABLE_REASON = "legacy_needle_has_no_isolated_preservation_label"
AUTHORITATIVE_REQUEST_EVENTS_BASIS = "authoritative_request_events"
MIXED_REQUEST_OUTCOMES_REASON = "mixed_request_outcomes"
REQUEST_EVENT_UNAVAILABLE_REASON = "request_event_unavailable"
REQUEST_EVENT_MISSING_REASON = "missing_request_events"
REQUEST_EVENT_DUPLICATE_REASON = "duplicate_request_events"
REQUEST_EVENT_FOREIGN_REASON = "foreign_request_events"
REQUEST_EVENT_IDENTITY_REASON = "request_event_identity_incomplete"
REQUEST_EVENT_TRANSFER_REASON = "transfer_metadata_unavailable"
REQUEST_EVENT_TRANSFER_FAILED_REASON = "transfer_failed_or_cancelled"
REQUEST_EVENT_ADMISSION_REASON = "admission_metadata_unavailable"
REQUEST_EVENT_ADMISSION_DECISION_REASON = "admission_decision_metadata_unavailable"
REQUEST_EVENT_PHYSICAL_REASON = "physical_block_metadata_unavailable"
REQUEST_EVENT_TP_REASON = "tp_contributor_metadata_unavailable"
REQUEST_EVENT_ACCOUNTING_REASON = "request_accounting_inconsistent"
REQUEST_EVENT_OUTCOME_REASON = "outcome_metadata_inconsistent"
REQUEST_EVENT_EPISODE_REASON = "request_episode_metadata_incomplete"
_CANONICAL_EXTERNAL_ID_RE = re.compile(r"^sc\|([^|]+)\|([^|]+)\|([^|]+)\|sample-(\d+)$")
_CANONICAL_REQUEST_ID_PREFIX_RE = re.compile(r"^sc\|([^|]+)\|([^|]+)\|[^|]+\|sample-$")

REQUEST_OUTCOME_FIELDS = (
    "preservation_outcome_basis",
    "preservation_request_ids_json",
    "preservation_request_outcome_details_json",
    "preservation_authoritative_request_count",
    "preservation_required_tokens",
    "preservation_reused_tokens",
    "preservation_loaded_tokens",
    "preservation_recomputed_tokens",
    "preservation_required_blocks",
    "preservation_reused_blocks",
    "preservation_loaded_blocks",
    "preservation_recomputed_blocks",
)
SAMPLE_IDENTITY_FIELDS = (
    "policy",
    "seed",
    "workload",
    "sub_workload",
    "request_rate",
    "parent_request_rate",
    "reference_count",
    "sample_index",
)


def _empty(value) -> bool:
    return value is None or str(value).strip() == ""


def _canonical_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "true":
            return True
        if value == "false":
            return False
    return None


def _mapping_fields(
    *,
    outcome: str = "unavailable",
    reason: str = REQUEST_EVENT_UNAVAILABLE_REASON,
    basis: str | None = None,
    request_ids: list[str] | None = None,
    details: list[dict] | None = None,
    authoritative_count: int | None = None,
    totals: dict[str, int] | None = None,
) -> dict[str, object]:
    totals = totals or {}
    return {
        "preservation_outcome": outcome,
        "preservation_reason": reason,
        "preservation_outcome_basis": basis,
        "preservation_request_ids_json": json.dumps(
            request_ids or [], separators=(",", ":"), sort_keys=True
        ),
        "preservation_request_outcome_details_json": json.dumps(
            details or [], separators=(",", ":"), sort_keys=True
        ),
        "preservation_authoritative_request_count": authoritative_count,
        "preservation_required_tokens": totals.get("required_tokens"),
        "preservation_reused_tokens": totals.get("reused_tokens"),
        "preservation_loaded_tokens": totals.get("loaded_tokens"),
        "preservation_recomputed_tokens": totals.get("recomputed_tokens"),
        "preservation_required_blocks": totals.get("required_blocks"),
        "preservation_reused_blocks": totals.get("reused_blocks"),
        "preservation_loaded_blocks": totals.get("loaded_blocks"),
        "preservation_recomputed_blocks": totals.get("recomputed_blocks"),
    }


def _nonnegative_int(value) -> int | None:
    """Return an actual non-negative integer; missing is not zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def validate_admission_decision_records(
    records: object,
    *,
    request_id: object,
    run_id: object,
) -> str | None:
    """Validate causal E04 records without accepting post-decision labels."""
    if not isinstance(records, list):
        return REQUEST_EVENT_ADMISSION_DECISION_REASON
    forbidden = {
        "future_reuse",
        "eventual_demand",
        "consumed",
        "discarded",
        "outcome",
        "target_output",
    }
    prior_sequence = 0
    for record in records:
        if not isinstance(record, dict) or forbidden.intersection(record):
            return REQUEST_EVENT_ADMISSION_DECISION_REASON
        if (
            record.get("schema") != "semantic-admission-decision-v1"
            or record.get("schema_version") != 1
            or record.get("event") != "admission_decision"
            or record.get("request_id") != request_id
            or record.get("engine_run_id") != run_id
            or record.get("mode")
            not in {"historical_frequency", "historical_frequency_cost_per_byte"}
            or record.get("causal_binding_complete") is not True
        ):
            return REQUEST_EVENT_ADMISSION_DECISION_REASON
        sequence = _nonnegative_int(record.get("decision_sequence"))
        observation_sequence = _nonnegative_int(record.get("observation_sequence"))
        if (
            sequence is None
            or sequence <= prior_sequence
            or observation_sequence is None
        ):
            return REQUEST_EVENT_ADMISSION_DECISION_REASON
        prior_sequence = sequence
        times = [
            _nonnegative_int(record.get(field))
            for field in (
                "observation_available_ns",
                "input_publication_ns",
                "consumer_decision_ns",
                "consumer_use_ns",
            )
        ]
        if None in times or times != sorted(times):
            return REQUEST_EVENT_ADMISSION_DECISION_REASON
        for field in (
            "model_revision",
            "tokenizer_revision",
            "adapter_namespace",
            "layout_namespace",
            "session_id",
            "generation",
            "scheduler_step",
            "key_chain_digest",
            "feature_provenance",
            "cost_sample_provenance",
            "candidate_config_hash",
        ):
            if record.get(field) in (None, ""):
                return REQUEST_EVENT_ADMISSION_DECISION_REASON
        config_hash = record["candidate_config_hash"]
        if not isinstance(config_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", config_hash
        ):
            return REQUEST_EVENT_ADMISSION_DECISION_REASON
        if record.get("action") not in {"admit", "reject"}:
            return REQUEST_EVENT_ADMISSION_DECISION_REASON
        if not isinstance(
            record.get("candidate_features_available"), bool
        ) or not isinstance(record.get("victim_features_available"), bool):
            return REQUEST_EVENT_ADMISSION_DECISION_REASON
    return None


def is_canonical_request_id_prefix(
    value, *, run_id: str | None = None, cell_id: str | None = None
) -> bool:
    if not isinstance(value, str):
        return False
    match = _CANONICAL_REQUEST_ID_PREFIX_RE.fullmatch(value)
    if match is None:
        return False
    return (run_id is None or match.group(1) == str(run_id)) and (
        cell_id is None or match.group(2) == str(cell_id)
    )


def expected_request_id_prefix(
    run_id: str | None,
    cell_id: str | None,
    workload: str | None,
    sub_workload: str | None,
) -> str | None:
    tag = sub_workload if workload == "mixed" and sub_workload else workload
    if _empty(run_id) or _empty(cell_id) or _empty(tag):
        return None
    return f"sc|{run_id}|{cell_id}|{tag}|sample-"


def _lookup_key_token(value) -> str | None:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _lookup_set_tokens(value) -> set[str] | None:
    if not isinstance(value, list):
        return None
    tokens: set[str] = set()
    for item in value:
        token = _lookup_key_token(item)
        if token is None or token in tokens:
            return None
        tokens.add(token)
    return tokens


def _subordinate_queried_tokens(lookup_episodes: list) -> set[str] | None:
    tokens: set[str] = set()
    for episode in lookup_episodes:
        groups = episode.get("groups") if isinstance(episode, dict) else None
        if not isinstance(groups, dict):
            return None
        for group in groups.values():
            if not isinstance(group, dict):
                return None
            queried = _lookup_set_tokens(group.get("queried"))
            if queried is None:
                return None
            tokens.update(queried)
    return tokens


def _validate_lookup_evidence(
    lookup_episodes: list,
    lookup_keys: list,
    required_tokens: int,
    *,
    allow_no_eligible: bool = False,
) -> str | None:
    declared = _lookup_set_tokens(lookup_keys)
    if declared is None:
        return REQUEST_EVENT_EPISODE_REASON
    queried_union: set[str] = set()
    all_empty_group_count = 0
    nonempty_lookup_evidence = False
    for episode in lookup_episodes:
        groups = episode.get("groups") if isinstance(episode, dict) else None
        if not isinstance(groups, dict):
            return REQUEST_EVENT_EPISODE_REASON
        if required_tokens > 0 and not groups:
            return REQUEST_EVENT_EPISODE_REASON
        for group in groups.values():
            if not isinstance(group, dict):
                return REQUEST_EVENT_EPISODE_REASON
            group_sets: dict[str, set[str]] = {}
            for name in ("eligible", "queried", "hits", "missing", "pending", "retry"):
                tokens = _lookup_set_tokens(group.get(name))
                if tokens is None:
                    return REQUEST_EVENT_EPISODE_REASON
                group_sets[name] = tokens
            group_all_empty = all(not tokens for tokens in group_sets.values())
            observed = group.get("observed")
            no_eligible = group.get("no_eligible")
            if observed is not True or not isinstance(no_eligible, bool):
                return REQUEST_EVENT_EPISODE_REASON
            if no_eligible != group_all_empty:
                return REQUEST_EVENT_EPISODE_REASON
            if group_all_empty:
                all_empty_group_count += 1
            nonempty_lookup_evidence = nonempty_lookup_evidence or not group_all_empty
            if not group_sets["queried"] <= group_sets["eligible"]:
                return REQUEST_EVENT_EPISODE_REASON
            if not group_sets["hits"] <= group_sets["queried"]:
                return REQUEST_EVENT_EPISODE_REASON
            if not group_sets["missing"] <= group_sets["queried"]:
                return REQUEST_EVENT_EPISODE_REASON
            if not group_sets["pending"] <= group_sets["queried"]:
                return REQUEST_EVENT_EPISODE_REASON
            if not group_sets["retry"] <= group_sets["queried"]:
                return REQUEST_EVENT_EPISODE_REASON
            terminal_sets = tuple(
                group_sets[name] for name in ("hits", "missing", "retry")
            )
            if any(
                left & right
                for index, left in enumerate(terminal_sets)
                for right in terminal_sets[index + 1 :]
            ):
                return REQUEST_EVENT_EPISODE_REASON
            if not group_sets["pending"] <= group_sets["hits"]:
                return REQUEST_EVENT_EPISODE_REASON
            if set().union(*terminal_sets) != group_sets["queried"]:
                return REQUEST_EVENT_EPISODE_REASON
            queried_union.update(group_sets["queried"])
    if all_empty_group_count:
        if nonempty_lookup_evidence and allow_no_eligible:
            return REQUEST_EVENT_EPISODE_REASON
        if not nonempty_lookup_evidence and not allow_no_eligible:
            return REQUEST_EVENT_EPISODE_REASON
    elif allow_no_eligible:
        return REQUEST_EVENT_EPISODE_REASON
    if not declared:
        declared = queried_union
    if queried_union != declared:
        return REQUEST_EVENT_EPISODE_REASON
    if required_tokens > 0 and not queried_union and not allow_no_eligible:
        return REQUEST_EVENT_EPISODE_REASON
    return None


def _validate_lookup_provenance(
    event: dict, declared: set[str]
) -> tuple[str | None, set[str], set[str], set[str]]:
    provenance: dict[str, set[str]] = {}
    for name in ("prior_evicted_keys", "prior_stored_keys", "cold_keys"):
        tokens = _lookup_set_tokens(event.get(name))
        if tokens is None:
            return REQUEST_EVENT_EPISODE_REASON, set(), set(), set()
        provenance[name] = tokens
    if any(not tokens <= declared for tokens in provenance.values()):
        return REQUEST_EVENT_EPISODE_REASON, set(), set(), set()
    if set().union(*provenance.values()) != declared:
        return REQUEST_EVENT_EPISODE_REASON, set(), set(), set()
    return (
        None,
        provenance["prior_evicted_keys"],
        provenance["prior_stored_keys"],
        provenance["cold_keys"],
    )


def _validate_transfer_evidence(
    event: dict, *, requires_load: bool, requires_store: bool
) -> tuple[str | None, dict[str, int], dict[str, int]]:
    """Require transfer records instead of trusting load/store totals."""
    statuses = event.get("transfer_job_statuses")
    counts = event.get("transfer_job_counts")
    if not isinstance(statuses, list) or not isinstance(counts, dict):
        return REQUEST_EVENT_TRANSFER_REASON, {}, {}
    load_count = _nonnegative_int(counts.get("load"))
    store_count = _nonnegative_int(counts.get("store"))
    if load_count is None or store_count is None:
        return REQUEST_EVENT_TRANSFER_REASON, {}, {}
    observed_counts = {"load": 0, "store": 0}
    observed_contributors = {"load": 0, "store": 0}
    expected_contributors = {"load": 0, "store": 0}
    job_ids: set[int] = set()
    for status in statuses:
        if not isinstance(status, dict):
            return REQUEST_EVENT_TRANSFER_REASON, {}, {}
        if status.get("is_prefetch") is True:
            return REQUEST_EVENT_TRANSFER_REASON, {}, {}
        direction = status.get("direction")
        if direction not in observed_counts:
            return REQUEST_EVENT_TRANSFER_REASON, {}, {}
        job_id = _nonnegative_int(status.get("job_id"))
        if job_id is None or job_id in job_ids:
            return REQUEST_EVENT_TRANSFER_REASON, {}, {}
        job_ids.add(job_id)
        contributors = _nonnegative_int(status.get("contributors"))
        expected = _nonnegative_int(status.get("expected_contributors"))
        if contributors is None or expected is None:
            return REQUEST_EVENT_TRANSFER_REASON, {}, {}
        if (
            status.get("status") != "success"
            or status.get("success") is not True
            or status.get("cancelled") is not False
        ):
            return REQUEST_EVENT_TRANSFER_FAILED_REASON, {}, {}
        if contributors != expected:
            return REQUEST_EVENT_TP_REASON, {}, {}
        observed_counts[direction] += 1
        observed_contributors[direction] += contributors
        expected_contributors[direction] += expected
    if observed_counts != {"load": load_count, "store": store_count}:
        return REQUEST_EVENT_TRANSFER_REASON, {}, {}
    if (
        _nonnegative_number(event.get("load_bytes")) is None
        or _nonnegative_number(event.get("store_bytes")) is None
    ):
        return REQUEST_EVENT_TRANSFER_REASON, {}, {}
    if requires_load and load_count == 0:
        return REQUEST_EVENT_TRANSFER_REASON, {}, {}
    if requires_store and store_count == 0:
        return REQUEST_EVENT_TRANSFER_REASON, {}, {}
    return None, observed_contributors, expected_contributors


def _validate_speculative_transfer_evidence(
    event: dict,
) -> tuple[str | None, dict[str, object] | None]:
    """Validate physical speculative work without treating it as demand reuse."""
    statuses = event.get("speculative_transfer_job_statuses")
    accounting = event.get("speculative_transfer_accounting")
    if statuses is None and accounting is None:
        # Legacy events remain readable only when speculative jobs were not
        # mixed into the demand-transfer channel.
        if any(
            isinstance(status, dict) and status.get("is_prefetch") is True
            for status in event.get("transfer_job_statuses", [])
        ):
            return REQUEST_EVENT_TRANSFER_REASON, None
        return None, None
    if not isinstance(statuses, list) or not isinstance(accounting, dict):
        return REQUEST_EVENT_TRANSFER_REASON, None
    demand_job_ids = {
        status.get("job_id")
        for status in event.get("transfer_job_statuses", [])
        if isinstance(status, dict)
    }
    job_ids: set[int] = set()
    successful_jobs = 0
    completed_bytes = 0
    actual_contributors = 0
    expected_contributors = 0
    device_time_s = 0.0
    for status in statuses:
        if (
            not isinstance(status, dict)
            or status.get("direction") != "load"
            or status.get("is_prefetch") is not True
        ):
            return REQUEST_EVENT_TRANSFER_REASON, None
        job_id = _nonnegative_int(status.get("job_id"))
        contributors = _nonnegative_int(status.get("contributors"))
        expected = _nonnegative_int(status.get("expected_contributors"))
        if (
            job_id is None
            or job_id in job_ids
            or job_id in demand_job_ids
            or contributors is None
            or expected is None
        ):
            return REQUEST_EVENT_TRANSFER_REASON, None
        job_ids.add(job_id)
        success = status.get("success")
        cancelled = status.get("cancelled")
        transfer_status = status.get("status")
        if success is True:
            transfer_bytes = _nonnegative_int(status.get("bytes"))
            transfer_time = _nonnegative_number(status.get("time_s"))
            if (
                transfer_status != "success"
                or cancelled is not False
                or transfer_bytes is None
                or transfer_bytes == 0
                or transfer_time is None
                or contributors != expected
            ):
                return REQUEST_EVENT_TRANSFER_REASON, None
            successful_jobs += 1
            completed_bytes += transfer_bytes
            device_time_s += transfer_time
        elif (
            success is not False
            or not isinstance(cancelled, bool)
            or not isinstance(transfer_status, str)
            or not transfer_status
        ):
            return REQUEST_EVENT_TRANSFER_REASON, None
        actual_contributors += contributors
        expected_contributors += expected
    integers = {
        name: _nonnegative_int(accounting.get(name))
        for name in (
            "job_count",
            "successful_job_count",
            "failed_job_count",
            "submitted_bytes",
            "completed_bytes",
            "consumed_bytes",
            "discarded_bytes",
            "released_reservation_bytes",
            "uncompleted_submitted_bytes",
            "unconsumed_completed_bytes",
            "actual_contributors",
            "expected_contributors",
        )
    }
    if any(value is None for value in integers.values()):
        return REQUEST_EVENT_TRANSFER_REASON, None
    declared_time = accounting.get("device_time_s")
    if successful_jobs:
        declared_time = _nonnegative_number(declared_time)
        if declared_time is None or not math.isclose(
            declared_time, device_time_s, rel_tol=1e-9, abs_tol=1e-12
        ):
            return REQUEST_EVENT_TRANSFER_REASON, None
    elif declared_time not in (None, 0, 0.0):
        return REQUEST_EVENT_TRANSFER_REASON, None
    if (
        integers["job_count"] != len(statuses)
        or integers["successful_job_count"] != successful_jobs
        or integers["failed_job_count"] != len(statuses) - successful_jobs
        or integers["completed_bytes"] != completed_bytes
        or integers["actual_contributors"] != actual_contributors
        or integers["expected_contributors"] != expected_contributors
        or integers["completed_bytes"]
        != integers["consumed_bytes"]
        + integers["discarded_bytes"]
        + integers["unconsumed_completed_bytes"]
        or integers["submitted_bytes"]
        != integers["completed_bytes"] + integers["uncompleted_submitted_bytes"]
        or integers["consumed_bytes"] + integers["discarded_bytes"]
        > integers["submitted_bytes"]
        or integers["discarded_bytes"] > integers["released_reservation_bytes"]
    ):
        return REQUEST_EVENT_TRANSFER_REASON, None
    attribution = event.get("prefetch_attribution")
    if not isinstance(attribution, dict):
        return REQUEST_EVENT_TRANSFER_REASON, None
    counts = event.get("prefetch_counts")
    provenance = attribution.get("provenance")
    config = provenance.get("config") if isinstance(provenance, dict) else None
    bytes_per_key = (
        _nonnegative_int(config.get("transfer_bytes_per_key"))
        if isinstance(config, dict)
        else None
    )
    if (
        attribution.get("transfer_jobs") != statuses
        or attribution.get("speculative_transfer") != accounting
        or not isinstance(counts, dict)
        or bytes_per_key is None
        or bytes_per_key == 0
    ):
        return REQUEST_EVENT_TRANSFER_REASON, None
    count_values = {
        name: _nonnegative_int((counts.get(name) or {}).get("blocks"))
        for name in ("attempted", "consumed", "discarded")
    }
    if (
        any(value is None for value in count_values.values())
        or count_values["attempted"] * bytes_per_key != integers["submitted_bytes"]
        or count_values["consumed"] * bytes_per_key != integers["consumed_bytes"]
        or count_values["discarded"] * bytes_per_key
        != integers["released_reservation_bytes"]
        or attribution.get("prefetch_h2d_bytes")
        != (integers["completed_bytes"] if successful_jobs else None)
    ):
        return REQUEST_EVENT_TRANSFER_REASON, None
    return None, dict(accounting)


def _consumed_speculative_destination_refs(
    event: dict,
    *,
    missing_external_refs: set[tuple[int, int]],
) -> tuple[str | None, set[tuple[int, int]]]:
    """Prove external blocks satisfied by a completed GPU splice.

    Demand loads and speculative loads use separate worker channels.  A
    consumed full-cover prefetch therefore has an external admission block
    but no demand-load job.  Accept that distinction only when the event
    carries exact key-to-destination identity and a completed splice fence.
    """
    if not missing_external_refs:
        return None, set()
    attribution = event.get("prefetch_attribution")
    physical_table = event.get("physical_block_table_provenance")
    counts = event.get("prefetch_counts")
    if (
        not isinstance(attribution, dict)
        or attribution.get("status") != "hint_consumed"
        or not isinstance(physical_table, dict)
        or physical_table.get("complete") is not True
        or not isinstance(counts, dict)
    ):
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    consumed = counts.get("consumed")
    consumed_blocks = (
        _nonnegative_int(consumed.get("blocks")) if isinstance(consumed, dict) else None
    )
    intersection_keys = attribution.get("intersection_keys")
    required_keys = attribution.get("required_keys")
    source_map = attribution.get("source_key_to_gpu_block")
    destination_map = attribution.get("destination_key_to_gpu_block")
    residency_proof = attribution.get("cpu_residency_proof")
    if (
        consumed_blocks is None
        or consumed_blocks != len(missing_external_refs)
        or not isinstance(intersection_keys, list)
        or len(intersection_keys) != consumed_blocks
        or any(not isinstance(key, str) or not key for key in intersection_keys)
        or len(intersection_keys) != len(set(intersection_keys))
        or not isinstance(required_keys, list)
        or any(not isinstance(key, str) or not key for key in required_keys)
        or not set(intersection_keys) <= set(required_keys)
        or not isinstance(source_map, dict)
        or set(source_map) != set(intersection_keys)
        or not isinstance(destination_map, dict)
        or set(destination_map) != set(intersection_keys)
        or not isinstance(residency_proof, dict)
        or set(residency_proof) != set(intersection_keys)
    ):
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    try:
        source_ids = [int(source_map[key]) for key in intersection_keys]
        destination_ids = [int(destination_map[key]) for key in intersection_keys]
    except (TypeError, ValueError):
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    if (
        any(block_id <= 0 for block_id in (*source_ids, *destination_ids))
        or len(source_ids) != len(set(source_ids))
        or len(destination_ids) != len(set(destination_ids))
        or set(source_ids) & set(destination_ids)
    ):
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    transfer_jobs = attribution.get("transfer_jobs")
    transferred_keys: set[str] = set()
    staging_destination_ids: set[int] = set()
    if not isinstance(transfer_jobs, list) or not transfer_jobs:
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    for job in transfer_jobs:
        keys = job.get("keys") if isinstance(job, dict) else None
        destinations = (
            job.get("destination_block_ids") if isinstance(job, dict) else None
        )
        if (
            not isinstance(job, dict)
            or job.get("status") != "success"
            or not isinstance(keys, list)
            or any(not isinstance(key, str) or not key for key in keys)
            or not isinstance(destinations, list)
        ):
            return REQUEST_EVENT_PHYSICAL_REASON, set()
        transferred_keys.update(keys)
        for destination in destinations:
            if (
                not isinstance(destination, (list, tuple))
                or len(destination) != 2
                or destination[0] != 0
                or isinstance(destination[1], bool)
                or not isinstance(destination[1], int)
                or destination[1] <= 0
            ):
                return REQUEST_EVENT_PHYSICAL_REASON, set()
            staging_destination_ids.add(destination[1])
    if transferred_keys != set(intersection_keys) or staging_destination_ids != set(
        source_ids
    ):
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    # The connector's splice path is deliberately restricted to one KV group.
    # Require the retained table to prove that same scope instead of guessing
    # a group for the destination IDs.
    table_groups = physical_table.get("group_block_ids")
    if not isinstance(table_groups, list) or len(table_groups) != 1:
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    speculative_refs = {(0, block_id) for block_id in destination_ids}
    if speculative_refs != missing_external_refs:
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    if (
        attribution.get("splice_acknowledged") is not True
        or _nonnegative_int(attribution.get("splice_fence_ns")) is None
        or not isinstance(attribution.get("splice_ack_id"), str)
        or not attribution["splice_ack_id"]
    ):
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    fence_evidence = attribution.get("splice_fence_evidence")
    if not isinstance(fence_evidence, list) or not fence_evidence:
        return REQUEST_EVENT_PHYSICAL_REASON, set()
    for evidence in fence_evidence:
        if (
            not isinstance(evidence, dict)
            or evidence.get("completion_event_satisfied") is not True
            or evidence.get("copied_on_cuda") is not True
            or evidence.get("same_default_stream") is not True
            or evidence.get("issued_before_base_bind_and_forward") is not True
            or _nonnegative_int(evidence.get("destination_block_count"))
            != consumed_blocks
            or _nonnegative_int(evidence.get("source_block_count")) != consumed_blocks
        ):
            return REQUEST_EVENT_PHYSICAL_REASON, set()
    return None, speculative_refs


def _json_list(value) -> list | None:
    return value if isinstance(value, list) else None


def _episode_ids(
    episodes: list, field: str, *, allow_repeats: bool = False
) -> tuple[list[int], str | None]:
    ids: list[int] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            return [], REQUEST_EVENT_EPISODE_REASON
        episode_id = _nonnegative_int(episode.get(field))
        if episode_id is None or (not allow_repeats and episode_id in ids):
            return [], REQUEST_EVENT_EPISODE_REASON
        if episode_id not in ids:
            ids.append(episode_id)
    return ids, None


def _block_ids(value) -> list[tuple]:
    """Normalize JSON physical references without treating malformed data as empty."""
    if not isinstance(value, list):
        raise TypeError("physical block identity is not a list")
    normalized: list[tuple] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise TypeError("physical block identity is not group-qualified")
        if any(isinstance(part, bool) or not isinstance(part, int) for part in item):
            raise TypeError("malformed physical block identity")
        ref = tuple(item)
        if ref[0] < 0 or ref[1] < 0 or ref[2] <= 0:
            raise ValueError("invalid group-qualified physical block identity")
        if ref in normalized:
            raise ValueError("duplicate physical block identity")
        normalized.append(ref)
    return normalized


def _validate_physical_block_table_provenance(
    event: dict,
    *,
    admission_episodes: list[dict],
    required_refs: list[tuple],
    local_refs: list[tuple],
    external_refs: list[tuple],
    recomputed_refs: list[tuple],
    scheduled_refs: list[tuple],
    transfer_refs: tuple[list[tuple], ...],
) -> str | None:
    provenance = event.get("physical_block_table_provenance")
    if provenance is None:
        return None
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema") != "semantic-physical-block-table-v1"
        or provenance.get("request_id") != event.get("request_id")
        or provenance.get("external_request_id") != event.get("external_request_id")
        or provenance.get("request_generation") != event.get("generation")
        or provenance.get("complete") is not True
        or provenance.get("errors") != []
    ):
        return REQUEST_EVENT_PHYSICAL_REASON
    lifetime_id = provenance.get("lifetime_id")
    event_observed_at_ns = _nonnegative_int(event.get("connector_observed_ns"))
    lifetime_observed_at_ns = _nonnegative_int(
        provenance.get("lifetime_observed_at_ns")
    )
    table_generation = _nonnegative_int(provenance.get("table_generation"))
    tokens_per_block = provenance.get("tokens_per_block")
    groups = provenance.get("group_block_ids")
    updates = provenance.get("updates")
    if (
        not isinstance(lifetime_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", lifetime_id) is None
        or event_observed_at_ns is None
        or lifetime_observed_at_ns is None
        or lifetime_observed_at_ns != event_observed_at_ns
        or table_generation is None
        or not isinstance(tokens_per_block, list)
        or not tokens_per_block
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in tokens_per_block
        )
        or not isinstance(groups, list)
        or len(groups) != len(tokens_per_block)
        or not isinstance(updates, list)
        or not updates
    ):
        return REQUEST_EVENT_PHYSICAL_REASON
    expected_lifetime_id = hashlib.sha256(
        json.dumps(
            [
                event.get("request_id"),
                event.get("external_request_id"),
                event.get("generation"),
                event_observed_at_ns,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if lifetime_id != expected_lifetime_id:
        return REQUEST_EVENT_PHYSICAL_REASON
    normalized_groups: list[list[int]] = []
    for group in groups:
        if (
            not isinstance(group, list)
            or any(
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or block_id < 0
                for block_id in group
            )
            or len(group) != len(set(group))
        ):
            return REQUEST_EVENT_PHYSICAL_REASON
        normalized_groups.append(group)
    table_digest = hashlib.sha256(
        json.dumps(normalized_groups, separators=(",", ":")).encode()
    ).hexdigest()
    if provenance.get("table_digest") != table_digest:
        return REQUEST_EVENT_PHYSICAL_REASON

    admissions_by_batch = {}
    for admission in admission_episodes:
        if not isinstance(admission, dict):
            return REQUEST_EVENT_PHYSICAL_REASON
        batch_id = _nonnegative_int(admission.get("scheduler_batch_id"))
        episode_id = _nonnegative_int(admission.get("episode_id"))
        scheduled_tokens = _nonnegative_int(admission.get("scheduled_tokens"))
        scheduled_prompt_tokens = _nonnegative_int(
            admission.get("prompt_tokens_locally_executed")
        )
        if (
            batch_id is None
            or batch_id in admissions_by_batch
            or episode_id is None
            or scheduled_tokens is None
            or scheduled_prompt_tokens is None
        ):
            return REQUEST_EVENT_PHYSICAL_REASON
        admissions_by_batch[batch_id] = (
            episode_id,
            scheduled_tokens,
            scheduled_prompt_tokens,
        )

    prior_sequence = -1
    prior_batch = -1
    prior_observed_at_ns = lifetime_observed_at_ns
    current_groups: list[list[int]] | None = None
    current_generation: int | None = None
    current_episode: int | None = None
    first_table_by_episode: dict[int, list[set[int]]] = {}
    last_start_by_episode: dict[int, int] = {}
    all_table_blocks = [set() for _ in normalized_groups]
    expected_scheduled_refs: set[tuple[int, int, int]] = set()
    observed_batches = set()
    for index, update in enumerate(updates):
        if not isinstance(update, dict):
            return REQUEST_EVENT_PHYSICAL_REASON
        mode = update.get("mode")
        scheduler_batch_id = _nonnegative_int(update.get("scheduler_batch_id"))
        episode_id = _nonnegative_int(update.get("episode_id"))
        sequence = _nonnegative_int(update.get("sequence"))
        start_tokens = _nonnegative_int(update.get("start_tokens"))
        observed_at_ns = _nonnegative_int(update.get("observed_at_ns"))
        generation = _nonnegative_int(update.get("table_generation"))
        scheduled_tokens = _nonnegative_int(update.get("scheduled_tokens"))
        scheduled_prompt_tokens = _nonnegative_int(
            update.get("scheduled_prompt_tokens")
        )
        update_tokens_per_block = update.get("tokens_per_block")
        partial_tail = update.get("partial_tail_tokens")
        scheduled_block_ids = update.get("scheduled_block_ids")
        before = update.get("group_lengths_before")
        after = update.get("group_lengths_after")
        update_groups = update.get("group_block_ids")
        if (
            mode
            not in {
                "scheduled_new_full",
                "cached_append",
                "cached_no_new_blocks",
                "cached_resumed_replace",
            }
            or scheduler_batch_id is None
            or episode_id is None
            or scheduler_batch_id <= prior_batch
            or sequence is None
            or sequence <= prior_sequence
            or start_tokens is None
            or observed_at_ns is None
            or observed_at_ns < prior_observed_at_ns
            or generation is None
            or scheduled_tokens is None
            or scheduled_prompt_tokens is None
            or scheduled_prompt_tokens > scheduled_tokens
            or update_tokens_per_block != tokens_per_block
            or not isinstance(partial_tail, list)
            or len(partial_tail) != len(normalized_groups)
            or partial_tail
            != [
                start_tokens % group_block_size for group_block_size in tokens_per_block
            ]
            or not isinstance(scheduled_block_ids, list)
            or not isinstance(after, list)
            or len(after) != len(normalized_groups)
            or any(_nonnegative_int(value) is None for value in after)
            or (
                before is not None
                and (
                    not isinstance(before, list)
                    or len(before) != len(normalized_groups)
                    or any(_nonnegative_int(value) is None for value in before)
                )
            )
            or update.get("table_digest") is None
            or not isinstance(update_groups, list)
            or len(update_groups) != len(normalized_groups)
        ):
            return REQUEST_EVENT_PHYSICAL_REASON
        admission = admissions_by_batch.get(scheduler_batch_id)
        if admission != (episode_id, scheduled_tokens, scheduled_prompt_tokens):
            return REQUEST_EVENT_PHYSICAL_REASON
        parsed_update_groups: list[list[int]] = []
        for group in update_groups:
            if (
                not isinstance(group, list)
                or any(
                    isinstance(block_id, bool)
                    or not isinstance(block_id, int)
                    or block_id < 0
                    for block_id in group
                )
                or len(group) != len(set(group))
            ):
                return REQUEST_EVENT_PHYSICAL_REASON
            parsed_update_groups.append(group)
        update_digest = hashlib.sha256(
            json.dumps(parsed_update_groups, separators=(",", ":")).encode()
        ).hexdigest()
        if update.get("table_digest") != update_digest or after != [
            len(group) for group in parsed_update_groups
        ]:
            return REQUEST_EVENT_PHYSICAL_REASON

        if index == 0:
            if mode != "scheduled_new_full" or before is not None or generation != 0:
                return REQUEST_EVENT_PHYSICAL_REASON
        elif (
            current_groups is None
            or current_generation is None
            or before != [len(group) for group in current_groups]
        ) or mode == "scheduled_new_full":
            return REQUEST_EVENT_PHYSICAL_REASON
        elif mode == "cached_resumed_replace":
            if (
                generation != current_generation + 1
                or current_episode is None
                or episode_id <= current_episode
            ):
                return REQUEST_EVENT_PHYSICAL_REASON
        elif generation != current_generation or episode_id != current_episode:
            return REQUEST_EVENT_PHYSICAL_REASON
        elif mode == "cached_no_new_blocks":
            if parsed_update_groups != current_groups:
                return REQUEST_EVENT_PHYSICAL_REASON
        elif any(
            len(new_group) < len(old_group) or new_group[: len(old_group)] != old_group
            for old_group, new_group in zip(
                current_groups, parsed_update_groups, strict=True
            )
        ):
            return REQUEST_EVENT_PHYSICAL_REASON

        prior_start = last_start_by_episode.get(episode_id)
        if prior_start is not None and start_tokens < prior_start:
            return REQUEST_EVENT_PHYSICAL_REASON
        last_start_by_episode[episode_id] = start_tokens
        calculated_scheduled_refs = []
        for group_idx, (group_block_size, group) in enumerate(
            zip(tokens_per_block, parsed_update_groups, strict=True)
        ):
            start_block = start_tokens // group_block_size
            end_block = (
                start_block
                if scheduled_prompt_tokens == 0
                else (start_tokens + scheduled_prompt_tokens + group_block_size - 1)
                // group_block_size
            )
            computed_prefix_blocks = start_tokens + group_block_size - 1
            if computed_prefix_blocks // group_block_size > len(
                group
            ) or end_block > len(group):
                return REQUEST_EVENT_PHYSICAL_REASON
            calculated_scheduled_refs.extend(
                (episode_id, group_idx, block_id)
                for block_id in group[start_block:end_block]
                if block_id != 0
            )
        try:
            declared_scheduled_refs = _block_ids(scheduled_block_ids)
        except (TypeError, ValueError):
            return REQUEST_EVENT_PHYSICAL_REASON
        if set(declared_scheduled_refs) != set(calculated_scheduled_refs):
            return REQUEST_EVENT_PHYSICAL_REASON
        expected_scheduled_refs.update(calculated_scheduled_refs)

        if episode_id not in first_table_by_episode:
            first_table_by_episode[episode_id] = [
                set(group) for group in parsed_update_groups
            ]
        for group_idx, group in enumerate(parsed_update_groups):
            all_table_blocks[group_idx].update(group)
        current_groups = parsed_update_groups
        current_generation = generation
        current_episode = episode_id
        observed_batches.add(scheduler_batch_id)
        prior_sequence = sequence
        prior_batch = scheduler_batch_id
        prior_observed_at_ns = observed_at_ns
    if (
        updates[-1].get("table_digest") != table_digest
        or updates[-1].get("group_lengths_after")
        != [len(group) for group in normalized_groups]
        or _nonnegative_int(updates[-1].get("table_generation")) != table_generation
        or current_groups != normalized_groups
        or observed_batches != set(admissions_by_batch)
    ):
        return REQUEST_EVENT_PHYSICAL_REASON

    for refs in (required_refs, local_refs, external_refs):
        for episode_id, group_idx, block_id in refs:
            episode_table = first_table_by_episode.get(episode_id)
            if (
                episode_table is None
                or group_idx >= len(episode_table)
                or block_id not in episode_table[group_idx]
            ):
                return REQUEST_EVENT_PHYSICAL_REASON
    if (
        set(recomputed_refs) != expected_scheduled_refs
        or set(scheduled_refs) != expected_scheduled_refs
    ):
        return REQUEST_EVENT_PHYSICAL_REASON
    for refs in transfer_refs:
        for _job_id, group_idx, block_id in refs:
            if (
                group_idx >= len(all_table_blocks)
                or block_id not in all_table_blocks[group_idx]
            ):
                return REQUEST_EVENT_PHYSICAL_REASON
    return None


def _unavailable_event_detail(event: dict, reason: str) -> dict:
    request_id = event.get("external_request_id")
    detail = {"request_id": request_id, "outcome": "unavailable", "reason": reason}
    if isinstance(event.get("sample_index"), int):
        detail["sample_index"] = event["sample_index"]
    return detail


def _validate_server_timing(event: dict) -> tuple[dict | None, str | None]:
    timing = event.get("server_timing")
    if timing is None:
        return None, None
    if not isinstance(timing, dict) or timing.get("schema") != (
        "semantic-server-request-timing-v1"
    ):
        return None, "server_timing_invalid"
    if timing.get("clock_domain") != "host_boot_monotonic_ns":
        return None, "server_timing_invalid"
    complete = timing.get("complete")
    if not isinstance(complete, bool) or not isinstance(
        timing.get("ordering_valid"), bool
    ):
        return None, "server_timing_invalid"
    names = (
        "arrival_aligned_monotonic_ns",
        "service_start_ns",
        "first_forward_ns",
        "completion_ns",
    )
    values = [_nonnegative_int(timing.get(name)) for name in names]
    if complete:
        if any(value is None for value in values) or values != sorted(values):
            return None, "server_timing_invalid"
        if timing.get("ordering_valid") is not True:
            return None, "server_timing_invalid"
        expected = {
            "queue_delay_ns": values[1] - values[0],
            "service_to_first_forward_ns": values[2] - values[1],
            "service_time_ns": values[3] - values[1],
            "completion_after_first_forward_ns": values[3] - values[2],
        }
        if any(timing.get(name) != value for name, value in expected.items()):
            return None, "server_timing_invalid"
    alignment_uncertainty = timing.get("arrival_alignment_uncertainty_ns")
    if (
        alignment_uncertainty is not None
        and _nonnegative_int(alignment_uncertainty) is None
    ):
        return None, "server_timing_invalid"
    return dict(timing), None


def _validate_prefill_cuda_events(event: dict) -> tuple[list[dict], str | None]:
    records = event.get("prefill_cuda_events", [])
    if not isinstance(records, list):
        return [], "prefill_cuda_timing_invalid"
    for record in records:
        if not isinstance(record, dict) or record.get("schema") not in {
            "semantic-bounded-prefill-cuda-v1",
            "semantic-bounded-prefill-cuda-v2",
            "semantic-bounded-prefill-cuda-v3",
        }:
            return [], "prefill_cuda_timing_invalid"
        started = _nonnegative_int(record.get("started_at_ns"))
        completed = _nonnegative_int(record.get("completed_at_ns"))
        device_time = record.get("device_time_s")
        if (
            started is None
            or completed is None
            or completed < started
            or isinstance(device_time, bool)
            or not isinstance(device_time, (int, float))
            or not math.isfinite(float(device_time))
            or float(device_time) < 0
        ):
            return [], "prefill_cuda_timing_invalid"
        if record.get("attribution_eligible") is True and not (
            record.get("singleton_request") is True
            and record.get("all_scheduled_tokens_are_prompt") is True
            and len(record.get("request_ids", [])) == 1
            and len(record.get("execution_attribution", [])) == 1
        ):
            return [], "prefill_cuda_timing_invalid"
        if (
            record.get("schema") == "semantic-bounded-prefill-cuda-v1"
            and record.get("attribution_eligible") is True
            and record.get("transfer_free") is not True
        ):
            return [], "prefill_cuda_timing_invalid"
        if record.get("schema") in {
            "semantic-bounded-prefill-cuda-v2",
            "semantic-bounded-prefill-cuda-v3",
        }:
            request_ids = record.get("request_ids")
            attribution = record.get("execution_attribution")
            categories = record.get("transfer_categories")
            if (
                not isinstance(request_ids, list)
                or any(not isinstance(value, str) or not value for value in request_ids)
                or len(request_ids) != len(set(request_ids))
                or not isinstance(attribution, list)
                or any(not isinstance(value, dict) for value in attribution)
                or not isinstance(categories, list)
                or any(not isinstance(value, str) or not value for value in categories)
                or len(categories) != len(set(categories))
                or record.get("tensor_parallel_size") != 1
                or record.get("host_clock_domain") != "host_boot_monotonic_ns"
                or record.get("device_time_source")
                != "cuda_event_current_model_runner_stream_elapsed"
                or record.get("synchronization_overhead")
                != "end_event_synchronize; attribution_cells_only"
                or (
                    record.get("schema") == "semantic-bounded-prefill-cuda-v3"
                    and (
                        _nonnegative_int(record.get("execution_sequence")) is None
                        or record.get("execution_sequence") == 0
                    )
                )
                or record.get("attribution_reason")
                not in {
                    "eligible_singleton_prompt_batch",
                    "non_singleton_or_unmatched_batch",
                    "decode_only_batch",
                    "mixed_prompt_decode_batch",
                    "transfer_or_splice_in_batch",
                    "eligible_singleton_prompt_batch_with_transfer",
                }
            ):
                return [], "prefill_cuda_timing_invalid"
            for value in attribution:
                scheduled = _nonnegative_int(value.get("scheduled_tokens"))
                prompt = _nonnegative_int(value.get("prompt_tokens"))
                if (
                    not isinstance(value.get("request_id"), str)
                    or not isinstance(value.get("external_request_id"), str)
                    or scheduled is None
                    or prompt is None
                    or prompt > scheduled
                    or _nonnegative_int(value.get("scheduler_batch_id")) is None
                ):
                    return [], "prefill_cuda_timing_invalid"
            eligible = record.get("attribution_eligible") is True
            expected_eligible = bool(
                len(request_ids) == 1
                and len(attribution) == 1
                and attribution[0].get("request_id") == request_ids[0]
                and attribution[0].get("prompt_tokens", 0) > 0
                and attribution[0].get("prompt_tokens")
                == attribution[0].get("scheduled_tokens")
            )
            exact_request = bool(
                len(request_ids) == 1
                and len(attribution) == 1
                and attribution[0].get("request_id") == request_ids[0]
            )
            prompt_tokens = (
                attribution[0].get("prompt_tokens", 0) if exact_request else 0
            )
            scheduled_tokens = (
                attribution[0].get("scheduled_tokens", 0) if exact_request else 0
            )
            if not exact_request:
                expected_reason = "non_singleton_or_unmatched_batch"
            elif prompt_tokens <= 0:
                expected_reason = "decode_only_batch"
            elif prompt_tokens != scheduled_tokens:
                expected_reason = "mixed_prompt_decode_batch"
            elif record.get("transfer_free") is not True:
                expected_reason = "eligible_singleton_prompt_batch_with_transfer"
            else:
                expected_reason = "eligible_singleton_prompt_batch"
            if (
                record.get("transfer_free") is not (not categories)
                or eligible != expected_eligible
                or record.get("attribution_reason") != expected_reason
            ):
                return [], "prefill_cuda_timing_invalid"
    if any(
        record.get("schema")
        in {
            "semantic-bounded-prefill-cuda-v2",
            "semantic-bounded-prefill-cuda-v3",
        }
        for record in records
    ) and isinstance(event.get("admission_episodes"), list):
        expected_prompt_batches = {
            episode.get("scheduler_batch_id"): episode.get(
                "prompt_tokens_locally_executed"
            )
            for episode in event["admission_episodes"]
            if isinstance(episode, dict)
            and _nonnegative_int(episode.get("prompt_tokens_locally_executed"))
            not in {None, 0}
        }
        observed_prompt_batches = {}
        for record in records:
            if record.get("schema") not in {
                "semantic-bounded-prefill-cuda-v2",
                "semantic-bounded-prefill-cuda-v3",
            }:
                continue
            for attribution in record.get("execution_attribution", []):
                if (
                    attribution.get("request_id") != event.get("request_id")
                    or attribution.get("prompt_tokens") == 0
                ):
                    continue
                batch_id = attribution.get("scheduler_batch_id")
                if batch_id in observed_prompt_batches:
                    return [], "prefill_cuda_timing_invalid"
                observed_prompt_batches[batch_id] = attribution.get("prompt_tokens")
        if observed_prompt_batches != expected_prompt_batches:
            return [], "prefill_cuda_timing_invalid"
    return [dict(record) for record in records], None


def summarize_prefill_cuda_work(events: list[dict]) -> dict[str, object]:
    """Summarize bounded CUDA work without treating partial coverage as total.

    CUDA elapsed intervals are request-attributable only for singleton,
    prompt-only TP=1 executions. Transfer/splice categories remain attached
    because the current-stream interval may include a wait on separately
    measured DMA and therefore must not be added to transfer time as if the
    two were disjoint. The observed subtotal is useful diagnostic evidence,
    but ``device_time_s`` is available only when eligible intervals cover
    every authoritative locally executed prompt token in the cell.
    """
    rows: list[dict[str, object]] = []
    seen_events: set[tuple[object, ...]] = set()
    observed_times: list[float] = []
    eligible_prompt_total = 0
    actual_prompt_total = 0
    eligible_event_total = 0
    observed_event_total = 0
    reason_counts: dict[str, int] = {}
    invalid_reason = None
    for event in events:
        if not isinstance(event, dict):
            invalid_reason = "prefill_cuda_event_container_invalid"
            break
        actual_prompt = _nonnegative_int(
            event.get("actual_prompt_tokens_locally_executed")
        )
        if actual_prompt is None:
            invalid_reason = "authoritative_prompt_tokens_unavailable"
            break
        records, error = _validate_prefill_cuda_events(event)
        if error is not None:
            invalid_reason = error
            break
        request_internal_id = event.get("request_id")
        request_external_id = event.get("external_request_id")
        request_eligible_tokens = 0
        request_device_times: list[float] = []
        request_eligible_events = 0
        request_reasons: dict[str, int] = {}
        for record in records:
            attribution = record.get("execution_attribution", [])
            scheduler_batches = tuple(
                item.get("scheduler_batch_id")
                for item in attribution
                if isinstance(item, dict)
            )
            identity = (
                tuple(record.get("request_ids", [])),
                scheduler_batches,
                record.get("started_at_ns"),
                record.get("completed_at_ns"),
            )
            if identity in seen_events:
                if record.get("attribution_eligible") is True:
                    invalid_reason = "prefill_cuda_event_duplicated"
                    break
                first_observation = False
            else:
                seen_events.add(identity)
                observed_event_total += 1
                first_observation = True
            reason = str(record.get("attribution_reason") or "legacy_unclassified")
            request_reasons[reason] = request_reasons.get(reason, 0) + 1
            if first_observation:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if record.get("attribution_eligible") is not True:
                continue
            item = attribution[0]
            if item.get("request_id") != request_internal_id:
                invalid_reason = "prefill_cuda_request_identity_mismatch"
                break
            prompt_tokens = _nonnegative_int(item.get("prompt_tokens"))
            if prompt_tokens is None or prompt_tokens <= 0:
                invalid_reason = "prefill_cuda_prompt_token_count_invalid"
                break
            elapsed = float(record["device_time_s"])
            request_eligible_tokens += prompt_tokens
            request_device_times.append(elapsed)
            eligible_prompt_total += prompt_tokens
            observed_times.append(elapsed)
            request_eligible_events += 1
            eligible_event_total += 1
        if invalid_reason is not None:
            break
        if request_eligible_tokens > actual_prompt:
            invalid_reason = "prefill_cuda_prompt_coverage_exceeds_authoritative"
            break
        complete = request_eligible_tokens == actual_prompt
        rows.append(
            {
                "request_id": request_external_id,
                "internal_request_id": request_internal_id,
                "actual_prompt_tokens_locally_executed": actual_prompt,
                "eligible_prompt_tokens": request_eligible_tokens,
                "eligible_event_count": request_eligible_events,
                "observed_event_count": len(records),
                "coverage_fraction": (
                    request_eligible_tokens / actual_prompt
                    if actual_prompt > 0
                    else None
                ),
                "complete": complete,
                "device_time_s": (
                    sum(request_device_times)
                    if request_device_times
                    else 0.0
                    if complete and actual_prompt == 0
                    else None
                ),
                "observed_device_time_s": (
                    sum(request_device_times) if request_device_times else None
                ),
                "ineligibility_reasons": request_reasons,
            }
        )
        actual_prompt_total += actual_prompt
    if invalid_reason is not None:
        return {
            "schema": "semantic-bounded-prefill-cuda-accounting-v1",
            "available": False,
            "reason": invalid_reason,
            "device_time_s": None,
            "observed_device_time_s": None,
            "per_request": [],
        }
    covered_requests = sum(row["complete"] is True for row in rows)
    requests_with_work = sum(
        int(row["actual_prompt_tokens_locally_executed"]) > 0 for row in rows
    )
    complete = bool(
        rows
        and covered_requests == len(rows)
        and eligible_prompt_total == actual_prompt_total
    )
    return {
        "schema": "semantic-bounded-prefill-cuda-accounting-v1",
        "available": complete,
        "reason": (
            "complete_singleton_prompt_cuda_coverage"
            if complete
            else "incomplete_prefill_cuda_prompt_coverage"
        ),
        "request_count": len(rows),
        "covered_request_count": covered_requests,
        "requests_with_prompt_work": requests_with_work,
        "actual_prompt_tokens_locally_executed": actual_prompt_total,
        "eligible_prompt_tokens": eligible_prompt_total,
        "prompt_token_coverage_fraction": (
            eligible_prompt_total / actual_prompt_total
            if actual_prompt_total > 0
            else None
        ),
        "observed_event_count": observed_event_total,
        "eligible_event_count": eligible_event_total,
        "device_time_s": sum(observed_times) if complete else None,
        "observed_device_time_s": sum(observed_times) if observed_times else None,
        "ineligibility_reasons": reason_counts,
        "attribution_scope": (
            "TP1 singleton prompt-only model-runner current CUDA stream; "
            "transfer/splice categories retained and overlap not disambiguated"
        ),
        "synchronization_overhead": "one CUDA end-event synchronize per model step",
        "per_request": rows,
    }


def summarize_gpu_model_runner_work(events: list[dict]) -> dict[str, object]:
    """Reconcile cell-level TP1 model-runner intervals without splitting batches."""
    expected_batches: dict[tuple[str, int], tuple[int, int]] = {}
    records_by_sequence: dict[int, dict] = {}
    owners_by_sequence: dict[int, set[str]] = {}
    appearances: set[tuple[int, str]] = set()
    invalid_reason = None
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("request_id"), str):
            invalid_reason = "gpu_model_runner_request_identity_invalid"
            break
        request_id = event["request_id"]
        for episode in event.get("admission_episodes", []):
            if not isinstance(episode, dict):
                invalid_reason = "gpu_model_runner_expected_batch_invalid"
                break
            batch_id = _nonnegative_int(episode.get("scheduler_batch_id"))
            scheduled = _nonnegative_int(episode.get("scheduled_tokens"))
            prompt = _nonnegative_int(episode.get("prompt_tokens_locally_executed"))
            key = (request_id, batch_id) if batch_id is not None else None
            if (
                key is None
                or scheduled is None
                or prompt is None
                or prompt > scheduled
                or key in expected_batches
            ):
                invalid_reason = "gpu_model_runner_expected_batch_invalid"
                break
            expected_batches[key] = (scheduled, prompt)
        if invalid_reason is not None:
            break
        records, error = _validate_prefill_cuda_events(event)
        if error is not None:
            invalid_reason = error
            break
        for record in records:
            if record.get("schema") != "semantic-bounded-prefill-cuda-v3":
                continue
            sequence = int(record["execution_sequence"])
            appearance = (sequence, request_id)
            if appearance in appearances:
                invalid_reason = "gpu_model_runner_event_duplicated"
                break
            appearances.add(appearance)
            prior = records_by_sequence.get(sequence)
            if prior is not None and prior != record:
                invalid_reason = "gpu_model_runner_sequence_conflict"
                break
            records_by_sequence[sequence] = dict(record)
            owners_by_sequence.setdefault(sequence, set()).add(request_id)
        if invalid_reason is not None:
            break
    if invalid_reason is not None:
        return {
            "schema": "semantic-gpu-model-runner-work-accounting-v1",
            "available": False,
            "reason": invalid_reason,
            "device_time_s": None,
            "observed_device_time_s": None,
        }
    observed_times = [
        float(record["device_time_s"]) for record in records_by_sequence.values()
    ]
    if not records_by_sequence:
        return {
            "schema": "semantic-gpu-model-runner-work-accounting-v1",
            "available": False,
            "reason": "gpu_model_runner_events_missing",
            "device_time_s": None,
            "observed_device_time_s": None,
            "execution_count": 0,
        }
    sequences = sorted(records_by_sequence)
    complete_sequences = sequences == list(range(1, len(sequences) + 1))
    observed_batches: dict[tuple[str, int], tuple[int, int]] = {}
    batch_kind_counts = {"prompt_only": 0, "decode_only": 0, "mixed": 0}
    transfer_category_counts: dict[str, int] = {}
    transfer_category_scheduled_tokens: dict[str, int] = {}
    scheduled_total = 0
    prompt_total = 0
    chronology_valid = True
    prior_completed_ns = None
    owner_sets_valid = True
    for sequence in sequences:
        record = records_by_sequence[sequence]
        started_ns = int(record["started_at_ns"])
        completed_ns = int(record["completed_at_ns"])
        if prior_completed_ns is not None and started_ns < prior_completed_ns:
            chronology_valid = False
        prior_completed_ns = completed_ns
        request_ids = set(record["request_ids"])
        owner_sets_valid &= owners_by_sequence.get(sequence, set()) == request_ids
        current_scheduled = 0
        current_prompt = 0
        for attribution in record["execution_attribution"]:
            key = (
                str(attribution["request_id"]),
                int(attribution["scheduler_batch_id"]),
            )
            if key in observed_batches:
                invalid_reason = "gpu_model_runner_batch_duplicated"
                break
            value = (
                int(attribution["scheduled_tokens"]),
                int(attribution["prompt_tokens"]),
            )
            observed_batches[key] = value
            current_scheduled += value[0]
            current_prompt += value[1]
        if invalid_reason is not None:
            break
        scheduled_total += current_scheduled
        prompt_total += current_prompt
        kind = (
            "decode_only"
            if current_prompt == 0
            else "prompt_only"
            if current_prompt == current_scheduled
            else "mixed"
        )
        batch_kind_counts[kind] += 1
        for category in record["transfer_categories"]:
            transfer_category_counts[category] = (
                transfer_category_counts.get(category, 0) + 1
            )
            transfer_category_scheduled_tokens[category] = (
                transfer_category_scheduled_tokens.get(category, 0) + current_scheduled
            )
    if invalid_reason is not None:
        return {
            "schema": "semantic-gpu-model-runner-work-accounting-v1",
            "available": False,
            "reason": invalid_reason,
            "device_time_s": None,
            "observed_device_time_s": sum(observed_times),
        }
    complete = bool(
        complete_sequences
        and chronology_valid
        and owner_sets_valid
        and expected_batches
        and observed_batches == expected_batches
    )
    return {
        "schema": "semantic-gpu-model-runner-work-accounting-v1",
        "available": complete,
        "reason": (
            "complete_exact_scheduler_batch_coverage"
            if complete
            else "incomplete_gpu_model_runner_batch_coverage"
        ),
        "device_time_s": sum(observed_times) if complete else None,
        "observed_device_time_s": sum(observed_times),
        "execution_count": len(records_by_sequence),
        "sequence_first": sequences[0],
        "sequence_last": sequences[-1],
        "sequence_contiguous": complete_sequences,
        "chronology_valid": chronology_valid,
        "owner_sets_valid": owner_sets_valid,
        "expected_request_batch_count": len(expected_batches),
        "observed_request_batch_count": len(observed_batches),
        "scheduled_tokens": scheduled_total,
        "prompt_tokens": prompt_total,
        "decode_tokens": scheduled_total - prompt_total,
        "batch_kind_counts": batch_kind_counts,
        "transfer_category_execution_counts": transfer_category_counts,
        "transfer_category_scheduled_tokens": transfer_category_scheduled_tokens,
        "attribution_scope": (
            "TP1 cell-total model-runner current CUDA stream; shared kernels "
            "not split by request; transfer overlap not disambiguated"
        ),
        "synchronization_overhead": "one CUDA end-event synchronize per real step",
    }


def _map_one_preservation_event(event: dict) -> tuple[dict | None, str | None]:
    """Validate one v1 event and return a normalized request detail.

    The connector has already aggregated repeated lookup/admission/transfer
    episodes.  We validate that their IDs are unique, then use only the
    event's aggregate token/block totals; summing episode totals here would
    double-count preemption/retry traffic.
    """
    if not isinstance(event, dict):
        return None, REQUEST_EVENT_UNAVAILABLE_REASON
    if not event.get("authoritative") or event.get("event_integrity") != "complete":
        return None, str(event.get("reason") or REQUEST_EVENT_UNAVAILABLE_REASON)
    terminal_status = str(event.get("terminal_status", "")).lower()
    if (
        terminal_status not in {"finished", "completed", "success"}
        and not terminal_status.startswith("finished_")
    ) or event.get("terminal_reason") != "completed":
        return None, REQUEST_EVENT_TRANSFER_FAILED_REASON
    if not event.get("request_identity_valid"):
        return None, REQUEST_EVENT_IDENTITY_REASON
    if not event.get("lookup_metadata_complete"):
        return None, "lookup_metadata_unavailable"
    lookup_no_eligible = event.get("lookup_no_eligible") is True
    if lookup_no_eligible and event.get("lookup_observed") is not True:
        return None, "lookup_metadata_unavailable"
    if not event.get("transfer_metadata_complete"):
        statuses = json.dumps(event.get("transfer_job_statuses"), sort_keys=True)
        if any(word in statuses.lower() for word in ("fail", "cancel", "abort")):
            return None, REQUEST_EVENT_TRANSFER_FAILED_REASON
        return None, REQUEST_EVENT_TRANSFER_REASON
    if not event.get("admission_metadata_complete"):
        return None, REQUEST_EVENT_ADMISSION_REASON
    if not event.get("physical_block_identity_complete"):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    if event.get("tp_contributors_complete") is not True:
        return None, REQUEST_EVENT_TP_REASON
    speculative_reason, speculative_accounting = (
        _validate_speculative_transfer_evidence(event)
    )
    if speculative_reason is not None:
        return None, speculative_reason
    speculative_costs = event.get("speculative_transfer_costs")
    if speculative_costs is not None:
        if not isinstance(speculative_costs, dict):
            return None, REQUEST_EVENT_TRANSFER_REASON
        duplicated_jobs = _nonnegative_int(
            speculative_costs.get("duplicated_job_count")
        )
        duplicated_bytes = _nonnegative_int(speculative_costs.get("duplicated_bytes"))
        waste_bytes = _nonnegative_int(speculative_costs.get("waste_completed_bytes"))
        if None in (duplicated_jobs, duplicated_bytes, waste_bytes):
            return None, REQUEST_EVENT_TRANSFER_REASON
        completed = _nonnegative_int(speculative_accounting.get("completed_bytes"))
        consumed = _nonnegative_int(speculative_accounting.get("consumed_bytes"))
        successful = _nonnegative_int(
            speculative_accounting.get("successful_job_count")
        )
        if (
            completed is None
            or consumed is None
            or successful is None
            or duplicated_jobs > successful
            or duplicated_bytes > completed
            or waste_bytes != completed - consumed
        ):
            return None, REQUEST_EVENT_TRANSFER_REASON
    server_timing, server_timing_error = _validate_server_timing(event)
    if server_timing_error is not None:
        return None, server_timing_error
    prefill_cuda_events, prefill_timing_error = _validate_prefill_cuda_events(event)
    if prefill_timing_error is not None:
        return None, prefill_timing_error
    staging_protection_records = event.get("staging_protection_records", [])
    if not isinstance(staging_protection_records, list) or any(
        not isinstance(record, dict)
        or record.get("schema") != "semantic-staging-protection-v1"
        or record.get("action") not in {"protect", "release"}
        or record.get("outcome") not in {"accepted", "rejected", "noop"}
        for record in staging_protection_records
    ):
        return None, "staging_protection_evidence_invalid"

    lookup_episodes = _json_list(event.get("lookup_episodes"))
    admission_episodes = _json_list(event.get("admission_episodes"))
    if lookup_episodes is None or admission_episodes is None:
        return None, REQUEST_EVENT_EPISODE_REASON
    admission_decision_error = validate_admission_decision_records(
        event.get("admission_decisions", []),
        request_id=event.get("request_id"),
        run_id=event.get("run_id"),
    )
    if admission_decision_error is not None:
        return None, admission_decision_error
    admission_decisions = event.get("admission_decisions", [])
    admission_accounting = event.get("admission_decision_accounting")
    if admission_accounting is not None:
        if not isinstance(admission_accounting, dict):
            return None, REQUEST_EVENT_ADMISSION_DECISION_REASON
        expected_accounting = {
            "admitted": sum(
                record.get("action") != "reject" for record in admission_decisions
            ),
            "rejected": sum(
                record.get("action") == "reject" for record in admission_decisions
            ),
            "compulsory_fallbacks": sum(
                record.get("reason") == "compulsory_fallback_unavailable_features"
                for record in admission_decisions
            ),
        }
        if admission_accounting != expected_accounting:
            return None, REQUEST_EVENT_ADMISSION_DECISION_REASON
    lookup_ids, error = _episode_ids(lookup_episodes, "lookup_id")
    if error:
        return None, error
    admission_ids, error = _episode_ids(
        admission_episodes, "episode_id", allow_repeats=True
    )
    if error:
        return None, error

    required_tokens = _nonnegative_int(event.get("required_tokens"))
    reused_tokens = _nonnegative_int(event.get("reused_tokens"))
    loaded_tokens = _nonnegative_int(event.get("loaded_tokens"))
    recomputed_tokens = _nonnegative_int(event.get("recomputed_tokens"))
    actual_prompt_tokens = _nonnegative_int(
        event.get("actual_prompt_tokens_locally_executed")
    )
    cumulative_prompt_tokens = _nonnegative_int(
        event.get("cumulative_prompt_tokens_locally_executed")
    )
    if cumulative_prompt_tokens is None:
        cumulative_prompt_tokens = actual_prompt_tokens
    scheduled_prompt_tokens = _nonnegative_int(event.get("prompt_tokens_scheduled"))
    if None in (
        required_tokens,
        reused_tokens,
        loaded_tokens,
        recomputed_tokens,
        actual_prompt_tokens,
        cumulative_prompt_tokens,
        scheduled_prompt_tokens,
    ):
        return None, REQUEST_EVENT_ACCOUNTING_REASON
    if (
        required_tokens != reused_tokens + recomputed_tokens
        or loaded_tokens > reused_tokens
        or cumulative_prompt_tokens < recomputed_tokens
        or actual_prompt_tokens != cumulative_prompt_tokens
        or actual_prompt_tokens > scheduled_prompt_tokens
    ):
        return None, REQUEST_EVENT_ACCOUNTING_REASON
    if required_tokens > 0 and (not lookup_episodes or not admission_episodes):
        return None, REQUEST_EVENT_EPISODE_REASON
    lookup_keys = event.get("lookup_keys")
    if not isinstance(lookup_keys, list):
        return None, REQUEST_EVENT_EPISODE_REASON
    lookup_error = _validate_lookup_evidence(
        lookup_episodes,
        lookup_keys,
        required_tokens,
        allow_no_eligible=lookup_no_eligible,
    )
    if lookup_error is not None:
        return None, lookup_error
    declared_lookup_tokens = _lookup_set_tokens(lookup_keys)
    if not declared_lookup_tokens:
        declared_lookup_tokens = _subordinate_queried_tokens(lookup_episodes)
    if declared_lookup_tokens is None:
        return None, REQUEST_EVENT_EPISODE_REASON
    provenance_error, evicted_tokens, _, cold_tokens = _validate_lookup_provenance(
        event, declared_lookup_tokens or set()
    )
    if provenance_error is not None:
        return None, provenance_error
    missing_tokens: set[str] = set()
    for lookup_episode in lookup_episodes:
        for group in lookup_episode["groups"].values():
            missing_tokens.update(_lookup_set_tokens(group["missing"]) or set())
    lookup_id_set = set(lookup_ids)
    scheduled_sum = 0
    actual_prompt_sum = 0
    for admission in admission_episodes:
        scheduled = _nonnegative_int(admission.get("scheduled_tokens"))
        actual_prompt = _nonnegative_int(
            admission.get("prompt_tokens_locally_executed")
        )
        if scheduled is None or actual_prompt is None:
            return None, REQUEST_EVENT_EPISODE_REASON
        if actual_prompt > scheduled:
            return None, REQUEST_EVENT_ACCOUNTING_REASON
        scheduled_sum += scheduled
        actual_prompt_sum += actual_prompt
        lookup_id = admission.get("lookup_id")
        if lookup_id is not None and lookup_id not in lookup_id_set:
            return None, REQUEST_EVENT_EPISODE_REASON
    if (
        scheduled_sum != scheduled_prompt_tokens
        or actual_prompt_sum != actual_prompt_tokens
    ):
        return None, REQUEST_EVENT_ACCOUNTING_REASON
    admission_count = _nonnegative_int(event.get("admission_count"))
    recompute_admission_count = _nonnegative_int(event.get("recompute_admission_count"))
    if (
        admission_count is None
        or admission_count < len(admission_ids)
        or recompute_admission_count is None
        or recompute_admission_count > admission_count
    ):
        return None, REQUEST_EVENT_EPISODE_REASON

    eligible = event.get("eligible_kv_blocks")
    physical = event.get("physical_block_ids")
    if not isinstance(eligible, dict) or not isinstance(physical, dict):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    block_counts = {
        name: _nonnegative_int(eligible.get(name))
        for name in ("required", "local", "external", "recomputed")
    }
    if any(value is None for value in block_counts.values()):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    try:
        required_ids = _block_ids(physical.get("required"))
        local_ids = _block_ids(physical.get("local"))
        external_ids = _block_ids(physical.get("external"))
        recomputed_ids = _block_ids(physical.get("recomputed"))
        scheduled_ids = _block_ids(physical.get("scheduled"))
        loaded_ids = _block_ids(physical.get("loaded"))
        stored_ids = _block_ids(physical.get("stored"))
    except (TypeError, ValueError):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    eligible_sets = (set(local_ids), set(external_ids), set(recomputed_ids))
    if any(
        left & right
        for index, left in enumerate(eligible_sets)
        for right in eligible_sets[index + 1 :]
    ):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    eligible_ids = set(local_ids) | set(external_ids)
    if set(required_ids) != eligible_ids:
        return None, REQUEST_EVENT_PHYSICAL_REASON
    if set(recomputed_ids) & eligible_ids:
        return None, REQUEST_EVENT_PHYSICAL_REASON
    if set(scheduled_ids) != set(recomputed_ids):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    if [len(required_ids), len(local_ids), len(external_ids), len(recomputed_ids)] != [
        block_counts["required"],
        block_counts["local"],
        block_counts["external"],
        block_counts["recomputed"],
    ]:
        return None, REQUEST_EVENT_PHYSICAL_REASON
    loaded_blocks = _nonnegative_int(event.get("loaded_kv_blocks"))
    stored_blocks = _nonnegative_int(event.get("stored_kv_blocks"))
    if loaded_blocks is None or stored_blocks is None:
        return None, REQUEST_EVENT_PHYSICAL_REASON
    if loaded_blocks != len(loaded_ids) or stored_blocks != len(stored_ids):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    # Demand transfer references are job-qualified while admission references
    # are episode-qualified. Compare their physical (group, block) identity,
    # then require any remaining external blocks to be proven by an exact,
    # consumed speculative splice.
    demand_loaded_refs = {(group, block) for _, group, block in loaded_ids}
    external_refs = {(group, block) for _, group, block in external_ids}
    if (
        len(demand_loaded_refs) != len(loaded_ids)
        or not demand_loaded_refs <= external_refs
    ):
        return None, REQUEST_EVENT_PHYSICAL_REASON
    speculative_reason, speculative_destination_refs = (
        _consumed_speculative_destination_refs(
            event,
            missing_external_refs=external_refs - demand_loaded_refs,
        )
    )
    if speculative_reason is not None:
        return None, speculative_reason
    if demand_loaded_refs | speculative_destination_refs != external_refs:
        return None, REQUEST_EVENT_PHYSICAL_REASON
    required_kv_blocks = _nonnegative_int(event.get("required_kv_blocks"))
    if required_kv_blocks != block_counts["required"]:
        return None, REQUEST_EVENT_PHYSICAL_REASON
    table_error = _validate_physical_block_table_provenance(
        event,
        admission_episodes=admission_episodes,
        required_refs=required_ids,
        local_refs=local_ids,
        external_refs=external_ids,
        recomputed_refs=recomputed_ids,
        scheduled_refs=scheduled_ids,
        transfer_refs=(loaded_ids, stored_ids),
    )
    if table_error is not None:
        return None, table_error

    transfer_reason, observed_contributors, expected_contributors = (
        _validate_transfer_evidence(
            event,
            requires_load=bool(demand_loaded_refs),
            requires_store=stored_blocks > 0,
        )
    )
    if transfer_reason is not None:
        return None, transfer_reason
    load_job_count = _nonnegative_int(
        (event.get("transfer_job_counts") or {}).get("load")
    )
    if load_job_count is None:
        return None, REQUEST_EVENT_TRANSFER_REASON
    real_load = bool(external_refs) or load_job_count > 0
    if load_job_count > 0 and not demand_loaded_refs:
        return None, REQUEST_EVENT_TRANSFER_REASON

    totals = {
        "required_tokens": required_tokens,
        "reused_tokens": reused_tokens,
        "loaded_tokens": loaded_tokens,
        "recomputed_tokens": recomputed_tokens,
        "required_blocks": block_counts["required"],
        "reused_blocks": block_counts["local"] + block_counts["external"],
        "loaded_blocks": block_counts["external"],
        "recomputed_blocks": block_counts["recomputed"],
        "demand_loaded_blocks": len(demand_loaded_refs),
        "speculatively_loaded_blocks": len(speculative_destination_refs),
    }
    for transfer_kind in ("load", "store"):
        contributor_totals = event.get("tp_transfer_contributor_totals") or {}
        contributors = event.get("tp_transfer_contributors") or {}
        actual = _nonnegative_int(
            (contributor_totals.get("actual") or {}).get(transfer_kind)
        )
        expected = _nonnegative_int(
            (contributor_totals.get("expected") or {}).get(transfer_kind)
        )
        direct_actual = _nonnegative_int(contributors.get(transfer_kind))
        direct_expected = _nonnegative_int(
            (contributors.get("expected") or {}).get(transfer_kind)
        )
        if (
            actual is None
            or expected is None
            or actual != expected
            or direct_actual != actual
            or direct_expected != expected
            or observed_contributors.get(transfer_kind) != actual
            or expected_contributors.get(transfer_kind) != expected
        ):
            return None, REQUEST_EVENT_TP_REASON
    world_size = _nonnegative_int(event.get("tp_world_size"))
    rank_count = _nonnegative_int(event.get("tp_rank_count"))
    if (
        world_size is None
        or rank_count is None
        or world_size != rank_count
        or world_size == 0
    ):
        return None, REQUEST_EVENT_TP_REASON

    pressure = event.get("pressure")
    if not isinstance(pressure, bool):
        return None, REQUEST_EVENT_OUTCOME_REASON
    missing_pressure_tokens = missing_tokens - cold_tokens
    derived_pressure = bool(evicted_tokens or missing_pressure_tokens or real_load)
    if pressure != derived_pressure:
        return None, REQUEST_EVENT_OUTCOME_REASON
    if not pressure:
        outcome = "not_pressured"
    elif loaded_tokens > 0 and recomputed_tokens == 0:
        outcome = "hit"
    elif loaded_tokens > 0 and recomputed_tokens > 0:
        outcome = "partial"
    elif recomputed_tokens > 0:
        outcome = "miss"
    else:
        outcome = "not_pressured"
    if not pressure and loaded_tokens > 0:
        return None, REQUEST_EVENT_OUTCOME_REASON
    if outcome in {"miss", "partial"} and not (
        evicted_tokens or missing_pressure_tokens or real_load
    ):
        return None, REQUEST_EVENT_OUTCOME_REASON
    if event.get("outcome") != outcome:
        return None, REQUEST_EVENT_OUTCOME_REASON
    preemption_count = _nonnegative_int(event.get("preemption_count"))
    if preemption_count is None or not isinstance(event.get("preempted"), bool):
        return None, REQUEST_EVENT_ACCOUNTING_REASON
    detail = {
        "request_id": event.get("external_request_id"),
        "sample_index": event.get("sample_index"),
        "outcome": outcome,
        "reason": str(event.get("reason") or "authoritative_offload_events"),
        "outcome_basis": AUTHORITATIVE_REQUEST_EVENTS_BASIS,
        **totals,
        "pressure": pressure,
        "preempted": event["preempted"],
        "preemption_count": preemption_count,
        "lookup_episode_count": len(lookup_ids),
        "admission_episode_count": len(admission_ids),
        "lookup_episode_ids": lookup_ids,
        "admission_episode_ids": admission_ids,
        "speculative_transfer_accounting": speculative_accounting,
        "speculative_transfer_costs": (
            dict(speculative_costs) if speculative_costs is not None else None
        ),
        "server_timing": server_timing,
        "prefill_cuda_events": prefill_cuda_events,
        "staging_protection_records": [
            dict(record) for record in staging_protection_records
        ],
        "block_pool_snapshots": list(
            (event.get("prefetch_attribution") or {}).get("block_pool_snapshots", [])
        ),
        "cpu_cache_snapshots": list(
            (event.get("prefetch_attribution") or {}).get("cpu_cache_snapshots", [])
        ),
    }
    return detail, None


def map_preservation_events(
    events: list[dict],
    *,
    expected_request_ids: list[str] | None = None,
    run_id: str | None = None,
    cell_id: str | None = None,
) -> dict[str, object]:
    """Map authoritative v1 request events to row-level preservation fields.

    The strict log join remains responsible for rejecting foreign, duplicate,
    missing, or malformed events in a live run.  This function also checks
    those identities for callers/tests and returns an explicit unavailable
    mapping rather than silently labelling an incomplete payload.  Multiple
    request episodes are represented in each detail item, while aggregate
    totals come from the connector's already-deduplicated event counters.
    """
    if not isinstance(events, list) or not events:
        return _mapping_fields(reason=REQUEST_EVENT_MISSING_REASON)
    expected = list(expected_request_ids) if expected_request_ids is not None else None
    if expected is not None and len(expected) != len(set(expected)):
        return _mapping_fields(reason=REQUEST_EVENT_DUPLICATE_REASON)
    ids: list[str] = []
    for event in events:
        request_id = (
            event.get("external_request_id") if isinstance(event, dict) else None
        )
        if (
            not isinstance(request_id, str)
            or _CANONICAL_EXTERNAL_ID_RE.fullmatch(request_id) is None
        ):
            return _mapping_fields(reason=REQUEST_EVENT_IDENTITY_REASON)
        if request_id in ids:
            return _mapping_fields(reason=REQUEST_EVENT_DUPLICATE_REASON)
        ids.append(request_id)
        match = _CANONICAL_EXTERNAL_ID_RE.fullmatch(request_id)
        if run_id is not None and event.get("run_id") != run_id:
            return _mapping_fields(reason=REQUEST_EVENT_FOREIGN_REASON, request_ids=ids)
        if cell_id is not None and event.get("cell_id") != cell_id:
            return _mapping_fields(reason=REQUEST_EVENT_FOREIGN_REASON, request_ids=ids)
        if event.get("sample_id") != request_id or event.get("sample_index") != int(
            match.group(4)
        ):
            return _mapping_fields(
                reason=REQUEST_EVENT_IDENTITY_REASON, request_ids=ids
            )
    if expected is not None:
        expected_set = set(expected)
        actual_set = set(ids)
        if actual_set - expected_set:
            return _mapping_fields(reason=REQUEST_EVENT_FOREIGN_REASON, request_ids=ids)
        if expected_set - actual_set:
            return _mapping_fields(reason=REQUEST_EVENT_MISSING_REASON, request_ids=ids)
        by_id = {event["external_request_id"]: event for event in events}
        events = [by_id[request_id] for request_id in expected]
        ids = list(expected)

    details: list[dict] = []
    invalid_reasons: list[str] = []
    event_reasons: list[str | None] = []
    for event in events:
        detail, reason = _map_one_preservation_event(event)
        if detail is None:
            unavailable_reason = reason or REQUEST_EVENT_UNAVAILABLE_REASON
            invalid_reasons.append(unavailable_reason)
            event_reasons.append(unavailable_reason)
            details.append(_unavailable_event_detail(event, unavailable_reason))
        else:
            event_reasons.append(None)
            details.append(detail)
    if invalid_reasons:
        reason = (
            invalid_reasons[0]
            if len(set(invalid_reasons)) == 1
            else REQUEST_EVENT_UNAVAILABLE_REASON
        )
        unavailable_details = [
            _unavailable_event_detail(
                event, event_reason or REQUEST_EVENT_UNAVAILABLE_REASON
            )
            for event, event_reason in zip(events, event_reasons)
        ]
        return _mapping_fields(
            reason=reason,
            request_ids=ids,
            details=unavailable_details,
        )

    outcomes = {detail["outcome"] for detail in details}
    totals = {
        key: sum(detail[key] for detail in details)
        for key in (
            "required_tokens",
            "reused_tokens",
            "loaded_tokens",
            "recomputed_tokens",
            "required_blocks",
            "reused_blocks",
            "loaded_blocks",
            "recomputed_blocks",
        )
    }
    if len(outcomes) != 1:
        return _mapping_fields(
            outcome="unavailable",
            reason=MIXED_REQUEST_OUTCOMES_REASON,
            basis=AUTHORITATIVE_REQUEST_EVENTS_BASIS,
            request_ids=ids,
            details=details,
            authoritative_count=len(details),
            totals=totals,
        )
    return _mapping_fields(
        outcome=next(iter(outcomes)),
        reason=AUTHORITATIVE_REQUEST_EVENTS_BASIS,
        basis=AUTHORITATIVE_REQUEST_EVENTS_BASIS,
        request_ids=ids,
        details=details,
        authoritative_count=len(details),
        totals=totals,
    )


def validate_preservation_row(row: dict, *, require_fields: bool = True) -> list[str]:
    """Return contract violations without inferring any preservation result."""
    errors: list[str] = []
    preservation_fields = (
        "preservation_outcome",
        "preservation_reason",
        "preservation_request_id",
        "preservation_request_id_supplied",
    )
    if require_fields and any(field not in row for field in preservation_fields):
        errors.append("missing preservation schema fields")
        return errors
    if not require_fields and not any(field in row for field in preservation_fields):
        return errors

    sample_index = row.get("sample_index")
    try:
        parsed_sample_index = int(sample_index)
        if str(sample_index).strip() != str(parsed_sample_index):
            raise ValueError
        if parsed_sample_index < 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append(f"invalid sample_index={sample_index!r}")

    outcome = row.get("preservation_outcome")
    reason = row.get("preservation_reason")
    request_id = row.get("preservation_request_id")
    request_id_supplied = _canonical_bool(row.get("preservation_request_id_supplied"))
    if request_id_supplied is None:
        errors.append("preservation_request_id_supplied must be canonical true/false")
    if outcome not in ALLOWED_PRESERVATION_OUTCOMES:
        errors.append(f"invalid preservation_outcome={outcome!r}")

    workload = row.get("workload")
    sub_workload = row.get("sub_workload")
    is_chat_like = workload in {"chat", "rag", "longdoc"} or (
        workload == "mixed" and sub_workload in {"chat", "rag", "longdoc"}
    )
    is_needle = workload == "needle-v2" or (
        workload == "mixed" and sub_workload == "needle"
    )
    is_legacy_needle = workload == "needle"
    if not is_needle:
        for field in ("needle_outcome", "recall_load_bytes", "recall_store_bytes"):
            if not _empty(row.get(field)):
                errors.append(f"non-needle preservation row has {field}")
    if is_chat_like:
        errors.extend(_validate_request_outcome_fields(row))
        mapped = row.get("preservation_join_status") == "complete"
        if not mapped and outcome != "unavailable":
            errors.append("Chat/RAG/longdoc preservation must be unavailable")
        if not mapped and reason != AUTHORITATIVE_METADATA_MISSING_REASON:
            errors.append(
                "Chat/RAG/longdoc preservation reason is not authoritative-metadata-missing"
            )
        if not _empty(request_id):
            errors.append("Chat/RAG/longdoc preservation request_id must be None")
        if request_id_supplied:
            errors.append(
                "Chat/RAG/longdoc preservation request_id cannot be validated"
            )
    elif is_needle:
        needle_outcome = row.get("needle_outcome")
        if _empty(needle_outcome):
            if outcome != "unavailable":
                errors.append("missing needle metrics require unavailable preservation")
            if reason != NEEDLE_METRICS_UNAVAILABLE_REASON:
                errors.append(
                    "missing needle metrics have the wrong preservation reason"
                )
            if not _empty(request_id) or request_id_supplied:
                errors.append("unavailable needle preservation must have no request ID")
            for counter in ("recall_load_bytes", "recall_store_bytes"):
                if not _empty(row.get(counter)):
                    errors.append(f"unavailable needle preservation has {counter}")
        else:
            if needle_outcome not in ALLOWED_PRESERVATION_OUTCOMES - {"unavailable"}:
                errors.append(f"invalid needle_outcome={needle_outcome!r}")
            if outcome != needle_outcome:
                errors.append("preservation_outcome does not match needle_outcome")
            if reason != NEEDLE_ISOLATED_COUNTER_REASON:
                errors.append("needle preservation lacks the isolated-counter reason")
            expected_request_id = (
                f"needle-v2-{row.get('seed')}-ref{row.get('reference_count')}"
                f"-sample{row.get('sample_index')}-recall"
            )
            if _empty(request_id) or not request_id_supplied:
                errors.append(
                    "authoritative needle preservation needs a supplied request ID"
                )
            elif str(request_id) != expected_request_id:
                errors.append("authoritative needle request ID is not deterministic")
            load = _finite_nonnegative(row.get("recall_load_bytes"))
            store = _finite_nonnegative(row.get("recall_store_bytes"))
            if load is None or store is None:
                errors.append(
                    "authoritative needle preservation needs both recall counters"
                )
            elif not _counter_outcome_matches(outcome, load, store):
                errors.append("needle outcome does not match both recall counters")
    elif is_legacy_needle:
        if outcome != "unavailable" or reason != LEGACY_NEEDLE_UNAVAILABLE_REASON:
            errors.append("legacy needle preservation must be unavailable")
        if not _empty(request_id) or request_id_supplied:
            errors.append("legacy needle preservation cannot claim a request ID")
    elif outcome == "unavailable" and _empty(reason):
        errors.append("unavailable preservation requires a reason")
    return errors


def _finite_nonnegative(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _counter_outcome_matches(outcome: str, load: float, store: float) -> bool:
    if outcome == "hit":
        return load > 0 and store == 0
    if outcome == "partial":
        return load > 0 and store > 0
    if outcome == "miss":
        return load == 0 and store > 0
    if outcome == "not_pressured":
        return load == 0 and store == 0
    return False


def _strict_nonnegative_integer(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if str(parsed) == value.strip() and parsed >= 0 else None
    return None


def _row_json(row: dict, field: str):
    value = row.get(field)
    if _empty(value):
        return None
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _validate_request_outcome_fields(row: dict) -> list[str]:
    """Validate the joined Chat/RAG projection without reclassifying it."""
    errors: list[str] = []
    join_status = row.get("preservation_join_status")
    basis = row.get("preservation_outcome_basis")
    if join_status != "complete" and _empty(basis):
        return errors
    if join_status != "complete":
        errors.append("request outcome fields require a complete preservation join")
        return errors
    request_ids = _row_json(row, "preservation_request_ids_json")
    details = _row_json(row, "preservation_request_outcome_details_json")
    if not isinstance(request_ids, list) or not all(
        isinstance(request_id, str) for request_id in request_ids
    ):
        errors.append("preservation request IDs are not a JSON string array")
        request_ids = []
    if len(request_ids) != len(set(request_ids)):
        errors.append("duplicate preservation request IDs in outcome projection")
    if not isinstance(details, list) or not all(
        isinstance(detail, dict) for detail in details
    ):
        errors.append(
            "preservation request outcome details are not a JSON object array"
        )
        details = []
    row_request_ids = _row_json(row, "request_ids_json")
    if not isinstance(row_request_ids, list) or not all(
        isinstance(request_id, str) for request_id in row_request_ids
    ):
        errors.append("row request IDs are not a JSON string array")
        row_request_ids = []
    elif row_request_ids != request_ids:
        errors.append("row request IDs do not match preservation request IDs")
    event_rows = _row_json(row, "preservation_events_json")
    if not isinstance(event_rows, list) or not all(
        isinstance(event, dict) for event in event_rows
    ):
        errors.append("preservation events are not a JSON object array")
        event_rows = []
    event_count = _strict_nonnegative_integer(row.get("preservation_event_count"))
    if event_count is None or event_count == 0:
        errors.append("complete request-event joins need an event count")
    elif event_count != len(event_rows):
        errors.append("preservation event count does not match event JSON")
    event_ids = [event.get("external_request_id") for event in event_rows]
    if event_ids != request_ids:
        errors.append("preservation event IDs do not match projected request IDs")
    if len(event_ids) != len(set(event_ids)):
        errors.append("duplicate preservation event IDs in archived projection")
    run_id = row.get("run_id")
    cell_id = row.get("cell_id")
    request_id_prefix = row.get("request_id_prefix")
    if _empty(run_id) or _empty(cell_id):
        errors.append("complete request-event joins need run_id and cell_id")
    for event in event_rows:
        event_id = event.get("external_request_id")
        if event.get("run_id") != run_id or event.get("cell_id") != cell_id:
            errors.append("preservation event is foreign to the archived row")
        if event.get("sample_id") != event_id:
            errors.append("preservation event sample identity is not bound")
        match = _CANONICAL_EXTERNAL_ID_RE.fullmatch(event_id or "")
        if match is None or event.get("sample_index") != int(match.group(4)):
            errors.append("preservation event sample index is not bound")
    if not is_canonical_request_id_prefix(request_id_prefix):
        errors.append("complete request-event joins need a canonical request_id_prefix")
    expected_prefix = expected_request_id_prefix(
        run_id, cell_id, row.get("workload"), row.get("sub_workload")
    )
    if expected_prefix is None or request_id_prefix != expected_prefix:
        errors.append(
            "request_id_prefix is not bound to row run/cell/workload identity"
        )
    if not _empty(request_id_prefix) and any(
        not isinstance(request_id, str) or not request_id.startswith(request_id_prefix)
        for request_id in request_ids
    ):
        errors.append("preservation request IDs do not match request_id_prefix")
    for request_id in request_ids:
        request_match = _CANONICAL_EXTERNAL_ID_RE.fullmatch(request_id or "")
        if request_match is None or not request_id.startswith(expected_prefix or ""):
            errors.append("preservation request ID is not bound to row identity")
    detail_ids = [detail.get("request_id") for detail in details]
    if detail_ids != request_ids:
        errors.append("request outcome detail IDs do not match the request ID array")
    recomputed = map_preservation_events(
        event_rows,
        expected_request_ids=row_request_ids,
        run_id=run_id,
        cell_id=cell_id,
    )
    json_projection_fields = {
        "preservation_request_ids_json",
        "preservation_request_outcome_details_json",
    }
    numeric_projection_fields = {
        "preservation_authoritative_request_count",
        "preservation_required_tokens",
        "preservation_reused_tokens",
        "preservation_loaded_tokens",
        "preservation_recomputed_tokens",
        "preservation_required_blocks",
        "preservation_reused_blocks",
        "preservation_loaded_blocks",
        "preservation_recomputed_blocks",
    }
    for field in (
        "preservation_outcome",
        "preservation_reason",
        *REQUEST_OUTCOME_FIELDS,
    ):
        expected_value = recomputed.get(field)
        if field in json_projection_fields:
            stored_value = _row_json(row, field)
            expected_value = _row_json(recomputed, field)
        elif field in numeric_projection_fields:
            stored_value = _strict_nonnegative_integer(row.get(field))
        else:
            stored_value = row.get(field)
        if expected_value is None:
            matches = _empty(row.get(field))
        else:
            matches = stored_value == expected_value
        if not matches:
            errors.append(f"stored {field} does not match recomputed event projection")
    if event_count is not None and event_count != len(event_rows):
        errors.append("stored preservation event count is not authoritative")
    count = _strict_nonnegative_integer(
        row.get("preservation_authoritative_request_count")
    )
    if basis == AUTHORITATIVE_REQUEST_EVENTS_BASIS:
        if (
            count is None
            or count == 0
            or count != len(request_ids)
            or count != len(details)
        ):
            errors.append("authoritative request outcome cardinality is inconsistent")
        detail_outcomes = {detail.get("outcome") for detail in details}
        if any(
            outcome not in ALLOWED_PRESERVATION_OUTCOMES - {"unavailable"}
            for outcome in detail_outcomes
        ):
            errors.append(
                "request outcome details contain an invalid authoritative outcome"
            )
        if len(detail_outcomes) == 1:
            if row.get("preservation_outcome") != next(iter(detail_outcomes)):
                errors.append(
                    "row preservation outcome does not match request outcomes"
                )
            if row.get("preservation_reason") != AUTHORITATIVE_REQUEST_EVENTS_BASIS:
                errors.append(
                    "uniform request outcomes have the wrong preservation reason"
                )
        elif (
            row.get("preservation_outcome") != "unavailable"
            or row.get("preservation_reason") != MIXED_REQUEST_OUTCOMES_REASON
        ):
            errors.append("mixed request outcomes must be explicitly unavailable")
        for field in REQUEST_OUTCOME_FIELDS[4:]:
            if _strict_nonnegative_integer(row.get(field)) is None:
                errors.append(f"authoritative request outcome lacks {field}")
        required = _strict_nonnegative_integer(row.get("preservation_required_tokens"))
        reused = _strict_nonnegative_integer(row.get("preservation_reused_tokens"))
        loaded = _strict_nonnegative_integer(row.get("preservation_loaded_tokens"))
        recomputed = _strict_nonnegative_integer(
            row.get("preservation_recomputed_tokens")
        )
        required_blocks = _strict_nonnegative_integer(
            row.get("preservation_required_blocks")
        )
        reused_blocks = _strict_nonnegative_integer(
            row.get("preservation_reused_blocks")
        )
        loaded_blocks = _strict_nonnegative_integer(
            row.get("preservation_loaded_blocks")
        )
        recomputed_blocks = _strict_nonnegative_integer(
            row.get("preservation_recomputed_blocks")
        )
        if None not in (required, reused, loaded, recomputed) and (
            required != reused + recomputed or loaded > reused
        ):
            errors.append("aggregate request token accounting is inconsistent")
        if None not in (
            required_blocks,
            reused_blocks,
            loaded_blocks,
            recomputed_blocks,
        ) and (required_blocks != reused_blocks or loaded_blocks > reused_blocks):
            errors.append("aggregate request block accounting is inconsistent")
    else:
        if basis not in (None, ""):
            errors.append(f"unknown preservation outcome basis={basis!r}")
        if row.get("preservation_outcome") != "unavailable":
            errors.append("unavailable request outcomes cannot claim a label")
        if event_count is None or event_count == 0:
            errors.append("complete request-event joins need an event count")
        elif len(request_ids) != event_count or len(details) != event_count:
            errors.append("unavailable request-event cardinality is inconsistent")
        elif [detail.get("request_id") for detail in details] != request_ids:
            errors.append(
                "unavailable request detail IDs do not match the request ID array"
            )
        if any(detail.get("outcome") != "unavailable" for detail in details):
            errors.append("unavailable request outcomes contain a labelled detail")
        if row.get("preservation_request_id_supplied") is True or not _empty(
            row.get("preservation_request_id")
        ):
            errors.append("request-event-unavailable rows cannot claim one request ID")
        if count is not None:
            errors.append(
                "unavailable request outcomes cannot claim an authoritative count"
            )
        for field in REQUEST_OUTCOME_FIELDS[4:]:
            if not _empty(row.get(field)):
                errors.append(f"unavailable request outcomes have {field}")
    return errors
