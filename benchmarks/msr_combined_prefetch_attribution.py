"""Fail-closed CPU analysis for the combined MSR/prefetch release comparison.

The analyzer consumes ordinary ``execute-cell`` result JSON.  It deliberately
uses authoritative request events for executed work instead of the campaign's
unique-recomputation metric: repeated prefill after preemption is provider work
and must remain in the numerator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from benchmarks.preservation import map_preservation_events

SCHEMA = "msr-combined-prefetch-attribution-v1"
DRAM_THRESHOLD = 2
TTFT_ALLOWANCE_PERCENT = 3.0
_ROLES = ("msr", "arc", "lru")
_FORBIDDEN_CAUSAL_FIELDS = frozenset(
    {"eventual_demand", "future_reuse", "consumed", "discarded", "output"}
)
_COALESCING_CONFIG_FIELDS = frozenset(
    {"wait_hint_coalescing_min_sources", "wait_hint_coalescing_max_ms"}
)


class AttributionError(ValueError):
    """The supplied observations cannot support the requested comparison."""


def _fail(message: str) -> None:
    raise AttributionError(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _records(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{field} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        _fail(f"{field} must contain objects")
    return list(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _extra_config(result: Mapping[str, Any]) -> Mapping[str, Any]:
    argv = result.get("resolved_launch_argv")
    if not isinstance(argv, list):
        _fail("resolved_launch_argv is unavailable")
    try:
        index = argv.index("--kv-transfer-config")
        transfer = json.loads(argv[index + 1])
    except (ValueError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise AttributionError("resolved KV transfer config is unavailable") from error
    config = _mapping(transfer, "kv-transfer-config")
    return _mapping(config.get("kv_connector_extra_config"), "extra_config")


def _canonical_policy_identity(config: Mapping[str, Any]) -> str:
    """Resolve the actual producer contract, not an experiment arm label."""
    if (
        config.get("spec_name") != "SemanticOffloadingSpec"
        or config.get("spec_module_path") != "semantic_offload.spec"
    ):
        _fail("unknown/non-instrumented offload spec identity")
    eviction = config.get("eviction_policy")
    if eviction in {"arc", "lru"}:
        if config.get("instrumentation_only") is not True:
            _fail("ARC/LRU control is not the real instrumentation-only producer")
        return str(eviction)
    if eviction is None:
        if (
            config.get("method") == "mean"
            and config.get("instrumentation_only") is not True
        ):
            return "msr"
        _fail("unknown semantic policy method or instrumentation identity")
    _fail(f"unknown eviction policy identity {eviction!r}")


def _demand_prefetch_is_disabled(config: Mapping[str, Any]) -> bool:
    return (
        _maybe_number(config.get("prefetch_budget_fraction")) == 0.0
        and config.get("prefetch_trigger") == "preemption"
    )


def _request_rows(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = _records(result.get("requests"), "requests")
    if not rows:
        _fail("requests is empty")
    ids: list[str] = []
    external: list[str] = []
    for index, row in enumerate(rows):
        ids.append(
            _string(
                row.get("manifest_request_id"), f"requests[{index}].manifest_request_id"
            )
        )
        external.append(
            _string(
                row.get("external_request_id"), f"requests[{index}].external_request_id"
            )
        )
        if row.get("success") is not True or row.get("terminal_complete") is not True:
            _fail(f"request {ids[-1]} did not complete successfully")
        _finite(row.get("ttft_ms"), f"requests[{index}].ttft_ms")
        _string(
            row.get("output_token_digest"), f"requests[{index}].output_token_digest"
        )
    if len(ids) != len(set(ids)) or len(external) != len(set(external)):
        _fail("request identities are duplicated")
    return rows


def _execution_by_manifest_request(
    result: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    events = _records(result.get("preservation_events"), "preservation_events")
    expected_external = [str(row["external_request_id"]) for row in requests]
    first_event = _mapping(events[0], "preservation_events[0]") if events else {}
    run_id = _string(first_event.get("run_id"), "preservation run_id")
    cell_id = _string(first_event.get("cell_id"), "preservation cell_id")
    mapped = map_preservation_events(
        [dict(event) for event in events],
        expected_request_ids=expected_external,
        run_id=run_id,
        cell_id=cell_id,
    )
    if mapped.get("preservation_basis") != "authoritative_request_events":
        _fail(
            "authoritative preservation mapping unavailable: "
            f"{mapped.get('preservation_reason')}"
        )
    by_external = {event.get("external_request_id"): event for event in events}
    totals: dict[str, int] = {}
    for row in requests:
        manifest_id = str(row["manifest_request_id"])
        event = _mapping(by_external.get(row["external_request_id"]), manifest_id)
        actual = _nonnegative_int(
            event.get("actual_prompt_tokens_locally_executed"),
            f"{manifest_id}.actual_prompt_tokens_locally_executed",
        )
        cumulative = event.get("cumulative_prompt_tokens_locally_executed", actual)
        if _nonnegative_int(cumulative, f"{manifest_id}.cumulative") != actual:
            _fail(f"{manifest_id}: actual and cumulative executed tokens differ")
        episodes = _records(
            event.get("admission_episodes"), f"{manifest_id}.admission_episodes"
        )
        episode_sum = sum(
            _nonnegative_int(
                item.get("prompt_tokens_locally_executed"), f"{manifest_id}.episode"
            )
            for item in episodes
        )
        if episode_sum != actual:
            _fail(f"{manifest_id}: admission episode execution does not reconcile")
        totals[manifest_id] = actual
    return totals


def _cohort_map(request_manifest: Mapping[str, Any]) -> dict[str, str]:
    requests = _records(request_manifest.get("requests"), "request manifest requests")
    result: dict[str, str] = {}
    for record in requests:
        request_id = _string(record.get("request_id"), "request manifest request_id")
        cohort = record.get("claim_cohort")
        if cohort is None:
            phase = record.get("phase")
            cohort = "return" if phase in {"return", "resume"} else "other"
        if cohort not in {"return", "non_return", "warmup", "competition", "other"}:
            _fail(f"{request_id}: unsupported claim_cohort {cohort!r}")
        result[request_id] = str(cohort)
    if len(result) != len(requests):
        _fail("request manifest contains duplicate request IDs")
    return result


def _match_manifest_id(request_id: str, cohorts: Mapping[str, str]) -> str:
    if request_id in cohorts:
        return request_id
    # Campaign profiles replace the split prefix (for example heldout -> screen)
    # while preserving the frozen suffix.  Accept only one unambiguous suffix.
    parts = request_id.split("-", 1)
    suffix = parts[1] if len(parts) == 2 else request_id
    matches = [candidate for candidate in cohorts if candidate.endswith(suffix)]
    if len(matches) != 1:
        _fail(f"{request_id}: no unique immutable workload binding")
    return matches[0]


def _validate_pairing(
    runs: Mapping[str, Mapping[str, Any]], rows: Mapping[str, list[Mapping[str, Any]]]
) -> list[str]:
    reference = rows["msr"]
    ids = [str(row["manifest_request_id"]) for row in reference]
    configs = {role: _extra_config(runs[role]) for role in _ROLES}
    expected_policy = {"msr": "msr", "arc": "arc", "lru": "lru"}
    for role, config in configs.items():
        if config.get("store_threshold") != DRAM_THRESHOLD:
            _fail(f"{role} does not use store_threshold=2")
        if _canonical_policy_identity(config) != expected_policy[role]:
            _fail(f"{role} has the wrong eviction policy")
        if role == "msr" and config.get("prefetch_selector") != "semantic":
            _fail("MSR does not activate the semantic prefetch selector")
    allowed_differences = {
        "eviction_policy",
        "method",
        "prefetch_selector",
        "prefetch_trigger",
        "prefetch_budget_fraction",
        "prefetch_score_max_age_ms",
        "soft_retention",
        "instrumentation_only",
        "shared_prefetch_scorer",
    }
    if all(_demand_prefetch_is_disabled(configs[role]) for role in ("arc", "lru")):
        allowed_differences.update(_COALESCING_CONFIG_FIELDS)
    generic_keys = (
        set().union(*(config.keys() for config in configs.values()))
        - allowed_differences
    )
    for key in generic_keys:
        values = {
            json.dumps(config.get(key), sort_keys=True) for config in configs.values()
        }
        if len(values) != 1:
            _fail(f"generic DRAM/runtime setting differs: {key}")
    for role in _ROLES:
        result = runs[role]
        if result.get("valid") is not True:
            _fail(f"{role} result is invalid")
        current = rows[role]
        if [row["manifest_request_id"] for row in current] != ids:
            _fail(f"{role} request order differs")
        for left, right in zip(reference, current):
            if left["output_token_digest"] != right["output_token_digest"]:
                _fail(f"{role} output differs for {left['manifest_request_id']}")
        for field in (
            "profile_id",
            "replicate_id",
            "seed",
            "request_manifest_sha256",
            "order_schedule_digest",
            "baseline_profile_hash",
        ):
            if runs["msr"].get(field) in (None, "") or result.get(field) in (None, ""):
                _fail(f"{role} missing matched identity {field}")
            if result.get(field) != runs["msr"].get(field):
                _fail(f"{role} mismatched {field}")
    return ids


def _validate_shared_prefetch_control(
    msr: Mapping[str, Any],
    control: Mapping[str, Any],
    msr_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require a true nomination ablation: same MSR retention, FIFO selector."""
    config = _extra_config(control)
    msr_config = _extra_config(msr)
    if (
        config.get("store_threshold") != DRAM_THRESHOLD
        or _canonical_policy_identity(config) != "msr"
        or config.get("prefetch_selector") != "request_order"
    ):
        _fail("shared-prefetch control must be MSR retention with FIFO nomination")
    allowed = {"prefetch_selector"}
    for key in set(config) | set(msr_config):
        if key not in allowed and config.get(key) != msr_config.get(key):
            _fail(f"shared-prefetch control changed non-selector setting: {key}")
    if control.get("valid") is not True:
        _fail("shared-prefetch control is invalid")
    for field in (
        "profile_id",
        "replicate_id",
        "seed",
        "request_manifest_sha256",
        "order_schedule_digest",
        "baseline_profile_hash",
    ):
        if control.get(field) in (None, "") or control.get(field) != msr.get(field):
            _fail(f"shared-prefetch control mismatched {field}")
    if [row["manifest_request_id"] for row in control_rows] != [
        row["manifest_request_id"] for row in msr_rows
    ]:
        _fail("shared-prefetch control request order differs")
    for left, right in zip(msr_rows, control_rows):
        if left["output_token_digest"] != right["output_token_digest"]:
            _fail(
                "shared-prefetch control output differs for "
                f"{left['manifest_request_id']}"
            )


def validate_semantic_nomination(
    events: Any, *, max_score_age_ns: int | None = None
) -> dict[str, Any]:
    """Validate cross-session nomination separately from within-source ranking."""
    records = _records(events, "semantic nomination events")
    selected = no_action = 0
    activated_observations = 0
    selected_event_indexes: list[int] = []
    validated_selected_keys: set[str] = set()
    signatures: list[dict[str, Any]] = []
    for index, event in enumerate(records):
        if (
            event.get("schema") != "semantic-cross-session-prefetch-nomination-v1"
            or event.get("event") != "nomination_decision"
            or event.get("selector") != "semantic"
        ):
            _fail(f"nomination[{index}] has unsupported schema")
        forbidden = _FORBIDDEN_CAUSAL_FIELDS.intersection(event)
        if forbidden:
            _fail(
                f"nomination[{index}] causal input contains future fields: {sorted(forbidden)}"
            )
        decision = _nonnegative_int(event.get("decision_ns"), "decision_ns")
        if event.get("clock_domain", "host_monotonic") != "host_monotonic":
            _fail(f"nomination[{index}] has an unsupported clock domain")
        candidates = _records(event.get("candidates"), "candidates")
        budget = _nonnegative_int(
            event.get("shared_allocation_budget_blocks"),
            "shared_allocation_budget_blocks",
        )
        _nonnegative_int(event.get("shared_reserved_blocks"), "shared_reserved_blocks")
        _nonnegative_int(event.get("nomination_cpu_ns"), "nomination_cpu_ns")
        eligible_blocks = 0
        ordered_event_ids: list[str] = []
        candidates_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
        aggregate_by_identity: dict[tuple[str, str], float | None] = {}
        rank_records: list[tuple[tuple[Any, ...], tuple[str, str]]] = []
        chosen_records = _records(event.get("chosen"), "chosen")
        chosen_identities = {
            (
                _string(item.get("event_id"), "chosen.event_id"),
                _string(item.get("session_id"), "chosen.session_id"),
            )
            for item in chosen_records
        }
        identities: set[tuple[str, str]] = set()
        exact_source_identities: set[tuple[Any, ...]] = set()
        for candidate in candidates:
            identity = (
                _string(candidate.get("event_id"), "candidate event_id"),
                _string(candidate.get("session_id"), "candidate session_id"),
            )
            identities.add(identity)
            exact_source_identity = (
                candidate.get("run_id"),
                candidate.get("cell_id"),
                candidate.get("session_id"),
                candidate.get("generation"),
                candidate.get("observed_prefix_id"),
                candidate.get("source_request_id"),
                candidate.get("prefix_boundary_tokens"),
                json.dumps(candidate.get("namespace"), sort_keys=True),
            )
            if exact_source_identity in exact_source_identities:
                _fail("semantic nomination contains duplicate exact source identities")
            exact_source_identities.add(exact_source_identity)
            candidates_by_identity[identity] = candidate
            ordered_event_ids.append(identity[0])
            observed = _nonnegative_int(
                candidate.get("source_observed_ns"), "source_observed_ns"
            )
            observation = _nonnegative_int(
                candidate.get("hint_observation_available_ns"),
                "hint_observation_available_ns",
            )
            publication = _nonnegative_int(
                candidate.get("hint_input_publication_ns"),
                "hint_input_publication_ns",
            )
            if not observed <= observation <= publication <= decision:
                _fail(f"nomination[{index}] violates causal clock order")
            namespace = _mapping(candidate.get("namespace"), "candidate namespace")
            for field in (
                "model_revision",
                "tokenizer_revision",
                "adapter_namespace",
                "layout_namespace",
            ):
                _string(namespace.get(field), f"namespace.{field}")
            _string(candidate.get("observed_prefix_id"), "observed_prefix_id")
            _nonnegative_int(candidate.get("generation"), "generation")
            _nonnegative_int(
                candidate.get("prefix_boundary_tokens"), "prefix_boundary_tokens"
            )
            scores = _records(
                candidate.get("semantic_key_scores"), "semantic_key_scores"
            )
            score_keys: list[str] = []
            for score in scores:
                score_key = _string(score.get("key"), "semantic score key")
                score_keys.append(score_key)
                _finite(score.get("score"), "semantic score")
                score_ns = _nonnegative_int(
                    score.get("observation_available_ns"),
                    "score observation_available_ns",
                )
                age_ns = decision - score_ns
                if age_ns < 0:
                    _fail("candidate semantic feature is not prior-only")
                if max_score_age_ns is not None and age_ns > max_score_age_ns:
                    _fail("candidate semantic feature is stale")
                if score.get("age_ns") != age_ns:
                    _fail("semantic score age does not reconcile decision clock")
                if score.get("max_age_ns") != max_score_age_ns:
                    _fail("semantic score max age differs from runtime config")
                if score.get("fresh_at_decision") is not True:
                    _fail("semantic score is not explicitly fresh at decision")
                _string(
                    score.get("observation_provenance"),
                    "semantic score observation_provenance",
                )
                captured_raw = score.get("feature_snapshot_captured_ns")
                feature_age = score.get("feature_age_ns")
                if captured_raw is None and feature_age is None:
                    if score.get("observation_provenance") != (
                        "manager_relevance_updated_ns"
                    ):
                        _fail("semantic feature snapshot provenance is unavailable")
                else:
                    captured = _nonnegative_int(
                        captured_raw,
                        "feature_snapshot_captured_ns",
                    )
                    if captured > decision or feature_age != decision - captured:
                        _fail("semantic feature snapshot clock does not reconcile")
            if len(score_keys) != len(set(score_keys)):
                _fail("candidate semantic score keys are duplicated")
            aggregate = candidate.get("aggregate_score")
            if scores:
                computed_mean = statistics.fmean(
                    _finite(score["score"], "semantic score") for score in scores
                )
                observed_aggregate = _finite(aggregate, "aggregate_score")
                if not math.isclose(
                    observed_aggregate, computed_mean, rel_tol=1e-12, abs_tol=1e-12
                ):
                    _fail("candidate aggregate is not the causal score mean")
                aggregate_by_identity[identity] = computed_mean
            elif aggregate is not None:
                _fail("unscored candidate has an aggregate score")
            else:
                aggregate_by_identity[identity] = None
            score_observed = candidate.get("score_observation_available_ns")
            expected_score_observed = (
                max(score["observation_available_ns"] for score in scores)
                if scores
                else None
            )
            if score_observed != expected_score_observed:
                _fail("candidate score observation timestamp does not reconcile")
            causal_inputs = _mapping(candidate.get("causal_inputs"), "causal_inputs")
            expected_causal_inputs = {
                "source_observed_ns": observed,
                "hint_observation_available_ns": observation,
                "hint_input_publication_ns": publication,
                "score_observation_available_ns": expected_score_observed,
                "score_max_age_ns": max_score_age_ns,
                "decision_ns": decision,
            }
            if any(
                causal_inputs.get(field) != value
                for field, value in expected_causal_inputs.items()
            ):
                _fail("candidate causal input snapshot does not reconcile")
            keys = candidate.get("eligible_resident_prefix_keys")
            if not isinstance(keys, list) or not keys:
                _fail("candidate lacks an eligible exact DRAM prefix")
            if not set(score_keys).issubset(set(keys)):
                _fail("semantic scores include a non-eligible exact key")
            eligible_blocks += len(keys)
            rank_records.append(
                (
                    (
                        aggregate_by_identity[identity] is None,
                        -(
                            aggregate_by_identity[identity]
                            if aggregate_by_identity[identity] is not None
                            else 0.0
                        ),
                        _string(candidate.get("run_id"), "candidate run_id"),
                        _string(candidate.get("cell_id"), "candidate cell_id"),
                        identity[1],
                        candidate["generation"],
                        candidate["observed_prefix_id"],
                        identity[0],
                    ),
                    identity,
                )
            )
        if event.get("eligible_candidate_blocks") != eligible_blocks:
            _fail("eligible candidate block total does not reconcile")
        scored_candidate_count = sum(
            score is not None for score in aggregate_by_identity.values()
        )
        if event.get("scored_candidate_count") != scored_candidate_count:
            _fail("scored candidate count does not reconcile")
        expected_constrained = bool(
            scored_candidate_count >= 2 and eligible_blocks > budget
        )
        if event.get("shared_budget_constrained") is not expected_constrained:
            _fail("shared budget constraint flag does not reconcile")
        if len(identities) != len(candidates):
            _fail("semantic nomination candidates are duplicated")
        expected_rank = [identity for _, identity in sorted(rank_records)]
        observed_rank = [
            (candidate["event_id"], candidate["session_id"]) for candidate in candidates
        ]
        if observed_rank != expected_rank:
            _fail("candidate order does not match recomputed fresh causal means")
        highest = event.get("highest_ranked_candidate")
        scored = {
            identity: score
            for identity, score in aggregate_by_identity.items()
            if score is not None
        }
        if scored:
            first_candidate_identity = expected_rank[0]
            highest_record = _mapping(highest, "highest_ranked_candidate")
            if (
                highest_record.get("event_id"),
                highest_record.get("session_id"),
            ) != first_candidate_identity or not math.isclose(
                _finite(
                    highest_record.get("aggregate_score"),
                    "highest aggregate score",
                ),
                scored[first_candidate_identity],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                _fail("highest-ranked candidate evidence does not reconcile")
        elif highest is not None:
            _fail("unscored/no-candidate poll has a highest-ranked candidate")
        if not chosen_identities:
            no_action += 1
            _string(event.get("no_action_reason"), "no_action_reason")
        elif not chosen_identities.issubset(identities):
            _fail("selected session is not a causal candidate")
        else:
            if event.get("no_action_reason") is not None:
                _fail("selected nomination cannot report a no-action reason")
            selected += len(chosen_identities)
            selected_event_indexes.append(index)
            first_candidate_identity = (
                candidates[0]["event_id"],
                candidates[0]["session_id"],
            )
            if first_candidate_identity not in chosen_identities:
                _fail("chosen session does not follow semantic candidate ranking")
            if not scored or aggregate_by_identity[first_candidate_identity] != max(
                scored.values()
            ):
                _fail("semantic ranking did not select the highest fresh causal mean")
        publication_order = event.get("candidate_publication_order_event_ids")
        if (
            not isinstance(publication_order, list)
            or len(publication_order) != len(set(publication_order))
            or set(publication_order) != set(ordered_event_ids)
        ):
            _fail("candidate publication order does not identify the competition")
        selected_blocks = _nonnegative_int(
            event.get("selected_blocks"), "selected_blocks"
        )
        selected_bytes = _nonnegative_int(event.get("selected_bytes"), "selected_bytes")
        selected_key_count = 0
        for item in chosen_records:
            keys = item.get("selected_keys")
            if (
                not isinstance(keys, list)
                or not keys
                or any(not isinstance(key, str) or not key for key in keys)
            ):
                _fail("chosen.selected_keys must contain exact key identities")
            identity = (item["event_id"], item["session_id"])
            eligible = candidates_by_identity[identity]["eligible_resident_prefix_keys"]
            if keys != eligible[: len(keys)]:
                _fail("chosen exact keys are not the contiguous selected source prefix")
            if not candidates_by_identity[identity]["semantic_key_scores"]:
                _fail("chosen semantic session has no causal score")
            selected_key_count += len(keys)
            validated_selected_keys.update(keys)
        if selected_blocks != selected_key_count:
            _fail("selected block total does not reconcile chosen exact keys")
        if selected_blocks > budget:
            _fail("selected blocks exceed the shared allocation budget")
        if selected_blocks and selected_bytes == 0:
            _fail("selected exact keys have zero transfer bytes")
        if chosen_records:
            signatures.append(
                {
                    "candidate_publication_order_event_ids": publication_order,
                    "shared_allocation_budget_blocks": budget,
                    "selected_bytes": selected_bytes,
                    "chosen_identities": sorted(chosen_identities),
                    "candidate_identities": sorted(identities),
                }
            )
            if expected_constrained and len(exact_source_identities) >= 2:
                activated_observations += 1
    if not records or activated_observations == 0:
        _fail("cross-session semantic nomination is inactive")
    return {
        "observations": len(records),
        "selected": selected,
        "no_action": no_action,
        "activated_observations": activated_observations,
        "selected_event_indexes": selected_event_indexes,
        "selected_keys": sorted(validated_selected_keys),
        "signatures": signatures,
    }


def validate_score_reversal(
    left: Any, right: Any, *, max_score_age_ns: int | None = None
) -> dict[str, Any]:
    """Require semantic scores, not FIFO order or bytes, to reverse a winner."""
    first = validate_semantic_nomination(left, max_score_age_ns=max_score_age_ns)
    second = validate_semantic_nomination(right, max_score_age_ns=max_score_age_ns)
    if len(first["signatures"]) != len(second["signatures"]):
        _fail("score-reversal observations are not paired")
    changed = 0
    for before, after in zip(first["signatures"], second["signatures"]):
        for field in (
            "candidate_publication_order_event_ids",
            "shared_allocation_budget_blocks",
            "selected_bytes",
            "candidate_identities",
        ):
            if before[field] != after[field]:
                _fail(f"score-reversal control changed {field}")
        if before["chosen_identities"] != after["chosen_identities"]:
            changed += 1
    if changed == 0:
        _fail("semantic score reversal did not change the chosen session")
    return {
        "paired_observations": len(first["signatures"]),
        "winner_reversals": changed,
    }


def _semantic_events(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = result.get("wait_hint_events")
    if raw is None:
        # Focused CPU fixtures may isolate just the schema observations.
        raw = result.get("semantic_prefetch_session_nominations")
    return [
        event
        for event in _records(raw, "wait_hint_events")
        if event.get("schema") == "semantic-cross-session-prefetch-nomination-v1"
    ]


def _lifecycle(
    result: Mapping[str, Any], *, nominated_keys: frozenset[str]
) -> dict[str, Any]:
    events = _records(result.get("preservation_events"), "preservation_events")
    byte_rows: list[dict[str, int | None]] = []
    device_times: list[float | None] = []
    exposed_stalls: list[float | None] = []
    wrong_or_no_return = 0
    staging_before_demand = usable_before_demand = usable_before_forward = 0
    for event in events:
        attribution = event.get("prefetch_attribution")
        if not isinstance(attribution, Mapping):
            continue
        transfer = attribution.get("speculative_transfer")
        if not isinstance(transfer, Mapping):
            continue
        values: dict[str, int | None] = {}
        for field in (
            "submitted_bytes",
            "completed_bytes",
            "consumed_bytes",
            "discarded_bytes",
            "unconsumed_completed_bytes",
        ):
            raw = transfer.get(field)
            values[field] = _nonnegative_int(raw, field) if raw is not None else None
        byte_rows.append(values)
        completed = values["completed_bytes"]
        submitted_bytes = values["submitted_bytes"]
        used = values["consumed_bytes"]
        discarded = values["discarded_bytes"]
        unconsumed = values["unconsumed_completed_bytes"]
        if None not in (completed, submitted_bytes, used, discarded, unconsumed):
            if used + discarded + unconsumed != completed:
                _fail("speculative transfer outcome bytes do not reconcile")
            if completed > submitted_bytes:
                _fail("completed speculative bytes exceed submitted bytes")
        device_times.append(
            _finite(transfer["device_time_s"], "device_time_s")
            if transfer.get("device_time_s") is not None
            else None
        )
        if completed and used == 0:
            wrong_or_no_return += 1
        demand = attribution.get("demand_boundary_ns")
        staging = attribution.get("staging_ready_ns")
        install_submitted = attribution.get("destination_install_submitted_ns")
        usable = attribution.get("request_usable_ns")
        forward = attribution.get("request_first_forward_ns")
        exposed_stalls.append(
            _finite(
                attribution["exposed_blocking_stall_s"],
                "exposed_blocking_stall_s",
            )
            if attribution.get("exposed_blocking_stall_s") is not None
            else None
        )
        if used is not None and used > 0:
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (staging, install_submitted, usable, forward, demand)
            ):
                _fail("consumed prefetch lacks complete comparable lifecycle clocks")
            if not staging <= install_submitted <= usable <= forward:
                _fail("selected transfer/install/use lifecycle clocks are out of order")
            required = attribution.get("required_keys")
            prefetched = attribution.get("prefetched_keys")
            intersection = attribution.get("intersection_keys")
            if not all(
                isinstance(value, list) and value
                for value in (required, prefetched, intersection)
            ):
                _fail("consumed speculative bytes lack exact-key attribution")
            if not set(intersection).issubset(set(required)) or not set(
                intersection
            ).issubset(set(prefetched)):
                _fail("consumed exact keys do not link selected transfer to demand")
            if not set(intersection).issubset(nominated_keys):
                _fail("consumed exact keys do not link semantic nomination to demand")
            proof = attribution.get("cpu_residency_proof")
            if (
                not isinstance(proof, Mapping)
                or not proof
                or any(
                    key not in proof
                    or proof[key]
                    not in {"manager_lookup_hit", "cpu_ready_snapshot_exact_key"}
                    for key in intersection
                )
            ):
                _fail("consumed prefetch lacks exact DRAM survival proof")
        if isinstance(staging, int) and isinstance(demand, int) and staging <= demand:
            staging_before_demand += 1
        if isinstance(usable, int) and isinstance(demand, int) and usable <= demand:
            usable_before_demand += 1
        if isinstance(usable, int) and isinstance(forward, int) and usable <= forward:
            usable_before_forward += 1

    def observed_total(field: str) -> dict[str, Any]:
        values = [row[field] for row in byte_rows if row[field] is not None]
        return {
            "value": sum(values) if len(values) == len(byte_rows) else None,
            "observed_subtotal": sum(values) if values else None,
            "observed_count": len(values),
            "expected_count": len(byte_rows),
        }

    submitted = observed_total("submitted_bytes")
    completed_total = observed_total("completed_bytes")
    consumed = observed_total("consumed_bytes")
    discarded = observed_total("discarded_bytes")
    unconsumed = observed_total("unconsumed_completed_bytes")
    return {
        "submitted_bytes": submitted,
        "completed_bytes": completed_total,
        "consumed_bytes": consumed,
        "waste_bytes": {
            "value": (
                discarded["value"] + unconsumed["value"]
                if discarded["value"] is not None and unconsumed["value"] is not None
                else None
            ),
            "observed_subtotal": (
                (discarded["observed_subtotal"] or 0)
                + (unconsumed["observed_subtotal"] or 0)
                if discarded["observed_subtotal"] is not None
                or unconsumed["observed_subtotal"] is not None
                else None
            ),
        },
        "transfer_device_time_s": {
            "value": (
                sum(value for value in device_times if value is not None)
                if device_times and all(value is not None for value in device_times)
                else None
            ),
            "observed_count": sum(value is not None for value in device_times),
            "expected_count": len(device_times),
        },
        "exposed_blocking_stall_s": {
            "value": (
                sum(value for value in exposed_stalls if value is not None)
                if exposed_stalls and all(value is not None for value in exposed_stalls)
                else None
            ),
            "observed_count": sum(value is not None for value in exposed_stalls),
            "expected_count": len(exposed_stalls),
        },
        "wrong_or_no_return_nominations": wrong_or_no_return,
        "staging_ready_before_demand": staging_before_demand,
        "request_usable_before_demand": usable_before_demand,
        "request_usable_before_first_forward": usable_before_forward,
        "strict_activation_met": bool(
            consumed["value"] and staging_before_demand and usable_before_forward
        ),
    }


def analyze(
    *,
    msr: Mapping[str, Any],
    arc: Mapping[str, Any],
    lru: Mapping[str, Any],
    request_manifest: Mapping[str, Any],
    shared_prefetch: Mapping[str, Any] | None = None,
    score_reversal_events: Any | None = None,
) -> dict[str, Any]:
    runs = {"msr": msr, "arc": arc, "lru": lru}
    rows = {role: _request_rows(run) for role, run in runs.items()}
    request_ids = _validate_pairing(runs, rows)
    cohorts = _cohort_map(request_manifest)
    cohort_by_result_id = {
        rid: cohorts[_match_manifest_id(rid, cohorts)] for rid in request_ids
    }
    executed = {
        role: _execution_by_manifest_request(runs[role], rows[role]) for role in _ROLES
    }
    arm_metrics: dict[str, Any] = {}
    for role in _ROLES:
        total = sum(executed[role].values())
        returns = [
            value
            for rid, value in executed[role].items()
            if cohort_by_result_id[rid] == "return"
        ]
        ttft = [float(row["ttft_ms"]) for row in rows[role]]
        arm_metrics[role] = {
            "total_actual_prompt_tokens_locally_executed": total,
            "completed_requests": len(request_ids),
            "actual_prompt_tokens_per_completed_request": total / len(request_ids),
            "return_actual_prompt_tokens_locally_executed": sum(returns),
            "return_completed_requests": len(returns),
            "return_actual_prompt_tokens_per_completed_request": (
                sum(returns) / len(returns) if returns else None
            ),
            "median_ttft_ms": statistics.median(ttft),
        }
    comparisons: dict[str, Any] = {}
    for baseline in ("arc", "lru"):
        base = arm_metrics[baseline]
        primary = (
            100.0
            * (
                base["total_actual_prompt_tokens_locally_executed"]
                - arm_metrics["msr"]["total_actual_prompt_tokens_locally_executed"]
            )
            / base["total_actual_prompt_tokens_locally_executed"]
            if base["total_actual_prompt_tokens_locally_executed"] > 0
            else None
        )
        if base["median_ttft_ms"] <= 0:
            _fail(f"{baseline} median TTFT must be positive")
        ttft_delta = (
            100.0
            * (arm_metrics["msr"]["median_ttft_ms"] - base["median_ttft_ms"])
            / base["median_ttft_ms"]
        )
        comparisons[baseline] = {
            "executed_prompt_token_reduction_percent": primary,
            "median_ttft_delta_percent": ttft_delta,
            "ttft_allowance_met": ttft_delta <= TTFT_ALLOWANCE_PERCENT,
        }
    nomination_events = _semantic_events(msr)
    score_max_age_ms = _extra_config(msr).get("prefetch_score_max_age_ms")
    max_score_age_ns = (
        int(_finite(score_max_age_ms, "prefetch_score_max_age_ms") * 1_000_000)
        if score_max_age_ms is not None
        else None
    )
    nomination = validate_semantic_nomination(
        nomination_events, max_score_age_ns=max_score_age_ns
    )
    reversed_events = (
        score_reversal_events
        if score_reversal_events is not None
        else msr.get("semantic_prefetch_score_reversal_events")
    )
    score_reversal = (
        {
            "available": True,
            **validate_score_reversal(
                nomination_events,
                reversed_events,
                max_score_age_ns=max_score_age_ns,
            ),
        }
        if reversed_events is not None
        else {
            "available": False,
            "reason": "matched_score_reversal_observation_missing",
        }
    )
    nominated_keys = frozenset(nomination["selected_keys"])
    lifecycle = _lifecycle(msr, nominated_keys=nominated_keys)
    if shared_prefetch is None:
        semantic_incremental_attribution = {
            "available": False,
            "reason": "matched_causal_nonsemantic_shared_prefetch_control_missing",
        }
    else:
        shared_rows = _request_rows(shared_prefetch)
        _validate_shared_prefetch_control(
            msr, shared_prefetch, rows["msr"], shared_rows
        )
        shared_execution = _execution_by_manifest_request(shared_prefetch, shared_rows)
        shared_total = sum(shared_execution.values())
        msr_total = arm_metrics["msr"]["total_actual_prompt_tokens_locally_executed"]
        semantic_incremental_attribution = {
            "available": True,
            "control": "MSR retention with FIFO nomination",
            "control_total_actual_prompt_tokens_locally_executed": shared_total,
            "control_completed_requests": len(shared_rows),
            "semantic_selector_reduction_percent": (
                100.0 * (shared_total - msr_total) / shared_total
                if shared_total > 0
                else None
            ),
        }
    return {
        "schema": SCHEMA,
        "status": "cpu_contract_validated",
        "benefit_measured": False,
        "claim": "less total actual locally executed prompt work per completed request",
        "arm_metrics": arm_metrics,
        "comparisons": comparisons,
        "semantic_nomination_activation": nomination,
        "semantic_score_reversal": score_reversal,
        "prefetch_lifecycle_activation": lifecycle,
        "semantic_incremental_attribution": semantic_incremental_attribution,
        "strict_activation_met": bool(
            lifecycle["strict_activation_met"] and score_reversal["available"]
        ),
    }


def _maybe_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[Any], *, expected: int) -> dict[str, Any]:
    observed = [
        number for value in values if (number := _maybe_number(value)) is not None
    ]
    return {
        "mean": statistics.fmean(observed) if observed else None,
        "p50": _percentile(observed, 0.50),
        "p95": _percentile(observed, 0.95),
        "p99": _percentile(observed, 0.99),
        "worst": max(observed) if observed else None,
        "observed_count": len(observed),
        "expected_count": expected,
        "complete": len(observed) == expected,
        "unavailable_reason": (None if observed else "no finite raw observations"),
    }


def _normalized(result: Mapping[str, Any], name: str) -> float | None:
    item = result.get("normalized_metrics", {}).get(name)
    if isinstance(item, Mapping) and item.get("available") is True:
        return _maybe_number(item.get("value"))
    return None


def _event_duration_ms(
    event: Mapping[str, Any],
    millisecond_fields: Sequence[str],
    second_fields: Sequence[str],
) -> tuple[float | None, str | None]:
    for field in millisecond_fields:
        value = _maybe_number(event.get(field))
        if value is not None:
            return value, field
    for field in second_fields:
        value = _maybe_number(event.get(field))
        if value is not None:
            return value * 1000.0, field
    return None, None


def _history_cohort(meta: Mapping[str, Any]) -> str:
    explicit = meta.get("exact_history_cohort") or meta.get("claim_cohort")
    if explicit:
        return str(explicit)
    phase = str(meta.get("phase", "")).lower()
    if phase in {"warm", "warm1", "cold"}:
        return "first_observation"
    if phase in {"warm2", "first_return"}:
        return "first_exact_return"
    if phase in {"resume", "return", "repeat", "revisit"}:
        return "subsequent_exact_return"
    if phase in {"pollution", "no_return", "nonreturn"}:
        return "no_return_within_horizon"
    return "unclassified"


def _manifest_records(
    request_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records = _records(request_manifest.get("requests"), "request manifest requests")
    expanded = request_manifest.get("msr_expanded_campaign")
    history = (
        expanded.get("request_history", {}) if isinstance(expanded, Mapping) else {}
    )
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        request_id = _string(record.get("request_id"), "request manifest request_id")
        augmented = dict(record)
        history_record = (
            history.get(request_id) if isinstance(history, Mapping) else None
        )
        if isinstance(history_record, Mapping):
            augmented["exact_history_cohort"] = history_record.get("history_role")
            augmented["history_identity"] = dict(history_record)
            for field in (
                "intervening_request_count",
                "idle_gap_ms",
                "prefix_token_digest",
                "known_prefix_boundary_tokens",
                "namespace",
            ):
                if field in history_record:
                    augmented[field] = history_record[field]
            if history_record.get("known_prefix_token_digest"):
                augmented["prefix_token_digest"] = history_record[
                    "known_prefix_token_digest"
                ]
            if "planned_intervening_request_count" in history_record:
                augmented["intervening_request_count"] = history_record[
                    "planned_intervening_request_count"
                ]
            for field in ("session_id", "generation", "known_prefix_boundary_tokens"):
                if field in history_record:
                    augmented[field] = history_record[field]
        result[request_id] = augmented
    return result


def _manifest_record(
    request_id: str, records: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    return records[_match_manifest_id(request_id, records)]


def _actual_execution_or_unavailable(
    event: Mapping[str, Any],
) -> tuple[int | None, str | None]:
    actual = event.get("actual_prompt_tokens_locally_executed")
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
        return None, "terminal actual execution counter missing"
    cumulative = event.get("cumulative_prompt_tokens_locally_executed", actual)
    episodes = event.get("admission_episodes")
    if cumulative != actual or not isinstance(episodes, list):
        return None, "actual/cumulative/episode execution views incomplete"
    episode_values = [
        item.get("prompt_tokens_locally_executed")
        for item in episodes
        if isinstance(item, Mapping)
    ]
    if (
        len(episode_values) != len(episodes)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in episode_values
        )
        or sum(episode_values) != actual
    ):
        return None, "actual execution does not reconcile admission episodes"
    return actual, None


def _integrate_samples(
    samples: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    points = sorted(
        (
            (int(timestamp), float(value))
            for sample in samples
            if (timestamp := _maybe_number(sample.get("timestamp_ns"))) is not None
            and (value := _maybe_number(sample.get(field))) is not None
        ),
        key=lambda item: item[0],
    )
    # Deduplicate repeated copies of the same physical observation and reject
    # conflicting values at one clock instant.
    values_by_time: dict[int, set[float]] = {}
    for timestamp, value in points:
        values_by_time.setdefault(timestamp, set()).add(value)
    if any(len(values) != 1 for values in values_by_time.values()):
        _fail(f"conflicting occupancy values at one timestamp for {field}")
    unique = [
        (timestamp, next(iter(values)))
        for timestamp, values in sorted(values_by_time.items())
    ]
    if not unique:
        return {
            "peak": None,
            "time_integral": None,
            "sample_count": 0,
            "raw_sample_count": 0,
            "duplicate_sample_count": 0,
            "observed_interval_s": None,
            "maximum_sample_gap_s": None,
            "mean_sample_gap_s": None,
            "unavailable_reason": "no timestamped physical observations",
        }
    integral = None
    observed_interval = None
    gaps: list[float] = []
    if len(unique) >= 2:
        gaps = [
            (right_time - left_time) / 1_000_000_000
            for (left_time, _), (right_time, _) in pairwise(unique)
        ]
        integral = sum(
            value * gap for ((_, value), _), gap in zip(pairwise(unique), gaps)
        )
        observed_interval = (unique[-1][0] - unique[0][0]) / 1_000_000_000
    return {
        "peak": max(value for _, value in unique),
        "time_integral": integral,
        "sample_count": len(unique),
        "raw_sample_count": len(points),
        "duplicate_sample_count": len(points) - len(unique),
        "observed_interval_s": observed_interval,
        "maximum_sample_gap_s": max(gaps) if gaps else None,
        "mean_sample_gap_s": statistics.fmean(gaps) if gaps else None,
        "first_sample_ns": unique[0][0],
        "last_sample_ns": unique[-1][0],
        "unavailable_reason": (
            None if integral is not None else "fewer than two timestamped observations"
        ),
    }


def _occupancy_samples(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = result.get("physical_occupancy_evidence")
    if not isinstance(evidence, Mapping):
        return []
    samples: list[dict[str, Any]] = []
    for sample in evidence.get("phase_samples", []):
        if not isinstance(sample, Mapping):
            continue
        samples.append(
            {
                "timestamp_ns": sample.get("timestamp_ns"),
                "gpu_blocks": sample.get("occupied_gpu_blocks"),
                "dram_bytes": sample.get("occupied_cpu_bytes"),
                "staging_reserved_blocks": sample.get("speculative_reserved_blocks"),
                "staging_inflight_blocks": sample.get("speculative_inflight_blocks"),
                "staging_ready_blocks": sample.get("speculative_ready_blocks"),
                "running_requests": (sample.get("gauges") or {}).get(
                    "vllm:num_requests_running"
                ),
                "waiting_requests": (sample.get("gauges") or {}).get(
                    "vllm:num_requests_waiting"
                ),
                "source": "physical_occupancy_evidence.phase_samples",
                "source_class": "phase_boundary",
            }
        )
    for owner in evidence.get("event_samples", []):
        if not isinstance(owner, Mapping):
            continue
        for snapshot in owner.get("block_pool_snapshots", []):
            if isinstance(snapshot, Mapping) and snapshot.get("available") is True:
                gpu_blocks = snapshot.get("gpu_usable_context_blocks")
                if isinstance(gpu_blocks, list):
                    physical_ids = [
                        item.get("block_id")
                        for item in gpu_blocks
                        if isinstance(item, Mapping)
                    ]
                    if len(physical_ids) != len(gpu_blocks) or len(physical_ids) != len(
                        set(physical_ids)
                    ):
                        _fail("GPU occupancy snapshot duplicates physical block IDs")
                samples.append(
                    {
                        "timestamp_ns": snapshot.get("observed_at_ns"),
                        "gpu_blocks": snapshot.get("estimated_referenced_blocks"),
                        "gpu_usable_context_bytes": snapshot.get(
                            "gpu_usable_context_bytes"
                        ),
                        "shared_gpu_physical_blocks": (
                            sum(
                                item.get("ref_cnt", 0) > 1
                                for item in snapshot.get(
                                    "gpu_usable_context_blocks", []
                                )
                                if isinstance(item, Mapping)
                            )
                            if isinstance(
                                snapshot.get("gpu_usable_context_blocks"), list
                            )
                            else None
                        ),
                        "staging_reserved_blocks": snapshot.get(
                            "scheduler_prefetch_reserved_blocks_total"
                        ),
                        "staging_reserved_bytes": snapshot.get(
                            "staging_reserved_bytes"
                        ),
                        "source": "block_pool_snapshots",
                        "source_class": "event_authoritative",
                    }
                )
        for snapshot in owner.get("cpu_cache_snapshots", []):
            if isinstance(snapshot, Mapping) and snapshot.get("available") is True:
                resident_blocks = snapshot.get("resident_blocks")
                if not isinstance(resident_blocks, list):
                    _fail("CPU occupancy snapshot lacks physical resident blocks")
                cpu_ids = [
                    item.get("cpu_block_id")
                    for item in resident_blocks
                    if isinstance(item, Mapping)
                ]
                if len(cpu_ids) != len(resident_blocks) or len(cpu_ids) != len(
                    set(cpu_ids)
                ):
                    _fail("CPU occupancy snapshot duplicates physical block IDs")
                block_bytes = _maybe_number(snapshot.get("physical_block_bytes"))
                capacity = _maybe_number(snapshot.get("cpu_capacity_bytes"))
                resident_bytes = _maybe_number(snapshot.get("resident_bytes"))
                ready_bytes = _maybe_number(snapshot.get("ready_resident_bytes"))
                total_blocks = _maybe_number(snapshot.get("total_blocks"))
                allocated_blocks = _maybe_number(snapshot.get("allocated_blocks"))
                configured_capacity = _maybe_number(
                    evidence.get("configured_cpu_bytes")
                )
                if (
                    snapshot.get("accounting_consistent") is not True
                    or _maybe_number(snapshot.get("accounting_underflow_blocks")) != 0
                    or block_bytes is None
                    or capacity is None
                    or resident_bytes is None
                    or ready_bytes is None
                    or total_blocks is None
                    or allocated_blocks is None
                    or capacity != total_blocks * block_bytes
                    or (
                        configured_capacity is not None
                        and capacity != configured_capacity
                    )
                    or resident_bytes != len(resident_blocks) * block_bytes
                    or allocated_blocks != len(resident_blocks)
                    or ready_bytes
                    != sum(item.get("ready") is True for item in resident_blocks)
                    * block_bytes
                    or resident_bytes > capacity
                    or ready_bytes > resident_bytes
                ):
                    _fail("CPU physical occupancy snapshot does not reconcile")
                samples.append(
                    {
                        "timestamp_ns": snapshot.get("observed_at_ns"),
                        "dram_bytes": snapshot.get("resident_bytes"),
                        "dram_ready_bytes": snapshot.get("ready_resident_bytes"),
                        "dram_write_pending_bytes": snapshot.get("write_pending_bytes"),
                        "dram_physical_blocks": len(resident_blocks),
                        "shared_dram_physical_blocks": sum(
                            _maybe_number(item.get("ref_cnt")) is not None
                            and float(item["ref_cnt"]) > 1
                            for item in resident_blocks
                        ),
                        "dram_capacity_bytes": snapshot.get("cpu_capacity_bytes"),
                        "source": "cpu_cache_snapshots",
                        "source_class": "event_authoritative",
                    }
                )
    return samples


def _maximum_concurrency(rows: Sequence[Mapping[str, Any]]) -> int | None:
    boundaries: list[tuple[int, int]] = []
    for row in rows:
        start, end = row.get("send_ns"), row.get("completion_ns")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            continue
        boundaries.extend(((start, 1), (end, -1)))
    if not boundaries:
        return None
    current = peak = 0
    for _, delta in sorted(boundaries, key=lambda item: (item[0], item[1])):
        current += delta
        peak = max(peak, current)
    return peak


def _gpu_prefill_work(
    result: Mapping[str, Any], events: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    raw = result.get("gpu_prefill_work_accounting")
    if raw is None:
        return (
            {
                "available": False,
                "status": "producer-field-absent",
                "reason": "gpu_prefill_work_accounting absent from retained result",
                "device_time_s": None,
                "observed_device_time_s": None,
                "request_count": 0,
                "covered_request_count": 0,
                "actual_prompt_tokens_locally_executed": None,
                "eligible_prompt_tokens": None,
                "prompt_token_coverage_fraction": None,
                "observed_event_count": 0,
                "eligible_event_count": 0,
                "overlap_transfer_categories": {},
            },
            {},
        )
    accounting = _mapping(raw, "gpu_prefill_work_accounting")
    if accounting.get("schema") != "semantic-bounded-prefill-cuda-accounting-v1":
        _fail("unsupported GPU prefill accounting schema")
    available = accounting.get("available")
    if not isinstance(available, bool):
        _fail("GPU prefill accounting availability must be boolean")
    counter_fields = (
        "request_count",
        "covered_request_count",
        "actual_prompt_tokens_locally_executed",
        "eligible_prompt_tokens",
        "observed_event_count",
        "eligible_event_count",
    )
    if any(accounting.get(field) is None for field in counter_fields):
        if (
            available is False
            and isinstance(accounting.get("reason"), str)
            and accounting.get("device_time_s") is None
            and accounting.get("observed_device_time_s") is None
            and accounting.get("per_request") == []
        ):
            return (
                {
                    **dict(accounting),
                    "status": "invalid-unavailable",
                    **{field: None for field in counter_fields},
                    "overlap_transfer_categories": {},
                },
                {},
            )
        _fail("GPU prefill accounting counters are incomplete")
    counts = {
        field: _nonnegative_int(accounting.get(field), field)
        for field in counter_fields
    }
    if (
        counts["covered_request_count"] > counts["request_count"]
        or counts["eligible_prompt_tokens"]
        > counts["actual_prompt_tokens_locally_executed"]
        or counts["eligible_event_count"] > counts["observed_event_count"]
    ):
        _fail("GPU prefill accounting coverage counts are inconsistent")
    expected_complete = bool(
        counts["request_count"] > 0
        and counts["covered_request_count"] == counts["request_count"]
        and counts["eligible_prompt_tokens"]
        == counts["actual_prompt_tokens_locally_executed"]
    )
    if available != expected_complete:
        _fail("GPU prefill accounting completeness does not reconcile coverage")
    device_time = accounting.get("device_time_s")
    if available:
        device_time = _finite(device_time, "gpu prefill device_time_s")
        if device_time < 0:
            _fail("GPU prefill device time must be non-negative")
    elif device_time is not None:
        _fail("partial GPU prefill accounting cannot publish complete device time")
    observed_time = accounting.get("observed_device_time_s")
    if observed_time is not None:
        observed_time = _finite(observed_time, "gpu prefill observed_device_time_s")
        if observed_time < 0:
            _fail("observed GPU prefill device time must be non-negative")
    actual_tokens = counts["actual_prompt_tokens_locally_executed"]
    coverage = accounting.get("prompt_token_coverage_fraction")
    expected_coverage = (
        counts["eligible_prompt_tokens"] / actual_tokens if actual_tokens > 0 else None
    )
    if coverage != expected_coverage:
        _fail("GPU prefill prompt-token coverage fraction does not reconcile")
    rows = _records(accounting.get("per_request"), "GPU prefill per_request")
    if len(rows) != counts["request_count"]:
        _fail("GPU prefill per-request count does not reconcile")
    by_external: dict[str, Mapping[str, Any]] = {}
    actual_sum = eligible_sum = 0
    device_sum = 0.0
    observed_request_times: list[float] = []
    covered_rows = 0
    eligible_events_sum = 0
    for index, row in enumerate(rows):
        request_id = _string(
            row.get("request_id"), f"gpu prefill row {index} request_id"
        )
        if request_id in by_external:
            _fail("GPU prefill per-request identities are duplicated")
        actual = _nonnegative_int(
            row.get("actual_prompt_tokens_locally_executed"),
            "per-request actual prompt tokens",
        )
        eligible = _nonnegative_int(
            row.get("eligible_prompt_tokens"), "per-request eligible prompt tokens"
        )
        if eligible > actual or row.get("complete") is not (eligible == actual):
            _fail("GPU prefill per-request coverage is inconsistent")
        row_eligible_events = _nonnegative_int(
            row.get("eligible_event_count"), "per-request eligible event count"
        )
        _nonnegative_int(
            row.get("observed_event_count"), "per-request observed event count"
        )
        eligible_events_sum += row_eligible_events
        covered_rows += row.get("complete") is True
        row_device = row.get("device_time_s")
        if row.get("complete") is True:
            row_device = _finite(row_device, "per-request GPU prefill device_time_s")
            if row_device < 0:
                _fail("per-request GPU prefill device time must be non-negative")
            device_sum += row_device
        elif row_device is not None:
            _fail("partial request cannot publish complete GPU prefill time")
        row_observed = row.get("observed_device_time_s")
        if row_observed is not None:
            row_observed = _finite(
                row_observed, "per-request observed GPU prefill device time"
            )
            if row_observed < 0:
                _fail("per-request observed GPU prefill time must be non-negative")
            observed_request_times.append(row_observed)
        actual_sum += actual
        eligible_sum += eligible
        by_external[request_id] = row
    authoritative_actual = sum(
        _nonnegative_int(
            event.get("actual_prompt_tokens_locally_executed"),
            "authoritative actual prompt tokens",
        )
        for event in events.values()
    )
    if (
        actual_sum != counts["actual_prompt_tokens_locally_executed"]
        or eligible_sum != counts["eligible_prompt_tokens"]
        or authoritative_actual != actual_sum
    ):
        _fail("GPU prefill accounting token totals do not reconcile raw events")
    if (
        covered_rows != counts["covered_request_count"]
        or eligible_events_sum != counts["eligible_event_count"]
    ):
        _fail("GPU prefill per-request event coverage does not reconcile")
    if available and not math.isclose(
        float(device_time), device_sum, rel_tol=1e-12, abs_tol=1e-12
    ):
        _fail("GPU prefill complete device time does not reconcile request chunks")
    if bool(observed_request_times) != (observed_time is not None) or (
        observed_time is not None
        and not math.isclose(
            observed_time,
            sum(observed_request_times),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        _fail("GPU prefill observed subtotal does not reconcile request chunks")
    overlap_categories: dict[str, int] = {}
    seen_cuda: set[tuple[Any, ...]] = set()
    for event in events.values():
        for record in event.get("prefill_cuda_events", []):
            if not isinstance(record, Mapping):
                continue
            identity = (
                tuple(record.get("request_ids", [])),
                tuple(
                    item.get("scheduler_batch_id")
                    for item in record.get("execution_attribution", [])
                    if isinstance(item, Mapping)
                ),
                record.get("started_at_ns"),
                record.get("completed_at_ns"),
            )
            if identity in seen_cuda:
                continue
            seen_cuda.add(identity)
            for category in record.get("transfer_categories", []):
                overlap_categories[str(category)] = (
                    overlap_categories.get(str(category), 0) + 1
                )
    return (
        {
            **dict(accounting),
            "status": "complete" if available else "partial",
            "device_time_s": device_time,
            "observed_device_time_s": observed_time,
            "overlap_transfer_categories": overlap_categories,
            "overlap_accounting_rule": (
                "current-stream prefill intervals may overlap separately measured DMA; "
                "never add them to transfer device time as disjoint work"
            ),
        },
        by_external,
    )


def _gpu_model_runner_work(
    result: Mapping[str, Any],
    events: Mapping[str, Mapping[str, Any]],
    *,
    completed_requests: int,
    expected_requests: int | None,
) -> dict[str, Any]:
    raw = result.get("gpu_model_runner_work_accounting")
    if raw is None:
        return {
            "available": False,
            "status": "producer-field-absent",
            "reason": "gpu_model_runner_work_accounting absent from retained result",
            "device_time_s": None,
            "observed_device_time_s": None,
            "per_completed_request_s": None,
            "overlap_accounting_rule": None,
        }
    accounting = _mapping(raw, "gpu_model_runner_work_accounting")
    if accounting.get("schema") != "semantic-gpu-model-runner-work-accounting-v1":
        _fail("unsupported GPU model-runner accounting schema")
    available = accounting.get("available")
    if not isinstance(available, bool):
        _fail("GPU model-runner availability must be boolean")
    device = accounting.get("device_time_s")
    observed = accounting.get("observed_device_time_s")
    if available:
        device = _finite(device, "GPU model-runner device_time_s")
        observed = _finite(observed, "GPU model-runner observed_device_time_s")
        if (
            device < 0
            or observed < 0
            or not math.isclose(device, observed, rel_tol=1e-12, abs_tol=1e-12)
        ):
            _fail("complete GPU model-runner device time is inconsistent")
    else:
        if device is not None:
            _fail("partial GPU model-runner accounting cannot publish total time")
        if observed is not None:
            observed = _finite(observed, "GPU model-runner observed_device_time_s")
            if observed < 0:
                _fail("GPU model-runner subtotal must be non-negative")
    detailed_fields = (
        "execution_count",
        "sequence_first",
        "sequence_last",
        "expected_request_batch_count",
        "observed_request_batch_count",
        "scheduled_tokens",
        "prompt_tokens",
        "decode_tokens",
    )
    details_present = all(
        accounting.get(field) is not None for field in detailed_fields
    )
    if available and not details_present:
        _fail("complete GPU model-runner accounting lacks coverage details")
    normalized = dict(accounting)
    if details_present:
        values = {
            field: _nonnegative_int(accounting.get(field), field)
            for field in detailed_fields
        }
        execution_count = values["execution_count"]
        for flag in ("sequence_contiguous", "chronology_valid", "owner_sets_valid"):
            if not isinstance(accounting.get(flag), bool):
                _fail(f"GPU model-runner {flag} must be boolean")
        computed_complete = bool(
            execution_count > 0
            and values["sequence_first"] == 1
            and values["sequence_last"] == execution_count
            and accounting["sequence_contiguous"] is True
            and accounting["chronology_valid"] is True
            and accounting["owner_sets_valid"] is True
            and values["expected_request_batch_count"]
            == values["observed_request_batch_count"]
        )
        if (
            available != computed_complete
            or values["prompt_tokens"] > values["scheduled_tokens"]
            or values["decode_tokens"]
            != values["scheduled_tokens"] - values["prompt_tokens"]
        ):
            _fail("GPU model-runner sequence/batch/token coverage is inconsistent")
        expected_batch_count = 0
        expected_scheduled = 0
        expected_prompt = 0
        seen_batches: set[tuple[str, int]] = set()
        for event in events.values():
            internal_id = _string(
                event.get("request_id"), "GPU model-runner event request_id"
            )
            for episode in _records(
                event.get("admission_episodes"), "admission_episodes"
            ):
                batch_id = _nonnegative_int(
                    episode.get("scheduler_batch_id"), "scheduler_batch_id"
                )
                batch_key = (internal_id, batch_id)
                if batch_key in seen_batches:
                    _fail("GPU model-runner admission batch identity duplicated")
                seen_batches.add(batch_key)
                scheduled = _nonnegative_int(
                    episode.get("scheduled_tokens"), "scheduled_tokens"
                )
                prompt = _nonnegative_int(
                    episode.get("prompt_tokens_locally_executed"),
                    "prompt_tokens_locally_executed",
                )
                if prompt > scheduled:
                    _fail("GPU model-runner admission prompt exceeds scheduled tokens")
                expected_batch_count += 1
                expected_scheduled += scheduled
                expected_prompt += prompt
        if values["expected_request_batch_count"] != expected_batch_count:
            _fail("GPU model-runner expected admission batch count does not reconcile")
        if available and (
            values["scheduled_tokens"] != expected_scheduled
            or values["prompt_tokens"] != expected_prompt
        ):
            _fail("GPU model-runner totals do not reconcile admission batches")
        if not available and (
            values["scheduled_tokens"] > expected_scheduled
            or values["prompt_tokens"] > expected_prompt
        ):
            _fail("GPU model-runner subtotal exceeds admission batch totals")
        kinds = _mapping(accounting.get("batch_kind_counts"), "batch_kind_counts")
        kind_counts = {
            kind: _nonnegative_int(kinds.get(kind), f"batch_kind_counts.{kind}")
            for kind in ("prompt_only", "decode_only", "mixed")
        }
        if sum(kind_counts.values()) != execution_count:
            _fail("GPU model-runner batch-kind counts do not reconcile")
        for field in (
            "transfer_category_execution_counts",
            "transfer_category_scheduled_tokens",
        ):
            categories = _mapping(accounting.get(field), field)
            for category, value in categories.items():
                _string(category, f"{field} category")
                _nonnegative_int(value, f"{field}.{category}")
    if available and (
        accounting.get("reason") != "complete_exact_scheduler_batch_coverage"
        or accounting.get("attribution_scope")
        != (
            "TP1 cell-total model-runner current CUDA stream; shared kernels "
            "not split by request; transfer overlap not disambiguated"
        )
        or accounting.get("synchronization_overhead")
        != "one CUDA end-event synchronize per real step"
    ):
        _fail("GPU model-runner attribution scope is inconsistent")
    complete_denominator = bool(
        expected_requests is not None
        and expected_requests > 0
        and completed_requests == expected_requests
    )
    normalized.update(
        {
            "status": "complete" if available else "partial-or-unavailable",
            "device_time_s": device,
            "observed_device_time_s": observed,
            "per_completed_request_s": (
                device / completed_requests
                if available and complete_denominator
                else None
            ),
            "complete_request_denominator": complete_denominator,
            "overlap_accounting_rule": (
                "cell-total current-stream time may overlap separately measured DMA; "
                "never add it to transfer device time as disjoint work"
            ),
        }
    )
    return normalized


def extract_cell_metrics(
    result: Mapping[str, Any], request_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Extract one real execute-cell record without turning absence into zero."""
    if result.get("schema") != "ep-campaign-cell-result-v1":
        _fail("unsupported execute-cell result schema")
    manifest = _manifest_records(request_manifest)
    input_identities = sorted(
        {
            str(
                record.get("prefix_token_digest")
                or record.get("context_sha256")
                or record.get("prompt_token_digest")
            )
            for record in manifest.values()
            if record.get("prefix_token_digest")
            or record.get("context_sha256")
            or record.get("prompt_token_digest")
        }
    )
    input_cohort_identity = (
        hashlib.sha256(
            json.dumps(input_identities, separators=(",", ":")).encode()
        ).hexdigest()
        if input_identities
        else None
    )
    events = {
        event.get("external_request_id"): event
        for event in _records(result.get("preservation_events"), "preservation_events")
        if event.get("external_request_id")
    }
    gpu_prefill_work, gpu_prefill_by_request = _gpu_prefill_work(result, events)
    request_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    ttl_rows: list[dict[str, Any]] = []
    policy_durations_ms: list[float] = []
    displacement_rows: list[dict[str, Any]] = []
    for request in _records(result.get("requests"), "requests"):
        request_id = _string(request.get("manifest_request_id"), "manifest_request_id")
        meta = _manifest_record(request_id, manifest)
        event = events.get(request.get("external_request_id"), {})
        actual, actual_reason = _actual_execution_or_unavailable(event)
        reused = _maybe_number(event.get("reused_tokens"))
        restored = _maybe_number(event.get("loaded_tokens"))
        gpu_reuse = (
            reused - restored if reused is not None and restored is not None else None
        )
        receipts = [
            int(value)
            for value in request.get("token_receipt_ns", [])
            if _maybe_number(value) is not None
        ]
        intervals = [
            (right - left) / 1_000_000
            for left, right in pairwise(receipts)
            if right >= left
        ]
        timing = event.get("server_timing") if isinstance(event, Mapping) else None
        timing = timing if isinstance(timing, Mapping) else {}
        queue_ns = _maybe_number(timing.get("queue_delay_ns"))
        service_ns = _maybe_number(timing.get("service_time_ns"))
        request_prefill = gpu_prefill_by_request.get(
            str(request.get("external_request_id"))
        )
        request_prefill = (
            request_prefill if isinstance(request_prefill, Mapping) else {}
        )
        gpu_prefill_ms = (
            float(request_prefill["device_time_s"]) * 1000
            if request_prefill.get("complete") is True
            and _maybe_number(request_prefill.get("device_time_s")) is not None
            else None
        )
        row = {
            "request_id": request_id,
            "external_request_id": request.get("external_request_id"),
            "cohort": _history_cohort(meta),
            "session_id": event.get("session_id", meta.get("session_id")),
            "generation": event.get("generation", meta.get("generation")),
            "source_id": event.get("source_id", meta.get("source_id")),
            "key_chain_digest": meta.get(
                "prefix_token_digest", meta.get("key_chain_digest")
            ),
            "prefix_boundary_tokens": meta.get("known_prefix_boundary_tokens"),
            "success": request.get("success") is True,
            "terminal_complete": request.get("terminal_complete") is True,
            "http_status": request.get("http_status"),
            "error": request.get("error"),
            "ttft_ms": _maybe_number(request.get("ttft_ms")),
            "release_to_first_token_ms": (
                (request["first_token_ns"] - request["planned_release_ns"]) / 1_000_000
                if isinstance(request.get("first_token_ns"), int)
                and isinstance(request.get("planned_release_ns"), int)
                and request["first_token_ns"] >= request["planned_release_ns"]
                else None
            ),
            "itl_ms": statistics.fmean(intervals) if intervals else None,
            "e2e_ms": _maybe_number(request.get("total_ms")),
            "queue_ms": queue_ns / 1_000_000 if queue_ns is not None else None,
            "service_ms": service_ns / 1_000_000 if service_ns is not None else None,
            "actual_prompt_tokens_locally_executed": actual,
            "actual_execution_unavailable_reason": actual_reason,
            "unique_required_tokens": _maybe_number(event.get("required_tokens")),
            "unique_recomputed_tokens": _maybe_number(event.get("recomputed_tokens")),
            "gpu_only_reused_tokens": gpu_reuse,
            "dram_restored_tokens": restored,
            "gpu_prefill_ms": gpu_prefill_ms,
            "gpu_prefill_observed_subtotal_ms": (
                float(request_prefill["observed_device_time_s"]) * 1000
                if _maybe_number(request_prefill.get("observed_device_time_s"))
                is not None
                else None
            ),
            "gpu_prefill_prompt_token_coverage_fraction": request_prefill.get(
                "coverage_fraction"
            ),
            "gpu_prefill_observed_events": request_prefill.get("observed_event_count"),
            "gpu_prefill_eligible_events": request_prefill.get("eligible_event_count"),
            "preemption_count": _maybe_number(event.get("preemption_count")),
            "planned_release_ns": request.get("planned_release_ns"),
            "release_lateness_ms": (
                request["release_lateness_ns"] / 1_000_000
                if _maybe_number(request.get("release_lateness_ns")) is not None
                else None
            ),
            "send_ns": request.get("send_ns"),
            "completion_ns": request.get("completion_ns"),
            "output_token_digest": request.get("output_token_digest"),
            "output_tokens": (
                len(request["output_token_ids"])
                if isinstance(request.get("output_token_ids"), list)
                else None
            ),
        }
        request_rows.append(row)
        for field in ("admission_decisions", "retention_decisions", "policy_decisions"):
            for decision_record in event.get(field, []):
                if not isinstance(decision_record, Mapping):
                    continue
                duration, _ = _event_duration_ms(
                    decision_record,
                    (
                        "policy_duration_ms",
                        "scoring_duration_ms",
                        "decision_duration_ms",
                        "duration_ms",
                    ),
                    (
                        "policy_duration_s",
                        "scoring_duration_s",
                        "decision_duration_s",
                        "duration_s",
                    ),
                )
                if duration is not None:
                    policy_durations_ms.append(duration)
        lifecycle = event.get("key_lifecycle")
        if isinstance(lifecycle, Mapping):
            for eviction in lifecycle.get("cpu_evictions", []):
                if isinstance(eviction, Mapping):
                    displacement_rows.append(
                        {
                            "request_id": request_id,
                            "kind": "cpu_eviction",
                            "key": eviction.get("key"),
                            "observed_at_ns": eviction.get("observed_at_ns"),
                            "provenance": eviction.get("provenance"),
                        }
                    )
        attr = event.get("prefetch_attribution")
        if isinstance(attr, Mapping):
            transfer = attr.get("speculative_transfer")
            transfer = transfer if isinstance(transfer, Mapping) else {}
            lifecycle_rows.append(
                {
                    "request_id": request_id,
                    "event_id": (attr.get("wait_hint") or {}).get("event_id")
                    if isinstance(attr.get("wait_hint"), Mapping)
                    else None,
                    "status": attr.get("status"),
                    "staging_ready_ns": attr.get("staging_ready_ns"),
                    "demand_boundary_ns": attr.get("demand_boundary_ns"),
                    "install_submitted_ns": attr.get(
                        "destination_install_submitted_ns"
                    ),
                    "request_usable_ns": attr.get("request_usable_ns"),
                    "first_forward_ns": attr.get("request_first_forward_ns"),
                    "submitted_bytes": transfer.get("submitted_bytes"),
                    "completed_bytes": transfer.get("completed_bytes"),
                    "consumed_bytes": transfer.get("consumed_bytes"),
                    "discarded_bytes": transfer.get("discarded_bytes"),
                    "uncompleted_bytes": transfer.get("uncompleted_submitted_bytes"),
                    "unconsumed_bytes": transfer.get("unconsumed_completed_bytes"),
                    "cancelled_jobs": transfer.get("cancelled_job_count"),
                    "failed_jobs": transfer.get("failed_job_count"),
                    "device_time_s": transfer.get("device_time_s"),
                    "exposed_stall_s": attr.get("exposed_blocking_stall_s"),
                    "required_keys": attr.get("required_keys"),
                    "prefetched_keys": attr.get("prefetched_keys"),
                    "intersection_keys": attr.get("intersection_keys"),
                }
            )
    successful = [
        row for row in request_rows if row["success"] and row["terminal_complete"]
    ]
    cohorts: dict[str, Any] = {}
    for cohort in sorted({row["cohort"] for row in request_rows} | {"whole_trace"}):
        selected = (
            request_rows
            if cohort == "whole_trace"
            else [row for row in request_rows if row["cohort"] == cohort]
        )
        latency_selected = [
            row for row in selected if row["success"] and row["terminal_complete"]
        ]
        cohorts[cohort] = {
            metric: {
                **_distribution(
                    [row[metric] for row in latency_selected],
                    expected=len(latency_selected),
                ),
                "population": "successful terminal requests only",
            }
            for metric in ("ttft_ms", "itl_ms", "e2e_ms", "queue_ms", "service_ms")
        }
        for metric in (
            "actual_prompt_tokens_locally_executed",
            "gpu_only_reused_tokens",
            "dram_restored_tokens",
            "unique_recomputed_tokens",
        ):
            values = [row[metric] for row in selected if row[metric] is not None]
            completed_selected = sum(
                row["success"] and row["terminal_complete"] for row in selected
            )
            complete = len(values) == len(selected) == completed_selected
            cohorts[cohort][metric] = {
                "value": sum(values) if complete else None,
                "per_completed_request": (
                    sum(values) / completed_selected
                    if complete and completed_selected > 0
                    else None
                ),
                "observed_subtotal": sum(values) if values else None,
                "observed_count": len(values),
                "expected_count": len(selected),
                "completed_count": completed_selected,
                "unavailable_reason": (
                    None
                    if complete
                    else "request observations or completions incomplete"
                ),
            }
    # TTL is defined only for exact-session histories with measured event clocks.
    by_session: dict[
        tuple[Any, ...], list[tuple[dict[str, Any], Mapping[str, Any]]]
    ] = {}
    for row in request_rows:
        meta = _manifest_record(row["request_id"], manifest)
        identity = (
            row["session_id"],
            row["generation"],
            row["key_chain_digest"],
            row["prefix_boundary_tokens"],
        )
        if all(value not in (None, "") for value in identity):
            by_session.setdefault(identity, []).append((row, meta))
    horizon = max(
        (
            row["completion_ns"]
            for row in request_rows
            if isinstance(row["completion_ns"], int)
        ),
        default=None,
    )
    for identity, history in by_session.items():
        history.sort(key=lambda item: item[1].get("request_index", 0))
        for previous, current in pairwise(history):
            prior, _ = previous
            row, meta = current
            idle_ns = (
                row["send_ns"] - prior["completion_ns"]
                if isinstance(row["send_ns"], int)
                and isinstance(prior["completion_ns"], int)
                and row["send_ns"] >= prior["completion_ns"]
                else None
            )
            actual_intervening = (
                sum(
                    candidate is not prior
                    and candidate is not row
                    and isinstance(candidate.get("completion_ns"), int)
                    and prior["completion_ns"]
                    < candidate["completion_ns"]
                    <= row["send_ns"]
                    for candidate in request_rows
                )
                if idle_ns is not None
                else None
            )
            survived = (
                row["dram_restored_tokens"] > 0
                if row["dram_restored_tokens"] is not None
                else None
            )
            ttl_rows.append(
                {
                    "identity": identity,
                    "return_request_id": row["request_id"],
                    "idle_ms": idle_ns / 1_000_000 if idle_ns is not None else None,
                    "actual_intervening_completed_requests": actual_intervening,
                    "planned_intervening_requests": meta.get(
                        "intervening_request_count",
                        meta.get("intervening_traffic_requests"),
                    ),
                    "planned_idle_gap_ms": meta.get("idle_gap_ms"),
                    "hint_lead_offset_ms": meta.get("hint_lead_ms"),
                    "dram_survived": survived,
                    "timely_reuse": None,
                    "timely_reuse_unavailable_reason": "requires matched baseline latency constraint",
                    "censored": False,
                    "horizon_ns": horizon,
                }
            )
        final_row, final_meta = history[-1]
        if horizon is not None and final_row["cohort"] in {
            "no_return_within_horizon",
            "cold_no_return",
            "censored",
        }:
            row = final_row
            ttl_rows.append(
                {
                    "identity": identity,
                    "return_request_id": None,
                    "idle_ms": (
                        (horizon - row["completion_ns"]) / 1_000_000
                        if isinstance(row["completion_ns"], int)
                        and horizon >= row["completion_ns"]
                        else None
                    ),
                    "dram_survived": None,
                    "timely_reuse": None,
                    "censored": True,
                    "horizon_ns": horizon,
                    "planned_intervening_traffic": final_meta.get(
                        "intervening_request_count"
                    ),
                }
            )
    occupancy = _occupancy_samples(result)
    phase_occupancy = [
        sample for sample in occupancy if sample.get("source_class") == "phase_boundary"
    ]
    event_occupancy = [
        sample
        for sample in occupancy
        if sample.get("source_class") == "event_authoritative"
    ]
    occupancy_summary: dict[str, Any] = {}
    for field in (
        "gpu_blocks",
        "dram_bytes",
        "staging_reserved_blocks",
        "staging_inflight_blocks",
        "staging_ready_blocks",
        "dram_ready_bytes",
        "dram_write_pending_bytes",
        "gpu_usable_context_bytes",
        "shared_gpu_physical_blocks",
        "staging_reserved_bytes",
        "dram_physical_blocks",
        "shared_dram_physical_blocks",
        "dram_capacity_bytes",
    ):
        event_summary = _integrate_samples(event_occupancy, field)
        phase_summary = _integrate_samples(phase_occupancy, field)
        authoritative = (
            event_summary if event_summary["sample_count"] > 0 else phase_summary
        )
        occupancy_summary[field] = {
            **authoritative,
            "selected_source": (
                "event_authoritative"
                if event_summary["sample_count"] > 0
                else "phase_boundary"
                if phase_summary["sample_count"] > 0
                else None
            ),
            "event_authoritative": event_summary,
            "phase_boundary": phase_summary,
            "peak_difference_event_minus_phase": (
                event_summary["peak"] - phase_summary["peak"]
                if event_summary["peak"] is not None
                and phase_summary["peak"] is not None
                else None
            ),
        }
    arrivals = sorted(
        row["send_ns"] for row in request_rows if isinstance(row["send_ns"], int)
    )
    interarrival_ms = [(right - left) / 1_000_000 for left, right in pairwise(arrivals)]
    planned_arrivals = sorted(
        row["planned_release_ns"]
        for row in request_rows
        if isinstance(row["planned_release_ns"], int)
    )
    planned_span_s = (
        (planned_arrivals[-1] - planned_arrivals[0]) / 1_000_000_000
        if len(planned_arrivals) >= 2
        else None
    )
    duration_s = (
        (
            max(
                row["completion_ns"]
                for row in request_rows
                if isinstance(row["completion_ns"], int)
            )
            - min(arrivals)
        )
        / 1_000_000_000
        if arrivals
        and any(isinstance(row["completion_ns"], int) for row in request_rows)
        else None
    )
    physical = result.get("physical_transfer_accounting")
    physical = dict(physical) if isinstance(physical, Mapping) else None
    raw_resources = result.get("resource_accounting")
    raw_resources = raw_resources if isinstance(raw_resources, Mapping) else {}
    nomination_events = [
        event
        for event in result.get("wait_hint_events", [])
        if isinstance(event, Mapping)
        and event.get("schema") == "semantic-cross-session-prefetch-nomination-v1"
    ]
    nomination_cpu = [
        event.get("nomination_cpu_ns") / 1_000_000
        for event in nomination_events
        if _maybe_number(event.get("nomination_cpu_ns")) is not None
    ]
    for scoring in result.get("async_scoring_events", []):
        if not isinstance(scoring, Mapping):
            continue
        duration, _ = _event_duration_ms(
            scoring,
            ("policy_duration_ms", "scoring_duration_ms", "duration_ms"),
            ("policy_duration_s", "scoring_duration_s", "duration_s"),
        )
        if duration is not None:
            policy_durations_ms.append(duration)
    drain_status = (
        (result.get("drain") or {}).get("status")
        if isinstance(result.get("drain"), Mapping)
        else None
    )
    expected_requests = (
        len(request_rows)
        if result.get("valid") is True
        and drain_status in {"drained", "complete", "success"}
        else None
    )
    gpu_model_runner_work = _gpu_model_runner_work(
        result,
        events,
        completed_requests=len(successful),
        expected_requests=expected_requests,
    )
    occupancy_evidence = result.get("physical_occupancy_evidence")
    occupancy_evidence = (
        occupancy_evidence if isinstance(occupancy_evidence, Mapping) else {}
    )
    try:
        runtime_config = _extra_config(result)
    except AttributionError:
        runtime_config = {}
    activation: dict[str, Any] = {
        "semantic_nomination_valid": False,
        "combined_activation_met": False,
        "unavailable_reason": "semantic nomination observations absent",
    }
    if nomination_events:
        try:
            max_age_ms = runtime_config.get("prefetch_score_max_age_ms")
            max_age_ns = (
                int(_finite(max_age_ms, "prefetch_score_max_age_ms") * 1_000_000)
                if max_age_ms is not None
                else None
            )
            nomination_summary = validate_semantic_nomination(
                nomination_events, max_score_age_ns=max_age_ns
            )
            nominated = frozenset(nomination_summary["selected_keys"])
            lifecycle_summary = _lifecycle(result, nominated_keys=nominated)
            activation = {
                "semantic_nomination_valid": True,
                "combined_activation_met": lifecycle_summary["strict_activation_met"],
                "nomination": nomination_summary,
                "lifecycle": lifecycle_summary,
                "unavailable_reason": (
                    None
                    if lifecycle_summary["strict_activation_met"]
                    else "exact staging/install/ack/consumption lifecycle inactive"
                ),
            }
        except AttributionError as error:
            activation["unavailable_reason"] = str(error)
    return {
        "schema": "msr-combined-prefetch-cell-aggregation-v1",
        "arm_id": result.get("arm_id"),
        "cell_id": result.get("cell_id"),
        "profile_id": result.get("profile_id"),
        "replicate_id": result.get("replicate_id"),
        "seed": result.get("seed"),
        "valid": result.get("valid"),
        "activation": activation,
        "input_cohort_identity": input_cohort_identity,
        "configuration": {
            "configured_dram_bytes": occupancy_evidence.get(
                "configured_cpu_bytes", runtime_config.get("cpu_bytes_to_use")
            ),
            "configured_gpu_blocks": occupancy_evidence.get("configured_gpu_blocks"),
            "workset_exact_contexts": len(
                {
                    (
                        row["session_id"],
                        row["generation"],
                        row["key_chain_digest"],
                        row["prefix_boundary_tokens"],
                    )
                    for row in request_rows
                    if all(
                        value not in (None, "")
                        for value in (
                            row["session_id"],
                            row["generation"],
                            row["key_chain_digest"],
                            row["prefix_boundary_tokens"],
                        )
                    )
                }
            ),
            "runtime_extra_config": dict(runtime_config),
        },
        "provenance": {
            **{
                field: result.get(field)
                for field in (
                    "campaign_manifest_sha256",
                    "request_manifest_sha256",
                    "baseline_profile_hash",
                    "order_schedule_digest",
                    "arm_config_hash",
                    "plan_digest",
                    "metric_contract_digest",
                )
            },
            "runtime_revision": (result.get("deployed_runtime") or {}).get("revision")
            if isinstance(result.get("deployed_runtime"), Mapping)
            else None,
            "runtime_tree": (result.get("deployed_runtime") or {}).get("tree")
            if isinstance(result.get("deployed_runtime"), Mapping)
            else None,
            "extension_revision": (result.get("deployed_extension") or {}).get(
                "revision"
            )
            if isinstance(result.get("deployed_extension"), Mapping)
            else None,
            "extension_tree": (result.get("deployed_extension") or {}).get("tree")
            if isinstance(result.get("deployed_extension"), Mapping)
            else None,
        },
        "request_counts": {
            "expected": expected_requests,
            "observed": len(request_rows),
            "completed": len(successful),
            "failed": sum(not row["success"] for row in request_rows),
            "timeouts": sum(
                "timeout" in str(row.get("error", "")).lower() for row in request_rows
            ),
            "missing": (
                expected_requests - len(request_rows)
                if expected_requests is not None
                else None
            ),
            "missing_unavailable_reason": (
                None
                if expected_requests is not None
                else "planned cell request set unavailable or cell did not drain validly"
            ),
        },
        "requests": request_rows,
        "cohorts": cohorts,
        "cell_serving": {
            "episode_makespan_ms": _normalized(result, "episode_makespan_ms"),
            "throughput_output_tokens_per_s": _normalized(
                result, "throughput_tokens_per_s"
            ),
            "throughput_completed_requests_per_s": (
                len(successful) / duration_s if duration_s and duration_s > 0 else None
            ),
            "preemptions": _normalized(result, "preemptions"),
            "drain_status": drain_status,
        },
        "prefetch_lifecycle": lifecycle_rows,
        "physical_transfer_accounting": physical,
        "occupancy_samples": occupancy,
        "occupancy_event_samples": event_occupancy,
        "occupancy_phase_samples": phase_occupancy,
        "occupancy_phase_peaks_reported": occupancy_evidence.get("phase_peaks"),
        "occupancy": occupancy_summary,
        "ttl_observations": ttl_rows,
        "displacements": displacement_rows,
        "resources": {
            "gpu_model_runner_work": gpu_model_runner_work,
            "gpu_prefill_ms": {
                "value": (
                    float(gpu_prefill_work["device_time_s"]) * 1000
                    if gpu_prefill_work.get("available") is True
                    else None
                ),
                "observed_subtotal": (
                    float(gpu_prefill_work["observed_device_time_s"]) * 1000
                    if _maybe_number(gpu_prefill_work.get("observed_device_time_s"))
                    is not None
                    else None
                ),
                "observed_requests": gpu_prefill_work.get("covered_request_count"),
                "expected_requests": gpu_prefill_work.get("request_count"),
                "actual_prompt_tokens_locally_executed": gpu_prefill_work.get(
                    "actual_prompt_tokens_locally_executed"
                ),
                "eligible_prompt_tokens": gpu_prefill_work.get(
                    "eligible_prompt_tokens"
                ),
                "prompt_token_coverage_fraction": gpu_prefill_work.get(
                    "prompt_token_coverage_fraction"
                ),
                "observed_event_count": gpu_prefill_work.get("observed_event_count"),
                "eligible_event_count": gpu_prefill_work.get("eligible_event_count"),
                "complete": gpu_prefill_work.get("available") is True,
                "status": gpu_prefill_work.get("status"),
                "unavailable_reason": (
                    None
                    if gpu_prefill_work.get("available") is True
                    else gpu_prefill_work.get("reason")
                ),
                "attribution_scope": gpu_prefill_work.get("attribution_scope"),
                "overlap_transfer_categories": gpu_prefill_work.get(
                    "overlap_transfer_categories", {}
                ),
                "overlap_accounting_rule": gpu_prefill_work.get(
                    "overlap_accounting_rule"
                ),
            },
            "nomination_cpu_ms": {
                "observed_subtotal": sum(nomination_cpu) if nomination_cpu else None,
                "observed_events": len(nomination_cpu),
                "expected_events": len(nomination_events),
            },
            "policy_decision_elapsed_ms": {
                "observed_subtotal": (
                    sum(policy_durations_ms) if policy_durations_ms else None
                ),
                "observed_events": len(policy_durations_ms),
                "note": "elapsed decision/scoring time, not process CPU utilization",
            },
            "causal_nomination_observations": len(nomination_events),
            "process_cpu_seconds": {
                "value": _maybe_number(raw_resources.get("process_cpu_seconds")),
                "status": (
                    "observed"
                    if _maybe_number(raw_resources.get("process_cpu_seconds"))
                    is not None
                    else "GPU-observation-missing"
                ),
            },
            "process_rss_byte_seconds": {
                "value": _maybe_number(raw_resources.get("process_rss_byte_seconds")),
                "status": (
                    "observed"
                    if _maybe_number(raw_resources.get("process_rss_byte_seconds"))
                    is not None
                    else "GPU-observation-missing"
                ),
            },
            "energy_joules": {
                "value": _maybe_number(raw_resources.get("energy_joules")),
                "status": (
                    "observed"
                    if _maybe_number(raw_resources.get("energy_joules")) is not None
                    else "optional-not-required"
                ),
            },
            "cost_per_completed_request": {
                "value": None,
                "status": "not-implemented-without-explicit-price-basis",
                "unavailable_reason": "no explicit resource price/cost basis supplied",
            },
        },
        "load": {
            "maximum_observed_concurrency": _maximum_concurrency(request_rows),
            "offered_requests_per_s": (
                (len(planned_arrivals) - 1) / planned_span_s
                if planned_span_s and planned_span_s > 0
                else None
            ),
            "planned_arrival_count": len(planned_arrivals),
            "interarrival_ms": _distribution(
                interarrival_ms, expected=max(0, len(arrivals) - 1)
            ),
            "observed_duration_s": duration_s,
            "release_lateness_ms": _distribution(
                [row["release_lateness_ms"] for row in request_rows],
                expected=len(request_rows),
            ),
            "completed_requests_per_s": (
                len(successful) / duration_s if duration_s and duration_s > 0 else None
            ),
        },
    }


_COVERAGE_SPEC = (
    (
        "executed_prompt_tokens",
        "preservation_events[].actual_prompt_tokens_locally_executed",
        "sum once per terminal request / completed requests",
        "post-GPU aggregate",
    ),
    (
        "gpu_only_reuse",
        "preservation_events[].reused_tokens-loaded_tokens",
        "sum distinct GPU-resident exact reuse",
        "post-GPU aggregate",
    ),
    (
        "dram_restore",
        "preservation_events[].loaded_tokens",
        "sum exact DRAM-restored tokens",
        "post-GPU aggregate",
    ),
    (
        "exact_history_cohorts",
        "request manifest session/generation/source/prefix in immutable order",
        "first observation/first return/subsequent/no-return groups",
        "CPU construction + post-GPU",
    ),
    (
        "request_latency",
        "requests[].ttft_ms,total_ms,token_receipt_ns",
        "mean/p50/p95/p99/worst with counts",
        "post-GPU aggregate",
    ),
    (
        "queue_service",
        "preservation_events[].server_timing",
        "ns to ms; never GPU compute",
        "post-GPU aggregate",
    ),
    (
        "makespan_throughput",
        "normalized_metrics + exact output/completion clocks",
        "first eligibility through drain; tokens/s and requests/s",
        "post-GPU aggregate",
    ),
    (
        "failures_timeouts",
        "requests success/terminal/http/error + manifest",
        "expected/observed/completed/failed/timeout/missing",
        "post-GPU aggregate",
    ),
    (
        "preemptions_collateral",
        "normalized preemptions + request preemption_count",
        "cell total and request distribution",
        "post-GPU aggregate",
    ),
    (
        "causal_nomination",
        "semantic-cross-session-prefetch-nomination-v1",
        "recompute fresh causal mean/rank/exact chosen prefix",
        "CPU validation + post-GPU",
    ),
    (
        "prefetch_stages",
        "wait_hint_events + prefetch_attribution timestamps",
        "publish/submit/ready/demand/install/fence/forward/consume",
        "post-GPU aggregate",
    ),
    (
        "physical_transfers",
        "physical_transfer_accounting",
        "reconciled cell jobs/bytes/device time; no hierarchical summing",
        "post-GPU aggregate",
    ),
    (
        "prefetch_waste",
        "speculative completed/consumed/discarded/uncompleted/unconsumed",
        "separate terminal waste from censored outstanding work",
        "post-GPU aggregate",
    ),
    (
        "policy_overhead",
        "nomination_cpu_ns and decision/scoring explicit durations",
        "observed subtotal/count; not process CPU",
        "post-GPU aggregate",
    ),
    (
        "displacement",
        "staging protection/block pool/key lifecycle events",
        "exact displaced physical identities and consequence",
        "post-GPU aggregate",
    ),
    (
        "occupancy_byte_time",
        "physical_occupancy_evidence snapshots",
        "deduplicated left-held integral and observed interval",
        "post-GPU aggregate",
    ),
    (
        "shared_physical_blocks",
        "snapshot pool/rank/block identities",
        "count each physical block once per timestamp",
        "post-GPU aggregate",
    ),
    (
        "staging_memory",
        "speculative reserved/inflight/ready snapshot fields",
        "peaks and block-seconds",
        "post-GPU aggregate",
    ),
    (
        "useful_sessions_contexts",
        "exact session/context consumed within latency constraint",
        "unique identities at matched budget/workset",
        "post-GPU comparison",
    ),
    (
        "capacity_frontier",
        "memory budget/workset + useful identities",
        "frontier across controlled budget/workset cells",
        "post-GPU comparison",
    ),
    (
        "ttl_survival",
        "exact history clocks/loaded tokens/intervening traffic",
        "survival/timely reuse/censor counts by idle horizon",
        "post-GPU comparison",
    ),
    (
        "gpu_resources",
        "gpu_model_runner_work_accounting + gpu_prefill_work_accounting + physical_transfer_accounting",
        "authoritative cell-total GPU model-runner work only at exact sequence/batch coverage; optional prefill total at complete prompt-token coverage; transfer overlap retained separately",
        "post-GPU aggregate",
    ),
    (
        "cpu_memory_resources",
        "process CPU/RSS/index memory",
        "observed resource-time per request",
        "GPU run instrumentation",
    ),
    (
        "optional_cost_per_request",
        "explicit resource prices + complete resource totals",
        "sum resource quantity*price / all requests",
        "post-GPU optional",
    ),
    (
        "load_concurrency_burst",
        "planned_release/send/completion clocks",
        "offered/realized rate, overlap peak, interarrival tails",
        "CPU construction + post-GPU",
    ),
    (
        "matched_provenance",
        "manifest/profile/order/config/runtime/source hashes",
        "exact equality gates before pairing",
        "post-GPU comparison",
    ),
    (
        "paired_uncertainty",
        "trace-level matched seed/repetition deltas",
        "mean/SD/min/max/95% normal interval; requests not independent",
        "post-GPU comparison",
    ),
)


def _coverage(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    has = lambda predicate: any(predicate(cell) for cell in cells)
    cost_basis_supplied = has(
        lambda c: (
            c["resources"]["cost_per_completed_request"]["status"]
            == "unavailable-incomplete"
        )
    )
    observed = {
        "executed_prompt_tokens": has(
            lambda c: any(
                r["actual_prompt_tokens_locally_executed"] is not None
                for r in c["requests"]
            )
        ),
        "gpu_only_reuse": has(
            lambda c: any(
                r["gpu_only_reused_tokens"] is not None for r in c["requests"]
            )
        ),
        "dram_restore": has(
            lambda c: any(r["dram_restored_tokens"] is not None for r in c["requests"])
        ),
        "exact_history_cohorts": bool(cells),
        "request_latency": has(
            lambda c: c["cohorts"]["whole_trace"]["ttft_ms"]["observed_count"] > 0
        ),
        "queue_service": has(
            lambda c: c["cohorts"]["whole_trace"]["queue_ms"]["observed_count"] > 0
        ),
        "makespan_throughput": has(
            lambda c: c["cell_serving"]["episode_makespan_ms"] is not None
        ),
        "failures_timeouts": bool(cells),
        "preemptions_collateral": has(
            lambda c: c["cell_serving"]["preemptions"] is not None
        ),
        "causal_nomination": has(
            lambda c: c["resources"]["causal_nomination_observations"] > 0
        ),
        "prefetch_stages": has(lambda c: bool(c["prefetch_lifecycle"])),
        "physical_transfers": has(
            lambda c: c["physical_transfer_accounting"] is not None
        ),
        "prefetch_waste": has(
            lambda c: any(
                r.get("consumed_bytes") is not None for r in c["prefetch_lifecycle"]
            )
        ),
        "policy_overhead": has(
            lambda c: (
                c["resources"]["nomination_cpu_ms"]["observed_events"] > 0
                or c["resources"]["policy_decision_elapsed_ms"]["observed_events"] > 0
            )
        ),
        "displacement": has(lambda c: bool(c["displacements"])),
        "occupancy_byte_time": has(
            lambda c: c["occupancy"]["dram_bytes"]["time_integral"] is not None
        ),
        "shared_physical_blocks": has(
            lambda c: c["occupancy"]["shared_gpu_physical_blocks"]["sample_count"] > 0
        ),
        "staging_memory": has(
            lambda c: c["occupancy"]["staging_reserved_blocks"]["sample_count"] > 0
        ),
        "useful_sessions_contexts": False,
        "capacity_frontier": False,
        "ttl_survival": has(lambda c: bool(c["ttl_observations"])),
        "gpu_resources": has(
            lambda c: (
                c["resources"]["gpu_model_runner_work"].get("available") is True
                or _maybe_number(
                    c["resources"]["gpu_model_runner_work"].get(
                        "observed_device_time_s"
                    )
                )
                is not None
                or c["resources"]["gpu_prefill_ms"]["complete"] is True
                or (
                    _maybe_number(c["resources"]["gpu_prefill_ms"]["observed_requests"])
                    or 0
                )
                > 0
            )
        ),
        "cpu_memory_resources": has(
            lambda c: (
                c["resources"]["process_cpu_seconds"]["value"] is not None
                or c["resources"]["process_rss_byte_seconds"]["value"] is not None
            )
        ),
        "optional_cost_per_request": has(
            lambda c: c["resources"]["cost_per_completed_request"]["value"] is not None
        ),
        "load_concurrency_burst": has(
            lambda c: c["load"]["maximum_observed_concurrency"] is not None
        ),
        "matched_provenance": len(cells) >= 2,
        "paired_uncertainty": False,
    }
    gpu_only = {
        "causal_nomination",
        "policy_overhead",
        "displacement",
        "useful_sessions_contexts",
        "capacity_frontier",
        "ttl_survival",
        "gpu_resources",
        "cpu_memory_resources",
        "optional_cost_per_request",
        "matched_provenance",
        "paired_uncertainty",
    }
    units = {
        "executed_prompt_tokens": "tokens/request",
        "gpu_only_reuse": "tokens",
        "dram_restore": "tokens",
        "request_latency": "milliseconds",
        "queue_service": "milliseconds",
        "makespan_throughput": "milliseconds; tokens/s; requests/s",
        "prefetch_stages": "host monotonic ns",
        "physical_transfers": "jobs; bytes; device seconds",
        "prefetch_waste": "jobs; bytes",
        "policy_overhead": "elapsed milliseconds",
        "occupancy_byte_time": "bytes; byte-seconds; block-seconds",
        "staging_memory": "blocks; bytes; time integral",
        "ttl_survival": "milliseconds; fractions",
        "load_concurrency_burst": "requests/s; count; milliseconds",
    }
    return [
        {
            "metric": metric,
            "raw_fields": raw,
            "calculation": calculation,
            "phase": phase,
            "unit": units.get(metric, "count/fraction or structured evidence"),
            "comparator": (
                "M vs A/L; M vs R for semantic nomination"
                if metric not in {"optional_cost_per_request"}
                else "per-arm then M vs A/L/R"
            ),
            "status": "implemented"
            if observed[metric]
            else (
                "not-implemented"
                if metric == "optional_cost_per_request" and not cost_basis_supplied
                else ("GPU-unmeasured" if metric in gpu_only else "missing")
            ),
            "unavailable_reason": None
            if observed[metric]
            else "required raw observations absent from supplied cells",
            "missing_data_behavior": "null with observed/expected counts; never infer zero",
            "cpu_fixture_evidence": ("tests/test_msr_combined_prefetch_attribution.py"),
            "retained_real_schema_evidence": (
                "observed" if observed[metric] else "not present in retained real cell"
            ),
        }
        for metric, raw, calculation, phase in _COVERAGE_SPEC
    ]


def _paired_uncertainty(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for cell in cells:
        arm_id = str(cell.get("arm_id") or "")
        role = arm_id.split("-", 1)[0]
        key = (
            cell.get("campaign_id"),
            cell.get("profile_id"),
            cell.get("replicate_id"),
            cell.get("seed"),
        )
        grouped.setdefault(key, {})[role] = cell
    pairs: list[dict[str, Any]] = []
    for key, arms in grouped.items():
        msr = arms.get("M")
        if msr is None:
            continue
        for baseline in ("A", "L", "P", "R"):
            other = arms.get(baseline)
            if other is None:
                continue
            matched = all(
                msr["provenance"].get(field) is not None
                and msr["provenance"].get(field) == other["provenance"].get(field)
                for field in (
                    "request_manifest_sha256",
                    "baseline_profile_hash",
                    "order_schedule_digest",
                    "runtime_revision",
                    "runtime_tree",
                    "extension_revision",
                    "extension_tree",
                )
            )
            left_config = msr["configuration"]["runtime_extra_config"]
            right_config = other["configuration"]["runtime_extra_config"]
            expected_policy = {"A": "arc", "L": "lru", "P": "arc", "R": "msr"}
            try:
                left_policy = _canonical_policy_identity(left_config)
                right_policy = _canonical_policy_identity(right_config)
            except AttributionError:
                left_policy = right_policy = "unknown"
            matched = bool(
                matched
                and left_config.get("store_threshold") == DRAM_THRESHOLD
                and right_config.get("store_threshold") == DRAM_THRESHOLD
                and left_policy == "msr"
                and left_config.get("prefetch_selector") == "semantic"
                and right_policy == expected_policy[baseline]
                and (
                    baseline not in {"R", "P"}
                    or right_config.get("prefetch_selector") == "request_order"
                )
            )
            allowed_config_differences = {
                "eviction_policy",
                "method",
                "prefetch_selector",
                "prefetch_trigger",
                "prefetch_budget_fraction",
                "prefetch_score_max_age_ms",
                "soft_retention",
                "instrumentation_only",
                "shared_prefetch_scorer",
            }
            if baseline == "R":
                allowed_config_differences = {"prefetch_selector"}
            elif baseline in {"A", "L"} and _demand_prefetch_is_disabled(right_config):
                allowed_config_differences.update(_COALESCING_CONFIG_FIELDS)
            for field in set(left_config) | set(right_config):
                if field not in allowed_config_differences and left_config.get(
                    field
                ) != right_config.get(field):
                    matched = False
            left_outputs = {
                row["request_id"]: row["output_token_digest"] for row in msr["requests"]
            }
            right_outputs = {
                row["request_id"]: row["output_token_digest"]
                for row in other["requests"]
            }
            matched = bool(
                matched
                and left_outputs
                and left_outputs == right_outputs
                and all(left_outputs.values())
                and msr["request_counts"]["completed"]
                == msr["request_counts"]["expected"]
                and other["request_counts"]["completed"]
                == other["request_counts"]["expected"]
            )
            left = msr["cohorts"]["whole_trace"]
            right = other["cohorts"]["whole_trace"]
            metric_deltas: dict[str, Any] = {}
            for metric in ("ttft_ms", "itl_ms", "e2e_ms", "queue_ms", "service_ms"):
                metric_deltas[f"delta_{metric}_p50"] = (
                    left[metric]["p50"] - right[metric]["p50"]
                    if matched
                    and left[metric]["p50"] is not None
                    and right[metric]["p50"] is not None
                    else None
                )
            for metric in (
                "actual_prompt_tokens_locally_executed",
                "gpu_only_reused_tokens",
                "dram_restored_tokens",
            ):
                metric_deltas[f"delta_{metric}"] = (
                    left[metric]["value"] - right[metric]["value"]
                    if matched
                    and left[metric]["value"] is not None
                    and right[metric]["value"] is not None
                    else None
                )
            left_gpu_prefill = msr["resources"]["gpu_prefill_ms"]["value"]
            right_gpu_prefill = other["resources"]["gpu_prefill_ms"]["value"]
            metric_deltas["delta_gpu_prefill_ms"] = (
                left_gpu_prefill - right_gpu_prefill
                if matched
                and left_gpu_prefill is not None
                and right_gpu_prefill is not None
                else None
            )
            left_model_runner = msr["resources"]["gpu_model_runner_work"].get(
                "device_time_s"
            )
            right_model_runner = other["resources"]["gpu_model_runner_work"].get(
                "device_time_s"
            )
            metric_deltas["delta_gpu_model_runner_device_time_s"] = (
                left_model_runner - right_model_runner
                if matched
                and msr["resources"]["gpu_model_runner_work"].get("available") is True
                and other["resources"]["gpu_model_runner_work"].get("available") is True
                and left_model_runner is not None
                and right_model_runner is not None
                else None
            )
            left_cost = msr["resources"]["cost_per_completed_request"]["value"]
            right_cost = other["resources"]["cost_per_completed_request"]["value"]
            metric_deltas["delta_cost_per_completed_request"] = (
                left_cost - right_cost
                if matched and left_cost is not None and right_cost is not None
                else None
            )
            baseline_actual = right["actual_prompt_tokens_locally_executed"]["value"]
            msr_actual = left["actual_prompt_tokens_locally_executed"]["value"]
            baseline_ttft = right["ttft_ms"]["p50"]
            msr_ttft = left["ttft_ms"]["p50"]
            metric_deltas["executed_prompt_token_reduction_percent"] = (
                100.0 * (baseline_actual - msr_actual) / baseline_actual
                if matched
                and baseline_actual is not None
                and msr_actual is not None
                and baseline_actual > 0
                else None
            )
            metric_deltas["median_ttft_regression_percent"] = (
                100.0 * (msr_ttft - baseline_ttft) / baseline_ttft
                if matched
                and baseline_ttft is not None
                and msr_ttft is not None
                and baseline_ttft > 0
                else None
            )
            metric_deltas["ttft_allowance_met"] = (
                metric_deltas["median_ttft_regression_percent"] <= 3.0
                if metric_deltas["median_ttft_regression_percent"] is not None
                else None
            )
            pairs.append(
                {
                    "pair_key": key,
                    "comparison": f"M-{baseline}",
                    "regime_identity": msr.get("regime_identity"),
                    "input_cohort_identity": msr.get("input_cohort_identity"),
                    "seed": msr.get("seed"),
                    "replicate_id": msr.get("replicate_id"),
                    "pairing_weight": (
                        msr.get("pairing_weight")
                        if msr.get("pairing_weight") == other.get("pairing_weight")
                        else None
                    ),
                    "msr_combined_activation_met": msr["activation"][
                        "combined_activation_met"
                    ],
                    "matched": matched,
                    **metric_deltas,
                }
            )
    results: list[dict[str, Any]] = []

    def pair_weight(value: Any) -> float | None:
        if value == "one matched trace per replicate; never request-weighted":
            return 1.0
        number = _maybe_number(value)
        return number if number is not None and number > 0 else None

    summary_keys = sorted(
        {
            (
                row["comparison"],
                row.get("regime_identity"),
                row.get("input_cohort_identity"),
            )
            for row in pairs
        },
        key=str,
    )
    for comparison, regime_identity, input_cohort_identity in summary_keys:
        rows = [
            row
            for row in pairs
            if (
                row["comparison"],
                row.get("regime_identity"),
                row.get("input_cohort_identity"),
            )
            == (comparison, regime_identity, input_cohort_identity)
        ]
        summary: dict[str, Any] = {
            "comparison": comparison,
            "regime_identity": regime_identity,
            "input_cohort_identity": input_cohort_identity,
            "group_by": [
                "comparison",
                "regime_identity",
                "input_cohort_identity",
            ],
            "weighting": "trace-level pairing_weight when complete; unweighted also reported",
            "matched_pairs": sum(row["matched"] for row in rows),
            "missing_pairs": sum(not row["matched"] for row in rows),
            "pairs": rows,
        }
        metric_names = sorted(
            {key for row in rows for key in row if key.startswith("delta_")}
        )
        metric_names.extend(
            [
                "executed_prompt_token_reduction_percent",
                "median_ttft_regression_percent",
            ]
        )
        for metric in metric_names:
            values = [
                row[metric]
                for row in rows
                if row["matched"] and row[metric] is not None
            ]
            mean = statistics.fmean(values) if values else None
            sd = statistics.stdev(values) if len(values) >= 2 else None
            weighted = [
                (row[metric], pair_weight(row.get("pairing_weight")))
                for row in rows
                if row["matched"] and row[metric] is not None
            ]
            weights_complete = bool(
                weighted
                and len(weighted) == len(values)
                and all(weight is not None and weight > 0 for _, weight in weighted)
            )
            summary[metric] = {
                "mean": mean,
                "sd": sd,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "n_independent_trace_pairs": len(values),
                "seed_count": len(
                    {
                        row["seed"]
                        for row in rows
                        if row["matched"] and row[metric] is not None
                    }
                ),
                "weighted_mean": (
                    sum(value * weight for value, weight in weighted)
                    / sum(weight for _, weight in weighted)
                    if weights_complete
                    else None
                ),
                "weighting_complete": weights_complete,
                "normal_95_percent_interval": (
                    [
                        mean - 1.96 * sd / math.sqrt(len(values)),
                        mean + 1.96 * sd / math.sqrt(len(values)),
                    ]
                    if len(values) >= 2 and mean is not None and sd is not None
                    else None
                ),
            }
        results.append(summary)
    return results


def _ttl_curves(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for cell in cells:
        for row in cell["ttl_observations"]:
            idle = row.get("idle_ms")
            if idle is None:
                continue
            grouped.setdefault(
                (
                    cell.get("campaign_id"),
                    cell.get("regime_identity"),
                    cell.get("input_cohort_identity"),
                    cell.get("arm_id"),
                    round(float(idle), 3),
                ),
                [],
            ).append(row)
    curves: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=str):
        campaign_id, regime_identity, input_cohort_identity, arm_id, idle_ms = key
        observed = [row for row in rows if not row.get("censored")]
        survived = [row for row in observed if row.get("dram_survived") is True]
        known_survival = [
            row for row in observed if isinstance(row.get("dram_survived"), bool)
        ]
        timely = [row for row in observed if row.get("timely_reuse") is True]
        known_timely = [
            row for row in observed if isinstance(row.get("timely_reuse"), bool)
        ]
        curves.append(
            {
                "arm_id": arm_id,
                "campaign_id": campaign_id,
                "regime_identity": regime_identity,
                "input_cohort_identity": input_cohort_identity,
                "idle_horizon_ms": idle_ms,
                "observations": len(rows),
                "survival_observed": len(known_survival),
                "survived": len(survived),
                "survival_fraction": (
                    len(survived) / len(known_survival) if known_survival else None
                ),
                "timely_reuse_observed": len(known_timely),
                "timely_reused": len(timely),
                "timely_reuse_fraction": (
                    len(timely) / len(known_timely) if known_timely else None
                ),
                "censored": sum(row.get("censored") is True for row in rows),
                "note": "no-return observations are censored; sampled evidence does not prove uninterrupted survival",
            }
        )
    return curves


def _exact_group_gate(
    arms: Mapping[str, Mapping[str, Any]], compared_roles: Sequence[str]
) -> tuple[bool, str | None]:
    required = {"M", *compared_roles}
    if not required.issubset(arms):
        return (
            False,
            f"missing requested comparison arm(s): {sorted(required - set(arms))}",
        )
    reference = arms["M"]
    reference_outputs = {
        row["request_id"]: row["output_token_digest"] for row in reference["requests"]
    }
    if not reference_outputs or not all(reference_outputs.values()):
        return False, "MSR output identity is incomplete"
    for role in sorted(required):
        cell = arms[role]
        if cell.get("valid") is not True:
            return False, f"{role} cell invalid"
        outputs = {
            row["request_id"]: row["output_token_digest"] for row in cell["requests"]
        }
        if outputs != reference_outputs:
            return False, f"{role} request/output identity differs"
        if cell["request_counts"]["completed"] != cell["request_counts"]["expected"]:
            return False, f"{role} completion denominator incomplete"
        for field in (
            "request_manifest_sha256",
            "baseline_profile_hash",
            "order_schedule_digest",
            "runtime_revision",
            "runtime_tree",
            "extension_revision",
            "extension_tree",
        ):
            if reference["provenance"].get(field) is None or cell["provenance"].get(
                field
            ) != reference["provenance"].get(field):
                return False, f"{role} provenance differs: {field}"
        for field in ("configured_dram_bytes", "configured_gpu_blocks"):
            if cell["configuration"].get(field) != reference["configuration"].get(
                field
            ):
                return False, f"{role} {field} differs"
        config = cell["configuration"]["runtime_extra_config"]
        expected_policy = {
            "A": "arc",
            "L": "lru",
            "P": "arc",
            "R": "msr",
            "M": "msr",
        }[role]
        try:
            policy = _canonical_policy_identity(config)
        except AttributionError as error:
            return False, f"{role} policy identity invalid: {error}"
        if policy != expected_policy or config.get("store_threshold") != DRAM_THRESHOLD:
            return False, f"{role} policy/threshold identity differs"
        if role == "M" and config.get("prefetch_selector") != "semantic":
            return False, "M semantic selector inactive"
        if role in {"P", "R"} and config.get("prefetch_selector") != "request_order":
            return False, f"{role} FIFO selector inactive"
        reference_config = reference["configuration"]["runtime_extra_config"]
        allowed = (
            {"prefetch_selector"}
            if role == "R"
            else {
                "eviction_policy",
                "method",
                "prefetch_selector",
                "prefetch_trigger",
                "prefetch_budget_fraction",
                "prefetch_score_max_age_ms",
                "soft_retention",
                "instrumentation_only",
                "shared_prefetch_scorer",
            }
        )
        if role in {"A", "L"} and _demand_prefetch_is_disabled(config):
            allowed.update(_COALESCING_CONFIG_FIELDS)
        for config_field in set(config) | set(reference_config):
            if config_field not in allowed and config.get(
                config_field
            ) != reference_config.get(config_field):
                return False, f"{role} generic runtime config differs: {config_field}"
    return True, None


def _capacity_rows(
    cells: Sequence[Mapping[str, Any]], *, fixed_service_target_ms: float | None
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for cell in cells:
        role = str(cell.get("arm_id") or "").split("-", 1)[0]
        key = (
            cell.get("campaign_id"),
            cell.get("profile_id"),
            cell.get("replicate_id"),
            cell.get("seed"),
        )
        grouped.setdefault(key, {})[role] = cell
    rows: list[dict[str, Any]] = []
    for key, arms in grouped.items():
        pair_gates = {
            role: _exact_group_gate(arms, (role,))
            for role in ("A", "L", "P", "R")
            if role in arms
        }
        primary_pair_available = any(
            pair_gates.get(role, (False, None))[0] for role in ("A", "L")
        )
        group_rows: dict[str, dict[str, Any]] = {}
        for role in ("A", "L", "P", "R", "M"):
            cell = arms.get(role)
            if cell is None:
                continue
            comparable, unavailable_reason = (
                (
                    primary_pair_available,
                    None
                    if primary_pair_available
                    else "no valid primary M-A or M-L comparison",
                )
                if role == "M"
                else pair_gates.get(role, (False, f"{role} comparison arm unavailable"))
            )
            median_ttft = cell["cohorts"]["whole_trace"]["ttft_ms"]["p50"]
            eligible: set[tuple[Any, ...]] = set()
            useful: set[tuple[Any, ...]] = set()
            service_observations = 0
            service_passes = 0
            requests_by_id = {
                request["request_id"]: request for request in cell["requests"]
            }
            for request in cell["requests"]:
                if request["cohort"] not in {
                    "first_exact_return",
                    "subsequent_exact_return",
                    "return",
                    "first_return",
                    "repeated_return",
                }:
                    continue
                boundary = request.get("prefix_boundary_tokens")
                identity = (
                    request.get("session_id"),
                    request.get("generation"),
                    request.get("key_chain_digest"),
                    boundary,
                )
                if all(value not in (None, "") for value in identity):
                    eligible.add(identity)
                release_latency = request.get("release_to_first_token_ms")
                request_service_pass = bool(
                    request.get("success")
                    and request.get("terminal_complete")
                    and release_latency is not None
                    and fixed_service_target_ms is not None
                    and release_latency <= fixed_service_target_ms
                )
                service_observations += 1
                service_passes += request_service_pass
                reused = request.get("gpu_only_reused_tokens")
                restored = request.get("dram_restored_tokens")
                if (
                    comparable
                    and request_service_pass
                    and reused is not None
                    and restored is not None
                    and boundary is not None
                    and reused + restored >= boundary
                    and identity in eligible
                ):
                    useful.add(identity)
            for ttl in cell["ttl_observations"]:
                request = requests_by_id.get(ttl.get("return_request_id"))
                if request is None or ttl.get("dram_survived") is None:
                    continue
                reused = request.get("gpu_only_reused_tokens")
                restored = request.get("dram_restored_tokens")
                boundary = request.get("prefix_boundary_tokens")
                full_exact = bool(
                    reused is not None
                    and restored is not None
                    and boundary is not None
                    and reused + restored >= boundary
                )
                ttl["timely_reuse"] = (
                    bool(
                        ttl["dram_survived"]
                        and full_exact
                        and request.get("success")
                        and request.get("terminal_complete")
                        and request.get("release_to_first_token_ms") is not None
                        and request["release_to_first_token_ms"]
                        <= fixed_service_target_ms
                    )
                    if comparable and fixed_service_target_ms is not None
                    else None
                )
                ttl["timely_reuse_unavailable_reason"] = (
                    None
                    if comparable and fixed_service_target_ms is not None
                    else unavailable_reason or "fixed service target unavailable"
                )
            result = {
                "pair_key": key,
                "role": role,
                "arm_id": cell.get("arm_id"),
                "campaign_id": cell.get("campaign_id"),
                "regime_identity": cell.get("regime_identity"),
                "input_cohort_identity": cell.get("input_cohort_identity"),
                "seed": cell.get("seed"),
                "replicate_id": cell.get("replicate_id"),
                **cell["configuration"],
                "fixed_service_target_ms": fixed_service_target_ms,
                "median_ttft_ms": median_ttft,
                "service_target_observations": service_observations,
                "service_target_passes": service_passes,
                "service_target_failures": service_observations - service_passes,
                "service_target_met": (
                    service_passes == service_observations
                    if comparable
                    and fixed_service_target_ms is not None
                    and service_observations > 0
                    else None
                ),
                "comparable_group": comparable,
                "pair_comparability": {
                    candidate: {
                        "available": gate[0],
                        "unavailable_reason": gate[1],
                    }
                    for candidate, gate in pair_gates.items()
                }
                if role == "M"
                else None,
                "unavailable_reason": unavailable_reason,
                "eligible_exact_sessions_contexts": len(eligible),
                "timely_useful_exact_sessions_contexts": (
                    len(useful)
                    if comparable and fixed_service_target_ms is not None
                    else None
                ),
                "sampled_useful_yield": (
                    len(useful) / len(eligible)
                    if comparable and fixed_service_target_ms is not None and eligible
                    else None
                ),
                "frontier_claim_supported": False,
                "note": "sampled yield at one declared budget/workset; not an automatic capacity frontier",
            }
            rows.append(result)
            group_rows[role] = result
        msr_row = group_rows.get("M")
        if msr_row is not None:
            for baseline in ("A", "L", "P", "R"):
                baseline_row = group_rows.get(baseline)
                if baseline_row is None:
                    continue
                pair_available = pair_gates.get(baseline, (False, None))[0]
                left = msr_row["timely_useful_exact_sessions_contexts"]
                right = baseline_row["timely_useful_exact_sessions_contexts"]
                msr_row[f"delta_useful_contexts_vs_{baseline}"] = (
                    left - right
                    if pair_available and left is not None and right is not None
                    else None
                )
                left_yield = msr_row["sampled_useful_yield"]
                right_yield = baseline_row["sampled_useful_yield"]
                msr_row[f"delta_sampled_yield_vs_{baseline}"] = (
                    left_yield - right_yield
                    if pair_available
                    and left_yield is not None
                    and right_yield is not None
                    else None
                )
    return rows


def _memory_capacity_contrasts(
    capacity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in capacity_rows:
        key = (
            row.get("role"),
            row.get("input_cohort_identity"),
            row.get("seed"),
            row.get("replicate_id"),
            row.get("workset_exact_contexts"),
        )
        grouped.setdefault(key, []).append(row)
    contrasts: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        by_budget = {
            row.get("configured_dram_bytes"): row
            for row in rows
            if row.get("configured_dram_bytes") is not None
        }
        if len(by_budget) < 2:
            continue
        low_budget, high_budget = min(by_budget), max(by_budget)
        low, high = by_budget[low_budget], by_budget[high_budget]
        low_yield, high_yield = (
            low.get("sampled_useful_yield"),
            high.get("sampled_useful_yield"),
        )
        contrasts.append(
            {
                "contrast_key": key,
                "low_dram_bytes": low_budget,
                "high_dram_bytes": high_budget,
                "low_sampled_useful_yield": low_yield,
                "high_sampled_useful_yield": high_yield,
                "delta_sampled_yield_high_minus_low": (
                    high_yield - low_yield
                    if low_yield is not None and high_yield is not None
                    else None
                ),
                "comparable": bool(
                    low.get("comparable_group") and high.get("comparable_group")
                ),
                "frontier_claim_supported": False,
                "note": "controlled sampled memory contrast only; not a capacity frontier",
            }
        )
    return contrasts


def aggregate_result_paths(
    result_paths: Sequence[Path],
    request_manifest: Mapping[str, Any],
    *,
    fixed_service_target_ms: float | None = None,
    resource_price_basis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cells = [
        extract_cell_metrics(_load(path), request_manifest) for path in result_paths
    ]
    return _assemble_expanded(
        cells,
        missing_results=[],
        fixed_service_target_ms=fixed_service_target_ms,
        resource_price_basis=resource_price_basis,
    )


def aggregate_bundle_results(
    bundle_path: Path,
    results_root: Path | Sequence[Path],
    *,
    fixed_service_target_ms: float | None = None,
    resource_price_basis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = _load(bundle_path)
    if bundle.get("schema") != "msr-combined-prefetch-expanded-campaign-v1":
        _fail("unsupported expanded workload bundle schema")
    claim = bundle.get("provider_claim")
    claim = claim if isinstance(claim, Mapping) else {}
    target_contract = claim.get("fixed_service_target")
    target_contract = target_contract if isinstance(target_contract, Mapping) else {}
    cells: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    roots = [results_root] if isinstance(results_root, Path) else list(results_root)
    for campaign in _records(bundle.get("campaigns"), "bundle campaigns"):
        campaign_id = _string(campaign.get("campaign_id"), "campaign_id")
        generated = _mapping(campaign.get("generated"), "campaign generated")
        if not campaign.get("input_cohort_identity") or not campaign.get(
            "regime_identity"
        ):
            _fail(f"{campaign_id}: input/regime identity is unavailable")
        if campaign.get("pairing_weight") in (None, ""):
            _fail(f"{campaign_id}: pairing weight is unavailable")
        dimensions = _mapping(campaign.get("dimensions"), "campaign dimensions")
        if dimensions.get("fixed_service_target") != target_contract:
            _fail(f"{campaign_id}: fixed service target differs from provider claim")
        portable = _string(
            generated.get("request_portable_path"), "request_portable_path"
        )
        request_path = bundle_path.parent / portable
        expected_request_sha = generated.get("request_sha256")
        if (
            not isinstance(expected_request_sha, str)
            or hashlib.sha256(request_path.read_bytes()).hexdigest()
            != expected_request_sha
        ):
            _fail(f"{campaign_id}: request manifest identity differs")
        request_manifest = _load(request_path)
        for selected in _records(campaign.get("selected_cells"), "selected_cells"):
            cell_id = _string(selected.get("cell_id"), "selected cell_id")
            candidates = [
                root / campaign_id / cell_id / "result.json" for root in roots
            ]
            existing = [path for path in candidates if path.is_file()]
            if len(existing) > 1:
                _fail(f"multiple result roots contain {campaign_id}/{cell_id}")
            if not existing:
                missing.append(
                    {
                        "campaign_id": campaign_id,
                        "cell_id": cell_id,
                        "arm_id": selected.get("arm_id"),
                        "replicate_id": selected.get("replicate_id"),
                        "seed": selected.get("seed"),
                        "expected_result_paths": [str(path) for path in candidates],
                    }
                )
                continue
            result_path = existing[0]
            raw_result = _load(result_path)
            for field in ("cell_id", "arm_id", "profile_id", "replicate_id", "seed"):
                expected = selected.get(field)
                if expected is not None and raw_result.get(field) != expected:
                    _fail(
                        f"declared cell identity mismatch for {campaign_id}/{cell_id}: {field}"
                    )
            cell = extract_cell_metrics(raw_result, request_manifest)
            cell["campaign_id"] = campaign_id
            cell["campaign_dimensions"] = campaign.get("dimensions")
            cell["manifest_input_cohort_identity"] = cell.get("input_cohort_identity")
            cell["input_cohort_identity"] = campaign.get("input_cohort_identity")
            cell["regime_identity"] = campaign.get("regime_identity")
            cell["pairing_weight"] = campaign.get("pairing_weight")
            cell["result_path"] = str(result_path)
            cells.append(cell)
    declared_target = target_contract.get(
        "maximum_ms", claim.get("fixed_service_target_ms")
    )
    target = fixed_service_target_ms
    if target is None and _maybe_number(declared_target) is not None:
        target = float(declared_target)
    report = _assemble_expanded(
        cells,
        missing_results=missing,
        fixed_service_target_ms=target,
        resource_price_basis=resource_price_basis,
        independent_input_proof=bundle.get("independent_input_cohort_proof"),
    )
    report["bundle"] = {
        "path": str(bundle_path),
        "result_layout": bundle.get("result_layout"),
        "metric_phase_table": bundle.get("metric_phase_table"),
        "regime_aggregation": bundle.get("regime_aggregation"),
        "future_aggregation_command": bundle.get("future_aggregation_command"),
    }
    return report


_PRICE_COMPONENTS = frozenset(
    {
        "gpu_prefill_ms",
        "gpu_model_runner_device_s",
        "speculative_transfer_device_ms",
        "dram_byte_seconds",
        "gpu_block_seconds",
        "policy_decision_elapsed_ms",
        "process_cpu_seconds",
        "process_rss_byte_seconds",
        "energy_joules",
    }
)


def _resource_quantities(cell: Mapping[str, Any]) -> dict[str, float | None]:
    gpu = cell["resources"]["gpu_prefill_ms"]
    model_runner = cell["resources"]["gpu_model_runner_work"]
    physical = cell.get("physical_transfer_accounting")
    physical = physical if isinstance(physical, Mapping) else {}
    policy = cell["resources"]["policy_decision_elapsed_ms"]
    return {
        "gpu_prefill_ms": (gpu["value"] if gpu.get("complete") is True else None),
        "gpu_model_runner_device_s": (
            model_runner.get("device_time_s")
            if model_runner.get("available") is True
            else None
        ),
        "speculative_transfer_device_ms": (
            float(physical["speculative_device_time_s"]) * 1000
            if _maybe_number(physical.get("speculative_device_time_s")) is not None
            else None
        ),
        "dram_byte_seconds": cell["occupancy"]["dram_bytes"]["time_integral"],
        "gpu_block_seconds": cell["occupancy"]["gpu_blocks"]["time_integral"],
        "policy_decision_elapsed_ms": policy["observed_subtotal"],
        "process_cpu_seconds": cell["resources"]["process_cpu_seconds"]["value"],
        "process_rss_byte_seconds": cell["resources"]["process_rss_byte_seconds"][
            "value"
        ],
        "energy_joules": cell["resources"]["energy_joules"]["value"],
    }


def _apply_resource_costs(
    cells: Sequence[Mapping[str, Any]], basis: Mapping[str, Any] | None
) -> None:
    if basis is None:
        return
    if basis.get("schema") != "msr-resource-price-basis-v1":
        _fail("unsupported resource price basis schema")
    currency = _string(basis.get("currency"), "resource price currency")
    prices = _mapping(basis.get("unit_prices"), "resource unit_prices")
    required = basis.get("required_components")
    if (
        not isinstance(required, list)
        or not required
        or len(required) != len(set(required))
        or any(component not in _PRICE_COMPONENTS for component in required)
    ):
        _fail("resource price basis has invalid required_components")
    unit_prices: dict[str, float] = {}
    for component in required:
        price = _finite(prices.get(component), f"unit price {component}")
        if price < 0:
            _fail("resource unit prices must be non-negative")
        unit_prices[component] = price
    for cell in cells:
        quantities = _resource_quantities(cell)
        missing = [
            component for component in required if quantities.get(component) is None
        ]
        completed = cell["request_counts"]["completed"]
        complete_denominator = (
            completed > 0 and completed == cell["request_counts"]["expected"]
        )
        value = (
            sum(
                quantities[component] * unit_prices[component] for component in required
            )
            / completed
            if not missing and complete_denominator
            else None
        )
        cell["resources"]["cost_per_completed_request"] = {
            "value": value,
            "currency": currency,
            "required_components": list(required),
            "quantities": quantities,
            "unit_prices": unit_prices,
            "status": "available" if value is not None else "unavailable-incomplete",
            "unavailable_reason": (
                None
                if value is not None
                else {
                    "missing_quantities": missing,
                    "complete_request_denominator": complete_denominator,
                }
            ),
        }


def _assemble_expanded(
    cells: Sequence[Mapping[str, Any]],
    *,
    missing_results: Sequence[Mapping[str, Any]],
    fixed_service_target_ms: float | None,
    resource_price_basis: Mapping[str, Any] | None,
    independent_input_proof: Any = None,
) -> dict[str, Any]:
    if fixed_service_target_ms is not None and (
        not math.isfinite(fixed_service_target_ms) or fixed_service_target_ms <= 0
    ):
        _fail("fixed service target must be a positive finite number")
    _apply_resource_costs(cells, resource_price_basis)
    uncertainty = _paired_uncertainty(cells)
    capacity = _capacity_rows(cells, fixed_service_target_ms=fixed_service_target_ms)
    memory_contrasts = _memory_capacity_contrasts(capacity)
    ttl_curves = _ttl_curves(cells)
    coverage = _coverage(cells)
    for row in coverage:
        if row["metric"] == "paired_uncertainty" and any(
            item["matched_pairs"] >= 2 for item in uncertainty
        ):
            row["status"] = "implemented"
            row["unavailable_reason"] = None
        if row["metric"] == "matched_provenance" and not any(
            item["matched_pairs"] for item in uncertainty
        ):
            row["status"] = "missing"
            row["unavailable_reason"] = "no exact matched result pair supplied"
        if row["metric"] == "ttl_survival" and not any(
            item["timely_reuse_observed"] for item in ttl_curves
        ):
            row["status"] = "GPU-unmeasured"
            row["unavailable_reason"] = (
                "survival rows exist but matched timely-reuse constraint is unavailable"
            )
        if row["metric"] == "useful_sessions_contexts" and any(
            item["timely_useful_exact_sessions_contexts"] is not None
            for item in capacity
        ):
            row["status"] = "implemented"
            row["unavailable_reason"] = None
        if row["metric"] == "capacity_frontier":
            row["status"] = "GPU-unmeasured"
            row["unavailable_reason"] = (
                "sampled yields/deltas are reported; no automatic frontier claim"
            )
    proof_valid = bool(
        isinstance(independent_input_proof, Mapping)
        and independent_input_proof.get("all_recorded_identity_intersections_empty")
        is True
        and independent_input_proof.get("working_sets_are_not_pooled") is True
        and independent_input_proof.get("reference_campaign_id")
        and independent_input_proof.get("independent_campaign_id")
    )
    confirmed = True
    decision_reasons: list[str] = []
    for comparison_name in ("M-A", "M-L"):
        summaries = [
            item for item in uncertainty if item["comparison"] == comparison_name
        ]
        if not summaries:
            confirmed = False
            decision_reasons.append("missing matched M-A or M-L comparison")
            continue
        qualifying = []
        positive_input_identities: set[str] = set()
        for summary in summaries:
            reduction = summary.get("executed_prompt_token_reduction_percent", {})
            matched_rows = [row for row in summary["pairs"] if row["matched"]]
            if (
                reduction.get("seed_count", 0) >= 3
                and reduction.get("mean") is not None
                and reduction["mean"] > 0
                and matched_rows
                and all(
                    row.get("ttft_allowance_met") is True
                    and row.get("msr_combined_activation_met") is True
                    for row in matched_rows
                )
            ):
                qualifying.append(summary)
            if summary.get("input_cohort_identity") and any(
                row.get("executed_prompt_token_reduction_percent") is not None
                and row["executed_prompt_token_reduction_percent"] > 0
                and row.get("ttft_allowance_met") is True
                and row.get("msr_combined_activation_met") is True
                for row in matched_rows
            ):
                positive_input_identities.add(summary["input_cohort_identity"])
        if not qualifying:
            confirmed = False
            decision_reasons.append(
                f"{comparison_name} lacks a positive three-seed result within one declared regime/input cohort"
            )
        if not proof_valid or len(positive_input_identities) < 2:
            confirmed = False
            decision_reasons.append(
                f"{comparison_name} lacks recorded disjoint-input confirmation"
            )
    comparison_status = {
        name: {
            "regime_input_summaries": sum(
                item["comparison"] == name for item in uncertainty
            ),
            "matched_trace_pairs": sum(
                item["matched_pairs"]
                for item in uncertainty
                if item["comparison"] == name
            ),
            "role": (
                "primary provider comparator"
                if name in {"M-A", "M-L"}
                else (
                    "practical ARC+FIFO comparator"
                    if name == "M-P"
                    else "semantic nomination isolation comparator"
                )
            ),
        }
        for name in ("M-A", "M-L", "M-P", "M-R")
    }
    cost_complete = bool(
        cells
        and all(
            cell["resources"]["cost_per_completed_request"]["value"] is not None
            for cell in cells
        )
    )
    return {
        "schema": "msr-combined-prefetch-expanded-aggregation-v1",
        "benefit_measured": any(item["matched_pairs"] > 0 for item in uncertainty),
        "fixed_service_target_ms": fixed_service_target_ms,
        "resource_price_basis_supplied": resource_price_basis is not None,
        "cells": cells,
        "missing_results": list(missing_results),
        "independent_input_cohort_proof": independent_input_proof,
        "paired_uncertainty": uncertainty,
        "ttl_curves": ttl_curves,
        "useful_capacity": capacity,
        "memory_capacity_contrasts": memory_contrasts,
        "coverage": coverage,
        "conclusions": {
            "provider_benefit": (
                "positive_candidate_on_supplied_regimes"
                if confirmed
                else "unavailable_or_unconfirmed"
            ),
            "provider_benefit_reasons": decision_reasons,
            "comparison_status": comparison_status,
            "average_inference_cost": (
                "available per cell under supplied explicit price basis"
                if cost_complete
                else "unavailable without complete resource quantities and explicit prices"
            ),
            "capacity_or_ttl_claim": "unavailable until controlled budget/workset/idle-horizon cells are supplied",
        },
    }


def write_expanded_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "aggregate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    with (output_dir / "requests.jsonl").open("w") as stream:
        for cell in report["cells"]:
            for row in cell["requests"]:
                stream.write(
                    json.dumps(
                        {"cell_id": cell["cell_id"], **row}, sort_keys=True, default=str
                    )
                    + "\n"
                )
    with (output_dir / "prefetch-lifecycle.jsonl").open("w") as stream:
        for cell in report["cells"]:
            for row in cell["prefetch_lifecycle"]:
                stream.write(
                    json.dumps(
                        {"cell_id": cell["cell_id"], **row}, sort_keys=True, default=str
                    )
                    + "\n"
                )
    with (output_dir / "coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["coverage"][0]))
        writer.writeheader()
        writer.writerows(report["coverage"])
    implemented = sum(row["status"] == "implemented" for row in report["coverage"])
    missing = [
        row["metric"] for row in report["coverage"] if row["status"] != "implemented"
    ]
    lines = [
        "# Combined MSR/prefetch expanded aggregation",
        "",
        f"Cells ingested: **{len(report['cells'])}**.",
        f"Coverage: **{implemented}/{len(report['coverage'])}** categories have supplied raw observations.",
        "",
        "CPU/report validation does not establish a provider benefit. Missing values remain null; observed subtotals are not complete totals.",
        "",
        "## Missing or GPU-unmeasured",
        "",
        ", ".join(f"`{metric}`" for metric in missing) if missing else "None.",
        "",
        "See `aggregate.json`, `requests.jsonl`, `prefetch-lifecycle.jsonl`, and `coverage.csv` for raw-field linkage and calculations.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def _load(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text()), str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msr", type=Path)
    parser.add_argument("--arc", type=Path)
    parser.add_argument("--lru", type=Path)
    parser.add_argument("--request-manifest", type=Path)
    parser.add_argument("--shared-prefetch", type=Path)
    parser.add_argument("--score-reversal-events", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--aggregate-results-root", "--results-root", type=Path, action="append"
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fixed-service-target-ms", type=float)
    parser.add_argument("--resource-price-basis", type=Path)
    args = parser.parse_args()
    if args.aggregate_results_root:
        if args.output_dir is None:
            parser.error("--output-dir is required with --aggregate-results-root")
        if args.bundle is not None:
            report = aggregate_bundle_results(
                args.bundle,
                args.aggregate_results_root,
                fixed_service_target_ms=args.fixed_service_target_ms,
                resource_price_basis=(
                    _load(args.resource_price_basis)
                    if args.resource_price_basis
                    else None
                ),
            )
            paths = [cell["result_path"] for cell in report["cells"]]
        else:
            if args.request_manifest is None:
                parser.error(
                    "--request-manifest is required without an expanded --bundle"
                )
            paths = sorted(
                path
                for root in args.aggregate_results_root
                for path in root.glob("**/result.json")
            )
            report = aggregate_result_paths(
                paths,
                _load(args.request_manifest),
                fixed_service_target_ms=args.fixed_service_target_ms,
                resource_price_basis=(
                    _load(args.resource_price_basis)
                    if args.resource_price_basis
                    else None
                ),
            )
        write_expanded_report(report, args.output_dir)
        return 0 if paths else 2
    if (
        args.msr is None
        or args.arc is None
        or args.lru is None
        or args.output is None
        or args.request_manifest is None
    ):
        parser.error(
            "comparison mode requires --msr, --arc, --lru, --request-manifest, and --output"
        )
    report = analyze(
        msr=_load(args.msr),
        arc=_load(args.arc),
        lru=_load(args.lru),
        request_manifest=_load(args.request_manifest),
        shared_prefetch=_load(args.shared_prefetch) if args.shared_prefetch else None,
        score_reversal_events=(
            json.loads(args.score_reversal_events.read_text())
            if args.score_reversal_events
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
