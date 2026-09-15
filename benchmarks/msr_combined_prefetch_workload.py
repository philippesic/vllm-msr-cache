# SPDX-License-Identifier: Apache-2.0
"""Build the exploratory combined MSR-retention/semantic-prefetch comparison.

This is a thin workload and plan adapter around
``benchmarks.eviction_prefetch_campaign``.  It deliberately does not simulate
serving benefit on CPU.  The CPU path freezes exact existing inputs, matched
arms, causal nomination requirements, and the result contract.  The execution
path delegates each selected cell to the existing campaign runner.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.eviction_prefetch_campaign import (
    _git_identity,
    _profile_payload,
    build_plan,
    digest,
    file_digest,
    validate_campaign_manifest,
    validate_request_manifest,
)

SCHEMA = "msr-combined-prefetch-workload-v1"
NOMINATION_SCHEMA = "semantic-cross-session-prefetch-nomination-v1"
RESULT_CONTRACT_SCHEMA = "msr-combined-prefetch-result-contract-v1"
EXPANDED_SCHEMA = "msr-combined-prefetch-expanded-campaign-v1"
FIXED_SERVICE_TARGET = {
    "id": "exact-return-release-to-first-token-le-1000ms-v1",
    "metric": "release_to_first_token_ms",
    "population": "all eligible exact-prefix first_return and repeated_return requests",
    "formula": "(requests.first_token_ns - requests.planned_release_ns) / 1e6",
    "maximum_ms": 1000.0,
    "inclusive_boundary": True,
    "release_lateness_included": True,
    "failure_or_timeout_is_violation": True,
    "applies_identically_to": ["all_policies", "all_memory_rows", "all_horizons"],
    "frozen_before_new_gpu_results": True,
    "selection_provenance": (
        "predeclared experimental service constraint; not a commercial SLO and not tuned "
        "from this campaign"
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    REPO_ROOT / "output/benefit-measurement-20260910-v1/execution-v1"
)
DEFAULT_BASE_CAMPAIGN = DEFAULT_BASE / "p4-campaign.json"
DEFAULT_BASE_REQUESTS = DEFAULT_BASE / "inputs/request-manifest-v1.json"
DEFAULT_COUNTERFACTUAL_EVIDENCE = (
    REPO_ROOT
    / "output/msr-combined-prefetch-cpu-20260910/owner-runtime/schema-refresh"
)
DEFAULT_REAL_CELL_RESULT = REPO_ROOT / (
    "output/minimum-benefit-screen/minimum-execution-20260910-v11/attempt2-ports/"
    "capture/attempt/block/minimum-activation-prefetch-pressure-screen2048-b320-"
    "rep-1-S-semantic-prefetch-threshold2/result.json"
)
DEFAULT_ACCEPTED_ACTIVATION_REQUESTS = REPO_ROOT / (
    "output/msr-combined-prefetch-cpu-20260910/owner-workload-expanded-review1/"
    "bundle-final-v3/reference-mixed/inputs/request-manifest-v1.json"
)
ACTIVATION_REPAIR_ID = "activation-admission-repair-a03-v1"
FRONTIER_DIAGNOSTIC_ID = "admission-frontier-probe-a04-v1"
TWO_SOURCE_ACTIVATION_ID = "two-source-activation-a05-v1"
DEFAULT_TWO_SOURCE_BUNDLE = REPO_ROOT / (
    "output/msr-combined-prefetch-gpu-20260910/two-source-coalesced-workload/"
    "bundle-final-v2/bundle.json"
)
DEFAULT_REMAINING_SLICES_V5B = REPO_ROOT / (
    "output/msr-combined-prefetch-gpu-20260910/remaining-slices-owner/"
    "bundle-final-v5b/bundle.json"
)
DEFAULT_REMAINING_SLICES_V5C = REPO_ROOT / (
    "output/msr-combined-prefetch-gpu-20260910/remaining-slices-owner/"
    "bundle-final-v5c/bundle.json"
)
DEFAULT_ACCEPTED_INDEPENDENT_REQUESTS = REPO_ROOT / (
    "output/msr-combined-prefetch-cpu-20260910/owner-workload-expanded-review1/"
    "bundle-final-v3/independent-inputs/inputs/request-manifest-v1.json"
)
DEMAND_CONTINUATION_ID = "two-source-demand-controls-a06-v1"
SEMANTIC_VALUE_SCENARIOS = (
    "semantic-value-order-ab-v1",
    "semantic-value-order-ba-v1",
)
REMAINING_SLICES_SCHEMA = "msr-remaining-activated-slices-proposal-v1"
REMAINING_SLICE_PROFILES = (
    "resource-reference",
    "resource-lower-memory",
    "resource-larger-workset",
    "horizon-short",
    "horizon-long",
    "load-burst-recovery",
)

PRODUCT_ARM = "M-msr-semantic-prefetch-threshold2"
ARC_DEMAND_ARM = "A-arc-demand-threshold2"
LRU_DEMAND_ARM = "L-lru-demand-threshold2"
NONSEMANTIC_PREFETCH_ARM = "P-arc-request-order-prefetch-threshold2"
MSR_FIFO_ARM = "R-msr-request-order-prefetch-threshold2"
_BASE_ARM_RENAMES = {
    "S-semantic-prefetch-threshold2": PRODUCT_ARM,
    "L-lru-prefetch-threshold2": LRU_DEMAND_ARM,
    "P-arc-prefetch-threshold2": NONSEMANTIC_PREFETCH_ARM,
}
SELECTED_ARMS = (
    ARC_DEMAND_ARM,
    LRU_DEMAND_ARM,
    NONSEMANTIC_PREFETCH_ARM,
    MSR_FIFO_ARM,
    PRODUCT_ARM,
)
SELECTED_REPLICATES = ("rep-1", "rep-2", "rep-3")


class WorkloadContractError(ValueError):
    """The combined-product workload is unsafe or scientifically unmatched."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkloadContractError(message)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _causal_requests(document: Mapping[str, object]) -> list[dict[str, object]]:
    requests = [
        copy.deepcopy(request)
        for request in document["requests"]
        if request.get("split") == "screen"
        and request.get("context_budget") == 2048
        and request.get("episode") == "causal_tool_resume"
    ]
    return sorted(requests, key=lambda request: int(request["request_index"]))


def _candidate_record(
    warm1: Mapping[str, object],
    warm2: Mapping[str, object],
    resume: Mapping[str, object],
) -> dict[str, object]:
    span = warm2["context_token_span"]
    context_tokens = int(span[1]) - int(span[0])
    return {
        "session_id": warm2["session_id"],
        "warm_request_ids": [warm1["request_id"], warm2["request_id"]],
        "last_observation_request_id": warm2["request_id"],
        "resume_request_id": resume["request_id"],
        "source_id": warm2["source_id"],
        "prompt_token_digest": warm2["prompt_token_digest"],
        "context_tokens": context_tokens,
        "cost_cohort": "expensive" if context_tokens >= 1024 else "cheap",
        "lead_ms": int(resume["relative_release_ms"]),
        "future_return_withheld_from_nomination": True,
    }


def build_workload_metadata(document: Mapping[str, object]) -> dict[str, object]:
    """Describe the existing exact trace without altering its serving inputs."""
    requests = _causal_requests(document)
    _require(len(requests) == 32, "expected causal_tool_resume32 workload")
    warm = requests[:16]
    pollution = requests[16:24]
    resumes = requests[24:]
    candidates = [
        _candidate_record(warm[2 * index], warm[2 * index + 1], resumes[index])
        for index in range(8)
    ]
    equal_cost = [item for item in candidates if item["context_tokens"] >= 1024]
    _require(len(equal_cost) >= 2, "need two meaningful equal-cost candidates")
    pair = equal_cost[:2]
    _require(
        pair[0]["context_tokens"] == pair[1]["context_tokens"],
        "nomination counterfactual candidates must have equal transfer size",
    )
    warm_sessions = {str(item["session_id"]) for item in candidates}
    pollution_sessions = {str(item["session_id"]) for item in pollution}
    return {
        "schema": SCHEMA,
        "classification_only": True,
        "serving_inputs_unchanged": True,
        "phase_counts": {"warm": 16, "pollution": 8, "resume": 8},
        "returning_candidates": candidates,
        "no_return_sessions": sorted(pollution_sessions - warm_sessions),
        "lead_times_ms": [int(item["lead_ms"]) for item in candidates],
        "cost_cohorts": {
            "expensive": sum(item["cost_cohort"] == "expensive" for item in candidates),
            "cheap": sum(item["cost_cohort"] == "cheap" for item in candidates),
        },
        "nomination_counterfactual": {
            "candidate_sessions": [item["session_id"] for item in pair],
            "candidate_source_ids": [item["source_id"] for item in pair],
            "candidate_context_tokens": [item["context_tokens"] for item in pair],
            "hint_publication_order": [item["session_id"] for item in pair],
            "shared_budget_blocks": int(320 * 0.05),
            "semantic_score_scenarios": [
                {
                    "id": "first_higher",
                    "scores": {
                        str(pair[0]["session_id"]): 0.9,
                        str(pair[1]["session_id"]): 0.1,
                    },
                    "expected_semantic_session": pair[0]["session_id"],
                    "expected_request_order_session": pair[0]["session_id"],
                },
                {
                    "id": "second_higher",
                    "scores": {
                        str(pair[0]["session_id"]): 0.1,
                        str(pair[1]["session_id"]): 0.9,
                    },
                    "expected_semantic_session": pair[1]["session_id"],
                    "expected_request_order_session": pair[0]["session_id"],
                },
            ],
            "future_demand_permutation_must_not_change_nomination": True,
        },
    }


def _exact_prefix_identity(request: Mapping[str, object]) -> str:
    boundary = request.get("known_prefix_boundary_tokens")
    token_digest = request.get("known_prefix_token_digest")
    if not isinstance(boundary, int):
        span = request["context_token_span"]
        boundary = int(span[1]) // 16 * 16
        token_digest = digest(request["prompt_token_ids"][:boundary])
    return digest(
        {
            "session_id": request["session_id"],
            "generation": request["generation"],
            "known_prefix_boundary_tokens": boundary,
            "known_prefix_token_digest": token_digest,
        }
    )


def derive_exact_request_history(
    requests: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Join reuse by exact prefix history, independent of position or phase labels."""
    ordered = sorted(
        requests,
        key=lambda request: (
            int(request["phase_index"]),
            int(request["relative_release_ms"]),
            int(request["request_index"]),
        ),
    )
    occurrences: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    history: dict[str, dict[str, object]] = {}
    return_counts: dict[str, int] = {}
    for trace_index, request in enumerate(ordered):
        identity = _exact_prefix_identity(request)
        boundary = request.get("known_prefix_boundary_tokens")
        token_digest = request.get("known_prefix_token_digest")
        if not isinstance(boundary, int):
            boundary = int(request["context_token_span"][1]) // 16 * 16
            token_digest = digest(request["prompt_token_ids"][:boundary])
        prior = occurrences.setdefault(identity, [])
        previous = prior[-1] if prior else None
        intervening = trace_index - previous[0] - 1 if previous else None
        prior_return_count = return_counts.get(identity, 0)
        if previous is None:
            role = "first_observation"
        elif prior_return_count > 0:
            role = "repeated_return"
            return_counts[identity] = prior_return_count + 1
        elif intervening == 0:
            role = "immediate_repeat"
        else:
            role = "first_return"
            return_counts[identity] = 1
        history[str(request["request_id"])] = {
            "exact_prefix_identity": identity,
            "session_id": request["session_id"],
            "generation": request["generation"],
            "known_prefix_boundary_tokens": boundary,
            "known_prefix_token_digest": token_digest,
            "exposure_index": len(prior) + 1,
            "history_role": role,
            "prior_source_request_id": (
                previous[1]["request_id"] if previous is not None else None
            ),
            "planned_intervening_request_count": intervening,
            "offline_only": True,
        }
        prior.append((trace_index, request))
    for identity, prior in occurrences.items():
        has_return = any(
            history[str(request["request_id"])]["history_role"]
            in {"first_return", "repeated_return"}
            for _, request in prior
        )
        if not has_return:
            final_id = str(prior[-1][1]["request_id"])
            history[final_id]["terminal_outcome"] = (
                "cold_no_return" if len(prior) == 1 else "censored_after_warm"
            )
        else:
            final_id = str(prior[-1][1]["request_id"])
            history[final_id]["terminal_outcome"] = "returned"
    return history


def meets_fixed_service_target(request: Mapping[str, object]) -> bool | None:
    if request.get("success") is not True or request.get("terminal_complete") is not True:
        return False
    planned = request.get("planned_release_ns")
    first_token = request.get("first_token_ns")
    if not isinstance(planned, int) or not isinstance(first_token, int):
        return None
    if first_token < planned:
        raise WorkloadContractError("first token precedes planned release")
    return (first_token - planned) / 1_000_000.0 <= float(
        FIXED_SERVICE_TARGET["maximum_ms"]
    )


def derive_actual_last_use_observations(
    request_rows: Sequence[Mapping[str, object]],
    request_history: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Measure completed-use gaps; planned predecessors never substitute for clocks."""
    rows = {
        str(row["manifest_request_id"]): row
        for row in request_rows
        if row.get("manifest_request_id") in request_history
    }
    observations = {}
    for request_id, history in request_history.items():
        if history.get("history_role") not in {"first_return", "repeated_return"}:
            continue
        row = rows.get(request_id)
        send_ns = row.get("send_ns") if row else None
        if not isinstance(send_ns, int):
            observations[request_id] = {
                "actual_last_use_gap_ns": None,
                "actual_last_use_source_request_id": None,
                "actual_intervening_completed_request_ids": None,
                "reason": "return send clock unavailable",
            }
            continue
        identity = history["exact_prefix_identity"]
        prior = []
        for candidate_id, candidate_history in request_history.items():
            candidate = rows.get(candidate_id)
            completed_ns = candidate.get("completion_ns") if candidate else None
            if (
                candidate_id != request_id
                and candidate_history.get("exact_prefix_identity") == identity
                and isinstance(completed_ns, int)
                and completed_ns <= send_ns
            ):
                prior.append((completed_ns, candidate_id))
        if not prior:
            observations[request_id] = {
                "actual_last_use_gap_ns": None,
                "actual_last_use_source_request_id": None,
                "actual_intervening_completed_request_ids": [],
                "reason": "no prior exact-prefix completion before send",
            }
            continue
        last_completion_ns, source_id = max(prior)
        intervening = sorted(
            candidate_id
            for candidate_id, candidate in rows.items()
            if candidate_id not in {source_id, request_id}
            and isinstance(candidate.get("completion_ns"), int)
            and last_completion_ns < candidate["completion_ns"] <= send_ns
        )
        observations[request_id] = {
            "actual_last_use_gap_ns": send_ns - last_completion_ns,
            "actual_last_use_source_request_id": source_id,
            "actual_intervening_completed_request_ids": intervening,
            "actual_intervening_completed_request_count": len(intervening),
            "hint_lead_ms": row.get("hint_lead_ms"),
            "reason": "measured_from_client_completion_to_return_send",
        }
    return observations


def _configure_arm(arm: dict[str, object]) -> None:
    config = arm["config"]
    extra = config["extra_config"]
    if arm["id"] in SELECTED_ARMS:
        extra["store_threshold"] = 2
        extra["decision_feature_trace"] = True
    if arm["id"] == PRODUCT_ARM:
        config["policy"] = "semantic-mean"
        extra.update(
            {
                "prefetch_trigger": "known_prefix_wait_hint",
                "prefetch_selector": "semantic",
                "prefetch_score_max_age_ms": 1000,
                "prefetch_budget_fraction": 0.05,
                "soft_retention": "bounded_hint",
                "shared_prefetch_scorer": False,
            }
        )
    elif arm["id"] in {NONSEMANTIC_PREFETCH_ARM, MSR_FIFO_ARM}:
        config["policy"] = (
            "semantic-mean" if arm["id"] == MSR_FIFO_ARM else "arc"
        )
        extra.update(
            {
                "prefetch_trigger": "known_prefix_wait_hint",
                "prefetch_selector": "request_order",
                "prefetch_budget_fraction": 0.05,
                "soft_retention": "bounded_hint",
                "shared_prefetch_scorer": False,
            }
        )
    elif arm["id"] in {ARC_DEMAND_ARM, LRU_DEMAND_ARM}:
        config["policy"] = "arc" if arm["id"] == ARC_DEMAND_ARM else "lru"
        extra.update(
            {
                "prefetch_trigger": "preemption",
                "prefetch_selector": "request_order",
                "prefetch_budget_fraction": 0.0,
                "soft_retention": "off",
                "shared_prefetch_scorer": False,
            }
        )
    arm["arm_config_hash"] = digest(config)


def _rename_arm_references(value: object) -> object:
    if isinstance(value, str):
        return _BASE_ARM_RENAMES.get(value, value)
    if isinstance(value, list):
        return [_rename_arm_references(item) for item in value]
    if isinstance(value, dict):
        return {key: _rename_arm_references(item) for key, item in value.items()}
    return value


def build_campaign(
    base: Mapping[str, object], request_path: Path, output_root: str
) -> dict[str, object]:
    campaign = _rename_arm_references(copy.deepcopy(dict(base)))
    campaign["request_manifest"] = {
        "path": str(Path("inputs") / request_path.name),
        "sha256": file_digest(request_path),
    }
    campaign["output_root"] = output_root
    for arm in campaign["arms"]:
        _configure_arm(arm)
    product = next(arm for arm in campaign["arms"] if arm["id"] == PRODUCT_ARM)
    msr_fifo = copy.deepcopy(product)
    msr_fifo["id"] = MSR_FIFO_ARM
    msr_fifo["scientific_row"] = MSR_FIFO_ARM
    msr_fifo["config"]["extra_config"]["prefetch_selector"] = "request_order"
    msr_fifo["arm_config_hash"] = digest(msr_fifo["config"])
    campaign["arms"].append(msr_fifo)
    profile = campaign["baseline_profiles"][0]
    profile["id"] = "msr-combined-prefetch-screen2048-dram6"
    profile["workload"] = {
        "split": "screen",
        "context_tokens": 2048,
        "episode": "causal_tool_resume",
    }
    profile["resources"]["cpu_bytes"] = 6 * 458_752
    profile["treatment_arm_ids"].append(MSR_FIFO_ARM)
    for replicate in profile["replication_plan"]:
        replicate["run_schedule"].append(
            {"arm_id": MSR_FIFO_ARM, "role": "treatment", "slot_id": MSR_FIFO_ARM}
        )
        replicate["order_schedule_digest"] = digest(
            {
                "seed": replicate["seed"],
                "run_schedule": replicate["run_schedule"],
                "request_order": replicate["request_order"],
            }
        )
    profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
    return campaign


def result_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": RESULT_CONTRACT_SCHEMA,
        "claim": (
            "combined MSR retention plus causal semantic prefetch reduces total "
            "actual prompt tokens locally executed per completed request"
        ),
        "comparators": [ARC_DEMAND_ARM, LRU_DEMAND_ARM],
        "semantic_attribution_control": MSR_FIFO_ARM,
        "practical_arc_prefetch_control": NONSEMANTIC_PREFETCH_ARM,
        "primary_metric": {
            "numerator": (
                "baseline_total_actual_prompt_tokens_locally_executed - "
                "msr_total_actual_prompt_tokens_locally_executed"
            ),
            "denominator": "baseline_total_actual_prompt_tokens_locally_executed",
            "formula_percent": "100 * numerator / denominator",
            "source": (
                "one authoritative terminal preservation event per completed request; "
                "actual_prompt_tokens_locally_executed must equal cumulative and the "
                "sum of admission episodes"
            ),
            "cohorts": ["whole_trace", "returning_candidates"],
        },
        "serving_constraint": {
            "metric": "median_ttft_ms",
            "maximum_msr_regression_percent_vs_each_baseline": 3.0,
        },
        "pair_key": ["profile_id", "replicate_id", "seed"],
        "required_result_fields": [
            "arm_id",
            "profile_id",
            "replicate_id",
            "seed",
            "valid",
            "request_order",
            "requests",
            "preservation_events",
            "wait_hint_publications",
            "wait_hint_events",
            "physical_transfer_accounting",
            "resolved_launch_argv",
            "config_delta",
        ],
        "required_activation": {
            "native_store_threshold": 2,
            "nomination_event_schema": NOMINATION_SCHEMA,
            "semantic_selector": "semantic",
            "observations_available_before_decision": True,
            "selected_exact_source_survives_in_dram": True,
            "staging_ready_before_demand": True,
            "request_usable_after_exact_bind_and_all_worker_fence_ack": True,
            "exact_kv_consumed": True,
            "exact_output_equality": True,
            "counts": [
                "misses",
                "wrong_or_no_return_nominations",
                "wasted_and_consumed_transfers",
                "displacements",
                "exposed_stall",
                "nomination_and_policy_overhead",
            ],
        },
        "nomination_counterfactual": metadata["nomination_counterfactual"],
        "interpretation": (
            "CPU validation establishes construction and fail-closed attribution only; "
            "it does not establish provider benefit or the TTFT constraint."
        ),
    }


def _runtime_nomination_capability() -> tuple[bool, str]:
    try:
        from semantic_offload import connector
    except ImportError as error:  # pragma: no cover - defensive import report
        return False, f"connector import failed: {error}"
    scheduler = getattr(connector, "SemanticOffloadingConnectorScheduler", None)
    required = (
        "_semantic_wait_key_scores",
        "_semantic_wait_nomination_candidate",
        "_retry_wait_prefetches",
    )
    missing = [name for name in required if scheduler is None or not hasattr(scheduler, name)]
    if missing:
        return False, "cross-session nomination methods unavailable: " + ", ".join(missing)
    return True, "cross-session nomination producer methods are present"


def _counterfactual_evidence(path: Path = DEFAULT_COUNTERFACTUAL_EVIDENCE) -> dict[str, object]:
    exit_path = path / "exit-status.txt"
    stdout_path = path / "stdout.txt"
    valid = (
        exit_path.is_file()
        and stdout_path.is_file()
        and exit_path.read_text(encoding="utf-8").strip() == "0"
        and "3 passed" in stdout_path.read_text(encoding="utf-8")
    )
    return {
        "path": str(path.resolve()),
        "valid": valid,
        "exit_status_sha256": file_digest(exit_path) if exit_path.is_file() else None,
        "stdout_sha256": file_digest(stdout_path) if stdout_path.is_file() else None,
        "required_tests": [
            "semantic_wait_nomination_reverses_winner_across_competing_sessions",
            "request_order_wait_nomination_preserves_publication_order",
            "semantic_nomination_excludes_future_score_and_keeps_exact_chain_prefix",
        ],
    }


def _retained_real_schema_evidence(path: Path = DEFAULT_REAL_CELL_RESULT) -> dict[str, object]:
    _require(path.is_file(), "retained real execute-cell result is unavailable")
    result = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "requests",
        "preservation_events",
        "wait_hint_publications",
        "wait_hint_events",
        "physical_transfer_accounting",
        "resolved_launch_argv",
    )
    missing = [field for field in required if field not in result]
    _require(
        result.get("schema") == "ep-campaign-cell-result-v1" and not missing,
        "retained execute-cell schema is incomplete",
    )
    return {
        "path": str(path.resolve()),
        "sha256": file_digest(path),
        "schema": result["schema"],
        "request_count": len(result["requests"]),
        "preservation_event_count": len(result["preservation_events"]),
        "ingested_fields": list(required),
        "attribution_compatibility": (
            "real schema accepted; historical policy labels are not current benefit evidence"
        ),
    }


def validate_nomination_event(
    event: Mapping[str, object], *, expected_selector: str
) -> None:
    _require(event.get("schema") == NOMINATION_SCHEMA, "nomination schema")
    _require(event.get("event") == "nomination_decision", "nomination event type")
    _require(event.get("selector") == expected_selector, "nomination selector")
    decision_ns = event.get("decision_ns")
    candidates = event.get("candidates")
    chosen = event.get("chosen")
    _require(isinstance(decision_ns, int) and decision_ns >= 0, "decision clock")
    _require(isinstance(candidates, list) and len(candidates) >= 2, "competing candidates")
    _require(isinstance(chosen, list) and chosen, "chosen nomination")
    forbidden = {"future_return", "future_demand", "resume_request_id"}
    for candidate in candidates:
        _require(not forbidden.intersection(candidate), "future demand leaked to nomination")
        for field in (
            "session_id",
            "source_request_id",
            "observed_prefix_id",
            "source_observed_ns",
            "hint_observation_available_ns",
            "hint_input_publication_ns",
            "eligible_resident_prefix_keys",
        ):
            _require(candidate.get(field) not in (None, ""), f"candidate missing {field}")
        clocks = [
            candidate["source_observed_ns"],
            candidate["hint_observation_available_ns"],
            candidate["hint_input_publication_ns"],
            decision_ns,
        ]
        _require(
            all(isinstance(value, int) and value >= 0 for value in clocks)
            and clocks == sorted(clocks),
            "nomination used a future or unordered observation",
        )
        if expected_selector == "semantic":
            scores = candidate.get("semantic_key_scores")
            _require(isinstance(scores, list) and scores, "semantic score evidence")
            _require(
                all(
                    isinstance(score.get("observation_available_ns"), int)
                    and score["observation_available_ns"] <= decision_ns
                    and isinstance(score.get("score"), (int, float))
                    for score in scores
                ),
                "semantic score availability",
            )


def validate_nomination_counterfactual(
    semantic_events: Sequence[Mapping[str, object]],
    request_order_winners: Sequence[str],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate the minimal score-reversal proof used to unlock GPU execution."""
    counterfactual = metadata["nomination_counterfactual"]
    scenarios = counterfactual["semantic_score_scenarios"]
    _require(len(semantic_events) == len(scenarios) == 2, "two semantic scenarios")
    _require(len(request_order_winners) == 2, "two request-order scenarios")
    publication_orders = []
    budgets = []
    selected_bytes = []
    semantic_winners = []
    for event, scenario in zip(semantic_events, scenarios, strict=True):
        validate_nomination_event(event, expected_selector="semantic")
        publication_orders.append(event.get("candidate_publication_order_event_ids"))
        budgets.append(event.get("shared_allocation_budget_blocks"))
        selected_bytes.append(event.get("selected_bytes"))
        chosen = event["chosen"]
        _require(len(chosen) == 1, "counterfactual requires one chosen session")
        winner = str(chosen[0]["session_id"])
        semantic_winners.append(winner)
        _require(
            winner == scenario["expected_semantic_session"],
            "semantic winner did not follow reversed prior scores",
        )
    _require(len({json.dumps(value, sort_keys=True) for value in publication_orders}) == 1,
             "publication order changed across score reversal")
    expected_budget = int(counterfactual["shared_budget_blocks"])
    _require(
        len(set(budgets)) == 1 and budgets[0] == expected_budget,
        "shared budget changed",
    )
    _require(len(set(selected_bytes)) == 1, "selected bytes changed")
    _require(len(set(semantic_winners)) == 2, "semantic winner did not reverse")
    _require(
        list(request_order_winners)
        == [
            scenario["expected_request_order_session"] for scenario in scenarios
        ],
        "request-order winner changed with semantic scores",
    )
    return {
        "schema": "msr-semantic-nomination-counterfactual-validation-v1",
        "valid": True,
        "semantic_winners": semantic_winners,
        "request_order_winners": list(request_order_winners),
        "shared_allocation_budget_blocks": budgets[0],
        "selected_bytes": selected_bytes[0],
        "candidate_publication_order_event_ids": publication_orders[0],
    }


def validate_expanded_live_competition(
    event: Mapping[str, object], dimensions: Mapping[str, object]
) -> dict[str, object]:
    validate_nomination_event(event, expected_selector="semantic")
    configured = int(dimensions["configured_shared_budget_blocks"])
    live = int(event["shared_allocation_budget_blocks"])
    _require(0 < live <= configured, "live allocation is outside configured budget")
    candidates = event["candidates"]
    eligible_counts = [len(candidate["eligible_resident_prefix_keys"]) for candidate in candidates]
    source_identities = {
        (candidate["session_id"], candidate["observed_prefix_id"])
        for candidate in candidates
    }
    _require(len(source_identities) == len(candidates), "candidate sources are duplicated")
    _require(
        len([count for count in eligible_counts if count > 0]) >= 2,
        "fewer than two surviving eligible exact sources",
    )
    _require(sum(eligible_counts) > live, "live allocation is not constrained")
    for candidate in candidates:
        eligible = candidate["eligible_resident_prefix_keys"]
        eligible_keys = {json.dumps(key, sort_keys=True) for key in eligible}
        scores = [float(score["score"]) for score in candidate["semantic_key_scores"]]
        _require(
            all(
                json.dumps(score["key"], sort_keys=True) in eligible_keys
                for score in candidate["semantic_key_scores"]
            ),
            "semantic score key is not eligible",
        )
        if scores:
            recomputed = sum(scores) / len(scores)
            _require(
                abs(float(candidate["aggregate_score"]) - recomputed) <= 1e-12,
                "candidate aggregate is not the recomputed mean",
            )
    chosen = event["chosen"]
    selected_count = sum(len(item["selected_keys"]) for item in chosen)
    _require(selected_count == int(event.get("selected_blocks", -1)), "selected count mismatch")
    _require(selected_count <= live, "selection exceeded live shared allocation")
    by_session = {candidate["session_id"]: candidate for candidate in candidates}
    for item in chosen:
        candidate = by_session.get(item["session_id"])
        _require(candidate is not None, "chosen source was not a candidate")
        selected = item["selected_keys"]
        _require(
            selected == candidate["eligible_resident_prefix_keys"][: len(selected)],
            "chosen keys are not an immutable contiguous eligible prefix",
        )
    ranked = [
        candidate
        for candidate in candidates
        if candidate.get("aggregate_score") is not None
    ]
    if ranked and chosen:
        highest = max(float(candidate["aggregate_score"]) for candidate in ranked)
        _require(
            float(by_session[chosen[0]["session_id"]]["aggregate_score"]) == highest,
            "first chosen source is not the highest recomputed mean",
        )
    return {
        "schema": "msr-expanded-live-competition-validation-v1",
        "valid": True,
        "configured_shared_budget_blocks": configured,
        "live_shared_allocation_budget_blocks": live,
        "eligible_source_count": len(eligible_counts),
        "eligible_block_demand": sum(eligible_counts),
    }


def validate_bundle_contract(
    campaign: Mapping[str, object],
    plan: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    arms = {arm["id"]: arm for arm in campaign["arms"]}
    _require(set(SELECTED_ARMS).issubset(arms), "missing selected arm")
    for arm_id in SELECTED_ARMS:
        extra = arms[arm_id]["config"]["extra_config"]
        _require(extra.get("store_threshold") == 2, f"{arm_id} threshold")
    _require(
        arms[PRODUCT_ARM]["config"]["policy"] == "semantic-mean"
        and arms[PRODUCT_ARM]["config"]["extra_config"]["prefetch_selector"]
        == "semantic",
        "product arm is not MSR plus semantic prefetch",
    )
    _require(
        arms[NONSEMANTIC_PREFETCH_ARM]["config"]["extra_config"]["prefetch_selector"]
        == "request_order",
        "ARC FIFO control",
    )
    product_config = copy.deepcopy(arms[PRODUCT_ARM]["config"])
    fifo_config = copy.deepcopy(arms[MSR_FIFO_ARM]["config"])
    product_config["extra_config"]["prefetch_selector"] = "request_order"
    _require(
        fifo_config == product_config,
        "MSR FIFO attribution control must differ only by selector",
    )
    for arm_id in (ARC_DEMAND_ARM, LRU_DEMAND_ARM):
        _require(
            arms[arm_id]["config"]["extra_config"]["prefetch_budget_fraction"] == 0.0,
            f"{arm_id} must be demand restore",
        )
    selected = [
        cell
        for cell in plan["cells"]
        if cell["arm_id"] in SELECTED_ARMS
        and cell["replicate_id"] in SELECTED_REPLICATES
    ]
    _require(len(selected) == 15, "expected three matched five-arm repetitions")
    for replicate in SELECTED_REPLICATES:
        cells = [cell for cell in selected if cell["replicate_id"] == replicate]
        _require({cell["arm_id"] for cell in cells} == set(SELECTED_ARMS), "matched arms")
        _require(len({cell["request_proof"]["selected_request_digest"] for cell in cells}) == 1,
                 "matched arms changed external requests")
        _require(len({cell["seed"] for cell in cells}) == 1, "matched arms changed seed")
    counterfactual = metadata["nomination_counterfactual"]
    scenarios = counterfactual["semantic_score_scenarios"]
    _require(
        scenarios[0]["expected_semantic_session"]
        != scenarios[1]["expected_semantic_session"],
        "reversed semantic scores must reverse chosen session",
    )
    _require(
        scenarios[0]["expected_request_order_session"]
        == scenarios[1]["expected_request_order_session"],
        "request-order control must ignore score reversal",
    )


def build_bundle(
    *,
    base_campaign_path: Path,
    base_request_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    base_campaign = json.loads(base_campaign_path.read_text(encoding="utf-8"))
    request_document = json.loads(base_request_path.read_text(encoding="utf-8"))
    validate_request_manifest(request_document)
    metadata = build_workload_metadata(request_document)
    request_document["msr_combined_prefetch_workload"] = metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in base_request_path.parent.iterdir():
        if source.name == base_request_path.name:
            continue
        destination = input_dir / source.name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    request_path = input_dir / "request-manifest-v1.json"
    _write_json(request_path, request_document)
    validate_request_manifest(request_document)

    campaign = build_campaign(base_campaign, request_path, future_result_root)
    campaign_path = output_dir / "campaign.json"
    _write_json(campaign_path, campaign)
    validate_campaign_manifest(campaign, campaign_path)
    plan = build_plan(campaign, campaign_path)
    plan_path = output_dir / "plan.json"
    _write_json(plan_path, plan)
    validate_bundle_contract(campaign, plan, metadata)

    capability, capability_reason = _runtime_nomination_capability()
    counterfactual_evidence = _counterfactual_evidence()
    gpu_ready = capability and bool(counterfactual_evidence["valid"])
    selected_cells = [
        {
            "cell_id": cell["cell_id"],
            "arm_id": cell["arm_id"],
            "profile_id": cell["profile_id"],
            "replicate_id": cell["replicate_id"],
            "seed": cell["seed"],
            "port": cell["port"],
            "request_digest": cell["request_proof"]["selected_request_digest"],
        }
        for cell in plan["cells"]
        if cell["arm_id"] in SELECTED_ARMS
        and cell["replicate_id"] in SELECTED_REPLICATES
    ]
    command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-selected --bundle {output_dir / 'bundle.json'} "
        f"--results {future_result_root} --replicates rep-1 --arms "
        f"{ARC_DEMAND_ARM} {LRU_DEMAND_ARM} {MSR_FIFO_ARM} {PRODUCT_ARM}"
    )
    bundle = {
        "schema": SCHEMA,
        "stage": "exploratory_cpu_validated_gpu_unmeasured",
        "base_inputs": {
            "campaign_path": str(base_campaign_path.resolve()),
            "campaign_sha256": file_digest(base_campaign_path),
            "request_path": str(base_request_path.resolve()),
            "request_sha256": file_digest(base_request_path),
        },
        "generated": {
            "campaign_path": str(campaign_path.resolve()),
            "campaign_portable_path": "campaign.json",
            "campaign_sha256": file_digest(campaign_path),
            "request_path": str(request_path.resolve()),
            "request_portable_path": "inputs/request-manifest-v1.json",
            "request_sha256": file_digest(request_path),
            "plan_path": str(plan_path.resolve()),
            "plan_portable_path": "plan.json",
            "plan_sha256": file_digest(plan_path),
            "plan_digest": plan["plan_digest"],
        },
        "selected_cells": selected_cells,
        "workload": metadata,
        "result_contract": result_contract(metadata),
        "gpu_readiness": {
            "ready": gpu_ready,
            "reason": (
                capability_reason
                if gpu_ready
                else capability_reason + "; retained CPU counterfactual is missing or failed"
            ),
            "counterfactual_evidence": counterfactual_evidence,
            "source_stage_required": (
                "run from the clean staged RTX 5080 source; execute-selected refreshes "
                "implementation revision/tree before launching existing execute-cell"
            ),
        },
        "future_gpu_command": command,
        "benefit_measured": False,
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-command.txt").write_text(command + "\n", encoding="utf-8")
    return bundle


EXPANDED_VARIANTS = (
    {
        "id": "reference-mixed",
        "purpose": "combined_recomputation_serving_and_prefetch_activation",
        "dram_blocks": 1024,
        "lead_pattern_ms": (50, 250, 750, 50, 250, 750, 250, 750),
        "input_cohort": "screen2048-primary",
        "source_split": "screen",
        "episode": "causal_tool_resume",
        "arrival_pattern": "mixed_resume_leads",
    },
    {
        "id": "capacity-pressure",
        "purpose": "useful_capacity_at_smaller_dram_budget",
        "dram_blocks": 320,
        "lead_pattern_ms": (50, 250, 750, 50, 250, 750, 250, 750),
        "input_cohort": "screen2048-primary",
        "source_split": "screen",
        "episode": "causal_tool_resume",
        "arrival_pattern": "mixed_resume_leads",
    },
    {
        "id": "short-horizon",
        "purpose": "short_idle_gap_with_fixed_intervening_traffic",
        "dram_blocks": 1024,
        "lead_pattern_ms": (50,) * 8,
        "input_cohort": "screen2048-primary",
        "source_split": "screen",
        "episode": "causal_tool_resume",
        "arrival_pattern": "eight_resume_burst_at_50ms",
    },
    {
        "id": "long-horizon",
        "purpose": "long_idle_gap_with_fixed_intervening_traffic",
        "dram_blocks": 1024,
        "lead_pattern_ms": (750,) * 8,
        "input_cohort": "screen2048-primary",
        "source_split": "screen",
        "episode": "causal_tool_resume",
        "arrival_pattern": "eight_resume_burst_at_750ms",
    },
    {
        "id": "middle-burst",
        "purpose": "bounded_concurrency_and_tail_behavior",
        "dram_blocks": 1024,
        "lead_pattern_ms": (250,) * 8,
        "input_cohort": "screen2048-primary",
        "source_split": "screen",
        "episode": "causal_tool_resume",
        "arrival_pattern": "eight_resume_burst_at_250ms",
    },
    {
        "id": "independent-inputs",
        "purpose": "independent_exact_input_cohort_confirmation",
        "dram_blocks": 1024,
        "lead_pattern_ms": (50, 250, 750, 50, 250, 750, 250, 750),
        "input_cohort": "heldout2048-relabeled-exploratory",
        "source_split": "heldout",
        "episode": "causal_tool_resume",
        "arrival_pattern": "mixed_resume_leads",
    },
    {
        "id": "lower-offered-rate-recovery",
        "purpose": "bounded_lower_offered_rate_and_recovery_observation",
        "dram_blocks": 1024,
        "input_cohort": "screen2048-reusable-load",
        "source_split": "screen",
        "episode": "reusable_prefix",
        "arrival_pattern": "pollution_500ms_spacing_then_recovery_125ms_spacing",
        "pollution_release_ms": tuple(500 * index for index in range(8)),
        "recovery_release_ms": tuple(125 * index for index in range(8)),
    },
    {
        "id": "higher-offered-burst-recovery",
        "purpose": "bounded_higher_offered_burst_and_post_burst_recovery",
        "dram_blocks": 1024,
        "input_cohort": "screen2048-reusable-load",
        "source_split": "screen",
        "episode": "reusable_prefix",
        "arrival_pattern": "eight_pollution_burst_then_recovery_125ms_spacing",
        "pollution_release_ms": (0,) * 8,
        "recovery_release_ms": tuple(125 * index for index in range(8)),
    },
)


def _swap_split_labels(document: dict[str, object]) -> None:
    for request in document["requests"]:
        request["split"] = "screen" if request["split"] == "heldout" else "heldout"
    order = document.get("episode_order_digests", {})
    replaced = {}
    for name, value in order.items():
        prefix, separator, suffix = str(name).partition(":")
        if prefix == "screen":
            prefix = "heldout"
        elif prefix == "heldout":
            prefix = "screen"
        replaced[prefix + separator + suffix] = value
    document["episode_order_digests"] = replaced


def _expanded_request_document(
    base: Mapping[str, object], variant: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    document = copy.deepcopy(dict(base))
    if variant["source_split"] == "heldout":
        _swap_split_labels(document)
    episode = str(variant["episode"])
    selected = sorted(
        (
            request
            for request in document["requests"]
            if request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == episode
        ),
        key=lambda request: int(request["request_index"]),
    )
    if episode == "causal_tool_resume":
        resumes = [request for request in selected if request.get("phase") == "resume"]
        _require(len(resumes) == 8, "expanded variant needs eight resume requests")
        for request, lead_ms in zip(resumes, variant["lead_pattern_ms"], strict=True):
            request["relative_release_ms"] = int(lead_ms)
            request["hint_lead_ms"] = int(lead_ms)
            request["arrival_origin"] = f"controlled_resume_release_{lead_ms}ms"
    else:
        pollution = [request for request in selected if request.get("phase") == "pollution"]
        recovery = [request for request in selected if request.get("phase") == "revisit"]
        _require(
            len(pollution) == len(recovery) == 8,
            "load contrast needs eight offered and eight recovery requests",
        )
        for request, release_ms in zip(
            pollution, variant["pollution_release_ms"], strict=True
        ):
            request["relative_release_ms"] = int(release_ms)
            request["arrival_origin"] = f"controlled_offered_release_{release_ms}ms"
        for request, release_ms in zip(
            recovery, variant["recovery_release_ms"], strict=True
        ):
            request["relative_release_ms"] = int(release_ms)
            request["arrival_origin"] = f"post_drain_recovery_release_{release_ms}ms"
    history = derive_exact_request_history(selected)
    for request in selected:
        request["msr_exact_history"] = history[str(request["request_id"])]
    return_sources = sorted(
        (
            request
            for request in selected
            if history[str(request["request_id"])]["history_role"]
            in {"first_return", "repeated_return"}
        ),
        key=lambda request: (
            history[str(request["request_id"])]["exact_prefix_identity"],
            str(request["request_id"]),
        ),
    )
    hint_only = (
        document.get("causal_tool_resume_extension", {}).get("hint_only_events", [])
        if episode == "causal_tool_resume"
        else []
    )
    _require(
        len(return_sources) >= len(hint_only),
        "hint-only events lack exact previously observed sources",
    )
    for spec, source in zip(hint_only, return_sources, strict=False):
        spec["source_request_id"] = source["request_id"]
        spec["source_exact_prefix_identity"] = history[str(source["request_id"])][
            "exact_prefix_identity"
        ]
    selected_ids = {str(request["request_id"]) for request in selected}
    _require(
        all(spec.get("source_request_id") in selected_ids for spec in hint_only),
        "hint-only source does not bind the selected exact cohort",
    )
    boundaries = [
        int(history[str(item["request_id"])]["known_prefix_boundary_tokens"]) // 16
        for item in selected
        if history[str(item["request_id"])]["history_role"]
        in {"first_return", "repeated_return"}
    ]
    shared_budget = int(320 * 0.05)
    eligible = sorted(blocks for blocks in boundaries if blocks >= shared_budget)
    if episode == "causal_tool_resume":
        _require(len(eligible) >= 2, "need two exact return sources above live budget")
    identity_fields = (
        "source_id",
        "source_sha256",
        "input_sha256",
        "context_sha256",
        "known_prefix_token_digest",
        "prompt_token_digest",
    )
    identity_sets = {
        field: sorted(
            {
                str(request[field])
                for request in selected
                if request.get(field) not in (None, "")
            }
        )
        for field in identity_fields
    }
    cohort_contexts = identity_sets["context_sha256"] or identity_sets["source_id"]
    metadata = {
        "schema": "msr-expanded-request-cohort-v1",
        "campaign_id": variant["id"],
        "input_cohort": variant["input_cohort"],
        "source_split": variant["source_split"],
        "execution_split": "screen",
        "episode": episode,
        "input_cohort_identity": digest(cohort_contexts),
        "input_identity_sets": identity_sets,
        "history_join": (
            "session+generation+known_prefix_boundary_tokens+known_prefix_token_digest"
        ),
        "request_history": history,
        "dimensions": {
            "gpu_blocks": 320,
            "dram_blocks": variant["dram_blocks"],
            "dram_bytes": int(variant["dram_blocks"]) * 458_752,
            "logical_return_working_set_blocks": sum(boundaries),
            "minimum_competing_source_blocks": eligible[0] if eligible else None,
            "shared_prefetch_budget_fraction": 0.05,
            "configured_shared_budget_blocks": shared_budget,
            "required_simultaneous_surviving_sources": (
                2 if episode == "causal_tool_resume" else 0
            ),
            "competition_demand_lower_bound_blocks": (
                eligible[0] + eligible[1]
                if episode == "causal_tool_resume"
                else None
            ),
            "prescribed_intervening_pollution_requests": 8,
            "hint_lead_ms": (
                sorted({int(value) for value in variant["lead_pattern_ms"]})
                if episode == "causal_tool_resume"
                else []
            ),
            "actual_last_use_gap_ns": (
                "later_request.send_ns - prior_exact_source.completion_ns"
            ),
            "actual_intervening_completed_request_count": (
                "count other completed requests with completion_ns after prior source "
                "completion and at or before later send_ns"
            ),
            "arrival_pattern": variant["arrival_pattern"],
            "offered_request_count": 8,
            "recovery_request_count": 8 if episode == "reusable_prefix" else 0,
            "recovery_phase": 17 if episode == "reusable_prefix" else None,
            "recovery_semantics": (
                "post-phase16-drain probes, not continuing-backlog recovery"
                if episode == "reusable_prefix"
                else None
            ),
            "prescribed_phase16_release_offsets_ms": (
                list(variant["pollution_release_ms"])
                if episode == "reusable_prefix"
                else [125 * index for index in range(8)]
            ),
            "prescribed_phase17_recovery_offsets_ms": (
                list(variant["recovery_release_ms"])
                if episode == "reusable_prefix"
                else []
            ),
            "offered_load_descriptor": (
                "one request every 500ms across 0..3500ms (2 requests/s interval rate)"
                if variant["id"] == "lower-offered-rate-recovery"
                else (
                    "8 requests at one phase origin (intentional bounded burst)"
                    if variant["id"] == "higher-offered-burst-recovery"
                    else "causal activity schedule; not an offered-load contrast"
                )
            ),
            "max_server_concurrency": 4,
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
        },
        "offline_labels_never_published_to_serving": True,
    }
    document["msr_expanded_campaign"] = metadata
    validate_request_manifest(document)
    return document, metadata


_STRUCTURAL_REQUEST_FIELDS = (
    "request_id",
    "request_index",
    "phase",
    "phase_index",
    "phase_ordinal",
    "relative_release_ms",
    "depends_on_phase_completion",
    "wait_origin",
    "arrival_origin",
    "pressure_origin",
    "revisit_origin",
    "fresh_server_id",
    "split",
    "context_budget",
    "episode",
)


def _replace_request_content(
    target: dict[str, object],
    source: Mapping[str, object],
    *,
    session_id: str,
) -> None:
    structural = {
        field: copy.deepcopy(target[field])
        for field in _STRUCTURAL_REQUEST_FIELDS
        if field in target
    }
    replacement = copy.deepcopy(dict(source))
    replacement.update(structural)
    replacement["session_id"] = session_id
    replacement["serving_inputs"] = copy.deepcopy(dict(source["serving_inputs"]))
    replacement["serving_inputs"]["session_id"] = session_id
    replacement["serving_inputs"]["generation"] = replacement["generation"]
    target.clear()
    target.update(replacement)


def _build_activation_repair_requests_budget1_draft(
    accepted: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Create a new a03-derived cohort that walks threshold-2 admission."""
    document = copy.deepcopy(dict(accepted))
    selected = sorted(
        (
            request
            for request in document["requests"]
            if request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == "causal_tool_resume"
        ),
        key=lambda request: int(request["request_index"]),
    )
    _require(len(selected) == 32, "repair cohort requires causal_tool_resume32")
    original = copy.deepcopy(selected)
    target_templates = (original[6], original[14])
    _require(
        all(int(item["known_prefix_boundary_tokens"]) // 16 == 129 for item in target_templates)
        and target_templates[0]["known_prefix_token_digest"]
        != target_templates[1]["known_prefix_token_digest"],
        "repair targets must be two distinct 129-block exact prefixes",
    )
    all_requests = accepted["requests"]
    pressure_pool = []
    seen_pressure = set()
    target_digests = {item["prompt_token_digest"] for item in target_templates}
    for request in all_requests:
        if (
            request.get("episode") == "causal_tool_resume"
            and request.get("family") == "repobench-p"
            and 2500 <= len(request.get("prompt_token_ids", [])) <= 5000
            and request.get("prompt_token_digest") not in target_digests
            and request.get("prompt_token_digest") not in seen_pressure
        ):
            seen_pressure.add(request["prompt_token_digest"])
            pressure_pool.append(request)
    _require(len(pressure_pool) >= 8, "repair cohort lacks eight distinct pressure prompts")

    target_request_ids = {0: [], 1: []}
    pressure_request_ids = []
    for pair_index in range(8):
        target_index = pair_index % 2
        target = target_templates[target_index]
        session_id = str(target["session_id"])
        warm1 = selected[2 * pair_index]
        warm2 = selected[2 * pair_index + 1]
        _replace_request_content(warm1, target, session_id=session_id)
        _replace_request_content(
            warm2, pressure_pool[pair_index], session_id=session_id
        )
        target_request_ids[target_index].append(warm1["request_id"])
        pressure_request_ids.append(warm2["request_id"])

    unrelated_resumes = []
    seen_sessions = {str(item["session_id"]) for item in target_templates}
    for request in all_requests:
        if (
            request.get("episode") == "causal_tool_resume"
            and request.get("phase") == "resume"
            and request.get("family") == "repobench-p"
            and request.get("session_id") not in seen_sessions
            and request.get("known_prefix_boundary_tokens", 0) >= 128
        ):
            unrelated_resumes.append(request)
            seen_sessions.add(str(request["session_id"]))
    _require(len(unrelated_resumes) >= 6, "repair cohort lacks unmatched resume controls")
    for index, target in enumerate(target_templates):
        resume = selected[24 + index]
        _replace_request_content(resume, target, session_id=str(target["session_id"]))
        resume["relative_release_ms"] = 750
        resume["hint_lead_ms"] = 750
        resume["arrival_origin"] = "repair_competing_hint_release_750ms"
    for offset, source in enumerate(unrelated_resumes[:6], start=2):
        resume = selected[24 + offset]
        _replace_request_content(resume, source, session_id=str(source["session_id"]))
        resume["relative_release_ms"] = 750
        resume["hint_lead_ms"] = 750
        resume["arrival_origin"] = "repair_unmatched_hint_release_750ms"

    document["causal_tool_resume_extension"]["hint_only_events"] = []
    history = derive_exact_request_history(selected)
    for request in selected:
        request["msr_exact_history"] = history[str(request["request_id"])]
    target_records = []
    for index, target in enumerate(target_templates):
        target_records.append(
            {
                "target_id": f"target-{index + 1}",
                "session_id": target["session_id"],
                "known_prefix_token_digest": target["known_prefix_token_digest"],
                "known_prefix_boundary_tokens": target["known_prefix_boundary_tokens"],
                "source_blocks": int(target["known_prefix_boundary_tokens"]) // 16,
                "separated_target_request_ids": target_request_ids[index],
                "expected_threshold_walk_ready_prefix_blocks": 2,
            }
        )
    metadata = {
        "schema": "msr-a03-activation-repair-cohort-v1",
        "campaign_id": ACTIVATION_REPAIR_ID,
        "stage": "new_exploratory_activation_cohort",
        "parent_frozen_request_sha256": file_digest(DEFAULT_ACCEPTED_ACTIVATION_REQUESTS),
        "observed_a03_cause": {
            "threshold2_counts_key_lookups_not_requests": True,
            "immediate_warm2_gpu_hits_did_not_admit_long_prefixes": True,
            "accepted_hint_events_repeated_one_128_token_session": True,
            "max_distinct_candidate_sessions": 1,
            "max_eligible_and_scored_blocks": 3,
            "observed_remaining_budget_blocks": 15,
        },
        "target_sources": target_records,
        "pressure_request_ids": pressure_request_ids,
        "admission_sequence": "target1,target2 alternating; each target followed by a distinct long pressure prompt",
        "hint_publication_contract": {
            "first_manifest_request_ids": [selected[24]["request_id"], selected[25]["request_id"]],
            "first_sessions": [item["session_id"] for item in target_records],
            "same_lead_ms": 750,
            "hint_only_events": 0,
            "remaining_hints_are_unmatched_controls": 6,
        },
        "configured_shared_budget_blocks": 1,
        "expected_eligible_demand_lower_bound_blocks": 4,
        "prefetch_score_max_age_ms": 15_000,
        "required_gpu_observations": {
            "distinct_candidate_sessions": 2,
            "each_ready_contiguous_dram_prefix_blocks_at_least": 2,
            "each_has_fresh_causal_score": True,
            "eligible_candidate_blocks_exceed_observed_remaining_budget": True,
            "shared_budget_constrained": True,
            "exact_output_and_consumption": True,
        },
        "falsifiable_experiment_contract": {
            "hypothesis": (
                "after native threshold2 admission walking, two distinct exact 2064-token "
                "sources are simultaneously fresh and DRAM-ready, so M ranks them under "
                "the observed shared budget while R preserves publication order"
            ),
            "available_causal_inputs": (
                "only completed pre-hint target/pressure requests, their exact prefix "
                "registrations, ready DRAM keys and score observations"
            ),
            "constrained_action": (
                "restore only a contiguous exact prefix from one of the published sources"
            ),
            "matched_controls": [ARC_DEMAND_ARM, LRU_DEMAND_ARM, MSR_FIFO_ARM, PRODUCT_ARM],
            "costs_included": [
                "all 41 admission-training requests",
                "first misses and recomputation",
                "stores, transfers, displacement, scoring and no-action hints",
            ],
            "fixed_stop_condition": (
                "after rep-1 M/R, stop if either lacks two distinct scored ready sources, "
                "deduplicated eligible keys do not exceed observed remaining budget, "
                "or exact output/consumption fails"
            ),
            "heldout_use": "none; this new exploratory cohort derives only from frozen screen inputs",
        },
        "benefit_measured": False,
    }
    document["msr_expanded_campaign"] = {
        "schema": "msr-expanded-request-cohort-v1",
        "campaign_id": ACTIVATION_REPAIR_ID,
        "input_cohort": "a03-preserved-inputs-admission-walk-v1",
        "source_split": "screen",
        "execution_split": "screen",
        "episode": "causal_tool_resume",
        "history_join": (
            "session+generation+known_prefix_boundary_tokens+known_prefix_token_digest"
        ),
        "request_history": history,
        "activation_repair": metadata,
        "offline_labels_never_published_to_serving": True,
    }
    validate_request_manifest(document)
    return document, metadata


def build_activation_repair_requests(
    accepted: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Extend one cell with enough pre-hint probes for threshold-2 admission."""
    document = copy.deepcopy(dict(accepted))
    original = sorted(
        (
            request
            for request in document["requests"]
            if request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == "causal_tool_resume"
        ),
        key=lambda request: int(request["request_index"]),
    )
    _require(len(original) == 32, "repair source requires causal_tool_resume32")
    targets = (copy.deepcopy(original[6]), copy.deepcopy(original[14]))
    _require(
        all(int(item["known_prefix_boundary_tokens"]) == 2064 for item in targets)
        and targets[0]["known_prefix_token_digest"]
        != targets[1]["known_prefix_token_digest"],
        "repair requires two distinct meaningful 2064-token targets",
    )
    pressure = next(
        request
        for request in document["requests"]
        if request.get("episode") == "causal_tool_resume"
        and request.get("split") == "screen"
        and request.get("family") == "repobench-p"
        and request.get("prompt_token_digest")
        not in {item["prompt_token_digest"] for item in targets}
        and 4500 <= len(request.get("prompt_token_ids", [])) <= 5000
    )

    def training_record(
        source: Mapping[str, object], index: int, label: str
    ) -> dict[str, object]:
        record = copy.deepcopy(dict(source))
        record.update(
            {
                "request_id": (
                    f"screen-2048-causal-tool-resume-{index:02d}-"
                    f"activation-train-{label}"
                ),
                "request_index": index,
                "phase": "activation_train",
                "phase_index": index,
                "phase_ordinal": 0,
                "relative_release_ms": 0,
                "depends_on_phase_completion": index - 1 if index else None,
                "wait_origin": (
                    "fresh_server_start"
                    if index == 0
                    else "all_prior_phase_requests_completed"
                ),
                "arrival_origin": "a03_repair_sequential_admission_walk",
                "pressure_origin": "a03_observed_first_miss_threshold_walk",
                "revisit_origin": None,
                "fresh_server_id": f"screen-2048-{ACTIVATION_REPAIR_ID}",
                "split": "screen",
                "context_budget": 2048,
                "episode": "causal_tool_resume",
                "suffix_budget_tokens": 1,
            }
        )
        serving = copy.deepcopy(dict(record["serving_inputs"]))
        serving["session_id"] = record["session_id"]
        boundary = int(record["context_token_span"][1]) // 16 * 16
        record["known_prefix_boundary_tokens"] = boundary
        record["known_prefix_token_digest"] = digest(
            record["prompt_token_ids"][:boundary]
        )
        serving["known_prefix_boundary_tokens"] = boundary
        serving["known_prefix_token_digest"] = record[
            "known_prefix_token_digest"
        ]
        record["serving_inputs"] = serving
        return record

    training = [
        training_record(targets[index % 2], index, f"target-{index % 2 + 1}")
        for index in range(40)
    ]
    training.append(training_record(pressure, 40, "final-pressure"))
    activity = []
    for ordinal, source in enumerate(original[16:24]):
        record = copy.deepcopy(source)
        record.update(
            {
                "request_id": (
                    f"screen-2048-causal-tool-resume-{41 + ordinal:02d}-"
                    f"pollution-{record['family']}"
                ),
                "request_index": 41 + ordinal,
                "phase_index": 41,
                "phase_ordinal": ordinal,
                "relative_release_ms": 125 * ordinal,
                "depends_on_phase_completion": 40,
                "fresh_server_id": f"screen-2048-{ACTIVATION_REPAIR_ID}",
            }
        )
        activity.append(record)

    unrelated = []
    target_sessions = {str(item["session_id"]) for item in targets}
    target_sessions.add(str(pressure["session_id"]))
    for request in document["requests"]:
        if (
            request.get("episode") == "causal_tool_resume"
            and request.get("split") == "screen"
            and request.get("phase") == "resume"
            and request.get("session_id") not in target_sessions
            and request.get("known_prefix_boundary_tokens", 0) >= 128
            and len(request.get("prompt_token_ids", [])) <= 5000
        ):
            unrelated.append(request)
            target_sessions.add(str(request["session_id"]))
    _require(len(unrelated) >= 6, "repair requires six unmatched hint controls")
    resume_sources = [targets[0], targets[1], *unrelated[:6]]
    for ordinal, source in enumerate(resume_sources):
        record = copy.deepcopy(dict(source))
        record.update(
            {
                "request_id": (
                    f"screen-2048-causal-tool-resume-{49 + ordinal:02d}-"
                    f"resume-{record['family']}"
                ),
                "request_index": 49 + ordinal,
                "phase": "resume",
                "phase_index": 41,
                "phase_ordinal": ordinal,
                "relative_release_ms": 750,
                "hint_lead_ms": 750,
                "depends_on_phase_completion": 40,
                "wait_origin": "a03_repair_common_750ms_wait",
                "arrival_origin": "a03_repair_competing_release_750ms",
                "pressure_origin": "a03_repair_post_admission_activity",
                "revisit_origin": "exact_repeat_after_admission_walk",
                "fresh_server_id": f"screen-2048-{ACTIVATION_REPAIR_ID}",
                "split": "screen",
                "context_budget": 2048,
                "episode": "causal_tool_resume",
            }
        )
        serving = copy.deepcopy(dict(record["serving_inputs"]))
        serving["session_id"] = record["session_id"]
        record["serving_inputs"] = serving
        activity.append(record)

    document["requests"] = [
        request
        for request in document["requests"]
        if not (
            request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == "causal_tool_resume"
        )
    ] + training + activity
    document["causal_tool_resume_extension"]["hint_only_events"] = []
    document["causal_tool_resume_extension"]["activation_training"] = {
        "schema": "msr-activation-training-extension-v1",
        "split": "screen",
        "context_budget": 2048,
        "training_request_count": 41,
        "activity_phase_index": 41,
        "activity_completion_dependency": 40,
        "target_sequence": [f"target-{index % 2 + 1}" for index in range(40)]
        + ["final-pressure"],
    }
    causal_order = [item["request_id"] for item in training + activity]
    document["episode_order_digests"][
        "screen:2048:causal_tool_resume"
    ] = digest(causal_order)
    history = derive_exact_request_history(training + activity)
    for request in (*training, *activity):
        request["msr_exact_history"] = history[request["request_id"]]

    ready_blocks_per_target = 10
    first_ready_block_digests = []
    target_records = []
    for index, target in enumerate(targets):
        block_digests = [
            digest(target["prompt_token_ids"][offset : offset + 16])
            for offset in range(0, ready_blocks_per_target * 16, 16)
        ]
        first_ready_block_digests.extend(block_digests)
        target_records.append(
            {
                "target_id": f"target-{index + 1}",
                "session_id": target["session_id"],
                "known_prefix_token_digest": target["known_prefix_token_digest"],
                "known_prefix_boundary_tokens": 2064,
                "source_blocks": 129,
                "separated_target_request_ids": [
                    training[position]["request_id"]
                    for position in range(index, 40, 2)
                ],
                "threshold2_probe_count": 20,
                "expected_ready_contiguous_prefix_blocks": ready_blocks_per_target,
                "expected_ready_block_token_digests": block_digests,
            }
        )
    unique_ready_lower_bound = len(set(first_ready_block_digests))
    _require(unique_ready_lower_bound > 16, "repair does not exceed live budget16")
    metadata = {
        "schema": "msr-a03-activation-repair-cohort-v2",
        "campaign_id": ACTIVATION_REPAIR_ID,
        "stage": "new_exploratory_activation_cohort",
        "parent_frozen_request_sha256": file_digest(DEFAULT_ACCEPTED_ACTIVATION_REQUESTS),
        "unused_budget1_draft_preserved": True,
        "observed_a03_cause": {
            "threshold2_counts_key_lookups_not_requests": True,
            "immediate_warm2_gpu_hits_did_not_admit_long_prefixes": True,
            "accepted_hint_events_repeated_one_128_token_session": True,
            "max_distinct_candidate_sessions": 1,
            "max_eligible_and_scored_blocks": 3,
            "observed_remaining_budget_blocks": 15,
        },
        "target_sources": target_records,
        "training_request_count": 41,
        "admission_sequence": "20 alternating probes per target, then final distinct pressure",
        "hint_publication_contract": {
            "first_manifest_request_ids": [activity[8]["request_id"], activity[9]["request_id"]],
            "first_sessions": [item["session_id"] for item in target_records],
            "same_lead_ms": 750,
            "hint_only_events": 0,
            "remaining_hints_are_unmatched_controls": 6,
            "publication_phase_after_completed_training": 41,
        },
        "configured_shared_budget_blocks": 16,
        "expected_unique_ready_key_lower_bound": unique_ready_lower_bound,
        "expected_eligible_demand_lower_bound_blocks": ready_blocks_per_target * 2,
        "prefetch_score_max_age_ms": 1_000,
        "fresh_score_strategy": (
            "preserve accepted 1000ms window; final target1,target2,pressure requests "
            "use one output token and GPU must confirm both score ages"
        ),
        "required_gpu_observations": {
            "distinct_candidate_sessions": 2,
            "each_ready_contiguous_dram_prefix_blocks_at_least": ready_blocks_per_target,
            "unique_ready_keys_exceed_observed_remaining_budget16": True,
            "each_has_fresh_causal_score": True,
            "shared_budget_constrained": True,
            "exact_output_and_consumption": True,
        },
        "benefit_measured": False,
    }
    document["msr_expanded_campaign"] = {
        "schema": "msr-expanded-request-cohort-v1",
        "campaign_id": ACTIVATION_REPAIR_ID,
        "input_cohort": "a03-preserved-inputs-admission-walk-v2",
        "source_split": "screen",
        "execution_split": "screen",
        "episode": "causal_tool_resume",
        "history_join": (
            "session+generation+known_prefix_boundary_tokens+known_prefix_token_digest"
        ),
        "request_history": history,
        "activation_repair": metadata,
        "offline_labels_never_published_to_serving": True,
    }
    validate_request_manifest(document)
    return document, metadata


def build_frontier_diagnostic_requests(
    accepted: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Four target probes with completed distinct pressure between each probe."""
    document = copy.deepcopy(dict(accepted))
    original = sorted(
        (
            request
            for request in document["requests"]
            if request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == "causal_tool_resume"
        ),
        key=lambda request: int(request["request_index"]),
    )
    _require(len(original) == 32, "frontier source requires causal_tool_resume32")
    target = copy.deepcopy(original[14])
    pressure = []
    seen = {target["prompt_token_digest"]}
    for request in document["requests"]:
        if (
            request.get("split") == "screen"
            and request.get("family") == "repobench-p"
            and 4200 <= len(request.get("prompt_token_ids", [])) <= 5000
            and request.get("prompt_token_digest") not in seen
        ):
            seen.add(request["prompt_token_digest"])
            pressure.append(copy.deepcopy(request))
    _require(len(pressure) >= 4, "frontier diagnostic needs four distinct pressure inputs")

    def training_record(
        source: Mapping[str, object], index: int, label: str
    ) -> dict[str, object]:
        record = copy.deepcopy(dict(source))
        record.update(
            {
                "request_id": (
                    f"screen-2048-causal-tool-resume-{index:02d}-"
                    f"frontier-{label}"
                ),
                "request_index": index,
                "phase": "activation_train",
                "phase_index": index,
                "phase_ordinal": 0,
                "relative_release_ms": 0,
                "depends_on_phase_completion": index - 1 if index else None,
                "wait_origin": (
                    "fresh_server_start"
                    if index == 0
                    else "all_prior_phase_requests_completed"
                ),
                "arrival_origin": "a04_frontier_sequential_probe_or_pressure",
                "pressure_origin": "two_completed_distinct_pressure_requests_per_gap",
                "revisit_origin": None,
                "fresh_server_id": f"screen-2048-{FRONTIER_DIAGNOSTIC_ID}",
                "split": "screen",
                "context_budget": 2048,
                "episode": "causal_tool_resume",
                "suffix_budget_tokens": 1,
            }
        )
        serving = copy.deepcopy(dict(record["serving_inputs"]))
        serving["session_id"] = record["session_id"]
        boundary = int(record["context_token_span"][1]) // 16 * 16
        record["known_prefix_boundary_tokens"] = boundary
        record["known_prefix_token_digest"] = digest(
            record["prompt_token_ids"][:boundary]
        )
        serving["known_prefix_boundary_tokens"] = boundary
        serving["known_prefix_token_digest"] = record[
            "known_prefix_token_digest"
        ]
        record["serving_inputs"] = serving
        return record

    training = [training_record(pressure[0], 0, "header-primer")]
    target_ids = []
    pressure_groups = []
    for cycle in range(4):
        target_record = training_record(target, len(training), f"target-cycle-{cycle + 1}")
        training.append(target_record)
        target_ids.append(target_record["request_id"])
        group = []
        for pressure_offset in range(2):
            source = pressure[(1 + cycle * 2 + pressure_offset) % len(pressure)]
            record = training_record(
                source,
                len(training),
                f"pressure-cycle-{cycle + 1}-{pressure_offset + 1}",
            )
            training.append(record)
            group.append(record["request_id"])
        pressure_groups.append(group)
    training_count = len(training)

    activity = []
    for ordinal, source in enumerate(original[16:24]):
        record = copy.deepcopy(source)
        record.update(
            {
                "request_id": (
                    f"screen-2048-causal-tool-resume-{training_count + ordinal:02d}-"
                    f"pollution-{record['family']}"
                ),
                "request_index": training_count + ordinal,
                "phase_index": training_count,
                "phase_ordinal": ordinal,
                "relative_release_ms": 125 * ordinal,
                "depends_on_phase_completion": training_count - 1,
                "fresh_server_id": f"screen-2048-{FRONTIER_DIAGNOSTIC_ID}",
            }
        )
        activity.append(record)
    observed_sessions = {str(item["session_id"]) for item in training}
    unrelated = []
    for request in document["requests"]:
        if (
            request.get("episode") == "causal_tool_resume"
            and request.get("phase") == "resume"
            and request.get("session_id") not in observed_sessions
            and len(request.get("prompt_token_ids", [])) <= 5000
        ):
            unrelated.append(request)
            observed_sessions.add(str(request["session_id"]))
    _require(len(unrelated) >= 7, "frontier diagnostic needs seven unmatched hint controls")
    resume_sources = [target, *unrelated[:7]]
    for ordinal, source in enumerate(resume_sources):
        record = copy.deepcopy(dict(source))
        record.update(
            {
                "request_id": (
                    f"screen-2048-causal-tool-resume-{training_count + 8 + ordinal:02d}-"
                    f"resume-{record['family']}"
                ),
                "request_index": training_count + 8 + ordinal,
                "phase": "resume",
                "phase_index": training_count,
                "phase_ordinal": ordinal,
                "relative_release_ms": 750,
                "hint_lead_ms": 750,
                "depends_on_phase_completion": training_count - 1,
                "wait_origin": "a04_frontier_common_750ms_wait",
                "arrival_origin": "a04_frontier_hint_release_750ms",
                "pressure_origin": "post_completed_frontier_pressure",
                "revisit_origin": "exact_repeat_after_four_probe_frontier",
                "fresh_server_id": f"screen-2048-{FRONTIER_DIAGNOSTIC_ID}",
                "split": "screen",
                "context_budget": 2048,
                "episode": "causal_tool_resume",
            }
        )
        serving = copy.deepcopy(dict(record["serving_inputs"]))
        serving["session_id"] = record["session_id"]
        record["serving_inputs"] = serving
        activity.append(record)

    document["requests"] = [
        request
        for request in document["requests"]
        if not (
            request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == "causal_tool_resume"
        )
    ] + training + activity
    document["causal_tool_resume_extension"]["hint_only_events"] = []
    sequence = [
        "header-primer",
        *[
            label
            for cycle in range(4)
            for label in (
                f"target-cycle-{cycle + 1}",
                f"pressure-cycle-{cycle + 1}-1",
                f"pressure-cycle-{cycle + 1}-2",
            )
        ],
    ]
    document["causal_tool_resume_extension"]["activation_training"] = {
        "schema": "msr-activation-training-extension-v1",
        "split": "screen",
        "context_budget": 2048,
        "training_request_count": training_count,
        "activity_phase_index": training_count,
        "activity_completion_dependency": training_count - 1,
        "target_sequence": sequence,
    }
    document["episode_order_digests"][
        "screen:2048:causal_tool_resume"
    ] = digest([item["request_id"] for item in (*training, *activity)])
    history = derive_exact_request_history(training + activity)
    for request in (*training, *activity):
        request["msr_exact_history"] = history[request["request_id"]]
    metadata = {
        "schema": "msr-a04-frontier-diagnostic-v1",
        "campaign_id": FRONTIER_DIAGNOSTIC_ID,
        "stage": "proposed_not_gpu_executed",
        "frozen_a04_negative": {
            "target1_hint_ready_blocks": 0,
            "target2_hint_ready_blocks": 2,
            "target2_ready_range_blocks": [25, 26],
            "target1_ready_range_blocks": [129, 137],
            "cause": "admission advanced behind surviving GPU prefixes, not from hinted-prefix start",
            "prefetch_consumed_blocks": 0,
            "discarded_transfer_bytes_per_prefetch_arm": 917_504,
        },
        "target": {
            "session_id": target["session_id"],
            "known_prefix_token_digest": target["known_prefix_token_digest"],
            "known_prefix_boundary_tokens": 2064,
            "probe_request_ids": target_ids,
        },
        "training_request_count": training_count,
        "pressure_request_ids_by_cycle": pressure_groups,
        "pressure_prompt_tokens_by_cycle": [
            [
                len(next(item for item in training if item["request_id"] == request_id)[
                    "prompt_token_ids"
                ])
                for request_id in group
            ]
            for group in pressure_groups
        ],
        "pressure_contract": (
            "two completed distinct long contexts after every target probe; GPU must "
            "show target local prefix <= common one-block header before next probe"
        ),
        "expected_frontier_after_cycle": [0, 0, 1, 1],
        "success_threshold_ready_contiguous_blocks": 2,
        "fail_closed_live_fields": [
            "lookup_episodes.groups.0.ordered_probes",
            "reused_tokens",
            "cpu_commit_events",
            "stored_kv_blocks",
            "prior_stored_keys",
            "prefetch_attribution.cpu_cache_snapshots",
            "wait_hint_publications.response.opportunity.local_gpu_prefix_count",
            "wait_hint_publications.response.eligible_resident_prefix_count",
        ],
        "decision": {
            "at_least_two_ready": "frontier mechanism proven; scalable matched construction uses pressure after every probe",
            "fewer_than_two_ready": "stop; report exact displacement/count/store limitation and do not enlarge",
        },
        "costs_included": "primer, four target probes, eight pressure requests and all activity",
        "benefit_measured": False,
    }
    document["msr_expanded_campaign"] = {
        "schema": "msr-expanded-request-cohort-v1",
        "campaign_id": FRONTIER_DIAGNOSTIC_ID,
        "input_cohort": "a04-single-source-four-probe-frontier-v1",
        "source_split": "screen",
        "execution_split": "screen",
        "episode": "causal_tool_resume",
        "history_join": (
            "session+generation+known_prefix_boundary_tokens+known_prefix_token_digest"
        ),
        "request_history": history,
        "frontier_diagnostic": metadata,
        "offline_labels_never_published_to_serving": True,
    }
    validate_request_manifest(document)
    return document, metadata


def build_two_source_activation_requests(
    accepted: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Scale the proven A05 pressure cycle to two exact source frontiers."""
    document = copy.deepcopy(dict(accepted))
    original = sorted(
        (
            request
            for request in document["requests"]
            if request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == "causal_tool_resume"
        ),
        key=lambda request: int(request["request_index"]),
    )
    _require(len(original) == 32, "two-source source requires causal_tool_resume32")
    targets = (copy.deepcopy(original[6]), copy.deepcopy(original[14]))
    target_digests = {item["prompt_token_digest"] for item in targets}
    target_prefix_digests = {item["known_prefix_token_digest"] for item in targets}
    pressure = []
    seen = set(target_digests)
    for request in document["requests"]:
        if (
            request.get("split") == "screen"
            and 1800 <= len(request.get("prompt_token_ids", [])) <= 5000
            and request.get("prompt_token_digest") not in seen
            and digest(
                request["prompt_token_ids"][
                    : int(request["context_token_span"][1]) // 16 * 16
                ]
            )
            not in target_prefix_digests
        ):
            seen.add(request["prompt_token_digest"])
            pressure.append(copy.deepcopy(request))
    large_pressure = [item for item in pressure if len(item["prompt_token_ids"]) >= 4200]
    _require(
        len(pressure) >= 12 and len(large_pressure) >= 4,
        "two-source activation lacks pressure diversity",
    )

    def training_record(
        source: Mapping[str, object], index: int, label: str
    ) -> dict[str, object]:
        record = copy.deepcopy(dict(source))
        record.update(
            {
                "request_id": (
                    f"screen-2048-causal-tool-resume-{index:03d}-"
                    f"two-source-{label}"
                ),
                "request_index": index,
                "phase": "activation_train",
                "phase_index": index,
                "phase_ordinal": 0,
                "relative_release_ms": 0,
                "depends_on_phase_completion": index - 1 if index else None,
                "wait_origin": (
                    "fresh_server_start"
                    if index == 0
                    else "all_prior_phase_requests_completed"
                ),
                "arrival_origin": "a05_proven_two_pressure_admission_cycle",
                "pressure_origin": "two_completed_distinct_pressure_requests_per_probe",
                "revisit_origin": None,
                "fresh_server_id": f"screen-2048-{TWO_SOURCE_ACTIVATION_ID}",
                "split": "screen",
                "context_budget": 2048,
                "episode": "causal_tool_resume",
                "suffix_budget_tokens": 1,
            }
        )
        serving = copy.deepcopy(dict(record["serving_inputs"]))
        serving["session_id"] = record["session_id"]
        boundary = int(record["context_token_span"][1]) // 16 * 16
        record["known_prefix_boundary_tokens"] = boundary
        record["known_prefix_token_digest"] = digest(
            record["prompt_token_ids"][:boundary]
        )
        serving["known_prefix_boundary_tokens"] = boundary
        serving["known_prefix_token_digest"] = record[
            "known_prefix_token_digest"
        ]
        record["serving_inputs"] = serving
        return record

    training = [training_record(large_pressure[0], 0, "header-primer")]
    target_request_ids = {0: [], 1: []}
    pressure_groups = []
    probe_counts = (18, 18)
    for round_index in range(max(probe_counts)):
        for target_index, target in enumerate(targets):
            if round_index >= probe_counts[target_index]:
                continue
            target_record = training_record(
                target,
                len(training),
                f"target-{target_index + 1}-probe-{round_index + 1}",
            )
            training.append(target_record)
            target_request_ids[target_index].append(target_record["request_id"])
            first = large_pressure[(round_index + target_index) % len(large_pressure)]
            second_candidates = [
                item
                for item in pressure
                if item["prompt_token_digest"] != first["prompt_token_digest"]
                and (
                    (len(item["prompt_token_ids"]) + 15) // 16
                    + (len(first["prompt_token_ids"]) + 15) // 16
                    > 319
                )
            ]
            _require(bool(second_candidates), "pressure pair does not exceed usable GPU")
            second = second_candidates[
                (round_index * 2 + target_index) % len(second_candidates)
            ]
            group = []
            for pressure_index, source in enumerate((first, second), start=1):
                record = training_record(
                    source,
                    len(training),
                    f"pressure-{round_index + 1}-{target_index + 1}-{pressure_index}",
                )
                training.append(record)
                group.append(record["request_id"])
            pressure_groups.append(group)
    training_count = len(training)
    _require(training_count == 109, "two-source training count")

    target_prefix_digests = {item["known_prefix_token_digest"] for item in targets}
    activity_sources = [
        item
        for item in pressure
        if digest(
            item["prompt_token_ids"][
                : int(item["context_token_span"][1]) // 16 * 16
            ]
        )
        not in target_prefix_digests
    ]
    _require(len(activity_sources) >= 8, "post-training activity touches target")
    activity = []
    for ordinal, source in enumerate(activity_sources[:8]):
        record = training_record(source, training_count + ordinal, f"pollution-{ordinal}")
        record.update(
            {
                "phase": "pollution",
                "phase_index": training_count,
                "phase_ordinal": ordinal,
                "relative_release_ms": 125 * ordinal,
                "depends_on_phase_completion": training_count - 1,
                "suffix_budget_tokens": 32,
            }
        )
        activity.append(record)
    resume_controls = [
        source for source in activity_sources if source.get("family") == "repobench-p"
    ]
    _require(len(resume_controls) >= 6, "two-source needs six unmatched resume controls")
    resume_sources = [targets[0], targets[1], *resume_controls[:6]]
    for ordinal, source in enumerate(resume_sources):
        record = training_record(
            source,
            training_count + 8 + ordinal,
            f"resume-{ordinal}",
        )
        record.update(
            {
                "session_id": (
                    source["session_id"]
                    if ordinal < 2
                    else f"a05-unobserved-control-{ordinal}"
                ),
                "phase": "resume",
                "phase_index": training_count,
                "phase_ordinal": ordinal,
                "relative_release_ms": 750,
                "hint_lead_ms": 750,
                "depends_on_phase_completion": training_count - 1,
                "wait_origin": "a05_two_source_common_750ms_wait",
                "arrival_origin": "a05_two_source_competing_hint_release",
                "pressure_origin": "post_completed_two_source_training",
                "revisit_origin": "exact_repeat_after_proven_pressure_cycles",
                "suffix_budget_tokens": 32,
            }
        )
        record["serving_inputs"]["session_id"] = record["session_id"]
        activity.append(record)

    document["requests"] = [
        request
        for request in document["requests"]
        if not (
            request.get("split") == "screen"
            and request.get("context_budget") == 2048
            and request.get("episode") == "causal_tool_resume"
        )
    ] + training + activity
    document["causal_tool_resume_extension"]["hint_only_events"] = []
    target_sequence = [item["request_id"].split("two-source-", 1)[1] for item in training]
    document["causal_tool_resume_extension"]["activation_training"] = {
        "schema": "msr-activation-training-extension-v1",
        "split": "screen",
        "context_budget": 2048,
        "training_request_count": training_count,
        "activity_phase_index": training_count,
        "activity_completion_dependency": training_count - 1,
        "target_sequence": target_sequence,
        "minimum_pending_hint_sources": 2,
    }
    document["episode_order_digests"][
        "screen:2048:causal_tool_resume"
    ] = digest([item["request_id"] for item in (*training, *activity)])
    history = derive_exact_request_history(training + activity)
    for request in (*training, *activity):
        request["msr_exact_history"] = history[request["request_id"]]

    ready_blocks_by_target = (9, 9)
    target_records = []
    unique_ready = set()
    for index, target in enumerate(targets):
        ready_blocks = ready_blocks_by_target[index]
        private_block_digests = [
            digest(target["prompt_token_ids"][offset : offset + 16])
            for offset in range(16, (ready_blocks + 1) * 16, 16)
        ]
        unique_ready.update(private_block_digests)
        target_records.append(
            {
                "target_id": f"target-{index + 1}",
                "session_id": target["session_id"],
                "known_prefix_token_digest": target["known_prefix_token_digest"],
                "known_prefix_boundary_tokens": 2064,
                "probe_request_ids": target_request_ids[index],
                "probe_count": probe_counts[index],
                "expected_ready_private_blocks": ready_blocks,
                "expected_ready_private_block_digests": private_block_digests,
            }
        )
    _require(len(unique_ready) > 16, "two-source unique eligible demand does not exceed16")
    prompt_tokens_per_arm = sum(
        len(request["prompt_token_ids"]) for request in (*training, *activity)
    )
    metadata = {
        "schema": "msr-a05-two-source-activation-proposal-v1",
        "campaign_id": TWO_SOURCE_ACTIVATION_ID,
        "stage": "proposed_not_gpu_executed",
        "a05_mechanism_evidence": {
            "request_count": 29,
            "prompt_tokens_per_arm": 94_748,
            "observed_frontier": [0, 0, 1, 1],
            "ready_contiguous_blocks": 2,
            "prefetch_consumed_bytes": 917_504,
            "prefetch_waste_bytes": 0,
            "whole_trace_token_delta_percent": 0.0,
        },
        "target_sources": target_records,
        "training_request_count": training_count,
        "pressure_request_ids_by_target_probe": pressure_groups,
        "expected_unique_eligible_blocks": len(unique_ready),
        "configured_shared_budget_blocks": 16,
        "trace_request_count_per_arm": len(training) + len(activity),
        "trace_prompt_tokens_per_arm": prompt_tokens_per_arm,
        "matched_two_arm_request_count": 2 * (len(training) + len(activity)),
        "matched_two_arm_prompt_tokens": 2 * prompt_tokens_per_arm,
        "publication_requirement": (
            "two distinct target hints must be pending before the first nomination decision"
        ),
        "required_runtime_capability": {
            "wait_hint_coalescing_min_sources": 2,
            "wait_hint_coalescing_max_ms": 25,
            "implemented": True,
            "reason": (
                "generic initial gate deduplicates exact sources and releases once at two "
                "eligible sources or bounded fallback"
            ),
        },
        "fixed_stop": {
            "success": (
                "M/R coalescing releases at two distinct eligible sources; one M decision "
                "has two fresh exact sources and deduplicated eligible demand > budget16; "
                "M semantic and R FIFO choices are recorded; exact staged keys are consumed "
                "and M/R outputs match"
            ),
            "failure": "stop and retain; do not run benefit slices or enlarge training",
            "alternative": (
                "stop now after A05 mechanism success and avoid a 458,029 prompt-token/arm "
                "proposal, 4.83x A05's 94,748-token trace"
            ),
        },
        "costs_included": "primer, 36 target probes, 72 pressure requests and 16 activity requests",
        "benefit_measured": False,
    }
    document["msr_expanded_campaign"] = {
        "schema": "msr-expanded-request-cohort-v1",
        "campaign_id": TWO_SOURCE_ACTIVATION_ID,
        "input_cohort": "a05-proven-pressure-two-source-v1",
        "source_split": "screen",
        "execution_split": "screen",
        "episode": "causal_tool_resume",
        "history_join": (
            "session+generation+known_prefix_boundary_tokens+known_prefix_token_digest"
        ),
        "request_history": history,
        "two_source_activation": metadata,
        "offline_labels_never_published_to_serving": True,
    }
    validate_request_manifest(document)
    return document, metadata


def build_semantic_value_requests(
    two_source: Mapping[str, object], *, useful_first: bool
) -> tuple[dict[str, object], dict[str, object]]:
    """Add one prior-only topical signal and counterbalanced target hints to A06."""
    document = copy.deepcopy(dict(two_source))
    causal = sorted(_causal_requests(document), key=lambda item: item["request_index"])
    _require(len(causal) == 125, "semantic-value source must be the 125-request A06 trace")
    training = [copy.deepcopy(item) for item in causal[:109]]
    activity = [copy.deepcopy(item) for item in causal[109:]]
    _require(
        [item["phase"] for item in activity[:8]] == ["pollution"] * 8
        and [item["phase"] for item in activity[8:]] == ["resume"] * 8,
        "semantic-value source activity layout",
    )
    proposal = document["msr_expanded_campaign"]["two_source_activation"]
    targets = list(proposal["target_sources"])
    _require(len(targets) == 2, "semantic-value target count")
    source_rows = []
    for target in targets:
        probes = [
            item
            for item in training
            if item["request_id"] in set(target["probe_request_ids"])
        ]
        _require(len(probes) == 18, "semantic-value target probe count")
        source_rows.append(probes[-1])
    useful_index = min(
        range(2), key=lambda index: str(source_rows[index]["source_id"])
    )
    distractor_index = 1 - useful_index
    useful = source_rows[useful_index]
    distractor = source_rows[distractor_index]

    signal = copy.deepcopy(useful)
    useful_tokens = list(useful["prompt_token_ids"])
    _require(len(useful_tokens) > 2064, "semantic-value signal source is not long")
    signal_tokens = (
        useful_tokens[:16]
        + [useful_tokens[16]]
        + useful_tokens[16:160]
        + useful_tokens[-64:]
    )
    signal.update(
        {
            "request_id": "semantic-value-prior-signal-000",
            "request_index": 109,
            "phase": "activation_train",
            "phase_index": 109,
            "phase_ordinal": 0,
            "relative_release_ms": 0,
            "depends_on_phase_completion": 108,
            "wait_origin": "all_a06_training_and_pressure_completed",
            "arrival_origin": "frozen_prior_only_topical_query_before_hints",
            "pressure_origin": "a06_completed_target_displacement",
            "revisit_origin": None,
            "session_id": "semantic-value-signal:" + digest(useful["source_id"])[:24],
            "source_id": "semantic-value-signal-" + digest(signal_tokens)[:24],
            "source_sha256": digest(
                {
                    "schema": "msr-prior-topical-signal-source-v1",
                    "derived_from_source_id": useful["source_id"],
                    "tokens": signal_tokens,
                }
            ),
            "input_sha256": digest(
                {
                    "schema": "msr-prior-topical-signal-input-v1",
                    "tokens": signal_tokens,
                }
            ),
            "content_origin": "frozen_token_shift_of_predeclared_candidate_content",
            "task_origin": "prior_query_related_to_predeclared_candidate_zero",
            "prompt_token_ids": signal_tokens,
            "prompt_token_digest": digest(signal_tokens),
            "context_token_span": [17, 161],
            "context_sha256": digest(signal_tokens[17:161]),
            "key_chain_digest": digest(
                [digest(signal_tokens[:offset]) for offset in range(16, 161, 16)]
            ),
            "known_prefix_boundary_tokens": 160,
            "known_prefix_token_digest": digest(signal_tokens[:160]),
            "suffix_budget_tokens": 1,
        }
    )
    signal["serving_inputs"] = {
        **copy.deepcopy(signal["serving_inputs"]),
        "session_id": signal["session_id"],
        "key_chain_digest": signal["key_chain_digest"],
        "known_prefix_boundary_tokens": 160,
        "known_prefix_token_digest": signal["known_prefix_token_digest"],
    }
    signal["truncation"] = {
        "strategy": "none",
        "original_token_count": 144,
        "selected_token_count": 144,
        "token_spans": [[0, 144]],
    }
    target_private_blocks = {
        digest(source["prompt_token_ids"][: offset + 16])
        for source in source_rows
        for offset in range(16, 160, 16)
    }
    signal_private_blocks = {
        digest(signal_tokens[: offset + 16])
        for offset in range(16, 2064, 16)
    }
    _require(
        not signal_private_blocks.intersection(target_private_blocks),
        "topical signal creates exact target-prefix overlap",
    )
    training.append(signal)

    pollution_offsets = [1_000 + 125 * index for index in range(8)]
    for index, item in enumerate(activity):
        item["request_index"] = 110 + index
        item["phase_index"] = 110
        item["depends_on_phase_completion"] = 109
        if item["phase"] == "pollution":
            item["relative_release_ms"] = pollution_offsets[item["phase_ordinal"]]
    resumes = activity[8:]
    useful_resume = next(
        item for item in resumes if item["session_id"] == useful["session_id"]
    )
    distractor_resume = next(
        item for item in resumes if item["session_id"] == distractor["session_id"]
    )
    neutral_source = next(
        item
        for item in resumes
        if item["session_id"] not in {useful["session_id"], distractor["session_id"]}
    )
    replacement = copy.deepcopy(neutral_source)
    replacement.update(
        {
            "request_id": distractor_resume["request_id"],
            "request_index": distractor_resume["request_index"],
            "phase_index": 110,
            "phase_ordinal": distractor_resume["phase_ordinal"],
            "depends_on_phase_completion": 109,
            "session_id": "semantic-value-unobserved-control-001",
            "arrival_origin": "matched_unobserved_control_return",
            "revisit_origin": "no_prior_exact_session_observation",
        }
    )
    replacement["serving_inputs"] = {
        **copy.deepcopy(replacement["serving_inputs"]),
        "session_id": replacement["session_id"],
    }
    activity[activity.index(distractor_resume)] = replacement
    resumes = activity[8:]
    useful_resume = next(
        item for item in resumes if item["session_id"] == useful["session_id"]
    )
    _require(
        sum(
            item["session_id"] == useful["session_id"]
            and item["known_prefix_token_digest"]
            == useful["known_prefix_token_digest"]
            for item in resumes
        )
        == 1
        and all(
            not (
                item["session_id"] == distractor["session_id"]
                or item["known_prefix_token_digest"]
                == distractor["known_prefix_token_digest"]
            )
            for item in resumes
        ),
        "semantic-value return/censored target identities",
    )
    distractor_hint_id = "semantic-value-candidate-001-hint"
    document["causal_tool_resume_extension"]["hint_only_events"] = [
        {
            "hint_id": distractor_hint_id,
            "source_request_id": distractor["request_id"],
            "lead_ms": 750,
        }
    ]
    remaining_hint_ids = [
        item["request_id"] for item in resumes if item is not useful_resume
    ]
    candidate_order = (
        [useful_resume["request_id"], distractor_hint_id]
        if useful_first
        else [distractor_hint_id, useful_resume["request_id"]]
    )
    publication_order = candidate_order + remaining_hint_ids
    activation = document["causal_tool_resume_extension"]["activation_training"]
    activation.update(
        {
            "training_request_count": 110,
            "activity_phase_index": 110,
            "activity_completion_dependency": 109,
            "target_sequence": [
                item["request_id"].split("two-source-", 1)[-1]
                for item in training
            ],
            "pollution_release_offsets_ms": pollution_offsets,
            "serialize_hint_publication": True,
            "hint_publication_order": publication_order,
        }
    )
    selected = training + activity
    document["requests"] = [
        item
        for item in document["requests"]
        if not (
            item.get("split") == "screen"
            and item.get("context_budget") == 2048
            and item.get("episode") == "causal_tool_resume"
        )
    ] + selected
    document["episode_order_digests"]["screen:2048:causal_tool_resume"] = digest(
        [item["request_id"] for item in selected]
    )
    history = derive_exact_request_history(selected)
    for item in selected:
        item["msr_exact_history"] = history[item["request_id"]]
    scenario_id = SEMANTIC_VALUE_SCENARIOS[0 if useful_first else 1]
    cost = sum(len(item["prompt_token_ids"]) for item in selected)
    metadata = {
        "schema": "msr-semantic-value-counterbalance-v1",
        "scenario_id": scenario_id,
        "stage": "cpu_frozen_gpu_unmeasured",
        "frozen_useful_target_rule": "lexicographically_smallest_source_id",
        "useful_target": {
            "candidate_id": "candidate-000",
            "source_id": useful["source_id"],
            "session_id": useful["session_id"],
            "known_prefix_token_digest": useful["known_prefix_token_digest"],
            "return_request_id": useful_resume["request_id"],
        },
        "censored_distractor": {
            "candidate_id": "candidate-001",
            "source_id": distractor["source_id"],
            "session_id": distractor["session_id"],
            "known_prefix_token_digest": distractor["known_prefix_token_digest"],
            "hint_id": distractor_hint_id,
            "return_request_id": None,
        },
        "prior_signal": {
            "request_id": signal["request_id"],
            "source_id": signal["source_id"],
            "derived_exclusively_from_source_id": useful["source_id"],
            "prompt_token_digest": signal["prompt_token_digest"],
            "private_exact_block_overlap_with_candidates": 0,
            "completed_before_hint_phase": True,
        },
        "candidate_publication_order": candidate_order,
        "all_hint_publication_order": publication_order,
        "runtime_future_labels_withheld": True,
        "expected_ready_blocks_by_candidate": [9, 9],
        "expected_unique_eligible_blocks": 18,
        "configured_shared_budget_blocks": 16,
        "trace_request_count_per_arm": len(selected),
        "trace_prompt_tokens_per_arm": cost,
        "costs_included": (
            "primer, 36 probes, 72 completed pressure requests, prior topical signal, "
            "8 delayed pollution requests and 8 return/control requests"
        ),
        "required_activation": {
            "semantic_related_candidate_rank": 1,
            "fifo_candidate_rank_matches_publication_order": True,
            "distinct_exact_candidates": 2,
            "eligible_blocks": 18,
            "budget_blocks": 16,
            "useful_exact_keys_consumed": True,
            "distractor_no_return_waste_or_censoring_accounted": True,
        },
        "fixed_stop": (
            "stop this configuration if semantic does not choose candidate-000 in both "
            "orders or if it has no token/stall/resource advantage over R across the "
            "counterbalance; add A/L only after M beats R"
        ),
        "benefit_measured": False,
    }
    document["msr_expanded_campaign"] = {
        "schema": "msr-expanded-request-cohort-v1",
        "campaign_id": scenario_id,
        "input_cohort": "a06-proven-two-source-prior-query-counterbalance-v1",
        "source_split": "screen",
        "execution_split": "screen",
        "episode": "causal_tool_resume",
        "history_join": (
            "session+generation+known_prefix_boundary_tokens+known_prefix_token_digest"
        ),
        "request_history": history,
        "semantic_value_counterbalance": metadata,
        "offline_labels_never_published_to_serving": True,
    }
    validate_request_manifest(document)
    return document, metadata


def build_remaining_slice_requests(
    two_source: Mapping[str, object],
    *,
    profile_id: str,
    working_set_source_count: int,
    pre_hint_idle_ms: int,
    load_release_offsets_ms: Sequence[int],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one bounded activated trace using two-access exact-chain admission."""
    _require(
        working_set_source_count in {8, 11}, "remaining working-set source count"
    )
    _require(
        pre_hint_idle_ms in {2_000, 5_000, 15_000},
        "remaining pre-hint idle interval",
    )
    _require(
        list(load_release_offsets_ms)
        in ([250 * index for index in range(8)], [0] * 8),
        "remaining offered-load schedule",
    )
    document = copy.deepcopy(dict(two_source))
    original_causal = _causal_requests(document)
    original_by_id = {item["request_id"]: item for item in original_causal}
    proposal = document["msr_expanded_campaign"]["two_source_activation"]
    original_targets = list(proposal["target_sources"])
    _require(len(original_targets) == 2, "remaining target count")
    target_sources = [
        copy.deepcopy(original_by_id[target["probe_request_ids"][0]])
        for target in original_targets
    ]
    candidate_prefixes = {
        item["known_prefix_token_digest"] for item in target_sources
    }
    pool = []
    seen_prefixes = set(candidate_prefixes)
    for item in sorted(
        document["requests"], key=lambda row: (str(row["source_id"]), row["request_id"])
    ):
        if (
            item.get("split") != "screen"
            or item.get("context_budget") != 2048
            or len(item.get("prompt_token_ids", [])) < 2064
        ):
            continue
        boundary = int(item["context_token_span"][1]) // 16 * 16
        if boundary < 2048:
            continue
        prefix_digest = digest(item["prompt_token_ids"][:boundary])
        if prefix_digest in seen_prefixes:
            continue
        seen_prefixes.add(prefix_digest)
        pool.append(copy.deepcopy(item))
    _require(len(pool) >= 14, "remaining slices lack distinct long sources")
    sources = target_sources + pool[: working_set_source_count - 2]

    def record(
        source: Mapping[str, object],
        index: int,
        label: str,
        *,
        session_id: str,
        phase: str = "activation_train",
        phase_index: int | None = None,
        phase_ordinal: int = 0,
        relative_release_ms: int = 0,
        dependency: int | None = None,
        suffix_budget_tokens: int = 1,
    ) -> dict[str, object]:
        item = copy.deepcopy(dict(source))
        boundary = 2064
        item.update(
            {
                "request_id": f"remaining-{index:03d}-{label}",
                "request_index": index,
                "phase": phase,
                "phase_index": index if phase_index is None else phase_index,
                "phase_ordinal": phase_ordinal,
                "relative_release_ms": relative_release_ms,
                "depends_on_phase_completion": dependency,
                "wait_origin": (
                    "fresh_server_start"
                    if dependency is None
                    else "all_prior_phase_requests_completed"
                ),
                "arrival_origin": "remaining_slices_frozen_schedule",
                "pressure_origin": "bounded_exact_chain_working_set",
                "revisit_origin": None,
                "fresh_server_id": f"screen-2048-{profile_id}",
                "session_id": session_id,
                "split": "screen",
                "context_budget": 2048,
                "episode": "causal_tool_resume",
                "generation": 0,
                "context_token_span": [24, 2072],
                "context_sha256": digest(item["prompt_token_ids"][24:2072]),
                "known_prefix_boundary_tokens": boundary,
                "known_prefix_token_digest": digest(
                    item["prompt_token_ids"][:boundary]
                ),
                "suffix_budget_tokens": suffix_budget_tokens,
            }
        )
        item["serving_inputs"] = {
            **copy.deepcopy(item["serving_inputs"]),
            "session_id": session_id,
            "generation": 0,
            "known_prefix_boundary_tokens": boundary,
            "known_prefix_token_digest": item["known_prefix_token_digest"],
        }
        item["truncation"] = {
            **copy.deepcopy(item["truncation"]),
            "selected_token_count": 2048,
        }
        return item

    source_sessions = {
        str(source["source_id"]): "remaining-source-"
        + digest([profile_id, source["source_id"]])[:24]
        for source in sources
    }
    training = []
    for source_index, source in enumerate(sources):
        training.append(
            record(
                source,
                len(training),
                f"source-{source_index}-access-1",
                session_id=source_sessions[str(source["source_id"])],
                dependency=len(training) - 1 if training else None,
            )
        )
    useful_source = min(target_sources, key=lambda item: str(item["source_id"]))
    distractor_source = next(
        item for item in target_sources if item["source_id"] != useful_source["source_id"]
    )
    second_pass = [
        source
        for source in sources
        if source["source_id"]
        not in {useful_source["source_id"], distractor_source["source_id"]}
    ] + [distractor_source, useful_source]
    second_access_ids = {}
    for source in second_pass:
        item = record(
            source,
            len(training),
            f"source-{sources.index(source)}-access-2",
            session_id=source_sessions[str(source["source_id"])],
            dependency=len(training) - 1,
        )
        training.append(item)
        second_access_ids[str(source["source_id"])] = item["request_id"]
    unused = pool[working_set_source_count - 2 :]
    for traffic_index, source in enumerate(unused[:3]):
        training.append(
            record(
                source,
                len(training),
                f"fixed-traffic-{traffic_index}",
                session_id=f"remaining-traffic-{traffic_index}",
                dependency=len(training) - 1,
            )
        )

    useful_tokens = list(useful_source["prompt_token_ids"])
    signal_tokens = (
        useful_tokens[:16]
        + [useful_tokens[16]]
        + useful_tokens[16:160]
        + useful_tokens[-64:]
    )
    signal_source = copy.deepcopy(useful_source)
    signal_source.update(
        {
            "source_id": "remaining-signal-" + digest(signal_tokens)[:24],
            "source_sha256": digest(
                {"schema": "remaining-prior-signal-v1", "tokens": signal_tokens}
            ),
            "input_sha256": digest(signal_tokens),
            "context_token_span": [17, 161],
            "context_sha256": digest(signal_tokens[17:161]),
            "prompt_token_ids": signal_tokens,
            "prompt_token_digest": digest(signal_tokens),
            "key_chain_digest": digest(
                [digest(signal_tokens[:offset]) for offset in range(16, 161, 16)]
            ),
            "truncation": {
                "strategy": "none",
                "original_token_count": 144,
                "selected_token_count": 144,
                "token_spans": [[0, 144]],
            },
        }
    )
    signal_source["serving_inputs"] = {
        **copy.deepcopy(signal_source["serving_inputs"]),
        "key_chain_digest": signal_source["key_chain_digest"],
    }
    signal = record(
        signal_source,
        len(training),
        "prior-topical-signal",
        session_id="remaining-prior-signal-session",
        dependency=len(training) - 1,
        relative_release_ms=pre_hint_idle_ms,
    )
    signal.update(
        {
            "content_origin": "frozen_token_shift_of_predeclared_candidate_content",
            "task_origin": "prior_query_related_to_candidate_zero",
            "wait_origin": "fixed_context_age_idle_then_causal_score_refresh",
            "context_token_span": [17, 161],
            "context_sha256": signal_source["context_sha256"],
            "known_prefix_boundary_tokens": 160,
            "known_prefix_token_digest": digest(signal_tokens[:160]),
            "truncation": {
                "strategy": "none",
                "original_token_count": 144,
                "selected_token_count": 144,
                "token_spans": [[0, 144]],
            },
        }
    )
    signal["serving_inputs"].update(
        {
            "known_prefix_boundary_tokens": 160,
            "known_prefix_token_digest": signal["known_prefix_token_digest"],
        }
    )
    training.append(signal)
    training_count = len(training)

    activity = []
    activity_sources = (unused[3:] + pool)[:8]
    _require(len(activity_sources) >= 8, "remaining activity source count")
    for ordinal, source in enumerate(activity_sources[:8]):
        activity.append(
            record(
                source,
                training_count + ordinal,
                f"activity-{ordinal}",
                session_id=f"remaining-activity-{ordinal}",
                phase="pollution",
                phase_index=training_count,
                phase_ordinal=ordinal,
                relative_release_ms=1_000 + 125 * ordinal,
                dependency=training_count - 1,
                suffix_budget_tokens=32,
            )
        )
    useful_resume = record(
        useful_source,
        training_count + 8,
        "candidate-000-return",
        session_id=source_sessions[str(useful_source["source_id"])],
        phase="resume",
        phase_index=training_count,
        phase_ordinal=0,
        relative_release_ms=750,
        dependency=training_count - 1,
        suffix_budget_tokens=32,
    )
    useful_resume.update(
        {
            "hint_lead_ms": 750,
            "wait_origin": "fixed_750ms_wait_after_measured_idle",
            "revisit_origin": "exact_repeat_after_chain_admission",
        }
    )
    activity.append(useful_resume)
    observed_control_sources = training[-4:-1]
    for ordinal in range(1, 8):
        source = observed_control_sources[(ordinal - 1) % len(observed_control_sources)]
        item = record(
            source,
            training_count + 8 + ordinal,
            f"control-return-{ordinal}",
            session_id=f"remaining-unobserved-return-{ordinal}",
            phase="resume",
            phase_index=training_count,
            phase_ordinal=ordinal,
            relative_release_ms=750,
            dependency=training_count - 1,
            suffix_budget_tokens=32,
        )
        item["hint_lead_ms"] = 750
        item["revisit_origin"] = "no_prior_exact_session_observation"
        item["task_origin"] = (
            "supplied_question_1"
            if item["family"] == "qasper"
            else "supplied_task_exact_repeat"
        )
        activity.append(item)
    _require(len(activity) == 16, "remaining activity count")

    load_phase_index = training_count + 1
    load = []
    load_sources = pool[-8:]
    for ordinal, source in enumerate(load_sources):
        load.append(
            record(
                source,
                training_count + 16 + ordinal,
                f"load-{ordinal}",
                session_id=f"remaining-load-{ordinal}",
                phase="load",
                phase_index=load_phase_index,
                phase_ordinal=ordinal,
                relative_release_ms=int(load_release_offsets_ms[ordinal]),
                dependency=training_count,
                suffix_budget_tokens=32,
            )
        )
    recovery = []
    for ordinal, source in enumerate(load[:4]):
        item = copy.deepcopy(source)
        item.update(
            {
                "request_id": f"remaining-{training_count + 24 + ordinal:03d}-recovery-{ordinal}",
                "request_index": training_count + 24 + ordinal,
                "phase": "recovery",
                "phase_index": training_count + 2,
                "phase_ordinal": ordinal,
                "relative_release_ms": 250 * (ordinal + 1),
                "depends_on_phase_completion": load_phase_index,
                "wait_origin": "load_phase_drained_plus_250ms_quiet",
                "arrival_origin": "post_drain_recovery_probe",
                "revisit_origin": "exact_load_request_repeat_after_drain",
            }
        )
        recovery.append(item)

    selected = training + activity + load + recovery
    document["requests"] = [
        item
        for item in document["requests"]
        if not (
            item.get("split") == "screen"
            and item.get("context_budget") == 2048
            and item.get("episode") == "causal_tool_resume"
        )
    ] + selected
    distractor_hint_id = "remaining-candidate-001-hint"
    hint_order = [
        useful_resume["request_id"],
        distractor_hint_id,
        *[item["request_id"] for item in activity[9:]],
    ]
    document["causal_tool_resume_extension"]["hint_only_events"] = [
        {
            "hint_id": distractor_hint_id,
            "source_request_id": second_access_ids[str(distractor_source["source_id"])],
            "lead_ms": 750,
        }
    ]
    document["causal_tool_resume_extension"]["activation_training"] = {
        "schema": "msr-activation-training-extension-v1",
        "split": "screen",
        "context_budget": 2048,
        "training_request_count": training_count,
        "activity_phase_index": training_count,
        "activity_completion_dependency": training_count - 1,
        "target_sequence": [item["request_id"] for item in training],
        "pollution_release_offsets_ms": [1_000 + 125 * index for index in range(8)],
        "minimum_pending_hint_sources": 2,
        "serialize_hint_publication": True,
        "hint_publication_order": hint_order,
        "score_refresh_request_id": signal["request_id"],
        "context_age_idle_before_score_refresh_ms": pre_hint_idle_ms,
        "pre_hint_idle_ms": 0,
        "serialize_within_phase": profile_id != "load-burst-recovery",
        "post_activity_load_request_count": 8,
        "post_activity_load_release_offsets_ms": list(load_release_offsets_ms),
        "post_drain_recovery_request_count": 4,
        "post_drain_recovery_release_offsets_ms": [250, 500, 750, 1_000],
    }
    document["episode_order_digests"]["screen:2048:causal_tool_resume"] = digest(
        [item["request_id"] for item in selected]
    )
    history = derive_exact_request_history(selected)
    for item in selected:
        item["msr_exact_history"] = history[item["request_id"]]
    chain_keys = {
        digest(source["prompt_token_ids"][: 16 * (position + 1)])
        for source in sources
        for position in range(128)
    }
    metadata = {
        "schema": "msr-remaining-activated-slice-v1",
        "profile_id": profile_id,
        "working_set_source_count": working_set_source_count,
        "deduplicated_constructed_chain_keys": len(chain_keys),
        "chain_admission_max_prefix_blocks": 128,
        "distinct_accesses_per_source": 2,
        "training_request_count": training_count,
        "trace_request_count_per_arm": len(selected),
        "trace_prompt_tokens_per_arm": sum(
            len(item["prompt_token_ids"]) for item in selected
        ),
        "context_age_idle_before_signal_ms": pre_hint_idle_ms,
        "score_refresh_to_hint_schedule": "signal phase completion immediately precedes hint phase",
        "serialize_within_phase": profile_id != "load-burst-recovery",
        "fixed_completed_intervening_request_ids": [
            item["request_id"] for item in training[-4:]
        ],
        "useful_target_source_id": useful_source["source_id"],
        "distractor_source_id": distractor_source["source_id"],
        "candidate_hint_order": hint_order[:2],
        "load_release_offsets_ms": list(load_release_offsets_ms),
        "post_drain_recovery_release_offsets_ms": [250, 500, 750, 1_000],
        "all_costs_included": True,
        "benefit_measured": False,
    }
    document["msr_expanded_campaign"] = {
        "schema": "msr-expanded-request-cohort-v1",
        "campaign_id": profile_id,
        "input_cohort": "remaining-chain128-activated-v1",
        "source_split": "screen",
        "execution_split": "screen",
        "episode": "causal_tool_resume",
        "history_join": (
            "session+generation+known_prefix_boundary_tokens+known_prefix_token_digest"
        ),
        "request_history": history,
        "remaining_slice": metadata,
        "offline_labels_never_published_to_serving": True,
    }
    validate_request_manifest(document)
    return document, metadata


def metric_phase_table() -> list[dict[str, object]]:
    return [
        {
            "metric": "fixed_service_target_yield",
            "raw_fields": [
                "requests.planned_release_ns",
                "requests.first_token_ns",
                "requests.success",
                "requests.terminal_complete",
            ],
            "calculation": (
                "count exact returns with (first_token_ns-planned_release_ns)/1e6 <= "
                "1000, divided by eligible exact returns; failures/timeouts violate"
            ),
            "campaigns": [variant["id"] for variant in EXPANDED_VARIANTS],
            "status": "implemented_gpu_unmeasured",
        },
        {
            "metric": "actual_prompt_execution_per_completed_request",
            "raw_fields": [
                "preservation_events.actual_prompt_tokens_locally_executed",
                "preservation_events.admission_episodes",
                "requests.terminal_complete",
            ],
            "calculation": "sum authoritative terminal actual tokens / completed requests",
            "campaigns": ["reference-mixed", "capacity-pressure", "independent-inputs"],
            "status": "implemented_gpu_unmeasured",
        },
        {
            "metric": "serving_distribution_and_tails",
            "raw_fields": ["requests.ttft_ms", "requests.total_ms", "episode_makespan_ms"],
            "calculation": "median/p90/p95/p99 and throughput by matched cell",
            "campaigns": ["reference-mixed", "middle-burst"],
            "status": "implemented_gpu_unmeasured",
        },
        {
            "metric": "semantic_prefetch_usefulness_and_waste",
            "raw_fields": [
                "wait_hint_events",
                "wait_hint_publications",
                "semantic_prefetch_session_nominations",
                "preservation_events.prefetch_attribution",
                "physical_transfer_accounting",
            ],
            "calculation": "causal nomination -> staging -> bind/install/fence -> exact consumption",
            "campaigns": ["reference-mixed", "short-horizon", "long-horizon"],
            "status": "retained_real_schema_validated_gpu_unmeasured",
        },
        {
            "metric": "useful_exact_context_capacity",
            "raw_fields": [
                "physical_occupancy_evidence",
                "preservation_events.cpu_residency_proof",
                "preservation_events.prefetch_attribution.request_usable_ns",
                "requests.first_token_ns",
                "requests.planned_release_ns",
            ],
            "calculation": (
                "unique full exact contexts consumed and first-token delivered within "
                "the fixed 1000ms release-to-first-token target at each actual DRAM budget"
            ),
            "campaigns": ["reference-mixed", "capacity-pressure"],
            "status": "implemented_gpu_unmeasured",
        },
        {
            "metric": "retention_survival_and_timely_reuse",
            "raw_fields": [
                "requests.planned_release_ns",
                "requests.send_ns",
                "requests.completion_ns",
                "wait_hint_publications.planned_resume_ns",
                "preservation_events.cpu_residency_proof",
                "msr_expanded_campaign.request_history",
            ],
            "calculation": (
                "exact-prefix consumed and timely-use yield at fixed 1000ms target by actual "
                "completion-to-send last-use gap and independently counted intervening "
                "completed traffic; hint lead is a separate field"
            ),
            "campaigns": ["short-horizon", "reference-mixed", "long-horizon"],
            "status": "implemented_gpu_unmeasured_joint_horizon",
        },
        {
            "metric": "bounded_offered_load_concurrency_and_burst_tails",
            "raw_fields": [
                "requests.planned_release_ns",
                "requests.send_ns",
                "requests.completion_ns",
                "requests.ttft_ms",
                "requests.total_ms",
            ],
            "calculation": (
                "phase16 achieved arrivals/overlap/drain plus phase17 post-drain recovery "
                "probe latency at lower staggered versus higher burst input, max_num_seqs=4"
            ),
            "campaigns": [
                "lower-offered-rate-recovery",
                "higher-offered-burst-recovery",
            ],
            "status": "implemented_gpu_unmeasured_post_drain_not_sustainable_frontier",
        },
        {
            "metric": "attributable_gpu_work_and_optional_cost",
            "raw_fields": ["prefill_cuda_events", "physical_transfer_accounting", "resource_prices"],
            "calculation": "measured work/request; cost unavailable unless explicit prices supplied",
            "campaigns": ["reference-mixed"],
            "status": "gpu_unmeasured_prices_unavailable",
        },
    ]


def build_expanded_bundle(
    *,
    base_campaign_path: Path,
    base_request_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    base_campaign = json.loads(base_campaign_path.read_text(encoding="utf-8"))
    base_requests = json.loads(base_request_path.read_text(encoding="utf-8"))
    validate_request_manifest(base_requests)
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_entries = []
    for variant in EXPANDED_VARIANTS:
        variant_dir = output_dir / str(variant["id"])
        input_dir = variant_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        for source in base_request_path.parent.iterdir():
            if source.name != base_request_path.name and source.is_file():
                destination = input_dir / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
        request_document, metadata = _expanded_request_document(base_requests, variant)
        request_path = input_dir / "request-manifest-v1.json"
        _write_json(request_path, request_document)
        campaign = build_campaign(
            base_campaign,
            request_path,
            str(Path(future_result_root) / str(variant["id"])),
        )
        profile = campaign["baseline_profiles"][0]
        profile["id"] = f"msr-expanded-{variant['id']}"
        profile["resources"]["cpu_bytes"] = int(variant["dram_blocks"]) * 458_752
        profile["workload"].update(
            {
                "campaign_id": variant["id"],
                "input_cohort": variant["input_cohort"],
                "arrival_pattern": variant["arrival_pattern"],
                "episode": variant["episode"],
            }
        )
        profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
        campaign_path = variant_dir / "campaign.json"
        _write_json(campaign_path, campaign)
        validate_campaign_manifest(campaign, campaign_path)
        plan = build_plan(campaign, campaign_path)
        plan_path = variant_dir / "plan.json"
        _write_json(plan_path, plan)
        selected_cells = [
            {
                "cell_id": cell["cell_id"],
                "arm_id": cell["arm_id"],
                "profile_id": cell["profile_id"],
                "replicate_id": cell["replicate_id"],
                "seed": cell["seed"],
                "port": cell["port"],
                "request_digest": cell["request_proof"]["selected_request_digest"],
            }
            for cell in plan["cells"]
            if cell["arm_id"] in SELECTED_ARMS
            and cell["replicate_id"] in SELECTED_REPLICATES
        ]
        campaign_entries.append(
            {
                "campaign_id": variant["id"],
                "purpose": variant["purpose"],
                "arm_roles": {
                    "arc_demand": ARC_DEMAND_ARM,
                    "lru_demand": LRU_DEMAND_ARM,
                    "arc_fifo_prefetch": NONSEMANTIC_PREFETCH_ARM,
                    "msr_fifo_prefetch": MSR_FIFO_ARM,
                    "msr_semantic_prefetch": PRODUCT_ARM,
                },
                "dimensions": metadata["dimensions"],
                "input_cohort_identity": metadata["input_cohort_identity"],
                "input_identity_sets": metadata["input_identity_sets"],
                "regime_identity": digest(
                    {
                        "campaign_id": variant["id"],
                        "dimensions": metadata["dimensions"],
                        "input_cohort_identity": metadata["input_cohort_identity"],
                    }
                ),
                "pairing_weight": "one matched trace per replicate; never request-weighted",
                "request_history_key": "msr_expanded_campaign.request_history",
                "generated": {
                    "campaign_path": str(campaign_path.resolve()),
                    "campaign_portable_path": str(
                        Path(str(variant["id"])) / "campaign.json"
                    ),
                    "campaign_sha256": file_digest(campaign_path),
                    "request_path": str(request_path.resolve()),
                    "request_portable_path": str(
                        Path(str(variant["id"])) / "inputs/request-manifest-v1.json"
                    ),
                    "request_sha256": file_digest(request_path),
                    "plan_path": str(plan_path.resolve()),
                    "plan_portable_path": str(Path(str(variant["id"])) / "plan.json"),
                    "plan_sha256": file_digest(plan_path),
                    "plan_digest": plan["plan_digest"],
                },
                "selected_cells": selected_cells,
            }
        )
    entries_by_id = {entry["campaign_id"]: entry for entry in campaign_entries}
    reference_identities = entries_by_id["reference-mixed"]["input_identity_sets"]
    independent_identities = entries_by_id["independent-inputs"]["input_identity_sets"]
    disjoint_intersections = {
        field: sorted(set(reference_identities[field]) & set(independent_identities[field]))
        for field in reference_identities
    }
    _require(
        all(not values for values in disjoint_intersections.values()),
        "independent input cohort has overlapping recorded identities",
    )
    independent_proof = {
        "reference_campaign_id": "reference-mixed",
        "independent_campaign_id": "independent-inputs",
        "fields": list(disjoint_intersections),
        "intersection_values": disjoint_intersections,
        "all_recorded_identity_intersections_empty": True,
        "working_sets_are_not_pooled": True,
    }
    capability, capability_reason = _runtime_nomination_capability()
    counterfactual = _counterfactual_evidence()
    gpu_ready = capability and bool(counterfactual["valid"])
    command_base = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'}"
    )
    primary_arms = f"{ARC_DEMAND_ARM} {LRU_DEMAND_ARM} {MSR_FIFO_ARM} {PRODUCT_ARM}"
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "stage": "expanded_cpu_validated_gpu_unmeasured",
        "benefit_measured": False,
        "provider_claim": {
            "primary_metric": (
                "100 * (baseline - MSR) / baseline using total authoritative "
                "actual_prompt_tokens_locally_executed"
            ),
            "denominator": "baseline actual prompt tokens over completed requests",
            "comparators": [ARC_DEMAND_ARM, LRU_DEMAND_ARM],
            "semantic_attribution_control": MSR_FIFO_ARM,
            "median_ttft_allowance_percent": 3.0,
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
            "benefit_measured": False,
        },
        "base_inputs": {
            "campaign_path": str(base_campaign_path.resolve()),
            "campaign_sha256": file_digest(base_campaign_path),
            "request_path": str(base_request_path.resolve()),
            "request_sha256": file_digest(base_request_path),
        },
        "campaigns": campaign_entries,
        "independent_input_cohort_proof": independent_proof,
        "regime_aggregation": {
            "group_by": [
                "campaign_id",
                "regime_identity",
                "input_cohort_identity",
                "replicate_id",
            ],
            "weighting": "paired trace effects; no pooling across input cohorts/regimes",
        },
        "metric_phase_table": metric_phase_table(),
        "retained_real_execute_cell_schema": _retained_real_schema_evidence(),
        "gpu_readiness": {
            "ready": gpu_ready,
            "reason": capability_reason,
            "counterfactual_evidence": counterfactual,
            "required_gpu_activation": {
                "minimum_simultaneous_surviving_exact_sources": 2,
                "configured_shared_budget_blocks": 16,
                "allocation_demand_must_exceed_live_budget": True,
                "missing_observations_are_unavailable_not_zero": True,
            },
        },
        "future_gpu_commands": {
            "activation": (
                command_base
                + f" --results {future_result_root}/activation"
                + " --campaigns reference-mixed --replicates rep-1 --arms "
                + primary_arms
            ),
            "followups_after_activation": [
                command_base
                + f" --results {future_result_root}/followup-{campaign_id}"
                + f" --campaigns {campaign_id} --replicates rep-1 --arms "
                + primary_arms
                for campaign_id in (
                    "capacity-pressure",
                    "short-horizon",
                    "long-horizon",
                    "middle-burst",
                    "independent-inputs",
                )
            ],
            "offered_load_and_recovery": (
                command_base
                + f" --results {future_result_root}/offered-load-recovery"
                + " --campaigns lower-offered-rate-recovery "
                + "higher-offered-burst-recovery --replicates rep-1 --arms "
                + f"{ARC_DEMAND_ARM} {LRU_DEMAND_ARM} {NONSEMANTIC_PREFETCH_ARM} "
                + f"{MSR_FIFO_ARM} {PRODUCT_ARM}"
            ),
            "practical_arc_prefetch_attribution": (
                command_base
                + f" --results {future_result_root}/arc-prefetch-attribution"
                + " --campaigns reference-mixed --replicates rep-1 --arms "
                + NONSEMANTIC_PREFETCH_ARM
            ),
            "paired_confirmation_after_positive": (
                command_base
                + f" --results {future_result_root}/paired-confirmation"
                + " --campaigns reference-mixed independent-inputs "
                + "--replicates rep-2 rep-3 --arms "
                + primary_arms
            ),
        },
        "result_layout": (
            "<command-results-root>/<campaign_id>/<cell_id>/result.json; each command "
            "uses a unique permanent root and retains state.json plus execution-freeze"
        ),
        "promised_practical_p_comparison": {
            "new_cells": [
                {
                    "campaign_id": "reference-mixed",
                    "replicate_id": "rep-1",
                    "arm_id": NONSEMANTIC_PREFETCH_ARM,
                }
            ],
            "retained_control_result_root": f"{future_result_root}/activation",
            "p_only_result_root": f"{future_result_root}/arc-prefetch-attribution",
            "comparison": (
                "P versus retained same-source A/R/M activation cells; no duplicate cells"
            ),
        },
        "future_aggregation_command": (
            "uv run --no-sync python -m benchmarks.msr_combined_prefetch_attribution "
            f"--bundle {output_dir / 'bundle.json'} "
            f"--results-root {future_result_root}/activation "
            f"--results-root {future_result_root}/arc-prefetch-attribution "
            f"--results-root {future_result_root}/offered-load-recovery "
            f"--results-root {future_result_root}/followup-capacity-pressure "
            f"--results-root {future_result_root}/followup-short-horizon "
            f"--results-root {future_result_root}/followup-long-horizon "
            f"--results-root {future_result_root}/followup-middle-burst "
            f"--results-root {future_result_root}/followup-independent-inputs "
            f"--results-root {future_result_root}/paired-confirmation "
            "--output-dir output/msr-combined-prefetch-cpu-20260910/"
            "owner-attribution-expanded/measured-review1-UNIQUE"
        ),
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-activation-command.txt").write_text(
        bundle["future_gpu_commands"]["activation"] + "\n", encoding="utf-8"
    )
    return bundle


def build_activation_repair_bundle(
    *,
    base_campaign_path: Path,
    accepted_request_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    base_campaign = json.loads(base_campaign_path.read_text(encoding="utf-8"))
    accepted = json.loads(accepted_request_path.read_text(encoding="utf-8"))
    validate_request_manifest(accepted)
    request_document, repair = build_activation_repair_requests(accepted)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / ACTIVATION_REPAIR_ID / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in accepted_request_path.parent.iterdir():
        if source.name != accepted_request_path.name and source.is_file():
            shutil.copy2(source, input_dir / source.name)
    request_path = input_dir / "request-manifest-v1.json"
    _write_json(request_path, request_document)
    campaign = build_campaign(
        base_campaign,
        request_path,
        str(Path(future_result_root) / ACTIVATION_REPAIR_ID),
    )
    for arm in campaign["arms"]:
        if arm["id"] in SELECTED_ARMS:
            extra = arm["config"]["extra_config"]
            extra["store_threshold"] = 2
            extra["prefetch_score_max_age_ms"] = 1_000
            if arm["id"] in {
                PRODUCT_ARM,
                MSR_FIFO_ARM,
                NONSEMANTIC_PREFETCH_ARM,
            }:
                extra["prefetch_budget_fraction"] = 0.05
            arm["arm_config_hash"] = digest(arm["config"])
    profile = campaign["baseline_profiles"][0]
    profile["id"] = f"msr-expanded-{ACTIVATION_REPAIR_ID}"
    profile["resources"]["cpu_bytes"] = 1024 * 458_752
    profile["workload"].update(
        {
            "campaign_id": ACTIVATION_REPAIR_ID,
            "input_cohort": "a03-preserved-inputs-admission-walk-v2",
            "arrival_pattern": "first-two-distinct-competing-hints-750ms",
            "episode": "causal_tool_resume",
        }
    )
    profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
    campaign_path = output_dir / ACTIVATION_REPAIR_ID / "campaign.json"
    _write_json(campaign_path, campaign)
    validate_campaign_manifest(campaign, campaign_path)
    plan = build_plan(campaign, campaign_path)
    plan_path = output_dir / ACTIVATION_REPAIR_ID / "plan.json"
    _write_json(plan_path, plan)
    selected_cells = [
        {
            "cell_id": cell["cell_id"],
            "arm_id": cell["arm_id"],
            "profile_id": cell["profile_id"],
            "replicate_id": cell["replicate_id"],
            "seed": cell["seed"],
            "port": cell["port"],
            "request_digest": cell["request_proof"]["selected_request_digest"],
        }
        for cell in plan["cells"]
        if cell["arm_id"] in SELECTED_ARMS
        and cell["replicate_id"] in SELECTED_REPLICATES
    ]
    input_cohort_identity = digest(
        sorted(str(request["input_sha256"]) for request in _causal_requests(request_document))
    )
    dimensions = {
        "gpu_blocks": 320,
        "dram_blocks": 1024,
        "configured_shared_budget_fraction": 0.05,
        "configured_shared_budget_blocks": 16,
        "prefetch_score_max_age_ms": 1_000,
        "store_threshold": 2,
        "expected_distinct_source_count": 2,
        "expected_eligible_demand_lower_bound_blocks": 20,
        "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
    }
    entry = {
        "campaign_id": ACTIVATION_REPAIR_ID,
        "purpose": "repair observed threshold2 admission and duplicate-source activation failure",
        "arm_roles": {
            "arc_demand": ARC_DEMAND_ARM,
            "lru_demand": LRU_DEMAND_ARM,
            "arc_fifo_prefetch": NONSEMANTIC_PREFETCH_ARM,
            "msr_fifo_prefetch": MSR_FIFO_ARM,
            "msr_semantic_prefetch": PRODUCT_ARM,
        },
        "dimensions": dimensions,
        "input_cohort_identity": input_cohort_identity,
        "regime_identity": digest(
            {
                "campaign_id": ACTIVATION_REPAIR_ID,
                "dimensions": dimensions,
                "input_cohort_identity": input_cohort_identity,
            }
        ),
        "pairing_weight": "one matched trace per replicate; never request-weighted",
        "activation_repair": repair,
        "generated": {
            "campaign_path": str(campaign_path.resolve()),
            "campaign_portable_path": str(Path(ACTIVATION_REPAIR_ID) / "campaign.json"),
            "campaign_sha256": file_digest(campaign_path),
            "request_path": str(request_path.resolve()),
            "request_portable_path": str(
                Path(ACTIVATION_REPAIR_ID) / "inputs/request-manifest-v1.json"
            ),
            "request_sha256": file_digest(request_path),
            "plan_path": str(plan_path.resolve()),
            "plan_portable_path": str(Path(ACTIVATION_REPAIR_ID) / "plan.json"),
            "plan_sha256": file_digest(plan_path),
            "plan_digest": plan["plan_digest"],
        },
        "selected_cells": selected_cells,
    }
    command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'} "
        f"--results {future_result_root}/activation-repair "
        f"--campaigns {ACTIVATION_REPAIR_ID} --replicates rep-1 --arms "
        f"{ARC_DEMAND_ARM} {LRU_DEMAND_ARM} {MSR_FIFO_ARM} {PRODUCT_ARM}"
    )
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "stage": "a03_evidence_directed_activation_repair_gpu_unmeasured",
        "benefit_measured": False,
        "provider_claim": {
            "primary_metric": (
                "100 * (baseline - MSR) / baseline using total authoritative "
                "actual_prompt_tokens_locally_executed"
            ),
            "denominator": "baseline actual prompt tokens over completed requests",
            "comparators": [ARC_DEMAND_ARM, LRU_DEMAND_ARM],
            "semantic_attribution_control": MSR_FIFO_ARM,
            "median_ttft_allowance_percent": 3.0,
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
            "benefit_measured": False,
        },
        "regime_aggregation": {
            "group_by": [
                "campaign_id",
                "regime_identity",
                "input_cohort_identity",
                "replicate_id",
            ],
            "weighting": "paired trace effects; this activation cohort is not pooled",
        },
        "parent_frozen_evidence": {
            "a03_root": str(
                REPO_ROOT
                / "output/msr-combined-prefetch-gpu-20260910/execution-owner/"
                "activation-attempt-03/execution"
            ),
            "accepted_request_path": str(accepted_request_path.resolve()),
            "accepted_request_sha256": file_digest(accepted_request_path),
        },
        "campaigns": [entry],
        "gpu_readiness": {
            "ready": True,
            "required_gpu_activation": repair["required_gpu_observations"],
            "early_stop": (
                "stop after M/R if either lacks two distinct ready scored sources or "
                "eligible demand does not exceed observed remaining budget"
            ),
        },
        "future_gpu_commands": {"activation_repair": command},
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-command.txt").write_text(command + "\n", encoding="utf-8")
    return bundle


def build_frontier_diagnostic_bundle(
    *,
    base_campaign_path: Path,
    accepted_request_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    base_campaign = json.loads(base_campaign_path.read_text(encoding="utf-8"))
    accepted = json.loads(accepted_request_path.read_text(encoding="utf-8"))
    validate_request_manifest(accepted)
    request_document, diagnostic = build_frontier_diagnostic_requests(accepted)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / FRONTIER_DIAGNOSTIC_ID / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in accepted_request_path.parent.iterdir():
        if source.name != accepted_request_path.name and source.is_file():
            shutil.copy2(source, input_dir / source.name)
    request_path = input_dir / "request-manifest-v1.json"
    _write_json(request_path, request_document)
    campaign = build_campaign(
        base_campaign,
        request_path,
        str(Path(future_result_root) / FRONTIER_DIAGNOSTIC_ID),
    )
    for arm in campaign["arms"]:
        if arm["id"] in SELECTED_ARMS:
            extra = arm["config"]["extra_config"]
            extra["store_threshold"] = 2
            extra["prefetch_score_max_age_ms"] = 1_000
            if arm["id"] in {
                PRODUCT_ARM,
                MSR_FIFO_ARM,
                NONSEMANTIC_PREFETCH_ARM,
            }:
                extra["prefetch_budget_fraction"] = 0.05
            arm["arm_config_hash"] = digest(arm["config"])
    profile = campaign["baseline_profiles"][0]
    profile["id"] = f"msr-expanded-{FRONTIER_DIAGNOSTIC_ID}"
    profile["resources"]["cpu_bytes"] = 1024 * 458_752
    profile["workload"].update(
        {
            "campaign_id": FRONTIER_DIAGNOSTIC_ID,
            "input_cohort": "a04-single-source-four-probe-frontier-v1",
            "arrival_pattern": "four_target_probes_with_two_completed_pressures_each",
            "episode": "causal_tool_resume",
        }
    )
    profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
    campaign_path = output_dir / FRONTIER_DIAGNOSTIC_ID / "campaign.json"
    _write_json(campaign_path, campaign)
    validate_campaign_manifest(campaign, campaign_path)
    plan = build_plan(campaign, campaign_path)
    plan_path = output_dir / FRONTIER_DIAGNOSTIC_ID / "plan.json"
    _write_json(plan_path, plan)
    selected_cells = [
        {
            "cell_id": cell["cell_id"],
            "arm_id": cell["arm_id"],
            "profile_id": cell["profile_id"],
            "replicate_id": cell["replicate_id"],
            "seed": cell["seed"],
            "port": cell["port"],
            "request_digest": cell["request_proof"]["selected_request_digest"],
        }
        for cell in plan["cells"]
        if cell["arm_id"] in {ARC_DEMAND_ARM, PRODUCT_ARM}
        and cell["replicate_id"] == "rep-1"
    ]
    input_cohort_identity = digest(
        sorted(str(request["input_sha256"]) for request in _causal_requests(request_document))
    )
    dimensions = {
        "gpu_blocks": 320,
        "dram_blocks": 1024,
        "configured_shared_budget_fraction": 0.05,
        "configured_shared_budget_blocks": 16,
        "prefetch_score_max_age_ms": 1_000,
        "store_threshold": 2,
        "target_probe_count": 4,
        "pressure_requests_per_probe": 2,
        "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
    }
    entry = {
        "campaign_id": FRONTIER_DIAGNOSTIC_ID,
        "purpose": "test native threshold2 admission-frontier progression only",
        "arm_roles": {
            "exact_output_control": ARC_DEMAND_ARM,
            "msr_diagnostic": PRODUCT_ARM,
        },
        "dimensions": dimensions,
        "input_cohort_identity": input_cohort_identity,
        "regime_identity": digest(
            {
                "campaign_id": FRONTIER_DIAGNOSTIC_ID,
                "dimensions": dimensions,
                "input_cohort_identity": input_cohort_identity,
            }
        ),
        "pairing_weight": "one diagnostic trace; no benefit estimate",
        "frontier_diagnostic": diagnostic,
        "generated": {
            "campaign_path": str(campaign_path.resolve()),
            "campaign_portable_path": str(Path(FRONTIER_DIAGNOSTIC_ID) / "campaign.json"),
            "campaign_sha256": file_digest(campaign_path),
            "request_path": str(request_path.resolve()),
            "request_portable_path": str(
                Path(FRONTIER_DIAGNOSTIC_ID) / "inputs/request-manifest-v1.json"
            ),
            "request_sha256": file_digest(request_path),
            "plan_path": str(plan_path.resolve()),
            "plan_portable_path": str(Path(FRONTIER_DIAGNOSTIC_ID) / "plan.json"),
            "plan_sha256": file_digest(plan_path),
            "plan_digest": plan["plan_digest"],
        },
        "selected_cells": selected_cells,
    }
    command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'} "
        f"--results {future_result_root}/frontier-diagnostic "
        f"--campaigns {FRONTIER_DIAGNOSTIC_ID} --replicates rep-1 --arms "
        f"{ARC_DEMAND_ARM} {PRODUCT_ARM}"
    )
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "stage": "a04_frontier_diagnostic_proposed_not_gpu_executed",
        "benefit_measured": False,
        "provider_claim": {
            "diagnostic_question": (
                "do four fully displaced probes advance one long exact source from "
                "zero toward at least two contiguous threshold2 DRAM-ready keys?"
            ),
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
            "benefit_measured": False,
        },
        "parent_frozen_evidence": {
            "a04_root": str(
                REPO_ROOT
                / "output/msr-combined-prefetch-gpu-20260910/execution-owner/"
                "activation-repair-attempt-04/execution"
            ),
            "accepted_request_path": str(accepted_request_path.resolve()),
            "accepted_request_sha256": file_digest(accepted_request_path),
        },
        "campaigns": [entry],
        "gpu_readiness": {
            "ready": True,
            "fixed_stop": diagnostic["decision"],
            "required_live_fields": diagnostic["fail_closed_live_fields"],
        },
        "future_gpu_commands": {"frontier_diagnostic": command},
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-command.txt").write_text(command + "\n", encoding="utf-8")
    return bundle


def build_two_source_activation_bundle(
    *,
    base_campaign_path: Path,
    accepted_request_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    _require(
        file_digest(REPO_ROOT / "semantic_offload/connector.py")
        == "22bd0bbc81d95bd15c989df2207af553f742a7bc1aff0c6b2ef9d5b4cabad2eb",
        "two-source proposal requires reviewed coalescing connector source",
    )
    base_campaign = json.loads(base_campaign_path.read_text(encoding="utf-8"))
    accepted = json.loads(accepted_request_path.read_text(encoding="utf-8"))
    validate_request_manifest(accepted)
    request_document, proposal = build_two_source_activation_requests(accepted)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / TWO_SOURCE_ACTIVATION_ID / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in accepted_request_path.parent.iterdir():
        if source.name != accepted_request_path.name and source.is_file():
            shutil.copy2(source, input_dir / source.name)
    request_path = input_dir / "request-manifest-v1.json"
    _write_json(request_path, request_document)
    campaign = build_campaign(
        base_campaign,
        request_path,
        str(Path(future_result_root) / TWO_SOURCE_ACTIVATION_ID),
    )
    for arm in campaign["arms"]:
        if arm["id"] in SELECTED_ARMS:
            extra = arm["config"]["extra_config"]
            extra["store_threshold"] = 2
            extra["prefetch_score_max_age_ms"] = 1_000
            if arm["id"] in {
                PRODUCT_ARM,
                MSR_FIFO_ARM,
                NONSEMANTIC_PREFETCH_ARM,
            }:
                extra["prefetch_budget_fraction"] = 0.05
            if arm["id"] in {PRODUCT_ARM, MSR_FIFO_ARM}:
                extra["wait_hint_coalescing_min_sources"] = 2
                extra["wait_hint_coalescing_max_ms"] = 25
            arm["arm_config_hash"] = digest(arm["config"])
    profile = campaign["baseline_profiles"][0]
    profile["id"] = f"msr-expanded-{TWO_SOURCE_ACTIVATION_ID}"
    profile["resources"]["cpu_bytes"] = 1024 * 458_752
    profile["workload"].update(
        {
            "campaign_id": TWO_SOURCE_ACTIVATION_ID,
            "input_cohort": "a05-proven-pressure-two-source-v1",
            "arrival_pattern": "two_source_threshold_walk_then_atomic_candidate_set",
            "episode": "causal_tool_resume",
        }
    )
    profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
    campaign_path = output_dir / TWO_SOURCE_ACTIVATION_ID / "campaign.json"
    _write_json(campaign_path, campaign)
    validate_campaign_manifest(campaign, campaign_path)
    plan = build_plan(campaign, campaign_path)
    plan_path = output_dir / TWO_SOURCE_ACTIVATION_ID / "plan.json"
    _write_json(plan_path, plan)
    selected_cells = [
        {
            "cell_id": cell["cell_id"],
            "arm_id": cell["arm_id"],
            "profile_id": cell["profile_id"],
            "replicate_id": cell["replicate_id"],
            "seed": cell["seed"],
            "port": cell["port"],
            "request_digest": cell["request_proof"]["selected_request_digest"],
        }
        for cell in plan["cells"]
        if cell["arm_id"] in {MSR_FIFO_ARM, PRODUCT_ARM}
        and cell["replicate_id"] == "rep-1"
    ]
    input_cohort_identity = digest(
        sorted(str(request["input_sha256"]) for request in _causal_requests(request_document))
    )
    dimensions = {
        "gpu_blocks": 320,
        "dram_blocks": 1024,
        "store_threshold": 2,
        "configured_shared_budget_fraction": 0.05,
        "configured_shared_budget_blocks": 16,
        "prefetch_score_max_age_ms": 1_000,
        "target_probe_counts": [18, 18],
        "predicted_ready_blocks": [9, 9],
        "predicted_unique_eligible_blocks": proposal[
            "expected_unique_eligible_blocks"
        ],
        "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
    }
    entry = {
        "campaign_id": TWO_SOURCE_ACTIVATION_ID,
        "purpose": "two-source constrained semantic nomination decision fork",
        "arm_roles": {
            "msr_fifo_prefetch": MSR_FIFO_ARM,
            "msr_semantic_prefetch": PRODUCT_ARM,
        },
        "dimensions": dimensions,
        "input_cohort_identity": input_cohort_identity,
        "regime_identity": digest(
            {
                "campaign_id": TWO_SOURCE_ACTIVATION_ID,
                "dimensions": dimensions,
                "input_cohort_identity": input_cohort_identity,
            }
        ),
        "pairing_weight": "one matched diagnostic trace; no benefit estimate",
        "two_source_activation": proposal,
        "generated": {
            "campaign_path": str(campaign_path.resolve()),
            "campaign_portable_path": str(Path(TWO_SOURCE_ACTIVATION_ID) / "campaign.json"),
            "campaign_sha256": file_digest(campaign_path),
            "request_path": str(request_path.resolve()),
            "request_portable_path": str(
                Path(TWO_SOURCE_ACTIVATION_ID) / "inputs/request-manifest-v1.json"
            ),
            "request_sha256": file_digest(request_path),
            "plan_path": str(plan_path.resolve()),
            "plan_portable_path": str(Path(TWO_SOURCE_ACTIVATION_ID) / "plan.json"),
            "plan_sha256": file_digest(plan_path),
            "plan_digest": plan["plan_digest"],
        },
        "selected_cells": selected_cells,
    }
    command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'} "
        f"--results {future_result_root}/two-source-activation "
        f"--campaigns {TWO_SOURCE_ACTIVATION_ID} --replicates rep-1 --arms "
        f"{MSR_FIFO_ARM} {PRODUCT_ARM}"
    )
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "stage": "a05_two_source_activation_proposal_not_gpu_executed",
        "benefit_measured": False,
        "provider_claim": {
            "diagnostic_question": (
                "can two proven threshold2 frontiers produce constrained semantic nomination?"
            ),
            "exact_output_control": (
                "M and R share MSR retention, training, coalescing, budget and freshness; "
                "only semantic versus request_order selection differs"
            ),
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
            "benefit_measured": False,
        },
        "parent_frozen_evidence": {
            "a05_root": str(
                REPO_ROOT
                / "output/msr-combined-prefetch-gpu-20260910/execution-owner/"
                "frontier-diagnostic-attempt-05/execution"
            ),
            "accepted_request_path": str(accepted_request_path.resolve()),
            "accepted_request_sha256": file_digest(accepted_request_path),
        },
        "source_identity": {
            "git_head": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "connector_sha256": file_digest(REPO_ROOT / "semantic_offload/connector.py"),
            "workload_sha256": file_digest(Path(__file__)),
            "campaign_runner_sha256": file_digest(
                REPO_ROOT / "benchmarks/eviction_prefetch_campaign.py"
            ),
            "connector_required_sha256": (
                "22bd0bbc81d95bd15c989df2207af553f742a7bc1aff0c6b2ef9d5b4cabad2eb"
            ),
        },
        "campaigns": [entry],
        "gpu_readiness": {
            "ready": True,
            "reason": (
                "generic two-source/25ms coalescing gate is implemented; GPU remains paused"
            ),
            "fixed_stop": proposal["fixed_stop"],
        },
        "future_gpu_commands": {"two_source_activation": command},
        "alternative": proposal["fixed_stop"]["alternative"],
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-command.txt").write_text(command + "\n", encoding="utf-8")
    return bundle


def build_semantic_value_bundle(
    *,
    parent_bundle_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    """Freeze the two-order M/R semantic-value diagnostic after A06 activation."""
    parent = json.loads(parent_bundle_path.read_text(encoding="utf-8"))
    _require(parent.get("schema") == EXPANDED_SCHEMA, "semantic-value parent schema")
    _require(len(parent.get("campaigns", [])) == 1, "semantic-value parent campaign")
    _require(
        file_digest(parent_bundle_path)
        == "e24570e4bba9b86168e2d1827927737225bbb2704ae4cc4836e0df8fc6259be3",
        "semantic-value parent bundle changed",
    )
    parent_entry = parent["campaigns"][0]
    parent_campaign_path = Path(parent_entry["generated"]["campaign_path"])
    parent_request_path = Path(parent_entry["generated"]["request_path"])
    _require(
        file_digest(parent_request_path)
        == "daa5c90868cefb00a9247cac5f199f39ba3ebb257e837129ec2572aad52ccdde",
        "semantic-value parent requests changed",
    )
    parent_campaign = json.loads(parent_campaign_path.read_text(encoding="utf-8"))
    parent_requests = json.loads(parent_request_path.read_text(encoding="utf-8"))
    parent_configs = {arm["id"]: arm["config"] for arm in parent_campaign["arms"]}
    entries = []
    for useful_first, scenario_id in zip(
        (True, False), SEMANTIC_VALUE_SCENARIOS, strict=True
    ):
        request_document, scenario = build_semantic_value_requests(
            parent_requests, useful_first=useful_first
        )
        scenario_dir = output_dir / scenario_id
        input_dir = scenario_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        for source in parent_request_path.parent.iterdir():
            if source.name != parent_request_path.name and source.is_file():
                shutil.copy2(source, input_dir / source.name)
        request_path = input_dir / "request-manifest-v1.json"
        _write_json(request_path, request_document)
        campaign = copy.deepcopy(parent_campaign)
        campaign["request_manifest"] = {
            "path": "inputs/request-manifest-v1.json",
            "sha256": file_digest(request_path),
        }
        campaign["output_root"] = str(Path(future_result_root) / scenario_id)
        profile = campaign["baseline_profiles"][0]
        profile["id"] = f"msr-expanded-{scenario_id}"
        profile["workload"].update(
            {
                "campaign_id": scenario_id,
                "input_cohort": "a06-proven-two-source-prior-query-counterbalance-v1",
                "arrival_pattern": "prior_signal_then_serial_counterbalanced_hints",
                "episode": "causal_tool_resume",
            }
        )
        profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
        campaign_path = scenario_dir / "campaign.json"
        _write_json(campaign_path, campaign)
        validate_campaign_manifest(campaign, campaign_path)
        configs = {arm["id"]: arm["config"] for arm in campaign["arms"]}
        for arm_id in (MSR_FIFO_ARM, PRODUCT_ARM):
            _require(
                configs[arm_id] == parent_configs[arm_id],
                f"{scenario_id} changed {arm_id} scientific config",
            )
            extra = configs[arm_id]["extra_config"]
            _require(
                extra["store_threshold"] == 2
                and extra["prefetch_budget_fraction"] == 0.05
                and extra["prefetch_score_max_age_ms"] == 1_000
                and extra["wait_hint_coalescing_min_sources"] == 2
                and extra["wait_hint_coalescing_max_ms"] == 25,
                f"{scenario_id} runtime activation config",
            )
        plan = build_plan(campaign, campaign_path)
        plan_path = scenario_dir / "plan.json"
        _write_json(plan_path, plan)
        selected_cells = [
            {
                "cell_id": cell["cell_id"],
                "arm_role": (
                    "semantic" if cell["arm_id"] == PRODUCT_ARM else "fifo"
                ),
                "arm_id": cell["arm_id"],
                "profile_id": cell["profile_id"],
                "replicate_id": cell["replicate_id"],
                "seed": cell["seed"],
                "port": cell["port"],
                "request_digest": cell["request_proof"]["selected_request_digest"],
                "request_count": cell["request_proof"]["count"],
            }
            for cell in plan["cells"]
            if cell["arm_id"] in {MSR_FIFO_ARM, PRODUCT_ARM}
            and cell["replicate_id"] == "rep-1"
        ]
        _require(
            len(selected_cells) == 2
            and {cell["request_count"] for cell in selected_cells} == {126},
            f"{scenario_id} M/R selection",
        )
        dimensions = {
            "gpu_blocks": 320,
            "dram_blocks": 1024,
            "store_threshold": 2,
            "configured_shared_budget_fraction": 0.05,
            "configured_shared_budget_blocks": 16,
            "prefetch_score_max_age_ms": 1_000,
            "wait_hint_coalescing_min_sources": 2,
            "wait_hint_coalescing_max_ms": 25,
            "expected_ready_blocks": [9, 9],
            "expected_unique_eligible_blocks": 18,
            "publication_order_scenario": scenario_id,
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
        }
        input_cohort_identity = digest(
            [item["prompt_token_digest"] for item in _causal_requests(request_document)]
        )
        entries.append(
            {
                "campaign_id": scenario_id,
                "purpose": "counterbalanced prior-query-conditioned semantic value",
                "arm_roles": {
                    "msr_fifo_prefetch": MSR_FIFO_ARM,
                    "msr_semantic_prefetch": PRODUCT_ARM,
                },
                "dimensions": dimensions,
                "input_cohort_identity": input_cohort_identity,
                "regime_identity": digest(
                    {
                        "input_cohort_identity": input_cohort_identity,
                        "scenario_id": scenario_id,
                        "dimensions": dimensions,
                    }
                ),
                "pairing_weight": "one matched M/R trace; counterbalance summarized equally",
                "semantic_value_counterbalance": scenario,
                "generated": {
                    "campaign_path": str(campaign_path.resolve()),
                    "campaign_portable_path": str(Path(scenario_id) / "campaign.json"),
                    "campaign_sha256": file_digest(campaign_path),
                    "request_path": str(request_path.resolve()),
                    "request_portable_path": str(
                        Path(scenario_id) / "inputs/request-manifest-v1.json"
                    ),
                    "request_sha256": file_digest(request_path),
                    "plan_path": str(plan_path.resolve()),
                    "plan_portable_path": str(Path(scenario_id) / "plan.json"),
                    "plan_sha256": file_digest(plan_path),
                    "plan_digest": plan["plan_digest"],
                },
                "selected_cells": selected_cells,
            }
        )
    _require(
        entries[0]["input_cohort_identity"] == entries[1]["input_cohort_identity"],
        "counterbalance changed external request content/order",
    )
    command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'} "
        f"--results {future_result_root} --campaigns "
        + " ".join(SEMANTIC_VALUE_SCENARIOS)
        + f" --replicates rep-1 --arms {MSR_FIFO_ARM} {PRODUCT_ARM}"
    )
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "bundle_id": "semantic-value-counterbalance-a06-v1",
        "stage": "a06_semantic_value_counterbalance_cpu_frozen_gpu_unmeasured",
        "benefit_measured": False,
        "provider_claim": {
            "diagnostic_question": (
                "does causal semantic selection preserve the one returning related exact "
                "source across FIFO-order reversal and reduce whole-trace prompt work, "
                "exposed stall, or resource waste versus matched MSR+FIFO?"
            ),
            "primary_metric": (
                "M-R total authoritative actual_prompt_tokens_locally_executed per "
                "completed request, summarized equally across both publication orders"
            ),
            "serving_tradeoff": "M median TTFT no more than 3% above R",
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
            "benefit_measured": False,
        },
        "parent_frozen_evidence": {
            "a06_result_root": str(
                REPO_ROOT
                / "output/msr-combined-prefetch-gpu-20260910/execution-owner/"
                "two-source-activation-attempt-06/execution/retained-two-source-results/"
                "two-source-activation"
            ),
            "parent_bundle_path": str(parent_bundle_path.resolve()),
            "parent_bundle_sha256": file_digest(parent_bundle_path),
            "parent_request_sha256": file_digest(parent_request_path),
            "target_selection_uses_a06_scores": False,
        },
        "source_identity": {
            "git_head": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "connector_sha256": file_digest(REPO_ROOT / "semantic_offload/connector.py"),
            "campaign_runner_sha256": file_digest(
                REPO_ROOT / "benchmarks/eviction_prefetch_campaign.py"
            ),
            "workload_sha256": file_digest(Path(__file__)),
        },
        "campaigns": entries,
        "gpu_readiness": {
            "ready": True,
            "reason": (
                "CPU construction proves matched counterbalance and deployable prior-only "
                "publication inputs; GPU benefit remains unmeasured"
            ),
            "required_activation": entries[0]["semantic_value_counterbalance"][
                "required_activation"
            ],
            "fixed_stop": entries[0]["semantic_value_counterbalance"]["fixed_stop"],
        },
        "future_gpu_commands": {"semantic_value_counterbalance": command},
        "future_analysis_command": (
            "uv run --no-sync python -m benchmarks.msr_combined_prefetch_attribution "
            f"--bundle {output_dir / 'bundle.json'} --results-root {future_result_root} "
            f"--output-dir {output_dir / 'measured-aggregate-UNIQUE'}"
        ),
        "conditional_next_step": (
            "Build source-coherent A/L continuations only if M beats R across the frozen "
            "counterbalance; otherwise stop this configuration"
        ),
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-command.txt").write_text(command + "\n", encoding="utf-8")
    return bundle


def build_remaining_slices_schema(
    *, output_dir: Path, future_result_root: str
) -> dict[str, object]:
    """Freeze unresolved all-metric rows without inventing admission/runner APIs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "gpu_blocks": 320,
        "prefetch_budget_blocks": 16,
        "prefetch_budget_fraction": 0.05,
        "store_threshold": 2,
        "prefetch_score_max_age_ms": 1_000,
        "wait_hint_coalescing_min_sources": 2,
        "wait_hint_coalescing_max_ms": 25,
        "hint_lead_ms": 750,
        "actual_last_use_gap": {
            "formula": (
                "return.eligibility_ns - latest prior exact-source completion_ns"
            ),
            "reported_from_runtime_not_schedule_label": True,
            "intervening_traffic_counted_separately": True,
        },
        "chain_admission_max_prefix_blocks": 128,
        "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
        "seed": 1103,
    }
    definitions = (
        {
            "profile_id": "resource-reference",
            "contrast_axis": "shared_reference",
            "max_server_concurrency": 4,
            "dram_blocks": 1_024,
            "working_set_source_count": 8,
            "nominal_context_working_set_blocks": 1_024,
            "context_age_idle_before_score_refresh_ms": 5_000,
            "completed_intervening_request_count": 4,
            "load_release_offsets_ms": [250 * index for index in range(8)],
            "recovery_release_offsets_ms": [250 * index for index in range(4)],
        },
        {
            "profile_id": "resource-lower-memory",
            "contrast_axis": "dram_budget_only_vs_resource_reference",
            "max_server_concurrency": 4,
            "dram_blocks": 512,
            "working_set_source_count": 8,
            "nominal_context_working_set_blocks": 1_024,
            "context_age_idle_before_score_refresh_ms": 5_000,
            "completed_intervening_request_count": 4,
            "load_release_offsets_ms": [250 * index for index in range(8)],
            "recovery_release_offsets_ms": [250 * index for index in range(4)],
        },
        {
            "profile_id": "resource-larger-workset",
            "contrast_axis": "working_set_only_vs_resource_reference",
            "max_server_concurrency": 4,
            "dram_blocks": 1_024,
            "working_set_source_count": 11,
            "nominal_context_working_set_blocks": 1_408,
            "context_age_idle_before_score_refresh_ms": 5_000,
            "completed_intervening_request_count": 4,
            "load_release_offsets_ms": [250 * index for index in range(8)],
            "recovery_release_offsets_ms": [250 * index for index in range(4)],
        },
        {
            "profile_id": "horizon-short",
            "contrast_axis": "actual_idle_gap_only_vs_resource_reference",
            "max_server_concurrency": 4,
            "dram_blocks": 1_024,
            "working_set_source_count": 8,
            "nominal_context_working_set_blocks": 1_024,
            "context_age_idle_before_score_refresh_ms": 2_000,
            "completed_intervening_request_count": 4,
            "load_release_offsets_ms": [250 * index for index in range(8)],
            "recovery_release_offsets_ms": [250 * index for index in range(4)],
        },
        {
            "profile_id": "horizon-long",
            "contrast_axis": "actual_idle_gap_only_vs_resource_reference",
            "max_server_concurrency": 4,
            "dram_blocks": 1_024,
            "working_set_source_count": 8,
            "nominal_context_working_set_blocks": 1_024,
            "context_age_idle_before_score_refresh_ms": 15_000,
            "completed_intervening_request_count": 4,
            "load_release_offsets_ms": [250 * index for index in range(8)],
            "recovery_release_offsets_ms": [250 * index for index in range(4)],
        },
        {
            "profile_id": "load-burst-recovery",
            "contrast_axis": "load_release_pattern_only_vs_resource_reference",
            "max_server_concurrency": 4,
            "dram_blocks": 1_024,
            "working_set_source_count": 8,
            "nominal_context_working_set_blocks": 1_024,
            "context_age_idle_before_score_refresh_ms": 5_000,
            "completed_intervening_request_count": 4,
            "load_release_offsets_ms": [0] * 8,
            "recovery_release_offsets_ms": [250 * index for index in range(4)],
        },
    )
    _require(
        tuple(item["profile_id"] for item in definitions) == REMAINING_SLICE_PROFILES,
        "remaining-slice profile identity drift",
    )
    arms = (ARC_DEMAND_ARM, LRU_DEMAND_ARM, MSR_FIFO_ARM, PRODUCT_ARM)
    profiles = []
    for definition in definitions:
        profile_id = str(definition["profile_id"])
        dimensions = {**copy.deepcopy(common), **copy.deepcopy(definition)}
        profiles.append(
            {
                "profile_id": profile_id,
                "purpose": definition["contrast_axis"],
                "dimensions": dimensions,
                "planned_cells": [
                    {
                        "cell_id": f"remaining-{profile_id}-rep-1-{arm_id}",
                        "arm_id": arm_id,
                        "replicate_id": "rep-1",
                        "seed": 1103,
                    }
                    for arm_id in arms
                ],
                "matched_contract": {
                    "same_exact_requests_and_order_within_profile": True,
                    "same_model_gpu_scheduler_and_generic_movement": True,
                    "policy_residency_may_evolve_independently": True,
                    "exact_output_and_key_chain_equality_required": True,
                    "failures_and_first_return_misses_counted": True,
                },
                "activation_gate": {
                    "minimum_distinct_exact_ready_sources": 2,
                    "minimum_unique_eligible_blocks": 17,
                    "actual_budget_blocks": 16,
                    "semantic_and_fifo_choices_recorded": True,
                    "exact_selected_keys_staged_and_consumed_when_returned": True,
                    "absence_under_pressure_is_a_measured_capacity_or_ttl_result": True,
                    "chain_admission": {
                        "schema": "semantic-exact-chain-admission-v1",
                        "required_per_target": {
                            "observed_accesses": 2,
                            "action": "admit",
                            "reason": "bounded_exact_chain_prefix_admitted",
                            "stored_keys": 128,
                        },
                        "all_training_and_pressure_requests_counted": True,
                    },
                },
                "metric_contract": {
                    "all_requests": [
                        "actual_prompt_tokens_locally_executed",
                        "gpu_prefill_time_ns_when_available",
                        "ttft_itl_makespan_throughput_failures",
                        "transfer_consumed_wasted_uncompleted_bytes",
                        "cpu_rss_and_explicit_price_inputs_when_available",
                    ],
                    "capacity": (
                        "unique exact contexts/tokens meeting fixed service target per actual "
                        "occupied DRAM+staging bytes"
                    ),
                    "ttl": (
                        "exact survival and timely reuse at measured completion-to-eligibility "
                        "gap with four fixed completed intervening requests"
                    ),
                    "load": (
                        "achieved throughput, queueing, drain duration, violations and four "
                        "post-drain recovery probes"
                    ),
                },
            }
        )
    bundle_path = output_dir / "bundle.json"
    command_prefix = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle-final/bundle.json'} "
        f"--results {future_result_root} --replicates rep-1 --arms "
    )
    commands = {
        "01_reference_a_m": (
            command_prefix
            + f"{ARC_DEMAND_ARM} {PRODUCT_ARM} --campaigns resource-reference"
        ),
        "02_reference_l_r_after_valid_a_m": (
            command_prefix
            + f"{LRU_DEMAND_ARM} {MSR_FIFO_ARM} --campaigns resource-reference"
        ),
        "03_lower_memory_if_reference_interpretable": (
            command_prefix
            + " ".join(arms)
            + " --campaigns resource-lower-memory"
        ),
        "04_larger_workset_if_capacity_can_change": (
            command_prefix
            + " ".join(arms)
            + " --campaigns resource-larger-workset"
        ),
        "05_horizons_if_survival_observable": (
            command_prefix
            + " ".join(arms)
            + " --campaigns horizon-short horizon-long"
        ),
        "06_burst_if_load_category_open": (
            command_prefix
            + " ".join(arms)
            + " --campaigns load-burst-recovery"
        ),
    }
    bundle = {
        "schema": REMAINING_SLICES_SCHEMA,
        "stage": "schema_frozen_integrated_apis_not_materialized",
        "benefit_measured": False,
        "provider_claims": {
            "sampled_resource_at_equal_service": (
                "useful exact contexts/tokens meeting the fixed 1000ms target at two actual "
                "DRAM budgets, with a separate working-set-only contrast"
            ),
            "controlled_short_horizon_ttl": (
                "survival and timely exact reuse after 2s/5s/15s pre-hint idle controls, "
                "grouped by measured completion-to-eligibility gaps with four fixed completed "
                "intervening requests"
            ),
            "bounded_load_and_recovery": (
                "lower-rate versus simultaneous eight-request load plus drain and four "
                "post-drain recovery observations"
            ),
        },
        "common_contract": common,
        "profiles": profiles,
        "dependencies": {
            "chain_admission": {
                "status": "integrated_manager_spec_connector_event_drain",
                "config_field": "chain_admission_max_prefix_blocks",
                "config_value": 128,
                "default_value": 0,
                "applies_identically_to": list(arms),
                "manager_source_sha256": (
                    "9964043ec074c939383e72f81a6ad6d5051ac9d61d0a0692e3063edc042f6fee"
                ),
                "spec_source_sha256": (
                    "c3c00ec67681b86e65a0ec4734bc5316f23741eb2899998a05d187e643a01e03"
                ),
                "record_schema": "semantic-exact-chain-admission-v1",
                "required_event_fields": [
                    "request_id",
                    "run_id",
                    "session_id",
                    "generation",
                    "key_chain_digest",
                    "namespace",
                    "observed_accesses",
                    "qualified_keys",
                    "admitted_keys",
                    "action",
                    "reason",
                    "overhead_wall_time_ns",
                    "overhead_cpu_time_ns",
                ],
                "required_accounting": [
                    "observations",
                    "matched_observations",
                    "qualified_chains",
                    "qualified_keys",
                    "admitted_keys",
                    "stored_keys",
                    "failed_store_keys",
                    "duplicate_request_observations",
                    "identity_rejections",
                    "prefix_rejections",
                    "tracker_evictions",
                    "overhead_wall_time_ns",
                    "overhead_cpu_time_ns",
                ],
                "construction_effect": (
                    "first exact miss plus second distinct exact-chain access may admit 128 "
                    "contiguous keys per target; no one-key-per-two-probes assumption"
                ),
                "connector_source_sha256": (
                    "38f3a1a81a2e9c2484c8f9448c932058dbf3d3e53af4b290665072368342a738"
                ),
                "reason": "opt-in path is fully bound; workload materialization remains",
            },
            "post_drain_recovery": {
                "status": "integrated_existing_runner_episode_extension",
                "owner_scope": "bounded activated post-load drain and recovery phases",
                "reason": "eight load requests drain before four recovery probes",
            },
        },
        "gpu_readiness": {
            "ready": False,
            "reason": (
                "schema-only artifact has no generated campaigns; use the materialized bundle"
            ),
        },
        "smallest_first_future_commands": commands,
        "future_analysis_command": (
            "uv run --no-sync python -m benchmarks.msr_combined_prefetch_attribution "
            f"--bundle {output_dir / 'bundle-final/bundle.json'} "
            f"--results-root {future_result_root} "
            f"--output-dir {output_dir / 'measured-aggregate-UNIQUE'}"
        ),
        "analyzer_compatibility": {
            "status": "deferred_until_materialized_expanded_bundle",
            "accepted_analyzer_sha256": file_digest(
                REPO_ROOT / "benchmarks/msr_combined_prefetch_attribution.py"
            ),
        },
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    _write_json(bundle_path, bundle)
    return bundle


def build_remaining_slices_bundle(
    *,
    parent_bundle_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    """Materialize the bounded resource, horizon, and load profiles."""
    _require(
        file_digest(REPO_ROOT / "semantic_offload/connector.py")
        == "38f3a1a81a2e9c2484c8f9448c932058dbf3d3e53af4b290665072368342a738",
        "remaining slices require chain-admission connector drain",
    )
    _require(
        file_digest(REPO_ROOT / "semantic_offload/manager.py")
        == "9964043ec074c939383e72f81a6ad6d5051ac9d61d0a0692e3063edc042f6fee"
        and file_digest(REPO_ROOT / "semantic_offload/spec.py")
        == "c3c00ec67681b86e65a0ec4734bc5316f23741eb2899998a05d187e643a01e03",
        "remaining slices require reviewed chain-admission manager/spec",
    )
    parent = json.loads(parent_bundle_path.read_text(encoding="utf-8"))
    parent_entry = parent["campaigns"][0]
    parent_campaign_path = Path(parent_entry["generated"]["campaign_path"])
    parent_request_path = Path(parent_entry["generated"]["request_path"])
    parent_campaign = json.loads(parent_campaign_path.read_text(encoding="utf-8"))
    parent_requests = json.loads(parent_request_path.read_text(encoding="utf-8"))
    schema_contract = build_remaining_slices_schema(
        output_dir=output_dir / "schema-contract",
        future_result_root=future_result_root,
    )
    entries = []
    for profile_contract in schema_contract["profiles"]:
        profile_id = str(profile_contract["profile_id"])
        dimensions = profile_contract["dimensions"]
        request_document, construction = build_remaining_slice_requests(
            parent_requests,
            profile_id=profile_id,
            working_set_source_count=int(dimensions["working_set_source_count"]),
            pre_hint_idle_ms=int(
                dimensions["context_age_idle_before_score_refresh_ms"]
            ),
            load_release_offsets_ms=dimensions["load_release_offsets_ms"],
        )
        profile_dir = output_dir / profile_id
        input_dir = profile_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        for source in parent_request_path.parent.iterdir():
            if source.name != parent_request_path.name and source.is_file():
                shutil.copy2(source, input_dir / source.name)
        request_path = input_dir / "request-manifest-v1.json"
        _write_json(request_path, request_document)
        campaign = copy.deepcopy(parent_campaign)
        campaign["request_manifest"] = {
            "path": "inputs/request-manifest-v1.json",
            "sha256": file_digest(request_path),
        }
        campaign["output_root"] = str(Path(future_result_root) / profile_id)
        for arm in campaign["arms"]:
            arm["config"]["extra_config"]["bounded_prefill_cuda_timing"] = True
            if arm["id"] in SELECTED_ARMS:
                extra = arm["config"]["extra_config"]
                extra["store_threshold"] = 2
                extra["chain_admission_max_prefix_blocks"] = 128
                if arm["id"] in {MSR_FIFO_ARM, PRODUCT_ARM}:
                    extra["wait_hint_coalescing_min_sources"] = 2
                    extra["wait_hint_coalescing_max_ms"] = 25
                    extra["prefetch_score_max_age_ms"] = 1_000
                    extra["prefetch_budget_fraction"] = 0.05
            arm["arm_config_hash"] = digest(arm["config"])
        profile = campaign["baseline_profiles"][0]
        profile["id"] = f"msr-expanded-{profile_id}"
        profile["comparator_config"] = copy.deepcopy(
            next(arm["config"] for arm in campaign["arms"] if arm["id"] == "G-gpu-only")
        )
        profile["resources"]["cpu_bytes"] = int(dimensions["dram_blocks"]) * 458_752
        profile["resources"]["max_num_seqs"] = int(
            dimensions["max_server_concurrency"]
        )
        profile["workload"].update(
            {
                "campaign_id": profile_id,
                "input_cohort": "remaining-chain128-activated-v1",
                "arrival_pattern": (
                    "chain128_two_access_then_idle_hint_load_recovery"
                ),
                "episode": "causal_tool_resume",
            }
        )
        profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
        campaign_path = profile_dir / "campaign.json"
        _write_json(campaign_path, campaign)
        validate_campaign_manifest(campaign, campaign_path)
        plan = build_plan(campaign, campaign_path)
        plan_path = profile_dir / "plan.json"
        _write_json(plan_path, plan)
        selected_cells = [
            {
                "cell_id": cell["cell_id"],
                "arm_id": cell["arm_id"],
                "profile_id": cell["profile_id"],
                "replicate_id": cell["replicate_id"],
                "seed": cell["seed"],
                "port": cell["port"],
                "request_digest": cell["request_proof"]["selected_request_digest"],
                "request_count": cell["request_proof"]["count"],
            }
            for cell in plan["cells"]
            if cell["arm_id"]
            in {ARC_DEMAND_ARM, LRU_DEMAND_ARM, MSR_FIFO_ARM, PRODUCT_ARM}
            and cell["replicate_id"] == "rep-1"
        ]
        _require(
            len(selected_cells) == 4
            and len({cell["request_digest"] for cell in selected_cells}) == 1,
            f"{profile_id} cells are not matched A/L/R/M",
        )
        actual_dimensions = {
            **copy.deepcopy(dimensions),
            "deduplicated_constructed_chain_keys": construction[
                "deduplicated_constructed_chain_keys"
            ],
            "trace_request_count_per_arm": construction["trace_request_count_per_arm"],
            "trace_prompt_tokens_per_arm": construction[
                "trace_prompt_tokens_per_arm"
            ],
            "configured_dram_bytes": int(dimensions["dram_blocks"]) * 458_752,
        }
        entries.append(
            {
                "campaign_id": profile_id,
                "purpose": profile_contract["purpose"],
                "arm_roles": {
                    "arc_demand": ARC_DEMAND_ARM,
                    "lru_demand": LRU_DEMAND_ARM,
                    "msr_fifo_prefetch": MSR_FIFO_ARM,
                    "msr_semantic_prefetch": PRODUCT_ARM,
                },
                "dimensions": actual_dimensions,
                "input_cohort_identity": digest(
                    [
                        item["prompt_token_digest"]
                        for item in _causal_requests(request_document)
                    ]
                ),
                "regime_identity": digest(actual_dimensions),
                "pairing_weight": "one matched trace per profile; profiles not request-weighted",
                "construction": construction,
                "activation_gate": profile_contract["activation_gate"],
                "metric_contract": {
                    **profile_contract["metric_contract"],
                    "gpu_prefill_work_accounting_schema": (
                        "semantic-bounded-prefill-cuda-accounting-v1"
                    ),
                    "gpu_prefill_complete_device_time_rule": (
                        "device_time_s only at 100% authoritative eligible-event coverage; "
                        "partial/absent remains null"
                    ),
                    "raw_prefill_event_schema": "semantic-bounded-prefill-cuda-v2",
                },
                "generated": {
                    "campaign_path": str(campaign_path.resolve()),
                    "campaign_portable_path": str(Path(profile_id) / "campaign.json"),
                    "campaign_sha256": file_digest(campaign_path),
                    "request_path": str(request_path.resolve()),
                    "request_portable_path": str(
                        Path(profile_id) / "inputs/request-manifest-v1.json"
                    ),
                    "request_sha256": file_digest(request_path),
                    "plan_path": str(plan_path.resolve()),
                    "plan_portable_path": str(Path(profile_id) / "plan.json"),
                    "plan_sha256": file_digest(plan_path),
                    "plan_digest": plan["plan_digest"],
                },
                "selected_cells": selected_cells,
            }
        )
    commands = copy.deepcopy(schema_contract["smallest_first_future_commands"])
    deferred_bundle_path = str(output_dir / "schema-contract/bundle-final/bundle.json")
    for key, command in commands.items():
        commands[key] = command.replace(
            deferred_bundle_path, str(output_dir / "bundle.json")
        )
    analysis_command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_attribution "
        f"--bundle {output_dir / 'bundle.json'} "
        f"--results-root {future_result_root} "
        f"--output-dir {output_dir / 'measured-aggregate-UNIQUE'}"
    )
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "bundle_id": "remaining-activated-all-metric-slices-v1",
        "stage": "cpu_materialized_gpu_unmeasured",
        "benefit_measured": False,
        "provider_claim": {
            "claims": schema_contract["provider_claims"],
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
            "benefit_measured": False,
        },
        "parent_frozen_evidence": {
            "parent_bundle_path": str(parent_bundle_path.resolve()),
            "parent_bundle_sha256": file_digest(parent_bundle_path),
            "parent_request_sha256": file_digest(parent_request_path),
        },
        "a12_fallback": {
            "excluded_attempt": (
                "all-metrics-reference-attempt-12 failed during upstream synthetic "
                "staged-write warmup before any result cell"
            ),
            "max_num_seqs": 4,
            "serialized_profiles": [
                profile_id
                for profile_id in REMAINING_SLICE_PROFILES
                if profile_id != "load-burst-recovery"
            ],
            "serialization_order": "relative_release_ms_then_request_index",
            "await_completion_before_next_send": True,
            "planned_phase_origin_and_release_offsets_preserved": True,
            "burst_profile_remains_concurrent": True,
        },
        "source_identity": {
            "git_head": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "connector_sha256": file_digest(REPO_ROOT / "semantic_offload/connector.py"),
            "manager_sha256": file_digest(REPO_ROOT / "semantic_offload/manager.py"),
            "spec_sha256": file_digest(REPO_ROOT / "semantic_offload/spec.py"),
            "campaign_runner_sha256": file_digest(
                REPO_ROOT / "benchmarks/eviction_prefetch_campaign.py"
            ),
            "workload_sha256": file_digest(Path(__file__)),
            "analyzer_sha256": file_digest(
                REPO_ROOT / "benchmarks/msr_combined_prefetch_attribution.py"
            ),
        },
        "campaigns": entries,
        "gpu_readiness": {
            "ready": True,
            "reason": (
                "chain admission, event draining, GPU timing, idle and post-drain schedule "
                "contracts are bound; max_num_seqs4 serialized fallback is CPU validated"
            ),
            "smallest_first": "01_reference_a_m",
            "profile_activation_rule": (
                "record two-source constrained activation or attributable source loss under "
                "the profile's frozen capacity/horizon pressure"
            ),
        },
        "smallest_first_future_commands": commands,
        "future_analysis_command": analysis_command,
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-smallest-command.txt").write_text(
        commands["01_reference_a_m"] + "\n", encoding="utf-8"
    )
    return bundle


def refresh_remaining_slices_provenance(
    *,
    source_bundle_path: Path,
    output_dir: Path,
    future_result_root: str,
) -> dict[str, object]:
    """Rebind source provenance without changing frozen scientific artifacts."""
    source_bundle = json.loads(source_bundle_path.read_text(encoding="utf-8"))
    _require(
        source_bundle.get("bundle_id") == "remaining-activated-all-metric-slices-v1"
        and source_bundle.get("schema") == EXPANDED_SCHEMA,
        "remaining provenance source bundle identity",
    )
    _require(
        file_digest(REPO_ROOT / "benchmarks/msr_combined_prefetch_attribution.py")
        == "23f5be9accab0225bf182b2e153b1d88a13c724cd75e25c60f0f9f94b76b467c",
        "remaining provenance requires final analyzer source",
    )
    _require(
        file_digest(REPO_ROOT / "semantic_offload/connector.py")
        == "f2549fdfcee8521b8b611a0f709dd91b1bad5f415ac2bfe235124a764db8d7c8",
        "remaining provenance requires final worker-queue connector source",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = copy.deepcopy(source_bundle)
    frozen_artifacts = []
    for entry in bundle["campaigns"]:
        generated = entry["generated"]
        profile_id = str(entry["campaign_id"])
        source_profile_dir = Path(generated["campaign_path"]).parent
        target_profile_dir = output_dir / profile_id
        _require(not target_profile_dir.exists(), "provenance target profile exists")
        shutil.copytree(source_profile_dir, target_profile_dir)
        for kind in ("campaign", "request", "plan"):
            relative = Path(generated[f"{kind}_portable_path"])
            copied = output_dir / relative
            expected = str(generated[f"{kind}_sha256"])
            _require(
                copied.is_file() and file_digest(copied) == expected,
                f"{profile_id} {kind} changed during provenance refresh",
            )
            generated[f"{kind}_path"] = str(copied.resolve())
            frozen_artifacts.append(
                {
                    "campaign_id": profile_id,
                    "kind": kind,
                    "sha256": expected,
                }
            )
    new_bundle_path = str(output_dir / "bundle.json")
    first_command = next(iter(bundle["smallest_first_future_commands"].values()))
    old_bundle_path = first_command.split("--bundle ", 1)[1].split(" ", 1)[0]
    old_result_root = first_command.split("--results ", 1)[1].split(" ", 1)[0]
    for key, command in bundle["smallest_first_future_commands"].items():
        bundle["smallest_first_future_commands"][key] = command.replace(
            old_bundle_path, new_bundle_path
        ).replace(old_result_root, future_result_root)
    bundle["future_analysis_command"] = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_attribution "
        f"--bundle {new_bundle_path} --results-root {future_result_root} "
        f"--output-dir {output_dir / 'measured-aggregate-UNIQUE'}"
    )
    bundle["stage"] = "a12_v5_science_frozen_final_source_provenance_refresh"
    bundle["source_identity"] = {
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "connector_sha256": file_digest(REPO_ROOT / "semantic_offload/connector.py"),
        "manager_sha256": file_digest(REPO_ROOT / "semantic_offload/manager.py"),
        "spec_sha256": file_digest(REPO_ROOT / "semantic_offload/spec.py"),
        "query_capture_sha256": file_digest(
            REPO_ROOT / "semantic_offload/query_capture.py"
        ),
        "campaign_runner_sha256": file_digest(
            REPO_ROOT / "benchmarks/eviction_prefetch_campaign.py"
        ),
        "workload_sha256": file_digest(Path(__file__)),
        "analyzer_sha256": file_digest(
            REPO_ROOT / "benchmarks/msr_combined_prefetch_attribution.py"
        ),
        "analyzer_test_sha256": file_digest(
            REPO_ROOT / "tests/test_msr_combined_prefetch_attribution.py"
        ),
    }
    bundle["provenance_refresh"] = {
        "source_bundle_path": str(source_bundle_path.resolve()),
        "source_bundle_sha256": file_digest(source_bundle_path),
        "scientific_artifacts_byte_identical": True,
        "frozen_artifacts": frozen_artifacts,
        "allowed_changes": [
            "bundle_stage",
            "source_identity",
            "absolute_generated_paths",
            "bundle_and_result_paths_in_future_commands",
            "provenance_refresh",
        ],
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-smallest-command.txt").write_text(
        bundle["smallest_first_future_commands"]["01_reference_a_m"] + "\n",
        encoding="utf-8",
    )
    return bundle


def _independent_two_source_base(document: Mapping[str, object]) -> dict[str, object]:
    independent = copy.deepcopy(dict(document))
    candidates = []
    seen = set()
    for item in _causal_requests(independent):
        identity = (
            item["source_id"],
            item["session_id"],
            item["generation"],
            item["known_prefix_token_digest"],
        )
        if len(item["prompt_token_ids"]) < 2064 or identity in seen:
            continue
        seen.add(identity)
        candidates.append(item)
    candidates.sort(
        key=lambda item: (
            str(item["source_id"]),
            str(item["known_prefix_token_digest"]),
            str(item["request_id"]),
        )
    )
    _require(len(candidates) >= 2, "independent cohort lacks two exact long sources")
    targets = candidates[:2]
    metadata = copy.deepcopy(independent.get("msr_expanded_campaign", {}))
    metadata["two_source_activation"] = {
        "target_sources": [
            {
                "target_id": f"independent-target-{index}",
                "session_id": item["session_id"],
                "known_prefix_token_digest": item["known_prefix_token_digest"],
                "probe_request_ids": [item["request_id"]],
            }
            for index, item in enumerate(targets)
        ]
    }
    independent["msr_expanded_campaign"] = metadata
    return independent


def build_uncertainty_disjoint_supplement(
    *,
    source_bundle_path: Path,
    independent_request_path: Path,
    output_dir: Path,
    uncertainty_result_root: str,
    disjoint_result_root: str,
    retained_reference_result_roots: Sequence[str],
) -> dict[str, object]:
    """Build rep2/3 uncertainty plus one disjoint rep1 confirmation."""
    source_bundle = json.loads(source_bundle_path.read_text(encoding="utf-8"))
    source_entries = {
        entry["campaign_id"]: entry for entry in source_bundle["campaigns"]
    }
    reference = copy.deepcopy(source_entries["resource-reference"])
    source_campaign_path = Path(reference["generated"]["campaign_path"])
    source_request_path = Path(reference["generated"]["request_path"])
    source_plan_path = Path(reference["generated"]["plan_path"])
    for kind, path in (
        ("campaign", source_campaign_path),
        ("request", source_request_path),
        ("plan", source_plan_path),
    ):
        _require(
            file_digest(path) == reference["generated"][f"{kind}_sha256"],
            f"uncertainty source {kind} changed",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir = output_dir / "resource-reference"
    shutil.copytree(source_campaign_path.parent, uncertainty_dir)
    reference["generated"] = {
        **reference["generated"],
        "campaign_path": str((uncertainty_dir / "campaign.json").resolve()),
        "request_path": str(
            (uncertainty_dir / "inputs/request-manifest-v1.json").resolve()
        ),
        "plan_path": str((uncertainty_dir / "plan.json").resolve()),
    }
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    new_reference_cells = [
        {
            "cell_id": cell["cell_id"],
            "arm_id": cell["arm_id"],
            "profile_id": cell["profile_id"],
            "replicate_id": cell["replicate_id"],
            "seed": cell["seed"],
            "port": cell["port"],
            "request_digest": cell["request_proof"]["selected_request_digest"],
            "request_count": cell["request_proof"]["count"],
        }
        for cell in source_plan["cells"]
        if cell["arm_id"]
        in {ARC_DEMAND_ARM, LRU_DEMAND_ARM, MSR_FIFO_ARM, PRODUCT_ARM}
        and cell["replicate_id"] in {"rep-1", "rep-2", "rep-3"}
    ]
    _require(
        len(new_reference_cells) == 12
        and {cell["replicate_id"] for cell in new_reference_cells}
        == {"rep-1", "rep-2", "rep-3"}
        and {cell["seed"] for cell in new_reference_cells}
        == {1103, 2207, 3301},
        "uncertainty selected cells",
    )
    reference["selected_cells"] = new_reference_cells
    reference["new_execution_cells"] = new_reference_cells
    reference["pairing_weight_description"] = reference["pairing_weight"]
    reference["pairing_weight"] = 1.0

    independent_source = json.loads(
        independent_request_path.read_text(encoding="utf-8")
    )
    independent_base = _independent_two_source_base(independent_source)
    independent_document, independent_construction = build_remaining_slice_requests(
        independent_base,
        profile_id="resource-reference-inputs",
        working_set_source_count=8,
        pre_hint_idle_ms=5_000,
        load_release_offsets_ms=[250 * index for index in range(8)],
    )
    independent_dir = output_dir / "resource-reference-independent"
    input_dir = independent_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in independent_request_path.parent.iterdir():
        if source.name != independent_request_path.name and source.is_file():
            shutil.copy2(source, input_dir / source.name)
    independent_path = input_dir / "request-manifest-v1.json"
    _write_json(independent_path, independent_document)
    campaign = json.loads(source_campaign_path.read_text(encoding="utf-8"))
    campaign["request_manifest"] = {
        "path": "inputs/request-manifest-v1.json",
        "sha256": file_digest(independent_path),
    }
    campaign["output_root"] = str(
        Path(disjoint_result_root) / "resource-reference-independent"
    )
    profile = campaign["baseline_profiles"][0]
    profile["id"] = "msr-expanded-resource-reference-independent"
    profile["workload"].update(
        {
            "campaign_id": "resource-reference-independent",
            "input_cohort": "accepted-independent-inputs-chain128-v1",
        }
    )
    profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
    campaign_path = independent_dir / "campaign.json"
    _write_json(campaign_path, campaign)
    validate_campaign_manifest(campaign, campaign_path)
    plan = build_plan(campaign, campaign_path)
    plan_path = independent_dir / "plan.json"
    _write_json(plan_path, plan)
    independent_cells = [
        {
            "cell_id": cell["cell_id"],
            "arm_id": cell["arm_id"],
            "profile_id": cell["profile_id"],
            "replicate_id": cell["replicate_id"],
            "seed": cell["seed"],
            "port": cell["port"],
            "request_digest": cell["request_proof"]["selected_request_digest"],
            "request_count": cell["request_proof"]["count"],
        }
        for cell in plan["cells"]
        if cell["arm_id"]
        in {ARC_DEMAND_ARM, LRU_DEMAND_ARM, MSR_FIFO_ARM, PRODUCT_ARM}
        and cell["replicate_id"] == "rep-1"
    ]
    _require(len(independent_cells) == 4, "independent selected cells")

    reference_document = json.loads(source_request_path.read_text(encoding="utf-8"))
    reference_rows = _causal_requests(reference_document)
    independent_rows = _causal_requests(independent_document)
    identity_fields = (
        "source_id",
        "source_sha256",
        "input_sha256",
        "context_sha256",
        "known_prefix_token_digest",
        "prompt_token_digest",
    )
    intersections = {}
    for field in identity_fields:
        left = {str(item[field]) for item in reference_rows}
        right = {str(item[field]) for item in independent_rows}
        intersections[field] = sorted(left.intersection(right))
    _require(
        all(not values for values in intersections.values()),
        "independent cohort identity overlap",
    )
    independent_dimensions = copy.deepcopy(reference["dimensions"])
    independent_dimensions.update(
        {
            "trace_request_count_per_arm": independent_construction[
                "trace_request_count_per_arm"
            ],
            "trace_prompt_tokens_per_arm": independent_construction[
                "trace_prompt_tokens_per_arm"
            ],
            "deduplicated_constructed_chain_keys": independent_construction[
                "deduplicated_constructed_chain_keys"
            ],
        }
    )
    independent_entry = {
        "campaign_id": "resource-reference-independent",
        "purpose": "genuinely disjoint exact-input confirmation",
        "arm_roles": copy.deepcopy(reference["arm_roles"]),
        "dimensions": independent_dimensions,
        "input_cohort_identity": digest(
            [item["prompt_token_digest"] for item in independent_rows]
        ),
        "regime_identity": digest(independent_dimensions),
        "pairing_weight": 1.0,
        "pairing_weight_description": (
            "one disjoint matched trace; never pooled as a repeat"
        ),
        "construction": independent_construction,
        "activation_gate": copy.deepcopy(reference["activation_gate"]),
        "metric_contract": copy.deepcopy(reference["metric_contract"]),
        "generated": {
            "campaign_path": str(campaign_path.resolve()),
            "campaign_portable_path": (
                "resource-reference-independent/campaign.json"
            ),
            "campaign_sha256": file_digest(campaign_path),
            "request_path": str(independent_path.resolve()),
            "request_portable_path": (
                "resource-reference-independent/inputs/request-manifest-v1.json"
            ),
            "request_sha256": file_digest(independent_path),
            "plan_path": str(plan_path.resolve()),
            "plan_portable_path": "resource-reference-independent/plan.json",
            "plan_sha256": file_digest(plan_path),
            "plan_digest": plan["plan_digest"],
        },
        "selected_cells": independent_cells,
    }
    uncertainty_command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'} "
        f"--results {uncertainty_result_root} --campaigns resource-reference "
        "--replicates rep-1 rep-2 rep-3 --arms "
        f"{ARC_DEMAND_ARM} {LRU_DEMAND_ARM} {MSR_FIFO_ARM} {PRODUCT_ARM}"
    )
    disjoint_command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'} "
        f"--results {disjoint_result_root} "
        "--campaigns resource-reference-independent --replicates rep-1 --arms "
        f"{ARC_DEMAND_ARM} {LRU_DEMAND_ARM} {MSR_FIFO_ARM} {PRODUCT_ARM}"
    )
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "bundle_id": "remaining-uncertainty-disjoint-supplement-v1",
        "stage": "cpu_frozen_gpu_unmeasured",
        "benefit_measured": False,
        "provider_claim": {
            "uncertainty": (
                "three predetermined paired reference repetitions on one fresh source"
            ),
            "disjoint_confirmation": (
                "one separately grouped exact-input cohort with zero six-field identity overlap"
            ),
            "fixed_service_target": copy.deepcopy(FIXED_SERVICE_TARGET),
            "benefit_measured": False,
        },
        "parent_frozen_evidence": {
            "v5c_bundle_path": str(source_bundle_path.resolve()),
            "v5c_bundle_sha256": file_digest(source_bundle_path),
            "historical_reference_rep1_excluded_from_pooling": True,
            "historical_reference_rep1_result_roots": list(
                retained_reference_result_roots
            ),
            "exclusion_reason": "deployed revision/tree differs from fresh supplement source",
        },
        "source_identity": {
            "git_head": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "connector_sha256": file_digest(REPO_ROOT / "semantic_offload/connector.py"),
            "manager_sha256": file_digest(REPO_ROOT / "semantic_offload/manager.py"),
            "spec_sha256": file_digest(REPO_ROOT / "semantic_offload/spec.py"),
            "campaign_runner_sha256": file_digest(
                REPO_ROOT / "benchmarks/eviction_prefetch_campaign.py"
            ),
            "workload_sha256": file_digest(Path(__file__)),
            "analyzer_sha256": file_digest(
                REPO_ROOT / "benchmarks/msr_combined_prefetch_attribution.py"
            ),
        },
        "campaigns": [reference, independent_entry],
        "disjoint_identity_proof": {
            "fields": list(identity_fields),
            "intersections": intersections,
            "all_empty": True,
            "reference_request_sha256": file_digest(source_request_path),
            "independent_source_sha256": file_digest(independent_request_path),
            "independent_generated_sha256": file_digest(independent_path),
        },
        "independent_input_cohort_proof": {
            "reference_campaign_id": "resource-reference",
            "independent_campaign_id": "resource-reference-independent",
            "fields": list(identity_fields),
            "intersection_values": intersections,
            "all_recorded_identity_intersections_empty": True,
            "working_sets_are_not_pooled": True,
        },
        "gpu_readiness": {
            "ready": True,
            "reason": (
                "fresh rep1/2/3 reuse exact v5c science; disjoint rep1 passes construction, leakage, "
                "identity-intersection and activation contracts"
            ),
            "fixed_stop": (
                "execute all 16 fresh cells for uncertainty/provenance; do not expand the grid"
            ),
            "new_execution_cell_count": 16,
            "retained_rep1_cell_count": 0,
        },
        "future_gpu_commands": {
            "reference_rep1_rep2_rep3": uncertainty_command,
            "disjoint_rep1": disjoint_command,
        },
        "future_analysis_command": (
            "uv run --no-sync python -m benchmarks.msr_combined_prefetch_attribution "
            f"--bundle {output_dir / 'bundle.json'} "
            + f"--results-root {uncertainty_result_root} "
            + f"--results-root {disjoint_result_root} "
            + f"--output-dir {output_dir / 'measured-aggregate-UNIQUE'}"
        ),
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    _write_json(output_dir / "bundle.json", bundle)
    (output_dir / "future-gpu-commands.txt").write_text(
        uncertainty_command + "\n" + disjoint_command + "\n", encoding="utf-8"
    )
    return bundle


def build_demand_control_continuation_bundle(
    *,
    parent_bundle_path: Path,
    output_dir: Path,
    future_result_root: str,
    retained_mr_result_root: str,
) -> dict[str, object]:
    parent = json.loads(parent_bundle_path.read_text(encoding="utf-8"))
    _require(parent.get("schema") == EXPANDED_SCHEMA, "parent bundle schema")
    _require(len(parent.get("campaigns", [])) == 1, "parent campaign count")
    parent_readiness = parent.get("gpu_readiness", {})
    _require(
        parent_readiness.get("ready") is True,
        "parent two-source bundle is not GPU ready",
    )
    parent_source = parent.get("source_identity", {})
    required_connector_sha = parent_source.get("connector_required_sha256")
    _require(
        isinstance(required_connector_sha, str)
        and parent_source.get("connector_sha256") == required_connector_sha
        and file_digest(REPO_ROOT / "semantic_offload/connector.py")
        == required_connector_sha,
        "parent/current connector source binding changed",
    )
    _require(
        parent_source.get("campaign_runner_sha256")
        == file_digest(REPO_ROOT / "benchmarks/eviction_prefetch_campaign.py"),
        "parent/current campaign runner source binding changed",
    )
    parent_entry = parent["campaigns"][0]
    _require(
        parent_entry.get("campaign_id") == TWO_SOURCE_ACTIVATION_ID,
        "parent two-source campaign identity",
    )
    parent_generated = parent_entry["generated"]
    parent_campaign_path = Path(parent_generated["campaign_path"])
    parent_request_path = Path(parent_generated["request_path"])
    parent_plan_path = Path(parent_generated["plan_path"])
    for kind, path in (
        ("campaign", parent_campaign_path),
        ("request", parent_request_path),
        ("plan", parent_plan_path),
    ):
        _require(path.is_file(), f"parent {kind} artifact missing")
        _require(
            file_digest(path) == parent_generated[f"{kind}_sha256"],
            f"parent {kind} artifact source binding changed",
        )
    _require(
        file_digest(parent_request_path)
        == "daa5c90868cefb00a9247cac5f199f39ba3ebb257e837129ec2572aad52ccdde",
        "continuation must reuse exact completed M/R requests",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / TWO_SOURCE_ACTIVATION_ID / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in parent_request_path.parent.iterdir():
        if source.is_file():
            shutil.copy2(source, input_dir / source.name)
    request_path = input_dir / parent_request_path.name
    _require(
        file_digest(request_path) == file_digest(parent_request_path),
        "continuation request copy changed bytes",
    )
    campaign = json.loads(parent_campaign_path.read_text(encoding="utf-8"))
    campaign["request_manifest"] = {
        "path": "inputs/request-manifest-v1.json",
        "sha256": file_digest(request_path),
    }
    campaign["output_root"] = str(
        Path(future_result_root) / TWO_SOURCE_ACTIVATION_ID
    )
    campaign_path = output_dir / TWO_SOURCE_ACTIVATION_ID / "campaign.json"
    _write_json(campaign_path, campaign)
    validate_campaign_manifest(campaign, campaign_path)
    plan = build_plan(campaign, campaign_path)
    plan_path = output_dir / TWO_SOURCE_ACTIVATION_ID / "plan.json"
    _write_json(plan_path, plan)
    cells_by_arm = {
        cell["arm_id"]: {
            "cell_id": cell["cell_id"],
            "arm_id": cell["arm_id"],
            "profile_id": cell["profile_id"],
            "replicate_id": cell["replicate_id"],
            "seed": cell["seed"],
            "port": cell["port"],
            "request_digest": cell["request_proof"]["selected_request_digest"],
        }
        for cell in plan["cells"]
        if cell["replicate_id"] == "rep-1"
    }
    execution_cells = [cells_by_arm[ARC_DEMAND_ARM], cells_by_arm[LRU_DEMAND_ARM]]
    parent_cells = {
        cell["arm_id"]: copy.deepcopy(cell)
        for cell in parent_entry["selected_cells"]
    }
    _require(
        set(parent_cells) == {MSR_FIFO_ARM, PRODUCT_ARM},
        "parent bundle does not contain exactly completed M/R cells",
    )
    all_cells = [
        cells_by_arm[ARC_DEMAND_ARM],
        cells_by_arm[LRU_DEMAND_ARM],
        parent_cells[MSR_FIFO_ARM],
        parent_cells[PRODUCT_ARM],
    ]
    _require(
        len({cell["cell_id"] for cell in all_cells}) == 4
        and len({cell["request_digest"] for cell in all_cells}) == 1
        and len({(cell["profile_id"], cell["replicate_id"], cell["seed"]) for cell in all_cells})
        == 1,
        "continuation cells are not exact matched peers",
    )
    configs_by_arm = {arm["id"]: arm["config"] for arm in campaign["arms"]}
    for arm_id, expected_policy in (
        (ARC_DEMAND_ARM, "arc"),
        (LRU_DEMAND_ARM, "lru"),
    ):
        config = configs_by_arm[arm_id]
        extra = config["extra_config"]
        _require(config["policy"] == expected_policy, f"{arm_id} policy changed")
        _require(extra["store_threshold"] == 2, f"{arm_id} threshold changed")
        _require(
            extra["prefetch_budget_fraction"] == 0.0,
            f"{arm_id} demand-only prefetch budget changed",
        )
    generated = {
        "campaign_path": str(campaign_path.resolve()),
        "campaign_portable_path": str(Path(TWO_SOURCE_ACTIVATION_ID) / "campaign.json"),
        "campaign_sha256": file_digest(campaign_path),
        "request_path": str(request_path.resolve()),
        "request_portable_path": str(
            Path(TWO_SOURCE_ACTIVATION_ID) / "inputs/request-manifest-v1.json"
        ),
        "request_sha256": file_digest(request_path),
        "plan_path": str(plan_path.resolve()),
        "plan_portable_path": str(Path(TWO_SOURCE_ACTIVATION_ID) / "plan.json"),
        "plan_sha256": file_digest(plan_path),
        "plan_digest": plan["plan_digest"],
    }
    entry_base = {
        "campaign_id": TWO_SOURCE_ACTIVATION_ID,
        "continuation_id": DEMAND_CONTINUATION_ID,
        "purpose": "same-input ARC/LRU demand continuation for completed M/R cohort",
        "dimensions": copy.deepcopy(parent_entry["dimensions"]),
        "input_cohort_identity": parent_entry["input_cohort_identity"],
        "regime_identity": parent_entry["regime_identity"],
        "pairing_weight": parent_entry["pairing_weight"],
        "generated": generated,
    }
    command = (
        "uv run --no-sync python -m benchmarks.msr_combined_prefetch_workload "
        f"execute-expanded --bundle {output_dir / 'bundle.json'} "
        f"--results {future_result_root} --campaigns {TWO_SOURCE_ACTIVATION_ID} "
        f"--replicates rep-1 --arms {ARC_DEMAND_ARM} {LRU_DEMAND_ARM}"
    )
    source_identity = {
        "parent_bundle_sha256": file_digest(parent_bundle_path),
        "parent_source_identity": copy.deepcopy(parent_source),
        "connector_sha256": file_digest(REPO_ROOT / "semantic_offload/connector.py"),
        "connector_required_sha256": required_connector_sha,
        "campaign_runner_sha256": file_digest(
            REPO_ROOT / "benchmarks/eviction_prefetch_campaign.py"
        ),
        "workload_sha256": file_digest(Path(__file__)),
    }
    readiness_binding = {
        "scope": "construction_time_provenance_before_clean_execution_freeze",
        "parent_bundle_sha256": source_identity["parent_bundle_sha256"],
        "parent_ready": True,
        "connector_sha256": source_identity["connector_sha256"],
        "connector_required_sha256": source_identity["connector_required_sha256"],
        "campaign_runner_sha256": source_identity["campaign_runner_sha256"],
        "request_sha256": generated["request_sha256"],
        "logical_campaign_id": TWO_SOURCE_ACTIVATION_ID,
        "profile_id": all_cells[0]["profile_id"],
        "replicate_id": "rep-1",
        "seed": all_cells[0]["seed"],
        "selected_arms": [ARC_DEMAND_ARM, LRU_DEMAND_ARM],
        "selected_arm_config_sha256": {
            arm_id: digest(configs_by_arm[arm_id])
            for arm_id in (ARC_DEMAND_ARM, LRU_DEMAND_ARM)
        },
        "semantic_activation_required": False,
    }
    bundle = {
        "schema": EXPANDED_SCHEMA,
        "bundle_id": DEMAND_CONTINUATION_ID,
        "stage": "a06_same_input_demand_controls_proposed_not_gpu_executed",
        "benefit_measured": False,
        "provider_claim": copy.deepcopy(parent["provider_claim"]),
        "source_identity": source_identity,
        "gpu_readiness": {
            "ready": True,
            "reason": (
                "A/L demand continuation has construction-time bindings to the parent-reviewed "
                "connector/campaign runner, exact requests, and accepted demand-control configs; "
                "semantic activation is not required for demand-only controls"
            ),
            "binding": readiness_binding,
        },
        "continuation_of": {
            "bundle_path": str(parent_bundle_path.resolve()),
            "bundle_sha256": file_digest(parent_bundle_path),
            "retained_mr_result_root": retained_mr_result_root,
            "logical_campaign_id": TWO_SOURCE_ACTIVATION_ID,
            "profile_id": all_cells[0]["profile_id"],
            "replicate_id": "rep-1",
            "seed": all_cells[0]["seed"],
            "request_sha256": file_digest(request_path),
            "request_count_per_arm": 125,
        },
        "campaigns": [
            {
                **entry_base,
                "arm_roles": {
                    "arc_demand": ARC_DEMAND_ARM,
                    "lru_demand": LRU_DEMAND_ARM,
                },
                "selected_cells": execution_cells,
            }
        ],
        "future_gpu_commands": {"demand_controls": command},
        "future_aggregation_command": (
            "uv run --no-sync python -m benchmarks.msr_combined_prefetch_attribution "
            f"--bundle {output_dir / 'consolidation-bundle.json'} "
            f"--results-root {retained_mr_result_root} "
            f"--results-root {future_result_root} "
            f"--output-dir {output_dir / 'measured-aggregate-UNIQUE'}"
        ),
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
    }
    consolidation = {
        **copy.deepcopy(bundle),
        "bundle_id": f"{DEMAND_CONTINUATION_ID}-consolidation",
        "stage": "a06_four_arm_consolidation_layout_gpu_unmeasured",
        "campaigns": [
            {
                **entry_base,
                "purpose": "consolidate retained M/R with new A/L exact peers",
                "arm_roles": {
                    "arc_demand": ARC_DEMAND_ARM,
                    "lru_demand": LRU_DEMAND_ARM,
                    "msr_fifo_prefetch": MSR_FIFO_ARM,
                    "msr_semantic_prefetch": PRODUCT_ARM,
                },
                "selected_cells": all_cells,
            }
        ],
    }
    _write_json(output_dir / "bundle.json", bundle)
    _write_json(output_dir / "consolidation-bundle.json", consolidation)
    (output_dir / "future-gpu-command.txt").write_text(command + "\n", encoding="utf-8")
    return bundle


def _replace_implementation_revision(value: object, revision: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "implementation_revision":
                value[key] = revision
            else:
                _replace_implementation_revision(item, revision)
    elif isinstance(value, list):
        for item in value:
            _replace_implementation_revision(item, revision)


def _freeze_staged_execution(
    bundle_path: Path, bundle: Mapping[str, object], results: Path
) -> tuple[Path, Path, dict[str, object]]:
    """Bind the plan to the clean staged source immediately before execution."""
    identity = _git_identity(REPO_ROOT)
    _require(identity["clean"] is True, "future GPU source must be a clean staged tree")
    generated = bundle["generated"]
    campaign_source = Path(generated["campaign_path"])
    request_source = Path(generated["request_path"])
    if not campaign_source.is_file():
        campaign_source = bundle_path.parent / generated["campaign_portable_path"]
    if not request_source.is_file():
        request_source = bundle_path.parent / generated["request_portable_path"]

    freeze_dir = results / "execution-freeze"
    freeze_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(request_source.parent, freeze_dir / "inputs")
    request_path = freeze_dir / "inputs/request-manifest-v1.json"
    campaign = json.loads(campaign_source.read_text(encoding="utf-8"))
    campaign["source"]["implementation_revision"] = identity["revision"]
    campaign["source"]["implementation_tree"] = identity["tree"]
    campaign["request_manifest"] = {
        "path": "inputs/request-manifest-v1.json",
        "sha256": file_digest(request_path),
    }
    campaign["output_root"] = str(results.resolve())
    _replace_implementation_revision(campaign, str(identity["revision"]))
    for arm in campaign["arms"]:
        arm["arm_config_hash"] = digest(arm["config"])
    for reference in campaign.get("control_references", {}).values():
        if isinstance(reference, dict) and isinstance(reference.get("config"), dict):
            reference["config_hash"] = digest(reference["config"])
    for profile in campaign["baseline_profiles"]:
        profile["baseline_profile_hash"] = digest(_profile_payload(campaign, profile))
    campaign_path = freeze_dir / "campaign.json"
    _write_json(campaign_path, campaign)
    validate_campaign_manifest(campaign, campaign_path)
    plan = build_plan(campaign, campaign_path)
    _write_json(freeze_dir / "plan.json", plan)
    _write_json(freeze_dir / "source-identity.json", identity)
    return campaign_path, request_path, plan


def execute_selected(
    bundle_path: Path,
    results: Path,
    replicates: Sequence[str] | None = None,
    arms: Sequence[str] | None = None,
) -> int:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    _require(bundle.get("schema") == SCHEMA, "bundle schema")
    readiness = bundle.get("gpu_readiness", {})
    _require(readiness.get("ready") is True, f"GPU execution gated: {readiness.get('reason')}")
    results.mkdir(parents=True, exist_ok=True)
    campaign, requests, execution_plan = _freeze_staged_execution(
        bundle_path, bundle, results
    )
    planned = {
        (cell["arm_id"], cell["replicate_id"]): cell
        for cell in execution_plan["cells"]
    }
    states = []
    for selected in bundle["selected_cells"]:
        if replicates is not None and selected["replicate_id"] not in replicates:
            continue
        if arms is not None and selected["arm_id"] not in arms:
            continue
        cell = planned[(selected["arm_id"], selected["replicate_id"])]
        cell_dir = results / cell["cell_id"]
        cell_dir.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            "-m",
            "benchmarks.eviction_prefetch_campaign",
            "execute-cell",
            "--manifest",
            str(campaign),
            "--request-manifest",
            str(requests),
            "--profile-id",
            cell["profile_id"],
            "--cell-id",
            cell["cell_id"],
            "--port",
            str(cell["port"]),
            "--output",
            str(cell_dir / "result.json"),
        ]
        with (cell_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            cell_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        state = {
            **cell,
            "argv": command,
            "exit_status": completed.returncode,
            "result_present": (cell_dir / "result.json").is_file(),
        }
        states.append(state)
        _write_json(results / "state.json", states)
        if completed.returncode:
            return completed.returncode
    return 0


def execute_expanded(
    bundle_path: Path,
    results: Path,
    campaign_ids: Sequence[str],
    replicates: Sequence[str] | None,
    arms: Sequence[str] | None,
) -> int:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    _require(bundle.get("schema") == EXPANDED_SCHEMA, "expanded bundle schema")
    _require(bundle.get("gpu_readiness", {}).get("ready") is True, "GPU execution gated")
    entries = {entry["campaign_id"]: entry for entry in bundle["campaigns"]}
    _require(set(campaign_ids).issubset(entries), "unknown expanded campaign")
    results.mkdir(parents=True, exist_ok=True)
    for campaign_id in campaign_ids:
        entry = copy.deepcopy(entries[campaign_id])
        generated = entry["generated"]
        for field in ("campaign", "request", "plan"):
            absolute = Path(generated[f"{field}_path"])
            if not absolute.is_file():
                generated[f"{field}_path"] = str(
                    bundle_path.parent / generated[f"{field}_portable_path"]
                )
        mini = {
            "schema": SCHEMA,
            "generated": generated,
            "selected_cells": entry["selected_cells"],
            "gpu_readiness": bundle["gpu_readiness"],
        }
        mini_path = results / f"{campaign_id}-execution-spec.json"
        _write_json(mini_path, mini)
        status = execute_selected(
            mini_path,
            results / campaign_id,
            replicates=replicates,
            arms=arms,
        )
        if status:
            return status
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--base-campaign", type=Path, default=DEFAULT_BASE_CAMPAIGN)
    build.add_argument("--base-requests", type=Path, default=DEFAULT_BASE_REQUESTS)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--future-result-root", required=True)
    expanded = subparsers.add_parser("build-expanded")
    expanded.add_argument("--base-campaign", type=Path, default=DEFAULT_BASE_CAMPAIGN)
    expanded.add_argument("--base-requests", type=Path, default=DEFAULT_BASE_REQUESTS)
    expanded.add_argument("--output", type=Path, required=True)
    expanded.add_argument("--future-result-root", required=True)
    repair = subparsers.add_parser("build-activation-repair")
    repair.add_argument("--base-campaign", type=Path, default=DEFAULT_BASE_CAMPAIGN)
    repair.add_argument(
        "--accepted-requests", type=Path, default=DEFAULT_ACCEPTED_ACTIVATION_REQUESTS
    )
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--future-result-root", required=True)
    frontier = subparsers.add_parser("build-frontier-diagnostic")
    frontier.add_argument("--base-campaign", type=Path, default=DEFAULT_BASE_CAMPAIGN)
    frontier.add_argument(
        "--accepted-requests", type=Path, default=DEFAULT_ACCEPTED_ACTIVATION_REQUESTS
    )
    frontier.add_argument("--output", type=Path, required=True)
    frontier.add_argument("--future-result-root", required=True)
    two_source = subparsers.add_parser("build-two-source-activation")
    two_source.add_argument(
        "--base-campaign", type=Path, default=DEFAULT_BASE_CAMPAIGN
    )
    two_source.add_argument(
        "--accepted-requests", type=Path, default=DEFAULT_ACCEPTED_ACTIVATION_REQUESTS
    )
    two_source.add_argument("--output", type=Path, required=True)
    two_source.add_argument("--future-result-root", required=True)
    continuation = subparsers.add_parser("build-demand-continuation")
    continuation.add_argument(
        "--parent-bundle", type=Path, default=DEFAULT_TWO_SOURCE_BUNDLE
    )
    continuation.add_argument("--output", type=Path, required=True)
    continuation.add_argument("--future-result-root", required=True)
    continuation.add_argument("--retained-mr-result-root", required=True)
    semantic_value = subparsers.add_parser("build-semantic-value")
    semantic_value.add_argument(
        "--parent-bundle", type=Path, default=DEFAULT_TWO_SOURCE_BUNDLE
    )
    semantic_value.add_argument("--output", type=Path, required=True)
    semantic_value.add_argument("--future-result-root", required=True)
    remaining = subparsers.add_parser("build-remaining-slices-schema")
    remaining.add_argument("--output", type=Path, required=True)
    remaining.add_argument("--future-result-root", required=True)
    remaining_bundle = subparsers.add_parser("build-remaining-slices")
    remaining_bundle.add_argument(
        "--parent-bundle", type=Path, default=DEFAULT_TWO_SOURCE_BUNDLE
    )
    remaining_bundle.add_argument("--output", type=Path, required=True)
    remaining_bundle.add_argument("--future-result-root", required=True)
    refresh_remaining = subparsers.add_parser("refresh-remaining-slices-provenance")
    refresh_remaining.add_argument(
        "--source-bundle", type=Path, default=DEFAULT_REMAINING_SLICES_V5B
    )
    refresh_remaining.add_argument("--output", type=Path, required=True)
    refresh_remaining.add_argument("--future-result-root", required=True)
    supplement = subparsers.add_parser("build-uncertainty-disjoint-supplement")
    supplement.add_argument(
        "--source-bundle", type=Path, default=DEFAULT_REMAINING_SLICES_V5C
    )
    supplement.add_argument(
        "--independent-requests",
        type=Path,
        default=DEFAULT_ACCEPTED_INDEPENDENT_REQUESTS,
    )
    supplement.add_argument("--output", type=Path, required=True)
    supplement.add_argument("--uncertainty-result-root", required=True)
    supplement.add_argument("--disjoint-result-root", required=True)
    supplement.add_argument(
        "--retained-reference-result-roots", nargs="+", required=True
    )
    execute = subparsers.add_parser("execute-selected")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--results", type=Path, required=True)
    execute.add_argument("--replicates", nargs="+")
    execute.add_argument("--arms", nargs="+")
    execute_expanded_parser = subparsers.add_parser("execute-expanded")
    execute_expanded_parser.add_argument("--bundle", type=Path, required=True)
    execute_expanded_parser.add_argument("--results", type=Path, required=True)
    execute_expanded_parser.add_argument("--campaigns", nargs="+", required=True)
    execute_expanded_parser.add_argument("--replicates", nargs="+")
    execute_expanded_parser.add_argument("--arms", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "build":
        bundle = build_bundle(
            base_campaign_path=args.base_campaign,
            base_request_path=args.base_requests,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-expanded":
        bundle = build_expanded_bundle(
            base_campaign_path=args.base_campaign,
            base_request_path=args.base_requests,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-activation-repair":
        bundle = build_activation_repair_bundle(
            base_campaign_path=args.base_campaign,
            accepted_request_path=args.accepted_requests,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-frontier-diagnostic":
        bundle = build_frontier_diagnostic_bundle(
            base_campaign_path=args.base_campaign,
            accepted_request_path=args.accepted_requests,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-two-source-activation":
        bundle = build_two_source_activation_bundle(
            base_campaign_path=args.base_campaign,
            accepted_request_path=args.accepted_requests,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-demand-continuation":
        bundle = build_demand_control_continuation_bundle(
            parent_bundle_path=args.parent_bundle,
            output_dir=args.output,
            future_result_root=args.future_result_root,
            retained_mr_result_root=args.retained_mr_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-semantic-value":
        bundle = build_semantic_value_bundle(
            parent_bundle_path=args.parent_bundle,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-remaining-slices-schema":
        bundle = build_remaining_slices_schema(
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-remaining-slices":
        bundle = build_remaining_slices_bundle(
            parent_bundle_path=args.parent_bundle,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "refresh-remaining-slices-provenance":
        bundle = refresh_remaining_slices_provenance(
            source_bundle_path=args.source_bundle,
            output_dir=args.output,
            future_result_root=args.future_result_root,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "build-uncertainty-disjoint-supplement":
        bundle = build_uncertainty_disjoint_supplement(
            source_bundle_path=args.source_bundle,
            independent_request_path=args.independent_requests,
            output_dir=args.output,
            uncertainty_result_root=args.uncertainty_result_root,
            disjoint_result_root=args.disjoint_result_root,
            retained_reference_result_roots=args.retained_reference_result_roots,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "execute-expanded":
        return execute_expanded(
            args.bundle,
            args.results,
            args.campaigns,
            args.replicates,
            args.arms,
        )
    return execute_selected(args.bundle, args.results, args.replicates, args.arms)


if __name__ == "__main__":
    raise SystemExit(main())
