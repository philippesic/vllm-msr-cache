"""Focused CPU checks for the combined MSR/prefetch claim analyzer."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import msr_combined_prefetch_attribution as attribution
from harness.policies import kv_transfer_config


def _config(policy: str, selector: str) -> list[str]:
    policy_identity = (
        {"method": "mean"}
        if policy == "semantic"
        else {"eviction_policy": policy, "instrumentation_only": True}
    )
    return [
        "python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "model",
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector_extra_config": {
                    "spec_name": "SemanticOffloadingSpec",
                    "spec_module_path": "semantic_offload.spec",
                    "store_threshold": 2,
                    "cpu_bytes_to_use": 268435456,
                    "prefetch_selector": selector,
                    "prefetch_trigger": "known_prefix_wait_hint",
                    "prefetch_score_max_age_ms": 1000,
                    **policy_identity,
                }
            }
        ),
    ]


def _nomination() -> dict:
    def candidate(event_id: str, session_id: str, score: float) -> dict:
        return {
            "event_id": event_id,
            "run_id": "run",
            "cell_id": "cell",
            "session_id": session_id,
            "generation": 0,
            "observed_prefix_id": f"prefix-{session_id}",
            "prefix_boundary_tokens": 32,
            "source_request_id": f"source-{session_id}",
            "namespace": {
                "model_revision": "model-rev",
                "tokenizer_revision": "tokenizer-rev",
                "adapter_namespace": "none",
                "layout_namespace": "dense-tp1-bf16-block16",
            },
            "source_observed_ns": 10,
            "hint_observation_available_ns": 20,
            "hint_input_publication_ns": 30,
            "eligible_resident_prefix_keys": [
                f"key-{session_id}-0",
                f"key-{session_id}-1",
            ],
            "semantic_key_scores": [
                {
                    "key": f"key-{session_id}-0",
                    "score": score,
                    "observation_available_ns": 20,
                    "observation_provenance": "synchronous_causal_observation",
                    "age_ns": 20,
                    "max_age_ns": 1_000_000_000,
                    "fresh_at_decision": True,
                    "feature_snapshot_captured_ns": 20,
                    "feature_age_ns": 20,
                }
            ],
            "aggregate_score": score,
            "score_observation_available_ns": 20,
            "score_aggregation": "mean",
            "causal_inputs": {
                "source_observed_ns": 10,
                "hint_observation_available_ns": 20,
                "hint_input_publication_ns": 30,
                "score_observation_available_ns": 20,
                "score_max_age_ns": 1_000_000_000,
                "decision_ns": 40,
            },
        }

    return {
        "schema": "semantic-cross-session-prefetch-nomination-v1",
        "event": "nomination_decision",
        "selector": "semantic",
        "clock_domain": "host_monotonic",
        "decision_ns": 40,
        "nomination_cpu_ns": 5,
        "shared_allocation_budget_blocks": 1,
        "shared_reserved_blocks": 0,
        "eligible_candidate_blocks": 4,
        "scored_candidate_count": 2,
        "shared_budget_constrained": True,
        "candidate_publication_order_event_ids": ["hint-a", "hint-b"],
        "tie_break": "immutable_hint_order",
        "candidates": [
            candidate("hint-a", "session-a", 0.9),
            candidate("hint-b", "session-b", 0.2),
        ],
        "chosen": [
            {
                "event_id": "hint-a",
                "session_id": "session-a",
                "selected_keys": ["key-session-a-0"],
            }
        ],
        "selected_blocks": 1,
        "selected_bytes": 128,
        "highest_ranked_candidate": {
            "event_id": "hint-a",
            "session_id": "session-a",
            "observed_prefix_id": "prefix-session-a",
            "source_request_id": "source-session-a",
            "aggregate_score": 0.9,
        },
    }


def _reversed_nomination() -> dict:
    event = _nomination()
    event["candidates"][0]["semantic_key_scores"][0]["score"] = 0.1
    event["candidates"][0]["aggregate_score"] = 0.1
    event["candidates"][1]["semantic_key_scores"][0]["score"] = 0.9
    event["candidates"][1]["aggregate_score"] = 0.9
    event["candidates"].reverse()
    event["chosen"] = [
        {
            "event_id": "hint-b",
            "session_id": "session-b",
            "selected_keys": ["key-session-b-0"],
        }
    ]
    event["highest_ranked_candidate"] = {
        "event_id": "hint-b",
        "session_id": "session-b",
        "observed_prefix_id": "prefix-session-b",
        "source_request_id": "source-session-b",
        "aggregate_score": 0.9,
    }
    return event


def _empty_nomination_poll() -> dict:
    return {
        "schema": "semantic-cross-session-prefetch-nomination-v1",
        "event": "nomination_decision",
        "selector": "semantic",
        "decision_ns": 5,
        "nomination_cpu_ns": 1,
        "shared_allocation_budget_blocks": 16,
        "shared_reserved_blocks": 0,
        "eligible_candidate_blocks": 0,
        "scored_candidate_count": 0,
        "shared_budget_constrained": False,
        "candidate_publication_order_event_ids": [],
        "tie_break": "aggregate_score_desc_then_source_identity",
        "highest_ranked_candidate": None,
        "candidates": [],
        "chosen": [],
        "selected_blocks": 0,
        "selected_bytes": 0,
        "no_action_reason": "no_causal_candidate_or_shared_budget",
    }


def _result(role: str, actual: tuple[int, int], ttft: float = 100.0) -> dict:
    policy = "semantic" if role == "msr" else role
    selector = "semantic" if role == "msr" else "disabled"
    requests = []
    events = []
    for index, (request_id, tokens) in enumerate(
        zip(("req-warm", "req-return"), actual)
    ):
        external = f"sc|run-{role}|cell-{role}|chat|sample-{index}"
        requests.append(
            {
                "manifest_request_id": request_id,
                "external_request_id": external,
                "success": True,
                "terminal_complete": True,
                "ttft_ms": ttft,
                "output_token_digest": f"output-{index}",
            }
        )
        events.append(
            {
                "external_request_id": external,
                "run_id": f"run-{role}",
                "cell_id": f"cell-{role}",
                "actual_prompt_tokens_locally_executed": tokens,
                "cumulative_prompt_tokens_locally_executed": tokens,
                "admission_episodes": [
                    {"prompt_tokens_locally_executed": tokens // 2},
                    {"prompt_tokens_locally_executed": tokens - tokens // 2},
                ],
            }
        )
    if role == "msr":
        events[1]["prefetch_attribution"] = {
            "staging_ready_ns": 50,
            "destination_install_submitted_ns": 60,
            "request_usable_ns": 70,
            "demand_boundary_ns": 80,
            "request_first_forward_ns": 90,
            "speculative_transfer": {
                "submitted_bytes": 100,
                "completed_bytes": 100,
                "consumed_bytes": 100,
                "discarded_bytes": 0,
                "unconsumed_completed_bytes": 0,
                "device_time_s": 0.001,
            },
            "required_keys": ["key-session-a-0"],
            "prefetched_keys": ["key-session-a-0"],
            "intersection_keys": ["key-session-a-0"],
            "cpu_residency_proof": {"key-session-a-0": "manager_lookup_hit"},
            "exposed_blocking_stall_s": 0.0,
        }
    return {
        "valid": True,
        "profile_id": "combined-small",
        "replicate_id": "rep-1",
        "seed": 17,
        "request_manifest_sha256": "manifest-sha",
        "order_schedule_digest": "order-sha",
        "baseline_profile_hash": "profile-sha",
        "resolved_launch_argv": _config(policy, selector),
        "requests": requests,
        "preservation_events": events,
        "semantic_prefetch_session_nominations": [_nomination()]
        if role == "msr"
        else [],
        "semantic_prefetch_score_reversal_events": [_reversed_nomination()]
        if role == "msr"
        else [],
    }


def _manifest() -> dict:
    return {
        "requests": [
            {"request_id": "req-warm", "claim_cohort": "warmup"},
            {"request_id": "req-return", "claim_cohort": "return"},
        ]
    }


def _accept_mapping(*args, **kwargs) -> dict:
    return {
        "preservation_basis": "authoritative_request_events",
        "preservation_reason": "authoritative_request_events",
    }


def test_primary_metric_counts_repeated_execution_once(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    report = attribution.analyze(
        msr=_result("msr", (64, 32), ttft=102),
        arc=_result("arc", (64, 56), ttft=100),
        lru=_result("lru", (64, 64), ttft=100),
        request_manifest=_manifest(),
    )

    assert (
        report["arm_metrics"]["msr"]["total_actual_prompt_tokens_locally_executed"]
        == 96
    )
    assert (
        report["arm_metrics"]["msr"]["actual_prompt_tokens_per_completed_request"] == 48
    )
    assert (
        report["arm_metrics"]["msr"]["return_actual_prompt_tokens_locally_executed"]
        == 32
    )
    assert report["comparisons"]["arc"]["executed_prompt_token_reduction_percent"] == 20
    assert report["comparisons"]["lru"]["executed_prompt_token_reduction_percent"] == 25
    assert report["comparisons"]["arc"]["median_ttft_delta_percent"] == 2
    assert report["comparisons"]["arc"]["ttft_allowance_met"] is True
    assert report["prefetch_lifecycle_activation"]["request_usable_before_demand"] == 1
    assert report["benefit_measured"] is False


def test_staging_and_post_bind_ack_are_separate_activation_stages(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    msr = _result("msr", (64, 32))
    lifecycle = msr["preservation_events"][1]["prefetch_attribution"]
    lifecycle["request_usable_ns"] = 85

    report = attribution.analyze(
        msr=msr,
        arc=_result("arc", (64, 56)),
        lru=_result("lru", (64, 64)),
        request_manifest=_manifest(),
    )
    assert report["prefetch_lifecycle_activation"]["staging_ready_before_demand"] == 1
    assert report["prefetch_lifecycle_activation"]["request_usable_before_demand"] == 0
    assert (
        report["prefetch_lifecycle_activation"]["request_usable_before_first_forward"]
        == 1
    )
    assert report["strict_activation_met"] is True


def test_consumed_exact_key_must_come_from_semantic_nomination(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    msr = _result("msr", (64, 32))
    lifecycle = msr["preservation_events"][1]["prefetch_attribution"]
    lifecycle["required_keys"] = ["other-key"]
    lifecycle["prefetched_keys"] = ["other-key"]
    lifecycle["intersection_keys"] = ["other-key"]
    with pytest.raises(attribution.AttributionError, match="nomination to demand"):
        attribution.analyze(
            msr=msr,
            arc=_result("arc", (64, 56)),
            lru=_result("lru", (64, 64)),
            request_manifest=_manifest(),
        )


def test_weak_boolean_residency_fixture_cannot_claim_exact_kv(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    msr = _result("msr", (64, 32))
    msr["preservation_events"][1]["prefetch_attribution"]["cpu_residency_proof"] = {
        "key-session-a-0": True
    }
    with pytest.raises(attribution.AttributionError, match="exact DRAM survival"):
        attribution.analyze(
            msr=msr,
            arc=_result("arc", (64, 56)),
            lru=_result("lru", (64, 64)),
            request_manifest=_manifest(),
        )


def test_missing_lifecycle_costs_remain_unavailable_not_zero():
    result = _result("msr", (64, 32))
    attr = result["preservation_events"][1]["prefetch_attribution"]
    attr["speculative_transfer"].pop("device_time_s")
    attr.pop("exposed_blocking_stall_s")
    summary = attribution._lifecycle(
        result, nominated_keys=frozenset({"key-session-a-0"})
    )
    assert summary["transfer_device_time_s"] == {
        "value": None,
        "observed_count": 0,
        "expected_count": 1,
    }
    assert summary["exposed_blocking_stall_s"]["value"] is None


def test_shared_prefetch_control_is_same_retention_fifo_ablation(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    shared = _result("msr", (64, 48))
    config = json.loads(shared["resolved_launch_argv"][-1])
    config["kv_connector_extra_config"]["prefetch_selector"] = "request_order"
    shared["resolved_launch_argv"][-1] = json.dumps(config)
    report = attribution.analyze(
        msr=_result("msr", (64, 32)),
        arc=_result("arc", (64, 56)),
        lru=_result("lru", (64, 64)),
        shared_prefetch=shared,
        request_manifest=_manifest(),
    )
    assert report["semantic_incremental_attribution"]["available"] is True
    assert (
        report["semantic_incremental_attribution"][
            "control_total_actual_prompt_tokens_locally_executed"
        ]
        == 112
    )

    wrong = _result("arc", (64, 48))
    config = json.loads(wrong["resolved_launch_argv"][-1])
    config["kv_connector_extra_config"]["prefetch_selector"] = "request_order"
    wrong["resolved_launch_argv"][-1] = json.dumps(config)
    with pytest.raises(attribution.AttributionError, match="MSR retention"):
        attribution.analyze(
            msr=_result("msr", (64, 32)),
            arc=_result("arc", (64, 56)),
            lru=_result("lru", (64, 64)),
            shared_prefetch=wrong,
            request_manifest=_manifest(),
        )


def test_pairing_rejects_output_and_input_identity_drift(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    arc = _result("arc", (64, 56))
    arc["requests"][1]["output_token_digest"] = "different"
    with pytest.raises(attribution.AttributionError, match="output differs"):
        attribution.analyze(
            msr=_result("msr", (64, 32)),
            arc=arc,
            lru=_result("lru", (64, 64)),
            request_manifest=_manifest(),
        )

    arc = _result("arc", (64, 56))
    arc["request_manifest_sha256"] = "different"
    with pytest.raises(
        attribution.AttributionError, match="mismatched request_manifest_sha256"
    ):
        attribution.analyze(
            msr=_result("msr", (64, 32)),
            arc=arc,
            lru=_result("lru", (64, 64)),
            request_manifest=_manifest(),
        )


def test_nomination_rejects_future_score_and_inactive_single_session():
    event = _nomination()
    event["candidates"][0]["semantic_key_scores"][0]["observation_available_ns"] = 41
    with pytest.raises(attribution.AttributionError, match="prior-only"):
        attribution.validate_semantic_nomination(
            [event], max_score_age_ns=1_000_000_000
        )


def test_score_reversal_keeps_fifo_budget_and_bytes_fixed():
    result = attribution.validate_score_reversal(
        [_nomination()],
        [_reversed_nomination()],
        max_score_age_ns=1_000_000_000,
    )
    assert result == {"paired_observations": 1, "winner_reversals": 1}

    changed = _reversed_nomination()
    changed["candidate_publication_order_event_ids"].reverse()
    with pytest.raises(
        attribution.AttributionError,
        match="candidate_publication_order_event_ids",
    ):
        attribution.validate_score_reversal(
            [_nomination()], [changed], max_score_age_ns=1_000_000_000
        )

    event = _nomination()
    event["candidates"] = event["candidates"][:1]
    event["candidate_publication_order_event_ids"] = ["hint-a"]
    event["eligible_candidate_blocks"] = 2
    event["scored_candidate_count"] = 1
    event["shared_budget_constrained"] = False
    with pytest.raises(attribution.AttributionError, match="inactive"):
        attribution.validate_semantic_nomination(
            [event], max_score_age_ns=1_000_000_000
        )


def test_empty_and_candidate_no_action_polls_do_not_mask_selected_competition():
    no_action_with_candidates = _nomination()
    no_action_with_candidates["chosen"] = []
    no_action_with_candidates["selected_blocks"] = 0
    no_action_with_candidates["selected_bytes"] = 0
    no_action_with_candidates["no_action_reason"] = (
        "no_causal_candidate_or_shared_budget"
    )

    summary = attribution.validate_semantic_nomination(
        [_empty_nomination_poll(), no_action_with_candidates, _nomination()],
        max_score_age_ns=1_000_000_000,
    )
    assert summary["observations"] == 3
    assert summary["no_action"] == 2
    assert summary["activated_observations"] == 1
    assert summary["selected_event_indexes"] == [2]
    assert summary["selected_keys"] == ["key-session-a-0"]


def test_malformed_selected_poll_and_duplicate_source_inflation_are_rejected():
    malformed = _empty_nomination_poll()
    malformed["chosen"] = [
        {
            "event_id": "not-a-candidate",
            "session_id": "not-a-session",
            "selected_keys": ["not-a-key"],
        }
    ]
    malformed["selected_blocks"] = 1
    malformed["selected_bytes"] = 128
    malformed["no_action_reason"] = None
    with pytest.raises(attribution.AttributionError, match="not a causal candidate"):
        attribution.validate_semantic_nomination(
            [malformed, _nomination()], max_score_age_ns=1_000_000_000
        )

    inflated = _nomination()
    duplicate = copy.deepcopy(inflated["candidates"][0])
    duplicate["event_id"] = "hint-duplicate-source"
    inflated["candidates"].append(duplicate)
    inflated["candidate_publication_order_event_ids"].append("hint-duplicate-source")
    inflated["eligible_candidate_blocks"] += len(
        duplicate["eligible_resident_prefix_keys"]
    )
    inflated["scored_candidate_count"] += 1
    with pytest.raises(attribution.AttributionError, match="duplicate exact source"):
        attribution.validate_semantic_nomination(
            [inflated], max_score_age_ns=1_000_000_000
        )


def test_store_threshold_and_execution_accounting_fail_closed(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    arc = _result("arc", (64, 56))
    config = json.loads(arc["resolved_launch_argv"][-1])
    config["kv_connector_extra_config"]["store_threshold"] = 0
    arc["resolved_launch_argv"][-1] = json.dumps(config)
    with pytest.raises(attribution.AttributionError, match="store_threshold=2"):
        attribution.analyze(
            msr=_result("msr", (64, 32)),
            arc=arc,
            lru=_result("lru", (64, 64)),
            request_manifest=_manifest(),
        )


def test_real_launch_config_producer_canonicalizes_m_a_l_r_and_rejects_unknown():
    produced = {
        "M": kv_transfer_config(
            "semantic-mean",
            1024,
            {
                "preservation_events": True,
                "store_threshold": 2,
                "prefetch_selector": "semantic",
            },
        ),
        "R": kv_transfer_config(
            "semantic-mean",
            1024,
            {
                "preservation_events": True,
                "store_threshold": 2,
                "prefetch_selector": "request_order",
            },
        ),
        "A": kv_transfer_config(
            "arc", 1024, {"preservation_events": True, "store_threshold": 2}
        ),
        "L": kv_transfer_config(
            "lru", 1024, {"preservation_events": True, "store_threshold": 2}
        ),
    }
    assert {
        role: attribution._canonical_policy_identity(
            config["kv_connector_extra_config"]
        )
        for role, config in produced.items()
    } == {"M": "msr", "R": "msr", "A": "arc", "L": "lru"}

    unknown = dict(produced["M"]["kv_connector_extra_config"])
    unknown["method"] = "mystery"
    with pytest.raises(attribution.AttributionError, match="unknown semantic"):
        attribution._canonical_policy_identity(unknown)


def test_execution_accounting_reconciliation_fails_closed(monkeypatch):
    monkeypatch.setattr(attribution, "map_preservation_events", _accept_mapping)
    msr = _result("msr", (64, 32))
    msr["preservation_events"][0]["admission_episodes"][0][
        "prompt_tokens_locally_executed"
    ] += 1
    with pytest.raises(attribution.AttributionError, match="does not reconcile"):
        attribution.analyze(
            msr=msr,
            arc=_result("arc", (64, 56)),
            lru=_result("lru", (64, 64)),
            request_manifest=_manifest(),
        )


def test_duplicate_terminal_event_is_rejected_without_dropping_request():
    run = _result("msr", (64, 32))
    run["preservation_events"].append(copy.deepcopy(run["preservation_events"][0]))
    with pytest.raises(
        attribution.AttributionError,
        match="authoritative preservation mapping unavailable",
    ):
        attribution._execution_by_manifest_request(run, run["requests"])


def _expanded_manifest() -> dict:
    return {
        "requests": [
            {
                "request_id": "req-warm",
                "phase": "warm1",
                "request_index": 0,
                "session_id": "session-a",
                "generation": 0,
                "source_id": "source-a",
                "key_chain_digest": "chain-a",
                "known_prefix_boundary_tokens": 32,
            },
            {
                "request_id": "req-return",
                "phase": "resume",
                "request_index": 1,
                "session_id": "session-a",
                "generation": 0,
                "source_id": "source-a",
                "key_chain_digest": "chain-a",
                "known_prefix_boundary_tokens": 32,
                "intervening_traffic_requests": 3,
            },
        ]
    }


def _expanded_result(role: str = "M") -> dict:
    source_role = {
        "M": "msr",
        "R": "msr",
        "P": "arc",
        "A": "arc",
        "L": "lru",
    }[role]
    result = _result(source_role, (64, 32), ttft=10)
    if role in {"R", "P"}:
        config = json.loads(result["resolved_launch_argv"][-1])
        config["kv_connector_extra_config"]["prefetch_selector"] = "request_order"
        result["resolved_launch_argv"][-1] = json.dumps(config)
    arm_ids = {
        "M": "M-msr-semantic-prefetch-threshold2",
        "R": "R-msr-request-order-prefetch-threshold2",
        "P": "P-arc-request-order-prefetch-threshold2",
        "A": "A-arc-demand-threshold2",
        "L": "L-lru-demand-threshold2",
    }
    result.update(
        schema="ep-campaign-cell-result-v1",
        arm_id=arm_ids[role],
        cell_id=f"cell-{role}",
        campaign_manifest_sha256="campaign",
        normalized_metrics={
            "episode_makespan_ms": {"available": True, "value": 100.0},
            "throughput_tokens_per_s": {"available": True, "value": 20.0},
            "preemptions": {"available": True, "value": 1},
        },
        drain={"status": "complete"},
        physical_transfer_accounting={
            "speculative_submitted_h2d_bytes": 100,
            "speculative_completed_h2d_bytes": 100,
            "speculative_consumed_h2d_bytes": 100,
            "speculative_device_time_s": None,
        },
        physical_occupancy_evidence={
            "phase_samples": [
                {
                    "timestamp_ns": 0,
                    "occupied_gpu_blocks": 2,
                    "occupied_cpu_bytes": 100,
                    "speculative_reserved_blocks": None,
                    "gauges": {"vllm:num_requests_running": 1},
                },
                {
                    "timestamp_ns": 1_000_000_000,
                    "occupied_gpu_blocks": 3,
                    "occupied_cpu_bytes": 200,
                    "speculative_reserved_blocks": None,
                    "gauges": {"vllm:num_requests_running": 1},
                },
            ],
            "event_samples": [],
        },
        wait_hint_events=[_nomination()],
        deployed_runtime={"revision": "runtime", "tree": "clean"},
        deployed_extension={"revision": "extension", "tree": "clean"},
    )
    for index, (request, event) in enumerate(
        zip(result["requests"], result["preservation_events"])
    ):
        request.update(
            send_ns=10 + index * 200,
            planned_release_ns=10 + index * 200,
            first_token_ns=20 + index * 200,
            completion_ns=110 + index * 200,
            token_receipt_ns=[20 + index * 200, 30 + index * 200],
            total_ms=0.1,
        )
        event.update(
            authoritative=True,
            event_integrity="complete",
            session_id="session-a",
            generation=0,
            source_id="source-a",
            reused_tokens=40 if index else 0,
            loaded_tokens=16 if index else 0,
            required_tokens=64,
            recomputed_tokens=48 if index else 64,
            preemption_count=index,
            server_timing={"queue_delay_ns": 1_000_000, "service_time_ns": 2_000_000},
            prefill_cuda_events=[],
        )
    # The expanded extractor must preserve missing device/stall values as null.
    lifecycle = result["preservation_events"][1].get("prefetch_attribution")
    if lifecycle is not None:
        lifecycle["speculative_transfer"].pop("device_time_s")
        lifecycle.pop("exposed_blocking_stall_s")
    return result


def test_expanded_extraction_preserves_missing_and_distinct_reuse():
    cell = attribution.extract_cell_metrics(_expanded_result(), _expanded_manifest())
    returned = cell["requests"][1]
    assert returned["gpu_only_reused_tokens"] == 24
    assert returned["dram_restored_tokens"] == 16
    assert cell["cohorts"]["whole_trace"]["ttft_ms"]["p50"] == 10
    assert (
        cell["cohorts"]["whole_trace"]["actual_prompt_tokens_locally_executed"]["value"]
        == 96
    )
    assert cell["prefetch_lifecycle"][0]["device_time_s"] is None
    assert cell["prefetch_lifecycle"][0]["exposed_stall_s"] is None
    assert cell["occupancy"]["dram_bytes"]["time_integral"] == 100
    assert cell["resources"]["gpu_prefill_ms"]["observed_subtotal"] is None
    assert cell["ttl_observations"][0]["dram_survived"] is True
    assert cell["ttl_observations"][0]["timely_reuse"] is None


def test_event_occupancy_peak_and_byte_time_override_lower_sparse_phase_samples():
    result = _expanded_result()

    def cpu_snapshot(timestamp: int, blocks: int) -> dict:
        resident = [
            {
                "cpu_block_id": index,
                "key": f"key-{index}",
                "ready": True,
                "ref_cnt": 2 if index == 0 else 0,
            }
            for index in range(blocks)
        ]
        return {
            "schema": "semantic-cpu-cache-snapshot-v1",
            "available": True,
            "observed_at_ns": timestamp,
            "accounting_consistent": True,
            "accounting_underflow_blocks": 0,
            "physical_block_bytes": 10,
            "cpu_capacity_bytes": 40,
            "total_blocks": 4,
            "resident_blocks": resident,
            "allocated_blocks": blocks,
            "resident_bytes": blocks * 10,
            "ready_resident_bytes": blocks * 10,
            "write_pending_bytes": 0,
        }

    result["physical_occupancy_evidence"] = {
        "configured_cpu_bytes": 40,
        "configured_gpu_blocks": 320,
        "phase_peaks": {"occupied_cpu_bytes": 10, "occupied_gpu_blocks": 0},
        "phase_samples": [
            {"timestamp_ns": 0, "occupied_cpu_bytes": 0},
            {"timestamp_ns": 10, "occupied_cpu_bytes": 10},
        ],
        "event_samples": [
            {
                "cpu_cache_snapshots": [
                    cpu_snapshot(1, 1),
                    cpu_snapshot(5, 4),
                    cpu_snapshot(9, 4),
                ],
                "block_pool_snapshots": [],
            }
        ],
    }
    cell = attribution.extract_cell_metrics(result, _expanded_manifest())
    dram = cell["occupancy"]["dram_bytes"]
    assert dram["selected_source"] == "event_authoritative"
    assert dram["peak"] == 40
    assert dram["event_authoritative"]["peak"] == 40
    assert dram["phase_boundary"]["peak"] == 10
    assert dram["peak_difference_event_minus_phase"] == 30
    assert dram["time_integral"] > dram["phase_boundary"]["time_integral"]
    assert dram["maximum_sample_gap_s"] == pytest.approx(4e-9)
    assert cell["occupancy"]["dram_physical_blocks"]["peak"] == 4
    assert cell["occupancy"]["shared_dram_physical_blocks"]["peak"] == 1
    assert cell["occupancy"]["dram_capacity_bytes"]["peak"] == 40
    assert cell["occupancy_phase_peaks_reported"]["occupied_cpu_bytes"] == 10

    duplicated = copy.deepcopy(result)
    blocks = duplicated["physical_occupancy_evidence"]["event_samples"][0][
        "cpu_cache_snapshots"
    ][1]["resident_blocks"]
    blocks[1]["cpu_block_id"] = blocks[0]["cpu_block_id"]
    with pytest.raises(attribution.AttributionError, match="duplicates physical"):
        attribution.extract_cell_metrics(duplicated, _expanded_manifest())


def test_expanded_report_writes_coverage_and_null_costs(tmp_path):
    report = attribution.aggregate_result_paths([], _expanded_manifest())
    output = tmp_path / "expanded"
    attribution.write_expanded_report(report, output)
    assert json.loads((output / "aggregate.json").read_text())["conclusions"][
        "average_inference_cost"
    ].startswith("unavailable")
    coverage = (output / "coverage.csv").read_text()
    assert "raw_fields" in coverage and "calculation" in coverage
    assert "phase" in coverage and "status" in coverage
    assert "missing_data_behavior" in coverage
    assert "GPU-unmeasured" in coverage


def test_bundle_aggregation_uses_declared_layout_and_retains_missing_cells(tmp_path):
    bundle_dir = tmp_path / "bundle"
    request_path = bundle_dir / "reference" / "inputs" / "requests.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(_expanded_manifest()))
    target = {"maximum_ms": 1000.0}
    bundle = {
        "schema": "msr-combined-prefetch-expanded-campaign-v1",
        "provider_claim": {"fixed_service_target": target},
        "result_layout": "<results-root>/<campaign_id>/<cell_id>/result.json",
        "metric_phase_table": [],
        "campaigns": [
            {
                "campaign_id": "reference",
                "input_cohort_identity": "inputs",
                "regime_identity": "regime",
                "pairing_weight": 1.0,
                "dimensions": {
                    "dram_blocks": 1024,
                    "fixed_service_target": target,
                },
                "generated": {
                    "request_portable_path": "reference/inputs/requests.json",
                    "request_sha256": hashlib.sha256(
                        request_path.read_bytes()
                    ).hexdigest(),
                },
                "selected_cells": [
                    {
                        "cell_id": "present",
                        "arm_id": "M-msr-semantic-prefetch-threshold2",
                        "profile_id": "combined-small",
                        "replicate_id": "rep-1",
                        "seed": 17,
                    },
                    {
                        "cell_id": "missing",
                        "arm_id": "A",
                        "replicate_id": "rep-1",
                        "seed": 1,
                    },
                ],
            }
        ],
    }
    bundle_path = bundle_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    result_path = tmp_path / "results" / "reference" / "present" / "result.json"
    result_path.parent.mkdir(parents=True)
    present = _expanded_result()
    present["cell_id"] = "present"
    result_path.write_text(json.dumps(present))

    report = attribution.aggregate_bundle_results(bundle_path, tmp_path / "results")
    assert len(report["cells"]) == 1
    assert report["cells"][0]["campaign_dimensions"]["dram_blocks"] == 1024
    assert report["cells"][0]["campaign_dimensions"]["fixed_service_target"] == target
    assert report["missing_results"][0]["cell_id"] == "missing"


def test_p_only_continuation_root_fills_missing_cell_without_duplicates(tmp_path):
    bundle_dir = tmp_path / "bundle"
    request_path = bundle_dir / "reference" / "inputs" / "requests.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(_expanded_manifest()))
    bundle = {
        "schema": "msr-combined-prefetch-expanded-campaign-v1",
        "provider_claim": {"fixed_service_target": {"maximum_ms": 1000.0}},
        "independent_input_cohort_proof": {"valid": False},
        "campaigns": [
            {
                "campaign_id": "reference",
                "input_cohort_identity": "inputs",
                "regime_identity": "regime",
                "pairing_weight": 1.0,
                "dimensions": {"fixed_service_target": {"maximum_ms": 1000.0}},
                "generated": {
                    "request_portable_path": "reference/inputs/requests.json",
                    "request_sha256": hashlib.sha256(
                        request_path.read_bytes()
                    ).hexdigest(),
                },
                "selected_cells": [
                    {
                        "cell_id": "m",
                        "arm_id": "M-msr-semantic-prefetch-threshold2",
                        "profile_id": "combined-small",
                        "replicate_id": "rep-1",
                        "seed": 17,
                    },
                    {
                        "cell_id": "p",
                        "arm_id": "P-arc-request-order-prefetch-threshold2",
                        "profile_id": "combined-small",
                        "replicate_id": "rep-1",
                        "seed": 17,
                    },
                ],
            }
        ],
    }
    bundle_path = bundle_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    for root_name, cell_id, role in (
        ("activation", "m", "M"),
        ("p-only", "p", "P"),
    ):
        result_path = tmp_path / root_name / "reference" / cell_id / "result.json"
        result_path.parent.mkdir(parents=True)
        result = _expanded_result(role)
        result["cell_id"] = cell_id
        result_path.write_text(json.dumps(result))

    report = attribution.aggregate_bundle_results(
        bundle_path, [tmp_path / "activation", tmp_path / "p-only"]
    )
    assert len(report["cells"]) == 2
    assert report["missing_results"] == []
    assert {cell["arm_id"][0] for cell in report["cells"]} == {"M", "P"}


def _expanded_comparable_cells(
    roles: tuple[str, ...] = ("A", "L", "P", "R", "M"),
) -> list[dict]:
    cells = []
    for role in roles:
        cell = attribution.extract_cell_metrics(
            _expanded_result(role), _expanded_manifest()
        )
        cell.update(
            campaign_id="reference",
            regime_identity="regime-reference",
            input_cohort_identity="inputs-a",
            pairing_weight=1.0,
        )
        cells.append(cell)
    return cells


def _apply_a09_coalescing_contract(cells: list[dict]) -> None:
    for cell in cells:
        role = cell["arm_id"][0]
        config = cell["configuration"]["runtime_extra_config"]
        if role in {"A", "L"}:
            config["prefetch_budget_fraction"] = 0.0
            config["prefetch_trigger"] = "preemption"
            config.pop("wait_hint_coalescing_min_sources", None)
            config.pop("wait_hint_coalescing_max_ms", None)
        else:
            config["prefetch_budget_fraction"] = 0.05
            config["prefetch_trigger"] = "known_prefix_wait_hint"
            config["wait_hint_coalescing_min_sources"] = 2
            config["wait_hint_coalescing_max_ms"] = 25


def test_a09_demand_controls_may_omit_unreachable_coalescing_only():
    cells = _expanded_comparable_cells(("A", "L", "R", "M"))
    _apply_a09_coalescing_contract(cells)
    summaries = {
        row["comparison"]: row for row in attribution._paired_uncertainty(cells)
    }
    assert summaries["M-A"]["matched_pairs"] == 1
    assert summaries["M-L"]["matched_pairs"] == 1
    assert summaries["M-R"]["matched_pairs"] == 1

    report = attribution._assemble_expanded(
        cells,
        missing_results=[],
        fixed_service_target_ms=1000.0,
        resource_price_basis=None,
        independent_input_proof=None,
    )
    assert all(row["comparable_group"] is True for row in report["useful_capacity"])

    # The exception is unreachable only for the exact demand-disabled signature.
    tampered = copy.deepcopy(cells)
    arc = next(cell for cell in tampered if cell["arm_id"].startswith("A-"))
    arc["configuration"]["runtime_extra_config"]["prefetch_trigger"] = (
        "known_prefix_wait_hint"
    )
    summaries = {
        row["comparison"]: row for row in attribution._paired_uncertainty(tampered)
    }
    assert summaries["M-A"]["matched_pairs"] == 0
    assert summaries["M-L"]["matched_pairs"] == 1
    assert summaries["M-R"]["matched_pairs"] == 1


def test_a09_r_and_p_must_match_m_generic_coalescing_exactly():
    cells = _expanded_comparable_cells()
    _apply_a09_coalescing_contract(cells)
    summaries = {
        row["comparison"]: row for row in attribution._paired_uncertainty(cells)
    }
    assert all(
        summaries[name]["matched_pairs"] == 1 for name in ("M-A", "M-L", "M-P", "M-R")
    )

    tampered = copy.deepcopy(cells)
    fifo_msr = next(cell for cell in tampered if cell["arm_id"].startswith("R-"))
    fifo_msr["configuration"]["runtime_extra_config"]["wait_hint_coalescing_max_ms"] = (
        24
    )
    arc_prefetch = next(cell for cell in tampered if cell["arm_id"].startswith("P-"))
    arc_prefetch["configuration"]["runtime_extra_config"].pop(
        "wait_hint_coalescing_min_sources"
    )
    summaries = {
        row["comparison"]: row for row in attribution._paired_uncertainty(tampered)
    }
    assert summaries["M-A"]["matched_pairs"] == 1
    assert summaries["M-L"]["matched_pairs"] == 1
    assert summaries["M-P"]["matched_pairs"] == 0
    assert summaries["M-R"]["matched_pairs"] == 0


def test_capacity_and_ttl_use_one_fixed_target_for_all_five_arms():
    report = attribution._assemble_expanded(
        _expanded_comparable_cells(),
        missing_results=[],
        fixed_service_target_ms=1000.0,
        resource_price_basis=None,
        independent_input_proof={
            "all_recorded_identity_intersections_empty": True,
            "working_sets_are_not_pooled": True,
            "reference_campaign_id": "reference",
            "independent_campaign_id": "independent",
        },
    )
    capacity = report["useful_capacity"]
    assert {row["role"] for row in capacity} == {"A", "L", "P", "R", "M"}
    assert all(row["fixed_service_target_ms"] == 1000.0 for row in capacity)
    assert all(row["comparable_group"] is True for row in capacity)
    assert all(row["service_target_observations"] == 1 for row in capacity)
    assert all(row["frontier_claim_supported"] is False for row in capacity)
    msr = next(row for row in capacity if row["role"] == "M")
    assert "delta_sampled_yield_vs_P" in msr
    assert {row["comparison"] for row in report["paired_uncertainty"]} == {
        "M-A",
        "M-L",
        "M-P",
        "M-R",
    }
    assert all(row["timely_reuse_observed"] > 0 for row in report["ttl_curves"])


def test_four_arm_capacity_and_horizon_structure_does_not_require_p():
    report = attribution._assemble_expanded(
        _expanded_comparable_cells(("A", "L", "R", "M")),
        missing_results=[],
        fixed_service_target_ms=1000.0,
        resource_price_basis=None,
        independent_input_proof=None,
    )
    capacity = report["useful_capacity"]
    assert {row["role"] for row in capacity} == {"A", "L", "R", "M"}
    assert all(row["comparable_group"] is True for row in capacity)
    msr = next(row for row in capacity if row["role"] == "M")
    assert set(msr["pair_comparability"]) == {"A", "L", "R"}
    assert msr["delta_sampled_yield_vs_A"] is not None
    assert msr["delta_sampled_yield_vs_L"] is not None
    assert msr["delta_sampled_yield_vs_R"] is not None
    assert "delta_sampled_yield_vs_P" not in msr
    assert all(row["timely_reuse_observed"] > 0 for row in report["ttl_curves"])


def test_mismatched_optional_p_invalidates_only_m_p_capacity_and_ttl_pair():
    cells = _expanded_comparable_cells()
    p_cell = next(cell for cell in cells if cell["arm_id"].startswith("P-"))
    p_cell["requests"][1]["output_token_digest"] = "mismatched-optional-p"
    report = attribution._assemble_expanded(
        cells,
        missing_results=[],
        fixed_service_target_ms=1000.0,
        resource_price_basis=None,
        independent_input_proof=None,
    )
    capacity = {row["role"]: row for row in report["useful_capacity"]}
    assert capacity["P"]["comparable_group"] is False
    assert "output identity differs" in capacity["P"]["unavailable_reason"]
    assert all(
        capacity[role]["comparable_group"] is True for role in ("A", "L", "R", "M")
    )
    assert capacity["M"]["delta_sampled_yield_vs_P"] is None
    assert capacity["M"]["delta_sampled_yield_vs_A"] is not None
    assert capacity["M"]["delta_sampled_yield_vs_L"] is not None
    assert capacity["M"]["delta_sampled_yield_vs_R"] is not None
    comparisons = {row["comparison"]: row for row in report["paired_uncertainty"]}
    assert comparisons["M-P"]["matched_pairs"] == 0
    assert all(
        comparisons[name]["matched_pairs"] == 1 for name in ("M-A", "M-L", "M-R")
    )
    ttl_by_role = {
        row["arm_id"][0]: row["timely_reuse_observed"] for row in report["ttl_curves"]
    }
    assert ttl_by_role["P"] == 0
    assert all(ttl_by_role[role] > 0 for role in ("A", "L", "R", "M"))


def test_regime_seed_summaries_do_not_pool_same_input_variants():
    first = _expanded_comparable_cells()
    second = copy.deepcopy(first)
    for cell in first:
        cell["pairing_weight"] = (
            "one matched trace per replicate; never request-weighted"
        )
    for cell in second:
        cell.update(
            campaign_id="short-horizon",
            regime_identity="regime-short",
            input_cohort_identity="inputs-a",
            pairing_weight="one matched trace per replicate; never request-weighted",
        )
    summaries = attribution._paired_uncertainty(first + second)
    practical = [row for row in summaries if row["comparison"] == "M-P"]
    assert len(practical) == 2
    assert {row["regime_identity"] for row in practical} == {
        "regime-reference",
        "regime-short",
    }
    assert {row["input_cohort_identity"] for row in practical} == {"inputs-a"}
    assert all(row["matched_pairs"] == 1 for row in practical)
    assert all(
        row["executed_prompt_token_reduction_percent"]["weighting_complete"] is True
        for row in practical
    )


def test_resource_cost_requires_explicit_complete_quantities(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_expanded_result()))
    basis = {
        "schema": "msr-resource-price-basis-v1",
        "currency": "USD",
        "required_components": ["dram_byte_seconds"],
        "unit_prices": {"dram_byte_seconds": 0.01},
    }
    report = attribution.aggregate_result_paths(
        [result_path], _expanded_manifest(), resource_price_basis=basis
    )
    cost = report["cells"][0]["resources"]["cost_per_completed_request"]
    assert cost["value"] == pytest.approx(0.5)
    assert cost["status"] == "available"
    assert report["cells"][0]["resources"]["energy_joules"]["status"] == (
        "optional-not-required"
    )

    incomplete = dict(basis)
    incomplete["required_components"] = ["gpu_prefill_ms"]
    incomplete["unit_prices"] = {"gpu_prefill_ms": 0.01}
    report = attribution.aggregate_result_paths(
        [result_path], _expanded_manifest(), resource_price_basis=incomplete
    )
    cost = report["cells"][0]["resources"]["cost_per_completed_request"]
    assert cost["value"] is None
    assert cost["unavailable_reason"]["missing_quantities"] == ["gpu_prefill_ms"]


def _attach_gpu_prefill_accounting(
    result: dict, *, complete: bool, authoritative_zero: bool = False
) -> None:
    actuals = [0, 0] if authoritative_zero else [64, 32]
    eligible = actuals if complete else [64, 0]
    per_request = []
    for event, actual, covered in zip(result["preservation_events"], actuals, eligible):
        event["actual_prompt_tokens_locally_executed"] = actual
        event["cumulative_prompt_tokens_locally_executed"] = actual
        event["admission_episodes"] = [{"prompt_tokens_locally_executed": actual}]
        per_request.append(
            {
                "request_id": event["external_request_id"],
                "internal_request_id": event.get("request_id"),
                "actual_prompt_tokens_locally_executed": actual,
                "eligible_prompt_tokens": covered,
                "eligible_event_count": 1 if covered else 0,
                "observed_event_count": 1 if covered else 0,
                "coverage_fraction": covered / actual if actual else None,
                "complete": covered == actual,
                "device_time_s": (
                    0.0 if actual == 0 else 0.01 if covered == actual else None
                ),
                "observed_device_time_s": 0.01 if covered else None,
                "ineligibility_reasons": {},
            }
        )
    # Retain an overlapping transfer category; it is current-stream context,
    # not another quantity to add to DMA time.
    result["preservation_events"][0]["prefill_cuda_events"] = [
        {
            "request_ids": ["internal-0"],
            "execution_attribution": [{"scheduler_batch_id": 1}],
            "started_at_ns": 1,
            "completed_at_ns": 2,
            "transfer_categories": ["demand_load"],
        }
    ]
    actual_total = sum(actuals)
    eligible_total = sum(eligible)
    observed_time = sum(row["observed_device_time_s"] or 0 for row in per_request)
    result["gpu_prefill_work_accounting"] = {
        "schema": "semantic-bounded-prefill-cuda-accounting-v1",
        "available": complete,
        "reason": (
            "complete_singleton_prompt_cuda_coverage"
            if complete
            else "incomplete_prefill_cuda_prompt_coverage"
        ),
        "request_count": 2,
        "covered_request_count": sum(row["complete"] for row in per_request),
        "requests_with_prompt_work": sum(actual > 0 for actual in actuals),
        "actual_prompt_tokens_locally_executed": actual_total,
        "eligible_prompt_tokens": eligible_total,
        "prompt_token_coverage_fraction": (
            eligible_total / actual_total if actual_total else None
        ),
        "observed_event_count": sum(row["observed_event_count"] for row in per_request),
        "eligible_event_count": sum(row["eligible_event_count"] for row in per_request),
        "device_time_s": observed_time if complete else None,
        "observed_device_time_s": observed_time or None,
        "ineligibility_reasons": {},
        "attribution_scope": (
            "TP1 singleton prompt-only model-runner current CUDA stream; "
            "transfer/splice categories retained and overlap not disambiguated"
        ),
        "synchronization_overhead": "one CUDA end-event synchronize per model step",
        "per_request": per_request,
    }


def test_gpu_prefill_cell_accounting_accepts_transfer_overlap_and_complete_zero():
    result = _expanded_result()
    _attach_gpu_prefill_accounting(result, complete=True)
    cell = attribution.extract_cell_metrics(result, _expanded_manifest())
    gpu = cell["resources"]["gpu_prefill_ms"]
    assert gpu["complete"] is True
    assert gpu["value"] == pytest.approx(20.0)
    assert gpu["observed_subtotal"] == pytest.approx(20.0)
    assert gpu["prompt_token_coverage_fraction"] == 1.0
    assert gpu["overlap_transfer_categories"] == {"demand_load": 1}
    assert "never add" in gpu["overlap_accounting_rule"]
    assert [row["gpu_prefill_ms"] for row in cell["requests"]] == [10.0, 10.0]

    zero = _expanded_result()
    _attach_gpu_prefill_accounting(zero, complete=True, authoritative_zero=True)
    zero_cell = attribution.extract_cell_metrics(zero, _expanded_manifest())
    assert zero_cell["resources"]["gpu_prefill_ms"]["value"] == 0.0
    assert zero_cell["resources"]["gpu_prefill_ms"]["complete"] is True


def test_partial_gpu_prefill_subtotal_is_not_total_or_cost_quantity(tmp_path):
    result = _expanded_result()
    _attach_gpu_prefill_accounting(result, complete=False)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    basis = {
        "schema": "msr-resource-price-basis-v1",
        "currency": "USD",
        "required_components": ["gpu_prefill_ms"],
        "unit_prices": {"gpu_prefill_ms": 1.0},
    }
    report = attribution.aggregate_result_paths(
        [result_path], _expanded_manifest(), resource_price_basis=basis
    )
    cell = report["cells"][0]
    gpu = cell["resources"]["gpu_prefill_ms"]
    assert gpu["complete"] is False
    assert gpu["value"] is None
    assert gpu["observed_subtotal"] == pytest.approx(10.0)
    assert gpu["prompt_token_coverage_fraction"] == pytest.approx(64 / 96)
    cost = cell["resources"]["cost_per_completed_request"]
    assert cost["value"] is None
    assert cost["unavailable_reason"]["missing_quantities"] == ["gpu_prefill_ms"]

    invalid = copy.deepcopy(result)
    invalid["gpu_prefill_work_accounting"]["device_time_s"] = 0.01
    with pytest.raises(attribution.AttributionError, match="partial GPU prefill"):
        attribution.extract_cell_metrics(invalid, _expanded_manifest())


def _attach_gpu_model_runner_accounting(
    result: dict, *, complete: bool, device_time_s: float = 0.03
) -> None:
    for index, event in enumerate(result["preservation_events"]):
        prompt = 64 if index == 0 else 32
        event["request_id"] = f"internal-{index}"
        event["admission_episodes"] = [
            {
                "scheduler_batch_id": 1,
                "scheduled_tokens": prompt,
                "prompt_tokens_locally_executed": prompt,
            }
        ]
    result["gpu_model_runner_work_accounting"] = {
        "schema": "semantic-gpu-model-runner-work-accounting-v1",
        "available": complete,
        "reason": (
            "complete_exact_scheduler_batch_coverage"
            if complete
            else "incomplete_gpu_model_runner_batch_coverage"
        ),
        "device_time_s": device_time_s if complete else None,
        "observed_device_time_s": device_time_s,
        "execution_count": 2 if complete else 1,
        "sequence_first": 1 if complete else 2,
        "sequence_last": 2,
        "sequence_contiguous": complete,
        "chronology_valid": True,
        "owner_sets_valid": True,
        "expected_request_batch_count": 2,
        "observed_request_batch_count": 2 if complete else 1,
        "scheduled_tokens": 96 if complete else 64,
        "prompt_tokens": 96 if complete else 64,
        "decode_tokens": 0,
        "batch_kind_counts": {
            "prompt_only": 2 if complete else 1,
            "decode_only": 0,
            "mixed": 0,
        },
        "transfer_category_execution_counts": {"demand_load": 1},
        "transfer_category_scheduled_tokens": {"demand_load": 64},
        "attribution_scope": (
            "TP1 cell-total model-runner current CUDA stream; shared kernels "
            "not split by request; transfer overlap not disambiguated"
        ),
        "synchronization_overhead": "one CUDA end-event synchronize per real step",
    }


def test_complete_gpu_model_runner_cell_total_and_cost_are_available(tmp_path):
    result = _expanded_result()
    _attach_gpu_model_runner_accounting(result, complete=True)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    basis = {
        "schema": "msr-resource-price-basis-v1",
        "currency": "USD",
        "required_components": ["gpu_model_runner_device_s"],
        "unit_prices": {"gpu_model_runner_device_s": 2.0},
    }
    report = attribution.aggregate_result_paths(
        [result_path], _expanded_manifest(), resource_price_basis=basis
    )
    cell = report["cells"][0]
    work = cell["resources"]["gpu_model_runner_work"]
    assert work["available"] is True
    assert work["device_time_s"] == pytest.approx(0.03)
    assert work["per_completed_request_s"] == pytest.approx(0.015)
    assert work["batch_kind_counts"] == {
        "prompt_only": 2,
        "decode_only": 0,
        "mixed": 0,
    }
    assert work["transfer_category_execution_counts"] == {"demand_load": 1}
    assert "never add" in work["overlap_accounting_rule"]
    assert all("gpu_model_runner" not in request for request in cell["requests"])
    assert cell["resources"]["cost_per_completed_request"]["value"] == (
        pytest.approx(0.03)
    )

    zero = _expanded_result()
    _attach_gpu_model_runner_accounting(zero, complete=True, device_time_s=0.0)
    zero_cell = attribution.extract_cell_metrics(zero, _expanded_manifest())
    assert zero_cell["resources"]["gpu_model_runner_work"]["device_time_s"] == 0.0

    shared = _expanded_result()
    _attach_gpu_model_runner_accounting(shared, complete=True, device_time_s=0.001)
    shared_work = shared["gpu_model_runner_work_accounting"]
    shared_work["execution_count"] = 1
    shared_work["sequence_last"] = 1
    shared_work["batch_kind_counts"]["prompt_only"] = 1
    shared_cell = attribution.extract_cell_metrics(shared, _expanded_manifest())
    assert (
        shared_cell["resources"]["gpu_model_runner_work"][
            "expected_request_batch_count"
        ]
        == 2
    )
    assert shared_cell["resources"]["gpu_model_runner_work"]["execution_count"] == 1
    assert shared_cell["resources"]["gpu_model_runner_work"][
        "device_time_s"
    ] == pytest.approx(0.001)


def test_partial_gpu_model_runner_subtotal_is_not_total_delta_or_cost(tmp_path):
    result = _expanded_result()
    _attach_gpu_model_runner_accounting(result, complete=False, device_time_s=0.02)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    basis = {
        "schema": "msr-resource-price-basis-v1",
        "currency": "USD",
        "required_components": ["gpu_model_runner_device_s"],
        "unit_prices": {"gpu_model_runner_device_s": 1.0},
    }
    report = attribution.aggregate_result_paths(
        [result_path], _expanded_manifest(), resource_price_basis=basis
    )
    work = report["cells"][0]["resources"]["gpu_model_runner_work"]
    assert work["available"] is False
    assert work["device_time_s"] is None
    assert work["observed_device_time_s"] == pytest.approx(0.02)
    assert work["per_completed_request_s"] is None
    cost = report["cells"][0]["resources"]["cost_per_completed_request"]
    assert cost["value"] is None
    assert cost["unavailable_reason"]["missing_quantities"] == [
        "gpu_model_runner_device_s"
    ]


def test_paired_gpu_model_runner_delta_requires_both_complete_cells():
    cells = []
    for role, seconds in (("A", 0.03), ("L", 0.02), ("R", 0.04), ("M", 0.04)):
        result = _expanded_result(role)
        _attach_gpu_model_runner_accounting(
            result, complete=True, device_time_s=seconds
        )
        cell = attribution.extract_cell_metrics(result, _expanded_manifest())
        cell.update(
            campaign_id="gpu-work",
            regime_identity="gpu-work-regime",
            input_cohort_identity="gpu-work-inputs",
            pairing_weight=1.0,
        )
        cells.append(cell)
    summaries = {
        row["comparison"]: row for row in attribution._paired_uncertainty(cells)
    }
    assert summaries["M-A"]["delta_gpu_model_runner_device_time_s"][
        "mean"
    ] == pytest.approx(0.01)
    assert summaries["M-L"]["delta_gpu_model_runner_device_time_s"][
        "mean"
    ] == pytest.approx(0.02)
    assert summaries["M-R"]["delta_gpu_model_runner_device_time_s"]["mean"] == 0

    partial = _expanded_result("R")
    _attach_gpu_model_runner_accounting(partial, complete=False, device_time_s=0.02)
    partial_cell = attribution.extract_cell_metrics(partial, _expanded_manifest())
    partial_cell.update(
        campaign_id="gpu-work",
        regime_identity="gpu-work-regime",
        input_cohort_identity="gpu-work-inputs",
        pairing_weight=1.0,
    )
    summaries = {
        row["comparison"]: row
        for row in attribution._paired_uncertainty(
            [cell for cell in cells if not cell["arm_id"].startswith("R-")]
            + [partial_cell]
        )
    }
    assert summaries["M-R"]["delta_gpu_model_runner_device_time_s"]["mean"] is None


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("sequence_contiguous", False, "sequence/batch/token"),
        ("chronology_valid", False, "sequence/batch/token"),
        ("owner_sets_valid", False, "sequence/batch/token"),
    ],
)
def test_gpu_model_runner_complete_total_rejects_coverage_tamper(field, value, match):
    result = _expanded_result()
    _attach_gpu_model_runner_accounting(result, complete=True)
    result["gpu_model_runner_work_accounting"][field] = value
    with pytest.raises(attribution.AttributionError, match=match):
        attribution.extract_cell_metrics(result, _expanded_manifest())


def test_gpu_model_runner_total_reconciles_exact_admission_map():
    result = _expanded_result()
    _attach_gpu_model_runner_accounting(result, complete=True)
    result["preservation_events"][0]["admission_episodes"][0]["scheduled_tokens"] = 65
    with pytest.raises(attribution.AttributionError, match="admission batches"):
        attribution.extract_cell_metrics(result, _expanded_manifest())


def test_semantic_aggregate_and_rank_are_recomputed_not_trusted():
    event = _nomination()
    event["candidates"][0]["aggregate_score"] = 123.0
    with pytest.raises(attribution.AttributionError, match="causal score mean"):
        attribution.validate_semantic_nomination(
            [event], max_score_age_ns=1_000_000_000
        )

    event = _nomination()
    event["candidates"].reverse()
    with pytest.raises(attribution.AttributionError, match="recomputed fresh causal"):
        attribution.validate_semantic_nomination(
            [event], max_score_age_ns=1_000_000_000
        )


_REAL_RESULT = Path(
    "output/benefit-measurement-20260910-v1/execution-v1/capture-moderate3/"
    "attempt/moderate3/benefit-moderate-screen2048-seed1103-rep-1-"
    "P-arc-prefetch-threshold2/result.json"
)
_REAL_MANIFEST = Path(
    "output/benefit-measurement-20260910-v1/execution-v1/inputs/"
    "request-manifest-v1.json"
)
_A06_RESULT = Path(
    "output/msr-combined-prefetch-gpu-20260910/execution-owner/"
    "two-source-activation-attempt-06/execution/retained-two-source-results/"
    "two-source-activation/two-source-activation-a05-v1/"
    "msr-expanded-two-source-activation-a05-v1-rep-1-"
    "M-msr-semantic-prefetch-threshold2/result.json"
)
_A16_ADAPTER = Path(
    "output/msr-combined-prefetch-gpu-20260910/gpu-work-owner/"
    "a16-validation-attempt-01/adapter.result.json"
)


@pytest.mark.skipif(
    not _REAL_RESULT.is_file() or not _REAL_MANIFEST.is_file(),
    reason="retained real execute-cell evidence is not present",
)
def test_retained_real_execute_cell_schema_is_ingested_without_fake_gpu_timing():
    cell = attribution.extract_cell_metrics(
        json.loads(_REAL_RESULT.read_text()), json.loads(_REAL_MANIFEST.read_text())
    )
    assert cell["schema"] == "msr-combined-prefetch-cell-aggregation-v1"
    assert cell["request_counts"]["completed"] == 32
    assert cell["physical_transfer_accounting"] is not None
    assert cell["occupancy"]["dram_bytes"]["sample_count"] > 0
    assert cell["resources"]["gpu_prefill_ms"]["observed_subtotal"] is None
    assert cell["resources"]["gpu_prefill_ms"]["value"] is None
    assert cell["resources"]["gpu_prefill_ms"]["status"] == ("producer-field-absent")
    assert cell["requests"][0]["gpu_prefill_observed_events"] is None
    model_runner = cell["resources"]["gpu_model_runner_work"]
    assert model_runner["status"] == "producer-field-absent"
    assert model_runner["device_time_s"] is None
    assert model_runner["observed_device_time_s"] is None
    assert model_runner["per_completed_request_s"] is None
    consumed = [
        row
        for row in cell["prefetch_lifecycle"]
        if row.get("consumed_bytes") not in (None, 0)
    ]
    assert consumed
    assert any(
        row["staging_ready_ns"]
        < row["demand_boundary_ns"]
        < row["install_submitted_ns"]
        < row["request_usable_ns"]
        <= row["first_forward_ns"]
        for row in consumed
    )


@pytest.mark.skipif(
    not _A06_RESULT.is_file(),
    reason="retained a06 two-source result is not present",
)
def test_retained_a06_accepts_empty_polls_and_one_constrained_selected_event():
    result = json.loads(_A06_RESULT.read_text())
    events = [
        event
        for event in result["wait_hint_events"]
        if event.get("schema") == "semantic-cross-session-prefetch-nomination-v1"
    ]
    summary = attribution.validate_semantic_nomination(
        events, max_score_age_ns=1_000_000_000
    )
    assert summary["observations"] == 760
    assert summary["no_action"] == 759
    assert summary["selected"] == 2
    assert summary["activated_observations"] == 1
    assert len(summary["selected_event_indexes"]) == 1
    assert len(summary["selected_keys"]) == 16


@pytest.mark.skipif(
    not _A16_ADAPTER.is_file(),
    reason="retained A16 GPU-work adapter result is not present",
)
def test_retained_a16_shared_kernel_cell_total_schema_is_consumed():
    adapter = json.loads(_A16_ADAPTER.read_text())
    accounting = adapter["v3_shared_kernel_exact_result"]
    result = _expanded_result()
    result["gpu_model_runner_work_accounting"] = accounting
    for event, batch_id, tokens in zip(
        result["preservation_events"], (50, 83), (3000, 2500)
    ):
        event["request_id"] = f"internal-{batch_id}"
        event["actual_prompt_tokens_locally_executed"] = tokens
        event["cumulative_prompt_tokens_locally_executed"] = tokens
        event["admission_episodes"] = [
            {
                "scheduler_batch_id": batch_id,
                "scheduled_tokens": tokens,
                "prompt_tokens_locally_executed": tokens,
            }
        ]
    cell = attribution.extract_cell_metrics(result, _expanded_manifest())
    work = cell["resources"]["gpu_model_runner_work"]
    assert work["available"] is True
    assert work["execution_count"] == 1
    assert work["expected_request_batch_count"] == 2
    assert work["observed_request_batch_count"] == 2
    assert work["scheduled_tokens"] == 5500
    assert work["prompt_tokens"] == 5500
    assert work["device_time_s"] == pytest.approx(0.001)
