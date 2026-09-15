# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks import msr_combined_prefetch_workload as workload
from benchmarks.eviction_prefetch_campaign import execute_request_schedule


def _require_historical_artifacts(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        relative = ", ".join(
            str(path.relative_to(workload.REPO_ROOT)) for path in missing
        )
        pytest.skip(f"requires external preserved experiment artifacts: {relative}")


def _build(tmp_path: Path) -> dict:
    _require_historical_artifacts(
        workload.DEFAULT_BASE_CAMPAIGN,
        workload.DEFAULT_BASE_REQUESTS,
    )
    return workload.build_bundle(
        base_campaign_path=workload.DEFAULT_BASE_CAMPAIGN,
        base_request_path=workload.DEFAULT_BASE_REQUESTS,
        output_dir=tmp_path / "bundle",
        future_result_root="/future/msr-combined-prefetch-small-sample",
    )


def test_build_freezes_matched_threshold2_combined_product_plan(tmp_path):
    bundle = _build(tmp_path)
    generated = bundle["generated"]
    campaign = json.loads(Path(generated["campaign_path"]).read_text())
    request_document = json.loads(Path(generated["request_path"]).read_text())
    plan = json.loads(Path(generated["plan_path"]).read_text())

    workload.validate_bundle_contract(
        campaign,
        plan,
        request_document["msr_combined_prefetch_workload"],
    )
    arms = {arm["id"]: arm for arm in campaign["arms"]}
    assert all(
        arms[arm_id]["config"]["extra_config"]["store_threshold"] == 2
        for arm_id in workload.SELECTED_ARMS
    )
    assert arms[workload.PRODUCT_ARM]["config"]["policy"] == "semantic-mean"
    assert (
        arms[workload.PRODUCT_ARM]["config"]["extra_config"]["prefetch_selector"]
        == "semantic"
    )
    assert (
        arms[workload.NONSEMANTIC_PREFETCH_ARM]["config"]["extra_config"]
        ["prefetch_selector"]
        == "request_order"
    )
    product = json.loads(json.dumps(arms[workload.PRODUCT_ARM]["config"]))
    msr_fifo = arms[workload.MSR_FIFO_ARM]["config"]
    product["extra_config"]["prefetch_selector"] = "request_order"
    assert msr_fifo == product
    assert len(bundle["selected_cells"]) == 15
    assert bundle["benefit_measured"] is False
    assert "uv run --no-sync" in bundle["future_gpu_command"]
    assert "--replicates rep-1" in bundle["future_gpu_command"]
    assert workload.NONSEMANTIC_PREFETCH_ARM not in bundle["future_gpu_command"]
    assert f"--arms {workload.ARC_DEMAND_ARM}" in bundle["future_gpu_command"]


def test_workload_has_competing_equal_cost_candidates_return_and_no_return():
    _require_historical_artifacts(workload.DEFAULT_BASE_REQUESTS)
    document = json.loads(workload.DEFAULT_BASE_REQUESTS.read_text())
    metadata = workload.build_workload_metadata(document)
    counterfactual = metadata["nomination_counterfactual"]

    assert len(metadata["returning_candidates"]) == 8
    assert metadata["no_return_sessions"]
    assert set(metadata["lead_times_ms"]) == {50, 250, 750}
    assert metadata["cost_cohorts"]["expensive"] > 0
    assert metadata["cost_cohorts"]["cheap"] > 0
    assert len(counterfactual["candidate_sessions"]) == 2
    assert len(set(counterfactual["candidate_context_tokens"])) == 1
    first, second = counterfactual["semantic_score_scenarios"]
    assert first["expected_semantic_session"] != second["expected_semantic_session"]
    assert first["expected_request_order_session"] == second[
        "expected_request_order_session"
    ]


def _nomination_event(
    selector: str = "semantic",
    *,
    sessions=("a", "b"),
    scores=(0.9, 0.1),
    winner=None,
) -> dict:
    def candidate(session: str, score: float) -> dict:
        return {
            "event_id": f"hint-{session}",
            "session_id": session,
            "source_request_id": f"warm-{session}",
            "observed_prefix_id": f"prefix-{session}",
            "source_observed_ns": 10,
            "hint_observation_available_ns": 20,
            "hint_input_publication_ns": 30,
            "eligible_resident_prefix_keys": [{"block_hash": session}],
            "semantic_key_scores": [
                {
                    "key": {"block_hash": session},
                    "score": score,
                    "observation_available_ns": 20,
                }
            ],
            "aggregate_score": score,
        }

    return {
        "schema": workload.NOMINATION_SCHEMA,
        "event": "nomination_decision",
        "selector": selector,
        "decision_ns": 40,
        "nomination_cpu_ns": 5,
        "shared_allocation_budget_blocks": 16,
        "candidate_publication_order_event_ids": [
            f"hint-{sessions[0]}",
            f"hint-{sessions[1]}",
        ],
        "selected_bytes": 128,
        "selected_blocks": 1,
        "candidates": [
            candidate(sessions[0], scores[0]),
            candidate(sessions[1], scores[1]),
        ],
        "chosen": [
            {
                "session_id": winner or sessions[0],
                "selected_keys": [{"block_hash": winner or sessions[0]}],
            }
        ],
    }


def test_nomination_event_requires_prior_observations_and_score_evidence():
    workload.validate_nomination_event(_nomination_event(), expected_selector="semantic")

    future = _nomination_event()
    future["candidates"][0]["semantic_key_scores"][0][
        "observation_available_ns"
    ] = 41
    with pytest.raises(workload.WorkloadContractError, match="score availability"):
        workload.validate_nomination_event(future, expected_selector="semantic")

    leakage = _nomination_event()
    leakage["candidates"][0]["future_return"] = True
    with pytest.raises(workload.WorkloadContractError, match="future demand leaked"):
        workload.validate_nomination_event(leakage, expected_selector="semantic")


def test_nomination_counterfactual_requires_score_reversal_not_fifo_change():
    _require_historical_artifacts(workload.DEFAULT_BASE_REQUESTS)
    document = json.loads(workload.DEFAULT_BASE_REQUESTS.read_text())
    metadata = workload.build_workload_metadata(document)
    sessions = tuple(metadata["nomination_counterfactual"]["candidate_sessions"])
    events = [
        _nomination_event(sessions=sessions, scores=(0.9, 0.1), winner=sessions[0]),
        _nomination_event(sessions=sessions, scores=(0.1, 0.9), winner=sessions[1]),
    ]
    result = workload.validate_nomination_counterfactual(
        events, [sessions[0], sessions[0]], metadata
    )
    assert result["valid"] is True
    assert result["semantic_winners"] == list(sessions)

    events[1]["chosen"][0]["session_id"] = sessions[0]
    with pytest.raises(workload.WorkloadContractError, match="reversed prior scores"):
        workload.validate_nomination_counterfactual(
            events, [sessions[0], sessions[0]], metadata
        )


def test_execute_selected_fails_closed_when_runtime_capability_is_absent(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        workload,
        "_runtime_nomination_capability",
        lambda: (False, "test runtime gap"),
    )
    bundle = _build(tmp_path)
    with pytest.raises(workload.WorkloadContractError, match="GPU execution gated"):
        workload.execute_selected(
            Path(bundle["generated"]["campaign_path"]).parent / "bundle.json",
            tmp_path / "results",
        )
    assert not (tmp_path / "results").exists()


def test_execution_freeze_binds_clean_staged_source_without_launching(tmp_path, monkeypatch):
    bundle = _build(tmp_path)
    bundle_path = Path(bundle["generated"]["campaign_path"]).parent / "bundle.json"
    monkeypatch.setattr(
        workload,
        "_git_identity",
        lambda _path: {
            "path": "/staged/source",
            "revision": "a" * 40,
            "tree": "b" * 40,
            "clean": True,
        },
    )
    campaign_path, request_path, plan = workload._freeze_staged_execution(
        bundle_path, bundle, tmp_path / "future-results"
    )
    frozen = json.loads(campaign_path.read_text())
    assert frozen["source"]["implementation_revision"] == "a" * 40
    assert frozen["source"]["implementation_tree"] == "b" * 40
    assert request_path.is_file()
    assert plan["cells"]
    assert not list((tmp_path / "future-results").glob("*/result.json"))


def test_result_contract_uses_terminal_actual_prompt_tokens():
    _require_historical_artifacts(workload.DEFAULT_BASE_REQUESTS)
    document = json.loads(workload.DEFAULT_BASE_REQUESTS.read_text())
    contract = workload.result_contract(workload.build_workload_metadata(document))
    primary = contract["primary_metric"]
    assert primary["denominator"] == "baseline_total_actual_prompt_tokens_locally_executed"
    assert "authoritative terminal preservation event" in primary["source"]
    assert contract["serving_constraint"][
        "maximum_msr_regression_percent_vs_each_baseline"
    ] == 3.0
    assert contract["required_activation"]["native_store_threshold"] == 2
    assert contract["semantic_attribution_control"] == workload.MSR_FIFO_ARM
    assert contract["required_activation"]["staging_ready_before_demand"] is True
    assert (
        contract["required_activation"]
        ["request_usable_after_exact_bind_and_all_worker_fence_ack"]
        is True
    )


def test_exact_history_join_survives_reordering_and_repeated_returns():
    def record(request_id, session, digest_value, phase_index, release):
        return {
            "request_id": request_id,
            "session_id": session,
            "generation": 0,
            "known_prefix_boundary_tokens": 32,
            "known_prefix_token_digest": digest_value,
            "phase_index": phase_index,
            "relative_release_ms": release,
            "request_index": len(request_id),
        }

    rows = [
        record("warm-a", "a", "a" * 64, 0, 0),
        record("warm-b", "b", "b" * 64, 1, 0),
        record("noise", "n", "c" * 64, 2, 0),
        record("return-b", "b", "b" * 64, 3, 100),
        record("return-a", "a", "a" * 64, 3, 50),
        record("return-a-again", "a", "a" * 64, 4, 0),
        record("return-a-adjacent", "a", "a" * 64, 5, 0),
    ]
    history = workload.derive_exact_request_history(rows)
    assert history["return-a"]["prior_source_request_id"] == "warm-a"
    assert history["return-a"]["history_role"] == "first_return"
    assert history["return-b"]["prior_source_request_id"] == "warm-b"
    assert history["return-a-again"]["history_role"] == "repeated_return"
    assert history["return-a-adjacent"]["history_role"] == "repeated_return"
    assert history["noise"]["terminal_outcome"] == "cold_no_return"


def test_fixed_service_target_is_absolute_and_includes_release_lateness():
    base = {
        "success": True,
        "terminal_complete": True,
        "planned_release_ns": 10_000_000,
    }
    assert workload.meets_fixed_service_target(
        {**base, "first_token_ns": 1_009_999_999}
    ) is True
    assert workload.meets_fixed_service_target(
        {**base, "first_token_ns": 1_010_000_000}
    ) is True
    assert workload.meets_fixed_service_target(
        {**base, "first_token_ns": 1_010_000_001}
    ) is False
    assert workload.meets_fixed_service_target(
        {**base, "first_token_ns": None}
    ) is None
    assert workload.meets_fixed_service_target(
        {**base, "success": False, "first_token_ns": None}
    ) is False


def test_actual_last_use_ignores_overlapping_nominal_predecessor():
    history = {
        "warm-complete": {"exact_prefix_identity": "exact", "history_role": "first_observation"},
        "warm-overlap": {"exact_prefix_identity": "exact", "history_role": "immediate_repeat"},
        "traffic": {"exact_prefix_identity": "other", "history_role": "first_observation"},
        "return": {"exact_prefix_identity": "exact", "history_role": "first_return"},
    }
    rows = [
        {"manifest_request_id": "warm-complete", "completion_ns": 100},
        {"manifest_request_id": "warm-overlap", "completion_ns": 300},
        {"manifest_request_id": "traffic", "completion_ns": 200},
        {
            "manifest_request_id": "return",
            "send_ns": 250,
            "completion_ns": 400,
            "hint_lead_ms": 50,
        },
    ]
    result = workload.derive_actual_last_use_observations(rows, history)["return"]
    assert result["actual_last_use_source_request_id"] == "warm-complete"
    assert result["actual_last_use_gap_ns"] == 150
    assert result["actual_intervening_completed_request_ids"] == ["traffic"]
    assert result["hint_lead_ms"] == 50


def test_expanded_campaign_has_controlled_nonfactorial_coverage(tmp_path):
    _require_historical_artifacts(
        workload.DEFAULT_BASE_CAMPAIGN, workload.DEFAULT_BASE_REQUESTS
    )
    bundle = workload.build_expanded_bundle(
        base_campaign_path=workload.DEFAULT_BASE_CAMPAIGN,
        base_request_path=workload.DEFAULT_BASE_REQUESTS,
        output_dir=tmp_path / "expanded",
        future_result_root="/future/msr-expanded",
    )
    assert bundle["schema"] == workload.EXPANDED_SCHEMA
    assert bundle["benefit_measured"] is False
    entries = {entry["campaign_id"]: entry for entry in bundle["campaigns"]}
    assert set(entries) == {
        "reference-mixed",
        "capacity-pressure",
        "short-horizon",
        "long-horizon",
        "middle-burst",
        "independent-inputs",
        "lower-offered-rate-recovery",
        "higher-offered-burst-recovery",
    }
    assert entries["reference-mixed"]["dimensions"]["dram_blocks"] == 1024
    assert entries["capacity-pressure"]["dimensions"]["dram_blocks"] == 320
    for entry in entries.values():
        dimensions = entry["dimensions"]
        assert dimensions["configured_shared_budget_blocks"] == 16
        assert dimensions["fixed_service_target"] == workload.FIXED_SERVICE_TARGET
        if dimensions["required_simultaneous_surviving_sources"]:
            assert dimensions["competition_demand_lower_bound_blocks"] > 16
        assert len(entry["selected_cells"]) == 15
        campaign = json.loads(Path(entry["generated"]["campaign_path"]).read_text())
        arms = {arm["id"]: arm for arm in campaign["arms"]}
        assert all(
            arms[arm_id]["config"]["extra_config"]["store_threshold"] == 2
            for arm_id in workload.SELECTED_ARMS
        )
    assert {row["metric"] for row in bundle["metric_phase_table"]} >= {
        "actual_prompt_execution_per_completed_request",
        "serving_distribution_and_tails",
        "semantic_prefetch_usefulness_and_waste",
    }
    activation = bundle["future_gpu_commands"]["activation"]
    assert "--campaigns reference-mixed" in activation
    assert "--replicates rep-1" in activation
    assert workload.NONSEMANTIC_PREFETCH_ARM not in activation
    assert workload.NONSEMANTIC_PREFETCH_ARM in bundle["future_gpu_commands"][
        "practical_arc_prefetch_attribution"
    ]
    p_command = bundle["future_gpu_commands"]["practical_arc_prefetch_attribution"]
    assert f"--arms {workload.NONSEMANTIC_PREFETCH_ARM}" in p_command
    assert workload.ARC_DEMAND_ARM not in p_command
    assert "arc-prefetch-attribution" in bundle["future_aggregation_command"]
    assert bundle["promised_practical_p_comparison"]["new_cells"] == [
        {
            "campaign_id": "reference-mixed",
            "replicate_id": "rep-1",
            "arm_id": workload.NONSEMANTIC_PREFETCH_ARM,
        }
    ]
    assert bundle["provider_claim"]["fixed_service_target"]["maximum_ms"] == 1000
    assert bundle["independent_input_cohort_proof"][
        "all_recorded_identity_intersections_empty"
    ] is True


def test_expanded_request_cohorts_are_exact_history_and_independent(tmp_path):
    _require_historical_artifacts(
        workload.DEFAULT_BASE_CAMPAIGN, workload.DEFAULT_BASE_REQUESTS
    )
    bundle = workload.build_expanded_bundle(
        base_campaign_path=workload.DEFAULT_BASE_CAMPAIGN,
        base_request_path=workload.DEFAULT_BASE_REQUESTS,
        output_dir=tmp_path / "expanded",
        future_result_root="/future/msr-expanded",
    )
    entries = {entry["campaign_id"]: entry for entry in bundle["campaigns"]}
    manifests = {}
    for campaign_id in ("reference-mixed", "independent-inputs"):
        path = Path(entries[campaign_id]["generated"]["request_path"])
        document = json.loads(path.read_text())
        metadata = document["msr_expanded_campaign"]
        selected = workload._causal_requests(document)
        assert metadata["request_history"] == workload.derive_exact_request_history(
            selected
        )
        manifests[campaign_id] = {
            request["known_prefix_token_digest"] for request in selected
        }
        selected_by_id = {request["request_id"]: request for request in selected}
        for spec in document["causal_tool_resume_extension"]["hint_only_events"]:
            source = selected_by_id[spec["source_request_id"]]
            history = metadata["request_history"][source["request_id"]]
            assert history["prior_source_request_id"] is not None
            assert spec["source_exact_prefix_identity"] == history[
                "exact_prefix_identity"
            ]
    assert manifests["reference-mixed"].isdisjoint(manifests["independent-inputs"])


def test_expanded_competition_uses_live_sixteen_block_budget():
    event = _nomination_event()
    event["shared_allocation_budget_blocks"] = 16
    event["selected_blocks"] = 16
    event["selected_bytes"] = 16 * 128
    for candidate in event["candidates"]:
        session = candidate["session_id"]
        candidate["eligible_resident_prefix_keys"] = [
            {"block_hash": f"{session}-{index}"} for index in range(20)
        ]
        candidate["semantic_key_scores"][0]["key"] = candidate[
            "eligible_resident_prefix_keys"
        ][0]
    winner = event["candidates"][0]
    event["chosen"][0]["selected_keys"] = winner["eligible_resident_prefix_keys"][:16]
    result = workload.validate_expanded_live_competition(
        event,
        {
            "configured_shared_budget_blocks": 16,
        },
    )
    assert result["live_shared_allocation_budget_blocks"] == 16
    assert result["eligible_source_count"] == 2
    assert result["eligible_block_demand"] == 40

    reversed_event = json.loads(json.dumps(event))
    reversed_event["candidates"][0]["semantic_key_scores"][0]["score"] = 0.1
    reversed_event["candidates"][0]["aggregate_score"] = 0.1
    reversed_event["candidates"][1]["semantic_key_scores"][0]["score"] = 0.9
    reversed_event["candidates"][1]["aggregate_score"] = 0.9
    reversed_event["chosen"][0]["session_id"] = reversed_event["candidates"][1][
        "session_id"
    ]
    reversed_event["chosen"][0]["selected_keys"] = reversed_event["candidates"][1][
        "eligible_resident_prefix_keys"
    ][:16]
    assert workload.validate_expanded_live_competition(
        reversed_event, {"configured_shared_budget_blocks": 16}
    )["valid"] is True


def test_execute_expanded_delegates_bounded_selection_without_launching(
    tmp_path, monkeypatch
):
    _require_historical_artifacts(
        workload.DEFAULT_BASE_CAMPAIGN, workload.DEFAULT_BASE_REQUESTS
    )
    bundle = workload.build_expanded_bundle(
        base_campaign_path=workload.DEFAULT_BASE_CAMPAIGN,
        base_request_path=workload.DEFAULT_BASE_REQUESTS,
        output_dir=tmp_path / "expanded",
        future_result_root="/future/msr-expanded",
    )
    bundle_path = tmp_path / "expanded/bundle.json"
    bundle["gpu_readiness"]["ready"] = True
    bundle_path.write_text(json.dumps(bundle))
    calls = []

    def fake_execute(bundle_path, results, replicates=None, arms=None):
        calls.append((bundle_path, results, replicates, arms))
        return 0

    monkeypatch.setattr(workload, "execute_selected", fake_execute)
    status = workload.execute_expanded(
        bundle_path,
        tmp_path / "results",
        ["reference-mixed"],
        ["rep-1"],
        [workload.PRODUCT_ARM],
    )
    assert status == 0
    assert len(calls) == 1
    assert calls[0][1] == tmp_path / "results/reference-mixed"
    assert calls[0][2] == ["rep-1"]
    assert calls[0][3] == [workload.PRODUCT_ARM]


def test_activation_repair_walks_two_distinct_large_admission_frontiers():
    _require_historical_artifacts(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS)
    accepted = json.loads(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS.read_text())
    repaired, contract = workload.build_activation_repair_requests(accepted)
    rows = workload._causal_requests(repaired)
    training = rows[:41]
    resumes = rows[49:]
    targets = contract["target_sources"]

    assert len(targets) == 2
    assert len({item["session_id"] for item in targets}) == 2
    assert all(item["known_prefix_boundary_tokens"] == 2064 for item in targets)
    assert all(len(item["separated_target_request_ids"]) == 20 for item in targets)
    assert [training[index]["session_id"] for index in range(40)] == [
        targets[index % 2]["session_id"] for index in range(40)
    ]
    assert training[-1]["session_id"] not in {item["session_id"] for item in targets}
    assert all(request["phase"] == "activation_train" for request in training)
    assert [request["session_id"] for request in resumes[:2]] == [
        item["session_id"] for item in targets
    ]
    assert not (
        {request["session_id"] for request in resumes[2:]}
        & {request["session_id"] for request in training}
    )
    assert all(request["hint_lead_ms"] == 750 for request in resumes)
    assert repaired["causal_tool_resume_extension"]["hint_only_events"] == []
    assert contract["configured_shared_budget_blocks"] == 16
    assert contract["expected_eligible_demand_lower_bound_blocks"] == 20
    assert contract["expected_unique_ready_key_lower_bound"] > 16


def test_activation_repair_bundle_is_new_matched_threshold2_cohort(tmp_path):
    _require_historical_artifacts(
        workload.DEFAULT_BASE_CAMPAIGN,
        workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS,
    )
    bundle = workload.build_activation_repair_bundle(
        base_campaign_path=workload.DEFAULT_BASE_CAMPAIGN,
        accepted_request_path=workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS,
        output_dir=tmp_path / "repair",
        future_result_root="/future/activation-repair",
    )
    assert bundle["stage"] == "a03_evidence_directed_activation_repair_gpu_unmeasured"
    assert bundle["benefit_measured"] is False
    assert len(bundle["campaigns"]) == 1
    entry = bundle["campaigns"][0]
    assert entry["campaign_id"] == workload.ACTIVATION_REPAIR_ID
    assert entry["dimensions"]["configured_shared_budget_blocks"] == 16
    assert entry["dimensions"]["expected_distinct_source_count"] == 2
    campaign = json.loads(Path(entry["generated"]["campaign_path"]).read_text())
    arms = {arm["id"]: arm for arm in campaign["arms"]}
    assert all(
        arms[arm_id]["config"]["extra_config"]["store_threshold"] == 2
        for arm_id in workload.SELECTED_ARMS
    )
    for arm_id in (
        workload.PRODUCT_ARM,
        workload.MSR_FIFO_ARM,
        workload.NONSEMANTIC_PREFETCH_ARM,
    ):
        extra = arms[arm_id]["config"]["extra_config"]
        assert extra["prefetch_budget_fraction"] == 0.05
        assert extra["prefetch_score_max_age_ms"] == 1_000
    assert bundle["provider_claim"]["fixed_service_target"] == workload.FIXED_SERVICE_TARGET
    assert entry["input_cohort_identity"]
    assert entry["regime_identity"]
    assert entry["pairing_weight"]
    assert "--replicates rep-1" in bundle["future_gpu_commands"]["activation_repair"]
    assert workload.NONSEMANTIC_PREFETCH_ARM not in bundle["future_gpu_commands"][
        "activation_repair"
    ]


def test_activation_repair_training_completes_before_atomicish_hint_phase():
    _require_historical_artifacts(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS)
    accepted = json.loads(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS.read_text())
    repaired, contract = workload.build_activation_repair_requests(accepted)
    rows = workload._causal_requests(repaired)

    async def run_schedule():
        async def send(_request):
            return {"success": True}

        return await execute_request_schedule(rows, send)

    results = asyncio.run(run_schedule())
    training = [result for result in results if result["phase_index"] < 41]
    activity = [result for result in results if result["phase_index"] == 41]
    assert len(training) == 41
    assert len(activity) == 16
    assert all(
        result["phase_origin_ns"] >= max(item["completion_ns"] for item in training)
        for result in activity
    )
    manifest_order = contract["hint_publication_contract"][
        "first_manifest_request_ids"
    ]
    activity_ids = [row["request_id"] for row in rows if row["phase"] == "resume"]
    assert activity_ids[:2] == manifest_order


def test_frontier_diagnostic_has_four_probes_and_completed_pressure_groups():
    _require_historical_artifacts(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS)
    accepted = json.loads(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS.read_text())
    repaired, contract = workload.build_frontier_diagnostic_requests(accepted)
    rows = workload._causal_requests(repaired)
    training = rows[:13]
    activity = rows[13:]
    assert len(contract["target"]["probe_request_ids"]) == 4
    assert len(contract["pressure_request_ids_by_cycle"]) == 4
    assert all(len(group) == 2 for group in contract["pressure_request_ids_by_cycle"])
    assert all(
        sum((tokens + 15) // 16 for tokens in group) > 319
        for group in contract["pressure_prompt_tokens_by_cycle"]
    )
    assert contract["expected_frontier_after_cycle"] == [0, 0, 1, 1]
    assert contract["success_threshold_ready_contiguous_blocks"] == 2
    assert all(request["phase"] == "activation_train" for request in training)
    assert all(request["depends_on_phase_completion"] == index - 1 for index, request in enumerate(training) if index)
    assert len(activity) == 16
    assert all(request["phase_index"] == 13 for request in activity)
    assert repaired["causal_tool_resume_extension"]["hint_only_events"] == []


def test_frontier_diagnostic_bundle_selects_only_m_and_exact_control(tmp_path):
    _require_historical_artifacts(
        workload.DEFAULT_BASE_CAMPAIGN,
        workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS,
    )
    bundle = workload.build_frontier_diagnostic_bundle(
        base_campaign_path=workload.DEFAULT_BASE_CAMPAIGN,
        accepted_request_path=workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS,
        output_dir=tmp_path / "frontier",
        future_result_root="/future/frontier",
    )
    assert bundle["stage"] == "a04_frontier_diagnostic_proposed_not_gpu_executed"
    entry = bundle["campaigns"][0]
    assert entry["campaign_id"] == workload.FRONTIER_DIAGNOSTIC_ID
    assert {cell["arm_id"] for cell in entry["selected_cells"]} == {
        workload.ARC_DEMAND_ARM,
        workload.PRODUCT_ARM,
    }
    command = bundle["future_gpu_commands"]["frontier_diagnostic"]
    assert f"--arms {workload.ARC_DEMAND_ARM} {workload.PRODUCT_ARM}" in command
    assert workload.LRU_DEMAND_ARM not in command
    assert bundle["benefit_measured"] is False


def test_two_source_activation_scales_proven_pressure_cycles_minimally():
    _require_historical_artifacts(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS)
    accepted = json.loads(workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS.read_text())
    proposed, contract = workload.build_two_source_activation_requests(accepted)
    rows = workload._causal_requests(proposed)
    training = rows[:109]
    activity = rows[109:]
    targets = contract["target_sources"]
    assert len(rows) == 125
    assert [target["probe_count"] for target in targets] == [18, 18]
    assert [target["expected_ready_private_blocks"] for target in targets] == [9, 9]
    assert contract["expected_unique_eligible_blocks"] == 18
    assert len(contract["pressure_request_ids_by_target_probe"]) == 36
    assert all(
        len(group) == 2 for group in contract["pressure_request_ids_by_target_probe"]
    )
    assert len(training) == 109 and len(activity) == 16
    target_prefixes = {target["known_prefix_token_digest"] for target in targets}
    assert not {
        request["known_prefix_token_digest"] for request in activity[:8]
    } & target_prefixes
    assert [request["session_id"] for request in activity[8:10]] == [
        target["session_id"] for target in targets
    ]
    assert all(request["hint_lead_ms"] == 750 for request in activity[8:])
    assert contract["trace_request_count_per_arm"] == 125
    assert contract["matched_two_arm_request_count"] == 250
    assert proposed["causal_tool_resume_extension"]["hint_only_events"] == []


def test_two_source_bundle_rejects_a_different_connector_source(tmp_path):
    with pytest.raises(
        workload.WorkloadContractError,
        match="two-source proposal requires reviewed coalescing connector source",
    ):
        workload.build_two_source_activation_bundle(
            base_campaign_path=workload.DEFAULT_BASE_CAMPAIGN,
            accepted_request_path=workload.DEFAULT_ACCEPTED_ACTIVATION_REQUESTS,
            output_dir=tmp_path / "two-source",
            future_result_root="/future/two-source",
        )


def test_semantic_value_counterbalance_is_prior_only_and_order_reversed():
    _require_historical_artifacts(workload.DEFAULT_TWO_SOURCE_BUNDLE)
    parent = json.loads(
        Path(
            json.loads(workload.DEFAULT_TWO_SOURCE_BUNDLE.read_text())["campaigns"][0][
                "generated"
            ]["request_path"]
        ).read_text()
    )
    useful_first, useful_contract = workload.build_semantic_value_requests(
        parent, useful_first=True
    )
    distractor_first, distractor_contract = workload.build_semantic_value_requests(
        parent, useful_first=False
    )
    useful_rows = workload._causal_requests(useful_first)
    distractor_rows = workload._causal_requests(distractor_first)
    assert len(useful_rows) == len(distractor_rows) == 126
    assert [row["request_id"] for row in useful_rows] == [
        row["request_id"] for row in distractor_rows
    ]
    assert [row["prompt_token_digest"] for row in useful_rows] == [
        row["prompt_token_digest"] for row in distractor_rows
    ]
    assert useful_contract["useful_target"] == distractor_contract["useful_target"]
    assert useful_contract["censored_distractor"] == distractor_contract[
        "censored_distractor"
    ]
    assert useful_contract["candidate_publication_order"] == list(
        reversed(distractor_contract["candidate_publication_order"])
    )
    signal = next(
        row
        for row in useful_rows
        if row["request_id"] == useful_contract["prior_signal"]["request_id"]
    )
    assert signal["phase_index"] == 109
    assert signal["depends_on_phase_completion"] == 108
    assert len(signal["prompt_token_ids"]) == 225
    assert signal["known_prefix_boundary_tokens"] == 160
    assert useful_contract["prior_signal"][
        "derived_exclusively_from_source_id"
    ] == useful_contract["useful_target"]["source_id"]
    assert useful_contract["prior_signal"][
        "private_exact_block_overlap_with_candidates"
    ] == 0
    useful_session = useful_contract["useful_target"]["session_id"]
    distractor_session = useful_contract["censored_distractor"]["session_id"]
    distractor_prefix = useful_contract["censored_distractor"][
        "known_prefix_token_digest"
    ]
    resumes = [row for row in useful_rows if row["phase"] == "resume"]
    assert sum(row["session_id"] == useful_session for row in resumes) == 1
    assert all(row["session_id"] != distractor_session for row in resumes)
    assert all(row["known_prefix_token_digest"] != distractor_prefix for row in resumes)
    activation = useful_first["causal_tool_resume_extension"]["activation_training"]
    assert activation["serialize_hint_publication"] is True
    assert activation["pollution_release_offsets_ms"][0] == 1_000
    assert min(activation["pollution_release_offsets_ms"]) > 750
    assert useful_contract["expected_unique_eligible_blocks"] == 18
    assert useful_contract["configured_shared_budget_blocks"] == 16

    forbidden = {"future_return", "eventual_demand", "useful", "distractor"}
    for row in useful_rows:
        assert forbidden.isdisjoint(row)
        assert forbidden.isdisjoint(row["serving_inputs"])
    for hint in useful_first["causal_tool_resume_extension"]["hint_only_events"]:
        assert forbidden.isdisjoint(hint)


def test_semantic_value_bundle_selects_only_matched_m_r(tmp_path):
    _require_historical_artifacts(workload.DEFAULT_TWO_SOURCE_BUNDLE)
    bundle = workload.build_semantic_value_bundle(
        parent_bundle_path=workload.DEFAULT_TWO_SOURCE_BUNDLE,
        output_dir=tmp_path / "semantic-value",
        future_result_root="/future/semantic-value",
    )
    assert bundle["gpu_readiness"]["ready"] is True
    assert bundle["benefit_measured"] is False
    assert [entry["campaign_id"] for entry in bundle["campaigns"]] == list(
        workload.SEMANTIC_VALUE_SCENARIOS
    )
    for entry in bundle["campaigns"]:
        assert {cell["arm_id"] for cell in entry["selected_cells"]} == {
            workload.MSR_FIFO_ARM,
            workload.PRODUCT_ARM,
        }
        assert {cell["request_count"] for cell in entry["selected_cells"]} == {126}
        campaign = json.loads(Path(entry["generated"]["campaign_path"]).read_text())
        configs = {arm["id"]: arm["config"] for arm in campaign["arms"]}
        for arm_id in (workload.MSR_FIFO_ARM, workload.PRODUCT_ARM):
            extra = configs[arm_id]["extra_config"]
            assert extra["store_threshold"] == 2
            assert extra["prefetch_budget_fraction"] == 0.05
            assert extra["prefetch_score_max_age_ms"] == 1_000
            assert extra["wait_hint_coalescing_min_sources"] == 2
            assert extra["wait_hint_coalescing_max_ms"] == 25
    command = bundle["future_gpu_commands"]["semantic_value_counterbalance"]
    assert all(scenario in command for scenario in workload.SEMANTIC_VALUE_SCENARIOS)
    assert f"--arms {workload.MSR_FIFO_ARM} {workload.PRODUCT_ARM}" in command
    assert workload.ARC_DEMAND_ARM not in command
    assert workload.LRU_DEMAND_ARM not in command
    assert "only if M beats R" in bundle["conditional_next_step"]


def test_remaining_slices_schema_freezes_nonfactorial_contrasts_without_guessing_api(
    tmp_path,
):
    bundle = workload.build_remaining_slices_schema(
        output_dir=tmp_path / "remaining",
        future_result_root="/future/remaining",
    )
    assert bundle["schema"] == workload.REMAINING_SLICES_SCHEMA
    assert bundle["gpu_readiness"]["ready"] is False
    assert bundle["dependencies"]["chain_admission"]["config_field"] == (
        "chain_admission_max_prefix_blocks"
    )
    assert bundle["dependencies"]["chain_admission"]["config_value"] == 128
    assert bundle["dependencies"]["chain_admission"]["default_value"] == 0
    assert bundle["dependencies"]["chain_admission"]["status"].startswith(
        "integrated"
    )
    assert bundle["dependencies"]["post_drain_recovery"]["status"].startswith(
        "integrated"
    )
    profiles = {item["profile_id"]: item for item in bundle["profiles"]}
    assert tuple(profiles) == workload.REMAINING_SLICE_PROFILES
    assert sum(len(item["planned_cells"]) for item in profiles.values()) == 24
    for profile in profiles.values():
        assert {cell["arm_id"] for cell in profile["planned_cells"]} == {
            workload.ARC_DEMAND_ARM,
            workload.LRU_DEMAND_ARM,
            workload.MSR_FIFO_ARM,
            workload.PRODUCT_ARM,
        }
        dimensions = profile["dimensions"]
        assert dimensions["store_threshold"] == 2
        assert dimensions["prefetch_budget_blocks"] == 16
        assert dimensions["prefetch_score_max_age_ms"] == 1_000
        assert dimensions["wait_hint_coalescing_min_sources"] == 2
        assert dimensions["wait_hint_coalescing_max_ms"] == 25
        assert dimensions["chain_admission_max_prefix_blocks"] == 128
        assert dimensions["fixed_service_target"]["maximum_ms"] == 1_000.0
        assert profile["activation_gate"]["chain_admission"][
            "required_per_target"
        ]["stored_keys"] == 128

    reference = profiles["resource-reference"]["dimensions"]
    lower = profiles["resource-lower-memory"]["dimensions"]
    larger = profiles["resource-larger-workset"]["dimensions"]
    assert (reference["dram_blocks"], lower["dram_blocks"]) == (1_024, 512)
    assert reference["working_set_source_count"] == lower[
        "working_set_source_count"
    ] == 8
    assert reference["nominal_context_working_set_blocks"] == 1_024
    assert reference["dram_blocks"] == larger["dram_blocks"] == 1_024
    assert larger["working_set_source_count"] == 11
    assert larger["nominal_context_working_set_blocks"] == 1_408
    assert {
        profiles[name]["dimensions"]["context_age_idle_before_score_refresh_ms"]
        for name in ("horizon-short", "resource-reference", "horizon-long")
    } == {2_000, 5_000, 15_000}
    assert {
        profiles[name]["dimensions"]["completed_intervening_request_count"]
        for name in ("horizon-short", "resource-reference", "horizon-long")
    } == {4}
    burst = profiles["load-burst-recovery"]["dimensions"]
    assert reference["max_server_concurrency"] == 4
    assert burst["max_server_concurrency"] == 4
    assert reference["load_release_offsets_ms"] == [250 * index for index in range(8)]
    assert burst["load_release_offsets_ms"] == [0] * 8
    assert burst["recovery_release_offsets_ms"] == reference[
        "recovery_release_offsets_ms"
    ]
    commands = bundle["smallest_first_future_commands"]
    assert next(iter(commands)) == "01_reference_a_m"
    assert f"{workload.ARC_DEMAND_ARM} {workload.PRODUCT_ARM}" in commands[
        "01_reference_a_m"
    ]
    assert all(workload.NONSEMANTIC_PREFETCH_ARM not in command for command in commands.values())


def test_remaining_slice_requests_use_chain_admission_idle_and_post_drain_phases():
    _require_historical_artifacts(workload.DEFAULT_TWO_SOURCE_BUNDLE)
    parent_bundle = json.loads(workload.DEFAULT_TWO_SOURCE_BUNDLE.read_text())
    parent = json.loads(
        Path(parent_bundle["campaigns"][0]["generated"]["request_path"]).read_text()
    )
    reference, reference_meta = workload.build_remaining_slice_requests(
        parent,
        profile_id="resource-reference",
        working_set_source_count=8,
        pre_hint_idle_ms=5_000,
        load_release_offsets_ms=[250 * index for index in range(8)],
    )
    _, larger_meta = workload.build_remaining_slice_requests(
        parent,
        profile_id="resource-larger-workset",
        working_set_source_count=11,
        pre_hint_idle_ms=5_000,
        load_release_offsets_ms=[250 * index for index in range(8)],
    )
    assert reference_meta["trace_request_count_per_arm"] == 48
    assert larger_meta["trace_request_count_per_arm"] == 54
    assert reference_meta["deduplicated_constructed_chain_keys"] == 825
    assert larger_meta["deduplicated_constructed_chain_keys"] == 1_142
    assert len(reference_meta["fixed_completed_intervening_request_ids"]) == 4
    assert reference_meta["chain_admission_max_prefix_blocks"] == 128
    rows = workload._causal_requests(reference)
    assert [row["request_index"] for row in rows] == list(range(48))
    extension = reference["causal_tool_resume_extension"]["activation_training"]
    assert extension["pre_hint_idle_ms"] == 0
    signal = next(row for row in rows if row["phase"] == "activation_train" and "prior-topical-signal" in row["request_id"])
    assert signal["relative_release_ms"] == 5_000
    assert signal["wait_origin"] == "fixed_context_age_idle_then_causal_score_refresh"
    assert reference_meta["context_age_idle_before_signal_ms"] == 5_000
    assert extension["post_activity_load_request_count"] == 8
    assert extension["post_drain_recovery_request_count"] == 4
    assert extension["post_drain_recovery_release_offsets_ms"] == [
        250,
        500,
        750,
        1_000,
    ]
    phases = [row["phase"] for row in rows]
    assert phases.count("load") == 8
    assert phases.count("recovery") == 4
    assert reference_meta["candidate_hint_order"][0].endswith(
        "candidate-000-return"
    )
    assert reference_meta["candidate_hint_order"][1] == (
        "remaining-candidate-001-hint"
    )
    assert reference_meta["all_costs_included"] is True
    assert reference_meta["serialize_within_phase"] is True


def test_serialized_phase_never_has_two_live_requests_and_preserves_order():
    rows = [
        {
            "request_id": f"request-{index}",
            "request_index": index,
            "phase_index": 0,
            "relative_release_ms": 0,
        }
        for index in range(4)
    ]
    active = 0
    maximum_active = 0
    observed = []

    async def run():
        async def send(request):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            observed.append(request["request_id"])
            await asyncio.sleep(0.001)
            active -= 1
            return {"success": True}

        return await execute_request_schedule(
            rows, send, serialize_within_phase=True
        )

    results = asyncio.run(run())
    assert maximum_active == 1
    assert observed == [row["request_id"] for row in rows]
    assert [row["manifest_request_id"] for row in results] == observed


def test_remaining_slices_bundle_rejects_a_different_connector_source(tmp_path):
    with pytest.raises(
        workload.WorkloadContractError,
        match="remaining slices require chain-admission connector drain",
    ):
        workload.build_remaining_slices_bundle(
            parent_bundle_path=workload.DEFAULT_TWO_SOURCE_BUNDLE,
            output_dir=tmp_path / "remaining-final",
            future_result_root="/future/remaining",
        )


def test_remaining_slices_refresh_rejects_a_different_analyzer_source(tmp_path):
    _require_historical_artifacts(workload.DEFAULT_REMAINING_SLICES_V5B)
    with pytest.raises(
        workload.WorkloadContractError,
        match="remaining provenance requires final analyzer source",
    ):
        workload.refresh_remaining_slices_provenance(
            source_bundle_path=workload.DEFAULT_REMAINING_SLICES_V5B,
            output_dir=tmp_path / "remaining-refresh",
            future_result_root="/future/remaining-source-refresh",
        )


def test_uncertainty_disjoint_supplement_selects_only_twelve_frozen_cells(tmp_path):
    _require_historical_artifacts(
        workload.DEFAULT_REMAINING_SLICES_V5C,
        workload.DEFAULT_ACCEPTED_INDEPENDENT_REQUESTS,
    )
    retained = ["/retained/reference-am", "/retained/reference-lr"]
    bundle = workload.build_uncertainty_disjoint_supplement(
        source_bundle_path=workload.DEFAULT_REMAINING_SLICES_V5C,
        independent_request_path=workload.DEFAULT_ACCEPTED_INDEPENDENT_REQUESTS,
        output_dir=tmp_path / "supplement",
        uncertainty_result_root="/future/reference-repeats",
        disjoint_result_root="/future/disjoint",
        retained_reference_result_roots=retained,
    )
    assert bundle["gpu_readiness"]["ready"] is True
    assert bundle["gpu_readiness"]["new_execution_cell_count"] == 16
    assert bundle["gpu_readiness"]["retained_rep1_cell_count"] == 0
    assert sum(len(entry["selected_cells"]) for entry in bundle["campaigns"]) == 16
    reference, independent = bundle["campaigns"]
    assert reference["campaign_id"] == "resource-reference"
    assert {cell["replicate_id"] for cell in reference["selected_cells"]} == {
        "rep-1",
        "rep-2",
        "rep-3",
    }
    assert {cell["seed"] for cell in reference["new_execution_cells"]} == {
        1103,
        2207,
        3301,
    }
    source_bundle = json.loads(workload.DEFAULT_REMAINING_SLICES_V5C.read_text())
    source_reference = next(
        entry
        for entry in source_bundle["campaigns"]
        if entry["campaign_id"] == "resource-reference"
    )
    for kind in ("campaign", "request", "plan"):
        assert reference["generated"][f"{kind}_sha256"] == source_reference[
            "generated"
        ][f"{kind}_sha256"]
    assert independent["campaign_id"] == "resource-reference-independent"
    assert len(independent["selected_cells"]) == 4
    assert {cell["replicate_id"] for cell in independent["selected_cells"]} == {
        "rep-1"
    }
    assert bundle["disjoint_identity_proof"]["all_empty"] is True
    assert all(
        values == []
        for values in bundle["disjoint_identity_proof"]["intersections"].values()
    )
    proof = bundle["independent_input_cohort_proof"]
    assert proof["all_recorded_identity_intersections_empty"] is True
    assert proof["working_sets_are_not_pooled"] is True
    assert proof["reference_campaign_id"] == "resource-reference"
    assert proof["independent_campaign_id"] == "resource-reference-independent"
    assert "source_sha256" in proof["fields"]
    assert reference["pairing_weight"] == independent["pairing_weight"] == 1.0
    request_document = json.loads(
        Path(independent["generated"]["request_path"]).read_text()
    )
    activation = request_document["causal_tool_resume_extension"][
        "activation_training"
    ]
    assert activation["serialize_within_phase"] is True
    assert activation["context_age_idle_before_score_refresh_ms"] == 5_000
    assert activation["pre_hint_idle_ms"] == 0
    assert independent["construction"]["deduplicated_constructed_chain_keys"] > 512
    for row in workload._causal_requests(request_document):
        assert {"future_return", "eventual_demand", "future_reuse"}.isdisjoint(
            row["serving_inputs"]
        )
    campaign = json.loads(Path(independent["generated"]["campaign_path"]).read_text())
    configs = {arm["id"]: arm["config"] for arm in campaign["arms"]}
    for arm_id in (
        workload.ARC_DEMAND_ARM,
        workload.LRU_DEMAND_ARM,
        workload.MSR_FIFO_ARM,
        workload.PRODUCT_ARM,
    ):
        extra = configs[arm_id]["extra_config"]
        assert extra["store_threshold"] == 2
        assert extra["chain_admission_max_prefix_blocks"] == 128
        assert extra["bounded_prefill_cuda_timing"] is True
    assert all(root not in bundle["future_analysis_command"] for root in retained)
    assert "rep-1 rep-2 rep-3" in bundle["future_gpu_commands"][
        "reference_rep1_rep2_rep3"
    ]
    assert "rep-1" in bundle["future_gpu_commands"]["disjoint_rep1"]


def test_a21_event_snapshots_prove_full_512_block_pressure_not_phase_peak():
    result_root = workload.REPO_ROOT / (
        "output/msr-combined-prefetch-gpu-20260910/execution-owner/"
        "all-metrics-lower-memory-attempt-21/execution/retained/"
        "all-metrics-results/resource-lower-memory"
    )
    _require_historical_artifacts(result_root)
    paths = sorted(result_root.glob("*/result.json"))
    assert len(paths) == 4
    output_digests = []
    for path in paths:
        result = json.loads(path.read_text())
        assert result["valid"] is True
        output_digests.append(
            {
                request["manifest_request_id"]: request["output_token_digest"]
                for request in result["requests"]
            }
        )
        final_accounting = result["preservation_events"][-1][
            "chain_admission_global_accounting"
        ]
        assert final_accounting["stored_keys"] == 1_269
        assert final_accounting["failed_store_keys"] == 0
        snapshots = [
            snapshot
            for event in result["preservation_events"]
            for snapshot in event["prefetch_attribution"].get(
                "cpu_cache_snapshots", []
            )
            if snapshot.get("available") is True
        ]
        assert max(snapshot["resident_bytes"] for snapshot in snapshots) == 512 * 458_752
        assert max(snapshot["ready_resident_bytes"] for snapshot in snapshots) == (
            512 * 458_752
        )
        assert result["physical_occupancy_evidence"]["phase_peaks"][
            "occupied_cpu_bytes"
        ] == 128 * 458_752
        if result["arm_id"] == workload.PRODUCT_ARM:
            activated = next(
                event
                for event in result["wait_hint_events"]
                if event.get("event") == "nomination_decision"
                and event.get("selected_blocks") == 16
            )
            assert [
                len(candidate["eligible_resident_prefix_keys"])
                for candidate in activated["candidates"]
            ] == [127, 127]
            assert activated["eligible_candidate_blocks"] == 254
    assert all(outputs == output_digests[0] for outputs in output_digests[1:])


def test_demand_continuation_rejects_a_different_connector_source(tmp_path):
    _require_historical_artifacts(workload.DEFAULT_TWO_SOURCE_BUNDLE)
    with pytest.raises(
        workload.WorkloadContractError,
        match="parent/current connector source binding changed",
    ):
        workload.build_demand_control_continuation_bundle(
            parent_bundle_path=workload.DEFAULT_TWO_SOURCE_BUNDLE,
            output_dir=tmp_path / "continuation",
            future_result_root="/future/two-source-demand-controls",
            retained_mr_result_root="/retained/two-source-activation",
        )


def test_load_contrast_preserves_inputs_and_observes_post_drain_recovery():
    _require_historical_artifacts(workload.DEFAULT_BASE_REQUESTS)
    base = json.loads(workload.DEFAULT_BASE_REQUESTS.read_text())
    variants = {variant["id"]: variant for variant in workload.EXPANDED_VARIANTS}
    low, _ = workload._expanded_request_document(
        base, variants["lower-offered-rate-recovery"]
    )
    high, _ = workload._expanded_request_document(
        base, variants["higher-offered-burst-recovery"]
    )
    low_rows = [
        request
        for request in low["requests"]
        if request["split"] == "screen"
        and request["context_budget"] == 2048
        and request["episode"] == "reusable_prefix"
    ]
    high_rows = [
        request
        for request in high["requests"]
        if request["split"] == "screen"
        and request["context_budget"] == 2048
        and request["episode"] == "reusable_prefix"
    ]
    assert [row["prompt_token_digest"] for row in low_rows] == [
        row["prompt_token_digest"] for row in high_rows
    ]
    assert [row["relative_release_ms"] for row in low_rows if row["phase"] == "pollution"] == [
        0, 500, 1000, 1500, 2000, 2500, 3000, 3500
    ]
    assert [row["relative_release_ms"] for row in high_rows if row["phase"] == "pollution"] == [0] * 8

    async def run_schedule():
        async def send(_request):
            return {"success": True}

        return await execute_request_schedule(high_rows, send)

    results = asyncio.run(run_schedule())
    phase16 = [row for row in results if row["phase_index"] == 16]
    phase17 = [row for row in results if row["phase_index"] == 17]
    assert len(phase16) == len(phase17) == 8
    assert all(row["phase_origin_ns"] >= max(item["completion_ns"] for item in phase16) for row in phase17)
