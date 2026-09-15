# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchmarks.eviction_prefetch_campaign import (
    CAUSAL_SCHEMA,
    COMBINATION_OBLIGATIONS,
    FEATURES,
    FROZEN_ORIGINAL_SEMANTIC_CONFIG,
    NATIVE_RUNTIME_VARIANTS,
    NATIVE_RUNTIME_VARIANTS_BINDING_SHA256,
    REQUEST_SCHEMA,
    REQUIRED_EFFECTIVE_ENVIRONMENT,
    REQUIRED_METRICS,
    REQUIRED_NEUTRAL_SETTINGS,
    REQUIRED_SETTINGS,
    RUNTIME_REVISION,
    SCHEMA,
    SEEDS,
    SOURCE_BASELINE,
    ManifestError,
    _control_config_matches_anchor,
    _engine_profile_matches,
    _engine_server_args,
    _normalized_metric_vector,
    _opaque_token,
    _parse_runtime_resolutions,
    _physical_transfer_accounting,
    _profile_payload,
    _scheduler_resolution,
    _scheduler_server_args,
    _validate_async_scoring_evidence,
    _validate_measured_cell,
    _validate_passive_key_evidence,
    _validate_wait_hint_publications,
    _wait_for_campaign_drain,
    _wait_hint_action_counts,
    _wait_hint_event_id,
    build_plan,
    build_serving_payload,
    consume_campaign_sse,
    digest,
    execute_request_schedule,
    extend_causal_tool_resume_manifest,
    file_digest,
    main,
    validate_campaign_event_bindings,
    validate_campaign_manifest,
    validate_causal_online_input,
    validate_request_manifest,
)
from semantic_offload.connector import _campaign_policy_params


def test_singleton_engine_profile_is_scoped_to_bounded_cuda_attribution():
    actual = {
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8192,
        "kv_block_size_tokens": 16,
        "max_model_len": 5120,
        "gpu_memory_utilization": 0.5,
    }

    assert _engine_profile_matches(actual, 320, bounded_prefill_cuda_timing=True)
    assert not _engine_profile_matches(actual, 320, bounded_prefill_cuda_timing=False)
    assert not _engine_profile_matches(
        {**actual, "max_num_seqs": 2},
        320,
        bounded_prefill_cuda_timing=True,
    )


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _request_manifest() -> dict:
    requests = []
    next_token = 1
    for split in ("screen", "heldout"):
        reusable_index = 0

        def add_reusable(family, group_index, phase, phase_index, split=split):
            nonlocal next_token, reusable_index
            tokens = [next_token, next_token + 1]
            next_token += 2
            session_id = f"session-{family}-{split}-{group_index}"
            request = {
                "request_id": f"{family}-{split}-{phase}-{reusable_index}",
                "request_index": reusable_index,
                "phase_index": phase_index,
                "relative_release_ms": 0,
                "depends_on_phase_completion": (
                    None if phase_index == 0 else phase_index - 1
                ),
                "source_id": f"source-{family}-{split}-{group_index}",
                "session_id": session_id,
                "family": family,
                "split": split,
                "context_budget": 2048,
                "generation": 0,
                "content_origin": "frozen_source_archive",
                "arrival_origin": "predeclared_episode_order",
                "wait_origin": "barrier",
                "pressure_origin": "predeclared_one_use_interleave",
                "prompt_token_ids": tokens,
                "prompt_token_digest": digest(tokens),
                "key_chain_digest": digest([session_id, 0]),
                "truncation": {"strategy": "none", "original_token_count": 2},
                "suffix_budget_tokens": 32,
                "episode": "reusable_prefix",
                "phase": phase,
                "task_origin": (
                    (
                        "supplied_question_1"
                        if phase == "warm2"
                        else "supplied_question_0"
                    )
                    if family == "qasper"
                    else "supplied_task_exact_repeat"
                ),
                "serving_inputs": {"session_id": session_id},
            }
            if phase == "revisit":
                request["revisit_origin"] = (
                    "supplied_question_0_after_distinct_question"
                    if family == "qasper"
                    else "exact_repeat_constructed_cache_workload"
                )
            requests.append(request)
            reusable_index += 1

        phase_index = 0
        for group_index in range(4):
            for family in ("qasper", "repobench-p"):
                add_reusable(family, group_index, "warm1", phase_index)
                phase_index += 1
                add_reusable(family, group_index, "warm2", phase_index)
                phase_index += 1
        for phase, group_range in (("pollution", range(4, 8)), ("revisit", range(4))):
            for ordinal, group_index in enumerate(group_range):
                for family in ("qasper", "repobench-p"):
                    add_reusable(family, group_index, phase, phase_index)
                    requests[-1]["relative_release_ms"] = ordinal * 250 + (
                        125 if family == "repobench-p" else 0
                    )
            phase_index += 1

        null_index = 0
        for group_index in range(4):
            for family in ("qasper", "repobench-p"):
                tokens = [next_token, next_token + 1]
                next_token += 2
                requests.append(
                    {
                        "request_id": f"{family}-{split}-null-{null_index}",
                        "request_index": null_index,
                        "phase_index": 0,
                        "relative_release_ms": null_index * 125,
                        "depends_on_phase_completion": None,
                        "source_id": f"null-source-{family}-{split}-{group_index}",
                        "session_id": f"null-session-{family}-{split}-{group_index}",
                        "family": family,
                        "split": split,
                        "context_budget": 2048,
                        "generation": 0,
                        "content_origin": "frozen_source_archive",
                        "arrival_origin": "predeclared_episode_order",
                        "wait_origin": "no_wait",
                        "pressure_origin": "predeclared_null_interleave",
                        "prompt_token_ids": tokens,
                        "prompt_token_digest": digest(tokens),
                        "key_chain_digest": digest(
                            [f"null-session-{family}-{split}-{group_index}", 0]
                        ),
                        "truncation": {
                            "strategy": "none",
                            "original_token_count": 2,
                        },
                        "suffix_budget_tokens": 32,
                        "episode": "nonreuse_null",
                        "phase": "one_use",
                        "serving_inputs": {"session_id": f"null-{null_index}"},
                    }
                )
                null_index += 1
    return {
        "schema": REQUEST_SCHEMA,
        "episode_states": {
            "reusable_prefix": "ready",
            "distinct_topic_shift": "pending",
            "nonreuse_null": "ready",
            "causal_tool_resume": "pending",
        },
        "episode_order_digests": {
            "screen:2048:reusable_prefix": "a" * 64,
            "heldout:2048:reusable_prefix": "b" * 64,
        },
        "requests": requests,
    }


def _campaign(tmp_path: Path) -> tuple[dict, Path]:
    corpus_path = tmp_path / "corpus.json"
    request_path = tmp_path / "requests.json"
    comparisons_path = tmp_path / "required-comparisons.json"
    episode_path = tmp_path / "episode-expansion.md"
    combination_path = tmp_path / "combination-semantics.md"
    original_control_path = tmp_path / "original-control-semantics.json"
    _write(corpus_path, {"schema": "test-corpus", "rows": 32})
    _write(request_path, _request_manifest())
    episode_path.write_text("# frozen episode\n")
    combination_path.write_text("# frozen combination semantics\n")
    _write(
        comparisons_path,
        {
            "schema": "ep-required-comparisons-v3",
            "items": {
                feature: {"settings": REQUIRED_SETTINGS[feature]}
                for feature in FEATURES
            },
            "episode_manifest": {
                "expansion_spec": {"sha256": file_digest(episode_path)}
            },
        },
    )
    _write(
        original_control_path,
        {
            "schema": "ep-original-control-semantics-v1",
            "source_revision": SOURCE_BASELINE,
            "original_semantic_no_prefetch": FROZEN_ORIGINAL_SEMANTIC_CONFIG,
            "original_semantic_dynamic_prefetch": {
                "only_behavior_delta": {"extra_config.prefetch_budget_fraction": 0.05}
            },
        },
    )
    baseline_config = copy.deepcopy(FROZEN_ORIGINAL_SEMANTIC_CONFIG)
    baseline_config["extra_config"]["implementation_revision"] = "1" * 40
    treatment_config = copy.deepcopy(baseline_config)
    treatment_config["extra_config"]["prefix_aware"] = True
    control_configs = {
        "original_semantic_no_prefetch": copy.deepcopy(baseline_config),
        "original_semantic_dynamic_prefetch": {
            **copy.deepcopy(baseline_config),
            "extra_config": {
                **baseline_config["extra_config"],
                "staging_mode": "dynamic_scratch",
                "prefetch_budget_fraction": 0.05,
            },
        },
        "ARC_same_admission_prefetch_staging": {
            **copy.deepcopy(baseline_config),
            "policy": "arc",
        },
        "LRU_same_admission_prefetch_staging": {
            **copy.deepcopy(baseline_config),
            "policy": "lru",
        },
        "best_supported_native_scheduler": {
            **copy.deepcopy(baseline_config),
            "scheduler_backend": "native_sync_fcfs",
        },
    }
    arms = [
        {
            "id": "baseline",
            "split": "screen",
            "features": [],
            "config": baseline_config,
            "arm_config_hash": digest(baseline_config),
            "comparator_arm_id": None,
            "config_delta": [],
            "execution_state": "pending",
        },
        {
            "id": "prefix-on",
            "split": "screen",
            "features": ["E01"],
            "config": treatment_config,
            "arm_config_hash": digest(treatment_config),
            "comparator_arm_id": "baseline",
            "config_delta": [
                {
                    "path": "extra_config.prefix_aware",
                    "before": False,
                    "after": True,
                }
            ],
            "execution_state": "pending",
        },
    ]
    profile = {
        "id": "screen-2048",
        "treatment_arm_ids": ["prefix-on"],
        "comparator_config": baseline_config,
        "workload": {
            "split": "screen",
            "context_tokens": 2048,
            "episode": "reusable_prefix",
        },
        "model_runtime": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "adapter_namespace": "none",
            "layout_namespace": "dense-tp1-bf16-block16",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
        },
        "scheduler_backend": "native_sync_fcfs",
        "resources": {
            "gpu": "RTX 5080",
            "total_gpu_blocks": 320,
            "kv_block_size_tokens": 16,
            "max_model_len": 5120,
            "gpu_memory_utilization": 0.5,
            "enforce_eager": True,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "cpu_bytes": 64 * 1024 * 1024,
            "speculative_cap_rule": "floor(fraction*total_gpu_blocks)",
        },
        "replication_plan": [
            {
                "replicate_id": f"rep-{index + 1}",
                "seed": seed,
                "run_schedule": [
                    {
                        "slot_id": "aa-a",
                        "arm_id": "baseline",
                        "role": "aa_baseline",
                    },
                    {
                        "slot_id": "aa-b",
                        "arm_id": "baseline",
                        "role": "aa_baseline",
                    },
                    {
                        "slot_id": "treatment",
                        "arm_id": "prefix-on",
                        "role": "treatment",
                    },
                ],
                "request_order": "manifest_order",
            }
            for index, seed in enumerate(SEEDS)
        ],
    }
    for replicate in profile["replication_plan"]:
        replicate["order_schedule_digest"] = digest(
            {
                "seed": replicate["seed"],
                "run_schedule": replicate["run_schedule"],
                "request_order": replicate["request_order"],
            }
        )
    manifest = {
        "schema": SCHEMA,
        "source": {
            "extension_revision": SOURCE_BASELINE,
            "runtime_revision": RUNTIME_REVISION,
            "implementation_revision": "1" * 40,
            "implementation_tree": "2" * 40,
            "runtime_source_path": str(tmp_path / "vllm"),
        },
        "port_base": 19000,
        "opaque_identity_namespace": "f" * 64,
        "corpus_selection": {
            "path": str(corpus_path),
            "sha256": file_digest(corpus_path),
        },
        "request_manifest": {
            "path": str(request_path),
            "sha256": file_digest(request_path),
        },
        "protocol_bindings": {
            "required_comparisons": {
                "path": str(comparisons_path),
                "sha256": file_digest(comparisons_path),
            },
            "episode_expansion": {
                "path": str(episode_path),
                "sha256": file_digest(episode_path),
            },
            "combination_semantics": {
                "path": str(combination_path),
                "sha256": file_digest(combination_path),
            },
            "original_control_semantics": {
                "path": str(original_control_path),
                "sha256": file_digest(original_control_path),
            },
        },
        "effective_environment": copy.deepcopy(REQUIRED_EFFECTIVE_ENVIRONMENT),
        "noise_protocol": {
            "seeds": list(SEEDS),
            "aa_pairs": 5,
            "treatment_repetitions": 5,
            "bootstrap_draws": 2000,
            "bootstrap_seed": 20260905,
            "outer_resampling_unit": "independent_run_pair",
            "nested_resampling_unit": "session",
            "metric_contracts": {
                metric: {
                    "unit": (
                        "tokens_per_second"
                        if metric == "throughput_tokens_per_s"
                        else "tokens"
                        if "tokens" in metric
                        else "bytes"
                        if "bytes" in metric
                        else "count"
                        if metric == "preemptions"
                        else "milliseconds"
                    ),
                    "effect_formula": "treatment_minus_control",
                    "beneficial_direction": (
                        "higher"
                        if metric
                        in {"throughput_tokens_per_s", "preservation_reused_tokens"}
                        else "lower"
                    ),
                }
                for metric in REQUIRED_METRICS
            },
        },
        "feature_neutral_settings": copy.deepcopy(REQUIRED_NEUTRAL_SETTINGS),
        "control_references": {
            name: {"config": config, "config_hash": digest(config)}
            for name, config in control_configs.items()
        },
        "arms": arms,
        "coverage": {
            feature: {
                "status": "pending",
                "arm_ids": [],
                "settings": REQUIRED_SETTINGS[feature],
            }
            for feature in FEATURES
        },
        "combination_obligations": [
            {"id": obligation, "status": "pending", "arm_ids": []}
            for obligation in sorted(COMBINATION_OBLIGATIONS)
        ],
        "baseline_profiles": [profile],
        "output_root": str(tmp_path / "results"),
    }
    profile["baseline_profile_hash"] = digest(_profile_payload(manifest, profile))
    manifest_path = tmp_path / "campaign.json"
    _write(manifest_path, manifest)
    return manifest, manifest_path


def _mark_ready(arm: dict) -> None:
    arm["execution_state"] = "ready"
    arm["runner_argv"] = [
        "python",
        "-m",
        "benchmarks.eviction_prefetch_campaign",
        "execute-cell",
        "--manifest={manifest}",
        "--request-manifest={request_manifest}",
        "--cell-id={cell_id}",
        "--profile-id={profile_id}",
        "--port={port}",
        "--output={output_dir}",
    ]


def test_request_manifest_enforces_exact_repeated_and_null_episode_counts():
    result = validate_request_manifest(_request_manifest())
    assert result["request_count"] == 80

    broken = _request_manifest()
    broken["requests"].pop()
    with pytest.raises(ManifestError, match="nonreuse_null must have 8"):
        validate_request_manifest(broken)


def test_causal_tool_resume_extension_preserves_base_and_uses_observed_q1(tmp_path):
    base = _request_manifest()
    for request in base["requests"]:
        request["context_token_span"] = [0, len(request["prompt_token_ids"])]
    frozen = copy.deepcopy(base)

    extended = extend_causal_tool_resume_manifest(
        base, base_sha256="c" * 64, kv_block_size_tokens=1
    )

    assert base == frozen
    assert len(extended["requests"]) == len(base["requests"]) + 64
    causal = [
        request
        for request in extended["requests"]
        if request["episode"] == "causal_tool_resume"
    ]
    resumes = [request for request in causal if request["phase"] == "resume"]
    pollution = [request for request in causal if request["phase"] == "pollution"]
    assert len(resumes) == 16
    assert all(
        request["task_origin"] == "supplied_question_1"
        for request in resumes
        if request["family"] == "qasper"
    )
    assert all("tool_gap_ms" not in request["serving_inputs"] for request in causal)
    assert (
        extended["causal_tool_resume_extension"]["future_suffix_withheld_until_release"]
        is True
    )
    assert extended["causal_tool_resume_extension"]["hint_boundary"] == (
        "final_warm2_completion"
    )
    assert (
        extended["causal_tool_resume_extension"]["hint_acknowledgement_gates_release"]
        is False
    )
    assert all(request["phase_index"] == 16 for request in pollution + resumes)
    assert all(
        request["depends_on_phase_completion"] == 15 for request in pollution + resumes
    )
    assert [request["relative_release_ms"] for request in pollution[:8]] == [
        125 * index for index in range(8)
    ]
    assert [request["relative_release_ms"] for request in resumes[:8]] == [250] * 8

    manifest, manifest_path = _campaign(tmp_path)
    profile = manifest["baseline_profiles"][0]
    cell = build_plan(manifest, manifest_path)["cells"][0]
    resume = resumes[0]
    payload = build_serving_payload(
        request=resume,
        arm=manifest["arms"][0],
        profile=profile,
        cell=cell,
    )
    params = payload["kv_transfer_params"]
    assert (
        params["observed_prefix_boundary_tokens"]
        == resume["known_prefix_boundary_tokens"]
    )
    assert "wait_hint_event_id" not in params
    assert "future_suffix" not in params

    wait_arm = copy.deepcopy(manifest["arms"][0])
    wait_arm["config"]["extra_config"]["prefetch_trigger"] = "known_prefix_wait_hint"
    wait_payload = build_serving_payload(
        request=resume,
        arm=wait_arm,
        profile=profile,
        cell=cell,
    )
    assert wait_payload["kv_transfer_params"]["wait_hint_event_id"] == (
        _wait_hint_event_id(resume, cell)
    )


def test_causal_tool_resume_is_an_executable_32_request_profile(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    base = _request_manifest()
    for request in base["requests"]:
        request["context_token_span"] = [0, len(request["prompt_token_ids"])]
    request_path = Path(manifest["request_manifest"]["path"])
    extended = extend_causal_tool_resume_manifest(
        base,
        base_sha256=file_digest(request_path),
        kv_block_size_tokens=1,
    )
    _write(request_path, extended)
    manifest["request_manifest"]["sha256"] = file_digest(request_path)
    profile = manifest["baseline_profiles"][0]
    profile["workload"]["episode"] = "causal_tool_resume"
    profile["baseline_profile_hash"] = digest(_profile_payload(manifest, profile))
    _write(manifest_path, manifest)

    plan = build_plan(manifest, manifest_path)

    assert plan["cells"][0]["request_proof"]["count"] == 32
    assert {
        binding["episode"] for binding in plan["cells"][0]["request_proof"]["bindings"]
    } == {"causal_tool_resume"}


def test_wait_hint_validation_accepts_identity_bound_zero_action_run():
    cell = {
        "cell_id": "human-cell",
        "policy_run_id": "run",
        "policy_cell_id": "cell",
        "opaque_identity_namespace": "f" * 64,
        "prefetch_trigger": "known_prefix_wait_hint",
        "tool_gap_ms": 50,
        "wait_hint_retention_ms": 2000,
    }
    binding = {
        "request_id": "resume",
        "episode": "causal_tool_resume",
        "phase": "resume",
        "session_id": "session",
        "generation": 0,
        "external_request_id": "sc|run|cell|work|sample-1",
        "known_prefix_boundary_tokens": 16,
        "relative_release_ms": 0,
    }
    event_id = _wait_hint_event_id(binding, cell)
    policy_session = _opaque_token(
        "f" * 64, "s", [binding["session_id"], binding["generation"]]
    )
    payload = {
        "schema": "semantic-known-prefix-wait-hint-v1",
        "event_id": event_id,
        "owner_id": binding["external_request_id"],
        "run_id": "run",
        "cell_id": "cell",
        "session_id": policy_session,
        "generation": 0,
        "observed_prefix_id": "a" * 64,
        "prefix_boundary_tokens": 16,
        "observation_available_ns": 9,
        "input_publication_ns": 10,
        "declared_wait_ms": 50,
        "wait_provenance": "driver_declared_tool_wait_budget",
        "namespace": {
            "model_revision": "model",
            "tokenizer_revision": "tokenizer",
            "adapter_namespace": "none",
            "layout_namespace": "layout",
        },
    }
    publication = {
        "manifest_request_id": "resume",
        "event_id": event_id,
        "hint_boundary_ns": 8,
        "phase_origin_ns": 8,
        "planned_resume_ns": 50_000_008,
        "publication_ns": 10,
        "completed_ns": 11,
        "acknowledgement_late_by_ns": 0,
        "http_status": 200,
        "response": {
            "status": "no_action",
            "reason": "resident_prefix_unavailable",
            "event_id": event_id,
            "run_id": "run",
            "cell_id": "cell",
            "session_id": policy_session,
            "generation": 0,
            "nomination_deadline_ns": 50_000_010,
            "expires_at_ns": 2_050_000_010,
        },
        "error": None,
        "payload": payload,
        "payload_digest": digest(payload),
        "future_suffix_published": False,
    }
    runtime_events = [
        {
            "event": "hint_result",
            "event_id": event_id,
            "status": "no_action",
            "run_id": "run",
            "cell_id": "cell",
            "session_id": policy_session,
            "generation": 0,
        },
        {
            "event": "resume_binding_rejected",
            "event_id": event_id,
            "run_id": "run",
            "cell_id": "cell",
            "session_id": policy_session,
            "generation": 0,
        },
    ]
    result = {
        "wait_hint_publications": [publication] * 8,
        "wait_hint_events": runtime_events,
        "wait_hint_action_counts": _wait_hint_action_counts(
            [publication] * 8, runtime_events
        ),
    }

    _validate_wait_hint_publications(
        result,
        cell,
        [binding] * 8,
        [
            {
                "manifest_request_id": "resume",
                "phase_origin_ns": 8,
                "planned_release_ns": 50_000_008,
                "send_ns": 60_000_010,
            }
        ]
        * 8,
    )

    late_publication = copy.deepcopy(publication)
    late_publication["completed_ns"] = 70_000_008
    late_publication["acknowledgement_late_by_ns"] = 20_000_000
    late_result = {
        "wait_hint_publications": [late_publication] * 8,
        "wait_hint_events": runtime_events,
        "wait_hint_action_counts": _wait_hint_action_counts(
            [late_publication] * 8, runtime_events
        ),
    }
    _validate_wait_hint_publications(
        late_result,
        cell,
        [binding] * 8,
        [
            {
                "manifest_request_id": "resume",
                "phase_origin_ns": 8,
                "planned_release_ns": 50_000_008,
                "send_ns": 60_000_010,
            }
        ]
        * 8,
    )


def test_campaign_plan_binds_profile_replicate_order_and_arm_hashes(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    plan = build_plan(manifest, manifest_path)

    assert plan["schema"] == "ep-campaign-plan-v1"
    assert len(plan["cells"]) == 15
    assert {cell["seed"] for cell in plan["cells"]} == set(SEEDS)
    assert all(cell["baseline_profile_hash"] for cell in plan["cells"])
    assert all(cell["order_schedule_digest"] for cell in plan["cells"])
    assert all(cell["cell_digest"] for cell in plan["cells"])
    assert all(cell["request_proof"]["count"] == 32 for cell in plan["cells"])
    assert all(
        cell["request_proof"]["bindings_digest"]
        == digest(cell["request_proof"]["bindings"])
        for cell in plan["cells"]
    )
    assert plan["plan_digest"]
    assert [cell["port"] for cell in plan["cells"]] == list(range(19000, 19015))
    assert plan["coverage"]["E03"]["status"] == "pending"
    assert plan["cells"][0]["runtime_resolution"]["max_num_seqs"] == 4
    assert plan["cells"][0]["runtime_resolution"]["max_num_batched_tokens"] == 8192
    server_args = _engine_server_args(manifest["baseline_profiles"][0], seed=1103)
    assert server_args[server_args.index("--block-size") + 1] == "16"
    assert "--enforce-eager" in server_args
    assert server_args[server_args.index("--max-num-seqs") + 1] == "4"
    assert server_args[server_args.index("--max-num-batched-tokens") + 1] == "8192"
    assert "--no-async-scheduling" in server_args


def test_ready_runner_expands_real_execute_cell_cli_and_bound_args(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    _write(manifest_path, manifest)
    plan = build_plan(manifest, manifest_path)
    cell = plan["cells"][0]
    assert "execute-cell" in cell["argv"]
    assert f"--request-manifest={manifest['request_manifest']['path']}" in cell["argv"]
    assert f"--profile-id={cell['profile_id']}" in cell["argv"]

    with patch(
        "benchmarks.eviction_prefetch_campaign.execute_serving_cell",
        new_callable=AsyncMock,
    ) as execute:
        assert (
            main(
                [
                    "execute-cell",
                    f"--manifest={manifest_path}",
                    f"--request-manifest={manifest['request_manifest']['path']}",
                    f"--cell-id={cell['cell_id']}",
                    f"--profile-id={cell['profile_id']}",
                    f"--port={cell['port']}",
                    f"--output={tmp_path / 'cell-result.json'}",
                ]
            )
            == 0
        )
    kwargs = execute.await_args.kwargs
    assert kwargs["profile_id"] == cell["profile_id"]
    assert kwargs["request_manifest_path"] == Path(manifest["request_manifest"]["path"])


def test_explicit_native_runtime_variant_requires_exact_frozen_child_binding(
    tmp_path,
):
    manifest, manifest_path = _campaign(tmp_path)
    child = NATIVE_RUNTIME_VARIANTS["wait_hint_utility_v1"]
    manifest["source"].update(
        {
            "runtime_variant": "wait_hint_utility_v1",
            "runtime_variants_binding_sha256": (NATIVE_RUNTIME_VARIANTS_BINDING_SHA256),
            "runtime_revision": child["revision"],
            "runtime_tree": child["tree"],
            "runtime_source_path": child["path"],
            "runtime_wait_hint_utility": True,
            "runtime_patch_sha256": child["patch_sha256"],
            "runtime_native_binary_inventory_sha256": child[
                "native_binary_inventory_sha256"
            ],
        }
    )
    for profile in manifest["baseline_profiles"]:
        profile["baseline_profile_hash"] = digest(_profile_payload(manifest, profile))

    validate_campaign_manifest(manifest, manifest_path)

    manifest["source"]["runtime_tree"] = "0" * 40
    with pytest.raises(ManifestError, match="frozen registry"):
        validate_campaign_manifest(manifest, manifest_path)


def test_ready_wait_hint_arm_rejects_legacy_original149_runtime(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    arm = manifest["arms"][1]
    arm["config"]["extra_config"].update(
        {
            "prefetch_trigger": "known_prefix_wait_hint",
            "prefetch_budget_fraction": 0.05,
        }
    )
    arm["config"]["request_controls"]["tool_gap_ms"] = 50
    arm["arm_config_hash"] = digest(arm["config"])
    _mark_ready(arm)

    with pytest.raises(ManifestError, match="pinned utility child"):
        validate_campaign_manifest(manifest, manifest_path)


def test_undeclared_delta_and_heldout_selection_fail_closed(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    manifest["arms"][1]["config"]["extra_config"]["alpha"] = 0.6
    manifest["arms"][1]["arm_config_hash"] = digest(manifest["arms"][1]["config"])
    with pytest.raises(ManifestError, match="undeclared config difference"):
        validate_campaign_manifest(manifest, manifest_path)


def test_async_scoring_age_and_consumption_evidence_are_causally_joined():
    cell = {
        "policy_run_id": "run",
        "policy_cell_id": "cell",
        "async_scoring_max_snapshot_age_ns": 100,
    }
    published = {
        "event": "published",
        "observation_sequence": 3,
        "request_ids": ["internal-request"],
        "snapshot_captured_ns": 10,
        "observation_available_ns": 10,
        "publication_ns": 20,
        "expires_at_ns": 110,
        "max_snapshot_age_ns": 100,
    }
    consumed = {
        "event": "consumed",
        "observation_sequence": 3,
        "request_ids": ["internal-request"],
        "snapshot_captured_ns": 10,
        "observation_available_ns": 10,
        "expires_at_ns": 110,
        "consumer_decision_ns": 30,
        "consumer_use_ns": 40,
        "request_bindings": [
            {"request_id": "internal-request", "run_id": "run", "cell_id": "cell"}
        ],
    }
    status = {
        "event": "status",
        "pending_items": 0,
        "pending_snapshot_bytes": 0,
        "max_snapshot_age_ns": 100,
        "next_observation_sequence": 4,
    }
    _validate_async_scoring_evidence([published, consumed, status], cell)

    stale = copy.deepcopy(consumed)
    stale["consumer_use_ns"] = 111
    with pytest.raises(ManifestError, match="causal age bounds"):
        _validate_async_scoring_evidence([published, stale, status], cell)

    fabricated_deadline = copy.deepcopy(published)
    fabricated_deadline["expires_at_ns"] = 1_000_000
    forged_consumption = copy.deepcopy(consumed)
    forged_consumption["expires_at_ns"] = 1_000_000
    with pytest.raises(ManifestError, match="derive from capture"):
        _validate_async_scoring_evidence(
            [fabricated_deadline, forged_consumption, status], cell
        )


def test_async_scoring_zero_publication_requires_explicit_drained_accounting():
    cell = {"async_scoring_max_snapshot_age_ns": 100}
    zero_status = {
        "event": "status",
        "pending_items": 0,
        "pending_snapshot_bytes": 0,
        "max_snapshot_age_ns": 100,
        "next_observation_sequence": 0,
    }
    _validate_async_scoring_evidence([zero_status], cell)

    expired_status = {**zero_status, "next_observation_sequence": 1}
    _validate_async_scoring_evidence(
        [
            {
                "event": "publication_rejected",
                "observation_sequence": 0,
                "reason": "snapshot_expired",
                "snapshot_captured_ns": 10,
                "observation_available_ns": 120,
                "expires_at_ns": 110,
                "publication_ns": 130,
            },
            expired_status,
        ],
        cell,
    )
    with pytest.raises(ManifestError, match="zero-action evidence"):
        _validate_async_scoring_evidence([expired_status], cell)


def test_async_scoring_manifest_requires_positive_finite_lifetime(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    manifest["arms"][1]["config"]["extra_config"].update(
        {"scoring_mode": "asynchronous", "async_scoring_max_snapshot_age_ms": 0}
    )
    with pytest.raises(ManifestError, match="positive snapshot age"):
        validate_campaign_manifest(manifest, manifest_path)


def test_profile_rejects_prompt_that_exceeds_model_or_kv_block_budget(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    profile = manifest["baseline_profiles"][0]
    profile["resources"]["max_model_len"] = 1
    with pytest.raises(ManifestError, match="exceeds max_model_len"):
        validate_campaign_manifest(manifest, manifest_path)


def test_capacity_indexed_model_limit_accepts_320_at_5120_and_rejects_8192(
    tmp_path,
):
    manifest, manifest_path = _campaign(tmp_path)
    profile = manifest["baseline_profiles"][0]

    assert profile["resources"]["total_gpu_blocks"] == 320
    assert profile["resources"]["max_model_len"] == 5120
    validate_campaign_manifest(manifest, manifest_path)

    profile["resources"]["max_model_len"] = 8192
    with pytest.raises(ManifestError, match="physical GPU KV token capacity"):
        validate_campaign_manifest(manifest, manifest_path)


def test_capacity_indexed_model_limit_keeps_640_at_8192(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    profile = manifest["baseline_profiles"][0]
    profile["resources"].update({"total_gpu_blocks": 640, "max_model_len": 8192})
    profile["baseline_profile_hash"] = digest(_profile_payload(manifest, profile))

    plan = build_plan(manifest, manifest_path)

    assert plan["cells"][0]["runtime_resolution"]["max_model_len"] == 8192
    assert plan["cells"][0]["runtime_resolution"]["num_gpu_blocks_override"] == 640


def test_passive_key_validator_requires_ordered_positions_lifecycle_and_first_use():
    request = {"prompt_token_ids": list(range(32))}
    positions = [
        {
            "key": f"key-{index}",
            "group_idx": 0,
            "chunk_index": index,
            "token_start": index * 16,
            "token_end": (index + 1) * 16,
            "complete_chunk": True,
            "provenance": "scheduler_request_group_state",
            "state": "planned",
        }
        for index in range(2)
    ]
    commit = {
        **positions[0],
        "state": "committed_cpu_ready",
        "sequence": 0,
        "committed_at_ns": 9,
        "completion_observed_at_ns": 10,
        "generation": 0,
        "run_id": "run",
        "cell_id": "cell",
    }
    eviction = {
        **positions[0],
        "state": "committed_cpu_eviction",
        "sequence": 1,
        "observed_at_ns": 20,
        "generation": 0,
        "run_id": "run",
        "cell_id": "cell",
        "prior_commit_sequence": 0,
        "prior_commit_ns": 9,
    }
    event = {
        "ordered_key_positions_complete": True,
        "ordered_key_positions": positions,
        "key_lifecycle": {
            "cpu_commits": [commit],
            "cpu_evictions": [eviction],
        },
        "first_admission": {
            "admission_index": 0,
            "episode_id": 0,
            "admitted_at_ns": 30,
            "provenance": "scheduler_update_state_after_alloc",
            "local_keys": [],
            "planned_load_keys": [positions[0]],
            "completed_load_keys": [positions[0]],
            "recomputed_keys": [positions[1]],
        },
    }

    assert (
        _validate_passive_key_evidence(
            event,
            request,
            external_id="request",
            tokens_per_block=16,
            blocks_per_chunk=1,
            kv_group_count=1,
        )
        == []
    )

    event["ordered_key_positions"] = list(reversed(positions))
    assert "request:ordered_key_position_invalid" in _validate_passive_key_evidence(
        event,
        request,
        external_id="request",
        tokens_per_block=16,
        blocks_per_chunk=1,
        kv_group_count=1,
    )


def test_passive_key_validator_rejects_empty_complete_map_for_long_prompt():
    errors = _validate_passive_key_evidence(
        {
            "ordered_key_positions_complete": True,
            "ordered_key_positions": [],
        },
        {
            "prompt_token_count": 2048,
            "context_token_span": [24, 2072],
        },
        external_id="request",
        tokens_per_block=16,
        blocks_per_chunk=1,
        kv_group_count=1,
    )

    assert "request:ordered_key_full_prompt_coverage_invalid" in errors


def test_replication_plan_requires_real_aa_slots_and_recomputed_order_digest(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    replicate = manifest["baseline_profiles"][0]["replication_plan"][0]
    replicate["run_schedule"][1]["role"] = "treatment"
    with pytest.raises(ManifestError, match="exactly two A/A"):
        validate_campaign_manifest(manifest, manifest_path)


def test_ready_treatment_requires_all_five_repeats(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    for replicate in manifest["baseline_profiles"][0]["replication_plan"][1:]:
        replicate["run_schedule"][-1]["arm_id"] = "baseline"
        replicate["order_schedule_digest"] = digest(
            {
                "seed": replicate["seed"],
                "run_schedule": replicate["run_schedule"],
                "request_order": replicate["request_order"],
            }
        )
    with pytest.raises(ManifestError, match="every treatment arm exactly once"):
        validate_campaign_manifest(manifest, manifest_path)


def test_ready_feature_and_combination_require_concrete_realizations(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    manifest["coverage"]["E02"].update({"status": "ready", "arm_ids": ["baseline"]})
    with pytest.raises(ManifestError, match="does not realize"):
        validate_campaign_manifest(manifest, manifest_path)


def test_ready_arm_duplicate_replicate_and_false_interaction_fail_closed(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    unscheduled = copy.deepcopy(manifest["arms"][1])
    unscheduled["id"] = "unscheduled-prefix"
    _mark_ready(unscheduled)
    manifest["arms"].append(unscheduled)
    with pytest.raises(ManifestError, match="globally ready arm"):
        validate_campaign_manifest(manifest, manifest_path)

    manifest, manifest_path = _campaign(tmp_path)
    replication = manifest["baseline_profiles"][0]["replication_plan"]
    replication[1]["replicate_id"] = replication[0]["replicate_id"]
    with pytest.raises(ManifestError, match="replicate ids must be unique"):
        validate_campaign_manifest(manifest, manifest_path)

    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    manifest["arms"][0]["features"] = ["E01", "E02", "E09"]
    obligation = next(
        item
        for item in manifest["combination_obligations"]
        if item["id"] == "interaction:E01+E02+E09"
    )
    obligation.update({"status": "ready", "arm_ids": ["baseline"]})
    with pytest.raises(ManifestError, match="exact behavior deltas"):
        validate_campaign_manifest(manifest, manifest_path)

    manifest, manifest_path = _campaign(tmp_path)
    obligation = manifest["combination_obligations"][0]
    obligation["status"] = "ready"
    with pytest.raises(ManifestError, match="requires concrete arms"):
        validate_campaign_manifest(manifest, manifest_path)

    manifest, manifest_path = _campaign(tmp_path)
    replicate = manifest["baseline_profiles"][0]["replication_plan"][0]
    replicate["order_schedule_digest"] = "0" * 64
    with pytest.raises(ManifestError, match="order schedule digest mismatch"):
        validate_campaign_manifest(manifest, manifest_path)

    manifest, manifest_path = _campaign(tmp_path)
    profile = manifest["baseline_profiles"][0]
    profile["resources"]["total_gpu_blocks"] = 1
    with pytest.raises(ManifestError, match="exceeds GPU KV block budget"):
        validate_campaign_manifest(manifest, manifest_path)

    manifest, manifest_path = _campaign(tmp_path)
    manifest["arms"][1]["split"] = "heldout"
    with pytest.raises(ManifestError, match="heldout"):
        validate_campaign_manifest(manifest, manifest_path)


def test_structured_heldout_lock_rejects_empty_self_declared_screen_evidence(tmp_path):
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir()
    screen_manifest, screen_manifest_path = _campaign(screen_dir)
    screen_plan = {
        "schema": "ep-campaign-plan-v1",
        "manifest_sha256": file_digest(screen_manifest_path),
        "source": screen_manifest["source"],
        "coverage": screen_manifest["coverage"],
        "combination_obligations": screen_manifest["combination_obligations"],
        "noise_protocol": screen_manifest["noise_protocol"],
        "cells": [],
    }
    screen_plan["plan_digest"] = digest(screen_plan)
    screen_plan_path = tmp_path / "empty-screen-plan.json"
    _write(screen_plan_path, screen_plan)
    screen_result = {
        "schema": "ep-screen-selection-result-v1",
        "valid": True,
        "selection_split": "screen",
        "screen_plan_sha256": file_digest(screen_plan_path),
        "selection_rule": "fake",
        "selected_configs": [],
        "completed_at_ns": 1,
        "cell_results": [],
    }
    screen_result_path = tmp_path / "fake-screen-result.json"
    _write(screen_result_path, screen_result)

    heldout_dir = tmp_path / "heldout"
    heldout_dir.mkdir()
    manifest, manifest_path = _campaign(heldout_dir)
    manifest["source"] = copy.deepcopy(screen_manifest["source"])
    manifest["request_manifest"] = copy.deepcopy(screen_manifest["request_manifest"])
    manifest["protocol_bindings"] = copy.deepcopy(screen_manifest["protocol_bindings"])
    manifest["noise_protocol"] = copy.deepcopy(screen_manifest["noise_protocol"])
    screen_arm = copy.deepcopy(manifest["arms"][1])
    screen_arm["id"] = "prefix-screen"
    manifest["arms"].append(screen_arm)
    manifest["arms"][1]["split"] = "heldout"
    manifest["arms"][1]["selected_from_screen_arm_id"] = "prefix-screen"
    manifest["heldout_selection_lock"] = {
        "schema": "ep-heldout-selection-lock-v1",
        "selection_split": "screen",
        "frozen_before_heldout": True,
        "screen_manifest": {
            "path": str(screen_manifest_path),
            "sha256": file_digest(screen_manifest_path),
        },
        "screen_plan": {
            "path": str(screen_plan_path),
            "sha256": file_digest(screen_plan_path),
        },
        "screen_result": {
            "path": str(screen_result_path),
            "sha256": file_digest(screen_result_path),
        },
        "selected_configs": [],
        "selection_rule": "fake",
        "frozen_at_ns": 2,
        "first_heldout_launch": {"cell_id": "fake", "not_before_ns": 2},
    }
    with pytest.raises(ManifestError, match="screen plan schema/digest"):
        validate_campaign_manifest(manifest, manifest_path)


def test_measured_rows_require_authoritative_plan_and_metric_bindings(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    bindings = []
    for arm in manifest["arms"]:
        for replicate in range(5):
            result_path = tmp_path / f"fake-{arm['id']}-{replicate}.json"
            _write(
                result_path,
                {
                    "schema": "ep-campaign-cell-result-v1",
                    "valid": True,
                    "arm_id": arm["id"],
                    "arm_config_hash": arm["arm_config_hash"],
                    "replicate_id": f"rep-{replicate}",
                    "campaign_manifest_sha256": "0" * 64,
                    "plan_digest": "0" * 64,
                    "cell_digest": "0" * 64,
                },
            )
            bindings.append(
                {"path": str(result_path), "sha256": file_digest(result_path)}
            )
    manifest["coverage"]["E01"].update(
        {
            "status": "measured",
            "arm_ids": ["baseline", "prefix-on"],
            "result_bindings": bindings,
        }
    )
    with pytest.raises(ManifestError, match="measured result plan binding"):
        validate_campaign_manifest(manifest, manifest_path)


def test_screen_selection_reuses_authoritative_measured_cell_validator(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    plan = build_plan(manifest, manifest_path)
    cell = plan["cells"][0]
    result = {
        "schema": "ep-campaign-cell-result-v1",
        "valid": True,
        "campaign_manifest_sha256": file_digest(manifest_path),
        "plan_digest": plan["plan_digest"],
        **{
            field: cell[field]
            for field in (
                "cell_id",
                "cell_digest",
                "profile_id",
                "baseline_profile_hash",
                "replicate_id",
                "seed",
                "order_schedule_digest",
                "arm_id",
                "arm_config_hash",
            )
        },
        "metric_contract_digest": digest(
            manifest["noise_protocol"]["metric_contracts"]
        ),
        "effective_environment": copy.deepcopy(cell["effective_environment"]),
        "resolved_scheduler": copy.deepcopy(cell["scheduler_resolution"]),
        "resolved_scheduler_runtime": copy.deepcopy(cell["runtime_resolution"]),
        "deployed_extension": {},
        "deployed_runtime": {},
        "normalized_metrics": {},
    }
    with pytest.raises(ManifestError, match="deployed source/runtime"):
        _validate_measured_cell(
            result,
            plan,
            source=manifest["source"],
            metric_contracts=manifest["noise_protocol"]["metric_contracts"],
            execution_manifest_sha256=file_digest(manifest_path),
        )

    result["deployed_extension"] = {
        "revision": manifest["source"]["implementation_revision"],
        "tree": manifest["source"]["implementation_tree"],
        "clean": True,
    }
    result["deployed_runtime"] = {
        "revision": manifest["source"]["runtime_revision"],
        "clean": True,
    }
    with pytest.raises(ManifestError, match="normalized metric vector"):
        _validate_measured_cell(
            result,
            plan,
            source=manifest["source"],
            metric_contracts=manifest["noise_protocol"]["metric_contracts"],
            execution_manifest_sha256=file_digest(manifest_path),
        )
    result["normalized_metrics"] = {
        metric: {
            "value": "fabricated",
            "unit": contract["unit"],
            "source": "fabricated",
            "available": True,
            "unavailable_reason": None,
        }
        for metric, contract in manifest["noise_protocol"]["metric_contracts"].items()
    }
    with pytest.raises(ManifestError, match="unavailable or unbound"):
        _validate_measured_cell(
            result,
            plan,
            source=manifest["source"],
            metric_contracts=manifest["noise_protocol"]["metric_contracts"],
            execution_manifest_sha256=file_digest(manifest_path),
        )


def test_reversed_leaveout_labels_cannot_replace_behavioral_ablation(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    manifest["arms"][0]["features"] = list(FEATURES)
    manifest["arms"][1]["features"] = [
        feature for feature in FEATURES if feature != "E01"
    ]
    manifest["arms"][1]["leave_one_out"] = "E01"
    obligation = next(
        item
        for item in manifest["combination_obligations"]
        if item["id"] == "leave_one_out:E01"
    )
    obligation.update(
        {
            "status": "ready",
            "arm_ids": ["prefix-on"],
            "full_bundle_arm_id": "baseline",
        }
    )
    with pytest.raises(ManifestError, match="lacks its leave-out arm"):
        validate_campaign_manifest(manifest, manifest_path)


def test_treatment_cannot_relabel_itself_as_named_control(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    manifest["arms"][1]["control_id"] = "original_semantic_no_prefetch"
    obligation = next(
        item
        for item in manifest["combination_obligations"]
        if item["id"] == "control:original_semantic_no_prefetch"
    )
    obligation.update({"status": "ready", "arm_ids": ["prefix-on"]})
    with pytest.raises(ManifestError, match="lacks its named control"):
        validate_campaign_manifest(manifest, manifest_path)


def test_treatment_cannot_redirect_original_control_reference(tmp_path):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    treatment = manifest["arms"][1]
    treatment["control_id"] = "original_semantic_no_prefetch"
    manifest["control_references"]["original_semantic_no_prefetch"] = {
        "config": copy.deepcopy(treatment["config"]),
        "config_hash": digest(treatment["config"]),
    }
    obligation = next(
        item
        for item in manifest["combination_obligations"]
        if item["id"] == "control:original_semantic_no_prefetch"
    )
    obligation.update({"status": "ready", "arm_ids": ["prefix-on"]})
    for profile in manifest["baseline_profiles"]:
        profile["baseline_profile_hash"] = digest(_profile_payload(manifest, profile))

    with pytest.raises(ManifestError, match="accepted original behavior"):
        validate_campaign_manifest(manifest, manifest_path)


@pytest.mark.parametrize("policy", ["arc", "lru"])
def test_arc_lru_profiles_keep_their_own_exact_aa_comparator(tmp_path, policy):
    manifest, manifest_path = _campaign(tmp_path)
    baseline = manifest["arms"][0]
    treatment = manifest["arms"][1]
    baseline["config"] = copy.deepcopy(
        manifest["control_references"][
            f"{policy.upper()}_same_admission_prefetch_staging"
        ]["config"]
    )
    baseline["arm_config_hash"] = digest(baseline["config"])
    treatment["config"] = copy.deepcopy(baseline["config"])
    treatment["config"]["extra_config"]["prefix_aware"] = True
    treatment["arm_config_hash"] = digest(treatment["config"])
    profile = manifest["baseline_profiles"][0]
    profile["comparator_config"] = copy.deepcopy(baseline["config"])
    profile["baseline_profile_hash"] = digest(_profile_payload(manifest, profile))

    validate_campaign_manifest(manifest, manifest_path)


def test_arc_semantic_prefetch_control_requires_real_shared_scorer():
    anchor = copy.deepcopy(FROZEN_ORIGINAL_SEMANTIC_CONFIG)
    anchor["extra_config"]["prefetch_budget_fraction"] = 0.05
    control = copy.deepcopy(anchor)
    control["policy"] = "arc"
    assert not _control_config_matches_anchor(
        "ARC_same_admission_prefetch_staging", control, anchor=anchor
    )

    control["extra_config"]["shared_prefetch_scorer"] = True
    assert _control_config_matches_anchor(
        "ARC_same_admission_prefetch_staging", control, anchor=anchor
    )


def test_measured_cell_rejects_incomplete_negative_summary_without_raw_proof(
    tmp_path,
):
    manifest, manifest_path = _campaign(tmp_path)
    _mark_ready(manifest["arms"][0])
    _mark_ready(manifest["arms"][1])
    plan = build_plan(manifest, manifest_path)
    cell = plan["cells"][0]
    contracts = manifest["noise_protocol"]["metric_contracts"]
    result = {
        **{
            field: cell[field]
            for field in (
                "cell_id",
                "cell_digest",
                "profile_id",
                "baseline_profile_hash",
                "replicate_id",
                "seed",
                "order_schedule_digest",
                "arm_id",
                "arm_config_hash",
            )
        },
        "schema": "ep-campaign-cell-result-v1",
        "valid": True,
        "campaign_manifest_sha256": file_digest(manifest_path),
        "plan_digest": plan["plan_digest"],
        "metric_contract_digest": digest(contracts),
        "effective_environment": copy.deepcopy(cell["effective_environment"]),
        "resolved_scheduler": copy.deepcopy(cell["scheduler_resolution"]),
        "resolved_scheduler_runtime": copy.deepcopy(cell["runtime_resolution"]),
        "deployed_extension": {
            "revision": manifest["source"]["implementation_revision"],
            "tree": manifest["source"]["implementation_tree"],
            "clean": True,
        },
        "deployed_runtime": {
            "revision": manifest["source"]["runtime_revision"],
            "clean": True,
        },
        "request_manifest_sha256": cell["request_proof"]["request_manifest_sha256"],
        "requests": [
            {
                "manifest_request_id": cell["request_proof"][
                    "ordered_manifest_request_ids"
                ][0],
                "success": True,
                "terminal_complete": False,
                "done_seen": False,
                "ttft_ms": -1,
                "output_token_ids": [-1],
            }
        ],
        "preservation_mapping": {
            "preservation_reason": "authoritative_request_events",
            "preservation_recomputed_tokens": -3,
            "preservation_reused_tokens": -4,
        },
        "metrics_delta": {
            "vllm:kv_offload_load_bytes_total": -5,
            "vllm:kv_offload_store_bytes_total": -6,
            "vllm:num_preemptions_total": -7,
        },
        "episode_makespan_ms": 1,
        "drain": {"status": "complete"},
        "campaign_binding_errors": [],
    }
    result["normalized_metrics"] = _normalized_metric_vector(
        metric_contracts=contracts,
        requests=result["requests"],
        preservation=result["preservation_mapping"],
        metric_deltas=result["metrics_delta"],
        episode_makespan_ms=1,
    )

    with pytest.raises(ManifestError, match="request set/order differs from plan"):
        _validate_measured_cell(
            result,
            plan,
            source=manifest["source"],
            metric_contracts=contracts,
            execution_manifest_sha256=file_digest(manifest_path),
        )


def _causal_record() -> dict:
    return {
        "schema": CAUSAL_SCHEMA,
        "schema_version": 1,
        "engine_run_id": "run-1",
        "model_revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "adapter_namespace": "none",
        "layout_namespace": "tp1-bf16",
        "request_id": "req-1",
        "session_id": "session-1",
        "generation": 1,
        "scheduler_step": 4,
        "observation_sequence": 9,
        "key_chain_digest": "a" * 64,
        "observation_available_ns": 10,
        "input_publication_ns": 11,
        "consumer_decision_ns": 12,
        "consumer_use_ns": 13,
        "feature_provenance": "lookup_history",
        "cost_sample_provenance": "configured_group_cost",
        "candidate_config_hash": "b" * 64,
    }


def test_causal_input_requires_publication_before_decision_and_no_future_label():
    validate_causal_online_input(_causal_record())

    record = _causal_record()
    record["input_publication_ns"] = 14
    with pytest.raises(ManifestError, match="order violated"):
        validate_causal_online_input(record)

    record = _causal_record()
    record["future_reuse"] = True
    with pytest.raises(ManifestError, match="future fields"):
        validate_causal_online_input(record)


def test_serving_payload_binds_causal_metadata_without_future_phase_labels():
    request = _request_manifest()["requests"][0]
    request["key_chain_digest"] = "c" * 64
    request["serving_inputs"].update(
        {
            "admission_cost_per_byte": 2.5,
            "admission_cost_provenance": "baseline-profile-cost-v1",
        }
    )
    arm = {
        "config": {
            "policy": "semantic-mean",
            "extra_config": {"admission_mode": "historical_frequency_cost_per_byte"},
            "request_controls": {"max_offload_tokens": "known_shared_prefix_tokens"},
        }
    }
    request["context_token_span"] = [2, 10]
    profile = {
        "workload": {"episode": "reusable_prefix"},
        "resources": {"kv_block_size_tokens": 16},
        "model_runtime": {"model": "model"},
    }
    cell = {
        "profile_id": "screen-2048",
        "replicate_id": "rep-1",
        "cell_id": "cell-1",
        "schedule_role": "treatment",
        "seed": 1103,
        "arm_config_hash": "d" * 64,
        "policy_run_id": "ropaque",
        "policy_cell_id": "copaque",
        "policy_workload_id": "wopaque",
        "opaque_identity_namespace": "f" * 64,
    }

    payload = build_serving_payload(
        request=request, arm=arm, profile=profile, cell=cell
    )

    params = payload["kv_transfer_params"]
    assert params["candidate_config_hash"] == "d" * 64
    assert params["max_offload_tokens"] == 10
    assert params["admission_cost_per_byte"] == 2.5
    assert "sample_index" not in params
    assert "benchmark_client_sequence" not in params
    assert all(
        alias not in json.dumps(params[key])
        for key in (
            "external_request_id",
            "run_id",
            "cell_id",
            "session_id",
        )
        for alias in ("treatment", "reusable_prefix", "screen", "warm", "pollution")
    )
    assert (
        params["admission_cost_observation_available_ns"]
        <= params["admission_cost_input_publication_ns"]
    )
    assert not {"episode", "phase", "revisit_origin", "future_reuse"}.intersection(
        params
    )
    policy_params = _campaign_policy_params(params)
    assert "campaign_provenance" not in policy_params
    assert not {"pressure_origin", "family", "split", "source_id"}.intersection(
        policy_params
    )
    for leaked_field in ("pressure_origin", "family", "split"):
        leaked = dict(params)
        leaked[leaked_field] = "forbidden-role"
        with pytest.raises(ValueError, match="unexpected policy-visible"):
            _campaign_policy_params(leaked)


def test_request_schedule_waits_for_prior_phase_completion():
    requests = [
        {
            "request_id": "first",
            "request_index": 0,
            "phase_index": 0,
            "relative_release_ms": 0,
        },
        {
            "request_id": "second",
            "request_index": 1,
            "phase_index": 1,
            "relative_release_ms": 0,
        },
    ]
    order = []

    async def send(request):
        order.append(f"start:{request['request_id']}")
        await asyncio.sleep(0)
        order.append(f"end:{request['request_id']}")
        return {"success": True}

    results = asyncio.run(execute_request_schedule(requests, send))

    assert order == ["start:first", "end:first", "start:second", "end:second"]
    assert results[0]["completion_ns"] <= results[1]["eligibility_ns"]


def test_request_schedule_runs_async_wait_boundary_before_resume_eligibility():
    requests = [
        {
            "request_id": "warm",
            "request_index": 0,
            "phase_index": 0,
            "relative_release_ms": 0,
        },
        {
            "request_id": "resume",
            "request_index": 1,
            "phase_index": 1,
            "relative_release_ms": 0,
        },
    ]
    boundary_completed_ns = None

    async def before_phase(phase_index, _requests, _phase_origin_ns):
        nonlocal boundary_completed_ns
        if phase_index == 1:
            await asyncio.sleep(0)
            boundary_completed_ns = time.monotonic_ns()

    async def send(_request):
        return {"success": True}

    results = asyncio.run(
        execute_request_schedule(requests, send, before_phase=before_phase)
    )

    assert boundary_completed_ns is not None
    assert results[1]["eligibility_ns"] >= boundary_completed_ns


def test_request_schedule_freezes_activity_origin_before_nonblocking_hook():
    requests = [
        {
            "request_id": "warm",
            "request_index": 0,
            "phase_index": 0,
            "relative_release_ms": 0,
        },
        {
            "request_id": "pollution",
            "request_index": 1,
            "phase_index": 1,
            "relative_release_ms": 0,
        },
        {
            "request_id": "resume",
            "request_index": 2,
            "phase_index": 1,
            "relative_release_ms": 250,
        },
    ]
    hook_origin_ns = None

    async def before_phase(phase_index, phase_requests, phase_origin_ns):
        nonlocal hook_origin_ns
        if phase_index == 1:
            hook_origin_ns = phase_origin_ns
            return {
                request["request_id"]: phase_origin_ns for request in phase_requests
            }
        return None

    async def send(_request):
        return {"success": True}

    results = asyncio.run(
        execute_request_schedule(requests, send, before_phase=before_phase)
    )

    assert hook_origin_ns == results[0]["completion_ns"]
    assert results[1]["phase_origin_ns"] == hook_origin_ns
    assert results[2]["phase_origin_ns"] == hook_origin_ns
    assert results[1]["planned_release_ns"] == hook_origin_ns
    assert results[2]["planned_release_ns"] == hook_origin_ns + 250_000_000


def test_campaign_event_binding_rejects_wrong_arm_hash_and_source_provenance():
    request = _request_manifest()["requests"][0]
    request["key_chain_digest"] = "c" * 64
    request["prompt_token_digest"] = digest(request["prompt_token_ids"])
    cell = {
        "profile_id": "screen-2048",
        "replicate_id": "rep-1",
        "cell_id": "cell-1",
        "schedule_role": "treatment",
        "arm_config_hash": "d" * 64,
        "policy_run_id": "ropaque",
        "policy_cell_id": "copaque",
        "policy_workload_id": "wopaque",
        "opaque_identity_namespace": "f" * 64,
    }
    profile = {
        "workload": {"episode": "reusable_prefix"},
        "resources": {"kv_block_size_tokens": 16},
        "model_runtime": {
            "model_revision": "model-rev",
            "tokenizer_revision": "tokenizer-rev",
            "adapter_namespace": "none",
            "layout_namespace": "tp1-bf16",
        },
    }
    policy_sample = int(_opaque_token("f" * 64, "", request["request_id"], 12), 16)
    external_id = f"sc|ropaque|copaque|wopaque|sample-{policy_sample}"
    policy_session = _opaque_token(
        "f" * 64, "s", [request["session_id"], request["generation"]]
    )
    event = {
        "external_request_id": external_id,
        **{
            event_field: request[request_field]
            for event_field, request_field in (
                ("manifest_request_id", "request_id"),
                ("manifest_request_index", "request_index"),
                ("content_origin", "content_origin"),
                ("arrival_origin", "arrival_origin"),
                ("wait_origin", "wait_origin"),
                ("pressure_origin", "pressure_origin"),
                ("source_id", "source_id"),
                ("family", "family"),
                ("split", "split"),
                ("generation", "generation"),
                ("prompt_token_digest", "prompt_token_digest"),
            )
        },
        "human_cell_id": "cell-1",
        "human_profile_id": "screen-2048",
        "schedule_role": "treatment",
        "admission_decisions": [
            {
                "candidate_config_hash": "d" * 64,
                "engine_run_id": "ropaque",
                "session_id": policy_session,
                "generation": request["generation"],
                "key_chain_digest": request["key_chain_digest"],
                "model_revision": "model-rev",
                "tokenizer_revision": "tokenizer-rev",
                "adapter_namespace": "none",
                "layout_namespace": "tp1-bf16",
            }
        ],
        "session_id": policy_session,
        "prefetch_attribution": {"provenance": {"config": {"cache_block_size": 16}}},
    }
    assert (
        validate_campaign_event_bindings([event], [request], cell=cell, profile=profile)
        == []
    )

    event["source_id"] = "wrong"
    event["admission_decisions"][0]["candidate_config_hash"] = "e" * 64
    errors = validate_campaign_event_bindings(
        [event], [request], cell=cell, profile=profile
    )
    assert f"{external_id}:source_id_mismatch" in errors
    assert f"{external_id}:candidate_config_hash_mismatch" in errors


def test_sse_requires_exact_tokens_and_explicit_terminal_marker():
    async def lines(*values):
        for value in values:
            yield value.encode()

    complete = asyncio.run(
        consume_campaign_sse(
            lines(
                'data: {"choices":[{"text":"x","token_ids":[7]}]}',
                'data: {"choices":[{"text":"","finish_reason":"length"}]}',
                "data: [DONE]",
            )
        )
    )
    assert complete["terminal_complete"] is True
    assert complete["output_token_ids"] == [7]
    assert len(complete["token_receipt_ns"]) == 1

    truncated = asyncio.run(
        consume_campaign_sse(lines('data: {"choices":[{"text":"x","token_ids":[7]}]}'))
    )
    assert truncated["terminal_complete"] is False

    with pytest.raises(ValueError, match="malformed stream token_ids"):
        asyncio.run(
            consume_campaign_sse(lines('data: {"choices":[{"token_ids":[7,"bad"]}]}'))
        )


def test_sse_allows_zero_tokens_only_with_explicit_terminal_outcome():
    async def lines():
        yield b'data: {"choices":[{"text":"","finish_reason":"stop"}]}'
        yield b"data: [DONE]"

    result = asyncio.run(consume_campaign_sse(lines()))
    assert result["terminal_complete"] is True
    assert result["output_token_ids"] == []

    async def empty_finish():
        yield b'data: {"choices":[{"finish_reason":""}]}'
        yield b"data: [DONE]"

    with pytest.raises(ValueError, match="finish_reason"):
        asyncio.run(consume_campaign_sse(empty_finish()))

    async def after_done():
        yield b"data: [DONE]"
        yield b'data: {"choices":[]}'

    with pytest.raises(ValueError, match="after DONE"):
        asyncio.run(consume_campaign_sse(after_done()))


def test_normalized_metric_vector_materializes_every_contract_metric():
    contracts = {
        metric: {
            "unit": "unit",
            "effect_formula": "ratio",
            "beneficial_direction": "lower",
        }
        for metric in REQUIRED_METRICS
    }
    vector = _normalized_metric_vector(
        metric_contracts=contracts,
        requests=[{"ttft_ms": 2.0, "output_token_ids": [1, 2]}],
        preservation={
            "preservation_recomputed_tokens": 3,
            "preservation_reused_tokens": 4,
        },
        metric_deltas={
            "vllm:kv_offload_load_bytes_total": 5.0,
            "vllm:kv_offload_store_bytes_total": 6.0,
            "vllm:num_preemptions_total": 0.0,
        },
        episode_makespan_ms=1000.0,
    )
    assert set(vector) == REQUIRED_METRICS
    assert all(entry["available"] for entry in vector.values())
    assert vector["throughput_tokens_per_s"]["value"] == 2.0


def _speculative_event(job_id=12, *, consumed=0, discarded=128):
    return {
        "run_id": "run",
        "cell_id": "cell",
        "transfer_job_statuses": [
            {
                "job_id": 10,
                "is_prefetch": False,
                "direction": "load",
                "status": "success",
                "success": True,
                "cancelled": False,
                "contributors": 1,
                "expected_contributors": 1,
                "bytes": 64,
                "time_s": 0.001,
            }
        ],
        "speculative_transfer_job_statuses": [
            {
                "job_id": job_id,
                "is_prefetch": True,
                "direction": "load",
                "status": "success",
                "success": True,
                "cancelled": False,
                "contributors": 1,
                "expected_contributors": 1,
                "bytes": 128,
                "time_s": 0.001,
            }
        ],
        "speculative_transfer_accounting": {
            "job_count": 1,
            "successful_job_count": 1,
            "failed_job_count": 0,
            "submitted_bytes": 128,
            "completed_bytes": 128,
            "consumed_bytes": consumed,
            "discarded_bytes": discarded,
            "released_reservation_bytes": discarded,
            "uncompleted_submitted_bytes": 0,
            "unconsumed_completed_bytes": 128 - consumed - discarded,
            "actual_contributors": 1,
            "expected_contributors": 1,
            "device_time_s": 0.001,
        },
    }


@pytest.mark.parametrize(("consumed", "discarded"), [(128, 0), (0, 128)])
def test_physical_load_bytes_include_speculative_completion_exactly_once(
    consumed, discarded
):
    accounting = _physical_transfer_accounting(
        [_speculative_event(consumed=consumed, discarded=discarded)],
        {"vllm:kv_offload_load_bytes_total": 192.0},
    )

    assert accounting["demand_load_bytes"] == 64.0
    assert accounting["prometheus_physical_load_bytes"] == 192.0
    assert accounting["speculative_completed_h2d_bytes"] == 128
    assert accounting["speculative_consumed_h2d_bytes"] == consumed
    assert accounting["speculative_discarded_h2d_bytes"] == discarded
    assert accounting["total_physical_h2d_bytes"] == 192.0


def test_physical_transfer_accounting_rejects_duplicate_speculative_job():
    duplicate = _speculative_event()
    duplicate_without_demand = copy.deepcopy(duplicate)
    duplicate_without_demand["transfer_job_statuses"] = []

    with pytest.raises(ManifestError, match="duplicated"):
        _physical_transfer_accounting(
            [duplicate, duplicate_without_demand],
            {"vllm:kv_offload_load_bytes_total": 192.0},
        )


def test_physical_transfer_accounting_rejects_omitted_speculative_counter():
    with pytest.raises(ManifestError, match="does not reconcile"):
        _physical_transfer_accounting(
            [_speculative_event()],
            {"vllm:kv_offload_load_bytes_total": 64.0},
        )


def test_scheduler_backend_and_source_identity_fail_closed(tmp_path):
    assert _scheduler_server_args("native_sync_fcfs") == ["--no-async-scheduling"]
    assert _scheduler_server_args("native_async_if_supported") == ["--async-scheduling"]
    assert _scheduler_resolution("native_sync_fcfs")["scheduler_class"] == (
        "vllm.v1.core.sched.scheduler.Scheduler"
    )
    assert (
        _scheduler_resolution("native_async_if_supported")["scheduler_class"]
        == "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
    )
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        'SEMANTIC_RUNTIME_RESOLUTION {"schema":"semantic-runtime-resolution-v1",'
        '"available":true,"async_scheduling":false,'
        '"scheduler_class":"vllm.v1.core.sched.scheduler.Scheduler"}\n'
    )
    assert _parse_runtime_resolutions(str(runtime_log))[0]["async_scheduling"] is False
    assert _scheduler_server_args("cache_affinity") == [
        "--no-async-scheduling",
        "--scheduler-cls",
        "semantic_offload.cache_affinity.CacheAffinityScheduler",
    ]

    manifest, manifest_path = _campaign(tmp_path)
    del manifest["source"]["implementation_tree"]
    with pytest.raises(ManifestError, match="implementation_tree"):
        validate_campaign_manifest(manifest, manifest_path)


def test_drain_requires_events_idle_requests_and_async_background_gauge(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text('SEMANTIC_PRESERVATION_EVENT {"event":"one"}\n')

    class Handle:
        def __init__(self):
            self.log_path = str(log_path)

        def metrics_url(self):
            return "unused"

    class SyncMetrics:
        @staticmethod
        def snapshot_gauges(_url):
            return {
                "vllm:num_requests_running": 0.0,
                "vllm:num_requests_waiting": 0.0,
                "vllm:semantic_async_scoring_pending": None,
            }

    complete = asyncio.run(
        _wait_for_campaign_drain(
            Handle(),
            expected_events=1,
            metrics_mod=SyncMetrics,
            background_mode="synchronous",
            timeout_s=0.1,
        )
    )
    assert complete["status"] == "complete"

    timed_out = asyncio.run(
        _wait_for_campaign_drain(
            Handle(),
            expected_events=1,
            metrics_mod=SyncMetrics,
            background_mode="asynchronous",
            timeout_s=0.001,
        )
    )
    assert timed_out["status"] == "timeout"
