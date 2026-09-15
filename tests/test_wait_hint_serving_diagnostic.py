# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
from pathlib import Path

import pytest

from benchmarks import wait_hint_serving_diagnostic as diagnostic


def _record(
    request_id: str,
    *,
    index: int,
    phase: str,
    family: str,
    session: str,
    task_origin: str,
    tokens: list[int],
    key_chain: str,
) -> dict:
    return {
        "request_id": request_id,
        "request_index": index,
        "phase": phase,
        "phase_index": index,
        "phase_ordinal": 0,
        "split": "screen",
        "context_budget": 2048,
        "episode": "reusable_prefix",
        "family": family,
        "session_id": session,
        "generation": 0,
        "task_origin": task_origin,
        "prompt_token_ids": tokens,
        "prompt_token_digest": diagnostic.digest(tokens),
        "key_chain_digest": key_chain,
        "context_token_span": [0, 32],
        "content_origin": "immutable_prepared_longbench_context",
        "arrival_origin": "fixed",
        "wait_origin": "prior_complete",
        "pressure_origin": "fixed",
        "source_id": request_id,
        "suffix_budget_tokens": 32,
    }


def _source_document() -> dict:
    prefix = list(range(32))
    return {
        "requests": [
            _record(
                "a-q0",
                index=0,
                phase="warm1",
                family="qasper",
                session="a",
                task_origin="supplied_question_0",
                tokens=prefix + [100],
                key_chain="a-chain",
            ),
            _record(
                "a-q1",
                index=1,
                phase="warm2",
                family="qasper",
                session="a",
                task_origin="supplied_question_1",
                tokens=prefix + [101],
                key_chain="a-chain",
            ),
            _record(
                "b-short",
                index=2,
                phase="pollution",
                family="qasper",
                session="b0",
                task_origin="supplied_question_0",
                tokens=list(range(36)),
                key_chain="b0-chain",
            ),
            _record(
                "b-long",
                index=3,
                phase="pollution",
                family="repobench-p",
                session="b1",
                task_origin="supplied_task_exact_repeat",
                tokens=list(range(40)),
                key_chain="b1-chain",
            ),
        ]
    }


def test_request_artifact_selects_exact_a_q0_q1_and_unique_longest_b(
    tmp_path, monkeypatch
):
    source = tmp_path / "requests.json"
    source.write_text(json.dumps(_source_document()))
    monkeypatch.setattr(diagnostic, "SOURCE_REQUEST_SHA256", diagnostic.file_digest(source))
    monkeypatch.setattr(diagnostic, "validate_request_manifest", lambda _document: None)

    artifact = diagnostic.build_request_artifact(_source_document(), source)

    assert artifact["selection"] == {
        "split": "screen",
        "context_budget": 2048,
        "a_rule": "first manifest-order Qasper warm1 target and its unique q1 warm2 continuation",
        "b_rule": "unique maximum prompt-token count among different-key-chain pollution records",
        "a_warm_request_id": "a-q0",
        "a_resume_request_id": "a-q1",
        "b_request_id": "b-long",
        "b_prompt_tokens": 40,
        "known_prefix_boundary_tokens": 32,
        "known_prefix_token_digest": diagnostic.digest(list(range(32))),
    }
    assert [entry["diagnostic_role"] for entry in artifact["requests"]] == [
        "a_warm_q0",
        "b_longest_pollution",
        "a_resume_q1",
    ]
    assert artifact["requests"][0]["original_record"] == _source_document()["requests"][0]
    assert artifact["requests"][2]["original_record"] == _source_document()["requests"][1]


def test_request_artifact_rejects_tied_longest_pollution(tmp_path, monkeypatch):
    document = _source_document()
    tied = copy.deepcopy(document["requests"][-1])
    tied["request_id"] = "b-long-tie"
    tied["source_id"] = "b-long-tie"
    document["requests"].append(tied)
    source = tmp_path / "requests.json"
    source.write_text(json.dumps(document))
    monkeypatch.setattr(diagnostic, "SOURCE_REQUEST_SHA256", diagnostic.file_digest(source))
    monkeypatch.setattr(diagnostic, "validate_request_manifest", lambda _document: None)

    with pytest.raises(diagnostic.DiagnosticError, match="not unique"):
        diagnostic.build_request_artifact(document, source)


def test_plan_requires_shared_staged_preflight_and_child_binary_inventory(
    tmp_path, monkeypatch
):
    source = tmp_path / "requests.json"
    source.write_text(json.dumps(_source_document()))
    monkeypatch.setattr(diagnostic, "SOURCE_REQUEST_SHA256", diagnostic.file_digest(source))
    monkeypatch.setattr(diagnostic, "validate_request_manifest", lambda _document: None)
    request_artifact = diagnostic.build_request_artifact(_source_document(), source)
    request_artifact_path = tmp_path / "request-artifact.json"
    request_artifact_path.write_text(json.dumps(request_artifact))

    plan = diagnostic.build_plan(
        request_artifact_path,
        request_runtime_path="/home/batch/requests.json",
        wheel_path="/home/batch/wheel.whl",
        wheel_sha256="a" * 64,
        plugin_root="/home/batch/plugin",
        binary_inventory_path="/home/batch/base-inventory.json",
        child_binary_inventory_path="/home/batch/child-inventory.json",
        child_binary_inventory_sha256="b" * 64,
        staged_preflight_manifest_path="/home/batch/preflight.json",
        staged_preflight_report_path="/home/batch/preflight-report.json",
        extension_root="/home/batch/extension",
        extension_revision="c" * 40,
        extension_tree="d" * 40,
        output_root="/home/batch/outputs",
    )

    assert plan["child_runtime_binary_inventory"] == {
        "path": "/home/batch/child-inventory.json",
        "sha256": "b" * 64,
        "verification": "hash child-resolved source-package binaries and import",
    }
    assert plan["preflight"]["argv"][0][2:] == [
        "benchmarks.staged_environment_preflight",
        "--manifest=/home/batch/preflight.json",
        "--output=/home/batch/preflight-report.json",
    ]


def test_strict_staged_path_never_uses_existing_review_fallback(
    tmp_path, monkeypatch
):
    review = tmp_path / "review.json"
    review.write_text("review")
    monkeypatch.setenv("EP_STRICT_STAGED_PATHS", "1")

    with pytest.raises(diagnostic.DiagnosticError, match="strict staged runtime"):
        diagnostic._bound_validation_path(
            Path("/home/missing/runtime.json"), review, "fixture"
        )


def test_request_payload_returns_exact_serving_envelope():
    record = _source_document()["requests"][0]
    entry = {
        "diagnostic_role": "a_warm_q0",
        "original_request_id": record["request_id"],
        "original_record_sha256": diagnostic.digest(record),
        "original_record": record,
    }

    payload = diagnostic._request_payload(
        entry,
        case={"case_id": "child-hint-250", "config_hash": "c" * 64},
        index=0,
        observed_prefix_id="a" * 64,
        boundary=32,
        hint_event_id="b" * 64,
    )

    assert isinstance(payload, dict)
    assert payload["prompt"] == record["prompt_token_ids"]
    assert payload["request_id"].startswith("sc|diag-run-child-hint-250|")
    assert payload["kv_transfer_params"]["observed_prefix_id"] == "a" * 64
    assert payload["kv_transfer_params"]["wait_hint_event_id"] == "b" * 64


def test_plugin_discovery_uses_isolated_distribution_metadata(tmp_path):
    package = tmp_path / "semantic_offload"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "wait_hint_endpoint.py").write_text(
        "class SemanticWaitHintEndpointPlugin:\n"
        "    name = 'semantic_wait_hint'\n"
        "    required_tasks = ('generate',)\n"
    )
    metadata = tmp_path / "semantic_offload-0.1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: semantic-offload\nVersion: 0.1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[vllm.endpoint_plugins]\n"
        "semantic_wait_hint = semantic_offload.wait_hint_endpoint:SemanticWaitHintEndpointPlugin\n"
    )
    expected = {
        "entry_point_group": "vllm.endpoint_plugins",
        "entry_point_name": "semantic_wait_hint",
        "entry_point_value": "semantic_offload.wait_hint_endpoint:SemanticWaitHintEndpointPlugin",
    }

    result = diagnostic.verify_plugin_discovery(tmp_path, expected)

    assert result["verified"] is True
    assert result["distribution_version"] == "0.1.0"
    assert Path(result["module_origin"]).is_relative_to(tmp_path)


def test_binary_inventory_verifies_actual_resolved_files(tmp_path):
    binary = tmp_path / "native.so"
    binary.write_bytes(b"native-binary")
    inventory_path = tmp_path / "inventory.json"
    inventory = {
        "schema": "ep-native149-binary-inventory-v1",
        "binary_files_count": 1,
        "files": [
            {
                "path": str(binary),
                "resolved_path": str(binary.resolve()),
                "bytes": binary.stat().st_size,
                "sha256": diagnostic.file_digest(binary),
                "symlink": False,
            }
        ],
    }
    inventory_path.write_text(json.dumps(inventory))

    result = diagnostic.verify_binary_inventory(
        inventory_path, diagnostic.file_digest(inventory_path)
    )

    assert result["verified"] is True
    assert result["verified_file_count"] == 1
    binary.write_bytes(b"changed")
    with pytest.raises(diagnostic.DiagnosticError, match="native binary mismatch"):
        diagnostic.verify_binary_inventory(
            inventory_path, diagnostic.file_digest(inventory_path)
        )


def _case_result(case: dict, output_tokens: list[list[int]]) -> dict:
    wait_origin = 100
    requests = [
        {
            "diagnostic_role": role,
            "success": True,
            "request_start_ns": wait_origin if index < 2 else wait_origin + case["tool_gap_ms"] * 1_000_000,
            "first_token_ns": (
                wait_origin
                if index < 2
                else wait_origin + case["tool_gap_ms"] * 1_000_000
            )
            + 3,
            "request_end_ns": (
                wait_origin
                if index < 2
                else wait_origin + case["tool_gap_ms"] * 1_000_000
            )
            + 10,
            "output_token_ids": output_tokens[index],
            "finish_reasons": ["stop"],
        }
        for index, role in enumerate(
            ("a_warm_q0", "b_longest_pollution", "a_resume_q1")
        )
    ]
    result = {
        "schema": diagnostic.CASE_RESULT_SCHEMA,
        "case_id": case["case_id"],
        "plan_digest": "plan",
        "case_config_hash": case.get("config_hash"),
        "requests": requests,
        "drain": {"status": "complete"},
        "wait_origin_ns": wait_origin,
        "resume_eligibility_ns": wait_origin + case["tool_gap_ms"] * 1_000_000,
        "resume_dispatch_ns": wait_origin + case["tool_gap_ms"] * 1_000_000,
        "resume_release_waited_for_hint_ack": False,
        "primary_hint": None,
        "negative_calls": [],
        "wait_hint_events": [],
        "wait_hint_event_counts": {},
        "preservation_events": [],
    }
    if case["endpoint_plugin_enabled"]:
        result["primary_hint"] = {
            "started_ns": (
                wait_origin + 1 if case["tool_gap_ms"] == 0 else wait_origin
            ),
            "payload": {"event_id": "primary"},
            "response": {"status": "expired"},
        }
    if case["case_id"] == "child-hint-250":
        resume_start = result["requests"][-1]["request_start_ns"]
        result["primary_hint"]["payload"] = {"event_id": "primary"}
        result["primary_hint"]["payload_digest"] = diagnostic.digest(
            result["primary_hint"]["payload"]
        )
        result["primary_hint"]["response"] = {
            "status": "accepted",
            "event_id": "primary",
            "opportunity": {
                "positive_opportunity": True,
                "local_gpu_coverage_available": True,
                "eligible_cpu_staging_prefix_count": 2,
                "local_gpu_prefix_count": 0,
                "source_key_count": 2,
            },
        }
        result["wait_hint_events"] = [
            {
                "event": "staging_submitted",
                "event_id": "primary",
                "job_id": 1,
                "selected_keys": ["k"],
                "bytes_per_key": 128,
                "submitted_bytes": 128,
                "submitted_at_ns": resume_start - 90,
            },
            {
                "event": "staging_ready",
                "event_id": "primary",
                "job_id": 1,
                "transfer_bytes": 128,
                "ready_at_ns": resume_start - 80,
            },
            {
                "event": "staging_consumed",
                "event_id": "primary",
                "splice_id": 2,
                "consumed_keys": 1,
                "discarded_keys": 0,
                "consumed_bytes": 128,
                "discarded_bytes": 0,
                "consumed_at_ns": resume_start + 2,
            },
        ]
        result["wait_hint_event_counts"] = diagnostic._event_action_counts(
            result["wait_hint_events"]
        )
        result["preservation_events"] = [
            {
                "schedule_role": "a_resume_q1",
                "external_request_id": "resume",
                "prefetch_attribution": {
                    "wait_hint": {
                        "event_id": "primary",
                        "owner_id": "resume",
                        "input_publication_ns": resume_start - 100,
                        "bound_at_ns": resume_start + 1,
                    },
                    "splice_acknowledged": True,
                    "splice_ack_id": "splice-2",
                    "splice_fence_ns": resume_start + 4,
                    "splice_fence_evidence": [
                        {
                            "splice_id": 2,
                            "source_block_count": 1,
                            "destination_block_count": 1,
                            "copied_layer_count": 28,
                            "copied_on_cuda": True,
                            "same_default_stream": True,
                            "issued_before_base_bind_and_forward": True,
                            "ordering_witness": "same_default_stream_copy_enqueued_before_base_bind_and_forward",
                            "completion_event_satisfied": True,
                            "issued_at_ns": resume_start + 2,
                            "completion_observed_at_ns": resume_start + 4,
                        }
                    ],
                    "terminal_cleanup": {"source_reuse_fenced": True},
                    "gpu_splice_bytes": 128,
                    "prefetch_h2d_bytes": 128,
                    "prefetched_keys": ["k"],
                    "intersection_keys": ["k"],
                    "source_key_to_gpu_block": {"k": "1"},
                    "destination_key_to_gpu_block": {"k": "2"},
                },
            }
        ]
    if case["negative_calls_after_primary"]:
        responses = [
            {"status": "expired", "reason": "nomination_deadline_elapsed"},
            {
                "status": "expired",
                "reason": "nomination_deadline_elapsed",
                "idempotent_replay": True,
            },
            {"status": "no_action", "reason": "observed_prefix_unavailable"},
            {"status": "rejected", "reason": "causal_clock_mismatch"},
        ]
        result["negative_calls"] = [
            {
                "payload": {"event_id": f"negative-{index}"},
                "response": responses[index],
                "http_status": 200,
                "error": None,
            }
            for index in range(4)
        ]
    result["status"] = "valid"
    result["valid"] = True
    result["validation_error"] = None
    result["result_digest"] = diagnostic.digest(result)
    return result


def test_suite_validation_requires_equal_outputs_and_positive_transfer():
    cases = [
        {
            "case_id": "original149-no-hint",
            "endpoint_plugin_enabled": False,
            "tool_gap_ms": 250,
            "negative_calls_after_primary": False,
            "config": {"same": True},
        },
        {
            "case_id": "child-no-hint",
            "endpoint_plugin_enabled": False,
            "tool_gap_ms": 250,
            "negative_calls_after_primary": False,
            "config": {"same": True},
        },
        {
            "case_id": "child-hint-250",
            "endpoint_plugin_enabled": True,
            "tool_gap_ms": 250,
            "negative_calls_after_primary": False,
            "config": {"hint": True},
        },
        {
            "case_id": "child-hint-0-negative",
            "endpoint_plugin_enabled": True,
            "tool_gap_ms": 0,
            "negative_calls_after_primary": True,
            "config": {"hint": True},
        },
    ]
    plan = {"cases": cases, "plan_digest": "plan"}
    outputs = [[1], [2], [3]]
    results = [_case_result(case, outputs) for case in cases]

    suite = diagnostic.validate_suite(results, plan)

    assert suite["valid"] is True
    broken = copy.deepcopy(results)
    broken[-1]["requests"][-1]["output_token_ids"] = [4]
    broken[-1].pop("result_digest")
    broken[-1]["result_digest"] = diagnostic.digest(broken[-1])
    with pytest.raises(diagnostic.DiagnosticError, match="exact outputs differ"):
        diagnostic.validate_suite(broken, plan)

    orphaned = copy.deepcopy(results)
    positive = next(
        result for result in orphaned if result["case_id"] == "child-hint-250"
    )
    positive["wait_hint_events"][0]["event_id"] = "orphan"
    positive.pop("result_digest")
    positive["result_digest"] = diagnostic.digest(positive)
    with pytest.raises(
        diagnostic.DiagnosticError, match="primary lifecycle cardinality"
    ):
        diagnostic.validate_suite(orphaned, plan)


def test_failed_case_validation_preserves_structured_raw_evidence(
    tmp_path, monkeypatch
):
    output = tmp_path / "failed-result.json"
    case = {"case_id": "child-hint-250", "config_hash": "c" * 64}
    plan = {"plan_digest": "p" * 64}

    async def raw_result(_plan, _case, _output):
        return {
            "schema": diagnostic.CASE_RESULT_SCHEMA,
            "case_id": "child-hint-250",
            "plan_digest": "p" * 64,
            "case_config_hash": "c" * 64,
            "requests": [{"diagnostic_role": "a_warm_q0", "success": True}],
            "primary_hint": {"response": {"status": "accepted"}},
            "wait_hint_events": [{"event": "hint_result"}],
            "metrics_before": {"raw": "before"},
            "metrics_after": {"raw": "after"},
            "execution_error": None,
        }

    monkeypatch.setattr(diagnostic, "_execute_case_raw", raw_result)

    with pytest.raises(diagnostic.DiagnosticError, match="case request completion"):
        asyncio.run(diagnostic.execute_case(plan, case, output))

    preserved = json.loads(output.read_text())
    assert preserved["valid"] is False
    assert preserved["status"] == "validation_failed_evidence_preserved"
    assert preserved["metrics_before"] == {"raw": "before"}
    assert preserved["primary_hint"]["response"]["status"] == "accepted"
    assert "case request completion" in preserved["validation_error"]
    assert preserved["result_digest"]


def test_zero_gap_async_release_does_not_require_hint_start_before_origin():
    events = []

    async def hint_call():
        events.append(("hint_start", diagnostic.time.monotonic_ns()))
        await asyncio.sleep(0.01)
        events.append(("hint_done", diagnostic.time.monotonic_ns()))
        return {"status": "expired"}

    async def resume_call():
        events.append(("resume_start", diagnostic.time.monotonic_ns()))
        return {"success": True}

    async def run():
        origin = diagnostic.time.monotonic_ns()
        result = await diagnostic._dispatch_at_release_boundary(
            wait_origin_ns=origin,
            tool_gap_ms=0,
            hint_call=hint_call,
            resume_call=resume_call,
        )
        return origin, result

    origin, result = asyncio.run(run())

    event_times = dict(events)
    assert result["resume_eligibility_ns"] == origin
    assert result["resume_release_waited_for_hint_ack"] is False
    assert event_times["hint_start"] >= origin
    assert event_times["resume_start"] < event_times["hint_done"]
