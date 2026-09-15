# SPDX-License-Identifier: Apache-2.0
"""Fail-closed MTP/EAGLE round-trip validator contract tests."""

from copy import deepcopy

from semantic_offload.mtp_roundtrip import SCHEMA, validate_roundtrip


def _phase(name, values, **extra):
    return {
        "name": name,
        "status": "complete",
        "metrics": {"source": "prometheus", "values": values},
        **extra,
    }


def valid_payload():
    return {
        "schema": SCHEMA,
        "dependency": {"vllm_revision": "main-abc", "candidate_sha": "fix-123"},
        "model": {"name": "Qwen/Qwen3.5-0.8B-Base", "mtp_enabled": True},
        "control": {
            "block_size": 16,
            "control_prefix_tokens": 256,
            "greedy_tokens": [1, 2, 3],
            "acceptance_rate": 0.9,
            "post_reset_ttft_s": 3.0,
        },
        "phases": [
            _phase(
                "cold_store",
                {
                    "store_bytes": 100,
                    "local_gpu_hit_tokens": 0,
                    "cpu_to_gpu_bytes": 0,
                    "external_tokens": 0,
                    "restored_prefix_tokens": 0,
                    "acceptance_rate": 0.0,
                    "ttft_s": 1.0,
                    "errors": 0,
                    "stalls": 0,
                },
                greedy_tokens=[1, 2, 3],
            ),
            _phase(
                "gpu_hit",
                {
                    "store_bytes": 0,
                    "local_gpu_hit_tokens": 256,
                    "cpu_to_gpu_bytes": 0,
                    "external_tokens": 0,
                    "restored_prefix_tokens": 256,
                    "acceptance_rate": 0.9,
                    "ttft_s": 0.1,
                    "errors": 0,
                    "stalls": 0,
                },
                greedy_tokens=[1, 2, 3],
            ),
            _phase(
                "local_reset",
                {
                    "store_bytes": 0,
                    "local_gpu_hit_tokens": 0,
                    "cpu_to_gpu_bytes": 0,
                    "external_tokens": 0,
                    "restored_prefix_tokens": 0,
                    "acceptance_rate": 0.0,
                    "ttft_s": 0.0,
                    "errors": 0,
                    "stalls": 0,
                },
                reset_scope="local_gpu_only",
                greedy_tokens=[1, 2, 3],
            ),
            _phase(
                "cpu_restore",
                {
                    "store_bytes": 0,
                    "local_gpu_hit_tokens": 0,
                    "cpu_to_gpu_bytes": 100,
                    "external_tokens": 256,
                    "restored_prefix_tokens": 256,
                    "acceptance_rate": 0.89,
                    "ttft_s": 0.8,
                    "errors": 0,
                    "stalls": 0,
                },
                greedy_tokens=[1, 2, 3],
            ),
        ],
    }


def test_valid_roundtrip_passes_and_reports_source_metrics():
    result = validate_roundtrip(
        valid_payload(),
        expected_vllm_revision="main-abc",
        expected_candidate_sha="fix-123",
    )
    assert result.status == "pass"
    assert result.metrics["restore_external_tokens"] == 256
    assert result.metrics["post_reset_ttft_speedup"] == 3.75


def test_missing_source_metrics_are_unavailable():
    payload = valid_payload()
    del payload["phases"][3]["metrics"]

    result = validate_roundtrip(payload)

    assert result.status == "unavailable"
    assert any("metrics" in reason for reason in result.unavailable)


def test_wrong_revision_is_invalid():
    result = validate_roundtrip(valid_payload(), expected_vllm_revision="other")

    assert result.status == "invalid"
    assert any("revision" in error for error in result.errors)


def test_output_mismatch_is_invalid():
    payload = valid_payload()
    payload["phases"][3]["greedy_tokens"] = [1, 9, 3]

    result = validate_roundtrip(payload)

    assert result.status == "invalid"
    assert any("greedy output" in error for error in result.errors)


def test_restore_must_be_external_not_local_gpu_hit():
    payload = valid_payload()
    payload["phases"][3]["metrics"]["values"]["local_gpu_hit_tokens"] = 1

    result = validate_roundtrip(payload)

    assert result.status == "invalid"
    assert any("local GPU hit" in error for error in result.errors)


def test_acceptance_loss_is_bounded():
    payload = valid_payload()
    payload["phases"][3]["metrics"]["values"]["acceptance_rate"] = 0.87

    result = validate_roundtrip(payload)

    assert result.status == "invalid"
    assert any("acceptance" in error for error in result.errors)


def test_prefix_must_be_within_one_block():
    payload = valid_payload()
    payload["phases"][3]["metrics"]["values"]["restored_prefix_tokens"] = 288

    result = validate_roundtrip(payload)

    assert result.status == "invalid"
    assert any("prefix" in error for error in result.errors)


def test_exact_token_accounting_can_replace_speedup_gate():
    payload = valid_payload()
    payload["phases"][3]["metrics"]["values"]["ttft_s"] = 2.0
    payload["token_accounting"] = {"exact": True, "restored_tokens": 256}

    assert validate_roundtrip(payload).status == "pass"


def test_missing_mtp_capability_is_unavailable():
    payload = valid_payload()
    payload["model"]["mtp_enabled"] = False

    result = validate_roundtrip(payload)

    assert result.status == "unavailable"
    assert any("MTP" in reason for reason in result.unavailable)


def test_phase_order_and_reset_scope_are_strict():
    payload = valid_payload()
    payload["phases"] = list(reversed(payload["phases"]))

    result = validate_roundtrip(payload)

    assert result.status == "invalid"
    assert any("ordered four-phase" in error for error in result.errors)


def test_errors_and_stalls_fail_closed_as_invalid():
    payload = valid_payload()
    payload["phases"][0]["metrics"]["values"]["stalls"] = 1

    result = validate_roundtrip(payload)

    assert result.status == "invalid"
    assert any("stalls" in error for error in result.errors)


def test_input_is_not_mutated():
    payload = valid_payload()
    before = deepcopy(payload)

    validate_roundtrip(payload)

    assert payload == before
