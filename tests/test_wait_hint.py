# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import patch

import pytest

from semantic_offload.wait_hint import (
    WAIT_HINT_PROVENANCE,
    WAIT_HINT_SCHEMA,
    KnownPrefixWaitHint,
    WaitHintLease,
    select_useful_on_time,
)


def _hint() -> dict:
    return {
        "schema": WAIT_HINT_SCHEMA,
        "event_id": "a" * 64,
        "owner_id": "owner:opaque",
        "run_id": "run:opaque",
        "cell_id": "cell:opaque",
        "session_id": "session:opaque",
        "generation": 2,
        "observed_prefix_id": "b" * 64,
        "prefix_boundary_tokens": 130,
        "observation_available_ns": 10,
        "input_publication_ns": 20,
        "declared_wait_ms": 50,
        "wait_provenance": WAIT_HINT_PROVENANCE,
        "namespace": {
            "model_revision": "model",
            "tokenizer_revision": "tokenizer",
            "adapter_namespace": "none",
            "layout_namespace": "dense-tp1-bf16-block16",
        },
    }


def test_wait_hint_rounds_only_to_complete_materialized_pages():
    hint = KnownPrefixWaitHint.from_mapping(_hint())

    assert hint.complete_page_boundary(16) == 128
    assert hint.namespace_dict()["model_revision"] == "model"


def test_wait_hint_accepts_real_campaign_external_owner_shape():
    raw = _hint()
    raw["owner_id"] = "sc|run-token|cell-token|work-token|sample-42"

    assert KnownPrefixWaitHint.from_mapping(raw).owner_id == raw["owner_id"]


def test_wait_hint_remaining_budget_subtracts_causal_elapsed_time():
    hint = KnownPrefixWaitHint.from_mapping(_hint())

    assert hint.remaining_wait_ns(20) == 50_000_000
    assert hint.remaining_wait_ns(30_000_020) == 20_000_000
    assert hint.remaining_wait_ns(70_000_020) == 0


def test_wait_hint_accepts_frozen_long_lead_cohort_only():
    raw = _hint()
    raw["declared_wait_ms"] = 750
    hint = KnownPrefixWaitHint.from_mapping(raw)

    assert hint.remaining_wait_ns(raw["input_publication_ns"]) == 750_000_000

    for unsupported in (1, 49, 51, 249, 251, 749, 751):
        raw = _hint()
        raw["declared_wait_ms"] = unsupported
        with pytest.raises(ValueError, match="frozen screen"):
            KnownPrefixWaitHint.from_mapping(raw)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("generation", True, "generation"),
        ("prefix_boundary_tokens", 0, "boundary"),
        ("declared_wait_ms", 51, "frozen screen"),
        ("event_id", "not-a-digest", "256-bit"),
    ],
)
def test_wait_hint_rejects_invalid_typed_fields(field, value, message):
    raw = _hint()
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        KnownPrefixWaitHint.from_mapping(raw)


def test_wait_hint_rejects_stale_clock_and_future_or_role_fields():
    raw = _hint()
    raw["input_publication_ns"] = 9
    with pytest.raises(ValueError, match="causal clocks"):
        KnownPrefixWaitHint.from_mapping(raw)

    for forbidden in ("future_suffix", "actual_resume_ns", "split", "phase"):
        raw = _hint()
        raw[forbidden] = "forbidden"
        with pytest.raises(ValueError, match="forbidden future"):
            KnownPrefixWaitHint.from_mapping(raw)


def test_wait_hint_requires_exact_namespace_and_estimate_provenance():
    raw = _hint()
    del raw["namespace"]["layout_namespace"]
    with pytest.raises(ValueError, match="namespace"):
        KnownPrefixWaitHint.from_mapping(raw)

    raw = _hint()
    raw["wait_provenance"] = "observed_future_resume"
    with pytest.raises(ValueError, match="provenance"):
        KnownPrefixWaitHint.from_mapping(raw)

    raw = _hint()
    raw["declared_wait_ms"] = None
    raw["wait_provenance"] = None
    assert KnownPrefixWaitHint.from_mapping(raw).remaining_wait_ns(25) is None


def test_wait_hint_lease_expires_cancels_and_resumes_without_reactivation():
    hint = KnownPrefixWaitHint.from_mapping(_hint())
    lease = WaitHintLease(
        hint,
        nomination_deadline_ns=50_000_020,
        expires_at_ns=2_050_000_020,
    )
    assert lease.active_at(2_050_000_020)
    assert not lease.active_at(2_050_000_021)

    for action, state in (
        ("expire", "expired"),
        ("cancel", "cancelled"),
        ("resume", "resumed"),
    ):
        candidate = copy.deepcopy(lease)
        getattr(candidate, action)()
        assert candidate.state == state
        assert not candidate.active_at(20)


def test_useful_on_time_selects_longest_chain_prefix_ready_by_deadline():
    result = select_useful_on_time(
        ["k0", "k1", "k2", "k3"],
        4,
        remaining_wait_ns=250,
        bytes_per_key=100,
        estimated_transfer_ns_per_byte=0.75,
        estimated_splice_ns_per_byte=0.25,
    )

    assert result.selected_keys == ("k0", "k1")
    assert result.estimated_selected_ns == 200
    assert result.status == "selected_on_time_prefix"


def test_useful_on_time_charges_transfer_and_splice_cost():
    result = select_useful_on_time(
        ["k0", "k1", "k2"],
        3,
        remaining_wait_ns=250,
        bytes_per_key=100,
        estimated_transfer_ns_per_byte=1.0,
        estimated_splice_ns_per_byte=0.5,
    )

    assert result.selected_keys == ("k0",)
    assert result.estimated_selected_ns == 150
    assert result.estimated_transfer_ns_per_byte == 1.0
    assert result.estimated_splice_ns_per_byte == 0.5


def test_useful_on_time_uses_explicit_ordered_fallback_for_missing_cost():
    result = select_useful_on_time(
        ["k0", "k1", "k2"],
        2,
        remaining_wait_ns=250,
        bytes_per_key=100,
        estimated_transfer_ns_per_byte=None,
        estimated_splice_ns_per_byte=0.25,
    )

    assert result.selected_keys == ("k0", "k1")
    assert result.status == "fallback_request_order"
    assert result.fallback == "missing_wait_or_cost_estimate"


def test_instrumentation_control_enables_probe_only_with_shared_scorer():
    from semantic_offload import spec as spec_module

    captured = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    spec = spec_module.SemanticOffloadingSpec.__new__(
        spec_module.SemanticOffloadingSpec
    )
    spec.extra_config = {
        "instrumentation_only": True,
        "shared_prefetch_scorer": True,
    }
    spec.tokens_per_block = [16]
    spec.num_blocks = 4
    spec.vllm_config = object()
    with (
        patch.object(spec_module, "SemanticOffloadingWorker", FakeWorker),
        patch.object(spec_module, "spec_blocks_per_chunk", return_value=1),
    ):
        spec.create_worker(None)

    assert captured["probe_layer"] == "middle"

    spec.extra_config = {"instrumentation_only": True}
    captured.clear()
    with (
        patch.object(spec_module, "SemanticOffloadingWorker", FakeWorker),
        patch.object(spec_module, "spec_blocks_per_chunk", return_value=1),
    ):
        spec.create_worker(None)
    assert captured["probe_layer"] is None
