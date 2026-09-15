# SPDX-License-Identifier: Apache-2.0
"""Decision-time feature trace identity, leakage, and provenance gates."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.outputs import KVConnectorOutput

from semantic_offload.connector import (
    SemanticOffloadingConnector,
    SemanticWorkerMetadata,
)
from semantic_offload.decision_trace import (
    TRACE_MARKER,
    build_decision_feature_records,
    validate_decision_feature_records,
)
from semantic_offload.manager import (
    PreservationOffloadingManager,
    SemanticOffloadingManager,
    compose_relevance_updates,
)
from semantic_offload.policy import SemanticPolicy


def _key(value: int, group: int = 0) -> bytes:
    return make_offload_key(f"block-{value}".encode(), group)


def _context(sample_index: int) -> dict[str, object]:
    external = f"sc|run-1|cell-rag|rag|sample-{sample_index}"
    return {
        "request_id": f"cmpl-{external}-0-deadbeef",
        "external_request_id": external,
        "run_id": "run-1",
        "cell_id": "cell-rag",
        "sample_index": sample_index,
    }


def _records(sample_index: int, sequence: int = 0) -> list[dict[str, object]]:
    key_a, key_b = _key(1), _key(2, group=1)
    blocks = []
    for block_id in (7, 8):
        block = BlockStatus(block_id)
        block.ref_cnt = 0
        blocks.append(block)
    records: list[dict[str, object]] = []
    build_decision_feature_records(
        context=_context(sample_index),
        decision_sequence=sequence,
        candidates=[(key_a, blocks[0]), (key_b, blocks[1])],
        ranked_keys=[key_b, key_a],
        selected_keys={key_b},
        relevance={key_a: 0.25, key_b: 0.75},
        latest_relevance={key_a: 0.2, key_b: 0.9},
        sink=records.extend,
    )
    return records


def test_trace_captures_physical_identity_features_and_cardinality():
    records = _records(7)
    validated = validate_decision_feature_records(
        records,
        expected_request_ids=["sc|run-1|cell-rag|rag|sample-7"],
        run_id="run-1",
        cell_id="cell-rag",
    )

    assert len(validated) == 2
    selected = next(item for item in validated if item["selected"])
    assert selected["offload_key"] == _key(2, group=1).hex()
    assert selected["group_idx"] == 1
    assert selected["cpu_block_id"] == 8
    assert selected["semantic_relevance_score"] == pytest.approx(0.9)
    assert selected["relevance_ema"] == pytest.approx(0.75)
    assert selected["rank"] == 0
    assert selected["recency_norm"] == pytest.approx(1.0)
    assert selected["frequency"] is None
    assert selected["frequency_status"] == "unavailable"
    assert {item["candidate_ordinal"] for item in validated} == {0, 1}


def test_policy_publishes_features_at_eviction_decision_time():
    records: list[dict[str, object]] = []
    key_a, key_b = _key(1), _key(2, group=1)
    relevance = {"mean": {key_a: 0.1, key_b: 0.9}}
    latest = {"mean": {key_a: 0.2, key_b: 0.8}}
    policy = SemanticPolicy(
        cache_capacity=2,
        relevance_ema=relevance,
        latest_relevance=latest,
        decision_feature_sink=records.extend,
    )
    for block_id, key in enumerate((key_a, key_b), start=7):
        block = BlockStatus(block_id)
        block.ref_cnt = 0
        policy.insert(key, block)
        policy.mark_evictable(key)
    policy.set_decision_feature_context(_context(7))

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert [item["selected"] for item in records] == [True, False]
    assert [item["rank"] for item in records] == [0, 1]
    assert all(item["decision_timestamp_ns"] is not None for item in records)
    assert all(item["clock_domain"] == "host_monotonic" for item in records)
    assert all(item["scoring_cpu_time_ns"] >= 0 for item in records)
    assert all(item["policy_cpu_time_ns"] >= 0 for item in records)
    assert all(item["critical_path_wall_time_ns"] >= 0 for item in records)
    assert all(item["instrumentation_cpu_time_ns"] >= 0 for item in records)
    assert all(item["instrumentation_wall_time_ns"] >= 0 for item in records)
    assert all(item["synchronization_wait_ns"] == 0 for item in records)
    assert all(
        item["synchronization_status"] == "no_explicit_synchronization"
        for item in records
    )


def test_manager_binds_eviction_trace_to_request_context():
    manager = SemanticOffloadingManager(num_blocks=1, decision_feature_trace=True)
    first = _key(1)
    second = _key(2)
    first_context = ReqContext(
        req_id=_context(0)["request_id"],
        kv_transfer_params={
            "run_id": "run-1",
            "cell_id": "cell-rag",
        },
    )
    second_context = ReqContext(
        req_id=_context(1)["request_id"],
        kv_transfer_params={
            "run_id": "run-1",
            "cell_id": "cell-rag",
        },
    )
    first_output = manager.prepare_store([first], first_context)
    assert first_output is not None
    manager.complete_store([first], first_context)
    second_output = manager.prepare_store([second], second_context)
    assert second_output is not None

    records = manager.drain_decision_feature_records()
    validated = validate_decision_feature_records(
        records,
        expected_request_ids=["sc|run-1|cell-rag|rag|sample-1"],
        run_id="run-1",
        cell_id="cell-rag",
    )
    assert len(validated) == 1
    assert validated[0]["external_request_id"].endswith("sample-1")


@pytest.mark.parametrize("cache_policy", ["arc", "lru"])
def test_matched_control_manager_emits_exact_victim_and_policy_overhead(cache_policy):
    manager = PreservationOffloadingManager(
        num_blocks=2,
        cache_policy=cache_policy,
        decision_feature_trace=True,
    )
    first, second, third = _key(31), _key(32), _key(33)
    initial_context = ReqContext(req_id=_context(0)["request_id"])
    initial = manager.prepare_store([first, second], initial_context)
    assert initial is not None
    manager.complete_store([first, second], initial_context)

    decision_context = ReqContext(
        req_id=_context(1)["request_id"],
        kv_transfer_params={"run_id": "run-1", "cell_id": "cell-rag"},
    )
    result = manager.prepare_store([third], decision_context)
    assert result is not None and len(result.evicted_keys) == 1

    records = manager.drain_decision_feature_records()
    validated = validate_decision_feature_records(
        records,
        expected_request_ids=["sc|run-1|cell-rag|rag|sample-1"],
        run_id="run-1",
        cell_id="cell-rag",
    )
    selected = [record for record in validated if record["selected"]]
    assert [record["offload_key"] for record in selected] == [
        result.evicted_keys[0].hex()
    ]
    assert all(record["policy_cpu_time_ns"] >= 0 for record in records)
    assert all(record["critical_path_wall_time_ns"] >= 0 for record in records)
    assert all(record["instrumentation_cpu_time_ns"] >= 0 for record in records)
    assert all(record["instrumentation_wall_time_ns"] >= 0 for record in records)
    assert all(record["scoring_cpu_time_ns"] is None for record in records)
    assert all(record["policy"] in {"ARCCachePolicy", "LRUCachePolicy"} for record in records)
    assert all(record["policy_state"] for record in records)


def test_trace_has_no_current_event_outcome_or_future_label_channel():
    record = _records(7)[0]
    record["outcome"] = "hit"

    with pytest.raises(ValueError, match="outcome/leakage"):
        validate_decision_feature_records(
            [record],
            expected_request_ids=["sc|run-1|cell-rag|rag|sample-7"],
            run_id="run-1",
            cell_id="cell-rag",
        )


def test_shuffled_rag_trace_rows_use_canonical_identity_not_input_order():
    first = _records(30, sequence=0)
    second = _records(10, sequence=1)
    shuffled = list(reversed(first + second))
    validated = validate_decision_feature_records(
        shuffled,
        expected_request_ids=[
            "sc|run-1|cell-rag|rag|sample-30",
            "sc|run-1|cell-rag|rag|sample-10",
        ],
        run_id="run-1",
        cell_id="cell-rag",
    )

    assert {record["external_request_id"] for record in validated} == {
        "sc|run-1|cell-rag|rag|sample-30",
        "sc|run-1|cell-rag|rag|sample-10",
    }
    assert [record["sequence_ordinal"] for record in validated] == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ("frequency", "frequency_status"),
    [(1.0, "unavailable"), (None, "available")],
)
def test_frequency_requires_matching_finite_availability(frequency, frequency_status):
    record = _records(7)[0]
    record["frequency"] = frequency
    record["frequency_status"] = frequency_status

    with pytest.raises(ValueError, match="frequency"):
        validate_decision_feature_records(
            [record],
            expected_request_ids=["sc|run-1|cell-rag|rag|sample-7"],
            run_id="run-1",
            cell_id="cell-rag",
        )


def test_compact_connector_metadata_carries_raw_score_into_eviction_trace(
    monkeypatch,
):
    key_a, key_b, key_c = _key(20), _key(21), _key(22)
    manager = SemanticOffloadingManager(num_blocks=2, decision_feature_trace=True)
    for index, key in enumerate((key_a, key_b)):
        context = ReqContext(req_id=_context(index)["request_id"])
        output = manager.prepare_store([key], context)
        assert output is not None
        manager.complete_store([key], context)

    scores = {
        "mean": {
            "req-a": [(key_a, 0.2), (key_b, 0.1)],
            "req-b": [(key_b, 0.9), (key_a, 0.8)],
        }
    }
    metadata = SemanticWorkerMetadata(
        relevance_updates=compose_relevance_updates(scores),
        latest_relevance={"mean": {key_a: 0.8, key_b: 0.9}},
        score_group_size=1,
        score_contributors=1,
    )
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector.connector_scheduler = SimpleNamespace(manager=manager)
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector."
        "OffloadingConnector.update_connector_output",
        lambda self, output: None,
    )

    connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=metadata)
    )
    manager.prepare_store([key_c], ReqContext(req_id=_context(9)["request_id"]))

    records = manager.drain_decision_feature_records()
    selected = next(record for record in records if record["selected"])
    expected_scores = {key_a: 0.8, key_b: 0.9}
    assert selected["semantic_relevance_score"] == pytest.approx(
        expected_scores[bytes.fromhex(selected["offload_key"])]
    )


def test_disabled_policy_does_not_emit_or_change_eviction_behavior():
    records: list[dict[str, object]] = []
    policy = SemanticPolicy(cache_capacity=2, decision_feature_sink=None)
    keys = [_key(1), _key(2)]
    for index, key in enumerate(keys):
        block = BlockStatus(index)
        block.ref_cnt = 0
        policy.insert(key, block)
        policy.mark_evictable(key)

    policy.set_decision_feature_context(_context(1))
    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert [key for key, _ in evicted] == [keys[0]]
    assert records == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records.pop(), "cardinality"),
        (
            lambda records: records[0].update(candidate_ordinal=0),
            "ordinals",
        ),
        (lambda records: records[0].update(group_idx=0), "group identity"),
        (lambda records: records[0].update(run_id="other"), "run/cell"),
    ],
)
def test_malformed_or_conflicting_trace_fails_closed(mutation, message):
    records = _records(7)
    mutation(records)

    with pytest.raises(ValueError, match=message):
        validate_decision_feature_records(
            records,
            expected_request_ids=["sc|run-1|cell-rag|rag|sample-7"],
            run_id="run-1",
            cell_id="cell-rag",
        )


def test_runner_trace_join_accepts_engine_prefix_and_rejects_foreign_data(tmp_path):
    from benchmarks.run_latency_suite import _read_decision_feature_records

    records = _records(7)
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + TRACE_MARKER
        + json.dumps(records[0], sort_keys=True)
        + "\n"
        + TRACE_MARKER
        + json.dumps(records[1], sort_keys=True)
        + "\n"
    )

    joined = _read_decision_feature_records(
        str(log_path),
        "sc|run-1|cell-rag|rag|sample-",
        {"sc|run-1|cell-rag|rag|sample-7"},
        "run-1",
        "cell-rag",
    )
    assert len(joined) == 2

    allowed_foreign = _records(8)[0]
    allowed_foreign["external_request_id"] = "sc|run-1|cell-rag|other|sample-8"
    log_path.write_text(TRACE_MARKER + json.dumps(allowed_foreign) + "\n")
    assert (
        _read_decision_feature_records(
            str(log_path),
            "sc|run-1|cell-rag|rag|sample-",
            ["sc|run-1|cell-rag|rag|sample-7"],
            "run-1",
            "cell-rag",
            allowed_request_prefixes=("sc|run-1|cell-rag|other|sample-",),
        )
        == []
    )

    foreign = deepcopy(records[0])
    foreign["external_request_id"] = "sc|run-1|other-cell|rag|sample-7"
    log_path.write_text(TRACE_MARKER + json.dumps(foreign) + "\n")
    with pytest.raises(ValueError, match="foreign decision feature request"):
        _read_decision_feature_records(
            str(log_path),
            "sc|run-1|cell-rag|rag|sample-",
            {"sc|run-1|cell-rag|rag|sample-7"},
            "run-1",
            "cell-rag",
        )

    unexpected = _records(8)[0]
    log_path.write_text(TRACE_MARKER + json.dumps(unexpected) + "\n")
    with pytest.raises(ValueError, match="unexpected decision feature request"):
        _read_decision_feature_records(
            str(log_path),
            "sc|run-1|cell-rag|rag|sample-",
            ["sc|run-1|cell-rag|rag|sample-7"],
            "run-1",
            "cell-rag",
        )
