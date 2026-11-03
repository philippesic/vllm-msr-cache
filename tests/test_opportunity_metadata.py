"""Regression: sync opportunity observations must not look like legacy scores."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.outputs import KVConnectorOutput

from semantic_offload.connector import SemanticOffloadingConnector, OffloadingWorkerMetadata
from tests.test_audit import manager
from tests.test_scoring_schedule import configured_worker
from tests.test_step_1_3_scoring import _summary, _seed_durable_summaries, to_key


def _connector(worker):
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector.connector_worker = SimpleNamespace(
        worker=worker, build_connector_worker_meta=OffloadingWorkerMetadata,
    )
    connector.connector_scheduler = None
    connector._score_group_size = 1
    return connector


def _observed_worker():
    worker = configured_worker()
    first, second = to_key(1), to_key(2)
    _seed_durable_summaries(worker, {first: [_summary(2)], second: [_summary(-1)]})
    for query in (torch.ones(1, 1, 4), -torch.ones(1, 1, 4)):
        worker._on_queries_captured(["source"], query)
        worker.receive_score_requests({"source": [first, second]})
    assert worker._pending_scores == {"mean": {}}  # Actual failing producer shape.
    return worker, [first, second]


def test_sync_opportunity_metadata_transports_ordered_observations_without_phantom_legacy(monkeypatch):
    worker, keys = _observed_worker()
    expected_observations = deepcopy(worker._pending_async_score_observations)
    connector = _connector(worker)
    metadata = connector.build_connector_worker_meta()
    assert metadata.pending_scores == {}
    assert metadata.score_head_counts == {}
    assert metadata.relevance_updates == {}
    assert metadata.complete_async_score_observations() == expected_observations
    assert [item["observation_sequence"] for item in expected_observations] == [1, 2]
    # Both clocks and feature expiry survive transport unchanged.
    for actual, expected in zip(metadata.async_score_observations, expected_observations):
        for field in ("snapshot_captured_ns", "observation_available_ns", "expires_at_ns"):
            assert actual[field] == expected[field]
    actual_manager, expected_manager = manager(keys), manager(keys)
    for observation in expected_observations:
        expected_manager.update_relevance(observation["scores"])
    connector.connector_scheduler = SimpleNamespace(
        manager=actual_manager, _preservation_requests={}, _preservation_finished={},
    )
    monkeypatch.setattr(OffloadingConnector, "update_connector_output", lambda *args: None)
    connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=metadata))
    assert actual_manager.relevance_ema["mean"] == pytest.approx(expected_manager.relevance_ema["mean"])
    for key in keys:
        assert actual_manager._relevance_feature_snapshot_ns["mean"][key] == expected_observations[-1]["snapshot_captured_ns"]
    second_metadata = connector.build_connector_worker_meta()
    assert second_metadata.pending_scores == {}
    assert second_metadata.async_score_observations == []


@pytest.mark.parametrize("ranked", [[(to_key(1), 3.0)], []])
def test_real_legacy_request_metadata_is_preserved_and_mixed_payload_still_rejected(ranked):
    worker, keys = _observed_worker()
    worker._pending_scores["mean"]["legacy"] = ranked
    worker._pending_score_head_counts["mean"]["legacy"] = 1
    metadata = _connector(worker).build_connector_worker_meta()
    assert metadata.pending_scores == {"mean": {"legacy": ranked}}
    assert metadata.score_head_counts == {"mean": {"legacy": 1}}
    with pytest.raises(ValueError, match="cannot coexist with legacy"):
        metadata.complete_async_score_observations()


def test_empty_method_normalization_preserves_actual_legacy_scores_and_head_counts():
    worker = configured_worker("continuous")
    worker._pending_scores = {"unused": {}, "mean": {"source": [(to_key(1), 2.0)]}}
    worker._pending_score_head_counts = {"unused": {}, "mean": {"source": 1}}
    metadata = _connector(worker).build_connector_worker_meta()
    assert metadata.complete_scores() == {"mean": {"source": [(to_key(1), 2.0)]}}
    assert metadata.score_head_counts == {"mean": {"source": 1}}
    assert metadata.async_score_observations == []
