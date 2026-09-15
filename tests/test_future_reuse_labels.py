# SPDX-License-Identifier: Apache-2.0
"""Leakage and split-boundary tests for future-reuse labels."""

import csv
import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from semantic_offload.decision_trace import TRACE_EVENT, TRACE_SCHEMA
from semantic_offload.future_reuse import (
    _canonical_digest,
    build_future_reuse_labels,
    build_future_reuse_labels_from_artifact,
    load_future_reuse_artifact,
)
from tests.test_preservation_mapping import _event as _preservation_event


def _event(
    *,
    sample_index,
    window_id,
    keys,
    workload="chat",
    run_id="run",
    cell_id="cell",
    hits=(),
    missing=(),
    prior_evicted=(),
    authoritative=True,
):
    if keys and not hits and not missing:
        hits = keys
    event = _preservation_event(
        sample_index=sample_index,
        run_id=run_id,
        cell_id=cell_id,
        mode="hit",
    )
    request_id = event["external_request_id"].replace("|chat|", f"|{workload}|")
    event["external_request_id"] = request_id
    event["request_id"] = event["request_id"].replace("|chat|", f"|{workload}|")
    event["sample_id"] = request_id
    group = event["lookup_episodes"][0]["groups"]["0"]
    group.update(
        {
            "eligible": list(keys),
            "queried": list(keys),
            "hits": list(hits),
            "missing": list(missing),
            "pending": [],
            "retry": [],
            "observed": True,
            "no_eligible": not keys,
        }
    )
    event.update(
        {
            "offline_window_id": window_id,
            "lookup_keys": list(keys),
            "prior_stored_keys": [key for key in keys if key not in prior_evicted],
            "prior_evicted_keys": list(prior_evicted),
            "cold_keys": [],
            "lookup_no_eligible": not keys,
            "lookup_observed": True,
            "authoritative": authoritative,
            "event_integrity": "complete" if authoritative else "unavailable",
        }
    )
    return event


def _artifact_kwargs(events, *, train_end=2, train_window_id=None, test_window_id=None):
    train_window_id = train_window_id or events[0].get("offline_window_id", "train")
    test_window_id = test_window_id or events[-1].get("offline_window_id", "test")
    root = "/tmp/future-reuse-artifact"
    config = {"model": "test-model", "seed": 1}
    manifest = {
        "artifact_root": root,
        "run_id": events[0]["run_id"],
        "cell_id": events[0]["cell_id"],
        "config": config,
    }
    manifest["manifest_sha256"] = _canonical_digest(manifest)
    request_ids = [event["external_request_id"] for event in events]
    row = {
        "artifact_root": root,
        "manifest_sha256": manifest["manifest_sha256"],
        "run_id": events[0]["run_id"],
        "cell_id": events[0]["cell_id"],
        "request_id_prefix": events[0]["external_request_id"].split("sample-")[0]
        + "sample-",
        "config": config,
        "config_digest": _canonical_digest(config),
        "preservation_request_ids_json": json.dumps(request_ids, separators=(",", ":")),
        "request_ids_json": json.dumps(request_ids, separators=(",", ":")),
        "preservation_events_json": json.dumps(
            events, sort_keys=True, separators=(",", ":")
        ),
        "preservation_event_count": str(len(events)),
        "csv_sha256": "d" * 64,
    }
    row["row_sha256"] = _canonical_digest(row)
    split_metadata = {
        "schema": "future-reuse-split-v1",
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "csv_sha256": row["csv_sha256"],
        "row_sha256": row["row_sha256"],
        "row_identity": row["row_sha256"],
        "ordered_request_ids": request_ids,
        "window_ids": [
            train_window_id if index < train_end else test_window_id
            for index in range(len(request_ids))
        ],
        "train_end": train_end,
        "train_window_id": train_window_id,
        "test_window_id": test_window_id,
    }
    return {
        "artifact_manifest": manifest,
        "artifact_root": root,
        "artifact_row": row,
        "config_digest": _canonical_digest(config),
        "split_metadata": split_metadata,
    }


def _build(events, *, train_end=2, future_window=1, decoder=lambda value: value):
    kwargs = dict(
        train_end=train_end,
        future_window=future_window,
        train_window_id="train",
        test_window_id="test",
        **_artifact_kwargs(events, train_end=train_end),
    )
    if decoder is not None:
        kwargs["key_decoder"] = decoder
    return build_future_reuse_labels(
        events,
        **kwargs,
    )


def _refresh_row_digest(row):
    row.pop("row_sha256", None)
    row["row_sha256"] = _canonical_digest(row)


def _feature_trace_for(events):
    records = []
    for sequence, event in enumerate(events):
        records.append(
            {
                "schema": TRACE_SCHEMA,
                "event": TRACE_EVENT,
                "event_instance": 1,
                "expected_event_count": 1,
                "request_id": event["request_id"],
                "external_request_id": event["external_request_id"],
                "run_id": event["run_id"],
                "cell_id": event["cell_id"],
                "sample_index": event["sample_index"],
                "sequence_ordinal": sequence,
                "candidate_ordinal": 0,
                "candidate_count": 1,
                "selected": True,
                "offload_key": "01020300000007",
                "group_idx": 7,
                "cpu_block_id": 3,
                "semantic_relevance_score": None,
                "semantic_relevance_score_status": "unavailable",
                "relevance_ema": None,
                "relevance_ema_status": "unavailable",
                "rank": 0,
                "recency_norm": 1.0,
                "frequency": None,
                "frequency_status": "unavailable",
            }
        )
    return records


def _write_real_artifact(
    tmp_path, events, *, manifest_overrides=None, row_overrides=None
):
    variant_dir = tmp_path / "rag_candidate"
    variant_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant": "rag_candidate",
        "command": "offline-fixture",
        "policies": "semantic-mean",
        "workloads": "rag",
        "request-rates": "8.0",
        "seeds": "1",
        "extra-config": json.dumps({"probe_layer": "middle"}, separators=(",", ":")),
        "semantic_revision_start": "semantic-857",
        "semantic_revision_end": "semantic-857",
        "semantic_state_start": "state-semantic-857",
        "semantic_state_end": "state-semantic-857",
        "vllm_revision_start": "vllm-test",
        "vllm_revision_end": "vllm-test",
        "vllm_state_start": "state-vllm-test",
        "vllm_state_end": "state-vllm-test",
    }
    manifest.update(manifest_overrides or {})
    manifest_path = variant_dir / "variant_manifest.txt"
    manifest_path.write_text(
        "".join(f"{key}={value}\n" for key, value in manifest.items())
    )
    row = {
        "policy": "semantic-mean",
        "seed": "1",
        "workload": "rag",
        "sub_workload": "",
        "request_rate": "8.0",
        "sample_index": "0",
        "run_id": events[0]["run_id"],
        "cell_id": events[0]["cell_id"],
        "request_id_prefix": "sc|run|cell|rag|sample-",
        "request_ids_json": json.dumps(
            [event["external_request_id"] for event in events], separators=(",", ":")
        ),
        "preservation_request_ids_json": json.dumps(
            [event["external_request_id"] for event in events], separators=(",", ":")
        ),
        "preservation_events_json": json.dumps(
            events, sort_keys=True, separators=(",", ":")
        ),
        "preservation_event_count": str(len(events)),
        "preservation_join_status": "complete",
        "error": "",
    }
    row.update(row_overrides or {})
    csv_path = variant_dir / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    row_digest = _canonical_digest(row)
    split_metadata = {
        "schema": "future-reuse-split-v1",
        "artifact_manifest_sha256": manifest_digest,
        "csv_sha256": csv_digest,
        "row_sha256": row_digest,
        "row_identity": row_digest,
        "ordered_request_ids": [event["external_request_id"] for event in events],
        "window_ids": [
            "train" if index < 2 else "test" for index in range(len(events))
        ],
        "train_end": 2,
        "train_window_id": "train",
        "test_window_id": "test",
    }
    (variant_dir / "future_reuse_split.json").write_text(
        json.dumps(split_metadata, sort_keys=True, separators=(",", ":"))
    )
    return manifest_path, csv_path


def test_labels_use_later_authoritative_hits_and_evictions_only():
    events = [
        _event(
            sample_index=0,
            window_id="train",
            keys=["a", "b"],
            hits=["a"],
            missing=["b"],
            prior_evicted=["b"],
        ),
        _event(
            sample_index=1,
            window_id="train",
            keys=["a", "b"],
            hits=["a"],
            missing=["b"],
            prior_evicted=["b"],
        ),
        _event(sample_index=2, window_id="test", keys=["a", "b"]),
    ]
    events[0]["lookup_keys"] = []

    labels = _build(events)

    assert dict(labels.entries) == {"a": True, "b": True}
    assert labels.train_request_ids == (
        "sc|run|cell|chat|sample-0",
        "sc|run|cell|chat|sample-1",
    )
    assert labels.test_request_ids == ("sc|run|cell|chat|sample-2",)
    assert len(labels.source_id) == 64
    assert labels.config_id == _artifact_kwargs(events)["config_digest"]


def test_current_hit_and_test_window_cannot_leak_into_training_labels():
    events = [
        _event(sample_index=0, window_id="train", keys=["current"], hits=["current"]),
        _event(sample_index=1, window_id="train", keys=["other"], hits=["other"]),
        _event(sample_index=2, window_id="test", keys=["current"], hits=["current"]),
    ]

    batch = _build(events)
    assert dict(batch.entries) == {"current": False}


def test_absence_from_a_complete_future_window_is_false():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=[]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]

    batch = _build(events)

    assert dict(batch.entries) == {"a": False}


def test_unavailable_future_window_invalidates_labels():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(
            sample_index=1,
            window_id="train",
            keys=["a"],
            authoritative=False,
        ),
        _event(
            sample_index=2,
            window_id="train",
            keys=["a"],
            hits=["a"],
        ),
        _event(sample_index=3, window_id="test", keys=["other"]),
    ]

    with pytest.raises(ValueError, match="authoritative and complete"):
        _build(events, train_end=3, future_window=2)


def test_default_decoder_makes_json_block_references_hashable():
    events = [
        _event(sample_index=0, window_id="train", keys=["01020300000007"]),
        _event(
            sample_index=1,
            window_id="train",
            keys=["01020300000007"],
            hits=["01020300000007"],
        ),
        _event(sample_index=2, window_id="test", keys=["0304"]),
    ]

    labels = _build(events, decoder=None)

    assert dict(labels.entries) == {bytes.fromhex("01020300000007"): True}


@pytest.mark.parametrize("wire_key", ["0102030000000A", "0102 0300000007", "010"])
def test_default_decoder_rejects_noncanonical_hex(wire_key):
    events = [
        _event(sample_index=0, window_id="train", keys=[wire_key]),
        _event(sample_index=1, window_id="train", keys=[wire_key], hits=[wire_key]),
        _event(sample_index=2, window_id="test", keys=["01020300000007"]),
    ]

    with pytest.raises(ValueError, match="OffloadKey|canonical"):
        _build(events, decoder=None)


def test_future_label_batch_rejects_dataclass_rebinding():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"], hits=["a"]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]
    batch = _build(events)

    with pytest.raises(TypeError, match="unexpected keyword argument 'entries'"):
        replace(batch, entries=(("a", False),))
    with pytest.raises((FrozenInstanceError, TypeError)):
        batch.entries = ()
    with pytest.raises(TypeError):
        type(batch)(entries=())
    assert not hasattr(type(batch), "_from_validated")
    assert not hasattr(type(batch), "_digest_for")


def test_split_metadata_digest_is_part_of_immutable_seal():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"], hits=["a"]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]
    batch = _build(events)
    assert len(batch.application_id) == 64
    object.__setattr__(batch, "_split_metadata_sha256", "0" * 64)
    with pytest.raises(ValueError, match="seal"):
        _ = batch.application_id


def test_manager_rejects_tampered_seal_before_mutation():
    from semantic_offload.manager import SemanticOffloadingManager

    event = _event(sample_index=0, window_id="train", keys=["a"])
    next_event = _event(sample_index=1, window_id="train", keys=["a"])
    held_out = _event(sample_index=2, window_id="test", keys=["b"])
    batch = _build([event, next_event, held_out])
    object.__setattr__(batch, "_entries", (("a", False),))

    manager = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    manager._future_reuse_feedback = True
    manager.future_reuse_ema = {}
    manager._future_feedback_windows = set()
    manager._max_tracker_size = 8

    with pytest.raises(ValueError, match="seal"):
        manager.update_future_reuse_labels(batch)
    assert manager.future_reuse_ema == {}
    assert manager._future_feedback_windows == set()


def test_mixed_run_or_cell_sequence_is_rejected():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(
            sample_index=1,
            window_id="train",
            keys=["a"],
            run_id="other-run",
        ),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]

    with pytest.raises(ValueError, match="run/cell"):
        _build(events)


def test_unavailable_held_out_event_is_rejected():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"]),
        _event(
            sample_index=2,
            window_id="test",
            keys=["b"],
            authoritative=False,
        ),
    ]

    with pytest.raises(ValueError, match="authoritative and complete"):
        _build(events)


def test_noncontiguous_sample_indices_follow_archived_row_order():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=2, window_id="train", keys=["a"]),
        _event(sample_index=3, window_id="test", keys=["b"]),
    ]

    assert dict(_build(events).entries) == {"a": True}


def test_duplicate_request_identity_is_rejected():
    event = _event(sample_index=0, window_id="train", keys=["a"])
    duplicate = dict(event)
    duplicate["offline_window_id"] = "test"
    events = [event, duplicate]

    with pytest.raises(ValueError, match="not unique"):
        _build(events, train_end=1)


def test_event_identity_and_cardinality_are_required():
    event = _event(sample_index=0, window_id="train", keys=["a"])
    event["event_instance"] = 2
    events = [event, _event(sample_index=1, window_id="test", keys=["b"])]

    with pytest.raises(ValueError, match="identity/cardinality"):
        _build(events, train_end=1)


def test_hit_missing_contradiction_is_not_overwritten():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"], hits=["a"]),
        _event(
            sample_index=1,
            window_id="train",
            keys=["a"],
            hits=["a"],
            missing=["a"],
            prior_evicted=["a"],
        ),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]

    with pytest.raises(ValueError, match="invalid authoritative future-reuse event"):
        _build(events)


def test_shuffled_rag_uses_canonical_archived_row_order():
    key = "a"
    chronological = [
        _event(
            sample_index=30,
            window_id="train",
            workload="rag",
            keys=[key],
            hits=[key],
        ),
        _event(
            sample_index=10,
            window_id="train",
            workload="rag",
            keys=[key],
            hits=[key],
        ),
        _event(sample_index=20, window_id="test", workload="rag", keys=[]),
    ]
    artifact = _artifact_kwargs(chronological)
    shuffled_input = [chronological[2], chronological[0], chronological[1]]

    labels = build_future_reuse_labels(
        shuffled_input,
        train_end=2,
        future_window=1,
        train_window_id="train",
        test_window_id="test",
        **artifact,
        key_decoder=lambda value: value,
    )

    assert dict(labels.entries) == {key: True}
    assert labels.train_request_ids == tuple(
        event["external_request_id"] for event in chronological[:2]
    )


def test_invented_order_metadata_is_rejected():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]
    artifact = _artifact_kwargs(events)
    artifact["artifact_row"]["ordered_request_ids"] = [
        event["external_request_id"] for event in events
    ]

    with pytest.raises(ValueError, match="ordered_request_ids"):
        build_future_reuse_labels(
            events,
            train_end=2,
            future_window=1,
            train_window_id="train",
            test_window_id="test",
            **artifact,
        )


def test_supplied_event_content_must_match_archived_row():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]
    artifact = _artifact_kwargs(events)
    events[1]["lookup_keys"] = ["forged"]

    with pytest.raises(ValueError, match="conflicts with archived"):
        build_future_reuse_labels(
            events,
            train_end=2,
            future_window=1,
            train_window_id="train",
            test_window_id="test",
            **artifact,
        )


@pytest.mark.parametrize("identity", ["manifest", "row", "config"])
def test_declared_artifact_identities_must_match_recomputed_content(identity):
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]
    artifact = _artifact_kwargs(events)
    if identity == "manifest":
        artifact["artifact_manifest"]["manifest_sha256"] = "f" * 64
    elif identity == "row":
        artifact["artifact_row"]["row_sha256"] = "f" * 64
    else:
        artifact["artifact_row"]["config_digest"] = "f" * 64
        _refresh_row_digest(artifact["artifact_row"])

    with pytest.raises(ValueError, match="digest|config"):
        build_future_reuse_labels(
            events,
            train_end=2,
            future_window=1,
            train_window_id="train",
            test_window_id="test",
            **artifact,
            key_decoder=lambda value: value,
        )


def test_caller_source_and_config_declarations_must_match_artifacts():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]
    artifact = _artifact_kwargs(events)
    for field in ("source_id", "config_id"):
        with pytest.raises(ValueError, match="derived|validated"):
            build_future_reuse_labels(
                events,
                train_end=2,
                future_window=1,
                train_window_id="train",
                test_window_id="test",
                **artifact,
                **{field: "forged"},
            )


def test_missing_or_forged_split_metadata_fails_closed():
    events = [
        _event(sample_index=0, window_id="train", keys=["a"]),
        _event(sample_index=1, window_id="train", keys=["a"]),
        _event(sample_index=2, window_id="test", keys=["b"]),
    ]
    artifact = _artifact_kwargs(events)
    artifact.pop("split_metadata")
    with pytest.raises(TypeError, match="split_metadata|required"):
        build_future_reuse_labels(
            events,
            train_end=2,
            future_window=1,
            train_window_id="train",
            test_window_id="test",
            **artifact,
        )

    artifact = _artifact_kwargs(events)
    artifact["split_metadata"] = dict(artifact["split_metadata"])
    artifact["split_metadata"]["train_window_id"] = "forged"
    with pytest.raises(ValueError, match="window IDs"):
        build_future_reuse_labels(
            events,
            train_end=2,
            future_window=1,
            train_window_id="train",
            test_window_id="test",
            **artifact,
        )


def test_file_backed_evaluator_uses_split_metadata_for_v8_shuffled_rag(tmp_path):
    key = "01020300000007"
    chronological = [
        _event(
            sample_index=30,
            window_id="train",
            workload="rag",
            keys=[key],
            hits=[key],
        ),
        _event(
            sample_index=10,
            window_id="train",
            workload="rag",
            keys=[key],
            hits=[key],
        ),
        _event(sample_index=20, window_id="test", workload="rag", keys=[]),
    ]
    # v8 preservation_events_json is serving output and has no offline split
    # annotation. The split artifact is the only source of membership.
    for event in chronological:
        event.pop("offline_window_id")
    raw_snapshot = json.dumps(chronological, sort_keys=True, separators=(",", ":"))
    _write_real_artifact(tmp_path, chronological)

    labels = build_future_reuse_labels_from_artifact(
        [chronological[2], chronological[0], chronological[1]],
        artifact_root=tmp_path,
        variant="rag_candidate",
        policy="semantic-mean",
        seed=1,
        workload="rag",
        request_rate="8.0",
        train_end=2,
        future_window=1,
        train_window_id="train",
        test_window_id="test",
    )

    assert dict(labels.entries) == {bytes.fromhex(key): True}
    assert labels.train_request_ids == tuple(
        event["external_request_id"] for event in chronological[:2]
    )
    assert all("offline_window_id" not in event for event in chronological)
    assert (
        json.dumps(chronological, sort_keys=True, separators=(",", ":")) == raw_snapshot
    )
    assert len(labels.split_metadata_sha256) == 64


def test_file_backed_evaluator_round_trips_large_preservation_events_json(tmp_path):
    key = "01020300000007"
    chronological = [
        _event(
            sample_index=0,
            window_id="train",
            workload="rag",
            keys=[key],
            hits=[key],
        ),
        _event(
            sample_index=1,
            window_id="train",
            workload="rag",
            keys=[key],
            hits=[key],
        ),
        _event(sample_index=2, window_id="test", workload="rag", keys=[]),
    ]
    for event in chronological:
        event.pop("offline_window_id")
    chronological[1]["diagnostic_padding"] = "x" * 340_000
    expected_events_json = json.dumps(
        chronological, sort_keys=True, separators=(",", ":")
    )
    assert len(expected_events_json.encode("utf-8")) > 330 * 1024
    _write_real_artifact(tmp_path, chronological)

    artifact = load_future_reuse_artifact(
        tmp_path,
        variant="rag_candidate",
        policy="semantic-mean",
        seed=1,
        workload="rag",
        request_rate="8.0",
    )
    assert artifact.row["preservation_events_json"] == expected_events_json

    labels = build_future_reuse_labels_from_artifact(
        [chronological[2], chronological[0], chronological[1]],
        artifact_root=tmp_path,
        variant="rag_candidate",
        policy="semantic-mean",
        seed=1,
        workload="rag",
        request_rate="8.0",
        train_end=2,
        future_window=1,
        train_window_id="train",
        test_window_id="test",
    )
    assert dict(labels.entries) == {bytes.fromhex(key): True}


def test_file_backed_evaluator_binds_opt_in_decision_features_to_manifest_and_csv(
    tmp_path,
):
    events = [
        _event(sample_index=0, window_id="train", workload="rag", keys=["a"]),
        _event(sample_index=1, window_id="train", workload="rag", keys=["a"]),
        _event(sample_index=2, window_id="test", workload="rag", keys=[]),
    ]
    features = _feature_trace_for(events)
    manifest_path, csv_path = _write_real_artifact(
        tmp_path,
        events,
        manifest_overrides={
            "extra-config": json.dumps(
                {
                    "probe_layer": "middle",
                    "preservation_events": True,
                    "decision_feature_trace": True,
                },
                separators=(",", ":"),
            )
        },
        row_overrides={
            "decision_feature_trace_json": json.dumps(
                features, sort_keys=True, separators=(",", ":")
            ),
            "decision_feature_trace_count": str(len(features)),
            "decision_feature_trace_join_status": "complete",
        },
    )

    artifact = load_future_reuse_artifact(
        tmp_path,
        variant="rag_candidate",
        policy="semantic-mean",
        seed=1,
        workload="rag",
        request_rate="8.0",
    )

    assert artifact.row["decision_feature_trace_count"] == str(len(features))
    assert json.loads(artifact.row["decision_feature_trace_json"]) == features
    assert manifest_path.is_file() and csv_path.is_file()


def test_file_backed_evaluator_rejects_conflicting_opt_in_decision_features(tmp_path):
    events = [
        _event(sample_index=0, window_id="train", workload="rag", keys=["a"]),
        _event(sample_index=1, window_id="train", workload="rag", keys=["a"]),
        _event(sample_index=2, window_id="test", workload="rag", keys=[]),
    ]
    features = _feature_trace_for(events)
    features[0]["group_idx"] = 0
    _write_real_artifact(
        tmp_path,
        events,
        manifest_overrides={
            "extra-config": json.dumps(
                {
                    "probe_layer": "middle",
                    "preservation_events": True,
                    "decision_feature_trace": True,
                },
                separators=(",", ":"),
            )
        },
        row_overrides={
            "decision_feature_trace_json": json.dumps(
                features, sort_keys=True, separators=(",", ":")
            ),
            "decision_feature_trace_count": str(len(features)),
            "decision_feature_trace_join_status": "complete",
        },
    )
    with pytest.raises(ValueError, match="group identity"):
        load_future_reuse_artifact(
            tmp_path,
            variant="rag_candidate",
            policy="semantic-mean",
            seed=1,
            workload="rag",
            request_rate="8.0",
        )


def test_file_backed_evaluator_allows_disabled_trace_for_lru_row(tmp_path):
    events = [
        _event(sample_index=0, window_id="train", keys=["a"], workload="rag"),
        _event(sample_index=1, window_id="train", keys=["a"], workload="rag"),
        _event(sample_index=2, window_id="test", keys=[], workload="rag"),
    ]
    manifest_path, csv_path = _write_real_artifact(
        tmp_path,
        events,
        manifest_overrides={
            "policies": "lru",
            "extra-config": json.dumps(
                {"preservation_events": True, "decision_feature_trace": True},
                separators=(",", ":"),
            ),
        },
        row_overrides={
            "policy": "lru",
            "decision_feature_trace_json": "[]",
            "decision_feature_trace_count": "",
            "decision_feature_trace_join_status": "not_requested",
        },
    )

    artifact = load_future_reuse_artifact(
        tmp_path,
        variant="rag_candidate",
        policy="lru",
        seed=1,
        workload="rag",
        request_rate="8.0",
    )

    assert artifact.row["policy"] == "lru"
    assert manifest_path.is_file() and csv_path.is_file()


def test_file_backed_evaluator_rejects_forged_manifest_and_csv_provenance(tmp_path):
    events = [
        _event(sample_index=0, window_id="train", workload="rag", keys=["a"]),
        _event(sample_index=1, window_id="train", workload="rag", keys=["a"]),
        _event(sample_index=2, window_id="test", workload="rag", keys=[]),
    ]
    _write_real_artifact(
        tmp_path,
        events,
        manifest_overrides={"semantic_revision_end": "forged"},
    )
    with pytest.raises(ValueError, match="source|manifest"):
        build_future_reuse_labels_from_artifact(
            events,
            artifact_root=tmp_path,
            variant="rag_candidate",
            policy="semantic-mean",
            seed=1,
            workload="rag",
            train_end=2,
            future_window=1,
            train_window_id="train",
            test_window_id="test",
            key_decoder=lambda value: value,
        )

    _manifest_path, csv_path = _write_real_artifact(tmp_path / "csv", events)
    rows = list(csv.DictReader(csv_path.open(newline="")))
    rows[0]["csv_sha256"] = "0" * 64
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV digest"):
        build_future_reuse_labels_from_artifact(
            events,
            artifact_root=tmp_path / "csv",
            variant="rag_candidate",
            policy="semantic-mean",
            seed=1,
            workload="rag",
            train_end=2,
            future_window=1,
            train_window_id="train",
            test_window_id="test",
            key_decoder=lambda value: value,
        )
