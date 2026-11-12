import json
from pathlib import Path

import pytest

from tools.benchmarking.report import summarize_probe


def test_exposure_tracks_source_ancestry_and_retains_failed_admissions(
    tmp_path: Path,
) -> None:
    from tools.benchmarking.exposure import admit_exposure

    root = tmp_path / "root"
    (root / "benchmarks").mkdir(parents=True)
    (root / "benchmarks/workloads.json").write_text(
        json.dumps(
            {"cases": [{"instance": "org__old-1", "exposure": "feasibility-pilot"}]}
        )
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    first = tmp_path / "first"
    first.mkdir()
    jobs = [
        {"kind": "application", "instance": instance}
        for instance in ("org__old-2", "org__fresh-1")
    ]
    admit_exposure(root, assets, first, jobs)
    assert (
        jobs[0]["source_group_previously_exposed"]
        and not jobs[1]["source_group_previously_exposed"]
    )
    second = tmp_path / "second"
    second.mkdir()
    later = [{"kind": "application", "instance": "org__fresh-2"}]
    admit_exposure(root, assets, second, later)
    assert later[0]["source_group_previously_exposed"]
    assert len(list((assets / "exposure").glob("*.json"))) == 2


def test_measurement_controls_detect_injected_delay_and_protected_cohort_harm() -> None:
    from typing import Any

    from tools.benchmarking.measurement_controls import controls

    observations: dict[str, dict[str, Any]] = {}
    for model in ("qwen35-4b-int4", "gemma4-12b-int4"):
        for block in range(2):
            for arm in ("native-arc", "wait-hint"):
                for case in ("T1", "T2", "T3"):
                    item = {
                        "model": model,
                        "block": block,
                        "system": arm,
                        "case": case,
                        "kind": "trace",
                        "status": "complete",
                        "requests": [
                            {"label": label, "seconds": 1.0}
                            for label in ("cold", "resident", "unhinted-return")
                        ],
                    }
                    observations[f"{model}/{block}/{arm}/{case}"] = item
            if model == "qwen35-4b-int4":
                for case, arm, seconds in (
                    ("A1", "native-arc", 1.0),
                    ("S1", "injected-delay", 1.2),
                ):
                    observations[f"{block}/{case}"] = {
                        "model": model,
                        "block": block,
                        "system": arm,
                        "case": case,
                        "kind": "trace",
                        "status": "complete",
                        "requests": [
                            {"label": label, "seconds": seconds}
                            for label in ("cold", "resident", "unhinted-return")
                        ],
                    }
    policy = {"absolute_tolerance_seconds": 0.02, "latency_ratio_limit": 1.05}
    assert controls(observations, policy)["status"] == "passed"
    observations["qwen35-4b-int4/0/wait-hint/T2"]["requests"][2]["seconds"] = 1.2
    result = controls(observations, policy)
    assert (
        result["status"] == "failed"
        and sum(not row["passed"] for row in result["protected_and_noop"]) == 1
    )
    observations["0/S1"]["requests"][0]["seconds"] = 1.0
    observations["0/S1"]["requests"][1]["seconds"] = 1.0
    assert not controls(observations, policy)["injected_delay"][0]["passed"]


def test_paired_report_keeps_censored_and_quality_regressing_outcomes() -> None:
    from tools.benchmarking.panel_report import paired_observations

    baseline = {
        "kind": "application",
        "model": "model",
        "case": "case",
        "block": 0,
        "system": "native-arc",
        "status": "complete",
        "quality": "resolved",
        "elapsed_seconds": 10,
    }
    candidate = {
        **baseline,
        "system": "wait-hint",
        "status": "failed",
        "quality": "unresolved",
        "elapsed_seconds": 2,
    }
    result = paired_observations({"control": baseline, "candidate": candidate})[0]
    assert result["quality_regression"]
    assert not result["complete_pair"]
    assert result["elapsed_ratio"] is None
    candidate.update(status="complete", elapsed_seconds=8)
    assert (
        paired_observations({"control": baseline, "candidate": candidate})[0][
            "elapsed_ratio"
        ]
        == 0.8
    )


def test_campaign_recovers_dead_owner_without_retrying_interrupted_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.benchmarking.campaign_recovery import (
        interrupted_results,
        recover_sessions,
        start_invocation,
    )

    (tmp_path / "admission.json").write_text("{}")
    session = tmp_path / "sessions/0"
    session.mkdir(parents=True)
    start_invocation(session, 1000)
    (session / "launch-000.json").write_text(
        json.dumps(
            {
                "directory": "sessions/0/0",
                "identities": ["interrupted"],
                "budget_seconds": 600,
            }
        )
    )
    with pytest.raises(ValueError, match="still running"):
        recover_sessions(tmp_path)
    monkeypatch.setattr(
        "tools.benchmarking.campaign_recovery.owner_alive", lambda _identity: False
    )
    recover_sessions(tmp_path)
    receipt = tmp_path / "terminal-000.json"
    saved = receipt.read_bytes()
    record = json.loads(saved)
    assert 600 <= record["elapsed_seconds"] <= 1000
    assert interrupted_results(tmp_path, record)[0]["status"] == "failed"
    recover_sessions(tmp_path)
    assert receipt.read_bytes() == saved
    (session / "launch-000.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt drift"):
        interrupted_results(tmp_path, record)


def test_resource_capture_records_unavailable_gpu_metrics_and_checks_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    from tools.benchmarking.resources import sample_resources, verify_resources

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("nvidia-smi unavailable")

    monkeypatch.setattr(
        "tools.benchmarking.resources.process_memory", lambda _session: {"1": 1024}
    )
    monkeypatch.setattr("tools.benchmarking.resources.subprocess.run", unavailable)
    record = sample_resources(1)
    assert record["sum_rss_bytes"] == 1024 and not record["complete"]
    destination = tmp_path / "resources.json"
    destination.write_text(json.dumps(record))
    assert verify_resources(destination) == record
    record["sum_rss_bytes"] = 0
    destination.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="invalid resource"):
        verify_resources(destination)


def test_report_requires_complete_requests_and_separates_hint_from_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = {
        "status": "restoration_observed",
        "elapsed_seconds": 30,
        "coding": {"quality": "unresolved", "execution_exit": 0, "elapsed_seconds": 20},
    }
    monkeypatch.setattr("tools.benchmarking.report.verify_probe", lambda path: terminal)
    (tmp_path / "admission.json").write_text(
        json.dumps({"system": "wait-hint", "coding": {"instance": "case"}})
    )
    requests = tmp_path / "requests"
    requests.mkdir()
    with pytest.raises(ValueError, match="incomplete request capture"):
        summarize_probe(tmp_path)
    (requests / "request-0001.admission.json").write_text(
        json.dumps({"generation": 1, "start_ns": 10})
    )
    record = {
        "schema": "benchmark-request-v1",
        "system": "wait-hint",
        "start_ns": 10,
        "received_ns": 1_000_000_010,
        "forwarded_ns": 1_100_000_010,
        "response": {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "metrics": {"time_to_first_token_ms": 500},
        },
        "compaction": {"removed_messages": 2},
        "hint": {"generation": 1},
        "acknowledgement": {"decision": "refreshed", "ready_cpu_prefix_keys": 4},
    }
    capture = requests / "request-0001.json"
    capture.write_text(json.dumps(record))
    result = summarize_probe(tmp_path)["coding"]
    assert result["quality"] == "unresolved"
    assert result["requests"] == 1
    assert result["counts"]["cached_tokens"] == 0
    assert result["counts"]["hint:refreshed"] == 1
    assert result["counts"]["compacted_turns"] == 1
    assert result["request_seconds"]["median"] == 1.1
    record["received_ns"] = 5
    capture.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="time ordering"):
        summarize_probe(tmp_path)
    (requests / "request-0002.admission.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete request capture"):
        summarize_probe(tmp_path)
