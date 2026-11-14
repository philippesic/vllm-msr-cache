from __future__ import annotations

from typing import Any

import pytest

from tools.benchmarking.pr_verdict import HARM_FLOORS, screen_verdict


def _estimate(point: float, low: float, high: float) -> dict[str, Any]:
    return {"point": point, "interval": [low, high], "delta_interval": [low, high]}


def _evidence() -> dict[str, Any]:
    return {
        "schema": "pr-screen-decision-input-v1",
        "route": "A4p",
        "model": "D8",
        "workload": "W1",
        "blocking_reasons": [],
        "valid": True,
        "hint_off_identity": True,
        "aa_diagnosed": True,
        "benefit": _estimate(0.2, 0.05, 0.35),
        "recompute": _estimate(-0.1, -0.15, -0.05),
        "harms": {metric: _estimate(0.0, -0.01, 0.01) for metric in HARM_FLOORS},
        "aa": {metric: _estimate(0.0, -0.01, 0.01) for metric in HARM_FLOORS},
        "baseline_recompute_share": 0.3,
        "evidence_paths": ["fixture"],
    }


def test_route_decision_preserves_precedence_and_requires_all_harms() -> None:
    evidence = _evidence()
    assert screen_verdict(evidence)["verdict"] == "CONTINUE"
    evidence["harms"]["one_shot_itl"] = _estimate(0.051, 0, 0.09)
    assert screen_verdict(evidence)["verdict"] == "INCONCLUSIVE"
    evidence["benefit"] = _estimate(0.01, -0.05, 0.1)
    assert screen_verdict(evidence)["verdict"] == "STOP"
    evidence["baseline_recompute_share"] = 0.19
    assert screen_verdict(evidence)["verdict"] == "INCONCLUSIVE"
    evidence["blocking_reasons"] = ["route-induced resource exhaustion"]
    assert screen_verdict(evidence)["basis"] == "blocking failure"


def test_sensitivity_disagreement_and_missing_identity_cannot_promote() -> None:
    evidence = _evidence()
    evidence["benefit"]["delta_interval"] = [-0.01, 0.3]
    assert screen_verdict(evidence)["verdict"] == "INCONCLUSIVE"
    evidence = _evidence()
    evidence["aa"]["unhinted_recompute"]["interval"] = None
    assert screen_verdict(evidence)["verdict"] == "INCONCLUSIVE"
    evidence = _evidence()
    evidence["hint_off_identity"] = False
    assert screen_verdict(evidence)["verdict"] == "INCONCLUSIVE"
    evidence["hint_off_identity"] = "passed"
    with pytest.raises(ValueError, match="qualifications"):
        screen_verdict(evidence)
