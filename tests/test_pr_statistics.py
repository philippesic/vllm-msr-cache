from __future__ import annotations

import copy
import statistics
from typing import Any

import pytest

from tools.benchmarking.pr_statistics import (
    bca_interval,
    compare,
    pooled,
    route_difference,
    summarize_cohort,
    validate_pairs,
)


def _summary(sessions: int, total: float, count: int) -> dict[str, Any]:
    return {
        "weights": {"s": 1},
        "strata": {
            "s": {
                "sessions": sessions,
                "numerator": total,
                "denominator": count,
                "failures": 0,
            }
        },
    }


def test_screen_pooling_weights_counted_returns_and_has_correct_cluster_influence() -> (
    None
):
    inputs = [_summary(1, 20, 2), _summary(2, 300, 10), _summary(1, 40, 4)]
    result = pooled(inputs)
    assert result["value"] == 360 / 16
    assert result["value"] != statistics.mean([10, 30, 10])
    assert sum(result["influence"]) == pytest.approx(0)
    for index in range(3):
        perturbed = copy.deepcopy(inputs)
        for key in ("sessions", "numerator", "denominator"):
            perturbed[index]["strata"]["s"][key] *= 1.000001
        derivative = (pooled(perturbed)["value"] - result["value"]) / 0.000001
        assert derivative == pytest.approx(result["influence"][index], rel=0.00001)
    stratified = [
        {
            "weights": {"s": 0.9, "t": 0.1},
            "strata": {
                "s": _summary(1, 20, 2)["strata"]["s"],
                "t": _summary(9, 900, 9)["strata"]["s"],
            },
        },
        {
            "weights": {"s": 0.9, "t": 0.1},
            "strata": {
                "s": _summary(3, 180, 6)["strata"]["s"],
                "t": _summary(1, 200, 2)["strata"]["s"],
            },
        },
    ]
    assert pooled(stratified)["value"] == pytest.approx(
        (0.9 * 200 / 4 + 0.1 * 1100 / 10) / (0.9 * 8 / 4 + 0.1 * 11 / 10)
    )


def test_bca_keeps_undefined_replicates_and_degenerate_evidence_unqualified() -> None:
    assert bca_interval(2, lambda indices: 1.0, "few")["interval"] is None
    assert bca_interval(4, lambda indices: 1.0, "constant")["interval"] is None
    partial = bca_interval(
        3,
        lambda indices: None if len(set(indices)) == 1 else float(sum(indices)),
        "missing",
    )
    assert partial["interval"] is None and partial["undefined_replicates"] > 0
    values = [1.0, 1.5, 2.0, 2.2, 3.0, 4.0, 4.2, 7.0]

    def statistic(indices: list[int]) -> float:
        return statistics.mean(values[index] for index in indices)

    first = bca_interval(8, statistic, "fixed")
    assert first == bca_interval(8, statistic, "fixed")
    assert first["interval"][0] < first["point"] < first["interval"][1]
    assert first["resamples"] == 10000


def test_paired_latency_ratio_and_censoring_cannot_create_support() -> None:
    baseline = [_summary(1, value, 1) for value in (1, 2, 4, 8)]
    treatment = [_summary(1, value, 1) for value in (1, 1, 2, 4)]
    result = compare(treatment, baseline, "benefit", "paired")
    assert result["point"] == pytest.approx(1 - 8 / 15)
    assert result["delta_interval"][0] < result["point"] < result["delta_interval"][1]
    treatment[0]["strata"]["s"]["failures"] = 1
    assert pooled(treatment)["value"] is None
    assert pooled(treatment, descriptive=True)["value"] == 2
    censored = compare(treatment, baseline, "benefit", "censored")
    assert censored["interval"] is None
    assert censored["descriptive_treatment"]["planned_sessions"] == 4
    assert censored["descriptive_treatment"]["failed_or_unoffered_requests"] == 1


def test_cohort_summary_preserves_failed_and_unoffered_return_denominators() -> None:
    rows = [
        {
            "cohort": "hinted_return",
            "stratum": "s",
            "session_id": "a",
            "outcome": "complete",
            "ttft_ns": 10,
        },
        {
            "cohort": "hinted_return",
            "stratum": "s",
            "session_id": "b",
            "outcome": "not_offered",
            "ttft_ns": None,
        },
    ]
    summary = summarize_cohort(
        {"requests": rows, "stratum_weights": {"s": 1}}, "hinted_return", "ttft_ns"
    )
    assert summary["strata"]["s"] == {
        "sessions": 2,
        "numerator": 10,
        "denominator": 1,
        "failures": 1,
    }
    assert pooled([summary])["value"] is None


def test_route_gap_uses_common_baseline_and_duplicate_seeds_are_refused() -> None:
    baseline = [_summary(1, value, 1) for value in (10, 20, 30, 40)]
    ttl = [_summary(1, value, 1) for value in (9, 20, 30, 40)]
    post = [_summary(1, value, 1) for value in (5, 10, 20, 30)]
    gap = route_difference(post, ttl, baseline, "route")
    assert gap["point"] == pytest.approx((99 - 65) / 100)
    assert gap["interval"] is not None and gap["delta_interval"] is not None
    cells = [
        {"model": "D8", "partition": "pilot", "seed": seed, "arm": "A2"}
        for seed in (12101, 12102)
    ]
    validate_pairs(cells, cells)
    with pytest.raises(ValueError, match="unique"):
        validate_pairs(cells + cells)
    with pytest.raises(ValueError, match="ordered"):
        validate_pairs(cells, cells[::-1])
