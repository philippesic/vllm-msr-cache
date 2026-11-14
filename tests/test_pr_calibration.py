from __future__ import annotations

import copy
from typing import Any

import pytest

from tools.benchmarking.pr_calibration import (
    block_quote,
    rank_pools,
    select_baseline,
    sizing_settings,
)
from tools.benchmarking.pr_workload import ARMS


def test_sizing_uses_failed_session_duration_and_fixed_arrival_seeds() -> None:
    cells: list[dict[str, Any]] = [
        {
            "model": "D8",
            "partition": "calibration",
            "seed": seed,
            "arm": "A2",
            "session_failures": 1,
            "sessions": [
                {
                    "one_shot": False,
                    "scheduled_arrival_ns": 1000000000,
                    "ended_ns": 11000000000,
                    "offered_turns": 2,
                }
                for _ in range(16)
            ],
        }
        for seed in (12001, 12002, 12003)
    ]
    result = sizing_settings(cells)
    assert result["arrival_rate"] == 0.8
    assert result["one_shot_rate"] == pytest.approx(9.6 / 720)
    assert result["failed_sessions"] == 3 and result["horizon_seconds"] == 120
    assert set(result["final_arrivals_seconds"]) == {
        12001,
        12002,
        12003,
        *range(12101, 12109),
    }
    cells[0]["sessions"][0]["ended_ns"] = 121000000000
    assert sizing_settings(cells)["seed_mean_session_seconds"][0] > 10


def test_pool_ranking_and_budget_quote_do_not_use_benefit() -> None:
    gib = 1024**3
    assert rank_pools([3 * gib] * 3, gib)["pool_order_gib"] == [1, 2, 4]
    assert rank_pools([3 * gib] * 3, gib + 1)["pool_order_gib"] == [2, 4]
    assert rank_pools([0] * 3, 5 * gib)["pool_order_gib"] == []
    assert block_quote(120, 30, 1470)["fits"] is True
    assert block_quote(120, 30, 1469)["fits"] is False


def test_baseline_selection_requires_all_calibration_arms_and_keeps_arc_ties() -> None:
    cells: list[dict[str, Any]] = [
        {
            "model": "D8",
            "partition": "calibration",
            "seed": seed,
            "arm": arm,
            "stratum_weights": {"s": 1},
            "requests": [
                {
                    "cohort": "hinted_return",
                    "stratum": "s",
                    "session_id": "a",
                    "outcome": "complete",
                    "eligible_tokens": 100,
                    "sources": {"unobserved_tokens": 0, "recomputed_tokens": 30},
                }
            ],
        }
        for arm in ARMS
        for seed in (12001, 12002, 12003)
    ]
    assert select_baseline(cells)["baseline"] == "A2"
    changed = copy.deepcopy(cells)
    changed[0]["requests"][0]["sources"]["recomputed_tokens"] = 20
    assert select_baseline(changed)["baseline"] == "A1"
    with pytest.raises(ValueError, match="complete"):
        select_baseline(cells[:-1])
    changed[0]["requests"][0]["outcome"] = "not_offered"
    with pytest.raises(ValueError, match="censored"):
        select_baseline(changed)
