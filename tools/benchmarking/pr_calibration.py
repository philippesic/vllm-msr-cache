from __future__ import annotations

import math
import random
import statistics
from typing import Any

from tools.benchmarking.pr_statistics import pooled, quantile, summarize_cohort
from tools.benchmarking.pr_workload import ARMS, seeded_order


def sizing_settings(cells: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(cells) != 3
        or {cell["seed"] for cell in cells} != {12001, 12002, 12003}
        or len({cell["model"] for cell in cells}) != 1
        or any(
            cell["arm"] != "A2" or cell["partition"] != "calibration" for cell in cells
        )
    ):
        raise ValueError("three native ARC calibration seeds required")
    durations, means = [], []
    offered = 0
    for cell in cells:
        sessions = [session for session in cell["sessions"] if not session["one_shot"]]
        if len(sessions) != 16 or len(cell["sessions"]) != 16:
            raise ValueError("sizing requires sixteen sessions and no one-shots")
        measured = [
            (session["ended_ns"] - session["scheduled_arrival_ns"]) / 1e9
            for session in sessions
        ]
        if any(not math.isfinite(value) or value <= 0 for value in measured):
            raise ValueError("invalid sizing duration")
        durations.extend(measured)
        means.append(statistics.mean(measured))
        offered += sum(session["offered_turns"] for session in sessions)
    rate = 8 / statistics.median(means)
    final_arrivals = {}
    for seed in [12001, 12002, 12003, *range(12101, 12109)]:
        rng = random.Random(int(seeded_order(seed, "arrivals", ""), 16))
        final_arrivals[seed] = sum(rng.expovariate(rate) for _ in range(16))
    p95 = quantile(sorted(durations), 0.95)
    horizon = min(900, max(120, math.ceil(max(final_arrivals.values()) + 2 * p95)))
    if max(final_arrivals.values()) >= horizon or not 0 < rate <= 100:
        raise ValueError(
            "sizing cannot admit sixteen arrivals within the frozen limits"
        )
    return {
        "schema": "pr-screen-sizing-v1",
        "model": cells[0]["model"],
        "arrival_rate": rate,
        "one_shot_rate": 0.1 * offered / (3 * 240),
        "horizon_seconds": horizon,
        "seed_mean_session_seconds": means,
        "pooled_p95_session_seconds": p95,
        "final_arrivals_seconds": final_arrivals,
        "failed_sessions": sum(cell["session_failures"] for cell in cells),
        "offered_requests": offered,
    }


def rank_pools(working_sets: list[float], largest_bytes: int) -> dict[str, Any]:
    if (
        len(working_sets) != 3
        or any(not math.isfinite(value) or value < 0 for value in working_sets)
        or largest_bytes <= 0
    ):
        raise ValueError(
            "three finite working sets and a positive largest conversation required"
        )
    mean = statistics.median(working_sets)
    candidates = sorted(
        [gib for gib in (1, 2, 4) if gib * 1024**3 >= largest_bytes],
        key=lambda gib: (abs(gib * 1024**3 - mean / 2), gib),
    )
    return {
        "median_mean_working_set_bytes": mean,
        "largest_conversation_bytes": largest_bytes,
        "pool_order_gib": candidates,
        "resource_status": "needs all-arm validation"
        if candidates
        else "no candidate holds largest conversation",
    }


def select_baseline(cells: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(cells) != 21
        or {(cell["arm"], cell["seed"]) for cell in cells}
        != {(arm, seed) for arm in ARMS for seed in (12001, 12002, 12003)}
        or len({cell["model"] for cell in cells}) != 1
        or any(cell["partition"] != "calibration" for cell in cells)
    ):
        raise ValueError("three complete seven-arm calibration blocks required")
    shares = {
        arm: pooled(
            [
                summarize_cohort(cell, "hinted_return", "recompute")
                for cell in cells
                if cell["arm"] == arm
            ]
        )["value"]
        for arm in ("A1", "A2")
    }
    if any(value is None for value in shares.values()):
        raise ValueError(
            "censored or missing baseline recomputation cannot select a baseline"
        )
    return {
        "baseline": "A1" if shares["A1"] < shares["A2"] else "A2",
        "pooled_hinted_recompute": shares,
    }


def block_quote(
    horizon: float, startup_cleanup: float, remaining: float
) -> dict[str, Any]:
    if (
        any(
            not math.isfinite(value) or value < 0
            for value in (horizon, startup_cleanup, remaining)
        )
        or horizon <= 0
    ):
        raise ValueError("finite positive horizon and nonnegative allowances required")
    per_job = max(30, math.ceil(horizon + startup_cleanup))
    reserved = len(ARMS) * (per_job + 60)
    return {
        "per_job_ceiling_seconds": per_job,
        "complete_block_reserved_seconds": reserved,
        "remaining_screen_seconds": remaining,
        "fits": per_job <= 1800 and reserved <= remaining,
        "scope": "seven complete arms including cleanup contingency; re-quote from the first complete decision block",
    }
