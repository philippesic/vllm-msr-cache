from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.benchmarking.pressure import (
    cohort_summary,
    contracts,
    load_definition,
    model_controls,
    projection,
    sessions_for,
    verify_pressure,
)
from tools.benchmarking.workload import load_shapes

ROOT = Path(__file__).resolve().parents[1]
DEFINITION = load_definition(ROOT)
ARMS = DEFINITION["arms"]


def row(
    turn: int,
    cohort: str,
    shared: int,
    prompt: int,
    cached: int,
    ttft: float = 100.0,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "session": "s",
        "turn": turn,
        "cohort": cohort,
        "abandons": False,
        "shared_tokens": shared,
        "prompt_tokens": prompt,
        "usage": {"prompt_tokens_details": {"cached_tokens": cached}},
        "metrics": {"time_to_first_token_ms": ttft},
        "wants_hint": False,
        "hint": None,
        **extra,
    }


def summary(
    hinted: tuple[int, int],
    unhinted: tuple[int, int],
    seed: int = 1,
    **changes: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "seed": seed,
        "status": "complete",
        "cohorts": {
            "hinted": {
                "returns": 4,
                "reusable_tokens": hinted[0],
                "recomputed_tokens": hinted[1],
            },
            "unhinted": {
                "returns": 2,
                "reusable_tokens": unhinted[0],
                "recomputed_tokens": unhinted[1],
            },
        },
        "advice": {
            "hints_wanted": 3,
            "hints_published": 3,
            "abandoned_hints": 1,
            "decisions": {"refreshed": 3},
            "mechanisms": {"reachable-hold": 3},
            "held_keys": 12,
            "refreshed_keys": 12,
        },
        "stored_keys": 300.0,
        "loaded_bytes": 10.0,
        "store_failures": 0.0,
        "pool_keys": 100,
        "largest_conversation_tokens": 4000,
        "distinct_tokens": 30000,
        "restoration": {"restore_observed": True, "keys_per_token": 0.0075},
        "valid": True,
    }
    return {**base, **changes}


def test_cohort_summary_uses_aligned_reusable_prefix_and_engine_ttft() -> None:
    rows = [
        row(0, "hinted", 0, 1000, 0),
        row(1, "hinted", 1000, 1500, 640, 200.0),
        row(2, "hinted", 1400, 1401, 1408, 100.0),
        row(1, "unhinted", 500, 600, 512),
    ]
    result = cohort_summary(rows, 64)
    assert result["hinted"] == {
        "returns": 2,
        "reusable_tokens": 960 + 1344,
        "recomputed_tokens": 320,
        "recomputed_share": round(320 / 2304, 4),
        "mean_ttft_ms": 150.0,
        "median_ttft_ms": 150.0,
    }
    assert result["unhinted"]["reusable_tokens"] == 448
    assert result["unhinted"]["recomputed_tokens"] == 0


def test_activity_contracts_detect_advice_off_small_pool_and_no_competition() -> None:
    keys_per_token = 0.0075
    good = contracts(summary((1000, 100), (500, 50)), ARMS["wait-hint"], keys_per_token)
    assert all(good.values())
    silent = summary(
        (1000, 100),
        (500, 50),
        advice={
            **summary((1, 1), (1, 1))["advice"],
            "decisions": {"disabled": 3},
            "mechanisms": {"disabled": 3},
            "held_keys": 0,
        },
    )
    checks = contracts(silent, ARMS["wait-hint"], keys_per_token)
    assert checks["advice_mechanism"] is False
    assert checks["advice_applied"] is False and checks["holds_applied"] is False
    small = contracts(
        summary((1000, 100), (500, 50), pool_keys=20), ARMS["native-arc"], 0.0075
    )
    assert small["pool_holds_one_conversation"] is False
    quiet = contracts(
        summary((1000, 0), (500, 0), stored_keys=50.0, loaded_bytes=0.0),
        ARMS["native-arc"],
        keys_per_token,
    )
    assert quiet["cpu_evictions"] is False and quiet["cpu_restores"] is False
    oracle = contracts(summary((1000, 0), (500, 0)), ARMS["oracle"], keys_per_token)
    assert oracle["oracle_pool_never_full"] is False and "cpu_evictions" not in oracle
    roomy = summary((1000, 0), (500, 0), pool_keys=400)
    assert contracts(roomy, ARMS["oracle"], keys_per_token)["oracle_pool_never_full"]
    idle = contracts(summary((1000, 0), (500, 0)), ARMS["gpu-only"], None)
    assert idle["gpu_displacement"] is False
    lmcache = contracts(summary((1000, 100), (500, 50)), ARMS["lmcache"], None)
    assert lmcache["cpu_restores"] is None and lmcache["cpu_evictions"] is None
    missing = contracts(
        summary((1000, 100), (500, 50), advice={**good_advice(), "abandoned_hints": 0}),
        ARMS["native-arc"],
        keys_per_token,
    )
    assert missing["abandoned_hints"] is False


def good_advice() -> dict[str, Any]:
    advice: dict[str, Any] = summary((1, 1), (1, 1))["advice"]
    return advice


def test_controls_fail_closed_without_separation_or_with_noisy_repeat() -> None:
    arms = {
        "gpu-only": [summary((1000, 600), (500, 300))],
        "oracle": [summary((1000, 50), (500, 20))],
        "native-arc": [summary((1000, 300), (500, 150))],
        "native-arc-repeat": [summary((1000, 320), (500, 150))],
    }
    controls = model_controls(arms, 0.1)
    assert controls["status"] == "discriminating"
    assert controls["separation"] == 0.55 and controls["aa_difference"] == 0.02
    flat = {**arms, "gpu-only": [summary((1000, 100), (500, 50))]}
    assert model_controls(flat, 0.1)["status"] == "non-discriminating"
    noisy = {**arms, "native-arc-repeat": [summary((1000, 900), (500, 150))]}
    assert model_controls(noisy, 0.1)["status"] == "non-discriminating"
    invalid = {**arms, "oracle": [summary((1000, 50), (500, 20), valid=False)]}
    assert model_controls(invalid, 0.1)["status"] == "controls-missing"


def write_cell(directory: Path, model: str, arm: str, cell: dict[str, Any]) -> None:
    path = directory / "cells" / model / arm
    path.mkdir(parents=True, exist_ok=True)
    (path / f"cell-{cell['seed']}.json").write_text(json.dumps(cell))


def test_verifier_recomputes_from_raw_requests_and_withholds_blind_comparisons(
    tmp_path: Path,
) -> None:
    (tmp_path / "admission.json").write_text(json.dumps({"definition": DEFINITION}))
    seed = DEFINITION["seeds"]["smoke"][0]
    sessions = sessions_for(DEFINITION, seed, "qwen35-4b-int4")
    model = "qwen35-4b-int4"
    for arm in ("gpu-only", "native-arc", "native-arc-repeat", "oracle"):
        rows = [
            row(
                index,
                "hinted" if session.hinted else "unhinted",
                turn.shared,
                turn.prompt,
                turn.shared,
                session=session.name,
                abandons=session.abandons,
                wants_hint=session.hinted,
            )
            for session in sessions
            if len(session.turns) > 1
            for index, turn in enumerate(session.turns)
        ]
        write_cell(
            tmp_path,
            model,
            arm,
            {
                "arm": arm,
                "seed": seed,
                "status": "complete",
                "pool_bytes": 0,
                "requests": rows,
                "restoration": None,
                "metrics_before": {},
                "metrics_after": {},
            },
        )
    report = verify_pressure(tmp_path)["models"][model]
    assert report["controls"]["status"] == "controls-missing"
    gpu = report["arms"]["gpu-only"]["cells"][0]
    assert gpu["state"] == "unexercised"
    assert gpu["contracts"]["gpu_displacement"] is False
    assert report["arms"]["gpu-only"]["pooled_hinted_share"] == 0.0


def test_release_01_evidence_is_rejected_as_unexercised() -> None:
    shapes = load_shapes(ROOT / "benchmarks/replay/release-01-shapes.json")
    for model, recorded in shapes.items():
        rows = [
            row(index, "hinted", turn.shared, turn.prompt, turn.shared)
            for session in recorded
            for index, turn in enumerate(session)
        ]
        alignment = DEFINITION["models"][model]["hit_alignment"]
        cohorts = cohort_summary(rows, alignment)
        assert cohorts["hinted"]["returns"] > 300
        assert cohorts["hinted"]["recomputed_tokens"] == 0
        assert cohorts["unhinted"]["returns"] == 0
        evidence = summary(
            (cohorts["hinted"]["reusable_tokens"], 0),
            (0, 0),
            cohorts={**cohorts},
        )
        assert contracts(evidence, ARMS["gpu-only"], None) == {
            "complete": True,
            "hinted_returns": True,
            "unhinted_returns": False,
            "abandoned_hints": True,
            "gpu_displacement": False,
        }


def test_profile_seeds_cover_every_cohort_before_gpu_work() -> None:
    from tools.benchmarking.pressure import covers_cohorts, profile_seeds

    for model in DEFINITION["models"]:
        for profile in DEFINITION["seeds"]:
            seeds = profile_seeds(DEFINITION, profile, model, [])
            assert all(covers_cohorts(DEFINITION, seed, model) for seed in seeds)
    assert profile_seeds(DEFINITION, "full", "gemma4-12b-int4", [])[0] == 305
    assert profile_seeds(DEFINITION, "full", "gemma4-12b-int4", [312]) == [312]
    assert not covers_cohorts(DEFINITION, 101, "qwen35-4b-int4")


def test_plan_projects_cost_from_measured_cells(tmp_path: Path) -> None:
    budget = projection(ROOT, "full", [])
    assert budget["projected_gpu_hours"] > 0
    assert {row["basis"] for row in budget["cells"]} == {"budget"}
    write_cell(
        tmp_path,
        "qwen35-4b-int4",
        "native-arc",
        {"seed": 1, "startup_seconds": 60.0, "workload_seconds": 200.0},
    )
    measured = projection(ROOT, "full", [tmp_path])
    row_ = next(
        r
        for r in measured["cells"]
        if r["model"] == "qwen35-4b-int4" and r["arm"] == "native-arc"
    )
    assert row_["basis"] == "measured" and row_["seconds"] == 60 + 30 + 3 * 200
