from __future__ import annotations

import math
from typing import Any

HARM_FLOORS = {
    "unhinted_recompute": 0.02,
    "unhinted_ttft": 0.05,
    "unhinted_itl": 0.05,
    "one_shot_ttft": 0.05,
    "one_shot_itl": 0.05,
}


def _interval(estimate: dict[str, Any], method: str) -> list[float] | None:
    value = estimate[method]
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            type(item) not in {float, int} or not math.isfinite(item) for item in value
        )
        or value[0] > value[1]
    ):
        raise ValueError("invalid confidence interval")
    return value


def _method_decision(evidence: dict[str, Any], method: str) -> dict[str, Any]:
    benefit, recompute = evidence["benefit"], evidence["recompute"]
    benefit_ci, recompute_ci = _interval(benefit, method), _interval(recompute, method)
    opportunity = evidence["baseline_recompute_share"]
    if benefit_ci is None or benefit["point"] is None:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "missing benefit interval",
            "criteria": {},
        }
    criteria = {
        "efficacy_stop": opportunity is not None
        and opportunity >= 0.2
        and benefit_ci[1] < 0.15,
        "worthwhile_point": benefit["point"] >= 0.15,
        "positive_benefit": benefit_ci[0] > 0,
        "recompute_reduction": recompute_ci is not None and recompute_ci[1] < 0,
    }
    margins = {}
    missing = []
    for metric, floor in HARM_FLOORS.items():
        harm, aa = evidence["harms"][metric], evidence["aa"][metric]
        harm_ci, aa_ci = _interval(harm, method), _interval(aa, method)
        if harm_ci is None or aa_ci is None or harm["point"] is None:
            missing.append(metric)
            criteria[metric] = False
            continue
        margin = max(floor, (aa_ci[1] - aa_ci[0]) / 2)
        margins[metric] = margin
        criteria[metric] = harm["point"] <= margin and harm_ci[1] <= 2 * margin
    if criteria["efficacy_stop"]:
        verdict, reason = (
            "STOP",
            "benefit interval excludes the minimum worthwhile benefit with opportunity",
        )
    elif all(value for key, value in criteria.items() if key != "efficacy_stop"):
        verdict, reason = (
            "CONTINUE",
            "benefit, recomputation and all harm thresholds pass",
        )
    else:
        verdict, reason = (
            "INCONCLUSIVE",
            "required efficacy or harm evidence is insufficient",
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "criteria": criteria,
        "harm_margins": margins,
        "missing_harm_intervals": missing,
    }


def screen_verdict(evidence: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "route",
        "model",
        "workload",
        "blocking_reasons",
        "valid",
        "hint_off_identity",
        "aa_diagnosed",
        "benefit",
        "recompute",
        "harms",
        "aa",
        "baseline_recompute_share",
        "evidence_paths",
    }
    if set(evidence) != fields or evidence["schema"] != "pr-screen-decision-input-v1":
        raise ValueError("closed screen decision input required")
    if (
        evidence["route"] not in {"A4t", "A4p"}
        or evidence["model"] not in {"D8", "HM", "HS"}
        or evidence["workload"] not in {"W1", "W2", "W2-12s"}
        or set(evidence["harms"]) != set(HARM_FLOORS)
        or set(evidence["aa"]) != set(HARM_FLOORS)
        or any(
            type(evidence[field]) is not bool
            for field in ("valid", "hint_off_identity", "aa_diagnosed")
        )
        or not isinstance(evidence["blocking_reasons"], list)
        or any(
            not isinstance(reason, str) or not reason
            for reason in evidence["blocking_reasons"]
        )
        or not isinstance(evidence["evidence_paths"], list)
        or not evidence["evidence_paths"]
    ):
        raise ValueError("invalid screen decision identity or qualifications")
    opportunity = evidence["baseline_recompute_share"]
    if opportunity is not None and (
        type(opportunity) not in {float, int} or not 0 <= opportunity <= 1
    ):
        raise ValueError("invalid opportunity share")
    for estimate in [
        evidence["benefit"],
        evidence["recompute"],
        *evidence["harms"].values(),
        *evidence["aa"].values(),
    ]:
        point = estimate["point"]
        if point is not None and (
            type(point) not in {float, int} or not math.isfinite(point)
        ):
            raise ValueError("invalid estimate point")
        for method in ("interval", "delta_interval"):
            interval = _interval(estimate, method)
            if point is None and interval is not None:
                raise ValueError("confidence interval lacks its point estimate")
    base = {
        "schema": "pr-screen-decision-v1",
        "route": evidence["route"],
        "model": evidence["model"],
        "workload": evidence["workload"],
        "opportunity": None if opportunity is None else opportunity >= 0.2,
        "evidence_paths": evidence["evidence_paths"],
        "scope": "decision arithmetic; supplied validity, identity and attribution still require receipt review",
    }
    if evidence["blocking_reasons"]:
        return base | {
            "verdict": "STOP",
            "basis": "blocking failure",
            "reasons": evidence["blocking_reasons"],
        }
    if not all(
        evidence[field] for field in ("valid", "hint_off_identity", "aa_diagnosed")
    ):
        return base | {
            "verdict": "INCONCLUSIVE",
            "basis": "missing validity, identity or A/A diagnosis",
        }
    primary, sensitivity = [
        _method_decision(evidence, method) for method in ("interval", "delta_interval")
    ]
    disagree = (
        primary["criteria"] != sensitivity["criteria"]
        or primary["verdict"] != sensitivity["verdict"]
    )
    return base | {
        "verdict": "INCONCLUSIVE" if disagree else primary["verdict"],
        "basis": "interval methods disagree across a threshold"
        if disagree
        else primary["reason"],
        "bca": primary,
        "delta": sensitivity,
    }
