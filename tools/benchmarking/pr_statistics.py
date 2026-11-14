from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Callable, Sequence
from typing import Any


def validate_pairs(*groups: Sequence[dict[str, Any]]) -> None:
    if not groups or not groups[0]:
        raise ValueError("at least one paired seed required")
    expected = [(cell["model"], cell["partition"], cell["seed"]) for cell in groups[0]]
    if len(set(expected)) != len(expected) or len({item[:2] for item in expected}) != 1:
        raise ValueError("one model/partition and unique paired seeds required")
    for group in groups:
        if [
            (cell["model"], cell["partition"], cell["seed"]) for cell in group
        ] != expected:
            raise ValueError(
                "contrast requires the same ordered model/partition/seed pairs"
            )
        if len({cell["arm"] for cell in group}) != 1:
            raise ValueError("each contrast group must contain one arm")


def summarize_cohort(cell: dict[str, Any], cohort: str, metric: str) -> dict[str, Any]:
    if cohort not in {"first_turn", "hinted_return", "unhinted_return", "one_shot"}:
        raise ValueError("unknown screen cohort")
    if metric not in {"ttft_ns", "client_itl_ns", "recompute"}:
        raise ValueError("unknown screen metric")
    rows = [row for row in cell["requests"] if row["cohort"] == cohort]
    strata = {}
    for stratum in {row["stratum"] for row in rows}:
        selected = [row for row in rows if row["stratum"] == stratum]
        total = denominator = 0.0
        failures = sum(row["outcome"] != "complete" for row in selected)
        for row in selected:
            if row["outcome"] != "complete":
                continue
            if metric == "recompute":
                if row["sources"] is None or row["sources"]["unobserved_tokens"]:
                    raise ValueError("complete return has unknown token source")
                total += row["sources"]["recomputed_tokens"]
                denominator += row["eligible_tokens"]
            elif row[metric] is not None:
                total += row[metric]
                denominator += 1
        strata[stratum] = {
            "sessions": len({row["session_id"] for row in selected}),
            "numerator": total,
            "denominator": denominator,
            "failures": failures,
        }
    return {
        "strata": strata,
        "weights": {"one_shot": 1} if cohort == "one_shot" else cell["stratum_weights"],
        "cohort": cohort,
        "metric": metric,
    }


def pooled(
    summaries: Sequence[dict[str, Any]], *, descriptive: bool = False
) -> dict[str, Any]:
    if not summaries:
        return {"value": None, "reason": "no seeds"}
    weights = summaries[0]["weights"]
    if any(summary["weights"] != weights for summary in summaries):
        raise ValueError("stratum weights drifted between seeds")
    numerator = denominator = 0.0
    dn, dd = [0.0] * len(summaries), [0.0] * len(summaries)
    failures = 0
    for stratum, weight in weights.items():
        cells = [
            summary["strata"].get(
                stratum,
                {"sessions": 0, "numerator": 0.0, "denominator": 0.0, "failures": 0},
            )
            for summary in summaries
        ]
        count = sum(cell["sessions"] for cell in cells)
        failures += sum(cell["failures"] for cell in cells)
        if count == 0:
            return {
                "value": None,
                "reason": "missing required stratum",
                "stratum": stratum,
            }
        top = sum(cell["numerator"] for cell in cells)
        bottom = sum(cell["denominator"] for cell in cells)
        numerator += weight * top / count
        denominator += weight * bottom / count
        for index, cell in enumerate(cells):
            dn[index] += (
                weight * (cell["numerator"] * count - top * cell["sessions"]) / count**2
            )
            dd[index] += (
                weight
                * (cell["denominator"] * count - bottom * cell["sessions"])
                / count**2
            )
    if denominator <= 0 or (failures and not descriptive):
        return {
            "value": None,
            "reason": "undefined denominator"
            if denominator <= 0
            else "censored required cohort",
            "failures": failures,
        }
    value = numerator / denominator
    influence = [
        (top * denominator - numerator * bottom) / denominator**2
        for top, bottom in zip(dn, dd, strict=True)
    ]
    return {
        "value": value,
        "influence": influence,
        "failures": failures,
        "numerator": numerator,
        "denominator": denominator,
    }


def quantile(ordered: Sequence[float], probability: float) -> float:
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("undefined quantile")
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    fraction = location - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    )


def describe(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = [cell for summary in summaries for cell in summary["strata"].values()]
    numerator = sum(cell["numerator"] for cell in cells)
    denominator = sum(cell["denominator"] for cell in cells)
    return {
        "weighted_complete_only": pooled(summaries, descriptive=True),
        "unweighted_complete_only": numerator / denominator if denominator else None,
        "raw_numerator": numerator,
        "raw_denominator": denominator,
        "planned_sessions": sum(cell["sessions"] for cell in cells),
        "failed_or_unoffered_requests": sum(cell["failures"] for cell in cells),
        "seed_strata": [summary["strata"] for summary in summaries],
        "scope": "complete-response description; censoring cannot establish efficacy or safety",
    }


def bca_interval(
    count: int, statistic: Callable[[list[int]], float | None], label: str
) -> dict[str, Any]:
    point = statistic(list(range(count))) if count else None
    if count < 3 or point is None:
        return {
            "point": point,
            "interval": None,
            "reason": "fewer than three pairs or undefined statistic",
        }
    rng = random.Random(int(hashlib.sha256(label.encode()).hexdigest(), 16))
    samples = [
        statistic([rng.randrange(count) for _ in range(count)]) for _ in range(10000)
    ]
    undefined = sum(sample is None or not math.isfinite(sample) for sample in samples)
    if undefined:
        return {
            "point": point,
            "interval": None,
            "reason": "undefined bootstrap replicates",
            "undefined_replicates": undefined,
        }
    values = sorted(float(sample) for sample in samples if sample is not None)
    jackknife = [
        statistic([index for index in range(count) if index != omitted])
        for omitted in range(count)
    ]
    if any(value is None or not math.isfinite(value) for value in jackknife):
        return {"point": point, "interval": None, "reason": "undefined jackknife"}
    jack_values = [float(value) for value in jackknife if value is not None]
    mean = statistics.mean(jack_values)
    squared = sum((mean - value) ** 2 for value in jack_values)
    probability = (
        sum(value < point for value in values)
        + 0.5 * sum(value == point for value in values)
    ) / len(values)
    if squared == 0 or values[0] == values[-1] or not 0 < probability < 1:
        return {
            "point": point,
            "interval": None,
            "reason": "degenerate bootstrap or jackknife",
        }
    acceleration = sum((mean - value) ** 3 for value in jack_values) / (
        6 * squared**1.5
    )
    normal = statistics.NormalDist()
    bias = normal.inv_cdf(probability)
    adjusted = []
    for probability in (0.025, 0.975):
        z = bias + normal.inv_cdf(probability)
        divisor = 1 - acceleration * z
        if divisor <= 0:
            return {
                "point": point,
                "interval": None,
                "reason": "undefined BCa adjustment",
            }
        adjusted.append(normal.cdf(bias + z / divisor))
    if (
        not all(math.isfinite(value) for value in adjusted)
        or adjusted[0] >= adjusted[1]
    ):
        return {
            "point": point,
            "interval": None,
            "reason": "undefined adjusted quantiles",
        }
    return {
        "point": point,
        "interval": [quantile(values, value) for value in adjusted],
        "bias": bias,
        "acceleration": acceleration,
        "resamples": 10000,
        "pairs": count,
    }


def compare(
    treatment: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    kind: str,
    label: str,
) -> dict[str, Any]:
    if len(treatment) != len(baseline) or kind not in {
        "benefit",
        "relative_harm",
        "absolute_change",
    }:
        raise ValueError("matched seeds and known contrast required")
    if treatment and treatment[0]["weights"] != baseline[0]["weights"]:
        raise ValueError("contrast population weights differ")

    def statistic(indices: list[int]) -> float | None:
        left = pooled([treatment[index] for index in indices])["value"]
        right = pooled([baseline[index] for index in indices])["value"]
        if left is None or right is None:
            return None
        if kind == "absolute_change":
            return float(left - right)
        if right <= 0:
            return None
        return float(1 - left / right if kind == "benefit" else left / right - 1)

    result = bca_interval(len(treatment), statistic, label)
    left, right = pooled(treatment), pooled(baseline)
    delta = None
    count = len(treatment)
    if count >= 2 and left["value"] is not None and right["value"] is not None:
        if kind == "absolute_change":
            influence = [
                a - b
                for a, b in zip(left["influence"], right["influence"], strict=True)
            ]
            point = left["value"] - right["value"]
            se = math.sqrt(count / (count - 1) * sum(value**2 for value in influence))
            delta = [point - 1.959963984540054 * se, point + 1.959963984540054 * se]
        elif left["value"] > 0 and right["value"] > 0:
            influence = [
                a / left["value"] - b / right["value"]
                for a, b in zip(left["influence"], right["influence"], strict=True)
            ]
            point = math.log(left["value"] / right["value"])
            se = math.sqrt(count / (count - 1) * sum(value**2 for value in influence))
            ratio = [
                math.exp(point - 1.959963984540054 * se),
                math.exp(point + 1.959963984540054 * se),
            ]
            delta = (
                [1 - ratio[1], 1 - ratio[0]]
                if kind == "benefit"
                else [ratio[0] - 1, ratio[1] - 1]
            )
    return result | {
        "delta_interval": delta,
        "treatment": left,
        "baseline": right,
        "descriptive_treatment": describe(treatment),
        "descriptive_baseline": describe(baseline),
        "contrast": kind,
    }


def route_difference(
    post: Sequence[dict[str, Any]],
    ttl: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    if len(post) != len(ttl) or len(post) != len(baseline):
        raise ValueError("whole-route difference requires matched seeds")
    if post and (
        post[0]["weights"] != ttl[0]["weights"]
        or post[0]["weights"] != baseline[0]["weights"]
    ):
        raise ValueError("whole-route population weights differ")

    def statistic(indices: list[int]) -> float | None:
        p, t, b = [
            pooled([group[index] for index in indices])["value"]
            for group in (post, ttl, baseline)
        ]
        if p is None or t is None or b is None or b <= 0:
            return None
        return float((t - p) / b)

    result = bca_interval(len(post), statistic, label)
    p, t, b = [pooled(group) for group in (post, ttl, baseline)]
    delta = None
    count = len(post)
    if count >= 2 and result["point"] is not None:
        influence = [
            (dt - dp) / b["value"] - (t["value"] - p["value"]) * db / b["value"] ** 2
            for dp, dt, db in zip(
                p["influence"], t["influence"], b["influence"], strict=True
            )
        ]
        se = math.sqrt(count / (count - 1) * sum(value**2 for value in influence))
        delta = [
            result["point"] - 1.959963984540054 * se,
            result["point"] + 1.959963984540054 * se,
        ]
    return result | {
        "delta_interval": delta,
        "contrast": "post_benefit_minus_ttl_benefit",
        "scope": "descriptive whole-route difference; not isolated delivery-hook benefit",
    }
