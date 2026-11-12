from __future__ import annotations

import statistics
from typing import Any


def controls(
    observations: dict[str, dict[str, Any]], policy: dict[str, Any] | None
) -> dict[str, Any]:
    indexed = {
        (item["model"], item["block"], item["system"], item["case"]): item
        for item in observations.values()
        if item["kind"] == "trace"
    }
    noise = []
    sensitivity = []
    protection = []
    if policy is None:
        return {
            "status": "unresolved",
            "reason": "no frozen measurement policy",
            "request_noise_seconds": None,
        }

    def matched(
        model: str,
        block: int,
        before_arm: str,
        before_case: str,
        after_arm: str,
        after_case: str,
        labels: set[str] | None = None,
    ) -> list[tuple[str, float, float]]:
        before = indexed.get((model, block, before_arm, before_case))
        after = indexed.get((model, block, after_arm, after_case))
        if (
            not before
            or not after
            or before["status"] != after["status"]
            or before["status"] != "complete"
        ):
            return []
        a = {record["label"]: record["seconds"] for record in before["requests"]}
        b = {record["label"]: record["seconds"] for record in after["requests"]}
        return [
            (label, a[label], b[label])
            for label in sorted(a.keys() & b.keys())
            if labels is None or label in labels
        ]

    for block in range(2):
        aa = matched("qwen35-4b-int4", block, "native-arc", "T3", "native-arc", "A1")
        delayed = matched(
            "qwen35-4b-int4", block, "native-arc", "A1", "injected-delay", "S1"
        )
        if aa:
            noise.append(
                statistics.median(abs(after - before) for _label, before, after in aa)
            )
        if delayed:
            difference = statistics.median(
                after - before for _label, before, after in delayed
            )
            sensitivity.append(
                {
                    "block": block,
                    "median_injected_difference_seconds": difference,
                    "passed": difference >= 0.15,
                }
            )
        for model in ("qwen35-4b-int4", "gemma4-12b-int4"):
            for before_arm, before_case, after_arm, after_case, labels in (
                ("native-arc", "T1", "wait-hint", "T1", {"cold", "resident"}),
                ("native-arc", "T2", "wait-hint", "T2", {"unhinted-return"}),
                ("native-arc", "T3", "wait-hint", "T3", {"unhinted-return"}),
            ):
                for label, before, after in matched(
                    model, block, before_arm, before_case, after_arm, after_case, labels
                ):
                    tolerance = max(
                        policy["absolute_tolerance_seconds"],
                        before * (policy["latency_ratio_limit"] - 1),
                    )
                    protection.append(
                        {
                            "model": model,
                            "block": block,
                            "case": before_case,
                            "label": label,
                            "control_seconds": before,
                            "candidate_seconds": after,
                            "absolute_delta_seconds": after - before,
                            "ratio": after / before if before else None,
                            "passed": after - before <= tolerance,
                        }
                    )
    complete = len(noise) == 2 and len(sensitivity) == 2 and len(protection) == 16
    return {
        "status": "unresolved"
        if not complete
        else "passed"
        if all(row["passed"] for row in sensitivity + protection)
        else "failed",
        "request_noise_seconds": max(noise) if noise else None,
        "aa_median_absolute_differences_seconds": noise,
        "injected_delay": sensitivity,
        "protected_and_noop": protection,
        "reason": "descriptive per-block controls with frozen max(20 ms, 5%) tolerance; no tail or population guarantee",
    }
