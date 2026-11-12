from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.benchmarking.measurement_controls import controls


def read_policy(root: Path) -> dict[str, Any] | None:
    path = root / "benchmarks/claims.json"
    if not path.exists():
        return None
    policy: dict[str, Any] = json.loads(path.read_text())
    if (
        set(policy)
        != {
            "schema",
            "latency_ratio_limit",
            "absolute_tolerance_seconds",
            "benefit_ratio_limit",
            "application_usefulness",
            "primary_outcome",
            "scope",
        }
        or policy["schema"] != "benchmark-claim-policy-v1"
    ):
        raise ValueError("unsupported benchmark claim policy")
    for key in (
        "latency_ratio_limit",
        "absolute_tolerance_seconds",
        "benefit_ratio_limit",
    ):
        if (
            type(policy[key]) not in (int, float)
            or not math.isfinite(policy[key])
            or policy[key] <= 0
        ):
            raise ValueError("claim policy requires finite positive thresholds")
    if policy["latency_ratio_limit"] < 1 or policy["benefit_ratio_limit"] >= 1:
        raise ValueError("claim policy ratios have invalid direction")
    return policy


def release_gates(
    admission: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    complete: bool,
    elapsed: float,
    functional: dict[str, Any],
) -> dict[str, Any]:
    policy = admission.get("claim_policy")
    gates: dict[str, dict[str, Any]] = {
        "acquisition": {
            "status": "passed" if complete and not admission["partial"] else "failed",
            "reason": "all mandatory cells must complete; partial campaigns cannot qualify release",
        },
        "budget": {
            "status": "passed" if elapsed <= admission["budget_seconds"] else "failed"
        },
        "functional_and_faults": functional,
    }
    regressions = [pair for pair in pairs if pair["quality_regression"]]
    gates["paired_quality"] = {
        "status": "unresolved" if regressions else "passed",
        "regressions": regressions,
        "reason": "every control-resolved/candidate-unresolved task needs investigation; no population parity claim",
    }
    controlled = [
        item
        for item in observations.values()
        if item["kind"] == "trace"
        and item["system"] == "wait-hint"
        and item["case"] in ("T1", "T2", "T3")
    ]
    gates["mechanism_activity"] = {
        "status": "passed"
        if len(controlled) == 12
        and all(
            item["status"] == "complete"
            and item.get("activity", {}).get("gpu_reuse_observed")
            and item.get("activity", {}).get("cpu_restore_observed")
            and item.get("activity", {}).get("ready_advice_observed")
            for item in controlled
        )
        else "unresolved",
        "reason": "both models/starts need real GPU reuse, CPU restore and ready causal advice; acknowledgement alone is not saved recomputation",
    }
    pressure = [item for item in controlled if item["case"] in ("T2", "T3")]
    gates["cpu_pressure"] = {
        "status": "passed"
        if len(pressure) == 8
        and all(
            (item.get("minimum_cpu_working_set_ratio") or 0) > 1
            and (item.get("trace_store_bytes") or 0) > 0
            for item in pressure
        )
        else "unresolved",
        "reason": "distinct completed prefixes must exceed native CPU capacity with observed stores; fixed prefix count alone does not qualify pressure",
    }
    soaks = [item for item in observations.values() if item["case"] == "sustained"]
    planned = {
        (job["model"], job["system"]): job["seconds"]
        for job in admission["jobs"]
        if job["case"] == "sustained"
    }
    gates["operating_exposure"] = {
        "status": "passed"
        if len(soaks) == 4
        and all(
            item["status"] == "complete"
            and (item.get("operating_seconds") or 0)
            >= planned[(item["model"], item["system"])] - 5
            and item.get("resources", {}).get("complete")
            and item.get("metadata", {}).get("within_limits")
            for item in soaks
        )
        else "unresolved",
        "reason": "four bounded mixed/abandoned/returning/churning runs, quiet replay and resource windows; no long-term reliability guarantee",
    }
    regions: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair["kind"] == "application":
            regions[(pair["model"], pair["case"], pair["reference"])].append(pair)
    useful = []
    observed_benefits = []
    measurement_controls = controls(observations, policy)
    noise_seconds = measurement_controls.get("request_noise_seconds")
    for (model, case, reference), blocks in regions.items():
        if {block["block"] for block in blocks} != {0, 1} or not all(
            block["complete_pair"]
            and block["candidate_quality"] == block["control_quality"] == "resolved"
            for block in blocks
        ):
            continue
        region = {
            "model": model,
            "case": case,
            "reference": reference,
            "elapsed_ratios": [block["elapsed_ratio"] for block in blocks],
        }
        useful.append(region)
        if (
            policy
            and all(
                block["elapsed_ratio"] <= policy["benefit_ratio_limit"]
                for block in blocks
            )
            and noise_seconds is not None
            and all(
                block["control_seconds"] - block["candidate_seconds"]
                > max(
                    policy["absolute_tolerance_seconds"],
                    3 * noise_seconds * block["maximum_request_count"],
                )
                for block in blocks
            )
        ):
            observed_benefits.append(region)
    gates["useful_application_evidence"] = {
        "status": "passed" if useful else "failed",
        "regions": useful,
        "reason": "both arms must actually resolve each claimed task in both blocks",
    }
    gates["noise_and_protected_traffic"] = measurement_controls
    fresh = [
        region
        for region in observed_benefits
        if any(
            item["model"] == region["model"]
            and item["case"] == region["case"]
            and not item.get("source_group_previously_exposed", True)
            for item in observations.values()
        )
    ]
    gates["fresh_confirmation"] = {
        "status": "passed" if fresh else "unresolved",
        "regions": fresh,
        "reason": "new source groups admitted before this campaign, with all prior/pilot admissions retained; repeated starts are not additional independent source tasks",
    }
    status = (
        "not_eligible"
        if any(gate["status"] == "failed" for gate in gates.values())
        else "unresolved"
        if any(gate["status"] == "unresolved" for gate in gates.values())
        else "eligible"
    )
    return {
        "release": status,
        "gates": gates,
        "observed_benefit_regions": observed_benefits,
        "confirmed_benefit_regions": fresh if status == "eligible" else [],
        "benefit": "observed_case_scoped"
        if observed_benefits and status == "eligible"
        else "unresolved"
        if observed_benefits
        else "not_established",
        "permitted_statement": (
            "All declared gates passed for the retained configurations. Benefit claims are limited to the listed confirmed task/configuration regions; population quality, competitive superiority and long-term reliability remain unestablished."
            if status == "eligible"
            else "Retained measurements describe the named tasks/configurations. Optimization release evidence remains incomplete or failed; competitive superiority and general quality guarantees are unestablished."
        ),
    }
