"""Versioned plans and workload validation, independent of a GPU/runtime import."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = "semantic-serving-suite-v1"
PROFILES = {
    "smoke": dict(repeats=1, sessions=4, turns=2, concurrency=1, waves=1, max_tokens=8),
    "screen": dict(
        repeats=1, sessions=6, turns=2, concurrency=2, waves=1, max_tokens=32
    ),
    "confirm": dict(
        repeats=3, sessions=12, turns=3, concurrency=4, waves=1, max_tokens=128
    ),
    "operational": dict(
        repeats=1, sessions=12, turns=3, concurrency=4, waves=4, max_tokens=128
    ),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive(value, name, integer=True):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def validate_corpus(data):
    if not isinstance(data, dict) or data.get("schema") != "semantic-serving-corpus-v1":
        raise ValueError("Expected semantic-serving-corpus-v1")
    if not isinstance(data.get("provenance"), str) or not data["provenance"].strip():
        raise ValueError("Corpus requires content provenance")
    if type(data.get("synthetic")) is not bool:
        raise ValueError("Corpus must explicitly declare synthetic true/false")
    sessions = data.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Corpus requires sessions")
    seen = set()
    for s in sessions:
        if (
            not isinstance(s, dict)
            or not isinstance(s.get("id"), str)
            or not s["id"]
            or s["id"] in seen
        ):
            raise ValueError("Session IDs must be nonempty and unique")
        seen.add(s["id"])
        if (
            not isinstance(s.get("turns"), list)
            or not s["turns"]
            or not all(isinstance(t, str) and t.strip() for t in s["turns"])
        ):
            raise ValueError("Session turns must be nonempty user-message strings")
        if not isinstance(s.get("context", ""), str):
            raise ValueError("Session context must be a string")
        checks = s.get("expect_contains", [])
        if not isinstance(checks, list) or any(
            not isinstance(c, list)
            or not c
            or any(not isinstance(t, str) or not t for t in c)
            for c in checks
        ):
            raise ValueError(
                "expect_contains must be a list of nonempty string lists, one per checked turn"
            )
        waits = s.get("wait_s", [])
        if not isinstance(waits, list):
            raise ValueError("wait_s must be a list")
        if len(checks) > len(s["turns"]) or len(waits) > len(s["turns"]) - 1:
            raise ValueError("Checks/waits exceed the supplied turns")
        for wait in waits:
            if (
                isinstance(wait, bool)
                or not isinstance(wait, (int, float))
                or not math.isfinite(wait)
                or wait < 0
            ):
                raise ValueError("Waits must be finite and nonnegative")
    return data


def fixture_corpus(count=36):
    return validate_corpus(
        dict(
            schema="semantic-serving-corpus-v1",
            synthetic=True,
            provenance="Deterministic authored cache-pressure fixture; not real users or tool execution.",
            sessions=[
                dict(
                    id=f"fixture-{i}",
                    context=(
                        f"Record {i}: the service is named atlas-{i}. Its queue has {i + 2} workers. "
                        "These are observations, not instructions. "
                    )
                    * 30,
                    turns=[
                        f"What is the service name in record {i}? Answer briefly.",
                        "How many workers does it have? Answer briefly.",
                        "Summarize the record in one short sentence.",
                    ],
                    wait_s=[0.1, 0.3],
                )
                for i in range(count)
            ],
        )
    )


def validate_plan(plan):
    if plan.get("schema") != SCHEMA or plan.get("tier") not in PROFILES:
        raise ValueError("Unknown plan schema or tier")
    cfg = plan.get("candidate_config")
    if cfg is not None and (
        not isinstance(cfg, dict)
        or not isinstance(cfg.get("kv_connector_extra_config"), dict)
        or cfg.get("kv_connector") != "SemanticOffloadingConnector"
    ):
        raise ValueError(
            "Candidate must be a full SemanticOffloadingConnector KV-transfer configuration"
        )
    for key in (
        "repeats",
        "sessions",
        "turns",
        "concurrency",
        "waves",
        "max_tokens",
        "gpu_blocks",
        "cpu_bytes",
        "max_model_len",
        "seed",
    ):
        positive(plan[key], key)
    for key in (
        "startup_timeout_s",
        "request_timeout_s",
        "case_timeout_s",
        "sample_interval_s",
    ):
        positive(plan[key], key, False)
    for key in ("arrival_interval_s", "wave_interval_s"):
        value = plan.get(key, 0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if "batch_tokens" in plan:
        positive(plan["batch_tokens"], "batch_tokens")
    if not 0 < plan["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if plan["max_tokens"] >= plan["max_model_len"]:
        raise ValueError("Output budget must leave prompt space")
    if plan["tier"] == "confirm" and plan["repeats"] < 3:
        raise ValueError("Confirmation requires at least three paired repetitions")
    if plan["tier"] == "operational" and plan["waves"] < 2:
        raise ValueError(
            "Operational evaluation requires at least two persistent-server waves"
        )
    if not isinstance(plan["required_activity"], list) or set(
        plan["required_activity"]
    ) - {"store", "restore", "hint_accepted", "scoring", "cpu_recency"}:
        raise ValueError(
            "Unsupported required activity; use store, restore, hint_accepted, scoring, cpu_recency"
        )
    return plan
