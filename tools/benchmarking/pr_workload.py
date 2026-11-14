from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import zlib
from pathlib import Path
from typing import Any

from wait_hint_offload.evidence_protocol import EvidenceRequest
from wait_hint_offload.policy_protocol import PolicySession

ARMS = ("A1", "A2", "A3", "A4t", "A3p", "A4p", "A5")


def seeded_order(seed: int, label: str, identity: str) -> str:
    return hashlib.sha256(
        json.dumps([seed, label, identity], separators=(",", ":")).encode()
    ).hexdigest()


def _rng(seed: int, label: str) -> random.Random:
    return random.Random(int(seeded_order(seed, label, ""), 16))


def load_corpus(
    data: Path, model: str, partition: str
) -> tuple[dict[str, Any], list[Any]]:
    frozen = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks/pr-evidence/w1-freeze.json"
        ).read_text()
    )
    if (
        hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
        != frozen["manifest_sha256"]
    ):
        raise ValueError("W1 manifest differs from committed content freeze")
    with (data / "episodes.sqlite").open("rb") as source:
        if (
            hashlib.file_digest(source, "sha256").hexdigest()
            != frozen["data_sha256"]["episodes.sqlite"]
        ):
            raise ValueError("W1 token corpus differs from committed content freeze")
    connection = sqlite3.connect(f"file:{data / 'episodes.sqlite'}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT task,metadata,payload FROM episodes WHERE model=? AND partition=? ORDER BY sort_key,task LIMIT 2048",
            (model, partition),
        ).fetchall()
    finally:
        connection.close()
    return frozen, rows


def w1_schedule(
    data: Path,
    model: str,
    partition: str,
    seed: int,
    arrival_rate: float,
    one_shot_rate: float,
    horizon_seconds: float,
    *,
    session_count: int = 16,
) -> dict[str, Any]:
    if model not in {"D8", "HM", "HS"} or partition not in {"calibration", "pilot"}:
        raise ValueError("screen never samples confirmation repositories")
    if (
        not 0 < arrival_rate <= 100
        or not 0 <= one_shot_rate <= 100
        or not 0 < horizon_seconds <= 900
    ):
        raise ValueError("invalid screen rates or horizon")
    if session_count != 16:
        raise ValueError("Stage 2 requires sixteen sessions")
    frozen, rows = load_corpus(data, model, partition)
    rows.sort(key=lambda row: (seeded_order(seed, "task", row[0]), row[0]))
    if not rows:
        raise ValueError("no admitted tasks")
    arrivals, gaps, shots = (
        _rng(seed, label) for label in ("arrivals", "gaps", "one-shots")
    )
    adopters = set(
        sorted(
            range(session_count),
            key=lambda index: seeded_order(seed, "adoption", str(index)),
        )[: session_count // 2]
    )
    sessions = []
    arrival = 0.0
    for index in range(session_count):
        task, metadata, compressed = rows[index % len(rows)]
        meta = json.loads(metadata)
        payload = zlib.decompress(compressed)
        if hashlib.sha256(payload).hexdigest() != meta["episode_sha256"]:
            raise ValueError("W1 episode identity drift")
        episode = json.loads(payload)
        arrival += arrivals.expovariate(arrival_rate)
        sessions.append(
            {
                "session_id": f"s{seed}-{index:04d}",
                "task": task,
                "repository": meta["repository"],
                "trajectory": meta["trajectory"],
                "source_resolved": meta["resolved"],
                "stratum": episode["stratum"],
                "ending": episode["ending"],
                "arrival_seconds": arrival,
                "adopter": index in adopters,
                "task_occurrence": index // len(rows) + 1,
                "turns": [
                    turn
                    | {
                        "gap_seconds": min(
                            300.0, gaps.lognormvariate(math.log(4.0), 0.8)
                        )
                    }
                    for turn in episode["turns"]
                ],
            }
        )
    if arrival >= horizon_seconds:
        raise ValueError("sixteen offered arrivals do not fit the frozen cell horizon")
    one_shots: list[dict[str, Any]] = []
    arrival = shots.expovariate(one_shot_rate) if one_shot_rate else horizon_seconds
    while arrival < horizon_seconds:
        index = len(one_shots)
        content = _rng(seed, f"one-shot-content-{index}")
        one_shots.append(
            {
                "session_id": f"o{seed}-{index:06d}",
                "arrival_seconds": arrival,
                "prompt_tokens": [content.randrange(1000, 30000) for _ in range(2048)],
                "requested_output_tokens": 128,
            }
        )
        arrival += shots.expovariate(one_shot_rate)
    return {
        "schema": "pr-w1-schedule-v1",
        "model": model,
        "partition": partition,
        "seed": seed,
        "manifest_sha256": frozen["manifest_sha256"],
        "arrival_rate": arrival_rate,
        "one_shot_rate": one_shot_rate,
        "horizon_seconds": horizon_seconds,
        "request_deadline_seconds": 120.0,
        "session_deadline_seconds": horizon_seconds,
        "ttl_ms": 12000,
        "sessions": sessions,
        "one_shots": one_shots,
        "arm_order": sorted(ARMS, key=lambda arm: seeded_order(seed, "arm-order", arm)),
        "stratum_weights": frozen["admitted"][model][partition]["stratum_counts"],
        "task_reuses": max(0, session_count - len(rows)),
    }


def request_payload(
    schedule: dict[str, Any], session: dict[str, Any], index: int, arm: str
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown arm")
    one_shot = "turns" not in session
    turn = session if one_shot else session["turns"][index]
    request_id = f"{session['session_id']}-t{index:02d}"
    ttl = (
        schedule["ttl_ms"]
        if not one_shot and session["adopter"] and arm in {"A3", "A4t", "A5"}
        else None
    )
    return EvidenceRequest(
        request_id,
        tuple(turn["prompt_tokens"]),
        turn["requested_output_tokens"],
        PolicySession(
            "pr-evidence", session["session_id"], str(schedule["seed"]), index + 1
        ),
        ttl,
    ).to_dict()


def validate_schedule(schedule: dict[str, Any]) -> None:
    fields = {
        "schema",
        "model",
        "partition",
        "seed",
        "manifest_sha256",
        "arrival_rate",
        "one_shot_rate",
        "horizon_seconds",
        "request_deadline_seconds",
        "session_deadline_seconds",
        "ttl_ms",
        "sessions",
        "one_shots",
        "arm_order",
        "stratum_weights",
        "task_reuses",
    }
    if set(schedule) != fields or schedule["schema"] != "pr-w1-schedule-v1":
        raise ValueError("closed screen schedule required")
    if schedule["model"] not in {"D8", "HM", "HS"} or schedule["partition"] not in {
        "calibration",
        "pilot",
    }:
        raise ValueError("screen model or partition outside protocol")
    expected_seeds = (
        range(12001, 12004)
        if schedule["partition"] == "calibration"
        else range(12101, 12109)
    )
    if type(schedule["seed"]) is not int or schedule["seed"] not in expected_seeds:
        raise ValueError("screen seed outside frozen family")
    if (
        schedule["ttl_ms"] != 12000
        or schedule["request_deadline_seconds"] != 120.0
        or schedule["session_deadline_seconds"] != schedule["horizon_seconds"]
    ):
        raise ValueError("screen deadline differs from protocol")
    if (
        any(
            type(schedule[name]) not in (int, float)
            or not math.isfinite(schedule[name])
            for name in ("arrival_rate", "one_shot_rate", "horizon_seconds")
        )
        or not 0 < schedule["arrival_rate"] <= 100
        or not 0 <= schedule["one_shot_rate"] <= 100
        or not 0 < schedule["horizon_seconds"] <= 900
    ):
        raise ValueError("invalid screen schedule rates or horizon")
    if (
        len(schedule["sessions"]) != 16
        or sum(session["adopter"] for session in schedule["sessions"]) != 8
    ):
        raise ValueError("screen needs sixteen sessions, eight preassigned adopters")
    if schedule["arm_order"] != sorted(
        ARMS, key=lambda arm: seeded_order(schedule["seed"], "arm-order", arm)
    ):
        raise ValueError("paired arm order differs")
    identities = set()
    for one_shot, entries in (
        (False, schedule["sessions"]),
        (True, schedule["one_shots"]),
    ):
        previous = -1.0
        for session in entries:
            expected = (
                {
                    "session_id",
                    "arrival_seconds",
                    "prompt_tokens",
                    "requested_output_tokens",
                }
                if one_shot
                else {
                    "session_id",
                    "task",
                    "repository",
                    "trajectory",
                    "source_resolved",
                    "stratum",
                    "ending",
                    "arrival_seconds",
                    "adopter",
                    "task_occurrence",
                    "turns",
                }
            )
            if (
                set(session) != expected
                or session["session_id"] in identities
                or not previous
                < session["arrival_seconds"]
                < schedule["horizon_seconds"]
            ):
                raise ValueError("session identity, arrival order or fields invalid")
            identities.add(session["session_id"])
            previous = session["arrival_seconds"]
            if one_shot:
                if (
                    len(session["prompt_tokens"]) != 2048
                    or session["requested_output_tokens"] != 128
                ):
                    raise ValueError("one-shot shape differs")
            elif (
                type(session["adopter"]) is not bool
                or not 2 <= len(session["turns"]) <= 8
            ):
                raise ValueError("invalid episode shape or adoption")
            for index, turn in enumerate([session] if one_shot else session["turns"]):
                if not one_shot:
                    if (
                        set(turn)
                        != {
                            "message_index",
                            "prompt_tokens",
                            "recorded_output_tokens",
                            "requested_output_tokens",
                            "output_template_prefix_changed_tokens",
                            "previous_request_lcp_tokens",
                            "previous_request_lost_suffix_tokens",
                            "post_response_eligible",
                            "gap_seconds",
                        }
                        or type(turn["post_response_eligible"]) is not bool
                        or not 0 < turn["gap_seconds"] <= 300
                    ):
                        raise ValueError("closed turn or gap required")
                EvidenceRequest.parse(request_payload(schedule, session, index, "A4t"))
