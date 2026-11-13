from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHAPES_SCHEMA = "session-shapes-v1"
MAX_SEQUENCE_TOKENS = 8192


@dataclass(frozen=True)
class Turn:
    prompt: int
    generated: int
    shared: int
    gap: float

    def __post_init__(self) -> None:
        if (
            min(self.prompt, self.generated) < 1
            or not 0 <= self.shared < self.prompt
            or self.prompt + self.generated > MAX_SEQUENCE_TOKENS
            or not 0 <= self.gap <= 3600
        ):
            raise ValueError("invalid turn shape")


@dataclass(frozen=True)
class Session:
    name: str
    start: float
    turns: tuple[Turn, ...]
    hinted: bool
    abandons: bool

    def __post_init__(self) -> None:
        if not self.turns or self.turns[0].shared:
            raise ValueError("session starts without shared context")
        for before, after in zip(self.turns, self.turns[1:]):
            if after.shared > before.prompt + before.generated - 1:
                raise ValueError("shared prefix exceeds computed context")
        if self.abandons and not self.hinted:
            raise ValueError("only hinted sessions abandon advice")


def turn_from_record(record: dict[str, Any]) -> tuple[int, int, int]:
    usage = record["response"]["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return (
        int(usage["prompt_tokens"]),
        int(usage["completion_tokens"]),
        int(details.get("cached_tokens") or 0),
    )


def extract_shapes(attempt: Path) -> dict[str, Any]:
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    models: dict[str, str] = {}
    for path in sorted(attempt.glob("sessions/*/*/panel/*/requests/request-*.json")):
        if path.name.endswith(".admission.json"):
            continue
        record = json.loads(path.read_text())
        if "usage" not in (record.get("response") or {}):
            continue
        cell = path.parents[3]
        if cell.as_posix() not in models:
            admission = json.loads((cell / "admission.json").read_text())
            model = admission["command"][admission["command"].index("--model") + 1]
            models[cell.as_posix()] = (
                "qwen35-4b-int4" if "Qwen" in model else "gemma4-12b-int4"
            )
        key = (path.parents[1].as_posix(), str(record.get("session")))
        sessions[key].append(record)
    shapes: dict[str, list[list[list[float]]]] = defaultdict(list)
    for (directory, _session), records in sorted(sessions.items()):
        records.sort(key=lambda record: record["start_ns"])
        turns: list[list[float]] = []
        previous_end = 0
        previous_context = 0
        for record in records:
            prompt, generated, cached = turn_from_record(record)
            if prompt + generated > MAX_SEQUENCE_TOKENS:
                break
            gap = round((record["start_ns"] - previous_end) / 1e9, 3) if turns else 0.0
            shared = min(cached, prompt - 1, max(0, previous_context - 1))
            turns.append([prompt, generated, shared if turns else 0, max(0.0, gap)])
            previous_end = record["received_ns"]
            previous_context = prompt + generated
        owner = Path(directory).parents[1].as_posix()
        if len(turns) >= 2:
            shapes[models[owner]].append(turns)
    return {
        "schema": SHAPES_SCHEMA,
        "source": "release-01 block 0 captured application requests",
        "shared_prefix": "cached_tokens observed without memory pressure, capped below the computed prior context",
        "turn_fields": ["prompt", "generated", "shared", "gap_seconds"],
        "models": {model: shapes[model] for model in sorted(shapes)},
    }


def load_shapes(path: Path) -> dict[str, list[list[Turn]]]:
    raw = json.loads(path.read_text())
    if raw.get("schema") != SHAPES_SCHEMA:
        raise ValueError("unsupported session shape schema")
    return {
        model: [
            [Turn(int(p), int(g), int(s), float(w)) for p, g, s, w in session]
            for session in sessions
        ]
        for model, sessions in raw["models"].items()
    }


SCHEDULE_FIELDS = {
    "seed",
    "sessions",
    "source",
    "turns",
    "first_prompt",
    "growth",
    "generated",
    "gap_median",
    "gap_sigma",
    "compaction",
    "compaction_keep",
    "unhinted",
    "abandon",
    "stagger",
    "one_shot",
    "one_shot_prompt",
    "one_shot_interval",
    "max_context",
}


def lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return min(3600.0, rng.lognormvariate(math.log(median), sigma))


def synthetic_turns(rng: random.Random, spec: dict[str, Any]) -> list[Turn]:
    turns: list[Turn] = []
    context = 0
    count = rng.randint(*spec["turns"])
    for index in range(count):
        generated = rng.randint(*spec["generated"])
        if index == 0:
            prompt, shared = rng.randint(*spec["first_prompt"]), 0
        elif rng.random() < spec["compaction"]:
            shared = min(spec["compaction_keep"], context - 1)
            prompt = shared + rng.randint(*spec["growth"])
        else:
            shared = context - 1
            prompt = context + rng.randint(*spec["growth"])
        if prompt + generated > spec["max_context"]:
            break
        gap = lognormal(rng, spec["gap_median"], spec["gap_sigma"]) if index else 0.0
        turns.append(Turn(prompt, generated, shared, round(gap, 3)))
        context = prompt + generated
    return turns


def schedule(
    spec: dict[str, Any], shapes: dict[str, list[list[Turn]]] | None = None
) -> list[Session]:
    if set(spec) != SCHEDULE_FIELDS:
        raise ValueError(
            f"schedule fields differ: {sorted(set(spec) ^ SCHEDULE_FIELDS)}"
        )
    if not 0 < spec["max_context"] <= MAX_SEQUENCE_TOKENS:
        raise ValueError("max_context outside the supported sequence length")
    rng = random.Random(spec["seed"])
    sessions: list[Session] = []
    pool = (
        [] if spec["source"] == "synthetic" else (shapes or {}).get(spec["source"], [])
    )
    if spec["source"] != "synthetic" and not pool:
        raise ValueError("recorded schedule requires session shapes")
    for index in range(spec["sessions"]):
        if pool:
            recorded = pool[rng.randrange(len(pool))]
            turns = [
                Turn(
                    turn.prompt,
                    turn.generated,
                    turn.shared,
                    round(lognormal(rng, spec["gap_median"], spec["gap_sigma"]), 3)
                    if position and spec["gap_median"]
                    else turn.gap,
                )
                for position, turn in enumerate(recorded)
            ]
            turns = turns[: rng.randint(*spec["turns"])] if spec["turns"] else turns
            for position, turn in enumerate(turns):
                if turn.prompt + turn.generated > spec["max_context"]:
                    turns = turns[:position]
                    break
        else:
            turns = synthetic_turns(rng, spec)
        if len(turns) < 2:
            continue
        hinted = rng.random() >= spec["unhinted"]
        sessions.append(
            Session(
                f"s{index}",
                round(rng.uniform(0, spec["stagger"]), 3),
                tuple(turns),
                hinted,
                hinted and rng.random() < spec["abandon"],
            )
        )
    clock = 0.0
    for index in range(spec["one_shot"]):
        clock += lognormal(rng, spec["one_shot_interval"], 0.5)
        prompt = rng.randint(*spec["one_shot_prompt"])
        sessions.append(
            Session(
                f"o{index}", round(clock, 3), (Turn(prompt, 16, 0, 0.0),), False, False
            )
        )
    return sessions


def cohort(session: Session) -> str:
    if len(session.turns) == 1:
        return "one-shot"
    return "hinted" if session.hinted else "unhinted"
