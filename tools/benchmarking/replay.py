from __future__ import annotations

import hashlib
import heapq
import importlib
import itertools
import json
import math
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tools.benchmarking.workload import (
    MAX_SEQUENCE_TOKENS,
    Session,
    Turn,
    cohort,
    load_shapes,
    schedule,
)
from tools.config import check_environment
from tools.process import run_owned
from tools.runtime_verify import write_record
from wait_hint_offload.observed import ObservedGroup

ARMS = (
    "gpu-only",
    "native",
    "recency",
    "promote",
    "hold",
    "promote-hold",
    "reachable-hold",
    "oracle",
)
COHORTS = ("hinted", "unhinted")


class Cache(Protocol):
    now: float

    def ready(self, key: bytes) -> bool: ...

    def touch(self, keys: Sequence[bytes]) -> None: ...

    def store(self, keys: Sequence[bytes]) -> tuple[list[bytes], int] | None: ...

    def complete(self, keys: Sequence[bytes]) -> None: ...

    def advise(
        self, arm: str, groups: Sequence[ObservedGroup], owner: str, deadline: float
    ) -> tuple[int, int]: ...

    def release(self, owner: str) -> None: ...

    def counters(self) -> dict[str, int]: ...


@dataclass(frozen=True)
class Geometry:
    block_tokens: int
    groups: tuple[tuple[int, int | None], ...]
    gpu_blocks: int
    cpu_keys: int
    max_running: int
    batch_tokens: int
    prefill_tokens_per_second: float
    decode_seconds_per_token: float
    store_seconds: float
    hint_delay_seconds: float
    hint_deadline_seconds: float
    hold_share: float
    gpu_prefix_reuse: bool

    def __post_init__(self) -> None:
        if (
            min(self.block_tokens, self.max_running, self.batch_tokens) < 1
            or not self.groups
            or self.groups[0][1] is not None
            or any(chunk % self.block_tokens for chunk, _ in self.groups)
            or any(window is not None and window < 1 for _, window in self.groups)
            or self.gpu_blocks * self.block_tokens < MAX_SEQUENCE_TOKENS
            or not 0 < self.hold_share <= 1
            or self.cpu_keys < 0
        ):
            raise ValueError("invalid replay geometry")

    @property
    def alignment(self) -> int:
        return math.lcm(*(chunk for chunk, _ in self.groups))

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> Geometry:
        names = set(cls.__dataclass_fields__)
        if set(raw) != names:
            raise ValueError(f"geometry fields differ: {sorted(set(raw) ^ names)}")
        values = dict(raw)
        values["groups"] = tuple(
            (int(chunk), None if window is None else int(window))
            for chunk, window in raw["groups"]
        )
        return cls(**values)

    def keys_per_token(self) -> float:
        return sum(1 / chunk for chunk, _ in self.groups)


@dataclass
class Running:
    session: Session
    turn: int
    total_blocks: int
    hit_tokens: int
    stored: list[int]


@dataclass
class Gpu:
    capacity: int
    running: int = 0
    cached: set[int] = field(default_factory=set)
    free: OrderedDict[int, None] = field(default_factory=OrderedDict)

    def hit(self, blocks: Sequence[int], limit: int) -> int:
        count = 0
        for block in blocks[:limit]:
            if block not in self.cached:
                break
            count += 1
        return count

    def trim(self) -> int:
        evicted = 0
        while self.running + len(self.free) > self.capacity:
            if not self.free:
                raise ValueError("GPU capacity exceeded by running requests")
            victim, _ = self.free.popitem(last=False)
            self.cached.discard(victim)
            evicted += 1
        return evicted

    def acquire(self, blocks: Sequence[int], total: int) -> int:
        for block in blocks:
            self.free.pop(block, None)
            self.cached.add(block)
        self.running += total
        return self.trim()

    def release(self, blocks: Sequence[int], total: int) -> int:
        self.running -= total
        for block in reversed(blocks):
            self.cached.add(block)
            self.free[block] = None
        return self.trim()


class Replay:
    def __init__(
        self,
        sessions: Sequence[Session],
        geometry: Geometry,
        arm: str,
        cache_factory: Callable[[int, float], Cache],
    ) -> None:
        if arm not in ARMS:
            raise ValueError("unknown replay arm")
        self.sessions = sessions
        self.geometry = geometry
        self.arm = arm
        self.cache = (
            None
            if arm == "gpu-only" or not geometry.cpu_keys
            else cache_factory(geometry.cpu_keys, geometry.hold_share)
        )
        self.gpu = Gpu(geometry.gpu_blocks)
        self.ids = itertools.count(1)
        self.order = itertools.count()
        self.events: list[tuple[float, int, str, Any]] = []
        self.content: dict[str, list[int]] = {s.name: [] for s in sessions}
        self.waiting: list[tuple[Session, int, float]] = []
        self.running = 0
        self.returns: list[dict[str, Any]] = []
        self.activity: Counter[str] = Counter()

    def push(self, time: float, kind: str, payload: Any) -> None:
        heapq.heappush(self.events, (time, next(self.order), kind, payload))

    def run(self) -> dict[str, Any]:
        for session in self.sessions:
            self.push(session.start, "arrive", (session, 0))
        handlers: dict[str, Callable[[float, Any], None]] = {
            "arrive": self.on_arrive,
            "progress": self.on_progress,
            "complete": self.on_complete,
            "finish": self.on_finish,
            "hint": self.on_hint,
        }
        while self.events:
            now, _, kind, payload = heapq.heappop(self.events)
            handlers[kind](now, payload)
        return self.summary()

    def keys(self, name: str, tokens: int) -> list[list[bytes]]:
        blocks = self.content[name]
        result = []
        for index, (chunk, _) in enumerate(self.geometry.groups):
            per_chunk = chunk // self.geometry.block_tokens
            result.append(
                [
                    b"%d:%d" % (blocks[(j + 1) * per_chunk - 1], index)
                    for j in range(tokens // chunk)
                ]
            )
        return result

    def ready_boundary(self, keys: list[list[bytes]], start: int, limit: int) -> int:
        assert self.cache is not None
        geometry = self.geometry
        full_chunk = geometry.groups[0][0]
        end = start // full_chunk
        while (end + 1) * full_chunk <= limit and self.cache.ready(keys[0][end]):
            end += 1
        boundary = end * full_chunk // geometry.alignment * geometry.alignment
        while boundary > start:
            if all(
                self.cache.ready(keys[index][j])
                for index, (chunk, window) in enumerate(geometry.groups)
                if window is not None
                for j in range(max(0, boundary // chunk - window), boundary // chunk)
            ):
                return boundary
            boundary -= geometry.alignment
        return start

    def touch(self, keys: list[list[bytes]], hit_tokens: int) -> None:
        assert self.cache is not None
        selected: list[bytes] = []
        for index, (chunk, window) in enumerate(self.geometry.groups):
            skip = 0 if window is None else max(0, hit_tokens // chunk - window)
            selected.extend(keys[index][skip:])
        self.cache.touch(selected)

    def on_arrive(self, now: float, payload: tuple[Session, int]) -> None:
        session, turn = payload
        self.waiting.append((session, turn, now))
        self.dispatch(now)

    def dispatch(self, now: float) -> None:
        while self.waiting and self.running < self.geometry.max_running:
            session, turn, arrived = self.waiting[0]
            shape = session.turns[turn]
            need = -(-(shape.prompt + shape.generated) // self.geometry.block_tokens)
            if self.gpu.running + need > self.gpu.capacity:
                self.activity["admission_waits"] += 1
                return
            self.waiting.pop(0)
            self.running += 1
            self.start(now, session, turn, arrived)

    def start(self, now: float, session: Session, index: int, arrived: float) -> None:
        if self.cache is not None:
            self.cache.now = now
        geometry = self.geometry
        block = geometry.block_tokens
        shape = session.turns[index]
        blocks = self.content[session.name]
        del blocks[shape.shared // block :]
        blocks.extend(
            next(self.ids) for _ in range(shape.prompt // block - len(blocks))
        )
        gpu_tokens = (
            self.gpu.hit(blocks, (shape.prompt - 1) // block) * block
            if geometry.gpu_prefix_reuse
            else 0
        )
        hit_tokens = gpu_tokens
        keys = self.keys(session.name, shape.prompt)
        stored = [0] * len(geometry.groups)
        if self.cache is not None:
            limit = shape.prompt - 1
            if len(geometry.groups) > 1:
                limit = limit // geometry.alignment * geometry.alignment
            boundary = self.ready_boundary(keys, gpu_tokens, limit)
            if boundary - gpu_tokens >= geometry.groups[0][0]:
                hit_tokens = boundary
                self.activity["cpu_restores"] += 1
                stored = [-(-boundary // chunk) for chunk, _ in geometry.groups]
            self.touch(keys, hit_tokens)
            self.cache.release(session.name)
        if index:
            reusable = shape.shared // block * block
            from_gpu = min(gpu_tokens, reusable)
            from_cpu = max(0, min(hit_tokens, reusable) - from_gpu)
            self.activity["gpu_displaced_returns"] += int(gpu_tokens < reusable)
            self.returns.append(
                {
                    "cohort": cohort(session),
                    "reusable_tokens": reusable,
                    "gpu_tokens": from_gpu,
                    "cpu_tokens": from_cpu,
                    "recomputed_tokens": reusable - from_gpu - from_cpu,
                    "ttft_seconds": now
                    - arrived
                    + (shape.prompt - hit_tokens) / geometry.prefill_tokens_per_second,
                }
            )
        total = -(-(shape.prompt + shape.generated) // block)
        self.activity["gpu_evictions"] += self.gpu.acquire(blocks, total)
        running = Running(session, index, total, hit_tokens, stored)
        nominal = shape.prompt - shape.shared
        steps = max(1, -(-nominal // geometry.batch_tokens))
        clock = now
        for step in range(1, steps + 1):
            computed = min(shape.prompt, shape.shared + step * geometry.batch_tokens)
            clock = now + (computed - shape.shared) / geometry.prefill_tokens_per_second
            self.push(clock, "progress", (running, computed))
        smallest = min(chunk for chunk, _ in geometry.groups)
        for token in range(1, shape.generated):
            if (shape.prompt + token) % smallest == 0:
                self.push(
                    clock + token * geometry.decode_seconds_per_token,
                    "progress",
                    (running, shape.prompt + token),
                )
        finish = clock + shape.generated * geometry.decode_seconds_per_token
        self.push(finish, "finish", running)

    def on_progress(self, now: float, payload: tuple[Running, int]) -> None:
        running, computed = payload
        name = running.session.name
        blocks = self.content[name]
        blocks.extend(
            next(self.ids)
            for _ in range(computed // self.geometry.block_tokens - len(blocks))
        )
        if self.cache is None:
            return
        keys = self.keys(name, computed)
        candidates = [
            key
            for index, group in enumerate(keys)
            for key in (
                group[running.stored[index] :][-1:]
                if self.geometry.groups[index][1] == 1
                else group[running.stored[index] :]
            )
        ]
        if not candidates:
            return
        self.cache.now = now
        result = self.cache.store(candidates)
        if result is None:
            self.activity["failed_stores"] += 1
            return
        stored, evicted = result
        self.activity["cpu_evictions"] += evicted
        running.stored = [len(group) for group in keys]
        if stored:
            self.activity["cpu_stored_keys"] += len(stored)
            self.touch(keys, running.hit_tokens)
            self.push(now + self.geometry.store_seconds, "complete", stored)

    def on_complete(self, now: float, keys: list[bytes]) -> None:
        if self.cache is not None:
            self.cache.complete(keys)

    def on_finish(self, now: float, running: Running) -> None:
        session = running.session
        shape = session.turns[running.turn]
        computed = shape.prompt + shape.generated - 1
        self.on_progress(now, (running, computed))
        blocks = self.content[session.name]
        self.activity["gpu_evictions"] += self.gpu.release(
            blocks[: computed // self.geometry.block_tokens], running.total_blocks
        )
        self.running -= 1
        last = running.turn + 1 == len(session.turns)
        if session.hinted and (not last or session.abandons):
            boundary = computed // self.geometry.alignment * self.geometry.alignment
            self.push(
                now + self.geometry.hint_delay_seconds,
                "hint",
                (session, self.keys(session.name, boundary), boundary, last),
            )
        if not last:
            gap = session.turns[running.turn + 1].gap
            self.push(now + gap, "arrive", (session, running.turn + 1))
        self.dispatch(now)

    def on_hint(
        self, now: float, payload: tuple[Session, list[list[bytes]], int, bool]
    ) -> None:
        session, keys, boundary, abandoned = payload
        self.activity["hints"] += 1
        self.activity["abandoned_hints"] += int(abandoned)
        if self.cache is None or self.arm == "native" or not boundary:
            return
        if self.arm == "oracle" and abandoned:
            return
        groups = tuple(
            ObservedGroup(chunk, window, tuple(keys[index]))
            for index, (chunk, window) in enumerate(self.geometry.groups)
        )
        self.cache.now = now
        deadline = now + self.geometry.hint_deadline_seconds
        ready, applied = self.cache.advise(self.arm, groups, session.name, deadline)
        self.activity["hints_without_ready_prefix"] += int(ready == 0)
        self.activity["hints_not_applied"] += int(ready > 0 and applied == 0)
        self.activity["advised_keys"] += applied

    def summary(self) -> dict[str, Any]:
        cohorts: dict[str, dict[str, float]] = {}
        for name in COHORTS:
            rows = [row for row in self.returns if row["cohort"] == name]
            totals: dict[str, float] = {"returns": len(rows)}
            for metric in (
                "reusable_tokens",
                "gpu_tokens",
                "cpu_tokens",
                "recomputed_tokens",
            ):
                totals[metric] = sum(row[metric] for row in rows)
            totals["mean_ttft_seconds"] = (
                round(sum(row["ttft_seconds"] for row in rows) / len(rows), 4)
                if rows
                else 0.0
            )
            cohorts[name] = totals
        if self.cache is not None:
            self.activity.update(self.cache.counters())
        return {
            "arm": self.arm,
            "cohorts": cohorts,
            "activity": dict(sorted(self.activity.items())),
        }


SCENARIO_FIELDS = {"name", "model", "geometry", "schedule", "seeds"}
SEPARATION = 0.1


def evaluate(
    scenario: dict[str, Any],
    shapes: dict[str, list[list[Turn]]],
    factory: Callable[[int, float], Cache],
    arms: Sequence[str] = ARMS,
) -> dict[str, Any]:
    if set(scenario) != SCENARIO_FIELDS:
        raise ValueError(
            f"scenario fields differ: {sorted(set(scenario) ^ SCENARIO_FIELDS)}"
        )
    geometry = Geometry.parse(scenario["geometry"])
    results: dict[str, Any] = {}
    for arm in arms:
        cohorts = {name: Counter[str]() for name in COHORTS}
        activity: Counter[str] = Counter()
        per_seed = []
        for seed in seed_list(scenario):
            sessions = schedule({**scenario["schedule"], "seed": seed}, shapes)
            run_result = Replay(sessions, geometry, arm, factory).run()
            for name in COHORTS:
                cohorts[name].update(
                    {
                        key: value
                        for key, value in run_result["cohorts"][name].items()
                        if key != "mean_ttft_seconds"
                    }
                )
            activity.update(run_result["activity"])
            per_seed.append(run_result["cohorts"]["hinted"]["recomputed_tokens"])
        results[arm] = {
            "cohorts": {name: dict(sorted(c.items())) for name, c in cohorts.items()},
            "activity": dict(sorted(activity.items())),
            "hinted_recomputed_per_seed": per_seed,
        }
    return {
        "name": scenario["name"],
        "model": scenario["model"],
        "working_set": working_set(scenario, shapes, geometry),
        "arms": results,
        "discrimination": discrimination(results),
    }


def seed_list(scenario: dict[str, Any]) -> list[int]:
    seeds = scenario["seeds"]
    if isinstance(seeds, list):
        return [int(seed) for seed in seeds]
    base = scenario["schedule"].get("seed", 0)
    return list(range(base, base + int(seeds)))


def recomputed_share(result: dict[str, Any], name: str = "hinted") -> float:
    cohort_totals = result["cohorts"][name]
    reusable = cohort_totals.get("reusable_tokens", 0)
    return cohort_totals.get("recomputed_tokens", 0) / reusable if reusable else 0.0


def discrimination(results: dict[str, Any]) -> dict[str, Any]:
    if "gpu-only" not in results or "oracle" not in results:
        return {"status": "not-evaluated"}
    separation = recomputed_share(results["gpu-only"]) - recomputed_share(
        results["oracle"]
    )
    return {
        "status": "discriminating"
        if separation >= SEPARATION
        else "non-discriminating",
        "control_minus_oracle_share": round(separation, 4),
        "required": SEPARATION,
    }


def working_set(
    scenario: dict[str, Any],
    shapes: dict[str, list[list[Turn]]],
    geometry: Geometry,
) -> dict[str, float]:
    sessions = schedule(
        {**scenario["schedule"], "seed": scenario["schedule"].get("seed", 0)}, shapes
    )
    peaks = [
        max(turn.prompt + turn.generated for turn in session.turns)
        for session in sessions
        if len(session.turns) > 1
    ]
    total = sum(peaks)
    keys_per_token = geometry.keys_per_token()
    return {
        "sessions": len(peaks),
        "working_set_tokens": total,
        "gpu_tokens": geometry.gpu_blocks * geometry.block_tokens,
        "cpu_tokens": round(geometry.cpu_keys / keys_per_token),
        "largest_conversation_tokens": max(peaks, default=0),
        "gpu_to_working_set": round(
            geometry.gpu_blocks * geometry.block_tokens / total, 3
        )
        if total
        else 0.0,
        "cpu_to_working_set": round(geometry.cpu_keys / keys_per_token / total, 3)
        if total
        else 0.0,
    }


SCENARIO_SCHEMA = "replay-scenarios-v1"
UNHINTED_TOLERANCE = 0.03


def load_scenarios(path: Path, names: Sequence[str] = ()) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if set(raw) != {"schema", "geometries", "schedules", "scenarios"} or (
        raw["schema"] != SCENARIO_SCHEMA
    ):
        raise ValueError("unsupported replay scenario definition")
    selected = []
    for entry in raw["scenarios"]:
        if set(entry) != {"name", "model", "geometry", "schedule", "seeds"}:
            raise ValueError("scenario entry fields differ")
        if names and entry["name"] not in names:
            continue
        selected.append(
            {
                **entry,
                "geometry": raw["geometries"][entry["geometry"]],
                "schedule": raw["schedules"][entry["schedule"]],
            }
        )
    missing = set(names) - {entry["name"] for entry in selected}
    if missing or not selected:
        raise ValueError(f"unknown or empty scenario selection: {sorted(missing)}")
    return selected


def compare(
    scenarios: Sequence[dict[str, Any]],
    candidate: str = "reachable-hold",
    reference: str = "recency",
    control: str = "native",
) -> dict[str, Any]:
    rows = []
    totals = Counter[str]()
    for scenario in scenarios:
        arms = scenario["arms"]
        if scenario["discrimination"]["status"] != "discriminating":
            rows.append({"name": scenario["name"], "status": "non-discriminating"})
            continue
        ours = arms[candidate]["cohorts"]["hinted"]["recomputed_tokens"]
        theirs = arms[reference]["cohorts"]["hinted"]["recomputed_tokens"]
        harm = recomputed_share(arms[candidate], "unhinted") - recomputed_share(
            arms[control], "unhinted"
        )
        failures = arms[candidate]["activity"].get("failed_stores", 0)
        baseline = arms[control]["activity"].get("failed_stores", 0)
        totals.update({"candidate": ours, "reference": theirs})
        rows.append(
            {
                "name": scenario["name"],
                "status": "compared",
                "candidate_hinted_recomputed": ours,
                "reference_hinted_recomputed": theirs,
                "no_worse_than_reference": ours <= theirs,
                "unhinted_share_change": round(harm, 4),
                "unhinted_within_tolerance": harm <= UNHINTED_TOLERANCE,
                "failed_stores": failures,
                "control_failed_stores": baseline,
                "no_added_store_failures": failures <= baseline,
            }
        )
    compared = [row for row in rows if row["status"] == "compared"]
    passed = bool(compared) and all(
        row["no_worse_than_reference"]
        and row["unhinted_within_tolerance"]
        and row["no_added_store_failures"]
        for row in compared
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "control": control,
        "scenarios": rows,
        "aggregate_candidate_hinted_recomputed": totals["candidate"],
        "aggregate_reference_hinted_recomputed": totals["reference"],
        "simulated_rule_met": passed
        and totals["candidate"] < 0.9 * totals["reference"],
    }


def simulate(
    root: Path,
    runtime_python: Path,
    scenarios: Path,
    shapes: Path,
    names: Sequence[str],
    output: Path,
) -> int:
    if output.exists():
        raise ValueError("simulation requires a new output directory")
    command = [
        str(runtime_python),
        "-m",
        "tools.benchmark",
        "simulate-native",
        "--scenarios",
        str(scenarios.resolve()),
        "--shapes",
        str(shapes.resolve()),
        "--out",
        str(output.resolve()),
    ]
    for name in names:
        command.extend(["--scenario", name])
    return run_owned(command, check_environment(root), root, 1800)


LIMITS = (
    "Simulated direction only: arrival times are arm-independent, loads and store "
    "completions are fixed delays, deferred lookups count as misses, and hybrid "
    "cache groups use the declared chunk/window geometry. Gemma's group layout is "
    "inferred from observed CPU usage, not read from the engine."
)


def simulate_native(
    scenarios: Path, shapes: Path, names: Sequence[str], output: Path
) -> dict[str, Any]:
    native = importlib.import_module("wait_hint_offload.vllm_adapter.replay")
    selected = load_scenarios(scenarios, names)
    recorded = load_shapes(shapes)
    results = [evaluate(entry, recorded, native.NativeCache) for entry in selected]
    record: dict[str, Any] = {
        "schema": "replay-simulation-v1",
        "native": native.native_identity(),
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (scenarios, shapes)
        },
        "arms": list(ARMS),
        "limits": LIMITS,
        "scenarios": results,
        "comparison": compare(results),
    }
    output.mkdir(parents=True)
    write_record(output / "simulation.json", record)
    comparison: dict[str, Any] = record["comparison"]
    return comparison
