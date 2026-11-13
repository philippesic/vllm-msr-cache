from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarking.replay import (
    Geometry,
    Gpu,
    Replay,
    compare,
    discrimination,
    load_scenarios,
)
from tools.benchmarking.workload import (
    Session,
    Turn,
    extract_shapes,
    load_shapes,
    schedule,
)
from wait_hint_offload.observed import ObservedGroup

ROOT = Path(__file__).resolve().parents[1]


class LruCache:
    def __init__(self, cpu_keys: int, hold_share: float) -> None:
        self.capacity = cpu_keys
        self.entries: OrderedDict[bytes, bool] = OrderedDict()
        self.now = 0.0
        self.evicted = 0

    def ready(self, key: bytes) -> bool:
        return self.entries.get(key, False)

    def touch(self, keys: Sequence[bytes]) -> None:
        for key in reversed(keys):
            if key in self.entries:
                self.entries.move_to_end(key)

    def store(self, keys: Sequence[bytes]) -> tuple[list[bytes], int] | None:
        new = [key for key in keys if key not in self.entries]
        victims = [
            key for key, ready in self.entries.items() if ready and key not in keys
        ][: max(0, len(self.entries) + len(new) - self.capacity)]
        if len(self.entries) - len(victims) + len(new) > self.capacity:
            return None
        for key in victims:
            del self.entries[key]
        self.entries.update(dict.fromkeys(new, False))
        return new, len(victims)

    def complete(self, keys: Sequence[bytes]) -> None:
        for key in keys:
            if key in self.entries:
                self.entries[key] = True

    def advise(
        self, arm: str, groups: Sequence[ObservedGroup], owner: str, deadline: float
    ) -> tuple[int, int]:
        ready = 0
        for key in groups[0].keys:
            if not self.ready(key):
                break
            ready += 1
        self.touch(groups[0].keys[:ready])
        return ready * groups[0].tokens_per_chunk, ready

    def release(self, owner: str) -> None:
        return None

    def counters(self) -> dict[str, int]:
        return {}


def geometry(**changes: Any) -> Geometry:
    values: dict[str, Any] = {
        "block_tokens": 16,
        "groups": ((64, None),),
        "gpu_blocks": 512,
        "cpu_keys": 1000,
        "max_running": 1,
        "batch_tokens": 2048,
        "prefill_tokens_per_second": 4000.0,
        "decode_seconds_per_token": 0.01,
        "store_seconds": 0.01,
        "hint_delay_seconds": 0.01,
        "hint_deadline_seconds": 60.0,
        "hold_share": 0.5,
        "gpu_prefix_reuse": True,
    }
    return Geometry(**{**values, **changes})


def sessions(competitor: bool) -> list[Session]:
    returning = Session(
        "a",
        0.0,
        (Turn(1000, 10, 0, 0.0), Turn(1500, 10, 1009, 10.0)),
        True,
        False,
    )
    other = Session("b", 2.0, (Turn(8000, 16, 0, 0.0),), False, False)
    return [returning, other] if competitor else [returning]


def hinted(result: dict[str, Any]) -> dict[str, float]:
    cohort: dict[str, float] = result["cohorts"]["hinted"]
    return cohort


def test_turns_and_sessions_refuse_impossible_shapes() -> None:
    with pytest.raises(ValueError):
        Turn(100, 10, 100, 0.0)
    with pytest.raises(ValueError):
        Turn(8000, 400, 0, 0.0)
    with pytest.raises(ValueError):
        Session("a", 0, (Turn(100, 1, 0, 0), Turn(200, 1, 101, 0)), True, False)
    with pytest.raises(ValueError):
        Session("a", 0, (Turn(100, 1, 0, 0),), False, True)


def test_gpu_prefix_cache_evicts_the_tail_of_a_released_request_first() -> None:
    gpu = Gpu(4)
    assert gpu.acquire([1, 2, 3], 3) == 0
    assert gpu.release([1, 2, 3], 3) == 0
    assert gpu.acquire([4, 5], 2) == 1
    assert gpu.cached == {1, 2, 4, 5}
    assert gpu.hit([1, 2, 3], 3) == 2


def test_replay_accounts_gpu_reuse_displacement_and_cpu_restore() -> None:
    alone = Replay(sessions(False), geometry(), "gpu-only", LruCache).run()
    assert hinted(alone)["reusable_tokens"] == 1008
    assert hinted(alone)["gpu_tokens"] == 1008
    assert hinted(alone)["recomputed_tokens"] == 0
    displaced = Replay(sessions(True), geometry(), "gpu-only", LruCache).run()
    assert hinted(displaced)["gpu_tokens"] == 176
    assert hinted(displaced)["recomputed_tokens"] == 832
    assert displaced["activity"]["gpu_displaced_returns"] == 1
    restored = Replay(sessions(True), geometry(), "native", LruCache).run()
    assert hinted(restored)["cpu_tokens"] == 784
    assert hinted(restored)["recomputed_tokens"] == 48
    assert restored["activity"]["cpu_restores"] == 1
    assert restored["activity"]["hints"] == 1
    small = Replay(sessions(True), geometry(cpu_keys=4), "native", LruCache).run()
    assert small["activity"]["failed_stores"] > 0
    assert hinted(small)["recomputed_tokens"] == 832


def test_geometry_and_scenarios_are_closed_and_tracked_definitions_load(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        geometry(groups=((64, 1),))
    with pytest.raises(ValueError):
        Geometry.parse({"block_tokens": 16})
    selected = load_scenarios(ROOT / "benchmarks/replay/scenarios.json")
    shapes = load_shapes(ROOT / "benchmarks/replay/release-01-shapes.json")
    for scenario in selected:
        Geometry.parse(scenario["geometry"])
        assert schedule({**scenario["schedule"], "seed": 0}, shapes)
    with pytest.raises(ValueError):
        load_scenarios(ROOT / "benchmarks/replay/scenarios.json", ["missing"])
    raw = json.loads((ROOT / "benchmarks/replay/scenarios.json").read_text())
    raw["extra"] = 1
    (tmp_path / "bad.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_scenarios(tmp_path / "bad.json")


def test_schedules_are_seeded_closed_and_need_recorded_shapes() -> None:
    spec = load_scenarios(
        ROOT / "benchmarks/replay/scenarios.json", ["qwen-10s-waits"]
    )[0]["schedule"]
    first = schedule({**spec, "seed": 3})
    assert first == schedule({**spec, "seed": 3})
    assert first != schedule({**spec, "seed": 4})
    assert {s.hinted for s in first} == {True, False}
    with pytest.raises(ValueError):
        schedule({**spec, "seed": 3, "unknown": 1})
    with pytest.raises(ValueError):
        schedule({**spec, "seed": 3, "source": "qwen35-4b-int4"}, {})


def test_shape_extraction_reads_unpressured_reuse_and_gaps(tmp_path: Path) -> None:
    cell = tmp_path / "sessions/0/3"
    requests = cell / "panel/0/requests"
    requests.mkdir(parents=True)
    (cell / "admission.json").write_text(
        json.dumps({"command": ["serve", "--model", "/models/Qwen3.5"]})
    )
    for index, (prompt, generated, cached, start, end) in enumerate(
        ((300, 20, 0, 0, 10**9), (500, 30, 304, 3 * 10**9, 4 * 10**9))
    ):
        record = {
            "session": "main",
            "start_ns": start,
            "received_ns": end,
            "response": {
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": generated,
                    "prompt_tokens_details": {"cached_tokens": cached},
                }
            },
        }
        (requests / f"request-{index:04d}.json").write_text(json.dumps(record))
        (requests / f"request-{index:04d}.admission.json").write_text("{}")
    shapes = extract_shapes(tmp_path)
    assert shapes["models"] == {
        "qwen35-4b-int4": [[[300, 20, 0, 0.0], [500, 30, 304, 2.0]]]
    }
    (tmp_path / "shapes.json").write_text(json.dumps(shapes))
    loaded = load_shapes(tmp_path / "shapes.json")
    assert loaded["qwen35-4b-int4"][0][1] == Turn(500, 30, 304, 2.0)


def arm(hinted: int, unhinted: int, failed: int = 0) -> dict[str, Any]:
    return {
        "cohorts": {
            "hinted": {"recomputed_tokens": hinted, "reusable_tokens": 1000},
            "unhinted": {"recomputed_tokens": unhinted, "reusable_tokens": 1000},
        },
        "activity": {"failed_stores": failed},
    }


def scenario(name: str, **arms: dict[str, Any]) -> dict[str, Any]:
    results = {"gpu-only": arm(900, 900), "oracle": arm(100, 100), **arms}
    return {"name": name, "arms": results, "discrimination": discrimination(results)}


def test_comparison_applies_every_clause_of_the_decision_rule() -> None:
    good = scenario(
        "good",
        native=arm(500, 300, 2),
        recency=arm(450, 300),
        **{"promote-hold": arm(200, 320, 2)},
    )
    assert good["discrimination"]["status"] == "discriminating"
    assert compare([good])["simulated_rule_met"]
    worse = scenario(
        "worse",
        native=arm(500, 300),
        recency=arm(450, 300),
        **{"promote-hold": arm(460, 300)},
    )
    assert not compare([good, worse])["simulated_rule_met"]
    harm = scenario(
        "harm",
        native=arm(500, 300),
        recency=arm(450, 300),
        **{"promote-hold": arm(200, 340)},
    )
    assert not compare([harm])["simulated_rule_met"]
    failing = scenario(
        "failing",
        native=arm(500, 300, 1),
        recency=arm(450, 300),
        **{"promote-hold": arm(200, 300, 2)},
    )
    assert not compare([failing])["simulated_rule_met"]
    close = scenario(
        "close",
        native=arm(500, 300),
        recency=arm(450, 300),
        **{"promote-hold": arm(440, 300)},
    )
    assert not compare([close])["simulated_rule_met"]
    blind: dict[str, Any] = {
        "gpu-only": arm(150, 150),
        "oracle": arm(100, 100),
        "native": arm(100, 100),
        "recency": arm(100, 100),
        "promote-hold": arm(100, 100),
    }
    flat: dict[str, Any] = {
        "name": "flat",
        "arms": blind,
        "discrimination": discrimination(blind),
    }
    assert flat["discrimination"]["status"] == "non-discriminating"
    assert not compare([flat])["simulated_rule_met"]
    assert compare([flat, good])["simulated_rule_met"]
