from __future__ import annotations

from tools.benchmarking.replay import Geometry, Replay
from tools.benchmarking.workload import Session, Turn
from wait_hint_offload.vllm_adapter.replay import NativeCache, native_identity


def geometry(cpu_keys: int) -> Geometry:
    return Geometry(
        16,
        ((64, None),),
        512,
        cpu_keys,
        1,
        2048,
        4000.0,
        0.01,
        0.01,
        0.01,
        60.0,
        0.5,
        True,
    )


def schedule() -> list[Session]:
    return [
        Session(
            "a",
            0.0,
            (Turn(1000, 10, 0, 0.0), Turn(1500, 10, 1009, 10.0)),
            True,
            False,
        ),
        Session("b", 2.0, (Turn(8000, 16, 0, 0.0),), False, False),
    ]


def test_simulator_drives_the_pinned_manager_and_product_advice() -> None:
    restored = Replay(schedule(), geometry(1000), "native", NativeCache).run()
    assert restored["cohorts"]["hinted"]["cpu_tokens"] == 784
    assert restored["cohorts"]["hinted"]["recomputed_tokens"] == 48
    pressured = {
        arm: Replay(schedule(), geometry(130), arm, NativeCache).run()
        for arm in ("native", "recency", "promote-hold", "reachable-hold")
    }
    assert pressured["native"]["activity"]["cpu_evictions"] > 0
    assert pressured["promote-hold"]["activity"]["advised_keys"] == 15
    assert (
        pressured["promote-hold"]["cohorts"]["hinted"]["recomputed_tokens"]
        < pressured["recency"]["cohorts"]["hinted"]["recomputed_tokens"]
    )
    assert set(native_identity()) == {"arc_policy_sha256", "cpu_manager_sha256"}


def test_recurrent_groups_store_only_scheduling_step_checkpoints() -> None:
    hybrid = Geometry(
        16,
        ((64, None), (64, 1)),
        512,
        1000,
        1,
        256,
        4000.0,
        0.01,
        0.01,
        0.01,
        60.0,
        0.5,
        False,
    )
    sessions = [Session("a", 0.0, (Turn(1024, 1, 0, 0.0),), False, False)]
    result = Replay(sessions, hybrid, "native", NativeCache).run()
    assert result["activity"]["cpu_stored_keys"] == 16 + 4
