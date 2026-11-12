from dataclasses import replace
from typing import Any, cast

import pytest

from wait_hint_offload.advice import advise
from wait_hint_offload.config import Settings
from wait_hint_offload.contracts import Hint, Namespace, prefix_digest
from wait_hint_offload.observed import (
    AdviceDecision,
    Observation,
    ObservedGroup,
    ObservedStore,
    RegistrationDecision,
)


class Target:
    def __init__(self) -> None:
        self.calls = 0
        self.available = True

    def ready_prefix(self, keys: tuple[bytes, ...]) -> tuple[bytes, ...]:
        return keys if self.available else ()

    def refresh_prefix(self, keys: tuple[bytes, ...]) -> int:
        self.calls += 1
        return len(self.ready_prefix(keys))


def sample() -> tuple[ObservedStore, Target, Hint]:
    ns = Namespace("m", "t", "none", "dense-fp16")
    digest = prefix_digest([1, 2], ns)
    observation = Observation("epoch", 1, digest, 2, ns, 10, (b"a", b"b"))
    store = ObservedStore(2, 2)
    assert store.register("scope", "session", observation).accepted
    hint = Hint(
        "scope", "session", "epoch", 1, "event", digest, 2, ns, 10, 11, 100, "boot"
    )
    return store, Target(), hint


def test_group_observations_require_aligned_boundary_and_full_anchor() -> None:
    store, _, _ = sample()
    observation = store.observation("scope", "session")
    assert observation is not None
    full = ObservedGroup(1, None, (b"a", b"b"))
    recurrent = ObservedGroup(2, 1, (b"state",))
    replacement = replace(observation, groups=(full, recurrent), generation=2)
    assert store.register("scope", "session", replacement).accepted
    for groups in ((recurrent,), (full, ObservedGroup(1, 1, (b"state",)))):
        with pytest.raises(ValueError, match="boundary or anchor"):
            replace(observation, groups=groups)
    with pytest.raises(ValueError, match="group keys"):
        ObservedGroup(2, 1, ())


def test_advice_replay_conflict_and_disabled() -> None:
    for enabled in (False, True):
        store, target, hint = sample()
        first = advise(store, target, hint, 12, "boot", enabled)
        assert first.decision is (
            AdviceDecision.REFRESHED if enabled else AdviceDecision.DISABLED
        )
        assert first.refreshed_keys == (2 if enabled else 0)
        target.available = False
        replay = advise(store, target, hint, 101, "boot", enabled)
        assert replay == replace(first, replayed=True)
        assert target.calls == int(enabled)
        assert (
            advise(
                store, target, replace(hint, deadline_ns=90), 101, "boot", enabled
            ).decision
            is AdviceDecision.EVENT_CONFLICT
        )
        for event in ("two", "three"):
            result = advise(
                store, target, replace(hint, event=event), 13, "boot", enabled
            )
        assert result.decision is AdviceDecision.EVENT_CAPACITY


def test_wrong_identity_is_no_action() -> None:
    store, target, hint = sample()
    for invalid in (
        replace(hint, generation=2),
        replace(hint, incarnation="old"),
        replace(hint, prefix="wrong"),
        replace(hint, boundary=1),
        replace(hint, namespace=Namespace("other", "t", "none", "dense-fp16")),
    ):
        assert (
            advise(store, target, invalid, 12, "boot", True).decision
            is AdviceDecision.IDENTITY_MISMATCH
        )
    assert target.calls == 0


def test_clock_and_reset() -> None:
    store, target, hint = sample()
    for now, clock in ((100, "boot"), (9, "boot"), (12, "other")):
        assert (
            advise(store, target, hint, now, clock, True).decision
            is AdviceDecision.CLOCK_OR_DEADLINE
        )
    advise(store, target, hint, 12, "boot", True)
    assert (
        advise(store, target, hint, 12, "other", True).decision
        is AdviceDecision.CLOCK_OR_DEADLINE
    )
    store.clear()
    assert store.charged_bytes == 0
    assert (
        advise(store, target, hint, 12, "boot", True).decision
        is AdviceDecision.UNOBSERVED
    )


def test_session_capacity_and_generation() -> None:
    store, target, hint = sample()
    original = store.observation("scope", "session")
    assert original is not None
    for generation in (0, 1):
        assert (
            store.register(
                "scope", "session", replace(original, generation=generation)
            ).decision
            is RegistrationDecision.GENERATION_NOT_NEW
        )
    assert store.register("scope", "session", replace(original, generation=2)).accepted
    assert (
        advise(store, target, hint, 12, "boot", True).decision
        is AdviceDecision.IDENTITY_MISMATCH
    )
    assert store.register(
        "scope", "session", replace(original, incarnation="new", generation=0)
    ).accepted
    for name in ("second", "third"):
        assert store.register("scope", name, original).accepted
    assert store.retired == 1
    assert (
        advise(store, target, hint, 12, "boot", True).decision
        is AdviceDecision.UNOBSERVED
    )


def test_schema_refusal_and_config() -> None:
    namespace = Namespace("m", "t", "none", "dense-fp16")
    settings = Settings.parse(
        {"namespace": namespace.mapping(), "cpu_bytes_to_use": 1024}
    )
    assert settings.cpu_bytes_to_use == 1024
    assert settings.mechanisms() == {
        "schema": "wait-hint-mechanisms-v1",
        "cpu_arc_recency_advice": False,
    }
    for key in (
        "wait_hint_publish_at_lookup",
        "scoring_mode",
        "prefetch_budget_fraction",
        "store_threshold",
        "eviction_policy",
    ):
        with pytest.raises(ValueError, match="unknown settings"):
            Settings.parse({key: True})
    with pytest.raises(ValueError, match="schema"):
        Hint.parse({"schema": "wait-hint-v99"})
    with pytest.raises(ValueError, match="token"):
        prefix_digest([-1], namespace)
    for value in (0, True, 1.5):
        with pytest.raises(ValueError, match="bounds"):
            ObservedStore(cast(Any, value), 1)


def test_retirement_and_reregistration_cannot_revalidate_old_hint() -> None:
    store, target, hint = sample()
    original = store.observation(hint.scope, hint.session)
    assert original is not None
    for name in ("two", "three"):
        assert store.register(hint.scope, name, original).accepted
    assert store.register(
        hint.scope, hint.session, replace(original, observed_ns=11)
    ).accepted
    assert (
        advise(store, target, hint, 12, "boot", True).decision
        is AdviceDecision.IDENTITY_MISMATCH
    )
    assert target.calls == 0


def test_memory_refusal_preserves_state_and_does_not_refresh() -> None:
    store, target, hint = sample()
    original = store.observation(hint.scope, hint.session)
    assert original is not None
    bounded = ObservedStore(2, 2, store.charged_bytes)
    assert bounded.register(hint.scope, hint.session, original).accepted
    charge = bounded.charged_bytes
    assert (
        advise(bounded, target, hint, 12, "boot", True).decision
        is AdviceDecision.MEMORY_CAPACITY
    )
    assert target.calls == 0 and bounded.charged_bytes == charge
    assert (
        bounded.register("scope", "second", original).decision
        is RegistrationDecision.MEMORY_CAPACITY
    )
    assert bounded.observation(hint.scope, hint.session) == original
    assert bounded.register(
        hint.scope, hint.session, replace(original, generation=2)
    ).accepted
    assert bounded.charged_bytes == charge
    bounded.clear()
    assert bounded.charged_bytes == 0


def test_new_generation_discards_replay_and_reclaims_memory() -> None:
    store, target, hint = sample()
    original = store.observation(hint.scope, hint.session)
    assert original is not None
    charge = store.charged_bytes
    advise(store, target, hint, 12, "boot", True)
    assert store.charged_bytes > charge
    assert store.register(
        hint.scope, hint.session, replace(original, generation=2)
    ).accepted
    assert store.charged_bytes == charge
    assert (
        advise(store, target, hint, 12, "boot", True).decision
        is AdviceDecision.IDENTITY_MISMATCH
    )


def test_direct_construction_rejects_invalid_state() -> None:
    from wait_hint_offload.contracts import Registration

    ns = Namespace("m", "t", "none", "dense-fp16")
    hint = Hint("s", "t", "i", 1, "e", "p", 4, ns, 10, 11, 20, "engine")
    changed: dict[str, Any]
    for changed in (
        {"generation": True},
        {"boundary": 0},
        {"observed_ns": -1},
        {"deadline_ns": 11},
        {"namespace": None},
        {"scope": ""},
    ):
        with pytest.raises(ValueError):
            replace(hint, **cast(Any, changed))
    with pytest.raises(ValueError):
        Namespace("", "t", "a", "l")
    with pytest.raises(ValueError):
        Registration("s", "t", "i", True)
    settings = Settings(ns, 1024, False, 2, 2)
    for changed in (
        {"cpu_bytes_to_use": True},
        {"advice_enabled": 1},
        {"max_sessions": 0},
        {"max_events": -1},
        {"metadata_bytes": 2**30},
    ):
        with pytest.raises(ValueError):
            replace(settings, **cast(Any, changed))
    observation = Observation("i", 1, "p", 4, ns, 10, (b"key",))
    for changed in (
        {"keys": ()},
        {"keys": (b"x" * 257,)},
        {"keys": [b"key"]},
        {"boundary": True},
    ):
        with pytest.raises(ValueError):
            replace(observation, **cast(Any, changed))
