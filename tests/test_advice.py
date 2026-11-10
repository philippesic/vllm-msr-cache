from dataclasses import replace

import pytest

from wait_hint_offload.advice import advise
from wait_hint_offload.config import Settings
from wait_hint_offload.contracts import Hint, Namespace, prefix_digest
from wait_hint_offload.observed import Observation, ObservedStore


class Target:
    def __init__(self) -> None:
        self.calls = 0

    def refresh_prefix(self, keys: tuple[bytes, ...]) -> int:
        self.calls += 1
        return len(keys)


def sample() -> tuple[ObservedStore, Target, Hint]:
    ns = Namespace("m", "t", "none", "dense-fp16")
    digest = prefix_digest([1, 2], ns)
    observation = Observation("epoch", 1, digest, 2, ns, 10, (b"a", b"b"))
    store = ObservedStore(2, 2)
    assert store.register("scope", "session", observation)
    hint = Hint(
        "scope", "session", "epoch", 1, "event", digest, 2, ns, 10, 11, 100, "boot"
    )
    return store, Target(), hint


def test_advice_replay_conflict_and_disabled() -> None:
    store, target, hint = sample()
    assert advise(store, target, hint, 12, "boot", False) == ("disabled", 0)
    assert advise(store, target, hint, 12, "boot", True) == ("refreshed", 2)
    assert advise(store, target, hint, 13, "boot", True) == ("refreshed", 2)
    assert target.calls == 1
    assert advise(store, target, replace(hint, deadline_ns=90), 13, "boot", True) == (
        "event_conflict",
        0,
    )
    for event in ("two", "three"):
        decision = advise(store, target, replace(hint, event=event), 13, "boot", True)
    assert decision == ("event_capacity", 0)


def test_wrong_identity_is_no_action() -> None:
    store, target, hint = sample()
    variants = (
        replace(hint, generation=2),
        replace(hint, incarnation="old"),
        replace(hint, prefix="wrong"),
        replace(hint, boundary=1),
        replace(hint, namespace=Namespace("other", "t", "none", "dense-fp16")),
    )
    for invalid in variants:
        assert advise(store, target, invalid, 12, "boot", True) == (
            "identity_mismatch",
            0,
        )
    assert target.calls == 0


def test_clock_and_reset() -> None:
    store, target, hint = sample()
    for now, clock in ((100, "boot"), (9, "boot"), (12, "other")):
        assert advise(store, target, hint, now, clock, True) == ("clock_or_deadline", 0)
    store.clear()
    assert advise(store, target, hint, 12, "boot", True) == ("unobserved", 0)


def test_session_capacity_and_generation() -> None:
    store, target, hint = sample()
    original = store.sessions[("scope", "session")].observation
    assert not store.register("scope", "session", replace(original, generation=0))
    assert not store.register("scope", "session", replace(original, incarnation="old"))
    assert store.register("scope", "session", original)
    assert store.register("scope", "session", replace(original, generation=2))
    assert advise(store, target, hint, 12, "boot", True) == ("identity_mismatch", 0)
    for name in ("second", "third"):
        assert store.register("scope", name, original)
    assert store.retired == 1
    assert advise(store, target, hint, 12, "boot", True) == ("unobserved", 0)


def test_schema_refusal_and_config() -> None:
    namespace = Namespace("m", "t", "none", "dense-fp16")
    settings = Settings.parse(
        {"namespace": namespace.mapping(), "cpu_bytes_to_use": 1024}
    )
    assert settings.cpu_bytes_to_use == 1024 and settings.store_threshold == 0
    assert settings.mechanisms()["cpu_arc_recency_advice"] is False
    for key in (
        "wait_hint_publish_at_lookup",
        "scoring_mode",
        "prefetch_budget_fraction",
    ):
        with pytest.raises(ValueError, match="unknown settings"):
            Settings.parse({key: True})
    with pytest.raises(ValueError, match="schema"):
        Hint.parse({"schema": "wait-hint-v99"})
    with pytest.raises(ValueError, match="token"):
        prefix_digest([-1], namespace)
    with pytest.raises(ValueError, match="bounds"):
        ObservedStore(0, 1)


def test_retirement_and_reregistration_cannot_revalidate_old_hint() -> None:
    store, target, hint = sample()
    original = store.sessions[(hint.scope, hint.session)].observation
    for name in ("two", "three"):
        assert store.register(hint.scope, name, original)
    assert store.register(hint.scope, hint.session, replace(original, observed_ns=11))
    assert advise(store, target, hint, 12, "boot", True) == ("identity_mismatch", 0)
    assert target.calls == 0
