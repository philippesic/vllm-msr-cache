import pytest

from wait_hint_offload.policy_holds import HoldDecision, HoldLedger


@pytest.mark.parametrize("capacity", [1, 2, 16, 1024])
@pytest.mark.parametrize("share", [0.0, 0.25, 0.5])
def test_shared_keys_count_once_and_release_preserves_other_owner(
    capacity: int, share: float
) -> None:
    book = HoldLedger(capacity, share)
    keys = frozenset(index.to_bytes(4, "big") for index in range(book.limit))
    assert book.hold("a", 1, 100, keys, 0) == HoldDecision.HELD
    assert book.hold("b", 1, 110, keys, 0) == HoldDecision.HELD
    assert len(book.references) == book.limit
    protection = book.protection({b"transfer"})
    assert b"transfer" in protection
    assert set(protection) == set(keys) | {b"transfer"}
    assert len(protection) == len(keys) + 1
    book.release("a")
    assert set(book.references) == keys
    assert book.yield_oldest()
    assert book.yields == 1
    assert book.references == {}
    assert book.metadata_bytes == 0
    assert not book.yield_oldest()


def test_hold_limits_and_expiry_preserve_accepted_state() -> None:
    book = HoldLedger(4, max_sessions=2, metadata_limit=8000)
    assert book.hold("a", 2, 10, frozenset({b"a"}), 0) == HoldDecision.HELD
    assert book.hold("a", 1, 20, frozenset({b"b"}), 0) == HoldDecision.STALE
    assert book.hold("b", 1, 20, frozenset({b"b"}), 0) == HoldDecision.HELD
    assert book.hold("c", 1, 20, frozenset(), 0) == HoldDecision.SESSION_LIMIT
    assert (
        book.hold("a", 2, 10, frozenset({b"a", b"c"}), 0) == HoldDecision.PHYSICAL_LIMIT
    )
    assert set(book.references) == {b"a", b"b"}
    book.expire(10)
    assert set(book.references) == {b"b"}
    assert book.hold("c", 1, 10, frozenset(), 10) == HoldDecision.EXPIRED
    assert book.hold("c", 1, 20, frozenset({b"z" * 256}), 10) == HoldDecision.HELD
    book.clear()
    assert not book.references and not book.deadlines and not book.leases
    assert book.metadata_bytes == 0


def test_metadata_refusal_and_replacement_churn_are_bounded() -> None:
    book = HoldLedger(1024, metadata_limit=4096)
    assert book.hold("a", 1, 1000, frozenset({b"a"}), 0) == HoldDecision.HELD
    assert book.hold("b", 1, 1000, frozenset({b"b"}), 0) == HoldDecision.METADATA_LIMIT
    charge = book.metadata_bytes
    for generation in range(2, 300):
        assert (
            book.hold("a", generation, 1000, frozenset({b"b"}), 0) == HoldDecision.HELD
        )
        assert book.metadata_bytes == charge
    assert len(book.deadlines) <= 64
    book.expire(1000)
    assert book.metadata_bytes == 0


def test_policy_wire_contracts_refuse_unknown_fields_and_boolean_numbers() -> None:
    from wait_hint_offload.policy_protocol import PolicyAction, PolicySession

    session = {
        "schema": "wait-hint-policy-session-v1",
        "scope": "scope",
        "session": "s",
        "incarnation": "i",
        "generation": 1,
    }
    action = {
        "schema": "wait-hint-policy-action-v1",
        "action_id": "event",
        "session": session,
        "kind": "hold",
        "ttl_ms": 12000,
    }
    assert PolicyAction.parse(action).to_dict() == action
    for changed in (
        session | {"extra": 1},
        session | {"generation": True},
        session | {"schema": "v2"},
    ):
        with pytest.raises(ValueError):
            PolicySession.parse(changed)
    for changed in (
        action | {"extra": 1},
        action | {"ttl_ms": True},
        action | {"schema": "v2"},
        action | {"kind": "prefetch"},
    ):
        with pytest.raises(ValueError):
            PolicyAction.parse(changed)
    unicode_session = PolicySession.parse(session | {"scope": "猫" * 128})
    assert len(unicode_session.owner) < 512
