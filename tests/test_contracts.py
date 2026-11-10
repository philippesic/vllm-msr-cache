from dataclasses import FrozenInstanceError, replace

import pytest

from wait_hint_offload.contracts import Hint, Namespace, prefix_digest


def namespace() -> Namespace:
    return Namespace("m", "t", "none", "dense-fp16")


def test_digest_external_vector() -> None:
    assert (
        prefix_digest([1, 2], namespace())
        == "9723481858567a97096d499e21cd956e9513ab7ee7f888477612f59c87252af3"
    )
    assert prefix_digest([2, 1], namespace()) != prefix_digest([1, 2], namespace())


@pytest.mark.parametrize(
    "field",
    ["model_revision", "tokenizer_revision", "adapter_namespace", "layout_namespace"],
)
def test_namespace_changes_digest(field: str) -> None:
    changed = replace(namespace(), **{field: "different"})
    assert prefix_digest([1, 2], changed) != prefix_digest([1, 2], namespace())


def test_contracts_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(namespace(), "model_revision", "changed")


def test_schema_and_causal_clock_refusal() -> None:
    hint = Hint(
        "scope",
        "session",
        "epoch",
        1,
        "event",
        "digest",
        4,
        namespace(),
        10,
        11,
        20,
        "engine",
    )
    assert Hint.parse(hint.mapping()) == hint
    for changed in (
        {"schema": "wait-hint-v999"},
        {"observed_ns": 12},
        {"published_ns": 30},
        {"unknown": True},
    ):
        with pytest.raises(ValueError):
            Hint.parse({**hint.mapping(), **changed})


def test_registration_generations_and_closed_fields() -> None:
    from wait_hint_offload.contracts import Registration

    raw = {
        "schema": "wait-registration-v2",
        "scope": "client",
        "session": "conversation",
        "incarnation": "instance",
        "generation": 0,
    }
    parsed = Registration.parse(raw)
    assert (parsed.scope, parsed.session, parsed.incarnation, parsed.generation) == (
        "client",
        "conversation",
        "instance",
        0,
    )
    for value in (True, False, -1, 1.0, "1", 2**63):
        with pytest.raises(ValueError, match="generation"):
            Registration.parse({**raw, "generation": value})
    for changed in ({"schema": "unknown"}, {"extra": 1}, {"scope": ""}):
        with pytest.raises(ValueError):
            Registration.parse({**raw, **changed})
