from dataclasses import FrozenInstanceError, replace

import pytest

from semantic_offload.contracts import (
    ExactKVIdentity,
    RetentionCandidate,
    RetentionScore,
    same_kv_identity,
)


def identity() -> ExactKVIdentity:
    return ExactKVIdentity((1, 2), "model-a", None, (0, 1), "layout-a")


def test_g005_exact_identity_positive() -> None:
    stored = identity()
    assert (
        stored.prefix_tokens,
        stored.model,
        stored.adapter,
        stored.positions,
        stored.layout,
    ) == ((1, 2), "model-a", None, (0, 1), "layout-a")
    assert same_kv_identity(stored, identity())


@pytest.mark.parametrize(
    "different",
    [
        replace(identity(), prefix_tokens=(1, 3)),
        replace(identity(), model="model-b"),
        replace(identity(), adapter="adapter-b"),
        replace(identity(), positions=(1, 2)),
        replace(identity(), layout="layout-b"),
    ],
    ids=["tokens", "model", "adapter", "positions", "layout"],
)
def test_g005_identity_difference_refuses_reuse(different: ExactKVIdentity) -> None:
    assert not same_kv_identity(identity(), different)


def test_g005_scores_rank_without_substitution() -> None:
    request = identity()
    matching = RetentionCandidate(request, RetentionScore(-1.0))
    high_score = RetentionCandidate(
        replace(request, prefix_tokens=(1, 3)), RetentionScore(1.0)
    )
    ranked = sorted([matching, high_score], key=lambda c: c.score, reverse=True)
    assert ranked[0] == high_score
    assert not same_kv_identity(ranked[0].identity, request)
    assert same_kv_identity(ranked[1].identity, request)


def test_g005_identity_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(identity(), "model", "changed")
