import pytest

from benchmarks.research_checks.hybrid_pilot_cache import reusable


def test_reuse_requires_same_tokens_model_and_capture_code():
    manifest = dict(model="model-a", capture_source_sha256="code-a")
    old = [dict(session_id="a", split="train", tokens=[1, 2])]
    prior = {"a": dict(session_id="a", split="train", scalar=[3.])}
    full = old + [dict(session_id="b", split="test", tokens=[4])]
    used, remaining = reusable(manifest, full, manifest, old, prior)
    assert list(used) == ["a"] and remaining == full[1:]
    changed = [dict(old[0], tokens=[1, 9])]
    used, remaining = reusable(manifest, changed, manifest, old, prior)
    assert not used and remaining == changed
    with pytest.raises(ValueError, match="changed"):
        reusable(dict(manifest, model="model-b"), full, manifest, old, prior)
    with pytest.raises(ValueError, match="changed"):
        reusable(dict(manifest, capture_source_sha256="code-b"), full, manifest, old, prior)
    with pytest.raises(ValueError, match="population"):
        reusable(manifest, full, manifest, old, {})
