"""Counterexamples to overclaiming a noisy pilot or per-request sample size."""
import pytest
from analyze_warm_confirmation import ci6

def test_positive_mean_with_noise_does_not_establish_win():
    result = ci6([100, -100, 100, -100, 100, -90])
    assert result['mean'] > 0 and result['lower95'] < 0

def test_all_six_repeats_needed():
    with pytest.raises(AssertionError):ci6([100] * 5)

def test_repeated_positive_shift_can_establish_win():
    result = ci6([60, 70, 65, 75, 80, 72])
    assert result['lower95'] > 0 and result['n_independent_workflow_pairs'] == 6
