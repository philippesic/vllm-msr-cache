import math
import pytest
from benchmarks.serving_suite.campaign_report import classify, paired_estimate, plan_repeats


def test_gain_and_time_reduction_differ():
    result = paired_estimate([1.10] * 6)
    assert result['throughput_gain_pct'] == pytest.approx(10)
    assert result['time_reduction_pct'] == pytest.approx(100 / 11)
    assert result['process_pairs'] == 6
    assert classify(result) == 'benefit'


def test_losses_and_uncertainty_retained():
    assert classify(paired_estimate([0.9] * 6)) == 'regression'
    assert classify(paired_estimate([1.0] * 6)) == 'bounded regression/neutral'
    assert classify(paired_estimate([0.8, 1.2, 1.0, 1.3, 0.7, 1.1])) == 'inconclusive'
    assert classify(paired_estimate([1.1])) == 'inconclusive'


def test_precision_is_bounded_and_balanced():
    assert plan_repeats([1, 1, 1])['repeats'] == 6
    assert plan_repeats([1, 1, 1], four_arm=True)['repeats'] == 8
    noisy = plan_repeats([0.7, 1, 1.3], four_arm=True)
    assert noisy['repeats'] == 12
    assert not noisy['target_expected_met']
    with pytest.raises(ValueError):
        plan_repeats([1, 1])


@pytest.mark.parametrize('bad', [[], [0], [-1], [math.nan], [math.inf]])
def test_invalid_ratios_rejected(bad):
    with pytest.raises(ValueError):
        paired_estimate(bad)
