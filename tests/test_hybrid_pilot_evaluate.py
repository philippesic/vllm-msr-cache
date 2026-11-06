import numpy as np

from benchmarks.research_checks.hybrid_pilot_evaluate import cheap_design, fit_logistic, paired_interval, probability, retained_value, retention_interval


def test_ridge_logistic_learns_signal_without_perfect_separation_instability():
    x = np.column_stack([np.ones(20), np.arange(20) - 9.5])
    y = np.array([0.] * 10 + [1.] * 10)
    weights = fit_logistic(x, y, 1.)
    predicted = probability(x, weights)
    assert np.isfinite(weights).all()
    assert predicted[0] < .1 and predicted[-1] > .9


def test_paired_interval_weights_tasks_not_number_of_model_runs():
    result = paired_interval([1., 1., -1.], ["same-task", "same-task", "other-task"])
    assert result["tasks"] == 2
    assert result["mean"] == 0.
    assert result["paired_95_interval"] == [-1., 1.]


def test_hypothetical_retention_uses_equal_capacity_and_never_overfills():
    result = retained_value([.9, .8, .1], [1, 0, 1], [10, 10, 20])
    assert result == dict(useful_tokens=10., retained_tokens=10., budget_tokens=10.)


def test_retention_uncertainty_is_paired_and_zero_for_identical_predictions():
    result = retention_interval([.9, .1], [.9, .1], [1., 0.], [10, 20], ["a", "b"])
    assert result["paired_95_interval"] == [0., 0.]


def test_adjusted_interval_is_never_labelled_as_95_percent():
    result = paired_interval([1., -1.], ["a", "b"], alpha=.01)
    assert result["confidence"] == .99
    assert result["paired_interval"] == [-1., 1.]
    assert "paired_95_interval" not in result


def test_single_task_cannot_supply_an_uncertainty_interval():
    result = paired_interval([1.], ["a"], alpha=.01)
    assert result["mean"] is None and result["paired_interval"] is None
    assert result["confidence"] == .99
    assert "paired_95_interval" not in result


def test_metadata_interactions_do_not_learn_heldout_categories():
    numeric = np.array([[1., 2.], [2., 3.]])
    train = np.array([True, False])
    original = cheap_design(numeric, [("a",), ("unseen",)], train, True)
    changed = cheap_design(numeric, [("a",), ("different",)], train, True)
    assert original.shape == changed.shape
    np.testing.assert_array_equal(original, changed)
    assert original.shape[1] > cheap_design(numeric, [("a",), ("unseen",)], train).shape[1]
