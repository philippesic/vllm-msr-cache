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


def test_mixed_mean_bound_is_separate_from_tail_alarm():
    from benchmarks.serving_suite.campaign_report import compare_classes
    def arm(mean,tail):
        return dict(case=dict(campaign=dict(traffic_classes=dict(unhinted=dict(
            completed_units_per_s=1,offered_units=16,requests=dict(end_user_ttft_mean_s=mean,
                end_user_ttft_p95_s=tail,e2e_mean_s=mean,e2e_p95_s=tail))))),
            request_observations=dict(unhinted=dict(requests=16,returning=0)))
    runs = [dict(systems=dict(arc=arm(1,1),wait=arm(1.01,1.2))) for _ in range(6)]
    result = compare_classes(runs,'arc','wait')['unhinted']
    assert result['five_percent_mean_latency_bound']=='bounded'
    assert result['ten_percent_p95_diagnostic_alarm']
    assert result['observations']['candidate']==dict(requests=96,returning=0)
    assert 'descriptive' in result['tail_scope']
    assert 'bootstrap' in result['end_user_ttft_p95_s']['method']
    assert result['end_user_ttft_p95_s']['paired_95_interval']==pytest.approx([1.2,1.2])


def test_analysis_uses_only_complete_blocks_and_can_retain_resume_source(tmp_path):
    from benchmarks.serving_suite.campaign_contracts import Ledger,atomic_json,seal_tree
    from benchmarks.serving_suite.campaign_report import completed_blocks
    atomic_json(tmp_path/'resolved.json',dict(cells=[dict(model='m',cell='B06',systems=['arc','wait'])]))
    ledger = Ledger(tmp_path)
    def finish(system,block,source,rate):
        identity = dict(phase='screen',model='m',cell='B06',repeat=0,system=system,source=source,workload='same')
        attempt,directory = ledger.begin(identity,block_id=block)
        row = ledger.index()[attempt]
        case = dict(evaluation=dict(status='passed'),summary=dict(failed=0),load_duration_s=10,
            campaign=dict(completed_units=16,offered_units=16,completed_units_per_s=rate))
        atomic_json(directory/'measurement/result.json',dict(cases=[case]))
        seal = seal_tree(directory)
        ledger.event(row['case_id'],attempt,'verified',**row['details'],seal=seal)
    finish('arc','partial-one','old',1)
    finish('wait','partial-two','old',2)
    assert completed_blocks(tmp_path,'screen')=={}
    finish('wait','partial-one','old',1.1)
    assert completed_blocks(tmp_path,'screen')[('m','B06',0)]['block_id']=='partial-one'
    finish('arc','new-whole','new',1.2)
    # A newer partial block cannot replace an existing complete comparison.
    assert completed_blocks(tmp_path,'screen')[('m','B06',0)]['block_id']=='partial-one'
    finish('wait','new-whole','new',1.3)
    assert completed_blocks(tmp_path,'screen')[('m','B06',0)]['block_id']=='new-whole'
    assert completed_blocks(tmp_path,'screen',source='old')[('m','B06',0)]['block_id']=='partial-one'
