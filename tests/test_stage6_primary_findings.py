import copy
import json

import pytest

from benchmarks.research_checks.stage6_primary_findings import build, markdown, summarize
from benchmarks.serving_suite.campaign_report import paired_estimate


def completed_report():
    ratios = [1.8, 2.0, 2.2, 1.9, 2.1, 2.05]

    def arm(rate, duration, ttft, native):
        return dict(
            completed_units=2, completed_units_per_s=rate, duration_s=duration,
            startup_s=20, attempt='retained-attempt',
            measurement_validity=dict(classification='original_measurement', **native),
            request_work=dict(completed=8, failed=0, prompt_tokens=100, output_tokens=30,
                              finish_reasons=dict(stop=7, length=1), end_user_ttft_mean_s=ttft,
                              end_user_ttft_p95_s=ttft * 2, e2e_mean_s=ttft * 3, e2e_p95_s=ttft * 4),
        )

    points = [dict(candidate=arm(r, 2 / r, .05, {}),
                   baseline=arm(1, 2, .10 + i * .01,
                                dict(cpu_pressure_evidence='observed', cpu_restore_activity='active')))
              for i, r in enumerate(ratios)]
    return dict(complete=True, throughput_acquisition_complete=True, missing_blocks=[],
                campaign_id='fixture', comparisons=[dict(
                    model='model', cell='B06', candidate='continuum', baseline='wait',
                    process_pairs=6, estimate=paired_estimate(ratios), points=points,
                    status='benefit', traffic_classes=dict(unhinted=dict(original_orientation=True)))])


def test_reciprocal_comparison_preserves_ci_order_and_whole_process_unit():
    report = completed_report()
    untouched = copy.deepcopy(report)
    row = summarize(report)['comparisons'][0]
    original = report['comparisons'][0]['estimate']
    assert report == untouched
    assert (row['numerator'], row['denominator']) == ('wait', 'continuum')
    assert row['throughput_ratio'] == pytest.approx(1 / original['geometric_throughput_ratio'])
    assert row['paired_95_interval'] == pytest.approx([1 / original['paired_95_interval'][1],
                                                    1 / original['paired_95_interval'][0]])
    independently_inverted = paired_estimate([1 / r for r in original['per_run_ratios']])
    assert row['paired_95_interval'] == pytest.approx(independently_inverted['paired_95_interval'])
    assert row['process_pairs'] == 6  # Not 48 request observations.
    assert row['numerator_summary']['serving_seconds']['mean'] == 2
    assert row['numerator_summary']['cpu_pressure_evidence'] == {'observed': 6}
    assert row['denominator_summary']['cpu_pressure_evidence'] == {'unavailable': 6}
    assert row['frozen_orientation'] == dict(candidate='continuum', baseline='wait')
    assert row['frozen_classification'] == 'benefit'


def test_latency_statistics_are_swapped_and_labeled_without_pooling():
    data = summarize(completed_report())
    row = data['comparisons'][0]
    assert row['numerator_summary']['latency']['end_user_ttft_mean_s']['mean_of_process_statistics'] == pytest.approx(.125)
    assert row['numerator_summary']['latency']['end_user_ttft_p95_s']['mean_of_process_statistics'] == pytest.approx(.250)
    assert row['numerator_summary']['length_caps'] == 6
    text = markdown(data)
    assert 'mean of process p95s is not a pooled p95' in text
    assert '125.000 | 250.000' in text


@pytest.mark.parametrize('change', [
    dict(complete=False), dict(throughput_acquisition_complete=False),
    dict(missing_blocks=[dict(cell='B06', repeat=6)]),
])
def test_partial_report_cannot_generate_final_findings(change, tmp_path):
    report = completed_report()
    report.update(change)
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='completed primary report'):
        build(path, tmp_path / 'results')
    assert not (tmp_path / 'results').exists()


@pytest.mark.parametrize('defect', ['work', 'rate', 'interval', 'failure'])
def test_mismatched_or_changed_measurements_are_rejected(defect):
    report = completed_report()
    row = report['comparisons'][0]
    if defect == 'work':
        row['points'][0]['candidate']['completed_units'] = 3
    elif defect == 'rate':
        row['points'][0]['candidate']['completed_units_per_s'] = 4
    elif defect == 'interval':
        row['estimate']['paired_95_interval'][0] *= .9
    else:
        row['points'][0]['candidate']['request_work']['failed'] = 1
    with pytest.raises(ValueError):
        summarize(report)


def test_native_orientation_and_missing_latency_remain_explicit():
    report = completed_report()
    row = report['comparisons'][0]
    row.update(candidate='wait', baseline='arc')
    row['points'][0]['candidate']['request_work']['end_user_ttft_p95_s'] = None
    output = summarize(report)['comparisons'][0]
    assert not output['orientation_inverted']
    assert output['throughput_ratio'] == row['estimate']['geometric_throughput_ratio']
    assert output['numerator_summary']['latency']['end_user_ttft_p95_s']['available'] is False
    assert 'unavailable' in markdown(summarize(report))


def test_materialized_artifacts_and_existing_findings_are_preserved(tmp_path):
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(completed_report()))
    destination = tmp_path / 'results'
    data = build(path, destination)
    assert len(data['comparisons']) == 1
    html = (destination / 'comparisons.html').read_text()
    assert '__DATA__' not in html and 'Candidate' in html
    assert not any(p.is_symlink() for p in destination.iterdir())
    previous = (destination / 'comparisons.json').read_bytes()
    with pytest.raises(FileExistsError):
        build(path, destination)
    assert (destination / 'comparisons.json').read_bytes() == previous


def test_explicit_incomplete_presentation_keeps_missing_repeat_and_frozen_estimate(tmp_path):
    report = completed_report()
    report.update(complete=False, throughput_acquisition_complete=False,
                  missing_blocks=[dict(model='model', cell='B06', repeat=0)],
                  completion=dict(complete=False, throughput=False, quality=True))
    untouched = copy.deepcopy(report)
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    data = build(path, tmp_path / 'conditional', retain_incomplete=True)
    assert report == untouched
    assert not data['report_complete'] and not data['findings_complete']
    assert not data['throughput_acquisition_complete']
    assert data['completion'] == report['completion']
    assert data['missing_blocks'] == report['missing_blocks']
    row = data['comparisons'][0]
    assert row['process_pairs'] == 6
    assert row['process_pairs_plus_reported_missing_blocks'] == 7
    assert row['missing_repeats'] == [0]
    assert row['paired_95_interval'] == pytest.approx(
        paired_estimate([1 / r for r in report['comparisons'][0]['estimate']['per_run_ratios']])['paired_95_interval'])
    text = (tmp_path / 'conditional/tables.md').read_text()
    assert 'Campaign report is incomplete' in text
    assert 'model / B06 / repeat 0' in text and '| 6/7 |' in text
    html = (tmp_path / 'conditional/comparisons.html').read_text()
    assert 'Campaign report is incomplete' in html and 'id="incomplete"' in html


def test_missing_audit_does_not_become_complete_when_all_throughput_blocks_exist():
    report = completed_report()
    report.update(complete=False, completion=dict(complete=False, quality=False))
    data = summarize(report, retain_incomplete=True)
    assert data['throughput_acquisition_complete']
    assert not data['findings_complete'] and not data['completion']['quality']
    assert 'Campaign report is incomplete' in markdown(data)


@pytest.mark.parametrize('defect', ['rate', 'interval', 'failure'])
def test_incomplete_mode_still_rejects_invalid_measurements(defect):
    report = completed_report()
    report['complete'] = False
    row = report['comparisons'][0]
    if defect == 'rate':
        row['points'][0]['candidate']['completed_units_per_s'] = 4
    elif defect == 'interval':
        row['estimate']['paired_95_interval'][0] *= .9
    else:
        row['points'][0]['candidate']['request_work']['failed'] = 1
    with pytest.raises(ValueError):
        summarize(report, retain_incomplete=True)


@pytest.mark.parametrize('missing', [
    [dict(reason='protocol_not_frozen')],
    [dict(model='model', cell='B06', repeat=0)] * 2,
])
def test_incomplete_mode_requires_unique_frozen_missing_identities(missing):
    report = completed_report()
    report.update(complete=False, missing_blocks=missing)
    with pytest.raises(ValueError, match='Missing blocks|Duplicate missing'):
        summarize(report, retain_incomplete=True)
