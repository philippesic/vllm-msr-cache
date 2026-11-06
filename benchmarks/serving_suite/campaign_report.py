"""Whole-process paired estimates; incomplete and adverse cases remain visible."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from .campaign_contracts import Ledger, atomic_json, load, local_path, verify_tree

# Two-sided 95% Student t critical values for the locked choices and pilot df.
T95 = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053,
       4: 2.7764451052, 5: 2.5705818356, 6: 2.4469118511,
       7: 2.3646242516, 8: 2.3060041352, 9: 2.2621571629,
       10: 2.2281388520, 11: 2.2009851601}


def paired_estimate(ratios):
    if not ratios or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError('Ratios must be positive finite whole-run measurements')
    logs = [math.log(r) for r in ratios]
    mean = statistics.mean(logs)
    n = len(logs)
    interval = None
    if n > 1:
        if n - 1 not in T95:
            raise ValueError('Unplanned repetition count; declare an interval method')
        half = T95[n - 1] * statistics.stdev(logs) / math.sqrt(n)
        interval = [math.exp(mean - half), math.exp(mean + half)]
    estimate = math.exp(mean)
    return dict(process_pairs=n, geometric_throughput_ratio=estimate,
                throughput_gain_pct=(estimate - 1) * 100,
                time_reduction_pct=(1 - 1 / estimate) * 100,
                paired_95_interval=interval,
                method='Student t on paired whole-process log throughput ratios',
                scope='Conditional on frozen cohort/host; not multiplicity adjusted',
                per_run_ratios=ratios)


def classify(estimate, *, benefit=0.02, no_regression=0.97):
    interval = estimate['paired_95_interval']
    if interval is None:
        return 'inconclusive'
    lower, upper = interval
    if estimate['geometric_throughput_ratio'] >= 1 + benefit and lower > 1:
        return 'benefit'
    if upper < no_regression:
        return 'regression'
    if lower >= no_regression:
        return 'bounded regression/neutral'
    return 'inconclusive'


def plan_repeats(pilot_ratios, *, four_arm=False, target=0.02):
    if len(pilot_ratios) != 3:
        raise ValueError('Precision planning requires three independent pilot blocks')
    if any(not math.isfinite(r) or r <= 0 for r in pilot_ratios):
        raise ValueError('Invalid pilot ratio')
    sd = statistics.stdev([math.log(r) for r in pilot_ratios])
    choices = [8, 12] if four_arm else [6, 8, 10, 12]
    predicted = {str(n): T95[n - 1] * sd / math.sqrt(n) for n in choices}
    n = next((n for n in choices if predicted[str(n)] <= target), choices[-1])
    return dict(repeats=n, log_ratio_sd=sd, predicted_log_half_widths=predicted,
                target_log_half_width=target, target_expected_met=predicted[str(n)] <= target,
                limitation='Variance estimated from only three development process blocks')


def report(out):
    out = Path(out)
    resolved = load(out / 'resolved.json')
    records, groups = [], {}
    for event in Ledger(out).index().values():
        details = event['details']
        identity = details.get('identity', {})
        row = dict(attempt_id=event['attempt_id'], state=event['state'], **identity)
        row['artifacts'] = details.get('path')
        records.append(row)
        if details.get('seal'):
            verify_tree(local_path(out, details['path']), details['seal'])
        if event['state'] != 'verified' or identity.get('phase') != 'confirm':
            continue
        attempt = local_path(out, details['path'])
        measured = load(attempt / 'measurement/result.json')
        case = measured['cases'][0]
        rate = case['summary'].get('completed_requests_per_s')
        if not rate or case['summary'].get('failed'):
            continue
        key = (identity['model'], identity['cell'], details.get('block_id'))
        groups.setdefault(key, {})[identity['system']] = dict(rate=rate, event=event,
                                                             requests=case['summary']['completed'],
                                                             duration_s=case['load_duration_s'])
    comparisons = []
    for cell in resolved['cells']:
        for baseline in ('arc', 'wait'):
            for system in cell['systems']:
                if system == baseline or (baseline == 'wait' and system == 'arc'):
                    continue
                pairs, points = [], []
                for key, block in groups.items():
                    if key[:2] == (cell['model'], cell['cell']) and baseline in block and system in block:
                        if block[system]['requests'] != block[baseline]['requests']:
                            raise ValueError('Matched completed workload counts differ')
                        pairs.append(block[system]['rate'] / block[baseline]['rate'])
                        points.append(dict(block=key[2], baseline=block[baseline], candidate=block[system]))
                record = dict(model=cell['model'], cell=cell['cell'], candidate=system,
                              baseline=baseline, status='unmeasured', process_pairs=len(pairs))
                if pairs:
                    estimate = paired_estimate(pairs)
                    record.update(estimate=estimate, status=classify(estimate), points=points,
                                  work_scope='Matched offered units; live outputs can differ; quality reported separately')
                comparisons.append(record)
    result = dict(schema='serving-campaign-report-v1', campaign_id=resolved['campaign_id'],
                  complete=False, comparisons=comparisons, all_attempts=records,
                  limitation='Acquisition completeness, quality, correctness and operational gates are separate requirements')
    atomic_json(out / 'analysis/current/results.json', result)
    path = out / 'analysis/current/results.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'cell', 'candidate', 'baseline', 'status', 'process_pairs', 'ratio', 'lower', 'upper'])
        writer.writeheader()
        for row in comparisons:
            estimate = row.get('estimate', {})
            bounds = estimate.get('paired_95_interval') or [None, None]
            writer.writerow({**{k: row[k] for k in ('model', 'cell', 'candidate', 'baseline', 'status', 'process_pairs')},
                             'ratio': estimate.get('geometric_throughput_ratio'), 'lower': bounds[0], 'upper': bounds[1]})
    return result
