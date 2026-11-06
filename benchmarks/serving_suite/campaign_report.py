"""Whole-process paired estimates; incomplete and adverse cases remain visible."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path

from .campaign_contracts import Ledger, atomic_json, load, local_path, read_protocol, verify_tree

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


def completed_blocks(out, phase, *, source=None):
    """Select whole matched blocks; preserve attempts outside the analysis set."""
    out = Path(out)
    resolved = load(out/'resolved.json')
    required = {(c['model'],c['cell']):set(c['systems']) for c in resolved['cells']}
    groups = {}
    for event in Ledger(out).index().values():
        details = event['details']
        identity = details.get('identity',{})
        if event['state'] != 'verified' or identity.get('phase') != phase:
            continue
        if source is not None and identity.get('source')!=source:
            continue
        attempt = local_path(out,details['path'])
        verify_tree(attempt,details['seal'])
        case = load(attempt/'measurement/result.json')['cases'][0]
        if case['evaluation']['status'] != 'passed' or case['summary']['failed']:
            continue
        observed = case.get('campaign')
        if not observed or observed['completed_units'] != observed['offered_units']:
            continue
        key = (identity['model'],identity['cell'],identity['repeat'],details['block_id'])
        request_counts = {}
        request_path = attempt/'measurement'/case.get('name','')/'requests.jsonl'
        if request_path.is_file():
            for line in request_path.read_text().splitlines():
                row = json.loads(line)
                if row.get('success') and not row.get('cancel_probe'):
                    count = request_counts.setdefault(row.get('hint_group','ordinary'),dict(requests=0,returning=0))
                    count['requests'] += 1
                    count['returning'] += bool(row.get('is_returning'))
        groups.setdefault(key,{})[identity['system']] = dict(event=event, case=case,
            rate=observed['completed_units_per_s'], units=observed['completed_units'],
            duration_s=case['load_duration_s'],request_observations=request_counts)
    selected = {}
    for key, block in groups.items():
        if set(block) != required.get(key[:2]):
            continue
        identities = [r['event']['details']['identity'] for r in block.values()]
        if len({i['source'] for i in identities}) != 1 or len({i['workload'] for i in identities}) != 1:
            raise ValueError('Comparison block mixes source/workload identities')
        if len({r['units'] for r in block.values()}) != 1:
            raise ValueError('Matched completed workload counts differ')
        logical = key[:3]
        sequence = max(r['event']['sequence'] for r in block.values())
        if logical not in selected or selected[logical]['sequence'] < sequence:
            selected[logical] = dict(block_id=key[3], sequence=sequence, systems=block)
    return selected


def pilot_precision(out):
    out = Path(out)
    resolved = load(out/'resolved.json')
    blocks = completed_blocks(out,'pilot')
    from .campaign import source_snapshot
    from .campaign_gates import current_serving,serving_files
    expected_source = serving_files(source_snapshot())
    blocks = {key:block for key,block in blocks.items() if all(current_serving(
        local_path(out,data['event']['details']['path']),expected_source) for data in block['systems'].values())}
    measured = {}
    for cell in resolved['cells']:
        key = cell['model']+':'+cell['cell']
        runs = [blocks.get((cell['model'],cell['cell'],repeat)) for repeat in range(3)]
        if not all(runs):
            continue
        checks = {}
        for baseline in ('arc','wait'):
            for system in cell['systems']:
                if system == baseline or baseline == 'wait' and system == 'arc':
                    continue
                ratios = [run['systems'][system]['rate']/run['systems'][baseline]['rate'] for run in runs]
                checks[system+'_vs_'+baseline] = plan_repeats(ratios,four_arm=len(cell['systems'])==4)
        durations = [r['duration_s'] for run in runs for r in run['systems'].values()]
        measured[key] = dict(comparisons=checks, repeats=max(v['repeats'] for v in checks.values()),
            mean_case_duration_s=statistics.mean(durations),
            block_ids=[run['block_id'] for run in runs])
    expected = ['qwen3_8b_awq:'+c for c in ('B02','B06','B09','B11','B12')]
    expected += [m+':B06' for m in ('ministral3_3b','qwen3_4b_2507')]
    missing = sorted(set(expected)-measured.keys())
    repeats, costs, propagation = {}, {}, {}
    for cell in resolved['cells']:
        key = cell['model']+':'+cell['cell']
        references = [key] if key in measured else [k for k in measured if k.startswith(cell['model']+':')]
        if not references:
            continue
        selected = max(measured[k]['repeats'] for k in references)
        if len(cell['systems'])==4:
            selected = 8 if selected<=8 else 12
        repeats[key] = selected
        # Cost is deliberately an estimate: unpiloted cells can have a different
        # context/output cost, so expose the conservative borrowed duration.
        duration = max(measured[k]['mean_case_duration_s'] for k in references)
        costs[key] = selected*len(cell['systems'])*(duration+60)
        propagation[key] = dict(pilots=references,rule='Maximum measured repetition requirement within model when not directly piloted',
                                mean_case_duration_s=duration,startup_allowance_s=60)
    result = dict(schema='serving-campaign-pilot-precision-v1',status='incomplete' if missing else 'complete',
        missing=missing,measured=measured,repeats=repeats,propagation=propagation,
        estimated_confirmation_seconds=sum(costs.values()),per_cell_seconds=costs,
        scope='Fixed bounded repetition choice before confirmation; no optional stopping for wins')
    atomic_json(out/'pilot-precision.json',result)
    return result


def latency_estimate(ratios, *, tail=False):
    result = paired_estimate(ratios)
    interval = result['paired_95_interval']
    method = 'Student t on paired whole-process log latency-statistic ratios; lower is better'
    if tail and len(ratios)>1:
        # Resample process pairs, never requests within a process. This estimates
        # variation in the geometric ratio of process p95s, not a pooled p95.
        logs = [math.log(r) for r in ratios]
        rng = random.Random(17011)
        samples = sorted(math.exp(sum(rng.choice(logs) for _ in logs)/len(logs)) for _ in range(10000))
        interval = [samples[249],samples[9749]]
        method = 'Paired whole-process percentile bootstrap of log process-p95 ratios; 10000 resamples, seed 17011; lower is better'
    return dict(process_pairs=result['process_pairs'], geometric_latency_ratio=result['geometric_throughput_ratio'],
                paired_95_interval=interval,per_run_ratios=ratios,method=method)


def compare_classes(runs, baseline, system):
    result = {}
    labels = set.intersection(*[set(run['systems'][system]['case']['campaign']['traffic_classes']) for run in runs]) if runs else set()
    for label in sorted(labels):
        pairs = [(run['systems'][baseline]['case']['campaign']['traffic_classes'][label],
                  run['systems'][system]['case']['campaign']['traffic_classes'][label]) for run in runs]
        result[label] = dict(throughput=paired_estimate([b['completed_units_per_s']/a['completed_units_per_s'] for a,b in pairs]),
                            offered_units_per_process=[a['offered_units'] for a,b in pairs])
        counts = {role:[run['systems'][name].get('request_observations',{}).get(label,{}) for run in runs]
                  for role,name in (('baseline',baseline),('candidate',system))}
        result[label]['observations'] = {role:dict(requests=sum(c.get('requests',0) for c in values),
            returning=sum(c.get('returning',0) for c in values)) for role,values in counts.items()}
        result[label]['tail_scope'] = ('descriptive: fewer than 200 returning observations in at least one arm; no tail guarantee'
            if any(v['returning']<200 for v in result[label]['observations'].values()) else
            'Process-p95 comparison conditional on this fixed cohort; no population tail guarantee')
        for statistic in ('end_user_ttft_mean_s','end_user_ttft_p95_s','e2e_mean_s','e2e_p95_s'):
            values = [(a['requests'][statistic],b['requests'][statistic]) for a,b in pairs]
            if all(a and b for a,b in values):
                result[label][statistic] = latency_estimate([b/a for a,b in values],tail='p95' in statistic)
        if label == 'unhinted':
            interval = result[label].get('end_user_ttft_mean_s',{}).get('paired_95_interval')
            result[label]['five_percent_mean_latency_bound'] = ('bounded' if interval and interval[1]<=1.05 else
                'regression' if interval and interval[0]>1.05 else 'inconclusive')
            tail = [b['requests']['end_user_ttft_p95_s']/a['requests']['end_user_ttft_p95_s']
                    for a,b in pairs if a['requests']['end_user_ttft_p95_s'] and b['requests']['end_user_ttft_p95_s']]
            result[label]['ten_percent_p95_diagnostic_alarm'] = any(r>1.10 for r in tail)
    return result


def report(out):
    out = Path(out)
    resolved = load(out/'resolved.json')
    locked = read_protocol(out) if (out/'protocol.lock.json').exists() else None
    blocks = completed_blocks(out,'confirm')
    records, comparisons, missing = [], [], []
    for event in Ledger(out).index().values():
        details = event['details']
        identity = details.get('identity',{})
        row = dict(attempt_id=event['attempt_id'],state=event['state'],**identity,artifacts=details.get('path'))
        records.append(row)
        if details.get('seal'):
            verify_tree(local_path(out,details['path']),details['seal'])
    if locked:
        for block in locked['schedule']:
            key = (block['model'],block['cell'],block['repeat'])
            selected = blocks.get(key)
            if not selected or any(r['event']['details']['identity']['source']!=locked['source']['digest']
                                   for r in selected['systems'].values()):
                missing.append(dict(model=key[0],cell=key[1],repeat=key[2]))
                blocks.pop(key,None)
    else:
        missing.append(dict(reason='protocol_not_frozen'))
    for cell in resolved['cells']:
        runs = [block for key,block in sorted(blocks.items()) if key[:2]==(cell['model'],cell['cell'])]
        for baseline in ('arc','wait'):
            for system in cell['systems']:
                if system==baseline or baseline=='wait' and system=='arc':
                    continue
                record = dict(model=cell['model'],cell=cell['cell'],candidate=system,baseline=baseline,
                              status='unmeasured',process_pairs=len(runs))
                if runs:
                    estimate = paired_estimate([r['systems'][system]['rate']/r['systems'][baseline]['rate'] for r in runs])
                    points = []
                    for run in runs:
                        point = dict(block=run['block_id'])
                        for role,name in (('baseline',baseline),('candidate',system)):
                            data = run['systems'][name]
                            case = data['case']
                            point[role] = dict(attempt=data['event']['attempt_id'],completed_units=data['units'],
                                completed_units_per_s=data['rate'],duration_s=data['duration_s'],startup_s=case['startup_s'],
                                request_work=case['summary'],metrics=case['metrics'],telemetry=case['telemetry'],
                                geometry=case.get('effective_geometry'),phases=case['campaign']['phases'])
                        points.append(point)
                    record.update(estimate=estimate,status=classify(estimate),points=points,
                                  traffic_classes=compare_classes(runs,baseline,system),
                                  work_scope='All matched offered units complete; live outputs can differ; quality remains separate')
                comparisons.append(record)
    quality_path = out/'analysis/quality/summary.json'
    quality = load(quality_path) if quality_path.exists() else dict(status='pending')
    acquisition_complete = not missing
    from .campaign_completion import readiness
    completion = readiness(out,acquisition_complete,quality) if locked else dict(complete=False)
    result = dict(schema='serving-campaign-report-v2',campaign_id=resolved['campaign_id'],
        complete=completion['complete'],completion=completion,throughput_acquisition_complete=acquisition_complete,missing_blocks=missing,
        comparisons=comparisons,all_attempts=records,quality=quality,
        limitation='Readiness also requires verified controls, operational, recovery and portability evidence')
    atomic_json(out/'analysis/current/results.json',result)
    with (out/'analysis/current/results.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['model','cell','candidate','baseline','status','process_pairs','ratio','lower','upper'])
        writer.writeheader()
        for row in comparisons:
            estimate = row.get('estimate',{})
            bounds = estimate.get('paired_95_interval') or [None,None]
            writer.writerow({**{k:row[k] for k in ('model','cell','candidate','baseline','status','process_pairs')},
                'ratio':estimate.get('geometric_throughput_ratio'),'lower':bounds[0],'upper':bounds[1]})
    return result
