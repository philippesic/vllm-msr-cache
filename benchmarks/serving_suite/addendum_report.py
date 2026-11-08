"""Bounded offline whole-block comparisons for the prespecified Stage 6 addendum."""
from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

from .campaign_contracts import Ledger, atomic_json, load, read_protocol
from .campaign_addendum import block_state, completed_attempt, pressure
from .report import percentile


def secondary_pressure(parent):
    """Retain every completed secondary cell's native pressure, including zeros."""
    from .campaign_validity import measurement_validity
    parent=Path(parent)
    selected={}
    for event in Ledger(parent).index().values():
        i=event['details'].get('identity',{})
        if (i.get('phase')!='confirm' or i.get('model')=='qwen3_8b_awq' or
            i.get('cell') not in ('B06','B08','B09','B11') or event['state'] not in ('verified','failed')):
            continue
        directory=parent/event['details']['path']
        if not measurement_validity(directory,event)['valid']:
            continue
        case=load(directory/'measurement/result.json')['cases'][0]
        key=(i['model'],i['cell'],i['repeat'],i['system'])
        if key not in selected or selected[key]['sequence']<event['sequence']:
            selected[key]=dict(model=i['model'],cell=i['cell'],repeat=i['repeat'],system=i['system'],
                sequence=event['sequence'],path=str(directory),native_cache=case['campaign'].get('native_cache'),
                pressure_observed=pressure(case),caps=case['campaign']['output_caps'])
    return dict(cases=list(selected.values()),
        scope='Actual completed secondary native receipts; a model/workload label never substitutes for activity. Whole primary completion and integrity are checked separately.')


def estimate(ratios, required, *, margin=None, latency=False):
    n = len(ratios)
    if any(not math.isfinite(r) or r<=0 for r in ratios):
        raise ValueError('Expected positive finite paired ratios')
    logs = [math.log(r) for r in ratios]
    point = math.exp(statistics.mean(logs)) if logs else None
    interval = None
    if n == required:
        critical = {4:3.1824463053, 24:2.0686576104}[required]
        half = critical*statistics.stdev(logs)/math.sqrt(n)
        interval = [math.exp(statistics.mean(logs)-half),math.exp(statistics.mean(logs)+half)]
    met = (interval[1]<=margin if latency else interval[0]>=margin) if interval and margin else None
    return dict(process_pairs=n, required_process_pairs=required, geometric_ratio=point,
        paired_95_interval=interval, declared_bound=margin, bound_met=met,
        status='fixed_set_complete' if n==required else 'partial_descriptive_only',
        metric='latency numerator/denominator; smaller is better' if latency else 'throughput numerator/denominator; larger is better',
        method='Student t on paired whole-process log ratios, conditional on fixed cohort/host; no multiplicity adjustment',
        per_pair_ratios=ratios)


def class_metrics(rows):
    result = {}
    for group in sorted({r['hint_group'] for r in rows}):
        selected = [r for r in rows if r['hint_group']==group and r.get('success')]
        ttft = [r['ttft_s']+r.get('client_queue_s',0) for r in selected if r.get('ttft_s') is not None]
        # Both transport latencies start after acquiring the client semaphore.
        # Add client queue for the user's actual offer-to-completion latency.
        e2e = [r['latency_s']+r.get('client_queue_s',0) for r in selected]
        metrics = dict(requests=len(selected), prompt_tokens=sum(r['prompt_token_count'] for r in selected),
            output_tokens=sum(len(r['output_token_ids']) for r in selected))
        for name, values in (('end_user_ttft',ttft),('end_user_e2e',e2e),
                             ('client_queue',[r.get('client_queue_s',0) for r in selected])):
            metrics.update({name+'_mean_s':statistics.mean(values) if values else None,
                name+'_p95_s':percentile(values,.95),name+'_p99_s':percentile(values,.99)})
        metrics['tool_to_first_token_s'] = [(r['first_token_ns']-r['tool_completed_ns'])/1e9
            for r in selected if r.get('first_token_ns') and r.get('tool_completed_ns')]
        result[group] = metrics
    return result


def report(out):
    out = Path(out).resolve()
    protocol = read_protocol(out)
    events = Ledger(out).index()
    blocks, cases, missing = {}, [], []
    for block in protocol['schedule']:
        done, _ = block_state(out, block, events)
        if not done:
            missing.append(block['id']); continue
        data = {}
        first_inputs = []
        for event in done:
            i = event['details']['identity']
            case = completed_attempt(out,event)
            directory = out/event['details']['path']/'measurement'/case['name']
            rows = [json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
            first_inputs.append({r['source_id']:r['payload']['prompt'] for r in rows if not r['is_returning']})
            label = i['system'] if block['section']!='advice' else i['cell']
            row = dict(block=block['id'],section=block['section'],arm=label,path=str(directory),
                seconds=case['load_duration_s'],startup_s=case['startup_s'],process_s=case['wall_s'],
                units=case['campaign']['completed_units'],requests=case['summary']['completed'],
                tools=case['campaign']['tools'],prompt_tokens=case['summary']['prompt_tokens'],
                output_tokens=case['summary']['output_tokens'],caps=case['campaign']['output_caps'],
                native_cache=case['campaign'].get('native_cache'),
                pressure=pressure(case) if i['system'] in ('arc','wait') else None,
                external_policy_activation=case['campaign'].get('policy_activation'),
                cpu_recency=case.get('activity',{}).get('cpu_recency'),
                traffic_classes=class_metrics(rows),
                class_workflows=case['campaign']['traffic_classes'])
            experiments = [h for h in case['hints'] if h.get('prediction_experiment')]
            row['false_positive_advice'] = dict(events=sum(bool(h['prediction_experiment']['false_positive']) for h in experiments),
                ready_key_touches=sum(h.get('response',{}).get('ready_cpu_prefix_keys',0)
                    for h in experiments if h['prediction_experiment']['false_positive']),
                scope='Repeated acknowledged key touches, not unique retained/consumed keys')
            data[label] = row; cases.append(row)
        if any(inputs!=first_inputs[0] for inputs in first_inputs):
            raise ValueError('Matched block first-request token inputs differ: '+block['id'])
        blocks[block['id']] = data
    comparisons = {}

    def compare(name, ids, numerator, denominator, count, *, latency=None, margin=None):
        pairs = [blocks[b] for b in ids if b in blocks]
        ratios = []
        for p in pairs:
            if latency:
                a = p[numerator]['traffic_classes']['unhinted'][latency]
                b = p[denominator]['traffic_classes']['unhinted'][latency]
                ratios.append(a/b)
            else:
                a, b = p[numerator],p[denominator]
                ratios.append((a['units']/a['seconds'])/(b['units']/b['seconds']))
        comparisons[name] = estimate(ratios,count,latency=bool(latency),margin=margin)
        if ids and ids[0].startswith(('repository-','advice-')):
            baseline='arc' if ids[0].startswith('repository-') else 'H00'
            comparisons[name]['baseline_pressure_in_all_fixed_blocks'] = len(pairs)==count and all(p[baseline]['pressure'] for p in pairs)
        if ids and ids[0].startswith('advice-'):
            comparisons[name]['no_return_cpu_recency_in_all_fixed_blocks'] = len(pairs)==count and all(
                p['H02']['false_positive_advice']['ready_key_touches']>0 for p in pairs)

    for a,b in itertools.combinations(('arc','wait','continuum','upstream'),2):
        compare('repository:'+b+'_vs_'+a,[f'repository-{r}' for r in range(4)],b,a,4)
    compare('overhead:O2_vs_O0',[f'overhead-{r}' for r in range(24)],'O2','O0',24,margin=.97)
    for a,b in (('H00','H01'),('H00','H02'),('H01','H02')):
        ids = [f'advice-{r}' for r in range(4)]
        compare('advice:'+b+'_vs_'+a,ids,b,a,4)
        for metric in ('end_user_ttft_mean_s','end_user_e2e_mean_s'):
            compare('unhinted:'+b+'_vs_'+a+':'+metric,ids,b,a,4,latency=metric,margin=1.05)
    elapsed = {}
    for section,arm in sorted({(c['section'],c['arm']) for c in cases if c['block'] not in ('repository-gate','repository-roomy')}):
        values = [c['seconds'] for c in cases if c['section']==section and c['arm']==arm and c['block'] not in ('repository-gate','repository-roomy')]
        elapsed[section+':'+arm] = dict(processes=len(values),mean_s=statistics.mean(values),median_s=statistics.median(values))
    result = dict(schema='stage6-addendum-comparison-v1',protocol=str(out/'protocol.lock.json'),
        missing_blocks=missing,complete_matched_blocks=len(blocks),comparisons=comparisons,
        elapsed_time_s=elapsed,cases=cases,
        secondary_confirmation_pressure=secondary_pressure(protocol['parent']),
        purpose_and_failure_state=load(out/'execution.json') if (out/'execution.json').exists() else None,
        quality='UNQUALIFIED until source-based final/intermediate capped-output and blinded review is complete',
        limitations=['Fixed selected cohorts and one host; fresh processes are not independent task samples',
            'External runtime/output differences; unavailable native counters remain null',
            'Per-process p95/p99 descriptive only; 16 unhinted requests/process is not tail-SLA evidence',
            'Completed units include initial-only abandoned sessions in advice traffic'])
    atomic_json(out/'analysis/addendum-comparison.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=report(args.out)
    print(json.dumps(dict(complete_matched_blocks=result['complete_matched_blocks'],
        missing_blocks=result['missing_blocks'],comparisons=result['comparisons']),indent=2))
