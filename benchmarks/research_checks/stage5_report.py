"""Compact per-case Stage 5 inventory, including failed and inactive candidates."""
import argparse
import json
from pathlib import Path
from benchmarks.serving_suite.contracts import write_json
from benchmarks.serving_suite.report import paired_interval
import math
import statistics


def cohort_completion(directory, corpus, plan):
    """Full-session aggregates; designated cohorts are also used for ARC."""
    paths = sorted(Path(directory).glob('workflow-*.jsonl'))
    if not paths:
        return None
    sources = corpus['sessions'][:plan['sessions']]
    groups = {s['id']: ('hinted' if s['publish_hints'] else 'unhinted')
              if 'publish_hints' in s else 'all' for s in sources}
    starts = {}; ends = {}
    for path in paths:
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['event'] not in ('session_start', 'session_end'):
                continue
            target = starts if e['event'] == 'session_start' else ends
            key = e['session_id']
            if key in target:
                raise ValueError('Duplicate session boundary: ' + key)
            target[key] = e
    expected = {f'w{wave}:{s["id"]}' for wave in range(plan['waves']) for s in sources}
    if set(starts) != expected or set(ends) != expected:
        raise ValueError('Incomplete cohort session boundaries')
    values = {}
    for key in sorted(expected):
        a, b = starts[key], ends[key]
        if a['source_id'] != b['source_id'] or b['ns'] < a['ns']:
            raise ValueError('Invalid cohort session boundaries')
        values.setdefault(groups[a['source_id']], []).append((b['ns']-a['ns'])/1e9)
    return {group:dict(count=len(v), mean_s=statistics.mean(v)) for group,v in values.items()}


def report(root):
    cases=[];comparisons=[]
    for path in sorted(Path(root).rglob('result.json')):
        d=json.loads(path.read_text())
        if 'qualification_conditions' not in d.get('plan',{}):continue
        corpus = json.loads((path.parent/'corpus.json').read_text())
        by={}
        for c in d.get('cases',[]):
            a=c.get('agent',{});cache=a.get('cache',{});hints=a.get('hints',{})
            row=dict(run=str(path.parent),name=c['name'],variant=c['variant'],repeat=c['repeat'],
                status=c['evaluation']['status'],error=c.get('error'),seconds=c['load_duration_s'],
                completed=c['summary']['completed'],expected=c['expected_requests'],
                output_tokens=c['summary']['output_tokens'],prefill=cache.get('locally_executed_prefill_tokens'),
                loaded=cache.get('loaded_tokens'),preemptions=cache.get('preemptions'),
                transfers=cache.get('physical_demand_load_bytes'),hint_calls=hints.get('publications'),
                hint_rtt=hints.get('rtt_s'),refreshed=hints.get('ready_cpu_keys_refreshed'),
                competition=a.get('opportunity',{}).get('cache_competition_observed'),
                return_ttft=a.get('latency',{}).get('tool_to_first_token_s'),
                cpu_s=c.get('telemetry',{}).get('process_cpu_elapsed_s'),
                clean=c.get('cleanup',{}).get('verified'))
            if row['status'] == 'passed':
                row['cohort_completion'] = cohort_completion(path.parent/c['name'],corpus,d['plan'])
            cases.append(row);by[c['variant'],c['repeat']]=row
        for baseline in ('arc','wait','linear-arc'):
            for candidate in d['plan']['qualification_conditions']:
                if baseline==candidate:continue
                ratios=[]; cohort_ratios={}
                for repeat in sorted({r for _,r in by}):
                    a,b=by.get((baseline,repeat)),by.get((candidate,repeat))
                    if a and b and a['completed']==a['expected'] and b['completed']==b['expected'] and a['status']==b['status']=='passed':
                        ratios.append(a['seconds']/b['seconds'])
                        for group, v in (a.get('cohort_completion') or {}).items():
                            other = (b.get('cohort_completion') or {}).get(group)
                            if other and other['count'] == v['count']:
                                cohort_ratios.setdefault(group,[]).append(other['mean_s']/v['mean_s'])
                if ratios:
                    comparisons.append(dict(run=str(path.parent),baseline=baseline,candidate=candidate,
                        pairs=len(ratios),workflow_ratio=math.exp(statistics.mean(map(math.log,ratios))),
                        ci95=paired_interval(ratios,logarithmic=True),ratios=ratios,
                        cohort_completion={g:dict(pairs=len(v),candidate_over_baseline=math.exp(statistics.mean(map(math.log,v))),
                            ci95=paired_interval(v,logarithmic=True)) for g,v in cohort_ratios.items()}))
    return dict(scope='Development comparisons, no selection-adjusted or fresh confirmation claim',cases=cases,comparisons=comparisons)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_json(a.output,report(a.root))
