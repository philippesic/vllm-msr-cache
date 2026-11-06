"""Descriptive workflow-group and finite-exposure memory reports.

No request-level significance test and no long-run failure-rate certification.
Failed or unfinished offered sessions remain in group coverage.
"""
from __future__ import annotations
import argparse
import json
import itertools
import math
import statistics
from pathlib import Path
from benchmarks.serving_suite.contracts import write_json
from benchmarks.serving_suite.report import percentile, paired_interval


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def report(directory):
    directory=Path(directory);result=json.loads((directory/'result.json').read_text());cases=[]
    assignments={s['id']:('hinted' if s.get('publish_hints',True) else 'unhinted')
                 for s in json.loads((directory/'corpus.json').read_text())['sessions']}
    for case in result['cases']:
        root=directory/case['name'];groups={};windows=[]
        resources=lines(root/'resources.jsonl')
        for workflow in sorted(root.glob('workflow-*.jsonl')):
            records=lines(workflow);starts={r['session_id']:r for r in records if r['event']=='session_start'}
            ends={r['session_id']:r for r in records if r['event']=='session_end'}
            for sid,start in starts.items():
                group=assignments[start['source_id']]
                entry=groups.setdefault(group,dict(offered=0,completed=0,unfinished=0,workflow_seconds=[]))
                entry['offered']+=1
                if sid in ends and ends[sid].get('success'):
                    entry['completed']+=1;entry['workflow_seconds'].append((ends[sid]['ns']-start['ns'])/1e9)
                else:entry['unfinished']+=1
            if starts and ends:
                lo=min(r['ns'] for r in starts.values())/1e9;hi=max(r['ns'] for r in ends.values())/1e9
                tail=[r['rss_bytes'] for r in resources if max(lo,hi-5)<=r['monotonic_s']<=hi]
                windows.append(dict(wave=next(iter(starts.values()))['wave'],start_s=lo,end_s=hi,
                    completed_sessions=len(ends),final_five_seconds_rss_median=statistics.median(tail) if tail else None))
        for group in groups.values():
            values=group.pop('workflow_seconds')
            group.update(mean_workflow_s=statistics.mean(values) if values else None,
                         p95_workflow_s=percentile(values,.95))
        rss=[w['final_five_seconds_rss_median'] for w in windows if w['final_five_seconds_rss_median'] is not None]
        cases.append(dict(name=case['name'],condition=case['variant'],repeat=case['repeat'],
            case_status=case['evaluation']['status'],error=case.get('error'),groups=groups,waves=windows,
            end_wave_rss_growth_bytes=rss[-1]-rss[0] if len(rss)>1 else None,
            growth_limit_bytes=536870912 if len(rss)>1 else None,
            within_predeclared_growth_limit=(rss[-1]-rss[0]<=536870912) if len(rss)>1 else None,
            exposure_s=case['load_duration_s'],completed_requests=case['summary']['completed'],
            failed_requests=case['summary']['failed'],
            operational=case.get('operational'),cleanup=case['cleanup']))
    comparisons={}
    by={(c['condition'],c['repeat']):c for c in cases}
    for baseline,candidate in itertools.combinations(result['plan']['qualification_conditions'],2):
        pairs=[]
        for repeat in sorted({c['repeat'] for c in cases}):
            left=by.get((baseline,repeat));right=by.get((candidate,repeat))
            if not left or not right:continue
            for name,group in left['groups'].items():
                other=right['groups'].get(name)
                eligible=other and group['offered']==other['offered'] and not group['unfinished'] and not other['unfinished'] and left['case_status']==right['case_status']=='passed'
                pairs.append(dict(repeat=repeat,group=name,eligible=bool(eligible),
                    candidate_workflow_time_ratio=other['mean_workflow_s']/group['mean_workflow_s'] if eligible else None))
        groups={}
        for name in sorted({p['group'] for p in pairs}):
            values=[p['candidate_workflow_time_ratio'] for p in pairs if p['group']==name and p['eligible']]
            ratio=math.exp(statistics.mean(map(math.log,values))) if values else None
            groups[name]=dict(paired_runs=len(values),candidate_mean_workflow_time_ratio=ratio,
                conditional_95ci=paired_interval(values,logarithmic=True),
                observed_regression_exceeds_five_pct=ratio>1.05 if ratio else None)
        comparisons[candidate+'_vs_'+baseline]=dict(pairs=pairs,groups=groups)
    return dict(schema='wait-hint-finite-stability-v1',cases=cases,comparisons=comparisons,
        scope='Descriptive full-workflow group coverage and finite repeated-task session churn. Summed process RSS includes shared mappings and is not unique allocated memory. No independent-arrival saturation or long-run failure-rate claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();write_json(args.output,report(args.directory))

if __name__=='__main__':main()
