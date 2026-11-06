"""Compact independently recomputed live-factorial metrics and native receipts."""
import argparse
import collections
import json
from pathlib import Path
from benchmarks.serving_suite.report import summarize_requests
from benchmarks.serving_suite.contracts import write_json


def extract(root):
    rows=[]
    for path in sorted(root.glob('*/*/case.json')):
        case=json.loads(path.read_text())
        requests=[json.loads(s) for s in (path.parent/'requests.jsonl').read_text().splitlines()]
        summary=summarize_requests(requests,case['load_duration_s'])
        assert summary == case['summary'], path
        log=(path.parent/'server.log').read_text(errors='replace')
        events=[]
        for s in log.split('SEMANTIC_PRESERVATION_EVENT ')[1:]:
            events.append(json.JSONDecoder().raw_decode(s)[0])
        spec=collections.Counter()
        selected=scored=0
        for event in events:
            for k,v in event.get('speculative_transfer_accounting',{}).items():
                if isinstance(v,(float,int)) and v is not None: spec[k]+=v
            for d in event.get('prefetch_selection_records',[]):
                selected+=1
                scored+=d.get('score_status')=='scored'
        hint_events = [json.JSONDecoder().raw_decode(s)[0] for s in log.split('SEMANTIC_WAIT_HINT_EVENT ')[1:]]
        native = dict(
            submitted_jobs=sum(e.get('event')=='staging_submitted' for e in hint_events),
            completed_jobs=sum(e.get('event')=='staging_ready' for e in hint_events),
            submitted_logical_bytes=sum(e.get('submitted_bytes',0) for e in hint_events if e.get('event')=='staging_submitted'),
            completed_physical_bytes=sum(e.get('transfer_bytes',0) for e in hint_events if e.get('event')=='staging_ready'),
            consumed_logical_bytes=sum(e.get('consumed_bytes',0) for e in hint_events if e.get('event')=='staging_consumed'),
            consumed_sources=sum(e.get('event')=='staging_consumed' for e in hint_events),
            scored_nominations=sum(e.get('scored_candidate_count',0) for e in hint_events if e.get('event')=='nomination_decision'),
            nomination_cpu_ns=sum(e.get('nomination_cpu_ns',0) for e in hint_events if e.get('event')=='nomination_decision'),
            demand_reclaimed_blocks=sum(e.get('reclaimed_ready_blocks',0) for e in hint_events if e.get('event')=='demand_reclaim'),
            prior_evicted_keys=sum(len(e.get('prior_evicted_keys',[])) for e in events))
        codec=path.parent.parent/'codec.jsonl'
        records=[json.loads(s) for s in codec.read_text().splitlines()] if codec.exists() else []
        row=dict(variant=path.parent.parent.name,summary=summary,telemetry=case.get('telemetry'),
                 activity=case.get('activity'),native_prefetch=native,speculative=dict(spec),codec={
                     'allocated_bytes':next((r['factor_bytes'] for r in records if r['event']=='allocate'),None),
                     'stores':sum(r['event']=='store' for r in records),
                     'loads':sum(r['event']=='load' for r in records),
                     'total_seconds':sum(r.get('seconds',0) for r in records),
                     'mean_relative_squared_error':sum(r.get('relative_squared_error') or 0 for r in records)/max(1,sum(r['event']=='store' for r in records))},
                 error=case.get('error'),evaluation=case['evaluation'],cleanup=case['cleanup'])
        rows.append(row)
    # Compare exact outputs by source/turn within each repeated workflow.
    paths = list(root.glob('*/*/case.json'))
    for row in rows:
        rep = row['variant'].split('-')[0]
        reference = next((p for p in paths if p.parent.parent.name == rep+'-wait'), None)
        target = next(p for p in paths if p.parent.parent.name == row['variant'])
        if reference:
            def keyed(path):
                return {(r['source_id'],r['turn']):r for r in
                        (json.loads(s) for s in (path.parent/'requests.jsonl').read_text().splitlines())
                        if not r.get('cancel_probe')}
            ref, candidate = keyed(reference), keyed(target)
            common = ref.keys() & candidate.keys()
            row['versus_wait'] = dict(
                matched_requests=len(common),
                exact_prompts=sum(ref[k]['payload']['prompt']==candidate[k]['payload']['prompt'] for k in common),
                exact_outputs=sum(ref[k]['output_token_ids']==candidate[k]['output_token_ids'] for k in common))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--seal',action='store_true');a=p.parse_args()
    rows=extract(a.root)
    write_json(a.root/'analysis.json',rows)
    if a.seal:
        from benchmarks.serving_suite.verify import seal, verify
        from benchmarks.serving_suite.contracts import SCHEMA
        cases=[]
        for path in sorted(a.root.glob('*/*/case.json')):
            case=json.loads(path.read_text())
            case['name']=str(path.parent.relative_to(a.root))
            cases.append(case)
        first=next(a.root.glob('*/plan/plan.json'))
        plan=json.loads(first.read_text())
        states={c['evaluation']['status'] for c in cases}
        status='failed' if 'failed' in states else 'inconclusive' if 'inconclusive' in states else 'passed'
        write_json(a.root/'result.json',dict(schema=SCHEMA,plan=plan,cases=cases,status=status,
                   comparison_scope='custom factorial coordinator using maintained run_case; no paired-suite qualification'))
        seal(a.root)
        print(json.dumps(verify(a.root)))
    for r in rows:
        s=r['summary'];print(r['variant'],s['completed'],round(s['output_tokens_per_s'] or 0,2),
            round(s['ttft_mean_s'] or 0,3),s['content_checks_failed'],r['activity'],r['codec'],
            {k:v for k,v in r['speculative'].items() if k in ['submitted_jobs','consumed_bytes','completed_bytes']})
if __name__=='__main__':main()
