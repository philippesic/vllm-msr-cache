"""Retain the first three fully repeated native cells of the 5080 campaign.

Read-only derived reporting; run from this campaign\'s exported source with
PYTHONDONTWRITEBYTECODE=1 and the literal pinned interpreter. It prints ready=false
without writing if B03 lacks a frozen matched pair, and preserves prior receipts.
This job-specific script does not acquire, replan or complete a campaign.
"""


def main():

    from pathlib import Path
    import json,statistics,sys
    source=Path('/home/philip/vllm-semantic-cache-ext/output/stage6-5080-20261108/primary-execution-bundle-01/source').resolve()
    if Path.cwd().resolve()!=source:
        raise ValueError('Run this derived helper as a file from the frozen exported source directory')
    sys.path.insert(0,str(source))
    from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,local_path
    from benchmarks.serving_suite.campaign_report import completed_blocks,paired_estimate,compare_classes
    from benchmarks.serving_suite import campaign_report
    assert Path(campaign_report.__file__).resolve().is_relative_to(source)
    out=Path('/home/philip/vllm-semantic-cache-ext/output/stage6-5080-20261108/linux-volume-01/primary-20261109')
    e=Path('/home/philip/vllm-semantic-cache-ext/evidence/stage6-5080-20261109')
    protocol=load(out/'protocol.lock.json')
    expected=sorted(b['repeat']for b in protocol['schedule']if b['model']=='qwen3_8b_awq'and b['cell']=='B03')
    blocks=completed_blocks(out,'confirm')
    observed=sorted(r for m,c,r in blocks if m=='qwen3_8b_awq'and c=='B03')
    if observed!=expected:
     print(json.dumps(dict(ready=False,cell='B03',complete_pairs=len(observed),expected_pairs=len(expected))))
     raise SystemExit(0)
    runs=[blocks[('qwen3_8b_awq','B03',r)]for r in expected]
    row=dict(model='qwen3_8b_awq',cell='B03',frozen_repeats=expected,all_frozen_repetitions_complete=True,candidate='wait',baseline='arc',throughput=paired_estimate([r['systems']['wait']['rate']/r['systems']['arc']['rate']for r in runs]),arms={},traffic_classes=compare_classes(runs,'arc','wait'),observed_native_output_matches=[])
    fields=('end_user_ttft_mean_s','end_user_ttft_p95_s','e2e_mean_s','e2e_p95_s')
    def raw(data):
     case=data['case'];directory=local_path(out,data['event']['details']['path'])/'measurement'/case['name']
     rows=[json.loads(line)for line in (directory/'requests.jsonl').read_text().splitlines()]
     rows=[r for r in rows if r.get('success')and not r.get('cancel_probe')and r.get('wave',0)>=0 and r.get('source_id')]
     keys=[(r.get('wave',0),r['source_id'],r['node_id'])for r in rows];assert len(keys)==len(set(keys))
     return{key:(r['generated_text'],r['finish_reason'])for key,r in zip(keys,rows)}
    for name in ('arc','wait'):
     points=[r['systems'][name]for r in runs];cases=[p['case']for p in points]
     assert all(c['evaluation']['status']=='passed'and c['cleanup']['verified']and c['summary']['completed']==16 and c['campaign']['completed_units']==16 for c in cases)
     row['arms'][name]=dict(processes=len(points),mean_serving_seconds=statistics.mean(c['load_duration_s']for c in cases),mean_startup_seconds=statistics.mean(c['startup_s']for c in cases),completed_units_per_process=[c['campaign']['completed_units']for c in cases],requests=sum(c['summary']['completed']for c in cases),output_tokens=sum(c['summary']['output_tokens']for c in cases),length_caps=sum(c['summary']['finish_reasons'].get('length',0)for c in cases),latency={field:dict(mean_of_process_statistics=statistics.mean(c['summary'][field]for c in cases),values=[c['summary'][field]for c in cases])for field in fields},cpu_restore_active_processes=sum(p['measurement_validity']['cpu_restore_activity']=='active'for p in points),cpu_pressure_observed_processes=sum(p['measurement_validity']['cpu_pressure_evidence']=='observed'for p in points),returning_cpu_eviction_affected_requests=sum(c['campaign']['native_cache']['returning_requests_with_prior_cpu_eviction']for c in cases),returning_loaded_tokens=sum(c['campaign']['native_cache']['returning_loaded_tokens']for c in cases),artifacts=[dict(attempt=p['event']['attempt_id'],path=p['event']['details']['path'],seal=p['event']['details']['seal'])for p in points])
    for r in runs:
     a,b=raw(r['systems']['arc']),raw(r['systems']['wait'])
     assert a.keys()==b.keys()and len(a)==16
     row['observed_native_output_matches'].append(dict(block=r['block_id'],requests=len(a),identical_generated_text_and_finish_reason=a==b))
    previous=load(e/'confirmation-first-cells.json')
    assert digest(protocol)==previous['protocol_digest']
    receipt=dict(schema=previous['schema'],campaign_complete=False,complete_for_listed_cells=True,campaign_id=previous['campaign_id'],protocol_digest=previous['protocol_digest'],acquisition_source=previous['acquisition_source'],cells=[*previous['cells'],row],scope='Only the first three native cells have all frozen repetitions: B01 twelve, B02 six, B03 twelve. Earlier two-cell checkpoint retained unchanged. Frozen Student t on whole-process paired log throughput ratios; no request-level independent repetition or multiplicity adjustment. Serving time excludes startup; latency means are means of process statistics, including process p95s, not pooled percentiles. Actual fixed-cohort output matching is not answer correctness or population quality equivalence. CPU-restore/pressure facts retain observed scope; this is not an offload-benefit or external-system estimate. Remaining confirmation, overhead, operational, source/cap/integrity/portability audits and final campaign report remain pending. No acquisition or stopping-rule change.')
    target=e/'confirmation-three-cells.json';assert not target.exists();atomic_json(target,receipt)
    print(json.dumps(dict(ready=True,cell='B03',estimate=row['throughput'],arms={k:{a:b for a,b in v.items()if a not in ('artifacts','latency','completed_units_per_process')}for k,v in row['arms'].items()},latency={k:{field:v['mean_of_process_statistics']for field,v in arm['latency'].items()}for k,arm in row['arms'].items()},native_outputs_identical=all(x['identical_generated_text_and_finish_reason']for x in row['observed_native_output_matches'])),indent=2))



if __name__ == "__main__":
    main()
