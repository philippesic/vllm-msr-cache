"""Use native counters/raw workflows for the audit-off profile; no invented history."""
import json,statistics,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent;sys.path.insert(0,str(b.parents[1]))
from harness.metrics import parse_metrics_evidence,diff_with_provenance
from benchmarks.serving_suite.report import percentile
root=b/'stage4-minimal-performance';r=json.loads((root/'result.json').read_text());assert r['status']=='passed' and len(r['cases'])==6
cases=[]
for c in r['cases']:
 d=root/c['name'];allrows=[json.loads(x) for x in (d/'requests.jsonl').read_text().splitlines()];rows=[x for x in allrows if x.get('source_id') in c['source_ids'] and not x.get('cancel_probe')];probes=[x for x in allrows if x not in rows]
 assert len(rows)==c['expected_requests']==64 and all(x['success'] for x in rows)
 assert len(probes)==1 and probes[0]['payload']['request_id']=='suite-cancel-probe'
 assert c['operational']['passed'] and c['cleanup']['verified']
 left=parse_metrics_evidence((d/'metrics-before.prom').read_text());right=parse_metrics_evidence((d/'metrics-after.prom').read_text());values,provenance=diff_with_provenance(left['values'],right['values'])
 names=('vllm:request_prefill_kv_computed_tokens_sum','vllm:request_prefill_kv_computed_tokens_count','vllm:kv_offload_load_bytes_total','vllm:kv_offload_store_bytes_total','vllm:num_preemptions_total')
 for n in names:assert values[n] is not None and values[n]==c['metrics'][n],(c['name'],n)
 assert values['vllm:request_prefill_kv_computed_tokens_count']==64
 events=[json.loads(x) for p in sorted(d.glob('workflow-*.jsonl')) for x in p.read_text().splitlines()];tools=[x for x in events if x['event']=='tool'];assert len(tools)==48 and all(x['success'] for x in tools)
 receipts=[x for e in events if e['event']=='hint_window' for x in e['receipts']];acks=[x['response'] for x in receipts if x.get('response') and not x.get('error')];positive=[x for x in acks if x.get('reason')=='cpu_prefix_recency_refreshed' and x.get('ready_cpu_prefix_keys',0)>0]
 if c['variant']=='wait':assert positive and all(x.get('reserved_gpu_blocks')==0 and x.get('speculative_transfer_bytes')==0 for x in acks)
 else:assert not receipts
 returns=[(x['first_token_ns']-x['tool_completed_ns'])/1e9 for x in rows if x['turn']>0 and x.get('first_token_ns') and x.get('tool_completed_ns')];assert len(returns)==48
 case=dict(case=c['name'],benchmark_requests=len(rows),separate_after_timing_cancel_records=len(probes),completed_tools=len(tools),batch_s=c['load_duration_s'],output_tokens=c['summary']['output_tokens'],metrics={n:dict(delta=values[n],before=left['sources'][n],after=right['sources'][n],difference=provenance[n]) for n in names},return_tool_ready_to_first_token_s=dict(count=len(returns),mean=statistics.mean(returns),p95=percentile(returns,.95)),hint_acknowledgements=len(acks),positive_cpu_refresh_acknowledgements=len(positive),ready_key_refresh_observations=sum(x['ready_cpu_prefix_keys'] for x in positive),operational=c['operational'],cleanup=c['cleanup'])
 cases.append(case);print(c['name'],'prefill',values[names[0]],'load_bytes',values[names[2]],'outputs',case['output_tokens'],flush=True)
out=dict(passed=True,checked_unix=time.time(),cases=cases,scope='Audit-off profile:64timed workflow requests and48real tools per case, separate after-timing cancellation/recovery, native before/after metrics with missing/reset provenance, raw returning-request latency and positive CPU-ready acknowledgments. Refreshed-key observations are not unique hits or saved work. No reconstructed per-key eviction/lifetime, first-admission, per-request transfer attribution, or production reliability claim. Generic shape reports include the separately labeled untimed cancellation record; these receipts explicitly separate it.')
(b/'stage4-minimal-native-receipts.json').write_text(json.dumps(out,indent=2)+'\n')
