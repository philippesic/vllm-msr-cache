from pathlib import Path
import json,gzip,collections,hashlib,statistics
H=Path(__file__).resolve().parent;R=H.parents[1];E=R/'evidence/live-optimization-2080ti-20261103';m=json.loads((E/'gpu-gate-confirmation-frozen.json').read_text())
def metric(text,name,label=None):
 vals=[float(line.rsplit(' ',1)[1]) for line in text.splitlines() if (line.startswith(name+'{') or line.startswith(name+' ')) and (label is None or label in line)]
 return sum(vals) if vals else None
rows=[]
for plan in m['execution_order']:
 p=H/'runs'/plan['name']/'raw.json.gz';r=json.load(gzip.open(p));gate=plan['arm']=='candidate';assert r['args']['gpu_gated_cpu_retention']==gate
 if gate:assert r['runtime_resolution'][0]['preservation_audit'] is False
 for q in r['requests']:
  a=q['payload'];assert len(a['prompt'])+a['max_tokens']==7680 and not a.get('ignore_eos',False) and not a.get('stop') and not a.get('stop_token_ids')
  assert q['done_seen'] and q['terminal_complete'] and q['finish_reasons']==['stop'] and len(q['output_token_ids'])<a['max_tokens']
 valid=[h['response'] for h in r['hints'] if h.get('response',{}).get('reason')=='cpu_prefix_recency_refreshed' and not h['response'].get('idempotent_replay')]
 for ack in valid:
  if gate:
   assert ack['cpu_retention_gpu_gate_policy']=='skip_CPU_advice_while_full_GPU_prefix_cached'
   assert 0<=ack['ready_cpu_prefix_keys']<=ack['cpu_retention_ready_prefix_keys']
   if ack['cpu_retention_skipped_full_gpu']:
    assert ack['cpu_retention_gpu_index_available'] and ack['cpu_retention_gpu_cached_prefix_keys']==ack['cpu_retention_source_keys'] and ack['ready_cpu_prefix_keys']==0
    assert ack['cpu_retention_skipped_ready_prefix_keys']==ack['cpu_retention_ready_prefix_keys']
   elif not ack['cpu_retention_gpu_index_available']:assert ack['cpu_retention_gpu_cached_prefix_keys'] is None
 deltas={}
 for k,(name,label) in {'H2D_bytes':('vllm:kv_offload_total_bytes_total','transfer_type="CPU_to_GPU"'),'D2H_bytes':('vllm:kv_offload_total_bytes_total','transfer_type="GPU_to_CPU"'),'local_compute':('vllm:prompt_tokens_by_source_total','source="local_compute"'),'GPU_cache_hit':('vllm:prompt_tokens_by_source_total','source="local_cache_hit"'),'external_transfer':('vllm:prompt_tokens_by_source_total','source="external_kv_transfer"'),'preemptions':('vllm:num_preemptions_total',None)}.items():
  a=metric(r['metrics_initial'],name,label);b=metric(r['metrics_final'],name,label);deltas[k]={'initial':a,'final':b,'delta':b-a if a is not None and b is not None else None}
 assert deltas['preemptions']['delta']==0
 qs={q['index']:q for q in r['requests']};contiguous=0;noncontiguous=[]
 for q in qs.values():
  if q['turn']==0:continue
  prev=qs[q['index']-1];assert prev['session_id']==q['session_id'] and prev['turn']==q['turn']-1
  prior=prev['payload']['prompt']+prev['output_token_ids'];boundary=(len(prior)//16)*16
  if q['payload']['prompt'][:boundary]==prior[:boundary]:contiguous+=1
  else:noncontiguous.append(q['index'])
 groups={}
 for label,qq in [('initial',[q for q in qs.values() if q['turn']==0]),('zero_wait_return',[q for q in qs.values() if q['turn'] and r['workload']['private_wait_ms'][f"{q['index']//4}:{q['turn']-1}"]==0]),('positive_wait_return',[q for q in qs.values() if q['turn'] and r['workload']['private_wait_ms'][f"{q['index']//4}:{q['turn']-1}"]>0])]:
  groups[label]={'n':len(qq),'mean_ttft_ms':statistics.mean((q['first_token_ns']-q['request_start_ns'])/1e6 for q in qq)}
 rows.append({'name':plan['name'],'arm':plan['arm'],'partition':plan['partition'],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'counters':deltas,'gate':{'valid_nonreplayed':len(valid),'full_GPU_skips':sum(a.get('cpu_retention_skipped_full_gpu',False) for a in valid),'partial_or_miss_fallback':sum(not a.get('cpu_retention_skipped_full_gpu',False) for a in valid),'actual_CPU_refresh_operations':sum(a['ready_cpu_prefix_keys'] for a in valid)} if gate else None,'actual_contiguous_returns':contiguous,'noncontiguous_return_indices':noncontiguous,'latency_subgroups_descriptive':groups,'peak_serving_process_RSS_sum_bytes':max(sum(p['rss_bytes'] for p in s['processes']) for s in r['resource_samples']),'startup_GPU_peak_bytes':max(s['gpu']['memory_used_bytes'] for s in r['startup_resource_samples'] if 'gpu' in s),'swap_used_bytes_range':[min(s['host_memory']['swap_used_bytes'] for s in r['resource_samples']),max(s['host_memory']['swap_used_bytes'] for s in r['resource_samples'])]})
out={'schema':'root-independent-gate-confirmation-mechanism-v1','all1728_measured_requests_natural_uncapped_generation_verified':True,'all24_no_preemptions':True,'all24_actual_contiguity_reconstructed':True,'rows':rows,'limits':['GPU snapshots and repeat refresh counts are not demand hits or saved bytes.','Prompt source counters count first admission, not cumulative physical GPU computation.','Missing metric series remains null.','RSS sums may double-count shared mappings; no per-process PSS observation.','Swap usage is host-wide and may predate these runs.']};(E/'gpu-gate-confirmation-root-mechanism.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(rows),'all_EOS_cap_and_contiguity_verified':True,'candidate_totals':{k:sum(row['gate'][k] for row in rows if row['gate']) for k in ['valid_nonreplayed','full_GPU_skips','partial_or_miss_fallback','actual_CPU_refresh_operations']}}))
