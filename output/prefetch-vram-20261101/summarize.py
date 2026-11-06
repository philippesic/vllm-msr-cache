import csv,gzip,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import analyze
items=[];runs={}
for p in sorted((H/'runs').glob('*/raw.json.gz')):
 raw,s,rs,_=analyze.analyze(p);runs[s['name']]=(raw,s,rs,[])
 ret=[r for r in rs if 'return' in r['role']];bg=[r for r in rs if r['role'].startswith('background')]
 mem=(raw.get('postload_export',{}).get('workers') or [{}])[0];before=(raw.get('before_load_export',{}).get('workers') or [{}])[0]
 a=analyze.aggregate(ret,s['all']['elapsed_s']);b=analyze.aggregate(bg,s['all']['elapsed_s'])
 adoptions=[r['native_adoption'] for r in ret if r['native_adoption']]
 bad=[r['index'] for r in rs if not r['success'] or not r['terminal_authoritative'] or not r['ordered_keys_complete'] or not r['physical_identity_complete'] or not r['block_table_complete']]
 staged=[e for e in raw['wait_hint_events'] if e.get('event')=='staging_submitted'];ready=[e for e in raw['wait_hint_events'] if e.get('event')=='staging_ready'];decision=[e for e in raw['wait_hint_events'] if e.get('event')=='nomination_decision']
 row={'name':s['name'],'phase':s['phase'],'workload':s['workload'],'seed':s['seed'],'policy':s['policy'],'status':s['status'],'gpu_blocks':raw['geometry']['gpu_blocks'],'kv_mib':raw['geometry']['gpu_kv_bytes']/2**20,'cpu_kv_mib':raw['geometry']['cpu_kv_bytes']/2**20,'tps':s['all']['output_tokens_s'],'elapsed_s':s['all']['elapsed_s'],'output_tokens':s['all']['output_tokens'],'ttft_mean_ms':s['all']['release_ttft_ms']['mean'],'return_ttft_ms':a['release_ttft_ms']['mean'],'return_p95_ttft_ms':a['release_ttft_ms']['p95'],'return_itl_ms':a['mean_itl_ms'],'background_ttft_ms':b['release_ttft_ms']['mean'],'background_itl_ms':b['mean_itl_ms'],'cpu_s':s['resources'].get('cpu_s_sampled'),'gpu_allocated_before_bytes':before.get('gpu_allocated_bytes'),'gpu_reserved_before_bytes':before.get('gpu_reserved_bytes'),'gpu_allocated_after_bytes':mem.get('gpu_allocated_bytes'),'gpu_reserved_after_bytes':mem.get('gpu_reserved_bytes'),'gpu_load_peak_allocated_bytes':s['load_gpu_peak_allocated_bytes'],'gpu_load_peak_reserved_bytes':s['load_gpu_peak_reserved_bytes'],'device_memory_load_peak_bytes':s['resources'].get('device_memory_used_peak_bytes_sampled'),'device_memory_before_bytes':next((r['gpu']['memory_used_bytes'] for r in reversed(raw.get('resource_samples',[])) if r['at_ns']<=raw['load_start_ns'] and 'gpu' in r),None),'captures':s['worker_capture_calls'],'applied_score_keys':s['manager_counts'].get('relevance_apply_keys',0),'score_apply_calls':s['manager_counts'].get('relevance_apply_calls',0),'scored_candidate_count':sum(e.get('scored_candidate_count',0) for e in decision),'constrained_decisions':sum(bool(e.get('shared_budget_constrained')) for e in decision),'staging_jobs':len(staged),'staged_bytes':sum(e.get('submitted_bytes',0) for e in staged),'ready_bytes':sum(e.get('transfer_bytes',0) for e in ready),'consumed_blocks':s['prefetch_consumed_blocks'],'consumed_bytes':s['prefetch_consumed_bytes'],'native_adoptions':len(adoptions),'all_adoptions_same_physical_ids':all(a.get('same_physical_blocks') for a in adoptions),'return_splice_bytes':sum(r['splice_bytes'] or 0 for r in ret),'return_demand_h2d_bytes':sum(r['demand_h2d_bytes'] or 0 for r in ret),'all_d2h_bytes':s['d2h_bytes'],'unused_completed_bytes':s['prefetch_unused_completed_bytes'],'wasted_bytes':s['prefetch_waste_bytes'],'hint_results':s['hint_result_counts'],'preemptions':s['preemptions'],'allocation_failures':s['prefetch_allocation_failures'],'bad_request_indices':bad,'transfer_failed_jobs':s['transfer_failed_jobs'],'owned_remaining':s['owned_descendants_remaining']}
 items.append(row)
comparisons=[]
for name,run in runs.items():
 if run[1]['policy']=='arc':continue
 for other,control in runs.items():
  if control[1]['policy']!='arc' or control[1]['seed']!=run[1]['seed'] or control[1]['workload']!=run[1]['workload']:continue
  comparisons.append(analyze.compare(control,run))
(H/'summaries.json').write_text(json.dumps(items,indent=2)+'\n');(H/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n');analyze.csvwrite(H/'measurements.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in items])
print(json.dumps(items,indent=2))
