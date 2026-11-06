import gzip,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import analyze
checks=[];pairs=[];supplement=[]
for family in ['confirm','normal','normalconfirm']:
 for workload in (['multi'] if family=='normalconfirm' else ['single','multi']):
  for seed in ([1103,2207,3301] if family!='normal' else [1103]):
   arms={}
   for arm in ['arc','candidate','original']:
    p=H/'runs'/f'{family}-{arm}-{workload}-{seed}-{"arc" if arm=="arc" else "semantic"}'/'raw.json.gz'
    raw,s,reqs,_=analyze.analyze(p);arms[arm]=(raw,s,reqs,[])
    checks.append({k:s[k] for k in ['name','status','terminal_events_matching_requests','terminal_events_authoritative','ordered_key_complete_requests','physical_identity_complete_requests','block_table_complete_requests','transfer_failed_jobs','owned_descendants_remaining']})
    checks[-1]['requests']=s['all']['requests'];checks[-1]['failures']=s['all']['failures']
    if family=='normal':supplement.append(s)
   for arm in ['candidate','original']:
    pairs.append(analyze.compare(arms['arc'],arms[arm]))
(H/'final-integrity.json').write_text(json.dumps({'runs':checks,'pairs':pairs},indent=2)+'\n')
(H/'ordinary-results.json').write_text(json.dumps(supplement,indent=2)+'\n')
extra=[]
for pattern in ['cpu-thread-*','arc-thread-*','multitoken-thread-*','cpu-multitoken-thread-*']:
 for p in sorted((H/'runs').glob(pattern+'/raw.json.gz')):
  raw,s,*_=analyze.analyze(p)
  s['threads']=raw['postload_export']['workers'][0].get('torch_threads');extra.append(s)
(H/'thread-results.json').write_text(json.dumps(extra,indent=2)+'\n')
print(json.dumps({'requests':sum(s['requests'] for s in checks),'bad_runs':[s for s in checks if s['status']!='complete' or s['failures'] or s['requests']!=s['terminal_events_authoritative'] or s['requests']!=s['ordered_key_complete_requests'] or s['requests']!=s['physical_identity_complete_requests'] or s['requests']!=s['block_table_complete_requests'] or s['transfer_failed_jobs'] or s['owned_descendants_remaining']],'pair_differences':[s for s in pairs if s['prompt_mismatch_indices'] or s['output_mismatch_indices'] or s['native_key_mismatch_indices'] or s['missing_native_key_indices']],'threads':[{'name':s['name'],'tps':s['all']['output_tokens_s'],'ttft_mean_ms':s['all']['release_ttft_ms']['mean'],'cpu_s':s['resources']['cpu_s_sampled'],'threads':s['threads'],'applied':s['manager_counts'].get('relevance_apply_keys'),'scored_evictions':s['manager_counts'].get('semantic_evict_calls_with_resident_scores')} for s in extra]},indent=2))
