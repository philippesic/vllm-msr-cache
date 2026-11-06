import json,sys,statistics,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));import analyze
H=Path(__file__).parent
records=[]
for p in sorted((H/'runs').glob('normalconfirm-*/raw.json.gz')):
 raw,s,requests,profiles=analyze.analyze(p);s['arm']=raw['phase'].replace('normalconfirm-','');s['summary_memory']=(raw.get('postload_export',{}).get('workers') or [{}])[0].get('summary_memory');s['requests_detail']=requests;records.append(s)
(H/'ordinary-confirmation-summaries.json').write_text(json.dumps(records,indent=2))
comparisons=[];pooled=[]
for w in ['multi']:
 for seed in [1103,2207,3301]:
  arms={s['arm']:s for s in records if s['workload']==w and s['seed']==seed and s['status']=='complete'}
  if not all(a in arms for a in ['arc','candidate','original']):continue
  a,c,o=[arms[k] for k in ['arc','candidate','original']]
  comparisons.append({'workload':w,'seed':seed,'arc_tps':a['all']['output_tokens_s'],'candidate_tps':c['all']['output_tokens_s'],'original_tps':o['all']['output_tokens_s'],'candidate_vs_arc_pct':100*(c['all']['output_tokens_s']/a['all']['output_tokens_s']-1),'candidate_vs_original_pct':100*(c['all']['output_tokens_s']/o['all']['output_tokens_s']-1),'candidate_ttft_mean_pct':100*(c['all']['release_ttft_ms']['mean']/a['all']['release_ttft_ms']['mean']-1)})
 for arm in ['arc','original','candidate']:
  cells=[s for s in records if s['workload']==w and s['arm']==arm and s['status']=='complete']
  if not cells:continue
  reqs=[r for s in cells for r in s['requests_detail']];elapsed=sum(s['all']['elapsed_s'] for s in cells)
  agg=analyze.aggregate(reqs,elapsed)
  pooled.append({'workload':w,'arm':arm,'runs':len(cells),'tokens_s':agg['output_tokens_s'],'mean_ttft_ms':agg['release_ttft_ms']['mean'],'p95_ttft_ms':agg['release_ttft_ms']['p95'],'median_ttft_ms':agg['release_ttft_ms']['median'],'mean_itl_ms':agg['mean_itl_ms'],'mean_cpu_s':statistics.mean(s['resources']['cpu_s_sampled'] for s in cells),'cpu_s_per_1000_tokens':1000*sum(s['resources']['cpu_s_sampled'] for s in cells)/agg['output_tokens'],'captures':sum(s['worker_capture_calls'] for s in cells),'applied_score_keys':sum(s['manager_counts'].get('relevance_apply_keys',0) for s in cells),'score_apply_calls':sum(s['manager_counts'].get('relevance_apply_calls',0) for s in cells),'scored_evictions':sum(s['manager_counts'].get('semantic_evict_calls_with_resident_scores',0) for s in cells),'rss_peak_gib':max(s['resources']['rss_peak_bytes_sampled'] for s in cells)/2**30,'summary_storage_bytes_max':max((s.get('summary_memory') or {}).get('retained_storage_bytes',0) for s in cells),'kv_restored_tokens':agg['external_reused_tokens']})
result={'pairs':comparisons,'pooled':pooled};(H/'ordinary-confirmation-results.json').write_text(json.dumps(result,indent=2))
for r in comparisons:print(r)
for r in pooled:print(r)
