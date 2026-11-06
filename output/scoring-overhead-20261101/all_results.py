import json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import analyze
rows=[]
for p in sorted((H/'runs').glob('*/raw.json.gz')):
 sp=p.parent/'summary.json'
 if not sp.exists():analyze.analyze(p)
 s=json.loads(sp.read_text());c=s['manager_counts'];a=s['all']
 rows.append({'run':s['name'],'phase':s['phase'],'workload':s['workload'],'seed':s['seed'],'policy':s['policy'],'threshold':s['threshold'],'status':s['status'],'error':s.get('error'),'output_tokens_s':a['output_tokens_s'],'ttft_mean_ms':a['release_ttft_ms']['mean'],'ttft_p95_ms':a['release_ttft_ms']['p95'],'itl_mean_ms':a['mean_itl_ms'],'cpu_s':s['resources'].get('cpu_s_sampled'),'captures':s['worker_capture_calls'],'applied_score_keys':c.get('relevance_apply_keys'),'score_apply_calls':c.get('relevance_apply_calls'),'evictions_with_resident_scores':c.get('semantic_evict_calls_with_resident_scores'),'restored_tokens':a['external_reused_tokens'],'requests':a['requests'],'failures':a['failures'],'owned_remaining':s['owned_descendants_remaining']})
analyze.csvwrite(H/'run-index.csv',rows);(H/'run-index.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'runs':len(rows),'complete':sum(r['status']=='complete' for r in rows),'failed':[{'run':r['run'],'error':r['error']} for r in rows if r['status']!='complete']},indent=2))
