import csv,json,gzip
from pathlib import Path
import analyze,campaign_metrics
H=Path(__file__).resolve().parent
rows=[];comparisons=[]
for plan in ['PLAN-v3.json','FOLLOWUP.json','KNEE.json','CONFIRM.json']:
 if not (H/plan).exists():continue
 p=json.loads((H/plan).read_text())
 for run in p['runs']:
  d=H/'runs'/run['name']
  if not (d/'frontier-disposition.json').exists():continue
  r=analyze.row(run,'screen');r['prefetch_fraction']=run['prefetch_fraction'];r['variant']='backoff' if ('backoff' in run['name']) else ('headroom' if 'pressure-' in run['name'] else ('ARC' if run['arm']=='arc' else 'cap'))
  n=r['prefetch_H2D_bytes'];r['useful_pct']=100*r['named_consumed_prefetch_bytes']/n if n else None
  rows.append(r)
for r in rows:
 controls=[a for a in rows if a['arm']=='arc' and a['seed']==r['seed'] and a['gpu_blocks']==r['gpu_blocks']]
 if not controls:continue
 a=controls[0];r['TPS_vs_ARC_pct']=100*(r['TPS']/a['TPS']-1);r['p95TTFT_vs_ARC_pct']=100*(r['TTFT_p95_ms']/a['TTFT_p95_ms']-1)
 if r['arm']=='arc':continue
 left,_=campaign_metrics.load(H/'runs'/a['name']);right,_=campaign_metrics.load(H/'runs'/r['name'])
 c=campaign_metrics.compare(left,right);comparisons.append({k:v for k,v in c.items() if k not in ['left_summary','right_summary']})
(H/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
(H/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
if rows:
 with (H/'RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
for r in rows: print({k:r.get(k) for k in ['name','TPS','TTFT_p95_ms','prefetch_H2D_bytes','useful_pct','native_adoptions','preemptions','TPS_vs_ARC_pct']})
