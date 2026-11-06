"""Read-only finite-exposure progress; no inference from unfinished waves."""
import json,sys,statistics
from pathlib import Path
b=Path(__file__).resolve().parent;root=b/'stage3-soak';spec=json.loads((root/'spec.json').read_text());p=root/'result.json';r=json.loads(p.read_text()) if p.exists() else {};cases=r.get('cases',[])
print('soak',r.get('status','starting'),len(cases),'/',len(spec['order']),'cases complete')
for repeat,name in spec['order']:
 d=root/f'{repeat:02d}-{name}'
 if not d.exists():continue
 waves=[]
 resources_path=d/'resources.jsonl'
 resources=[]
 if resources_path.exists():
  for line in resources_path.read_text().splitlines():
   try:resources.append(json.loads(line))
   except json.JSONDecodeError:pass
 for path in sorted(d.glob('workflow-*.jsonl')):
  rows=[]
  for line in path.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
  starts=[x for x in rows if x['event']=='session_start'];ends=[x for x in rows if x['event']=='session_end']
  entry=dict(file=path.name,offered=len(starts),finished=len(ends),success=sum(bool(x.get('success')) for x in ends))
  if len(ends)==16:
   end=max(x['ns'] for x in ends)/1e9
   tail=[x['rss_bytes'] for x in resources if end-5<=x['monotonic_s']<=end]
   entry['last5s_summed_server_rss_mib']=round(statistics.median(tail)/1048576,2) if tail else None
  waves.append(entry)
 requests=d/'requests.jsonl';count=len(requests.read_text().splitlines()) if requests.exists() else 0
 print(d.name,count,'responses',json.dumps(waves))
