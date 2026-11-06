"""Retrospective evaluation only; no future reuse information enters policy."""
from pathlib import Path
import json,collections,statistics
H=Path(__file__).resolve().parent;P=H.parent/'vram-frontier-20261102'
rows=[]
for p in (P/'runs').glob('*/summary.json'):
 s=json.loads(p.read_text());u=s.get('offline_staging_utility',{});events=u.get('event_rows',[])
 if not events:continue
 groups=collections.defaultdict(lambda:dict(jobs=0,bytes=0,lifetimes=[]))
 for e in events:
  if not e.get('submitted'):continue
  g=groups[str(e.get('retirement_reason'))];g['jobs']+=1;g['bytes']+=e.get('explicit_successful_completion_bytes') or 0
  if e.get('reservation_lifetime_ms') is not None:g['lifetimes'].append(e['reservation_lifetime_ms'])
 for g in groups.values():
  xs=g.pop('lifetimes');g['median_reservation_ms']=statistics.median(xs) if xs else None
 rows.append(dict(run=p.parent.name,completed=u.get('global_explicit_successful_completion_bytes_observed'),consumed=u.get('global_named_demand_consumed_bytes'),retirement=dict(groups)))
(H/'historical-waste.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 if 'b768-' in r['run']:print(json.dumps(r,indent=2))
