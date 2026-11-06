"""Descriptive complete and failed workflows; only full natural EOS is eligible."""
import gzip,json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent

def metrics(path):
 r=json.load(gzip.open(path));q=r.get('requests',[])
 base=dict(name=r['name'],status=r['status'],error=r.get('error'),requests=len(q),planned=r.get('planned_requests'),performance_eligible=False,owned_descendants_remaining=r.get('owned_descendants_remaining'))
 if not q or 'load_end_ns' not in r:return base
 duration=(r['load_end_ns']-r['load_start_ns'])/1e9
 def dist(v):
  if not v:return {'mean':None,'p95':None,'observations':0}
  v=sorted(v);pos=(len(v)-1)*.95;i=int(pos)
  return {'mean':statistics.mean(v),'p95':v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i),'observations':len(v)}
 mem=max((x.get('gpu',{}).get('memory_used_bytes',0) for x in r.get('resource_samples',[])),default=0)
 tokens=sum(len(x.get('output_token_ids',[])) for x in q)
 completed=sum(bool(x.get('success')) for x in q)
 eligible=(r.get('performance_eligible',False) and not any(r.get('args',{}).get(k,False) for k in ['audit','quality','boundary_diagnostic','feasibility_only','observe_native_queries']) and r['status']=='complete' and len(q)==r.get('planned_requests') and all(x.get('natural_eos') and x.get('success') for x in q))
 return dict(base,performance_eligible=bool(eligible),completed_requests=completed,natural_eos=sum(x.get('natural_eos',False) for x in q),length_terminations=sum(x.get('finish_reasons')==['length'] for x in q),tokens=tokens,load_s=duration,tokens_s=tokens/duration if duration>0 else None,requests_s=completed/duration if duration>0 else None,peak_nvml_mib=mem/2**20,tokens_s_per_gib=tokens/duration/(mem/2**30) if duration>0 and mem>0 else None,ttft_ms=dist([(x['first_token_ns']-x['request_start_ns'])/1e6 for x in q if 'first_token_ns' in x]),e2e_ms=dist([(x['request_end_ns']-x['request_start_ns'])/1e6 for x in q if 'request_end_ns' in x]),latency_scope='all planned requests' if eligible else 'observed subset only; not an eligible workflow')
if __name__=='__main__':
 rows=[metrics(p) for p in sorted((H/'runs').glob('live-nov3-*/raw.json.gz'))]
 (H/'live-nov3-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
