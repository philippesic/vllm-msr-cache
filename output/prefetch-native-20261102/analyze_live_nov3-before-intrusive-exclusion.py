import gzip,json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent

def metrics(path):
 r=json.load(gzip.open(path));q=r['requests'];
 if not q or 'load_end_ns' not in r:return dict(name=r['name'],status=r['status'],error=r.get('error'),performance_eligible=False)
 duration=(r['load_end_ns']-r['load_start_ns'])/1e9
 def dist(v):
  v=sorted(v);pos=(len(v)-1)*.95;i=int(pos)
  return {'mean':statistics.mean(v),'p95':v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)}
 mem=max(x.get('gpu',{}).get('memory_used_bytes',0) for x in r['resource_samples'])
 tokens=sum(len(x['output_token_ids']) for x in q)
 return dict(name=r['name'],status=r['status'],requests=len(q),planned=r['planned_requests'],performance_eligible=(r['status']=='complete' and len(q)==r['planned_requests'] and all(x.get('natural_eos') for x in q)),natural_eos=sum(x.get('natural_eos',False) for x in q),length_terminations=sum(x.get('finish_reasons')==['length'] for x in q),tokens=tokens,load_s=duration,tokens_s=tokens/duration,requests_s=len(q)/duration,peak_nvml_mib=mem/2**20,tokens_s_per_gib=tokens/duration/(mem/2**30),ttft_ms=dist([(x['first_token_ns']-x['request_start_ns'])/1e6 for x in q]),e2e_ms=dist([(x['request_end_ns']-x['request_start_ns'])/1e6 for x in q]),owned_descendants_remaining=r['owned_descendants_remaining'])
if __name__=='__main__':
 rows=[metrics(p) for p in sorted((H/'runs').glob('live-nov3-*/raw.json.gz'))]
 (H/'live-nov3-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
