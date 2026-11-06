"""Independent raw-only arithmetic, without importing the campaign analyzers."""
from pathlib import Path
import json,gzip,math,statistics,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent;E=ROOT/'evidence/live-optimization-2080ti-20261103'
m=json.loads((E/'confirmation-frozen.json').read_text());other=json.loads((H/'live-nov3-confirmation-independent-analysis.json').read_text())
def pct95(v):
 v=sorted(v);p=.95*(len(v)-1);i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
def read(name):
 path=H/'runs'/name/'raw.json.gz';r=json.load(gzip.open(path));qs=r['requests'];a=r['args']
 assert r['status']=='complete' and r['performance_eligible'] and len(qs)==r['planned_requests']==72
 assert all(q['success'] and q['natural_eos'] and q['finish_reasons']==['stop'] for q in qs)
 assert not any(a[k] for k in ['audit','quality','boundary_diagnostic','feasibility_only','observe_native_queries'])
 assert r['owned_descendants_remaining']==[]
 duration=(r['load_end_ns']-r['load_start_ns'])/1e9;tokens=sum(len(q['output_token_ids']) for q in qs)
 ttft=[(q['first_token_ns']-q['request_start_ns'])/1e6 for q in qs]
 memory=max(s['gpu']['memory_used_bytes'] for s in r['resource_samples'] if 'gpu' in s)/2**20
 return {'name':name,'load_s':duration,'tokens':tokens,'tokens_s':tokens/duration,'memory_mib':memory,'mean_ttft_ms':statistics.mean(ttft),'p95_ttft_ms':pct95(ttft),'first_prompts':{q['index']:q['payload']['prompt'] for q in qs if q['turn']==0},'waits':r['workload']['private_wait_ms'],'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
pairs=[]
for p in m['pairs']:
 n=read(p['runs']['native']);c=read(p['runs']['candidate']);assert n.pop('first_prompts')==c.pop('first_prompts');assert n.pop('waits')==c.pop('waits')
 pairs.append({'pair':p['pair'],'partition':p['partition'],'native':n,'candidate':c,'workflow_log_ratio':math.log(n['load_s']/c['load_s']),'token_log_ratio':math.log(c['tokens_s']/n['tokens_s']),'mean_ttft_ms':c['mean_ttft_ms']-n['mean_ttft_ms'],'p95_ttft_ms':c['p95_ttft_ms']-n['p95_ttft_ms'],'vram_saving_mib':n['memory_mib']-c['memory_mib']})
def interval(v,log=False):
 avg=statistics.mean(v);t={6:2.570581835636305,12:2.200985160082949}[len(v)];half=t*statistics.stdev(v)/math.sqrt(len(v));vals=[avg,avg-half,avg+half]
 if log:vals=[100*math.expm1(x) for x in vals]
 return dict(zip(['mean','lower95','upper95'],vals))
reports={}
for label,rows in [('all',pairs),('partition_2207',[p for p in pairs if p['partition']==2207]),('partition_3301',[p for p in pairs if p['partition']==3301])]:
 report={k:interval([p[k] for p in rows],k.endswith('log_ratio')) for k in ['workflow_log_ratio','token_log_ratio','mean_ttft_ms','p95_ttft_ms']};report['vram_savings_mib']=[p['vram_saving_mib'] for p in rows];reports[label]=report
 for k,ok in [('workflow_log_ratio','workflow_rate_log_ratio'),('token_log_ratio','token_rate_log_ratio'),('mean_ttft_ms','mean_ttft_difference_ms'),('p95_ttft_ms','p95_ttft_difference_ms')]:
  theirs=other['paired_analysis'][label][ok];expected=[theirs['relative_change_percent'],*theirs['relative_change_95_percent']] if k.endswith('log_ratio') else [theirs['mean'],*theirs['two_sided_95_percent']]
  assert all(math.isclose(x,y,abs_tol=1e-8) for x,y in zip(report[k].values(),expected)),(label,k,report[k],expected)
out={'schema':'root-independent-confirmation-verification-v1','all26_eos_and_cleanup_verified':all(read(r['name']) for r in m['execution_order']),'independent_astra_arithmetic_agrees':True,'scope':'12whole-workflow pairs, conditional on two fixed24session input sets; not12independent source traces','reports':reports,'pairs':pairs,'judgment':'Pooled unique benefit unproven; continue optimization. Single input-set opportunity signal is not broad production qualification.'}
(E/'confirmation-root-verification.json').write_text(json.dumps(out,indent=2)+'\n')
with (E/'confirmation-pairs.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['pair','partition','native_seconds','candidate_seconds','workflow_gain_percent','token_rate_gain_percent','VRAM_saved_MiB','mean_TTFT_change_ms','p95_TTFT_change_ms'])
 for p in pairs:w.writerow([p['pair'],p['partition'],p['native']['load_s'],p['candidate']['load_s'],100*math.expm1(p['workflow_log_ratio']),100*math.expm1(p['token_log_ratio']),p['vram_saving_mib'],p['mean_ttft_ms'],p['p95_ttft_ms']])
print(json.dumps({'agrees':True,'all':reports['all']},indent=2))
