"""Independent raw arithmetic and HTTP/EOS checks for the frozen triage block."""
from pathlib import Path
import json,gzip,hashlib,math,statistics,collections,csv
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=ROOT/'evidence/live-optimization-2080ti-20261103';m=json.loads((E/'triage-confirmation-frozen.json').read_text())
def p95(v):
 v=sorted(v);p=.95*(len(v)-1);i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
def read(name):
 p=H/'runs'/name/'raw.json.gz';r=json.load(gzip.open(p));qs=r['requests'];assert r['status']=='complete' and r['performance_eligible'] and len(qs)==r['planned_requests']==72 and all(q['success'] and q['natural_eos'] for q in qs) and not r['owned_descendants_remaining']
 assert not any(r['args'][k] for k in ['audit','quality','boundary_diagnostic','feasibility_only','observe_native_queries'])
 hc=r['http_client_configuration'];assert hc['force_close'] and hc['automatic_POST_retry'] is False and hc['application_retries']==0
 events=r['http_connection_events'];counts=collections.Counter(e['event'] for e in events);assert counts['connection_reuse']==0 and counts['request_exception']==0
 n=72+len(r['hints']);assert all(counts[k]==n for k in ['request_start','connection_create_start','connection_create_end','request_headers_received']);assert all(e['status']==200 for e in events if e['event']=='request_headers_received')
 seconds=(r['load_end_ns']-r['load_start_ns'])/1e9;tokens=sum(len(q['output_token_ids']) for q in qs);ttft=[(q['first_token_ns']-q['request_start_ns'])/1e6 for q in qs];e2e=[(q['request_end_ns']-q['request_start_ns'])/1e6 for q in qs];memory=max(s['gpu']['memory_used_bytes'] for s in r['resource_samples'] if 'gpu' in s)/2**20
 return {'name':name,'load_s':seconds,'tokens':tokens,'tokens_s':tokens/seconds,'vram_mib':memory,'mean_ttft_ms':statistics.mean(ttft),'p95_ttft_ms':p95(ttft),'mean_e2e_ms':statistics.mean(e2e),'p95_e2e_ms':p95(e2e),'initial_prompts':{q['index']:q['payload']['prompt'] for q in qs if q['turn']==0},'waits':r['workload']['private_wait_ms'],'http_requests':n,'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
pairs=[]
for p in m['pairs']:
 n=read(p['runs']['native']);c=read(p['runs']['candidate']);assert n.pop('initial_prompts')==c.pop('initial_prompts');assert n.pop('waits')==c.pop('waits')
 row={'pair':p['pair'],'partition':p['partition'],'native':n,'candidate':c,'workflow_log_ratio':math.log(n['load_s']/c['load_s']),'token_log_ratio':math.log(c['tokens_s']/n['tokens_s']),'vram_saving_mib':n['vram_mib']-c['vram_mib']}
 row.update({k:c[k]-n[k] for k in ['mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','p95_e2e_ms']});pairs.append(row)
def interval(v,log):
 t={4:3.182446305284263,12:2.200985160082949}[len(v)];avg=statistics.mean(v);h=t*statistics.stdev(v)/math.sqrt(len(v));vals=[avg,avg-h,avg+h]
 if log:vals=[100*math.expm1(x) for x in vals]
 return dict(zip(['mean','lower95','upper95'],vals))
reports={}
for label,rows in [('all',pairs),*[(str(seed),[p for p in pairs if p['partition']==seed]) for seed in [7703,8803,9901]]]:
 reports[label]={k:interval([p[k] for p in rows],k.endswith('log_ratio')) for k in ['workflow_log_ratio','token_log_ratio','mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','p95_e2e_ms']};reports[label]['vram_savings_mib']=[p['vram_saving_mib'] for p in rows]
for run in m['execution_order']:read(run['name'])
out={'schema':'root-independent-triage-confirmation-verification-v1','all27_workflows_1944_EOS_and_HTTP_verified':True,'all_planned_outcomes_retained':True,'scope':'12whole-workflow pairs conditional on3fixed24session traces, not12independent source samples. All27invocations checked.','reports':reports,'pairs':pairs,'judgment':'Pooled throughput benefit unproven and p95TTFT materially worse; current policy not production-qualified. Continue policy optimization rather than postselecting8803.'};(E/'triage-confirmation-root-verification.json').write_text(json.dumps(out,indent=2)+'\n')
with (E/'triage-confirmation-pairs.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['pair','partition','native_seconds','candidate_seconds','workflow_gain_percent','token_rate_gain_percent','VRAM_saved_MiB','mean_TTFT_change_ms','p95_TTFT_change_ms'])
 for p in pairs:w.writerow([p['pair'],p['partition'],p['native']['load_s'],p['candidate']['load_s'],100*math.expm1(p['workflow_log_ratio']),100*math.expm1(p['token_log_ratio']),p['vram_saving_mib'],p['mean_ttft_ms'],p['p95_ttft_ms']])
print(json.dumps({'all_EOS_HTTP':True,'pooled':reports['all']},indent=2))
