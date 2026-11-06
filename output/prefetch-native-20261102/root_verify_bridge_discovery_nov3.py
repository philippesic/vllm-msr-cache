"""Independent raw arithmetic and HTTP/EOS checks for the frozen triage block."""
from pathlib import Path
import json,gzip,hashlib,math,statistics,collections,csv
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=ROOT/'evidence/live-optimization-2080ti-20261103';m=json.loads((H/'bridge-discovery-frozen-plan-nov3.json').read_text())
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

rows={x['name']:read(x['name']) for x in m['execution_order']}
groups=[]
for group in m['pairs']:
 arms={a:rows[n].copy() for a,n in group['runs'].items()}
 n=arms['native'];assert all(a['initial_prompts']==n['initial_prompts'] and a['waits']==n['waits'] for a in arms.values())
 for a in arms.values():a.pop('initial_prompts');a.pop('waits')
 differences={}
 for a,b in [('old','native'),('bridge','native'),('bridge','old')]:
  c=arms[a];n=arms[b];differences[a+'_vs_'+b]={'workflow_percent':100*(n['load_s']/c['load_s']-1),'token_rate_percent':100*(c['tokens_s']/n['tokens_s']-1),'vram_saved_mib':n['vram_mib']-c['vram_mib'],**{k:c[k]-n[k] for k in ['mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','p95_e2e_ms']}}
 raw=json.load(gzip.open(H/'runs'/group['runs']['bridge']/'raw.json.gz'))
 acks=[h['response'] for h in raw['hints'] if h.get('response',{}).get('reason')=='cpu_prefix_recency_refreshed']
 assert all('refreshed_cpu_key_count' in a for a in acks)
 activation={'eligible_ack_count':len(acks),'helper_attempts':sum(a['gpu_prefix_lookup_performed'] for a in acks),'coverage_available_count':sum(a['local_gpu_coverage_available'] is True for a in acks),'positive_suffix_ack_count':sum(a['additional_ready_cpu_suffix_keys']>0 for a in acks),'additional_suffix_touches':sum(a['additional_ready_cpu_suffix_keys'] for a in acks),'actual_refreshed_touches':sum(a['refreshed_cpu_key_count'] for a in acks),'scope':'Repeated key touches, not unique keys, bytes saved or a restoration certificate.'}
 groups.append({'partition':group['partition'],'arms':arms,'differences':differences,'bridge_activation':activation})
out={'schema':'root-independent-bridge-discovery-verification-v1','all9_workflows648EOS_HTTP_verified':True,'groups':groups,'judgment':'No incremental bridge gain on these spent workloads. All three bridge workflows slower than the previous policy. Discovery only; do not advance bridge to confirmation.'}
(E/'bridge-discovery-root-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verified':True,'groups':[{'partition':x['partition'],'differences':x['differences'],'activation':x['bridge_activation']} for x in groups]},indent=2))
