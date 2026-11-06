"""Independent raw arithmetic and HTTP/EOS checks for the frozen triage block."""
from pathlib import Path
import json,gzip,hashlib,math,statistics,collections,csv
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=ROOT/'evidence/live-optimization-2080ti-20261103';m=json.loads((H/'promotion-discovery-frozen-plan-nov3.json').read_text())
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

rows={x['name']:read(x['name']) for x in m['execution_order']};groups=[]
for group in m['pairs']:
 arms={a:rows[n].copy() for a,n in group['runs'].items()};n=arms['native']
 assert all(a['initial_prompts']==n['initial_prompts'] and a['waits']==n['waits'] for a in arms.values())
 counts={}
 for arm,a in arms.items():
  a.pop('initial_prompts');a.pop('waits');a['tokens_s_gib']=a['tokens_s']/(a['vram_mib']/1024)
  raw=json.load(gzip.open(H/'runs'/group['runs'][arm]/'raw.json.gz'));bywait=collections.defaultdict(list)
  for h in raw['hints']:bywait[(h['hint']['session_id'],h['source_completed_at_ns'])].append(h)
  stops=0
  for key,hs in bywait.items():
   hs.sort(key=lambda h:h['started_ns']);prior=[q for q in raw['requests'] if q['session_id']==key[0] and q['request_end_ns']==key[1]];assert len(prior)==1
   q=prior[0];computed=(q['payload']['prompt']+q['output_token_ids'][:-1]);boundary=len(computed)//16*16
   assert all(h['hint']['prefix_boundary_tokens']==boundary and h['started_ns']>=q['request_end_ns'] for h in hs)
   markers=[i for i,h in enumerate(hs) if h.get('producer_policy_stop')]
   if arm in ['promotion_once_positive']:
    qualify=lambda h: h['response'].get('status')=='no_action' and h['response'].get('reason')=='cpu_prefix_recency_refreshed' and isinstance(h['response'].get('ready_cpu_prefix_keys'),int) and not isinstance(h['response'].get('ready_cpu_prefix_keys'),bool) and (h['response']['ready_cpu_prefix_keys']>0 if arm=='promotion_once_positive' else h['response']['ready_cpu_prefix_keys']==boundary//16>0)
    eligible=[i for i,h in enumerate(hs) if qualify(h)]
    assert markers==eligible and (not markers or markers==[len(hs)-1]);stops+=len(markers)
   else:assert not markers
  counts[arm]={'hints':len(raw['hints']),'waits':len(bywait),'stop_markers':stops}
  if arm.startswith('promotion_'):
   valid=[h['response'] for h in raw['hints'] if h.get('response',{}).get('reason')=='cpu_prefix_recency_refreshed' and not h['response'].get('idempotent_replay')]
   assert all(a['cpu_advice_queue_policy']=='experimental_ready_resident_T2_promotion' and type(a['cpu_advice_promoted_t1_keys']) is type(a['cpu_advice_refreshed_t2_keys']) is int and a['cpu_advice_promoted_t1_keys']>=0 and a['cpu_advice_refreshed_t2_keys']>=0 and a['cpu_advice_promoted_t1_keys']+a['cpu_advice_refreshed_t2_keys']==a['ready_cpu_prefix_keys'] for a in valid)
   counts[arm].update(valid_action_ack_count=len(valid),promoted_t1_touches=sum(a['cpu_advice_promoted_t1_keys'] for a in valid),already_t2_touches=sum(a['cpu_advice_refreshed_t2_keys'] for a in valid))
 differences={}
 for a in ['periodic','promotion_periodic','promotion_once_positive']:
  for b in ['native','periodic']:
   if a==b:continue
   c=arms[a];n=arms[b];differences[a+'_vs_'+b]={'workflow_percent':100*(n['load_s']/c['load_s']-1),'token_rate_percent':100*(c['tokens_s']/n['tokens_s']-1),'tokens_s_gib_percent':100*(c['tokens_s_gib']/n['tokens_s_gib']-1),'vram_saved_mib':n['vram_mib']-c['vram_mib'],**{k:c[k]-n[k] for k in ['mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','p95_e2e_ms']}}
 groups.append({'partition':group['partition'],'arms':arms,'differences':differences,'hint_accounting':counts})
out={'schema':'root-independent-promotion-discovery-verification-v1','all12_workflows864EOS_HTTP_verified':True,'groups':groups,'judgment':'No robust incremental promotion gain. Sparse actual T1 promotions; many repeated already-T2 touches. Discovery only, no pooled confidence interval or production claim.'}
(E/'promotion-discovery-root-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verified':True,'groups':[{'partition':g['partition'],'differences':g['differences'],'hint_accounting':g['hint_accounting']} for g in groups]},indent=2))
