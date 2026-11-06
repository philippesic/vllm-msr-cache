"""Independent raw arithmetic and HTTP/EOS checks for the frozen triage block."""
from pathlib import Path
import json,gzip,hashlib,math,statistics,collections,csv
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=ROOT/'evidence/live-optimization-2080ti-20261103';m=json.loads((H/'gpu-recency-frozen-plan-nov3.json').read_text())
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
def metric(text,name,label=None):
 vals=[float(line.rsplit(' ',1)[1]) for line in text.splitlines() if line.startswith(name+'{') and (label is None or label in line)]
 return sum(vals) if vals else None
for group in m['pairs']:
 arms={a:rows[n].copy() for a,n in group['runs'].items()};native=arms['native']
 assert all(a['initial_prompts']==native['initial_prompts'] and a['waits']==native['waits'] for a in arms.values())
 for arm,a in arms.items():
  a.pop('initial_prompts');a.pop('waits');a['tokens_s_gib']=a['tokens_s']/(a['vram_mib']/1024)
  raw=json.load(gzip.open(H/'runs'/group['runs'][arm]/'raw.json.gz'));a['prometheus_load_deltas']={}
  series={'H2D_bytes':('vllm:kv_offload_total_bytes_total','transfer_type="CPU_to_GPU"'),'local_compute':('vllm:prompt_tokens_by_source_total','source="local_compute"'),'GPU_cache_hit':('vllm:prompt_tokens_by_source_total','source="local_cache_hit"'),'external_transfer':('vllm:prompt_tokens_by_source_total','source="external_kv_transfer"'),'preemptions':('vllm:num_preemptions_total',None)}
  for k,(name,label) in series.items():
   v0=metric(raw['metrics_initial'],name,label);v1=metric(raw['metrics_final'],name,label)
   a['prometheus_load_deltas'][k]={'initial':v0,'final':v1,'delta':v1-v0 if v0 is not None and v1 is not None else None}
  if arm=='gpu':
   valid=[h['response'] for h in raw['hints'] if h.get('response',{}).get('reason')=='cpu_prefix_recency_refreshed' and not h['response'].get('idempotent_replay')]
   for ack in valid:
    assert ack['gpu_recency_policy']=='experimental_free_cached_prefix_LRU_refresh' and ack['gpu_observation_available'] and ack['gpu_observation_free_queue_validated']
    assert ack['gpu_recency_eligible_free_blocks']==ack['gpu_recency_refreshed_free_blocks']<=ack['gpu_observation_cached_free_prefix_keys']
   a['gpu_operations']={'valid_nonreplayed_ack_count':len(valid),'refreshed_free_block_operations':sum(ack['gpu_recency_refreshed_free_blocks'] for ack in valid),'scope':'Repeated policy operations, not unique keys, demand hits or saved bytes.'}
 differences={}
 for a,b in [('cpu','native'),('gpu','native'),('gpu','cpu')]:
  c=arms[a];n=arms[b];differences[a+'_vs_'+b]={'workflow_percent':100*(n['load_s']/c['load_s']-1),'token_rate_percent':100*(c['tokens_s']/n['tokens_s']-1),'tokens_s_gib_percent':100*(c['tokens_s_gib']/n['tokens_s_gib']-1),'vram_saved_mib':n['vram_mib']-c['vram_mib'],**{k:c[k]-n[k] for k in ['mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','p95_e2e_ms']}}
 groups.append({'partition':group['partition'],'arms':arms,'differences':differences})
out={'schema':'root-independent-GPU-recency-discovery-verification-v1','all9workflows648EOS_HTTP_verified':True,'groups':groups,'judgment':'No robust winner. Mean and p95TTFT worse than CPU-only in all3sets; do not advance GPUrecency to confirmation. Discovery only, no pooled CI/production claim.','source_attribution_limit':'First-admission token attribution is not cumulative physical GPU compute, even without preemptions. Missing metric series remains unavailable rather than zero.'}
(E/'gpu-recency-root-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verified':True,'groups':[{'partition':g['partition'],'diffs':g['differences'],'GPUops':g['arms']['gpu']['gpu_operations']} for g in groups]},indent=2))
