from pathlib import Path
import json,gzip,hashlib,math,statistics,collections
H=Path(__file__).resolve().parent;R=H.parents[1];E=R/'evidence/live-optimization-2080ti-20261103';m=json.loads((H/'return-priority-frozen-plan-nov3.json').read_text())
def p95(v):
 v=sorted(v);p=.95*(len(v)-1);i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
def metric(text,name,label=None):
 v=[float(s.rsplit(' ',1)[1]) for s in text.splitlines() if (s.startswith(name+'{') or s.startswith(name+' ')) and (label is None or label in s)];return sum(v) if v else None
for n,pin in m['file_pins'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==pin,n
rows={}
for plan in m['execution_order']:
 p=H/'runs'/plan['name']/'raw.json.gz';r=json.load(gzip.open(p));qs=r['requests'];assert r['status']=='complete' and r['performance_eligible'] and len(qs)==r['planned_requests']==72 and not r['owned_descendants_remaining'];a=r['args'];credit=plan['priority_credit_ms'];gate=plan['gate_enabled'];assert a['gpu_gated_cpu_retention']==gate and a['return_priority_ms']==credit
 assert not any(a[k] for k in ['audit','quality','boundary_diagnostic','feasibility_only','observe_native_queries'])
 assert '--scheduling-policy' in r['server_command'] and r['server_command'][r['server_command'].index('--scheduling-policy')+1]=='priority'
 assert hashlib.sha256((p.parent/'workload.py').read_bytes()).hexdigest()==m['file_pins']['output/prefetch-native-20261102/live_return_priority_workload_nov3.py']
 for q in qs:
  body=q['payload'];assert q['success'] and q['natural_eos'] and q['done_seen'] and q['terminal_complete'] and q['finish_reasons']==['stop'];assert len(body['prompt'])+body['max_tokens']==7680 and len(q['output_token_ids'])<body['max_tokens'] and not body.get('ignore_eos',False) and not body.get('stop') and not body.get('stop_token_ids')
  assert q['priority_origin_ns']==r['load_start_ns'] and q['priority_clock_ns']==q['request_start_ns'] and q['priority_credit_ms']==credit
  priority=0 if credit==0 else (q['request_start_ns']-r['load_start_ns'])//1000000-(credit if q['turn']>0 else 0)
  assert body['priority']==priority
 hc=r['http_client_configuration'];assert hc['force_close'] and hc['automatic_POST_retry'] is False and hc['application_retries']==0
 events=r['http_connection_events'];cc=collections.Counter(e['event'] for e in events);n=72+len(r['hints']);assert all(cc[k]==n for k in ['request_start','connection_create_start','connection_create_end','request_headers_received']) and not cc['connection_reuse'] and not cc['request_exception'];assert all(e['status']==200 for e in events if e['event']=='request_headers_received')
 seconds=(r['load_end_ns']-r['load_start_ns'])/1e9;tokens=sum(len(q['output_token_ids']) for q in qs);vram=max(s['gpu']['memory_used_bytes'] for s in r['resource_samples'] if 'gpu' in s)/2**20;groups={}
 for label,qq in [('all',qs),('initial',[q for q in qs if q['turn']==0]),('zero_wait_return',[q for q in qs if q['turn'] and r['workload']['private_wait_ms'][f"{q['index']//4}:{q['turn']-1}"]==0]),('positive_wait_return',[q for q in qs if q['turn'] and r['workload']['private_wait_ms'][f"{q['index']//4}:{q['turn']-1}"]>0])]:
  ttft=[(q['first_token_ns']-q['request_start_ns'])/1e6 for q in qq];e2e=[(q['request_end_ns']-q['request_start_ns'])/1e6 for q in qq];groups[label]={'count':len(qq),'mean_ttft_ms':statistics.mean(ttft),'p95_ttft_ms':p95(ttft),'max_ttft_ms':max(ttft),'mean_e2e_ms':statistics.mean(e2e),'p95_e2e_ms':p95(e2e),'max_e2e_ms':max(e2e)}
 deltas={}
 for k,(name,label) in {'H2D_bytes':('vllm:kv_offload_total_bytes_total','transfer_type="CPU_to_GPU"'),'D2H_bytes':('vllm:kv_offload_total_bytes_total','transfer_type="GPU_to_CPU"'),'local_compute':('vllm:prompt_tokens_by_source_total','source="local_compute"'),'GPU_cache_hit':('vllm:prompt_tokens_by_source_total','source="local_cache_hit"'),'external_transfer':('vllm:prompt_tokens_by_source_total','source="external_kv_transfer"'),'preemptions':('vllm:num_preemptions_total',None)}.items():
  b=metric(r['metrics_initial'],name,label);f=metric(r['metrics_final'],name,label);deltas[k]={'initial':b,'final':f,'delta':f-b if b is not None and f is not None else None}
 qmap={q['index']:q for q in qs};noncontig=[]
 for q in qs:
  if q['turn']==0:continue
  prior=qmap[q['index']-1];assert prior['session_id']==q['session_id'] and prior['turn']==q['turn']-1
  history=prior['payload']['prompt']+prior['output_token_ids'][:-1];boundary=(len(history)//16)*16
  if q['payload']['prompt'][:boundary]!=history[:boundary]:noncontig.append(q['index'])
 valid=[h['response'] for h in r['hints'] if h.get('response',{}).get('reason')=='cpu_prefix_recency_refreshed' and not h['response'].get('idempotent_replay')]
 if gate:
  assert r['runtime_resolution'][0]['preservation_audit'] is False
  for ack in valid:
   assert ack['cpu_retention_gpu_gate_policy']=='skip_CPU_advice_while_full_GPU_prefix_cached' and 0<=ack['ready_cpu_prefix_keys']<=ack['cpu_retention_ready_prefix_keys']
   if ack['cpu_retention_skipped_full_gpu']:assert ack['cpu_retention_gpu_index_available'] and ack['cpu_retention_gpu_cached_prefix_keys']==ack['cpu_retention_source_keys'] and ack['ready_cpu_prefix_keys']==0 and ack['cpu_retention_skipped_ready_prefix_keys']==ack['cpu_retention_ready_prefix_keys']
 rows[plan['name']]={'name':plan['name'],'arm':plan['arm'],'partition':plan['partition'],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'load_s':seconds,'tokens':tokens,'tokens_s':tokens/seconds,'tokens_s_gib':tokens/seconds/(vram/1024),'vram_mib':vram,'groups':groups,'counters':deltas,'actual_contiguous_returns':48-len(noncontig),'noncontiguous_return_indices':noncontig,'http_requests':n,'startup_GPU_peak_bytes':max(s['gpu']['memory_used_bytes'] for s in r['startup_resource_samples'] if 'gpu' in s),'peak_process_RSS_sum_bytes':max(sum(x['rss_bytes'] for x in s['processes']) for s in r['resource_samples']),'peak_swap_used_bytes':max(s['host_memory']['swap_used_bytes'] for s in r['resource_samples']),'gate_ops':{'valid':len(valid),'full_GPU_skips':sum(x['cpu_retention_skipped_full_gpu'] for x in valid),'actual_CPU_refresh_operations':sum(x['ready_cpu_prefix_keys'] for x in valid)} if gate else None,'initial_prompts':{q['index']:q['payload']['prompt'] for q in qs if q['turn']==0},'private_waits':r['workload']['private_wait_ms']}
groups=[]
for p in m['pairs']:
 arms={a:rows[n].copy() for a,n in p['runs'].items()};n=arms['native0'];assert all(v['initial_prompts']==n['initial_prompts'] and v['private_waits']==n['private_waits'] for v in arms.values())
 for v in arms.values():v.pop('initial_prompts');v.pop('private_waits')
 comparisons={}
 for a,b in [('native2000','native0'),('native8000','native0'),('gate0','native0'),('gate2000','native2000'),('gate8000','native8000'),('gate2000','gate0'),('gate8000','gate0'),('gate8000','native0')]:
  c=arms[a];n=arms[b];comparisons[a+'_vs_'+b]={'workflow_percent':100*(n['load_s']/c['load_s']-1),'token_rate_percent':100*(c['tokens_s']/n['tokens_s']-1),'token_rate_per_GiB_percent':100*(c['tokens_s_gib']/n['tokens_s_gib']-1),'vram_saved_mib':n['vram_mib']-c['vram_mib'],'groups':{g:{k:c['groups'][g][k]-n['groups'][g][k] for k in c['groups'][g] if k!='count'} for g in c['groups']}}
 groups.append({'partition':p['partition'],'arms':arms,'comparisons':comparisons})
out={'schema':'root-independent-return-priority-discovery-v1','all36_workflows_2592_naturalEOS_HTTP_priority_arithmetic_verified':True,'all407_pins_held':True,'groups':groups,'limits':['Discoveryonearm/tracenoinferentialCI.','Allzero is nativepriority reference, not guaranteedFCFSunderpreemption.','Priority also controls preemption victimselection; finiteburst doesnotprovewallclockfairness.','First-admission promptsourcecounters are not totalphysicalGPUcomputation.','RSSsumcan doublecountsharedmappings;swapishostwide.','Differentactualoutputsandarrivals are retained,notreplayedtoequalize.']};(E/'return-priority-root-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verified':True,'preemptions':[v['counters']['preemptions']['delta'] for v in rows.values()],'summary':[{'partition':g['partition'],'gate8_vs_native8':{k:v for k,v in g['comparisons']['gate8000_vs_native8000'].items() if k!='groups'},'gate8_vs_native8_TTFT':g['comparisons']['gate8000_vs_native8000']['groups']['all'],'native8_vs_native0':{k:v for k,v in g['comparisons']['native8000_vs_native0'].items() if k!='groups'}} for g in groups]},indent=2))
