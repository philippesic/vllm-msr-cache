from pathlib import Path
import gzip,json,collections,hashlib,statistics
H=Path(__file__).resolve().parent;R=H.parents[1];E=R/'evidence/live-optimization-2080ti-20261103';plan=json.loads((H/'gpu-coverage-frozen-plan-nov3.json').read_text());rows=[]
for item in plan['execution_order']:
 p=H/'runs'/item['name']/'raw.json.gz';r=json.load(gzip.open(p));qs=r['requests']
 assert r['status']=='complete' and r['performance_eligible'] is False and len(qs)==r['planned_requests']==72 and all(q['success'] and q['natural_eos'] for q in qs)
 assert r['config']['kv_connector_extra_config'].get('wait_hint_cpu_retention_gpu_aware',False) is False and r['config']['kv_connector_extra_config']['wait_hint_gpu_prefix_observation'] is True
 assert not r['owned_descendants_remaining']
 events=collections.Counter(e['event'] for e in r['http_connection_events']);n=72+len(r['hints'])
 assert events['connection_reuse']==events['request_exception']==0 and all(events[k]==n for k in ['request_start','connection_create_start','connection_create_end','request_headers_received'])
 cfg=r['http_client_configuration'];assert cfg['force_close'] is True and cfg['automatic_POST_retry'] is False and cfg['application_retries']==0
 bywait=collections.defaultdict(list);other=collections.Counter();acks=[]
 for h in r['hints']:
  a=h.get('response',{});assert h['http_status']==200
  if a.get('reason')!='cpu_prefix_recency_refreshed':other[a.get('reason')]+=1;continue
  if a.get('idempotent_replay'):continue
  assert a['gpu_observation_available'] is a['gpu_observation_free_queue_validated'] is True
  assert a['gpu_observation_cached_free_prefix_keys']+a['gpu_observation_cached_active_prefix_keys']==a['gpu_observation_cached_prefix_keys']<=a['gpu_observation_source_keys']
  assert a['gpu_observation_source_keys']==h['hint']['prefix_boundary_tokens']//16
  bywait[(h['hint']['session_id'],h['source_completed_at_ns'])].append(h);acks.append(a)
 waits=[]
 for key,hs in bywait.items():
  hs.sort(key=lambda h:h['started_ns']);prior=[q for q in qs if q['session_id']==key[0] and q['request_end_ns']==key[1]];assert len(prior)==1
  q=prior[0];computed=q['payload']['prompt']+q['output_token_ids'][:-1];boundary=len(computed)//16*16
  assert all(h['started_ns']>=q['request_end_ns'] and h['hint']['prefix_boundary_tokens']==boundary for h in hs)
  first,last=hs[0]['response'],hs[-1]['response']
  waits.append({'session_id':key[0],'source_completed_at_ns':key[1],'observations':len(hs),'source_keys':first['gpu_observation_source_keys'],'first_gpu_cached':first['gpu_observation_cached_prefix_keys'],'first_gpu_free':first['gpu_observation_cached_free_prefix_keys'],'first_gpu_active':first['gpu_observation_cached_active_prefix_keys'],'first_CPU_ready':first['ready_cpu_prefix_keys'],'last_gpu_cached':last['gpu_observation_cached_prefix_keys'],'last_gpu_free':last['gpu_observation_cached_free_prefix_keys'],'first_full':first['gpu_observation_full_prefix_cached'],'last_full':last['gpu_observation_full_prefix_cached']})
 rows.append({'partition':item['partition'],'name':item['name'],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'valid_nonreplayed_snapshots':len(acks),'other_ack_reasons':dict(other),'HTTP_attempts':n,'waits':waits,'first_full_count':sum(w['first_full'] for w in waits),'first_with_free_count':sum(w['first_gpu_free']>0 for w in waits),'first_full_then_partial_last':sum(w['first_full'] and not w['last_full'] for w in waits),'first_CPU_full_count':sum(w['first_CPU_ready']==w['source_keys'] for w in waits)})
out={'schema':'root-independent-GPU-prefix-coverage-verification-v1','all3workflows216EOS_HTTP_verified':True,'rows':rows,'summary':{'waits':sum(len(r['waits']) for r in rows),'first_full_GPU':sum(r['first_full_count'] for r in rows),'first_with_free_GPU':sum(r['first_with_free_count'] for r in rows),'became_partial_by_last_snapshot':sum(r['first_full_then_partial_last'] for r in rows),'first_CPU_full':sum(r['first_CPU_full_count'] for r in rows),'valid_nonreplayed_snapshots':sum(r['valid_nonreplayed_snapshots'] for r in rows)},'scope':'Read-only snapshots during real unbounded generation. Prefix-position counts can include shared/repeated keys. No proof of demand hits, avoided bytes, future survival or performance benefit.'}
(E/'gpu-coverage-root-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
