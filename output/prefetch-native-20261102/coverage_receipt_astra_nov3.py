"""Observation-only coverage receipt. Snapshot counts, never performance evidence."""
from collections import Counter,defaultdict
from cadence_receipt_astra_nov3 import verify_cadence

def verify_coverage(raw):
 cadence=verify_cadence(raw,'periodic');issues=[];rows=[];replays=[]
 cfg=raw.get('config',{}).get('kv_connector_extra_config',{})
 if cfg.get('wait_hint_gpu_prefix_observation') is not True:issues.append('observation_not_configured')
 if cfg.get('wait_hint_cpu_retention_gpu_aware',False) is not False:issues.append('bridge_enabled')
 if raw.get('performance_eligible') is not False or raw.get('args',{}).get('feasibility_only') is not True:issues.append('diagnostic_not_performance_ineligible')
 for h in raw.get('hints',[]):
  ack=h.get('response',{})
  if ack.get('reason')!='cpu_prefix_recency_refreshed':continue
  keys=['gpu_observation_source_keys','gpu_observation_available','gpu_observation_free_queue_validated','gpu_observation_cached_prefix_keys','gpu_observation_cached_free_prefix_keys','gpu_observation_cached_active_prefix_keys','gpu_observation_full_prefix_cached']
  if any(k not in ack for k in keys):issues.append('missing_coverage_fields');continue
  x={k:ack[k] for k in keys};x.update(event_id=h['hint'].get('event_id'),session_id=h['hint'].get('session_id'),owner_id=h['hint'].get('owner_id'),source_completed_at_ns=h.get('source_completed_at_ns'),started_ns=h.get('started_ns'),CPU_ready_keys=ack.get('ready_cpu_prefix_keys'))
  n=x['gpu_observation_source_keys'];ready=x['CPU_ready_keys'];available=x['gpu_observation_available'];counts=[x[k] for k in keys[3:6]]
  if type(n) is not int or n<=0 or type(ready) is not int or not 0<=ready<=n:issues.append('invalid_source_or_CPU_counts')
  if n!=h['hint'].get('prefix_boundary_tokens',-1)//16:issues.append('source_boundary_count_mismatch')
  if ack.get('reserved_gpu_blocks')!=0 or ack.get('speculative_transfer_bytes')!=0:issues.append('unexpected_GPU_work')
  if type(available) is not bool:issues.append('invalid_availability')
  if available:
   if not all(type(v) is int and 0<=v<=n for v in counts) or counts[1]+counts[2]!=counts[0]:issues.append('invalid_GPU_counts')
   if x['gpu_observation_free_queue_validated'] is not True or x['gpu_observation_full_prefix_cached']!=(counts[0]==n):issues.append('invalid_GPU_flags')
  else:
   x['gpu_observation_unavailable_reason']=ack.get('gpu_observation_unavailable_reason')
   if not x['gpu_observation_unavailable_reason'] or any(v is not None for v in counts) or x['gpu_observation_full_prefix_cached'] is not None:issues.append('unavailable_not_explicit')
  (replays if ack.get('idempotent_replay') is True else rows).append(x)
 if not rows:issues.append('no_fresh_coverage_ACKs')
 groups=defaultdict(list)
 for x in sorted(rows,key=lambda x:x['started_ns']):groups[(x['session_id'],x['owner_id'],x['source_completed_at_ns'])].append(x)
 def category(x):
  n=x['gpu_observation_source_keys'];g=x['gpu_observation_cached_prefix_keys'];cpu=x['CPU_ready_keys']
  return ('unavailable' if not x['gpu_observation_available'] else 'complete' if g==n else 'zero' if g==0 else 'partial', 'complete' if cpu==n else 'zero' if cpu==0 else 'partial')
 def describe(xs):
  av=[x for x in xs if x['gpu_observation_available']]
  return {'observations':len(xs),'availability':dict(Counter(str(x['gpu_observation_available']) for x in xs)),'unavailable_reasons':dict(Counter(x.get('gpu_observation_unavailable_reason') for x in xs if not x['gpu_observation_available'])),'GPU_CPU_coverage_crosstab':dict(Counter('/'.join(category(x)) for x in xs)),'GPU_prefix_has_free':sum(x['gpu_observation_cached_free_prefix_keys']>0 for x in av),'GPU_prefix_has_active':sum(x['gpu_observation_cached_active_prefix_keys']>0 for x in av),'GPU_cached_key_observations':sum(x['gpu_observation_cached_prefix_keys'] for x in av) if av else None,'GPU_free_key_observations':sum(x['gpu_observation_cached_free_prefix_keys'] for x in av) if av else None,'GPU_active_key_observations':sum(x['gpu_observation_cached_active_prefix_keys'] for x in av) if av else None,'source_key_observations_available_subset':sum(x['gpu_observation_source_keys'] for x in av) if av else None}
 return {'qualified':cadence['qualified'] and not issues,'issues':issues,'cadence':cadence,'replayed_ACKs_excluded':len(replays),'per_hint':describe(rows),'per_wait_first':describe([xs[0] for xs in groups.values()]),'per_wait_last':describe([xs[-1] for xs in groups.values()]),'wait_groups':[{'key':list(k),'snapshots':len(xs),'first':xs[0],'last':xs[-1]} for k,xs in groups.items()],'fresh_observation_rows':rows,'scope':'Read-only instantaneous contiguous native cached-prefix coverage; free and active counts partition members, not separate contiguous subprefixes. Repeated observations are not unique keys, future survival, demand hits or latency benefit. Unavailable coverage is not zero.'}
