"""Independent raw actual-cap/overlap/isolation arithmetic beyond common receipt."""
import collections,gzip,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];M=ROOT/'evidence/live-optimization-2080ti-20261103/operational-v3-frozen.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={'schema':'astra-operational-v3-extra-verification-v1','manifest_sha256':sha(M),'cases':{}}
for item in json.loads(M.read_text())['execution_order']:
 p=H/'runs'/item['name']/'raw.json.gz';r=json.load(gzip.open(p,'rt'));origin=r['operational_contract']['gateway_origin_ns'];issues=[];wave_rows=[];ids=set();namespaces=set();caps=[]
 for w in r['waves']:
  ns=w['application_namespace'];assert ns not in namespaces;namespaces.add(ns)
  reqs=w['requests'];byindex={q['index']:q for q in reqs};capacity=[]
  for q in reqs:
   payload=q['payload'];params=payload['kv_transfer_params'];rid=payload['request_id'];assert rid not in ids;ids.add(rid)
   assert params['run_id']==ns and params['session_id']==q['session_id'] and q['session_id'].startswith(ns+'::')
   assert q['priority_origin_ns']==origin and q['priority_credit_ms']==8000
   assert payload['priority']==(q['priority_clock_ns']-origin)//1000000-(8000 if q['turn']>0 else 0)
   assert q['natural_eos'] and payload['max_tokens']==7680-len(payload['prompt']) and not payload.get('ignore_eos',False) and 'stop' not in payload
   if q['turn']:
    prev=byindex[q['index']-1];prefix=prev['payload']['prompt']+prev['output_token_ids'][:-1];prefix=prefix[:len(prefix)//16*16];assert payload['prompt'][:len(prefix)]==prefix
  for hint in w['hints']:
   h=hint['hint'];a=hint.get('response',{});assert h['run_id']==ns and h['session_id'].startswith(ns+'::')
   if a.get('reason')=='session_registry_capacity':
    assert a['status']=='no_action' and a['event_id']==h['event_id'] and a['session_id']==h['session_id']
    assert not any(a.get(k,0) for k in ('ready_cpu_prefix_keys','reserved_gpu_blocks','speculative_transfer_bytes'))
    capacity.append(hint);caps.append(hint)
  cap_sessions={h['hint']['session_id'] for h in capacity}
  row={'wave':w['wave_index'],'requests':len(reqs),'natural_EOS':sum(q['natural_eos'] for q in reqs),'application_namespace':ns,
       'actual_release_offset_s':(w['actual_wave_start_ns']-origin)/1e9,'normal_load_start_offset_s':(w['load_start_ns']-origin)/1e9,'normal_load_end_offset_s':(w['load_end_ns']-origin)/1e9,
       'capacity_ACKs':len(capacity),'capacity_distinct_sessions':len(cap_sessions),'EOS_requests_for_capacity_sessions':sum(q['natural_eos'] for q in reqs if q['session_id'] in cap_sessions),
       'hint_reasons':dict(collections.Counter(h.get('response',{}).get('reason') for h in w['hints']))}
  if item['phase']=='D':
   scalar=w['settled_export']['scheduler']['operational_scalar_state'];row['registry']=scalar['scheduler_lengths']['_observed_prefix_generations'];row['prefixes']=scalar['scheduler_lengths']['_observed_prefixes'];row['ACK_history']=scalar['scheduler_lengths']['_wait_hint_results']
   assert row['registry']<=128 and row['prefixes']<=1024 and row['ACK_history']<=4096
  wave_rows.append(row)
 if item['phase']=='D':
  assert [q['registry'] for q in wave_rows]==[24,48,72,96,120,128,128,128]
  assert sum(q['capacity_ACKs'] for q in wave_rows)==702
  assert all(q['hint_reasons'].keys()=={'session_registry_capacity'} and q['natural_EOS']==72 for q in wave_rows[-2:])
  overlap=None
 else:
  assert [w['intended_wave_release_ns']-origin for w in r['waves']]==[0,20000000000,40000000000]
  overlap=(min(w['load_end_ns'] for w in r['waves'])-max(w['load_start_ns'] for w in r['waves']))/1e9
  assert overlap>0 and all(w['immediate_export'] is None for w in r['waves'])
  assert r['overlap_settled_export']['scheduler']['operational_scalar_state']['native_request_count']==0
 assert r['operational_contract']['between_wave_cache_reset'] is False and r['operational_contract']['between_wave_counter_reset'] is False
 out['cases'][item['name']]={'raw_sha256':sha(p),'qualified':True,'waves':wave_rows,'three_normal_wave_intervals_overlap_s':overlap,'unique_external_request_ids':len(ids),'unique_application_namespaces':len(namespaces),'one_gateway_epoch':True,'natural_EOS':len(ids),'capacity_ACKs':len(caps),'scope':'Exact raw accounting; capacity128 only, finite overlap only. Semantic model namespace unchanged. No physical-prefill or unbounded-stability claim.'}
(H/'live-nov3-operational-v3-extra-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({n:{k:v[k] for k in ('qualified','natural_EOS','capacity_ACKs','three_normal_wave_intervals_overlap_s')} for n,v in out['cases'].items()}))
