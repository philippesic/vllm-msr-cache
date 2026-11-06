"""Independent root verification of all selected-geometry audit rows and native counters."""
from pathlib import Path
import json,gzip,hashlib,collections,re,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=ROOT/'evidence/live-optimization-2080ti-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(text,name,direction):
 values=[]
 for line in text.splitlines():
  if line.startswith(name+'{') and 'transfer_type="'+direction+'"' in line:values.append(float(line.rsplit(' ',1)[1]))
 assert values,('missing metric',name,direction)
 return sum(values)
manifest_path=H/'return-priority-bytes-frozen-plan-nov3.json';manifest=json.loads(manifest_path.read_text())
for name,pin in manifest['file_pins'].items():
 p=Path(name);p=p if p.is_absolute() else ROOT/p;assert sha(p)==pin,('pin drift',name)
receipts={}
for arm in ['native','candidate']:
 receipt_path=H/f'live-nov3-return-priority-byte-{arm}-qualification.json';receipt=json.loads(receipt_path.read_text());rawpath=Path(receipt['raw_path']);assert sha(rawpath)==receipt['raw_sha256'];raw=json.load(gzip.open(rawpath));assert raw['performance_eligible'] is False and raw['status']=='complete' and len(raw['requests'])==72 and all(q['success'] and q['natural_eos'] for q in raw['requests'])
 assert raw['args']['gpu_blocks']==1536 and raw['args']['cpu_blocks']==2048 and raw['args']['active']==8 and raw['args']['batch_tokens']==1024
 assert raw['args']['service_profile']=='repository_triage_v2' and raw['args']['audit'] and raw['args']['graphs']
 assert raw['http_client_configuration']['force_close'] and not raw['owned_descendants_remaining']
 assert raw['args']['return_priority_ms']==8000 and raw['server_command'][raw['server_command'].index('--scheduling-policy')+1]=='priority'
 for q in raw['requests']:
  assert q['payload']['priority']==(q['priority_clock_ns']-q['priority_origin_ns'])//1000000-(8000 if q['turn'] else 0) and q['priority_origin_ns']==raw['load_start_ns'] and q['priority_clock_ns']==q['request_start_ns']
 assert all(q['payload']['max_tokens']==7680-len(q['payload']['prompt']) and not q['payload'].get('ignore_eos',False) and not q['payload'].get('stop') and q['finish_reasons']==['stop'] and q['terminal_complete'] for q in raw['requests'])
 assert not any(x['event'] in ['connection_reuse','request_exception'] for x in raw['http_connection_events'])
 source=H/'sources'/('nov3-gpu-gate-integrated' if arm=='candidate' else 'nov3-hardened')
 for name,pin in raw['source_pins'].items():assert manifest['file_pins'][str((source/name).relative_to(ROOT))]==pin
 if arm=='candidate':
  assert raw['args']['no_audit'] and raw['config']['kv_connector_extra_config']['preservation_audit'] is False and raw['config']['kv_connector_extra_config']['wait_hint_cpu_retention_gpu_gate'] is True
 for name,pin in raw['runtime_pins'].items():assert manifest['file_pins'][str((H/name).relative_to(ROOT))]==pin
 all_counts=collections.Counter();all_bytes=collections.Counter();load_counts=collections.Counter();load_bytes=collections.Counter();seen=set();components=0
 for entry in receipt['audit_files']:
  p=Path(entry['path']);assert sha(p)==entry['sha256']
  with gzip.open(p,'rt') as f:
   for line in f:
    row=json.loads(line);direction=row['direction'];assert direction in ['H2D','D2H'];identity=(row['at_ns'],row['job_id'],direction);assert identity not in seen;seen.add(identity)
    n=len(row['source_physical_ids']);assert n==len(row['destination_physical_ids']) and n>0
    assert row['intrusive_synchronization'] is True and len(row['components'])==28
    assert {c['tensor_idx'] for c in row['components']}==set(range(28))
    for c in row['components']:
     assert c['page_size_bytes']==32768 and c['bytes']==n*32768 and c['equal'] is True and c['invalid_byte_rejected'] is True
     assert re.fullmatch('[0-9a-f]{64}',c['sha256']);components+=1
    assert row['total_bytes']==sum(c['bytes'] for c in row['components'])==n*917504
    all_counts[direction]+=1;all_bytes[direction]+=row['total_bytes']
    if row['at_ns']>=raw['load_start_ns']:load_counts[direction]+=1;load_bytes[direction]+=row['total_bytes']
 assert all_counts['D2H']>0 and all_counts['H2D']>0
 for direction,label in [('D2H','GPU_to_CPU'),('H2D','CPU_to_GPU')]:
  for name,total,load in [('vllm:kv_offload_total_bytes_total',all_bytes[direction],load_bytes[direction]),('vllm:kv_offload_size_count',all_counts[direction],load_counts[direction])]:
   final=metric(raw['metrics_final'],name,label);initial=metric(raw['metrics_initial'],name,label)
   assert final==total and final-initial==load,(direction,name,initial,final,total,load)
 receipts[arm]={'run':raw['name'],'qualification_receipt':str(receipt_path.relative_to(ROOT)),'qualification_receipt_sha256':sha(receipt_path),'counts':dict(all_counts),'bytes':dict(all_bytes),'component_comparisons_and_corruption_controls':components,'all_rows_and_aggregate_reconciled':True,'per_job_native_Transfer_num_bytes_observed':False,'scope':'Mapped pages × independently pinned geometry and native aggregate counters; not an extra observed per-job native byte field.'}
raw=json.load(gzip.open(H/'runs/live-nov3-return-priority-byte-candidate/raw.json.gz'))
valid=[h['response'] for h in raw['hints'] if h.get('response',{}).get('reason')=='cpu_prefix_recency_refreshed' and not h['response'].get('idempotent_replay')]
skips=[a for a in valid if a['cpu_retention_skipped_full_gpu']];fallbacks=[a for a in valid if not a['cpu_retention_skipped_full_gpu']]
assert skips and fallbacks and all(a['ready_cpu_prefix_keys']==0 and a['cpu_retention_gpu_cached_prefix_keys']==a['cpu_retention_source_keys'] for a in skips)
assert any(a['cpu_retention_ready_prefix_keys']>0 for a in skips) and any(a['ready_cpu_prefix_keys']>0 for a in fallbacks)
for arm,expected in [('native',1),('candidate',0)]:
 r=json.load(gzip.open(H/('runs/live-nov3-return-priority-byte-'+arm)/'raw.json.gz'))
 values=lambda text:[float(line.rsplit(' ',1)[1]) for line in text.splitlines() if line.startswith('vllm:num_preemptions_total{') or line.startswith('vllm:num_preemptions_total ')]
 assert sum(values(r['metrics_final']))-sum(values(r['metrics_initial']))==expected
out={'schema':'root-independent-return-priority-bytes-verification-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all414pins_verified':True,'both_selected_arm_qualifications':receipts,'all_rows_and_final_load_native_counters_reconciled':True,'candidate_bounded_preservation_verified':True,'actual_gate_activation':{'valid_ACKs':len(valid),'full_GPU_skips':len(skips),'partial_fallbacks':len(fallbacks),'actual_CPU_refreshes':sum(a['ready_cpu_prefix_keys'] for a in valid)},'total_components':sum(r['component_comparisons_and_corruption_controls'] for r in receipts.values()),'observed_preemptions':{'native':1,'candidate':0},'preemption_scope':'Native intrusive run counter1; cannot attribute per-job H2D specifically to that preemption; no performance claim.','scope':'Every exercised mapped D2H/H2D pagecomponent plus onebyte comparator control. Intrusive qualification is not performance, quality or endurance. Per-job native Transfer.num_bytes not independently observed.'}
(E/'return-priority-bytes-root-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'all414pins':True,'components':out['total_components'],'arm_counts':{a:r['counts'] for a,r in receipts.items()}},indent=2))
