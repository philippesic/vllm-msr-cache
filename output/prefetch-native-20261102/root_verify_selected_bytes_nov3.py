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
manifest_path=E/'triage-confirmation-frozen.json';manifest=json.loads(manifest_path.read_text())
for name,pin in manifest['file_pins'].items():
 p=Path(name);p=p if p.is_absolute() else ROOT/p;assert sha(p)==pin,('pin drift',name)
receipts={}
for arm in ['native','candidate']:
 receipt_path=H/f'live-nov3-selected-g1536-c2048-byte-{arm}-qualification.json';receipt=json.loads(receipt_path.read_text());rawpath=Path(receipt['raw_path']);assert sha(rawpath)==receipt['raw_sha256'];raw=json.load(gzip.open(rawpath));assert raw['performance_eligible'] is False and raw['status']=='complete' and len(raw['requests'])==72 and all(q['success'] and q['natural_eos'] for q in raw['requests'])
 assert raw['args']['gpu_blocks']==1536 and raw['args']['cpu_blocks']==2048 and raw['args']['active']==8 and raw['args']['batch_tokens']==1024
 assert raw['args']['service_profile']=='repository_triage_v2' and raw['args']['audit'] and raw['args']['graphs']
 assert raw['http_client_configuration']['force_close'] and not raw['owned_descendants_remaining']
 assert not any(x['event'] in ['connection_reuse','request_exception'] for x in raw['http_connection_events'])
 for name,pin in raw['source_pins'].items():assert manifest['file_pins'][str((H/'sources/nov3-hardened'/name).relative_to(ROOT))]==pin
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
for item in manifest['execution_order']:assert not (H/'runs'/item['name']).exists(),('fresh planned output already exists',item['name'])
# Every prior invocation of the only runner accepting a custom corpus used old inputs.
prior_custom_runs=0
for p in (H/'runs').glob('live-nov3-triage-*/raw.json.gz'):
 raw=json.load(gzip.open(p));cp=raw.get('args',{}).get('corpus_path')
 if cp:
  assert Path(cp).resolve()!=(H/'fresh-agent-nov3/session-corpus.json').resolve(),p;prior_custom_runs+=1
out={'schema':'root-selected-bytes-and-launch-gate-v1','gate_open':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':str(manifest_path.relative_to(ROOT)),'manifest_sha256':sha(manifest_path),'all205_file_pins_match':True,'both_selected_arm_qualifications':receipts,'all_fresh_planned_names_absent':True,'prior_custom_runner_runs_checked':prior_custom_runs,'fresh_corpus_previously_generated':False,'limitations':['Intrusive exercised transfer integrity, not performance/answer-quality/endurance certification.','Historical pre-header disconnect cause remains a hypothesis; new common client prohibits socket reuse and has no retries.'],'root_verifier':str(Path(__file__).relative_to(ROOT)),'root_verifier_sha256':sha(Path(__file__))}
(E/'triage-confirmation-launch-gate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'gate_open':True,'native_jobs':receipts['native']['counts'],'candidate_jobs':receipts['candidate']['counts'],'total_component_checks':sum(r['component_comparisons_and_corruption_controls'] for r in receipts.values()),'prior_custom_runs_checked':prior_custom_runs}))
