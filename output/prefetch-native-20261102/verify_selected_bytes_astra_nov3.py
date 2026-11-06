"""Independent all-component verification; no GPU/model imports."""
import collections,gzip,hashlib,json
from pathlib import Path
from analyze_confirmation_astra_nov3 import counter,delta
from http_receipt_astra_nov3 import verify_http

def verify_bytes(raw_path):
 raw_path=Path(raw_path);r=json.load(gzip.open(raw_path,'rt'));issues=[];files=sorted(raw_path.parent.glob('byte-audit-*.jsonl'))+sorted(raw_path.parent.glob('byte-audit-*.jsonl.gz'));rows=[];file_receipts=[]
 for f in files:
  opener=gzip.open if f.suffix=='.gz' else open
  with opener(f,'rt') as stream:
   for line_no,line in enumerate(stream,1):
    if line.strip():rows.append((f.name.removesuffix('.gz'),line_no,json.loads(line)))
  file_receipts.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 if not rows:issues.append('missing_byte_audit_rows')
 if r.get('performance_eligible') is not False or r.get('args',{}).get('audit') is not True:issues.append('not_explicitly_intrusive_ineligible')
 req=r.get('requests',[])
 if r.get('status')!='complete' or len(req)!=72 or not all(q.get('success') and q.get('natural_eos') for q in req):issues.append('incomplete_natural_EOS_workflow')
 http=verify_http(r)
 if not http['qualified']:issues.append('HTTP_unqualified')
 geometry=r['geometry'];expected_block_bytes=geometry['block_bytes'];directions=collections.Counter();bytedirs=collections.Counter();jobs=set();bad=[];inload=collections.Counter();inloadbytes=collections.Counter()
 for file,line,row in rows:
  errs=[];key=(file,row.get('job_id'));src=row.get('source_physical_ids',[]);dst=row.get('destination_physical_ids',[]);components=row.get('components',[]);direction=row.get('direction')
  if key in jobs:errs.append('duplicate_job_identity')
  jobs.add(key)
  if direction not in ['D2H','H2D']:errs.append('unknown_direction')
  if not src or len(src)!=len(dst):errs.append('invalid_block_mapping')
  if len(components)!=28 or {v.get('tensor_idx') for v in components}!=set(range(28)):errs.append('missing_or_duplicate_tensor_component')
  if any(v.get('equal') is not True or v.get('invalid_byte_rejected') is not True for v in components):errs.append('byte_or_corruption_control_failure')
  if any(v.get('page_size_bytes')!=32768 or v.get('bytes')!=len(src)*v.get('page_size_bytes',-1) for v in components):errs.append('component_page_byte_mismatch')
  if sum(v.get('page_size_bytes',0) for v in components)!=expected_block_bytes:errs.append('geometry_page_size_mismatch')
  expected_bytes=len(src)*expected_block_bytes
  if row.get('total_bytes')!=expected_bytes or sum(v.get('bytes',0) for v in components)!=expected_bytes:errs.append('job_geometry_byte_mismatch')
  if row.get('intrusive_synchronization') is not True:errs.append('missing_sync_receipt')
  src_limit=geometry['gpu_blocks'] if direction=='D2H' else geometry['cpu_blocks'];dst_limit=geometry['cpu_blocks'] if direction=='D2H' else geometry['gpu_blocks']
  if any(not isinstance(v,int) or v<0 or v>=src_limit for v in src) or any(not isinstance(v,int) or v<0 or v>=dst_limit for v in dst):errs.append('physical_ID_out_of_range')
  if errs:bad.append({'file':file,'line':line,'job_id':row.get('job_id'),'errors':errs})
  directions[direction]+=1;bytedirs[direction]+=row.get('total_bytes',0)
  if row.get('at_ns',0)>=r.get('load_start_ns',0):inload[direction]+=1;inloadbytes[direction]+=row.get('total_bytes',0)
 if bad:issues.append('invalid_audit_rows')
 if not all(directions.get(d,0)>0 and inload.get(d,0)>0 for d in ['D2H','H2D']):issues.append('both_directions_not_exercised_during_load')
 prometheus={}
 for d,label in [('D2H','GPU_to_CPU'),('H2D','CPU_to_GPU')]:
  metricbytes=counter(r.get('metrics_final',''),'vllm:kv_offload_total_bytes_total',f'transfer_type="{label}"');metriccount=counter(r.get('metrics_final',''),'vllm:kv_offload_size_count',f'transfer_type="{label}"')
  prometheus[d]={'final_total_bytes':metricbytes,'final_job_count':metriccount,'all_audit_bytes':bytedirs.get(d),'all_audit_jobs':directions.get(d),'bytes_match':metricbytes==bytedirs.get(d) if metricbytes is not None else None,'counts_match':metriccount==directions.get(d) if metriccount is not None else None,'load_delta_bytes':delta(r,'vllm:kv_offload_total_bytes_total',f'transfer_type="{label}"'),'load_delta_job_count':delta(r,'vllm:kv_offload_size_count',f'transfer_type="{label}"'),'audit_jobs_at_or_after_load_start':inload.get(d),'audit_bytes_at_or_after_load_start':inloadbytes.get(d)}
  if prometheus[d]['bytes_match'] is False or prometheus[d]['counts_match'] is False:issues.append('native_Prometheus_aggregate_mismatch_'+d)
 counts=r.get('postload_export',{}).get('scheduler',{}).get('counts',{})
 if counts.get('store_failed_keys',0)>0:issues.append('explicit_failed_store_keys')
 if 'store_accepted_keys' not in counts or 'store_completed_keys' not in counts or counts['store_accepted_keys']!=counts['store_completed_keys']:issues.append('store_completion_not_reconciled')
 first={s:delta(r,'vllm:prompt_tokens_by_source_total',f'source="{s}"') for s in ['local_compute','local_cache_hit','external_kv_transfer']}
 if first['external_kv_transfer'] is None or first['external_kv_transfer']<=0:issues.append('first_admission_restoration_not_evidenced')
 return {'schema':'astra-selected-geometry-byte-qualification-v1','qualified':not issues,'issues':issues,'run':r['name'],'raw_path':str(raw_path),'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'performance_eligible':r.get('performance_eligible'),'requests':len(req),'natural_eos':sum(bool(q.get('natural_eos')) for q in req),'geometry':geometry,'audit_files':file_receipts,'directions':dict(directions),'bytes_by_direction':dict(bytedirs),'components_per_job':28,'component_page_bytes':32768,'component_observations':28*len(rows),'all_equal_and_corruption_controls_rejected':not bad and bool(rows),'bad_rows':bad,'native_aggregate_reconciliation':prometheus,'native_per_job_num_bytes_observed':False,'per_job_reconciliation_scope':'Each componentbytes equals mappedblocks*32768;28components sum to mappedblocks*pinned917504-byte geometry and auditrow total_bytes. Pinned native single-group1:1 handler calculates Transfer.num_bytes with the same page-sum formula. Native per-job num_bytes field is not independently recorded; Prometheus whole-run aggregates provide separate observed totals.','preemptions':delta(r,'vllm:num_preemptions_total'),'first_admission_attribution':first,'store_accepted_keys':counts.get('store_accepted_keys'),'store_completed_keys':counts.get('store_completed_keys'),'explicit_store_failed_keys':counts.get('store_failed_keys'),'missing_counter_policy':'Unavailable counters remain null; absent failurekey alone is not zero proof.','HTTP_receipt':http,'owned_descendants_remaining':r.get('owned_descendants_remaining'),'pins':{k:r.get(k) for k in ['source_pins','runtime_pins','pins','harness_pins','artifact_pins','native_scheduler_source','model_identity']},'limits':['Intrusive owned-source snapshots/readbacks/synchronization alter scheduling; never performance evidence.','Copies checked at transfer completion, not logical semantic token boundary or all futureadoption lifetimes. Finished-token clamp and ownership tests complement byte checks.','No answer-quality/task-success or endurance claim.','Qualification limited to pinnedTP1dense28layerFP16AWQ geometry and exercised directions; heldoutinputs unconsumed.']}
