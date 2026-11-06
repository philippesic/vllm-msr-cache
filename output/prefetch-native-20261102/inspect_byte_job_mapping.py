import fcntl,gzip,json
from collections import Counter
from pathlib import Path
from campaign_metrics_warm import terminal_index,nested
H=Path(__file__).resolve().parent
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 d=H/'runs/cpu-recency-v2-byte-qualified-c24-b2048-candidate-s1103-v2';raw=json.load(gzip.open(d/'raw.json.gz'));idx=terminal_index(raw)
 audits={a['job_id']:a for f in d.glob('byte-audit-*.jsonl') for line in f.read_text().splitlines() if line for a in [json.loads(line)]};seen=set();mismatch=[]
 for r in raw['requests']:
  for j in idx[nested(r,'payload.request_id')][0]['transfer_job_statuses']:
   if j['job_id'] in seen:continue
   seen.add(j['job_id']);a=audits[j['job_id']]
   fields=[('source_block_ids','source_physical_ids'),('destination_block_ids','destination_physical_ids')]
   wrong=[p for p,q in fields if [i[1] for i in j[p]]!=a[q]]
   if wrong:
    src=[i[1] for i in j['source_block_ids']];dst=[i[1] for i in j['destination_block_ids']]
    mismatch.append({'job_id':j['job_id'],'direction':a['direction'],'wrong_fields':wrong,'source_sets_equal':Counter(src)==Counter(a['source_physical_ids']),'destination_sets_equal':Counter(dst)==Counter(a['destination_physical_ids']),'paired_mapping_equal':Counter(zip(src,dst))==Counter(zip(a['source_physical_ids'],a['destination_physical_ids'])),'planned_source_first':src[:8],'planned_destination_first':dst[:8],'actual_source_first':a['source_physical_ids'][:8],'actual_destination_first':a['destination_physical_ids'][:8]})
    if len(mismatch)==1:print('first_full_job',json.dumps(j)[:2400])
 out={'unique_serving_jobs':len(seen),'mismatched_order_jobs':len(mismatch),'all_source_sets_equal':all(m['source_sets_equal'] for m in mismatch),'all_destination_sets_equal':all(m['destination_sets_equal'] for m in mismatch),'all_paired_mappings_equal':all(m['paired_mapping_equal'] for m in mismatch),'examples':mismatch[:3]}
 (H/'qualification/byte-job-mapping-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
