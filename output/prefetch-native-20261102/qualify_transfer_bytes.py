"""Require actual all-component byte checks for every serving transfer job."""
import argparse,fcntl,gzip,json
from pathlib import Path
from campaign_metrics_warm import terminal_index,nested
H=Path(__file__).resolve().parent
def validate(raw,summary,audits):
 assert raw['status']=='complete' and raw['performance_eligible'] is False
 assert summary['valid'] and summary['proofs']['all_passed']
 assert summary['all']['requests']==summary['proofs']['passed']==raw['planned_requests']==72
 block_bytes=raw['geometry']['block_bytes'];index={}
 for a in audits:
  key=a['job_id'];assert key not in index,'duplicate byte-audit job'
  index[key]=a;components=a['components']
  assert len(components)==28 and len({c['tensor_idx'] for c in components})==28
  assert sum(c['page_size_bytes'] for c in components)==block_bytes
  assert len(a['source_physical_ids'])==len(a['destination_physical_ids'])>0
  assert a['total_bytes']==len(a['source_physical_ids'])*block_bytes
  assert all(c['equal'] is True and c['invalid_byte_rejected'] is True
             and c['bytes']==len(a['source_physical_ids'])*c['page_size_bytes'] for c in components)
 jobs={};terminals=terminal_index(raw)
 for r in raw['requests']:
  ts=terminals.get(nested(r,'payload.request_id'),[]);assert len(ts)==1
  for job in ts[0]['transfer_job_statuses']:
   old=jobs.get(job['job_id']);assert old is None or old==job
   jobs[job['job_id']]=job
 assert jobs
 for job_id,j in jobs.items():
  assert j['success'] is True and j['cancelled'] is False and j['contributors']==j['expected_contributors']==1
  assert job_id in index,'serving transfer has no actual byte audit'
  a=index[job_id];assert a['direction']==('H2D' if j['direction']=='load' else 'D2H')
  assert a['total_bytes']==j['bytes']
  for field,physical in [('source_block_ids','source_physical_ids'),('destination_block_ids','destination_physical_ids')]:
   if j.get(field) is not None:
    assert [i[1] for i in j[field]]==a[physical]
 directions={index[j]['direction'] for j in jobs};assert directions=={'H2D','D2H'}
 return dict(passed=True,strict_requests=72,serving_transfer_jobs=len(jobs),
  serving_bytes_checked=sum(index[j]['total_bytes'] for j in jobs),all_kv_components_per_job=28,
  invalid_byte_rejected_for_every_component=True,all_serving_jobs_covered=True,
  audit_jobs_including_warmup=len(index),intrusive_not_performance=True,
  speculative_gpu_copies=summary['offline_staging_utility']['global_explicit_successful_completion_bytes_observed'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('name');args=ap.parse_args()
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  d=H/'runs'/args.name;raw=json.load(gzip.open(d/'raw.json.gz'));summary=json.loads((d/'summary.json').read_text())
  audits=[json.loads(line) for f in d.glob('byte-audit-*.jsonl') for line in f.read_text().splitlines() if line]
  result=validate(raw,summary,audits);(d/'qualification-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
