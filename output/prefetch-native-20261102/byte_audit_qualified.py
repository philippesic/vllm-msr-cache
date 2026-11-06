"""Intrusive real-transfer byte qualification. Never enable for performance."""
import hashlib,json,os,time
from pathlib import Path

def patch(module,directory):
 import torch
 cls=module.SingleDirectionOffloadingHandler;original=cls.transfer_async
 def audited(self,job_id,src_spec,dst_spec):
  assert self.src_blocks_per_chunk==self.dst_blocks_per_chunk==1
  assert len(self.kv_cache_groups_data_refs)==1,'audit only qualified dense single group'
  src=list(map(int,src_spec.block_ids));dst=list(map(int,dst_spec.block_ids));assert len(src)==len(dst)
  expected=[]
  # Source ownership is still held here. The clone is enqueued on the current
  # producer stream, which the native D2H transfer waits on before consuming KV.
  for ref in self.kv_cache_groups_data_refs[0]:
   expected.append((ref,self.src_tensors[ref.tensor_idx][src,:ref.page_size_bytes].clone()))
  local_ids=tuple(int(x) for x in os.getenv('REAL_KV_LOCAL_PREFIX_IDS','').split(',') if x) if not self.gpu_to_cpu else ()
  local=[]
  if local_ids:
   assert set(local_ids).isdisjoint(dst),'suffix destination overlaps shared prefix'
   for ref in self.kv_cache_groups_data_refs[0]:
    local.append((ref,self.dst_tensors[ref.tensor_idx][list(local_ids),:ref.page_size_bytes].clone()))
  ok=original(self,job_id,src_spec,dst_spec)
  assert ok and job_id in self._transfer_events
  self._transfer_events[job_id].synchronize()
  records=[]
  for ref,owned in expected:
   a=owned.cpu();b=self.dst_tensors[ref.tensor_idx][dst,:ref.page_size_bytes].cpu().clone()
   equal=bool(torch.equal(a,b));assert equal,f'actual KV transfer mismatch job={job_id} tensor={ref.tensor_idx}'
   bad=b.clone();bad.view(-1)[0]^=1
   assert not torch.equal(a,bad),'invalid one-byte oracle failed'
   records.append({'tensor_idx':ref.tensor_idx,'page_size_bytes':ref.page_size_bytes,'bytes':a.numel(),'equal':equal,'invalid_byte_rejected':True,'sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest()})
  if local_ids:
   owned=[]
   for ref,value in local:
    expected_local=value.cpu();actual_local=self.dst_tensors[ref.tensor_idx][list(local_ids),:ref.page_size_bytes].cpu().clone()
    assert torch.equal(expected_local,actual_local),'H2D suffix modified shared local prefix'
    owned.append((ref,expected_local.clone()))
   LOCAL_SNAPSHOTS[(job_id,local_ids)]={'job_id':job_id,'components':owned,'gpu_tensors':self.dst_tensors}
  row={'at_ns':time.monotonic_ns(),'job_id':job_id,'direction':'D2H' if self.gpu_to_cpu else 'H2D','source_physical_ids':src,'destination_physical_ids':dst,'components':records,'total_bytes':sum(x['bytes'] for x in records),'intrusive_synchronization':True,'local_prefix_checked_ids':list(local_ids),'local_prefix_bytes_checked':sum(ref.page_size_bytes*len(local_ids) for ref,_ in local)}
  with (Path(directory)/f'byte-audit-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return ok
 cls.transfer_async=audited

LOCAL_SNAPSHOTS={}
def verify_local_prefix_on_adoption(prefix_ids,request_id,directory,source_job_id):
 """Actual local pages are protected by native positive request references.
 Expected bytes are owned pre-transfer snapshots, never late source reads.
 """
 import torch
 ids=tuple(prefix_ids);entry=LOCAL_SNAPSHOTS.pop((source_job_id,ids),None)
 assert entry is not None,'missing owned local-prefix snapshot for native adoption'
 torch.cuda.synchronize();components=[]
 for ref,expected in entry['components']:
  actual=entry['gpu_tensors'][ref.tensor_idx][list(ids),:ref.page_size_bytes].cpu().clone()
  assert torch.equal(expected,actual),'local prefix KV mutated across suffix publication/adoption'
  wrong=actual.clone();wrong.view(-1)[0]^=1
  assert not torch.equal(expected,wrong),'local prefix 1-byte wrong control failed'
  components.append({'tensor_idx':ref.tensor_idx,'bytes':actual.numel(),'equal':True,'invalid_byte_rejected':True,'sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
 row={'event':'local_prefix_bytes_unchanged_at_native_adoption','at_ns':time.monotonic_ns(),'request_id':request_id,'prefix_physical_ids':list(ids),'source_h2d_job_id':entry['job_id'],'components':components,'total_bytes':sum(x['bytes'] for x in components),'intrusive_synchronization':True}
 with (Path(directory)/f'local-prefix-audit-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
