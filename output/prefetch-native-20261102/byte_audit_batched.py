"""All-component byte audit with one host fence per transfer, never performance."""
import hashlib,json,os,time
from pathlib import Path
# Keep existing scheduler/adoption checks; CPU-only mode never stages a prefix.
from byte_audit_dynamic import patch_scheduler

def patch(module,directory):
 import numpy as np
 import torch
 cls=module.SingleDirectionOffloadingHandler;original=cls.transfer_async
 def audited(self,job_id,src_spec,dst_spec):
  assert self.src_blocks_per_chunk==self.dst_blocks_per_chunk==1
  assert len(self.kv_cache_groups_data_refs)==1
  src=list(map(int,src_spec.block_ids));dst=list(map(int,dst_spec.block_ids))
  assert len(src)==len(dst)>0
  assert not (Path(directory)/f'audit-layout-{job_id}.json').exists(), 'CPU-only qualification: no speculative GPU jobs'
  refs=self.kv_cache_groups_data_refs[0];expected=[];actual=[];owned=[]
  current=torch.cuda.current_stream()
  def snapshot_gpu(tensors,ids):
   index=torch.tensor(ids,device=tensors[0].device,dtype=torch.long)
   values=[]
   for ref in refs:
    # index_select owns the source before native transfer submission or the
    # destination after its fence, while native job ownership is still held.
    gathered=tensors[ref.tensor_idx][:,:ref.page_size_bytes].index_select(0,index)
    host=torch.empty(gathered.shape,dtype=gathered.dtype,device='cpu',pin_memory=True)
    host.copy_(gathered,non_blocking=True)
    owned.append(gathered);values.append(host)
   return values
  if self.gpu_to_cpu:
   expected=snapshot_gpu(self.src_tensors,src)
  else:
   expected=[self.src_tensors[ref.tensor_idx].numpy()[src,:ref.page_size_bytes].copy() for ref in refs]
  ok=original(self,job_id,src_spec,dst_spec)
  assert ok and job_id in self._transfer_events
  fence=self._transfer_events[job_id]
  if self.gpu_to_cpu:
   # Native D2H waits for current producer stream, including owned snapshots.
   fence.synchronize()
   actual=[self.dst_tensors[ref.tensor_idx].numpy()[dst,:ref.page_size_bytes].copy() for ref in refs]
  else:
   current.wait_event(fence)
   actual=snapshot_gpu(self.dst_tensors,dst)
   done=torch.cuda.Event();done.record(current);done.synchronize()
  records=[]
  for ref,left,right in zip(refs,expected,actual):
   a=left.numpy() if isinstance(left,torch.Tensor) else left
   b=right.numpy() if isinstance(right,torch.Tensor) else right
   equal=bool(np.array_equal(a,b));assert equal,f'actual KV transfer mismatch job={job_id} tensor={ref.tensor_idx}'
   bad=b.copy();bad.view(np.uint8).reshape(-1)[0]^=1
   assert not np.array_equal(a,bad),'invalid one-byte oracle failed'
   records.append({'tensor_idx':ref.tensor_idx,'page_size_bytes':ref.page_size_bytes,'bytes':a.nbytes,'equal':equal,'invalid_byte_rejected':True,'sha256':hashlib.sha256(memoryview(a)).hexdigest()})
  row={'at_ns':time.monotonic_ns(),'job_id':job_id,'direction':'D2H' if self.gpu_to_cpu else 'H2D','source_physical_ids':src,'destination_physical_ids':dst,'components':records,'total_bytes':sum(x['bytes'] for x in records),'intrusive_synchronization':True,'local_prefix_checked_ids':[],'local_prefix_bytes_checked':0,'audit_implementation':'batched-owned-snapshots'}
  with (Path(directory)/f'byte-audit-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return ok
 cls.transfer_async=audited
