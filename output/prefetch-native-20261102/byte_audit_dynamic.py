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
  layout_path=Path(directory)/f'audit-layout-{job_id}.json'
  layout=json.loads(layout_path.read_text()) if not self.gpu_to_cpu and layout_path.exists() else None
  if layout:
   assert layout['job_id']==job_id and layout['suffix_ids']==dst
  local_ids=tuple(layout['local_ids']) if layout else ()
  local=[]
  if local_ids:
   assert set(local_ids).isdisjoint(dst),'suffix destination overlaps shared prefix'
   for ref in self.kv_cache_groups_data_refs[0]:
    local.append((ref,self.dst_tensors[ref.tensor_idx][list(local_ids),:ref.page_size_bytes].clone()))
  ok=original(self,job_id,src_spec,dst_spec)
  assert ok and job_id in self._transfer_events
  self._transfer_events[job_id].synchronize()
  records=[];cpu_expected=[]
  for ref,owned in expected:
   a=owned.cpu();b=self.dst_tensors[ref.tensor_idx][dst,:ref.page_size_bytes].cpu().clone()
   equal=bool(torch.equal(a,b));assert equal,f'actual KV transfer mismatch job={job_id} tensor={ref.tensor_idx}'
   bad=b.clone();bad.view(-1)[0]^=1
   assert not torch.equal(a,bad),'invalid one-byte oracle failed'
   cpu_expected.append((ref,a.clone()))
   records.append({'tensor_idx':ref.tensor_idx,'page_size_bytes':ref.page_size_bytes,'bytes':a.numel(),'equal':equal,'invalid_byte_rejected':True,'sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest()})
  if local_ids:
   owned=[]
   for ref,value in local:
    expected_local=value.cpu();actual_local=self.dst_tensors[ref.tensor_idx][list(local_ids),:ref.page_size_bytes].cpu().clone()
    assert torch.equal(expected_local,actual_local),'H2D suffix modified shared local prefix'
    owned.append((ref,expected_local.clone()))
   LOCAL_SNAPSHOTS[(job_id,local_ids)]={'job_id':job_id,'components':owned,'gpu_tensors':self.dst_tensors}
  if layout:
   ADOPTION_SNAPSHOTS[job_id]={'local_ids':local_ids,'local_components':[(ref,value.cpu().clone()) for ref,value in local],'suffix_ids':dst,'suffix_components':cpu_expected,'gpu_tensors':self.dst_tensors}
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


ADOPTION_SNAPSHOTS={}
def verify_native_prefix_bytes(prefix_ids,local_count,request_id,directory,source_job_id):
 import torch
 entry=ADOPTION_SNAPSHOTS.pop(source_job_id,None)
 assert entry is not None,'missing owned per-job transfer snapshots'
 assert local_count==len(entry['local_ids'])
 assert list(prefix_ids[local_count:])==entry['suffix_ids']
 assert len(prefix_ids)==local_count+len(entry['suffix_ids'])
 torch.cuda.synchronize();components=[]
 for label,ids,owned in [('local',prefix_ids[:local_count],entry['local_components']),
                          ('suffix',prefix_ids[local_count:],entry['suffix_components'])]:
  for ref,expected in owned:
   actual=entry['gpu_tensors'][ref.tensor_idx][list(ids),:ref.page_size_bytes].cpu().clone()
   assert torch.equal(expected,actual),label+' KV bytes changed before native adoption'
   wrong=actual.clone();wrong.view(-1)[0]^=1
   assert not torch.equal(expected,wrong),'one-byte wrong control was not rejected'
   components.append({'region':label,'tensor_idx':ref.tensor_idx,'bytes':actual.numel(),'equal':True,'invalid_byte_rejected':True,'sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
 row={'event':'owned_prefix_and_suffix_bytes_equal_at_native_adoption','request_id':request_id,
      'source_h2d_job_id':source_job_id,'prefix_physical_ids':list(prefix_ids),'local_count':local_count,
      'components':components,'total_bytes':sum(c['bytes'] for c in components),'intrusive_synchronization':True}
 with (Path(directory)/f'native-adoption-byte-audit-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

def patch_scheduler(cls,directory):
 original_attempt=cls._attempt_wait_prefetch
 def attempt(self,state,*args,**kwargs):
  result=original_attempt(self,state,*args,**kwargs)
  if result:
   assert state.job_id is not None and state.gpu_block_ids and state.pending_contributors>0
   row={'job_id':state.job_id,'event_id':state.lease.hint.event_id,
        'local_ids':list(state.publication_local_gpu_block_ids),'suffix_ids':list(state.gpu_block_ids)}
   path=Path(directory)/f'audit-layout-{state.job_id}.json';assert not path.exists()
   path.write_text(json.dumps(row)+'\n')
  return result
 cls._attempt_wait_prefetch=attempt
 original_publish=cls._try_publish_wait_prefix
 def publish(self,state,**kwargs):
  job_id=state.job_id;result=original_publish(self,state,**kwargs)
  if result:
   record=self._published_wait_prefixes[state.bound_request_id]
   assert record['event_id']==state.lease.hint.event_id
   assert record['block_ids'][record.get('local_prefix_count',0):]==list(state.gpu_block_ids)
   record['_audit_transfer_job_id']=job_id
  return result
 cls._try_publish_wait_prefix=publish
 original_adoption=cls._record_native_prefix_adoption
 def adoption(self,request,blocks,num_external_tokens):
  published=getattr(self,'_published_wait_prefixes',{}).get(request.request_id)
  result=original_adoption(self,request,blocks,num_external_tokens)
  proof=getattr(self,'_preservation_requests',{}).get(request.request_id,{})
  if published and proof.get('prefetch_native_adoption',{}).get('same_physical_blocks') is True:
   verify_native_prefix_bytes(published['block_ids'],published.get('local_prefix_count',0),
                              request.request_id,directory,published['_audit_transfer_job_id'])
  return result
 cls._record_native_prefix_adoption=adoption
