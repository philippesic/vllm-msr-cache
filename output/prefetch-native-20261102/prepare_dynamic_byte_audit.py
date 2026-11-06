"""Intrusive qualification for exact per-job local heads and copied suffixes."""
import ast,shutil
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'byte_audit_qualified.py').read_text()
old="  local_ids=tuple(int(x) for x in os.getenv('REAL_KV_LOCAL_PREFIX_IDS','').split(',') if x) if not self.gpu_to_cpu else ()"
new='''  layout_path=Path(directory)/f'audit-layout-{job_id}.json'
  layout=json.loads(layout_path.read_text()) if not self.gpu_to_cpu and layout_path.exists() else None
  if layout:
   assert layout['job_id']==job_id and layout['suffix_ids']==dst
  local_ids=tuple(layout['local_ids']) if layout else ()'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('  records=[]\n','  records=[];cpu_expected=[]\n')
s=s.replace("   records.append({'tensor_idx':ref.tensor_idx", "   cpu_expected.append((ref,a.clone()))\n   records.append({'tensor_idx':ref.tensor_idx")
needle="  row={'at_ns':time.monotonic_ns(),'job_id':job_id"
assert s.count(needle)==1
s=s.replace(needle,"  if layout:\n   ADOPTION_SNAPSHOTS[job_id]={'local_ids':local_ids,'local_components':[(ref,value.cpu().clone()) for ref,value in local],'suffix_ids':dst,'suffix_components':cpu_expected,'gpu_tensors':self.dst_tensors}\n"+needle)
s+='''

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
 with (Path(directory)/f'native-adoption-byte-audit-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')

def patch_scheduler(cls,directory):
 original_attempt=cls._attempt_wait_prefetch
 def attempt(self,state,*args,**kwargs):
  result=original_attempt(self,state,*args,**kwargs)
  if result:
   assert state.job_id is not None and state.gpu_block_ids and state.pending_contributors>0
   row={'job_id':state.job_id,'event_id':state.lease.hint.event_id,
        'local_ids':list(state.publication_local_gpu_block_ids),'suffix_ids':list(state.gpu_block_ids)}
   path=Path(directory)/f'audit-layout-{state.job_id}.json';assert not path.exists()
   path.write_text(json.dumps(row)+'\\n')
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
'''
ast.parse(s);(H/'byte_audit_dynamic.py').write_text(s)
target=H/'hooks-dynamic-byte';shutil.copytree(H/'hooks-teacher-replay',target,dirs_exist_ok=True)
f=target/'sitecustomize.py';s=f.read_text();s=s.replace('        import byte_audit\n','        import byte_audit_dynamic as byte_audit\n')
start=s.index("        if os.getenv('REAL_KV_BYTE_AUDIT')=='1':",s.index("elif n=='semantic_offload.connector':"));end=s.index("        if os.getenv('MATRIX_BOUNDARY_DIAGNOSTIC')",start)
s=s[:start]+"        if os.getenv('REAL_KV_BYTE_AUDIT')=='1':\n            import byte_audit_dynamic\n            byte_audit_dynamic.patch_scheduler(cls,DIRECTORY)\n"+s[end:]
ast.parse(s);f.write_text(s)
print('Prepared intrusive dynamic per-job byte audit; not run on GPU.')
