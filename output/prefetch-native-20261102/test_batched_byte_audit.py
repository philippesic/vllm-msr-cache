from types import SimpleNamespace as NS
import json,sys
import pytest,torch
import byte_audit_batched as audit
@pytest.mark.parametrize('direction',['D2H','H2D'])
@pytest.mark.parametrize('corrupt',[None,0,13,27])
def test_owned_all_component_snapshots_and_wrong_bytes(tmp_path,monkeypatch,direction,corrupt):
 events=[]
 class Stream:
  def wait_event(self,event):events.append('wait');event.synchronize()
 class Fence:
  def synchronize(self):events.append('fence')
  def record(self,stream):events.append('record')
 monkeypatch.setattr(torch.cuda,'current_stream',lambda:Stream())
 monkeypatch.setattr(torch.cuda,'Event',Fence)
 empty=torch.empty
 monkeypatch.setattr(torch,'empty',lambda *a,**kw:empty(*a,**{k:v for k,v in kw.items() if k!='pin_memory'}))
 class Handler:
  def __init__(self):
   self.src_blocks_per_chunk=self.dst_blocks_per_chunk=1;self.gpu_to_cpu=direction=='D2H'
   self.kv_cache_groups_data_refs=[[NS(tensor_idx=i,page_size_bytes=4) for i in range(28)]]
   self.src_tensors=[(torch.arange(24,dtype=torch.int8)+i).reshape(6,4) for i in range(28)]
   self.dst_tensors=[torch.zeros((6,4),dtype=torch.int8) for _ in range(28)];self._transfer_events={}
  def transfer_async(self,j,s,d):
   for i in range(28):
    self.dst_tensors[i][d.block_ids]=self.src_tensors[i][s.block_ids]
    # Source is overwritten after actual copy. Only owned snapshots can pass.
    self.src_tensors[i][s.block_ids]=0
   if corrupt is not None:self.dst_tensors[corrupt][d.block_ids[0],0]^=1
   self._transfer_events[j]=Fence();return True
 m=NS(SingleDirectionOffloadingHandler=Handler);audit.patch(m,tmp_path);obj=Handler()
 call=lambda:obj.transfer_async(5,NS(block_ids=[0,2]),NS(block_ids=[3,1]))
 if corrupt is not None:
  with pytest.raises(AssertionError,match='actual KV transfer mismatch'):call()
 else:
  assert call();r=json.loads(next(tmp_path.glob('byte-audit-*.jsonl')).read_text())
  assert r['source_physical_ids']==[0,2] and r['destination_physical_ids']==[3,1]
  assert r['total_bytes']==28*2*4 and len(r['components'])==28
  assert all(c['equal'] and c['invalid_byte_rejected'] for c in r['components'])
  assert events==(['fence'] if direction=='D2H' else ['wait','fence','record','fence'])
