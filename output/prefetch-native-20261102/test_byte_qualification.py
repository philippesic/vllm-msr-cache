import json
import pytest
import torch
import byte_audit_qualified as audit
from types import SimpleNamespace as NS

@pytest.mark.parametrize('bad',[None,'wrong_job','changed_byte'])
def test_owned_local_snapshot_requires_exact_job_and_identical_bytes(tmp_path,monkeypatch,bad):
 monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
 data=torch.arange(16,dtype=torch.uint8).reshape(4,4)
 ref=NS(tensor_idx=0,page_size_bytes=4)
 audit.LOCAL_SNAPSHOTS.clear()
 audit.LOCAL_SNAPSHOTS[(7,(1,2))]={'job_id':7,'components':[(ref,data[[1,2]].clone())],'gpu_tensors':[data]}
 if bad=='changed_byte':data[1,0]^=1
 if bad:
  with pytest.raises(AssertionError,match='snapshot' if bad=='wrong_job' else 'mutated'):
   audit.verify_local_prefix_on_adoption([1,2],'request',tmp_path,8 if bad=='wrong_job' else 7)
 else:
  audit.verify_local_prefix_on_adoption([1,2],'request',tmp_path,7)
  row=json.loads(next(tmp_path.glob('local-prefix-audit-*.jsonl')).read_text())
  assert row['source_h2d_job_id']==7 and row['components'][0]['invalid_byte_rejected']
