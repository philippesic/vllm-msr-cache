import json
from types import SimpleNamespace as NS
import pytest,torch
import byte_audit_dynamic as audit
@pytest.mark.parametrize('bad',[None,'wrong_job','changed_local','changed_suffix','changed_suffix_id','wrong_local_count'])
def test_all_native_adopted_regions_match_owned_pretransfer_snapshots(tmp_path,monkeypatch,bad):
 monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
 data=torch.arange(32,dtype=torch.uint8).reshape(8,4);ref=NS(tensor_idx=0,page_size_bytes=4)
 audit.ADOPTION_SNAPSHOTS.clear()
 audit.ADOPTION_SNAPSHOTS[7]={'local_ids':[1,2],'local_components':[(ref,data[[1,2]].clone())],
   'suffix_ids':[3,4],'suffix_components':[(ref,data[[3,4]].clone())],'gpu_tensors':[data]}
 if bad=='changed_local':data[1,0]^=1
 if bad=='changed_suffix':data[3,0]^=1
 args=([1,2,3,5] if bad=='changed_suffix_id' else [1,2,3,4],
       1 if bad=='wrong_local_count' else 2,'r',tmp_path,8 if bad=='wrong_job' else 7)
 if bad:
  with pytest.raises(AssertionError):audit.verify_native_prefix_bytes(*args)
 else:
  audit.verify_native_prefix_bytes(*args)
  row=json.loads(next(tmp_path.glob('native-adoption-byte-audit-*.jsonl')).read_text())
  assert row['total_bytes']==16 and {c['region'] for c in row['components']}=={'local','suffix'}
  assert all(c['equal'] and c['invalid_byte_rejected'] for c in row['components'])
def test_busy_judge_checks_every_one_of_72_unique_requests():
 import teacher_busy_judge
 refs=[dict(session_id=f's{i}',turn=t,prompt=[i+1,t+1],reference_output_token_ids=[3,4]) for i in range(24) for t in range(3)]
 raw={'requests':[dict(session_id=r['session_id'],turn=r['turn'],success=True,payload=dict(prompt=r['prompt'][:],max_tokens=2,ignore_eos=True,vllm_xargs={'replay_token_ids':[3,4]}),output_token_ids=[3,4]) for r in refs]}
 assert teacher_busy_judge.validate(raw,{'requests':refs})['requests']==72
 assert teacher_busy_judge.reject_invalid_controls(raw,{'requests':refs})
