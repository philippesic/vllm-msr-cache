from types import SimpleNamespace as NS
import copy,types
import pytest,torch
import teacher_replay,teacher_judge
@pytest.fixture
def sampler():
 class Original:
  def __init__(self,*a,**kw):self.calls=0
  def add_request(self,*a):pass
  def __call__(self,logits,batch):self.calls+=1;return NS(sampled_token_ids=logits.argmax(-1).reshape(-1,1))
 module=NS(Sampler=Original);teacher_replay.patch(module,lambda *a:None)
 return module.Sampler(4,20,'cpu',None)
def batch(mapping,position):return NS(expanded_idx_mapping=torch.tensor(mapping),positions=torch.tensor(position),logits_indices=torch.arange(len(mapping)))
def params(ids):return NS(extra_args={'replay_token_ids':ids},max_tokens=len(ids),ignore_eos=True)
def test_original_model_sampling_executes_before_recorded_token_override(sampler):
 sampler.add_request(2,10,params([4,5,6]))
 for pos,expected in [(9,4),(10,5),(11,6)]:
  out=sampler(torch.zeros((1,20)),batch([2],[pos]));assert out.sampled_token_ids.item()==expected
 assert sampler.calls==3
def test_reordered_batch_uses_persistent_request_mapping_and_absolute_position(sampler):
 sampler.add_request(1,7,params([8,9]));sampler.add_request(3,12,params([10,11]))
 out=sampler(torch.zeros((2,20)),batch([3,1],[12,6]))
 assert out.sampled_token_ids.flatten().tolist()==[11,8]
def test_normal_request_and_reused_slot_are_unchanged(sampler):
 sampler.add_request(1,7,params([8,9]));sampler.add_request(1,7,NS(extra_args=None))
 x=torch.zeros((1,20));x[0,3]=1
 assert sampler(x,batch([1],[8])).sampled_token_ids.item()==3
@pytest.mark.parametrize('invalid',[[],[True],[-1],[20],[1]*97])
def test_invalid_recorded_tokens_rejected(sampler,invalid):
 with pytest.raises(ValueError):sampler.add_request(0,2,params(invalid))
def test_recorded_count_and_ignore_eos_required(sampler):
 p=params([1]);p.max_tokens=2
 with pytest.raises(ValueError):sampler.add_request(0,2,p)
 p=params([1]);p.ignore_eos=False
 with pytest.raises(ValueError):sampler.add_request(0,2,p)
def test_judge_requires_all_input_output_ids_and_rejects_wrong_controls():
 rows=[dict(session_id=f's{s}',turn=t,prompt=[s+1,t+1],reference_output_token_ids=[3,4,5]) for s in range(4) for t in range(6)]
 raw={'requests':[dict(session_id=r['session_id'],turn=r['turn'],success=True,payload=dict(prompt=r['prompt'][:],max_tokens=3,ignore_eos=True,vllm_xargs={'replay_token_ids':[3,4,5]}),output_token_ids=[3,4,5]) for r in rows]}
 assert teacher_judge.validate(raw,{'requests':rows})['decode_tokens']==72
 assert teacher_judge.reject_invalid_controls(raw,{'requests':rows})
