"""Real native Request/NewRequestData/allocator and strict observer invalid controls."""
import copy,time
from types import SimpleNamespace
import pytest,torch
from test_shared_prefix import base
import readmission_observer as observer
import semantic_offload.connector as module
from vllm.v1.core.sched.output import NewRequestData

def setup():
 f=base._native_adoption_fixture(torch.float16)
 f.sched._retire_wait_prefetch(f.state,'cancel',now_ns=time.monotonic_ns());base._native_adoption_complete(f)
 req=base._native_adoption_request('actual',[10]*35)
 assert f.cache.allocate_slots(req,35) is not None
 f.cache.cache_blocks(req,32);req.num_computed_tokens=32;req.num_preemptions=1
 f.sched._req_status[req.request_id]=SimpleNamespace(req=req,group_states=[SimpleNamespace(offload_keys=[module.make_offload_key(v,0) for v in req.block_hashes])])
 groups=f.cache.get_blocks(req.request_id).get_block_ids()
 data=NewRequestData.from_request(req,groups,list(req.all_token_ids))
 state={'request_id':req.request_id,'episode_id':1,'preemption_count':1,'preempted':True,'physical_block_table':((9,10,11),),'physical_block_table_generation':0,'physical_block_table_updates':[{'episode_id':0,'table_generation':0}],'physical_block_table_errors':[],'physical_block_identity_complete':True,'prompt_token_count':35,'prompt_token_digest':observer.digest(req.prompt_token_ids),'ordered_key_positions':f.sched._ordered_request_key_positions(req)[0]}
 native=SimpleNamespace(use_v2_model_runner=True,requests={req.request_id:req},kv_cache_manager=f.cache)
 out=SimpleNamespace(scheduled_new_reqs=[data]);kwargs=dict(mode='scheduled_new_full',start_tokens=32,scheduled_tokens=3,scheduled_prompt_tokens=3,scheduler_batch_id=9)
 class Observed(type(f.sched)):pass
 f.sched.__class__=Observed;events=[];observer.install(Observed,lambda:native,lambda kind,row:events.append(row),module)
 return SimpleNamespace(f=f,req=req,groups=groups,data=data,state=state,native=native,out=out,kwargs=kwargs,events=events)

def call(x):
 token=observer.CURRENT.set(x.out)
 try:return x.f.sched._update_physical_block_table(x.state,x.groups,**x.kwargs)
 finally:observer.CURRENT.reset(token)

def test_actual_v2_full_readmission_strictly_replaces_only_observed_table():
 x=setup();before=[(b.ref_cnt,b.block_hash) for b in x.f.cache.block_pool.blocks]
 result=call(x)
 assert result==x.groups
 assert x.state['physical_block_table_generation']==1
 last=x.state['physical_block_table_updates'][-1]
 assert last['mode']=='cached_resumed_replace' and last['native_payload_kind']=='scheduled_new_reqs'
 assert x.events[-1]['accepted'] is True
 assert before==[(b.ref_cnt,b.block_hash) for b in x.f.cache.block_pool.blocks]
 assert x.req.num_computed_tokens==32

@pytest.mark.parametrize('bad',['no_preemption','stale_episode','stale_native_count','different_request','wrong_prompt','wrong_alltokens','wrong_digest','wrong_count','duplicate_payload','wrong_native_table','wrong_hash','wrong_key_chain','unowned_block','v1','wrong_generation'])
def test_invalid_or_duplicate_newfull_cannot_bypass_strict_proof(bad):
 x=setup();old_table=x.state['physical_block_table']
 if bad=='no_preemption':x.state.update(episode_id=0,preemption_count=0,preempted=False);x.req.num_preemptions=0
 elif bad=='stale_episode':x.state['physical_block_table_updates'][-1]['episode_id']=1
 elif bad=='stale_native_count':x.req.num_preemptions=2
 elif bad=='different_request':x.native.requests[x.req.request_id]=base._native_adoption_request(x.req.request_id,[10]*35)
 elif bad=='wrong_prompt':x.data.prompt_token_ids=[11]*35
 elif bad=='wrong_alltokens':x.data.prefill_token_ids=[11]*35
 elif bad=='wrong_digest':x.state['prompt_token_digest']='bad'
 elif bad=='wrong_count':x.data.num_computed_tokens=31
 elif bad=='duplicate_payload':x.out.scheduled_new_reqs.append(x.data)
 elif bad=='wrong_native_table':x.groups=([1,2],);x.data.block_ids=x.groups
 elif bad=='wrong_hash':x.f.cache.block_pool.blocks[x.groups[0][0]]._block_hash=b'bad'
 elif bad=='wrong_key_chain':x.state['ordered_key_positions'][0]['key']='bad'
 elif bad=='unowned_block':x.f.cache.block_pool.blocks[x.groups[0][-1]].ref_cnt=0
 elif bad=='v1':x.native.use_v2_model_runner=False
 elif bad=='wrong_generation':x.state['physical_block_table_generation']=1
 assert call(x) is None
 assert x.state['physical_block_table']==old_table and x.state['physical_block_identity_complete'] is False
 assert x.events[-1]['accepted'] is False


def test_uncontextualized_duplicate_newfull_remains_rejected():
 x=setup()
 assert x.f.sched._update_physical_block_table(x.state,x.groups,**x.kwargs) is None
 assert x.state['physical_block_table_errors'][-1]['reason']=='invalid_new_full_block_table'


def test_real_scheduler_ordered_keys_are_native_bytes_not_event_serialization():
 x=setup();key=x.state['ordered_key_positions'][0]['key']
 assert isinstance(key,bytes) and key!=module._event_key(key)
 assert call(x)==x.groups

def test_serialized_event_key_cannot_substitute_for_owned_scheduler_key():
 x=setup()
 x.state['ordered_key_positions'][0]['key']=module._event_key(x.state['ordered_key_positions'][0]['key'])
 assert call(x) is None
 assert x.events[-1]['reason']=='prompt_native_key_chain_mismatch'
