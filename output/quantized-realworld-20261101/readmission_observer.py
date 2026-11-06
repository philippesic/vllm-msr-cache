"""Strict observer-only handling of V2 full native readmission payloads.
Never changes requests, native block tables, KV, transfer jobs or scheduling.
"""
import contextvars,hashlib,json,time
from contextlib import contextmanager
CURRENT=contextvars.ContextVar('actual_native_scheduler_output',default=None)

def integer(x):return isinstance(x,int) and not isinstance(x,bool)
def digest(tokens):return hashlib.sha256(json.dumps(list(tokens),separators=(',',':')).encode()).hexdigest()

def check(sched,state,groups,kwargs,out,native,module):
 rid=state.get('request_id');req_status=getattr(sched,'_req_status',{}).get(rid);req=getattr(req_status,'req',None)
 payloads=[x for x in getattr(out,'scheduled_new_reqs',[]) if x.req_id==rid]
 if len(payloads)!=1 or native is None or getattr(native,'use_v2_model_runner',False) is not True:return False,'not_unique_actual_v2_payload'
 data=payloads[0]
 if req is None or native.requests.get(rid) is not req or data.req_id!=req.request_id:return False,'native_request_identity_mismatch'
 updates=state.get('physical_block_table_updates',[]);episode=state.get('episode_id');count=state.get('preemption_count');prior=updates[-1] if updates else {}
 if (not integer(episode) or episode<=0 or not integer(count) or count!=episode
     or getattr(req,'num_preemptions',None)!=count or state.get('preempted') is not True
     or not integer(prior.get('episode_id')) or prior['episode_id']>=episode
     or prior.get('table_generation')!=state.get('physical_block_table_generation')):return False,'missing_or_stale_preemption_episode'
 start=kwargs.get('start_tokens');scheduled=kwargs.get('scheduled_tokens')
 if not integer(start) or not integer(scheduled) or start<0 or scheduled<0 or data.num_computed_tokens!=start or req.num_computed_tokens!=start:return False,'native_computed_count_mismatch'
 prompt=getattr(data,'prompt_token_ids',None);alltokens=getattr(data,'prefill_token_ids',None)
 if (not isinstance(prompt,list) or prompt!=req.prompt_token_ids or len(prompt)!=state.get('prompt_token_count')
     or digest(prompt)!=state.get('prompt_token_digest') or alltokens!=list(req.all_token_ids)):
  return False,'native_prompt_or_token_identity_mismatch'
 if groups!=data.block_ids:return False,'full_payload_table_mismatch'
 pool=getattr(sched,'gpu_block_pool',None);cache=getattr(native,'kv_cache_manager',None)
 if pool is None or cache is None or cache.block_pool is not pool:return False,'native_pool_identity_mismatch'
 actual=cache.get_blocks(rid).get_block_ids()
 if groups!=actual:return False,'native_live_full_table_mismatch'
 configs=sched.config.kv_group_configs
 if not isinstance(groups,(tuple,list)) or len(groups)!=len(configs):return False,'native_group_count_mismatch'
 positions={(x['group_idx'],x['chunk_index']):x['key'] for x in state.get('ordered_key_positions',[])}
 for group,config in zip(groups,configs):
  block_size=module.group_config_tokens_per_block(config);computed=start//block_size
  if (not isinstance(group,(tuple,list)) or len(group)<(start+scheduled+block_size-1)//block_size
      or any(not integer(i) or not 0<i<pool.num_gpu_blocks for i in group)
      or len(set(group))!=len(group)):return False,'invalid_native_full_table_geometry'
  for index,bid in enumerate(group):
   block=pool.blocks[bid]
   if block.ref_cnt<=0:return False,'unowned_native_full_table_block'
   if index<computed:
    if index>=len(req.block_hashes):return False,'missing_native_computed_hash'
    key=module.make_offload_key(req.block_hashes[index],config.group_idx)
    if block.block_hash!=key:return False,'computed_physical_hash_mismatch'
    if index<len(prompt)//block_size and positions.get((config.group_idx,index))!=key:return False,'prompt_native_key_chain_mismatch'
 return True,'v2_full_readmission_after_observed_preemption'

def install(cls,native_getter,emit,module):
 if getattr(cls,'_matrix_v2_readmission_observer',False):return
 original_record=cls._record_scheduler_execution;original_table=cls._update_physical_block_table
 def record(self,out):
  token=CURRENT.set(out)
  try:return original_record(self,out)
  finally:CURRENT.reset(token)
 def table(self,state,groups,**kwargs):
  out=CURRENT.get();mode=kwargs.get('mode')
  if mode!='scheduled_new_full' or state.get('physical_block_table') is None or out is None:
   return original_table(self,state,groups,**kwargs)
  accepted,reason=check(self,state,groups,kwargs,out,native_getter(),module)
  if not accepted:
   self._invalidate_physical_block_table(state,'v2_readmission_'+reason,kwargs['scheduler_batch_id'])
   emit('READMISSION_OBSERVER',{'request_id':state.get('request_id'),'episode_id':state.get('episode_id'),'accepted':False,'reason':reason,'observed_at_ns':time.monotonic_ns()})
   return None
  normalized=dict(kwargs,mode='cached_resumed_replace')
  result=original_table(self,state,groups,**normalized)
  if result is not None:
   state['physical_block_table_updates'][-1].update(native_payload_kind='scheduled_new_reqs',replacement_reason=reason)
  emit('READMISSION_OBSERVER',{'request_id':state.get('request_id'),'episode_id':state.get('episode_id'),'accepted':result is not None,'reason':reason,'observed_at_ns':time.monotonic_ns(),'native_payload_kind':'scheduled_new_reqs','table_generation':state.get('physical_block_table_generation'),'group_lengths':[len(g) for g in groups]})
  return result
 cls._record_scheduler_execution=record;cls._update_physical_block_table=table;cls._matrix_v2_readmission_observer=True
