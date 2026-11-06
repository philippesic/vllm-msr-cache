"""Evaluation-only token replay. Normal model forwards/sampling still execute.

Current-request recorded completions are supplied by the evaluation driver.
They never enter the wait-hint producer or speculative nomination policy.
This measures serving compute/cache behavior; it is not unconstrained generation.
"""
def patch(module,emit):
 import torch
 cls=module.Sampler;init=cls.__init__;add=cls.add_request;call=cls.__call__
 def initialize(self,max_num_reqs,vocab_size,device,req_states,*args,**kwargs):
  init(self,max_num_reqs,vocab_size,device,req_states,*args,**kwargs)
  self._teacher_ids=torch.zeros((max_num_reqs,96),dtype=torch.long,device=device)
  self._teacher_prompt_len=torch.zeros(max_num_reqs,dtype=torch.long,device=device)
  self._teacher_enabled=torch.zeros(max_num_reqs,dtype=torch.bool,device=device)
  self._teacher_vocab_size=vocab_size;self._teacher_seen=False
 def add_request(self,req_idx,prompt_len,sampling_params):
  reference=(sampling_params.extra_args or {}).get('replay_token_ids')
  if reference is not None:
   if (not isinstance(reference,list) or not 1<=len(reference)<=96
       or any(type(t) is not int or not 0<=t<self._teacher_vocab_size for t in reference)
       or sampling_params.max_tokens!=len(reference) or not sampling_params.ignore_eos):
    raise ValueError('invalid evaluation-only recorded completion')
  add(self,req_idx,prompt_len,sampling_params)
  self._teacher_enabled[req_idx]=reference is not None
  if reference is not None:
   self._teacher_ids[req_idx,:len(reference)].copy_(torch.tensor(reference,device=self._teacher_ids.device))
   self._teacher_prompt_len[req_idx]=prompt_len;self._teacher_seen=True
   emit('TEACHER_REQUEST',{'req_idx':req_idx,'prompt_len':prompt_len,'recorded_tokens':len(reference),'evaluation_only':True})
 def sample(self,logits,input_batch):
  output=call(self,logits,input_batch)
  if not self._teacher_seen:return output
  mapping=input_batch.expanded_idx_mapping
  position=input_batch.positions[input_batch.logits_indices]
  offset=(position-self._teacher_prompt_len[mapping]+1).clamp(0,95)
  reference=self._teacher_ids[mapping,offset].reshape(-1,1)
  enabled=self._teacher_enabled[mapping].reshape(-1,1)
  assert reference.shape==output.sampled_token_ids.shape
  output.sampled_token_ids.copy_(torch.where(enabled,reference,output.sampled_token_ids))
  return output
 cls.__init__=initialize;cls.add_request=add_request;cls.__call__=sample
