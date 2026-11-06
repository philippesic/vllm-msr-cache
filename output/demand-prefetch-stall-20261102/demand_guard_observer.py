"""Passive exact-branch counts and bounded per-request episodes; never waits."""
import time,copy

def data(s):
 if not hasattr(s,'_demand_guard_probe_data'):
  s._demand_guard_probe_data={'calls':0,'blocked_calls':0,'rejected_observations':0,'episodes':[],'active':{}}
 return s._demand_guard_probe_data

def metadata(s,request):
 pool=getattr(s,'gpu_block_pool',None);status=getattr(request,'status',None);native=getattr(s,'_req_status',{}).get(request.request_id)
 return {'native_request_status':getattr(status,'name',str(status)),'gpu_free_blocks':pool.get_num_free_blocks() if pool is not None else None,'prefetch_reserved_blocks':getattr(s,'_prefetch_reserved_blocks',None),'native_pending_transfer_jobs':len(getattr(native,'transfer_jobs',{}))}

def eligible(row,at,reason,status):
 if row.get('eligible_at_ns') is None:
  row.update(eligible_at_ns=at,eligibility_reason=reason,eligible_status=status)

def observe(s,request,eid,state,blocked,*,at_ns=None):
 now=time.monotonic_ns() if at_ns is None else at_ns;d=data(s);d['calls']+=1;rid=request.request_id;status=getattr(state,'status',None);active=d['active'];row=active.get(rid)
 if blocked:
  d['blocked_calls']+=1
  # Source branch is exact; independently reject malformed identity telemetry.
  native=getattr(s,'_req_status',{}).get(rid)
  if not eid or status not in ('pending','in_flight') or getattr(s,'_wait_hint_request_bindings',{}).get(rid)!=eid or getattr(state,'bound_request_id',None)!=rid or getattr(native,'req',None) is not request:
   d['rejected_observations']+=1;return
  if row is None or row['event_id']!=eid:
   if row is not None:eligible(row,now,'binding_changed',status);row['next_native_path_ns']=now
   row={'request_id':rid,'event_id':eid,'first_guard_ns':now,'last_guard_ns':now,'guard_calls':0,'status_calls':{},'transitions':[],'eligible_at_ns':None,'next_native_path_ns':None,'first_metadata':metadata(s,request)};d['episodes'].append(row);active[rid]=row
  row['guard_calls']+=1;row['last_guard_ns']=now;row['status_calls'][status]=row['status_calls'].get(status,0)+1;row['last_metadata']=metadata(s,request)
  if not row['transitions'] or row['transitions'][-1]['status']!=status:row['transitions'].append({'at_ns':now,'status':status})
 elif row is not None:
  eligible(row,now,'guard_released_to_native_path_observed',status);row['next_native_path_ns']=now;row['next_native_metadata']=metadata(s,request);active.pop(rid,None)

def on_event(s,event):
 if s is None:return
 kind=event.get('event')
 if kind not in ('staging_ready','staging_retired','staging_retired_pending_fence'):return
 stamp=event.get('ready_at_ns') if kind=='staging_ready' else event.get('retired_at_ns')
 if not isinstance(stamp,int):return
 for row in data(s)['active'].values():
  if row['event_id']==event.get('event_id') and stamp>=row['first_guard_ns']:
   eligible(row,stamp,kind+(':'+str(event.get('reason')) if event.get('reason') else ''),'ready' if kind=='staging_ready' else 'retired')

def finished(s,rid,at_ns=None):
 row=data(s)['active'].pop(rid,None)
 if row is not None:
  now=time.monotonic_ns() if at_ns is None else at_ns;eligible(row,now,'request_finished_or_cancelled','finished');row['request_finished_ns']=now

def snapshot(s):
 if s is None:return None
 d=data(s);return {'calls':d['calls'],'blocked_calls':d['blocked_calls'],'rejected_observations':d['rejected_observations'],'episodes':copy.deepcopy(d['episodes']),'active_request_ids':list(d['active']),'semantics':'nonblocking scheduler return; spans bound possible added per-request latency, never causal GPU idle or TPS loss'}

def install(cls,module,scheduler):
 original_record=module.record_wait_hint_event
 def record(event):
  result=original_record(event);on_event(scheduler(),event);return result
 module.record_wait_hint_event=record
 original_finished=cls.request_finished
 def finish(self,request,*a,**kw):
  try:return original_finished(self,request,*a,**kw)
  finally:finished(self,request.request_id)
 cls.request_finished=finish
