"""Observe native pending-transfer no-query returns; never manufacture a lookup."""
import contextvars,hashlib,inspect,time
QUERY=contextvars.ContextVar('native_pending_guard_query_check',default=None)

def patch_manager_class(cls):
 if cls.__dict__.get('_frontier_lookup_counter',False):return
 original=cls.lookup
 def lookup(self,*a,**kw):
  current=QUERY.get()
  if current is not None and self is current['manager']:current['manager_queries']+=1
  return original(self,*a,**kw)
 cls.lookup=lookup;cls._frontier_lookup_counter=True

def install(cls,base,emit):
 if cls.__dict__.get('_frontier_pending_defer',False):return
 original_finish=cls._finish_lookup_episode
 if not base.__dict__.get('_frontier_pending_guard_capture',False):
  original_base=base.get_num_new_matched_tokens
  source_sha=hashlib.sha256(inspect.getsource(original_base).encode()).hexdigest()
  def matched(self,request,num_computed_tokens):
   state=getattr(self,'_preservation_requests',{}).get(request.request_id);status=getattr(self,'_req_status',{}).get(request.request_id)
   if state is not None:state.pop('_frontier_actual_native_defer',None)
   pending=tuple(sorted(status.transfer_jobs)) if status is not None and status.transfer_jobs else ()
   if not pending or state is None:return original_base(self,request,num_computed_tokens)
   episode=state.get('active_lookup_episode');jobs=getattr(self,'_jobs',{})
   valid=(status.req is request and isinstance(episode,dict) and not episode['groups']
          and all(j in jobs and jobs[j].req_id==request.request_id and isinstance(jobs[j].pending_count,int) and not isinstance(jobs[j].pending_count,bool) and jobs[j].pending_count>0 for j in pending))
   counter={'manager':getattr(self,'manager',None),'manager_queries':0};token=QUERY.set(counter)
   try:result=original_base(self,request,num_computed_tokens)
   finally:QUERY.reset(token)
   if (valid and result==(None,False) and status is self._req_status.get(request.request_id)
       and status.req is request and tuple(sorted(status.transfer_jobs))==pending
       and state.get('active_lookup_episode') is episode and not episode['groups']
       and counter['manager_queries']==0):
    state['_frontier_actual_native_defer']={'episode':episode,'request':request,'event':{'request_id':request.request_id,'episode_id':episode.get('episode_id'),'attempt_id':episode.get('lookup_id'),'started_at_ns':episode.get('started_at_ns'),'completed_at_ns':time.monotonic_ns(),'reason':'actual_native_pending_transfer_guard','result':[None,False],'native_pending_jobs':[{'job_id':j,'request_id':jobs[j].req_id,'pending_workers':jobs[j].pending_count,'is_store':jobs[j].is_store} for j in pending],'same_live_request_object':True,'native_base_source_sha256':source_sha,'manager_queries_observed':counter['manager_queries'] if type(self.manager).__dict__.get('_frontier_lookup_counter',False) else None,'manager_query_counter_enabled':type(self.manager).__dict__.get('_frontier_lookup_counter',False),'actual_lookup_executed':False}}
   return result
  base.get_num_new_matched_tokens=matched;base._frontier_pending_guard_capture=True
 def finish(self,request):
  state=getattr(self,'_preservation_requests',{}).get(request.request_id);evidence=state.pop('_frontier_actual_native_defer',None) if state is not None else None
  if (evidence is not None and evidence['request'] is request
      and state.get('active_lookup_episode') is evidence['episode']
      and not evidence['episode']['groups']):
   # The native call did not reach lookup. Preserve an explicit separate
   # diagnostic event, and leave last real lookup ID untouched. All genuine
   # lookup episodes still use the unmodified strict observer/validator.
   state.pop('active_lookup_episode');emit('NATIVE_LOOKUP_DEFERRED',evidence['event'])
   return original_finish(self,request)
  return original_finish(self,request)
 cls._finish_lookup_episode=finish;cls._frontier_pending_defer=True
