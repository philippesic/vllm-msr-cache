"""Use actual native guard and Request/TransferJobStatus, with fail-closed cases."""
from types import SimpleNamespace
import pytest,torch
from vllm import SamplingParams
from vllm.v1.request import Request
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler,TransferJobStatus
from semantic_offload.connector import SemanticOffloadingConnectorScheduler,_lookup_metadata_complete
import lookup_deferral_observer as obs

class Manager:
 def __init__(self):self.calls=0
 def lookup(self,*a,**kw):self.calls+=1;return None
obs.patch_manager_class(Manager)

def fixture(base=OffloadingConnectorScheduler):
 class Scheduler(base):
  _finish_lookup_episode=SemanticOffloadingConnectorScheduler._finish_lookup_episode
  def _lookup(self,*a):raise AssertionError('native pending guard must never lookup')
 x=Scheduler.__new__(Scheduler);req=Request('real-native-request',[10]*35,SamplingParams(max_tokens=8),None);e={'lookup_id':5,'episode_id':1,'started_at_ns':1,'groups':{}};x._preservation_requests={req.request_id:{'active_lookup_episode':e,'lookup_episodes':[],'lookup_observed':False,'lookup_keys':[],'prior_evicted_keys':[],'prior_stored_keys':[],'cold_keys':[],'last_lookup_id':4}};x._preservation_evicted_keys=set();x._preservation_known_stored_keys=set();x._req_status={req.request_id:SimpleNamespace(req=req,transfer_jobs={7},group_states=[SimpleNamespace(block_ids=[1])])};x._jobs={7:TransferJobStatus(req.request_id,1,{b'key'},True)};x.manager=Manager();events=[];obs.install(Scheduler,base,lambda k,v:events.append((k,v)))
 return SimpleNamespace(x=x,req=req,e=e,state=x._preservation_requests[req.request_id],events=events)

def test_actual_native_pending_guard_no_query_deferral_is_separate_from_lookup():
 f=fixture();assert f.x.get_num_new_matched_tokens(f.req,0)==(None,False);f.x._finish_lookup_episode(f.req)
 assert f.state['lookup_episodes']==[] and f.state['last_lookup_id']==4
 assert len(f.events)==1 and f.events[0][1]['native_pending_jobs'][0]['job_id']==7
 assert f.events[0][1]['manager_queries_observed']==0 and f.x.manager.calls==0
 assert f.x._req_status[f.req.request_id].transfer_jobs=={7} and f.x._jobs[7].pending_count==1

@pytest.mark.parametrize('bad',['wrong_native_request','foreign_job','missing_job','zero_workers','bool_workers','stale_episode','wrong_finish_request','observed_group'])
def test_invalid_evidence_does_not_erase_an_empty_lookup(bad):
 f=fixture()
 if bad=='wrong_native_request':f.x._req_status[f.req.request_id].req=object()
 elif bad=='foreign_job':f.x._jobs[7].req_id='foreign'
 elif bad=='missing_job':f.x._jobs.clear()
 elif bad=='zero_workers':f.x._jobs[7].pending_count=0
 elif bad=='bool_workers':f.x._jobs[7].pending_count=True
 f.x.get_num_new_matched_tokens(f.req,0)
 if bad=='stale_episode':f.state['active_lookup_episode']=dict(f.e)
 elif bad=='observed_group':f.e['groups']={'0':{'eligible':[],'queried':[],'hits':[],'missing':[],'pending':[],'retry':[],'ordered_probes':[],'observed':False}}
 request=Request(f.req.request_id,[10]*35,SamplingParams(max_tokens=8),None) if bad=='wrong_finish_request' else f.req
 f.x._finish_lookup_episode(request)
 assert not f.events and len(f.state['lookup_episodes'])==1
 assert not _lookup_metadata_complete(f.state['lookup_episodes'])

def test_ordinary_empty_lookup_without_native_guard_marker_stays_invalid():
 f=fixture();f.x._finish_lookup_episode(f.req)
 assert not _lookup_metadata_complete(f.state['lookup_episodes']) and not f.events

def test_observed_manager_query_forbids_defer_normalization():
 class QueryingBase:
  def get_num_new_matched_tokens(self,request,num_computed_tokens):self.manager.lookup(b'bad');return None,False
 f=fixture(QueryingBase);f.x.get_num_new_matched_tokens(f.req,0);f.x._finish_lookup_episode(f.req)
 assert f.x.manager.calls==1 and not f.events and not _lookup_metadata_complete(f.state['lookup_episodes'])

def test_no_cuda_initialized():assert not torch.cuda.is_initialized()
