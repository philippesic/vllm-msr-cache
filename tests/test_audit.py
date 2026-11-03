"""Native CPU audit: real ARC manager, native metadata, actual tensor scoring."""
import threading,time
from types import SimpleNamespace as N
from unittest.mock import patch
import pytest,torch
from semantic_offload.worker import SemanticOffloadingWorker
from semantic_offload.async_scoring import AsyncScoringController
from semantic_offload.manager import PreservationOffloadingManager
from semantic_offload.connector import SemanticOffloadingConnector,SemanticWorkerMetadata,SemanticOffloadingConnectorWorker
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.outputs import KVConnectorOutput
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from tests.test_async_scoring import _worker,_key,_summary,_wait_async_observations
from tests.test_step_1_5_prefetch import _make_connector_scheduler,_StubBlockPool,_register_wait_prefix,_wait_hint

def manager(keys,capacity=None):
 m=PreservationOffloadingManager(num_blocks=capacity or len(keys),cache_policy='arc',store_threshold=2,shared_prefetch_scorer=True)
 ctx=ReqContext(req_id='source')
 for k in keys:m.lookup(k,ctx);m.lookup(k,ctx)
 out=m.prepare_store(keys,ctx);assert out and set(out.keys_to_store)==set(keys)
 m.complete_store(out.keys_to_store,ctx)
 return m

def consume(m,observations,sched=None):
 c=SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
 c.connector_scheduler=sched or N(manager=m,_preservation_requests={},_preservation_finished={})
 c.connector_scheduler._score_context=lambda:{'epoch':'fixture','sources':{r:1 for o in observations for r in o['request_ids']}}
 meta=SemanticWorkerMetadata(async_score_observations=observations,score_group_size=1,score_contributors=1)
 with patch.object(OffloadingConnector,'update_connector_output',lambda *a:None):
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))

def worker():
 w=_worker('asynchronous');w._score_transport_context={'epoch':'fixture','sources':dict.fromkeys(['source','source2','warm-a','warm-b'],1)};w._scoring_schedule='opportunity';w._query_snapshots={};w._query_snapshot_max_age_ns=1_000_000_000;w._query_snapshot_max_requests=8;w._query_snapshot_events={};w._retired_async_request_ids=set();w._decision_async_signatures={};return w

def gated(w):
 w._async_scoring.shutdown();gate=threading.Event();fn=w._score_async_payload
 def score(p):
  assert gate.wait(2);return fn(p)
 w._async_scoring=AsyncScoringController(score,max_queue_items=2,max_snapshot_bytes=4096,workers=1)
 return gate

def test_actual_constructor_accepts_only_implemented_opportunity_async_combination():
 import semantic_offload.worker as mod
 config=N(parallel_config=N(tensor_parallel_size=1,pipeline_parallel_size=1,world_size=1),model_config=N(get_total_num_kv_heads=lambda:8,use_mla=False),compilation_config=N(static_forward_context={}),cache_config=N(block_size=16),use_v2_model_runner=False)
 with patch.object(mod,'init_cpu_offloading_worker_base',lambda *a,**kw:None):
  w=SemanticOffloadingWorker([],1,8,config,scoring_mode='asynchronous',scoring_schedule='opportunity')
 assert w._scoring_schedule=='opportunity';assert w._async_scoring is not None;w._async_scoring.shutdown()
 with patch.object(mod,'init_cpu_offloading_worker_base',lambda *a,**kw:None),pytest.raises(ValueError):
  SemanticOffloadingWorker([],1,8,config,scoring_mode='asynchronous',scoring_schedule='opportunity',scoring_search='ivf')

def test_empty_and_nonempty_retirement_do_not_wait_unrelated_future_and_keep_final_lease():
 w=worker();gate=gated(w)
 try:
  w._on_queries_captured(['source'],torch.ones(1,1,4));stamp=w._query_snapshots['source'][0]
  w.receive_score_requests({'source':list(w.durable_summaries)})
  cw=SemanticOffloadingConnectorWorker.__new__(SemanticOffloadingConnectorWorker);cw.worker=w
  t=time.monotonic()
  with patch.object(cw.__class__.__mro__[1],'get_finished',lambda *a:(set(),set())):
   cw.get_finished(set());cw.get_finished({'unrelated','source'})
  assert time.monotonic()-t<.1 and w._async_scoring.stats()['pending_items']==1
  assert w._query_snapshots['source'][0]==stamp
  gate.set();obs=_wait_async_observations(w);assert obs and obs[0]['snapshot_captured_ns']==stamp
 finally:gate.set();w._async_scoring.shutdown()

def test_projected_payload_actual_unique_backing_storage_and_queue_precheck():
 w=worker();gate=gated(w)
 try:
  backing=torch.ones(64);w.durable_summaries={_key(1):[N(mean=backing[:4],min=torch.ones(400),max=torch.ones(400),mad=torch.ones(400))]}
  w._durable_summary_publications={_key(1):(0,1,1)}
  w._submit_async_scoring(['source'],torch.ones(1,1,4))
  pending=next(iter(w._async_scoring._pending.values()));p=pending.snapshot.payload
  assert vars(p.summaries[0][0]).keys()=={'mean'} and pending.snapshot.snapshot_bytes>=backing.untyped_storage().nbytes()
  w._submit_async_scoring(['source2'],torch.ones(1,1,4))
  class Bad(dict):
   def __iter__(self):raise AssertionError('full queue scanned')
  w.durable_summaries=Bad();w._submit_async_scoring(['third'],torch.ones(1,1,4))
  assert w._async_scoring.stats()['pending_items']==2
 finally:gate.set();w._async_scoring.shutdown()

def test_real_arc_worker_scores_metadata_and_same_key_reinsert_consumer_rejects():
 w=worker();keys=list(w.durable_summaries);m=manager(keys,capacity=2)
 try:
  versions={k:m._policy.get(k)._semantic_score_version for k in keys}
  w._on_queries_captured(['source'],torch.ones(1,1,4))
  w.receive_score_requests({'source':keys},{'source':versions});obs=_wait_async_observations(w)
  assert obs[0]['scores']['mean']['source']==[(keys[0],8.),(keys[1],-4.)]
  consume(m,obs);assert m.relevance_ema['mean'][keys[0]]==8.
  # Actual native ARC eviction + same-key reinsert, not a fake version assignment.
  other=[_key(3),_key(4)];ctx=ReqContext(req_id='pressure')
  for k in other:m.lookup(k,ctx);m.lookup(k,ctx)
  out=m.prepare_store(other,ctx);m.complete_store(out.keys_to_store,ctx)
  for k in keys:m.lookup(k,ctx);m.lookup(k,ctx)
  out=m.prepare_store(keys,ctx);m.complete_store(out.keys_to_store,ctx)
  m.relevance_ema.clear();consume(m,obs);assert not m.relevance_ema
 finally:w._async_scoring.shutdown()

@pytest.mark.parametrize('action',['cancel','generation','expire','same_id'])
def test_delayed_request_versions_and_feature_expiry_are_fail_closed(action):
 w=worker();gate=gated(w)
 try:
  w._on_queries_captured(['source'],torch.ones(1,1,4));w.receive_score_requests({'source':list(w.durable_summaries)})
  if action=='cancel':w.cancel_async_requests(['source'])
  elif action=='generation':w._summary_generation+=1
  elif action=='expire':w._async_scoring_max_snapshot_age_ns=1
  else:w.retire_async_requests(['source']);w._on_queries_captured(['source'],torch.zeros(1,1,4))
  gate.set();assert _wait_async_observations(w)==[]
 finally:gate.set();w._async_scoring.shutdown()

def test_completed_source_real_arc_two_competing_sessions_native_metadata_nomination():
 w=worker();a=[_key(i) for i in range(101,117)];b=[_key(i) for i in range(201,217)];m=manager(a+b)
 w.durable_summaries={k:[_summary(2. if k in a else -.5)] for k in a+b};w._durable_summary_publications={k:(0,1,i) for i,k in enumerate(a+b)}
 sched=_make_connector_scheduler(m,_StubBlockPool(320,free_blocks=320));sched._prefetch_selector='semantic';sched._prefetch_score_max_age_ns=1_000_000_000;sched._preservation_requests={};sched._preservation_finished={}
 try:
  for keys,label in [(a,'a'),(b,'b')]:
   prefix,bound=_register_wait_prefix(sched,keys,session_id='session-'+label,request_id='warm-'+label)
   assert sched.publish_wait_hint(_wait_hint(sched,prefix,bound,session_id='session-'+label,event_id=label*64))['status']=='accepted'
  w._on_queries_captured(['warm-a','warm-b'],torch.ones(2,1,4));w.retire_async_requests(['warm-a','warm-b'])
  versions={k:m._policy.get(k)._semantic_score_version for k in a+b}
  for keys,label in [(a,'a'),(b,'b')]:
   w.receive_score_requests({'warm-'+label:keys},{'warm-'+label:{k:versions[k] for k in keys}},{'warm-'+label:[label*64]})
  w._drain_async_scores(wait=True);obs=w.pop_async_score_observations();assert len(obs)==2
  consume(m,obs,sched);assert m.relevance_ema['mean'][a[0]]>m.relevance_ema['mean'][b[0]]
  before=m._policy.__class__.__name__;sched._retry_wait_prefetches();assert m._policy.__class__.__name__==before
  assert any(state.status=='in_flight' for state in sched._wait_prefetch.values())
  assert sum(len(state.keys) for state in sched._wait_prefetch.values())<=16
 finally:w._async_scoring.shutdown()


def test_cancel_after_publication_and_before_metadata_or_consumer_is_rejected():
 w=worker();keys=list(w.durable_summaries);m=manager(keys)
 try:
  versions={k:m._policy.get(k)._semantic_score_version for k in keys}
  w._on_queries_captured(['source'],torch.ones(1,1,4))
  w.receive_score_requests({'source':keys},{'source':versions},{'source':['lease']})
  w._drain_async_scores(wait=True);obs=w.pop_async_score_observations();assert obs
  sched=N(manager=m,_preservation_requests={},_preservation_finished={},_wait_prefetch={})
  consume(m,obs,sched);assert not m.relevance_ema
  # Already queued worker observations are pruned without a device drain.
  w._pending_async_score_observations=obs
  w.cancel_async_requests(['source']);assert w.pop_async_score_observations()==[]
 finally:w._async_scoring.shutdown()


def test_retired_bookkeeping_is_bounded_by_owned_sources_not_total_completed_ids():
 w=worker()
 try:
  w.retire_async_requests({str(i) for i in range(10000)})
  assert w._retired_async_request_ids==set()
  w._on_queries_captured(['source'],torch.ones(1,1,4));w.retire_async_requests(['source'])
  assert w._retired_async_request_ids=={'source'}
 finally:w._async_scoring.shutdown()

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_score_rejects_entire_request_without_ema_or_fake_zero(value):
 w=worker();keys=list(w.durable_summaries);m=manager(keys)
 try:
  w.durable_summaries[keys[0]]=[_summary(value)]
  w._on_queries_captured(['source'],torch.ones(1,1,4))
  w.receive_score_requests({'source':keys})
  w._drain_async_scores(wait=True)
  obs=w.pop_async_score_observations();assert obs==[]
  consume(m,obs);assert not m.relevance_ema
  assert any(e.get('reason')=='nonfinite_score' for e in w.pop_async_scoring_telemetry())
 finally:w._async_scoring.shutdown()


def test_repeated_idle_async_status_keeps_metadata_and_events_logs_only_transitions():
 import semantic_offload.connector as mod
 m=manager([_key(1)])
 c=SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
 c.connector_scheduler=N(manager=m,_preservation_requests={},_preservation_finished={})
 statuses=[{'pending_items':0},{'pending_items':0},{'pending_items':1},{'pending_items':1},{'pending_items':0}]
 with patch.object(OffloadingConnector,'update_connector_output',lambda *a:None),patch.object(mod,'record_async_scoring_event') as emit:
  for i,status in enumerate(statuses):
   meta=SemanticWorkerMetadata(async_scoring_status=status,async_scoring_events=[{'event':'proof','step':i}],score_group_size=1,score_contributors=1)
   c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
   assert meta.async_scoring_status==status
  values=[args.args[0] for args in emit.call_args_list]
  assert [v['pending_items'] for v in values if v['event']=='status']==[0,1,0]
  assert [v['step'] for v in values if v['event']=='proof']==list(range(5))
