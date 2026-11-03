"""Native CPU contracts. Only native CUDA transfer-handler construction is stubbed.
Scheduler, ARC manager, requests, BlockPool, metadata/bind/consumer are real.
CPU query tensors substitute the model probe; no GPU correctness claim here.
"""
from contextlib import ExitStack
from types import SimpleNamespace as N
from unittest.mock import patch
import copy,time
import pytest,torch
from semantic_offload.worker import SemanticOffloadingWorker
from semantic_offload.connector import SemanticOffloadingConnector,SemanticOffloadingConnectorScheduler,SemanticOffloadingConnectorWorker
from semantic_offload.async_scoring import AsyncScoringController
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import CanonicalKVCaches,ReqContext,OffloadingKVEventsConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.request import Request,RequestStatus
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import KVConnectorOutput
from tests.test_audit import manager,worker,consume
from tests.test_async_scoring import _key,_summary
from tests.test_step_1_5_prefetch import _wait_hint
from semantic_offload.connector import _observed_prefix_digest as observed_prefix_digest

NS={'model_revision':'model','tokenizer_revision':'tokenizer','adapter_namespace':'none','layout_namespace':'dense-tp1-bf16-block16'}

def native(schedule='opportunity',method='mean'):
 keys_a=[_key(i) for i in range(101,117)];keys_b=[_key(i) for i in range(201,217)];m=manager(keys_a+keys_b,32)
 cfg=N(parallel_config=N(tensor_parallel_size=1,pipeline_parallel_size=1,world_size=1,rank=0),model_config=N(model='cpu-contract',get_total_num_kv_heads=lambda:1,use_mla=False),compilation_config=N(static_forward_context={}),cache_config=N(block_size=1,enable_prefix_caching=True),speculative_config=None,use_v2_model_runner=False)
 extra={'eviction_policy':'arc','shared_prefetch_scorer':True,'preservation_events':True,'prefetch_trigger':'known_prefix_wait_hint','prefetch_selector':'semantic','prefetch_budget_fraction':.05,'scoring_schedule':schedule,'scoring_mode':'asynchronous','admission_namespace':NS,'soft_retention':'off'}
 spec=N(extra_config=extra,replicated_layout=False,cpu_page_size_per_worker=128,kv_bytes_per_chunk=128,tokens_per_block=[1],tokens_per_hash=1,blocks_per_chunk=1,offload_prompt_only=False,kv_events_config=OffloadingKVEventsConfig(False,False),get_manager=lambda:m)
 cache=KVCacheConfig(num_blocks=320,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(layer_names=['layer'],kv_cache_spec=FullAttentionSpec(block_size=1,num_kv_heads=1,head_size=4,dtype=torch.float32))])
 sched=SemanticOffloadingConnectorScheduler(spec,cfg,cache);sched.gpu_block_pool=BlockPool(320,True,1)
 # Actual native CPUOffloadingWorker constructor runs; hardware transfer backend
 # has no CUDA buffers in CPU tests and is the only constructor stub.
 with patch('vllm.v1.kv_offload.cpu.gpu_worker.SingleDirectionOffloadingHandler',return_value=N()):
  w=SemanticOffloadingWorker(CanonicalKVCaches(tensors=[],group_data_refs=[]),1,32,cfg,scoring_mode='asynchronous',scoring_schedule=schedule,method=method)
 w.durable_summaries={k:[_summary(2. if k in keys_a else -.5)] for k in keys_a+keys_b};w._durable_summary_publications={k:(0,1,i) for i,k in enumerate(keys_a+keys_b)}
 cw=SemanticOffloadingConnectorWorker(spec,cache);cw.worker=w
 # Thin role facade needs no model/transport init; its REAL bind/build/update
 # methods and native connector-worker/scheduler methods execute unpatched.
 c=SemanticOffloadingConnector.__new__(SemanticOffloadingConnector);c.connector_scheduler=sched;c.connector_worker=cw;c._score_group_size=1
 return c,sched,w,m,keys_a,keys_b

def source(sched,label,keys):
 tokens=list(range(len(keys)+1));prefix=observed_prefix_digest(tokens[:len(keys)],sched._wait_hint_namespace)
 req=Request(request_id='warm-'+label,prompt_token_ids=tokens,sampling_params=SamplingParams(max_tokens=1),pooling_params=None)
 req.kv_transfer_params={'run_id':'run','cell_id':'cell','session_id':'session-'+label,'generation':0,'observed_prefix_id':prefix,'observed_prefix_boundary_tokens':len(keys)}
 from vllm.v1.kv_offload.base import get_offload_block_hash
 req.block_hashes=[get_offload_block_hash(k) for k in keys]
 sched.on_new_request(req);sched._req_status[req.request_id].update_offload_keys()
 return req,prefix

def compete():
 c,s,w,m,a,b=native();requests=[]
 for keys,label in [(a,'a'),(b,'b')]:requests.append((source(s,label,keys),label))
 c.bind_connector_metadata(s.build_connector_meta(SchedulerOutput.make_empty()))
 w._on_queries_captured(['warm-a','warm-b'],torch.ones(2,1,4));stamps={r:v[0] for r,v in w._query_snapshots.items()}
 for (req,prefix),label in requests:
  req.status=RequestStatus.FINISHED_STOPPED;s.request_finished(req)
  assert s.publish_wait_hint(_wait_hint(s,prefix,16,session_id='session-'+label,event_id=label*64))['status']=='accepted'
 first=s.build_connector_meta(SchedulerOutput.make_empty())
 assert first.scoring_requests=={'warm-a':a,'warm-b':b} or first.scoring_requests=={'warm-b':b,'warm-a':a}
 assert all(x.status=='pending' for x in s._wait_prefetch.values())
 assert not first.prefetch_load_jobs
 c.bind_connector_metadata(first);w._drain_async_scores(wait=True)
 meta=c.build_connector_worker_meta();assert len(meta.async_score_observations)==2
 for o in meta.async_score_observations:assert o['snapshot_captured_ns']==stamps[o['request_ids'][0]]
 return c,s,w,m,a,b,meta

def test_actual_opportunity_scheduler_metadata_worker_consumer_exact_winner_and_ema():
 c,s,w,m,a,b,meta=compete()
 try:
  assert 'arc' in type(m._policy).__name__.lower()
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
  assert m.relevance_ema['mean'][a[0]]>m.relevance_ema['mean'][b[0]]
  values=copy.deepcopy(m.relevance_ema)
  # Duplicate native metadata cannot apply the EMA twice.
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta));assert m.relevance_ema==values
  s.build_connector_meta(SchedulerOutput.make_empty())
  flight=[x for x in s._wait_prefetch.values() if x.status=='in_flight']
  assert len(flight)==1 and flight[0].source.source_request_id=='warm-a'
  assert list(flight[0].keys)==a and len(flight[0].gpu_block_ids)==16
 finally:w._async_scoring.shutdown()

@pytest.mark.parametrize('action',['reinsert','same_id_start','cancel','reset','expired_lease','reorder'])
def test_real_opportunity_delayed_consumer_invalidations(action):
 c,s,w,m,a,b,meta=compete()
 try:
  if action=='reinsert':
   ctx=ReqContext(req_id='pressure');other=[_key(i) for i in range(501,533)]
   for keys in [other,a+b]:
    for k in keys:m.lookup(k,ctx);m.lookup(k,ctx)
    out=m.prepare_store(keys,ctx);assert out;m.complete_store(out.keys_to_store,ctx)
  elif action=='same_id_start':
   source(s,'a',a);source(s,'b',b)
   c.bind_connector_metadata(s.build_connector_meta(SchedulerOutput.make_empty()))
   assert not w._query_snapshots # before the next stride callback
  elif action=='cancel':
   for label in ['a','b']:s._score_request_finished(N(request_id='warm-'+label,status='FINISHED_ABORTED'))
  elif action=='reset':s.reset_cache()
  elif action=='expired_lease':
   for state in s._wait_prefetch.values():state.status='retired'
  elif action=='reorder':
   obs=sorted(meta.async_score_observations,key=lambda o:o['observation_sequence']);meta.async_score_observations=[obs[1]]
   c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta));saved=copy.deepcopy(m.relevance_ema)
   meta.async_score_observations=[obs[0]];c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta));assert m.relevance_ema==saved;return
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta));assert not m.relevance_ema
 finally:w._async_scoring.shutdown()

def test_actual_scheduler_bounded_50ms_timeout_with_no_fake_scores():
 c,s,w,m,a,b,meta=compete()
 try:
  first=min(s._decision_scoring_attempts.values())
  with patch('semantic_offload.connector.time.monotonic_ns',return_value=first+49_000_000):s.build_connector_meta(SchedulerOutput.make_empty())
  assert all(x.status=='pending' for x in s._wait_prefetch.values())
  with patch('semantic_offload.connector.time.monotonic_ns',return_value=first+51_000_000):s.build_connector_meta(SchedulerOutput.make_empty())
  assert any(x.status=='in_flight' for x in s._wait_prefetch.values()) and not m.relevance_ema
 finally:w._async_scoring.shutdown()

@pytest.mark.parametrize('action',['reinsert','same_id_start','abort','reset','duplicate'])
def test_continuous_async_publication_in_transit_is_bound_to_native_context(action):
 c,s,w,m,a,b=native(schedule='continuous')
 try:
  req,_=source(s,'a',a);c.bind_connector_metadata(s.build_connector_meta(SchedulerOutput.make_empty()))
  w._on_queries_captured(['warm-a'],torch.ones(1,1,4));w._drain_async_scores(wait=True);meta=c.build_connector_worker_meta();assert meta.async_score_observations
  if action=='reinsert':
   ctx=ReqContext(req_id='x');other=[_key(i) for i in range(600,632)]
   for keys in [other,a+b]:
    for k in keys:m.lookup(k,ctx);m.lookup(k,ctx)
    out=m.prepare_store(keys,ctx);m.complete_store(out.keys_to_store,ctx)
  elif action=='same_id_start':source(s,'a',a)
  elif action=='abort':req.status=RequestStatus.FINISHED_ABORTED;s.request_finished(req)
  elif action=='reset':s.reset_cache()
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
  if action=='duplicate':
   assert m.relevance_ema;saved=copy.deepcopy(m.relevance_ema);c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta));assert m.relevance_ema==saved
  else:assert not m.relevance_ema
 finally:w._async_scoring.shutdown()

def test_query100ms_async1000ms_uses_original_minimum_expiry():
 w=worker()
 try:
  w._query_snapshot_max_age_ns=100_000_000;stamp=time.monotonic_ns();w._on_queries_captured(['source'],torch.ones(1,1,4));w._query_snapshots['source']=(stamp,torch.ones(1,4))
  with patch('semantic_offload.worker.time.monotonic_ns',return_value=stamp+90_000_000):w.receive_score_requests({'source':list(w.durable_summaries)})
  with patch('semantic_offload.worker.time.monotonic_ns',return_value=stamp+200_000_000):w._drain_async_scores(wait=True)
  assert not w.pop_async_score_observations()
 finally:w._async_scoring.shutdown()

@pytest.mark.parametrize('method',['mean','minmax','cuboid_mean'])
def test_R16_K1024_H8_D128_real_payload_exceeds_64MiB_and_is_rejected(method):
 w=worker();w._method=method
 try:
  w._async_scoring.shutdown();w._async_scoring=AsyncScoringController(w._score_async_payload,max_queue_items=2,max_snapshot_bytes=64*1024*1024,workers=1)
  w.durable_summaries={_key(i):[_summary(1.) for _ in range(8)] for i in range(1024)}
  for row in w.durable_summaries.values():
   for s in row:
    for f in ['mean','min','max','mad']:setattr(s,f,torch.ones(128))
  w._durable_summary_publications={k:(0,1,i) for i,k in enumerate(w.durable_summaries)}
  assert w._submit_async_scoring([str(i) for i in range(16)],torch.ones(16,8,128)) is None
  telemetry=w.pop_async_scoring_telemetry();assert any(e.get('submitted_snapshot_bytes',0)>64*1024*1024 for e in telemetry)
  assert w._async_scoring.stats()['pending_snapshot_bytes']==0
 finally:w._async_scoring.shutdown()


def test_continuous_completed_source_transport_lease_preserves_final_observation():
 c,s,w,m,a,b=native(schedule='continuous')
 try:
  req,_=source(s,'a',a);c.bind_connector_metadata(s.build_connector_meta(SchedulerOutput.make_empty()))
  w._on_queries_captured(['warm-a'],torch.ones(1,1,4));w._drain_async_scores(wait=True)
  meta=c.build_connector_worker_meta();req.status=RequestStatus.FINISHED_STOPPED;s.request_finished(req)
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta));assert m.relevance_ema['mean'][a[0]]==8.
 finally:w._async_scoring.shutdown()


def test_summary_shape_mismatch_cannot_escape_workspace_bound():
 w=worker()
 try:
  with pytest.raises(ValueError,match='heads/dim'):w._submit_async_scoring(['source'],torch.ones(1,8,128))
  assert w._async_scoring.stats()['pending_snapshot_bytes']==0
 finally:w._async_scoring.shutdown()


def test_ordered_real_consumer_observations_match_existing_ema_composition():
 from semantic_offload.connector import compose_relevance_updates,compose_latest_relevance
 c,s,w,m,a,b=native(schedule='continuous')
 try:
  source(s,'a',a);c.bind_connector_metadata(s.build_connector_meta(SchedulerOutput.make_empty()))
  expected=[]
  for value in [1.,.5]:
   w._on_queries_captured(['warm-a'],torch.full((1,1,4),value));w._drain_async_scores(wait=True)
   meta=c.build_connector_worker_meta();expected.extend(copy.deepcopy(meta.async_score_observations))
   c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
  other=manager(a+b,32)
  for o in expected:other.apply_relevance_updates(compose_relevance_updates(o['scores'],'default',0.),compose_latest_relevance(o['scores']))
  assert m.relevance_ema==other.relevance_ema
 finally:w._async_scoring.shutdown()


@pytest.mark.parametrize('field',['async_contract','source_tokens','residency_versions','transport_epoch'])
def test_native_async_consumer_missing_mandatory_identity_is_fail_closed(field):
 c,s,w,m,a,b,meta=compete()
 try:
  for o in meta.async_score_observations:o.pop(field)
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta));assert not m.relevance_ema
 finally:w._async_scoring.shutdown()
