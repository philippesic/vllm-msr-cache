"""Native CPU cooperative wait contracts; actual scoring/metadata suite retained."""
import threading,time,weakref
from types import SimpleNamespace as N
from unittest.mock import patch
import pytest
from semantic_offload.async_scoring import AsyncScoringController,wait_for_owned_fence

def test_cooperative_wait_queries_pending_only_then_synchronizes():
 calls=[];states=iter([False,False,True])
 fence=N(query=lambda:calls.append('query') or next(states),synchronize=lambda:calls.append('sync'))
 with patch('semantic_offload.async_scoring.time.sleep',side_effect=lambda delay:calls.append(delay)):
  wait_for_owned_fence(fence)
 assert calls==['query',.0005,'query',.0005,'query','sync']

def test_ready_event_has_no_sleep_and_generic_fence_unchanged():
 calls=[]
 with patch('semantic_offload.async_scoring.time.sleep',side_effect=AssertionError('unexpected sleep')):
  wait_for_owned_fence(N(query=lambda:True,synchronize=lambda:calls.append('ready')))
  wait_for_owned_fence(N(synchronize=lambda:calls.append('generic')))
 assert calls==['ready','generic']

def test_query_error_fences_all_inputs_before_controller_releases_budget():
 entered=threading.Event();release=threading.Event();seen=[]
 class Payload:pass
 payload=Payload();ref=weakref.ref(payload)
 def badquery():raise RuntimeError('query error')
 def sync():entered.set();assert release.wait(3);seen.append('bad_synced')
 c=AsyncScoringController(lambda p:pytest.fail('must not score'),max_queue_items=1,max_snapshot_bytes=100,workers=1)
 try:
  assert c.submit(payload,generation=0,snapshot_bytes=80,key_versions={},observation_available_ns=time.monotonic_ns(),readiness_fences=[N(query=badquery,synchronize=sync),N(synchronize=lambda:seen.append('other_synced'))])==0
  del payload;assert entered.wait(3);assert ref() is not None
  assert c.stats()['pending_snapshot_bytes']==80 and c.drain_ready()==([],[])
  release.set();done,fail=c.drain_all();assert not done and len(fail)==1
  assert 'other_synced' in seen and seen.count('bad_synced')==2
  assert c.stats()['pending_snapshot_bytes']==0
 finally:release.set();c.shutdown()

def test_reset_waits_for_cooperative_event_and_holds_charge():
 ready=threading.Event();queried=threading.Event();reset_entered=threading.Event();reset_done=threading.Event()
 def query():queried.set();return ready.is_set()
 c=AsyncScoringController(lambda value:value,max_queue_items=1,max_snapshot_bytes=16,workers=1)
 try:
  c.submit(1,generation=0,snapshot_bytes=16,key_versions={},observation_available_ns=time.monotonic_ns(),readiness_fences=[N(query=query,synchronize=lambda:None)])
  assert queried.wait(3)
  def reset():reset_entered.set();c.reset();reset_done.set()
  t=threading.Thread(target=reset);t.start();assert reset_entered.wait(3)
  assert not reset_done.is_set() and c.stats()['pending_snapshot_bytes']==16
  ready.set();assert reset_done.wait(3);t.join();assert c.drain_ready()==([],[]) and c.stats()['pending_snapshot_bytes']==0
 finally:ready.set();c.shutdown()
