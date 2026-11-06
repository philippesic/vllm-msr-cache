"""Request-owned staging lifecycle and synchronous native lookup controls."""
import time
from dataclasses import replace
import pytest
from test_shared_prefix import fixture,base


def bound():
 f=fixture();f.sched._wait_hint_publish_at_lookup=True
 req=base._native_adoption_bind(f)
 assert f.state.status=='ready' and f.state.bound_native_owned_ns is not None
 assert not f.sched._published_wait_prefixes
 assert f.sched._prefetch_reserved_blocks==1
 return f,req


def test_no_publication_at_enqueue_then_exact_native_lookup_adoption():
 f,req=bound();oldrefs=f.local.ref_cnt
 f.sched.before_prefix_cache_lookup(req)
 assert f.state.status=='published' and f.sched._prefetch_reserved_blocks==0
 blocks=base._native_adoption_allocate(f,req)
 assert f.local.ref_cnt==oldrefs+1
 assert not f.sched._published_wait_prefixes
 f.cache.free(req)


def test_bound_actual_request_outlives_speculative_lease_without_future_refresh():
 f,req=bound();f.state.lease.expires_at_ns=time.monotonic_ns()-1
 f.sched._expire_wait_prefetches(time.monotonic_ns())
 assert f.state.status=='ready' and f.sched._prefetch_reserved_blocks==1
 f.sched.before_prefix_cache_lookup(req)
 assert f.state.status=='published'
 blocks=base._native_adoption_allocate(f,req);f.cache.free(req)


@pytest.mark.parametrize('bad',['different_object','progressed','prefix_evicted','fence_missing','cancelled','stale_generation'])
def test_lookup_invalid_boundary_releases_only_owned_suffix(bad):
 f,req=bound();pool=f.cache.block_pool;oldrefs=f.local.ref_cnt;ids=f.state.gpu_block_ids
 if bad=='different_object':req=base._native_adoption_request(req.request_id)
 elif bad=='progressed':req.num_computed_tokens=16
 elif bad=='prefix_evicted':pool.evict_blocks({f.local.block_id})
 elif bad=='fence_missing':f.state.transfer_record['success']=False
 elif bad=='cancelled':f.state.lease.cancel()
 elif bad=='stale_generation':f.sched._observed_prefix_generations[(f.state.lease.hint.run_id,f.state.lease.hint.cell_id,f.state.lease.hint.session_id)]+=1
 f.sched.before_prefix_cache_lookup(req)
 assert not f.sched._published_wait_prefixes and f.sched._prefetch_reserved_blocks==0
 assert all(pool.blocks[i].ref_cnt==0 for i in ids)
 assert f.local.ref_cnt==oldrefs


def test_finish_and_reset_release_bound_owner_without_native_lookup():
 for action in ['finish','reset']:
  f,req=bound();ids=f.state.gpu_block_ids;pool=f.cache.block_pool
  if action=='finish':f.sched.request_finished(req)
  else:
   from unittest.mock import patch
   from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
   with patch.object(OffloadingConnectorScheduler,'reset_cache'):f.sched.reset_cache()
  assert f.sched._prefetch_reserved_blocks==0
  assert not f.sched._wait_prefetch and not f.sched._wait_hint_request_bindings
  assert all(pool.blocks[i].ref_cnt==0 for i in ids)


def test_completion_after_bind_does_not_publish_until_native_lookup():
 from test_shared_prefix import finish
 f=fixture(complete=False);f.sched._wait_hint_publish_at_lookup=True
 req=base._native_adoption_bind(f)
 assert f.state.status=='in_flight'
 f.sched.before_prefix_cache_lookup(req)  # no wait, no publication
 assert not f.sched._published_wait_prefixes
 finish(f)
 assert f.state.status=='ready' and f.sched._prefetch_reserved_blocks==1
 assert not f.sched._published_wait_prefixes
 f.sched.before_prefix_cache_lookup(req)
 assert f.state.status=='published'
 base._native_adoption_allocate(f,req);f.cache.free(req)


def test_unbound_speculation_still_expires_at_original_ttl():
 f=fixture();f.sched._wait_hint_publish_at_lookup=True
 f.state.lease.expires_at_ns=time.monotonic_ns()-1
 f.sched._expire_wait_prefetches(time.monotonic_ns())
 assert f.state.status=='retired' and f.sched._prefetch_reserved_blocks==0


def test_publication_exception_cleans_owned_suffix_before_propagating(monkeypatch):
 f,req=bound();pool=f.cache.block_pool;ids=f.state.gpu_block_ids;refs=f.local.ref_cnt
 def fail(*a,**kw):raise RuntimeError('injected callback failure')
 monkeypatch.setattr(pool,'publish_completed_suffix',fail)
 with pytest.raises(RuntimeError,match='injected callback failure'):f.sched.before_prefix_cache_lookup(req)
 assert f.sched._prefetch_reserved_blocks==0 and not f.sched._wait_prefetch
 assert all(pool.blocks[i].ref_cnt==0 for i in ids) and f.local.ref_cnt==refs
