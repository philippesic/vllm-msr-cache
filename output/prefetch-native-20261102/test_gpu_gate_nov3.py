from unittest.mock import patch
import pytest
from tests.test_wait_hint_cpu_recency import retention
from test_gpu_coverage_nov3 import snapshot


def enable(retention):
 s,m,p,keys,hint=retention;s._wait_hint_cpu_retention_gpu_gate=True
 return s,m,p,keys,hint

def test_full_GPU_prefix_skips_CPU_touch_and_does_not_mutate_queues(retention):
 s,m,p,keys,hint=enable(retention);before=list(m._policy.t1)
 with patch('semantic_offload.connector._local_gpu_prefix',return_value=(4,[1,2,3,4],True)),patch.object(m,'refresh_wait_hint_cpu_recency',side_effect=AssertionError('unexpected CPUtouch')):r=s.publish_wait_hint(hint)
 assert list(m._policy.t1)==before and not m._policy.t2
 assert r['cpu_retention_skipped_full_gpu'] and r['cpu_retention_skipped_ready_prefix_keys']==4
 assert r['cpu_retention_ready_prefix_keys']==4 and r['ready_cpu_prefix_keys']==0
 assert r['reserved_gpu_blocks']==r['speculative_transfer_bytes']==0 and not s._jobs

@pytest.mark.parametrize('count,available',[(2,True),(0,True),(0,False)])
def test_partial_miss_unavailable_fall_back_to_original_CPU_roots(retention,count,available):
 s,m,p,keys,hint=enable(retention)
 with patch('semantic_offload.connector._local_gpu_prefix',return_value=(count,list(range(count)),available)),patch.object(m,'refresh_wait_hint_cpu_recency',wraps=m.refresh_wait_hint_cpu_recency) as refresh:r=s.publish_wait_hint(hint)
 assert refresh.call_args.args[0]==keys and refresh.call_count==1
 assert r['ready_cpu_prefix_keys']==4 and not r['cpu_retention_skipped_full_gpu']
 assert r['cpu_retention_gpu_cached_prefix_keys']==(count if available else None)
 assert r['cpu_retention_gpu_index_available'] is available
 assert list(m._policy.t1)==list(reversed(keys)) and not m._policy.t2

def test_full_then_partial_GPU_starts_CPU_advice_without_reconstructing_evicted_data(retention):
 s,m,p,keys,hint=enable(retention)
 with patch('semantic_offload.connector._local_gpu_prefix',return_value=(4,[1,2,3,4],True)):s.publish_wait_hint(hint)
 assert m._policy.evict(1,set())[0][0]==keys[0]
 before=list(m._policy.t1);hint=dict(hint,event_id='f'*64)
 with patch('semantic_offload.connector._local_gpu_prefix',return_value=(0,[],True)):r=s.publish_wait_hint(hint)
 assert not r['cpu_retention_skipped_full_gpu'] and r['cpu_retention_ready_prefix_keys']==r['ready_cpu_prefix_keys']==0
 assert m._policy.get(keys[0]) is None and list(m._policy.t1)==before

def test_default_off_never_queries_gpu_index(retention):
 s,m,p,keys,hint=retention
 with patch('semantic_offload.connector._local_gpu_prefix',side_effect=AssertionError('default lookup')):r=s.publish_wait_hint(hint)
 assert r['ready_cpu_prefix_keys']==4 and 'cpu_retention_gpu_index_available' not in r

def test_skipped_replayed_event_never_queries_or_touches_again(retention):
 s,m,p,keys,hint=enable(retention)
 with patch('semantic_offload.connector._local_gpu_prefix',return_value=(4,[1,2,3,4],True)):first=s.publish_wait_hint(hint)
 with patch('semantic_offload.connector._local_gpu_prefix',side_effect=AssertionError('replayed lookup')),patch.object(m,'refresh_wait_hint_cpu_recency',side_effect=AssertionError('replayed CPUtouch')):second=s.publish_wait_hint(hint)
 assert first['ready_cpu_prefix_keys']==second['ready_cpu_prefix_keys']==0 and second['idempotent_replay']

@pytest.mark.parametrize('bad',['namespace','generation','source','future_clock','expired','global_disable'])
def test_causal_guards_precede_GPU_gate(retention,bad):
 s,m,p,keys,hint=enable(retention)
 if bad=='namespace':hint['namespace']=dict(hint['namespace'],model_revision='other')
 elif bad=='generation':s._observed_prefix_generations[('run','cell','session')]=1
 elif bad=='source':s._observed_prefixes.clear()
 elif bad=='future_clock':hint['input_publication_ns']+=10_000_000_000
 elif bad=='global_disable':s._prefetch_budget_fraction=0
 clock=hint['observation_available_ns']+(3_000_000_000 if bad=='expired' else 1)
 with patch('semantic_offload.connector.time.monotonic_ns',return_value=clock),patch('semantic_offload.connector._local_gpu_prefix',side_effect=AssertionError('invalid lookup')):r=s.publish_wait_hint(hint)
 assert r['reason']!='cpu_prefix_recency_refreshed'

@pytest.mark.parametrize('flag',[0,1,None,'true'])
def test_GPU_gate_flag_requires_boolean(flag):
 from tests.test_preservation_audit_mode import native
 with pytest.raises(TypeError,match='must be a JSON boolean'):native(wait_hint_cpu_retention_gpu_gate=flag)

def test_GPU_gate_requires_qualified_layout():
 from tests.test_preservation_audit_mode import native
 with pytest.raises(ValueError,match='dense TP1'):native(wait_hint_cpu_retention_gpu_gate=True)

def test_real_native_index_full_then_partial_is_readonly_audit_off():
 from tests.test_preservation_audit_mode import native,source,_wait_hint
 s,m=native(wait_hint_native_adoption=True,wait_hint_cpu_retention_gpu_gate=True)
 req,prefix,keys=source(s);ctx=s._req_status[req.request_id].req_context
 prepared=m.prepare_store(keys,ctx);m.complete_store(prepared.keys_to_store,ctx);s._register_observed_prefix(req)
 p=s.gpu_block_pool;b=p.get_new_blocks(2);p.cache_full_blocks(req,b,0,2,16,0);p.free_blocks(b)
 before=snapshot(p);qbefore=list(m._policy.t1);hint=_wait_hint(s,prefix,32,session_id='one')
 with patch.object(p,'get_new_blocks',side_effect=AssertionError('allocation')),patch.object(p,'touch',side_effect=AssertionError('refs')):r=s.publish_wait_hint(hint)
 assert snapshot(p)==before and list(m._policy.t1)==qbefore and r['cpu_retention_skipped_full_gpu'] and r['ready_cpu_prefix_keys']==0
 assert p._maybe_evict_cached_block(b[1]);before=snapshot(p)
 r=s.publish_wait_hint(dict(hint,event_id='e'*64))
 assert snapshot(p)==before and r['cpu_retention_gpu_cached_prefix_keys']==1 and r['ready_cpu_prefix_keys']==2
 assert not s._jobs and not s._wait_prefetch and not s._preservation_requests
 assert len(s._observed_prefix_generations)==len(s._observed_prefixes)==1

def test_actual_refresh_zero_is_distinct_from_ready_before_snapshot(retention):
 s,m,p,keys,hint=enable(retention);before=list(m._policy.t1)
 with patch('semantic_offload.connector._local_gpu_prefix',return_value=(0,[],False)),patch.object(m,'refresh_wait_hint_cpu_recency',return_value=0):r=s.publish_wait_hint(hint)
 assert r['cpu_retention_ready_prefix_keys']==4 and r['ready_cpu_prefix_keys']==0
 assert not r['cpu_retention_skipped_full_gpu'] and r['cpu_retention_gpu_cached_prefix_keys'] is None
 assert list(m._policy.t1)==before
