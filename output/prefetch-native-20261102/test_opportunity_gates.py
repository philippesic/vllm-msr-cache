import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent

@pytest.fixture
def case():
 t=ast.parse((H/'sources/opportunity-gates/semantic_offload/connector.py').read_text())
 m=next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name=='_retry_prefix_mean_prefetches')
 reads=[];attempts=[]
 ns={'PROFILE':NS(count=lambda *a:None),'record_wait_hint_event':lambda x:None,'_event_key':lambda k:k,'_ready_resident_prefix':lambda manager,keys:(reads.append(tuple(keys)) or keys)}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[m],type_ignores=[])),'gates','exec'),ns)
 w=NS(_prefetch_require_full_prefix=True,_admitted_prefill_in_progress=lambda:False,gpu_block_pool=NS(num_gpu_blocks=100),_prefetch_budget_fraction=.6,_prefetch_reserved_blocks=10,_prefetch_headroom_blocks=1,_prefetch_bytes_per_key=1,_prefetch_free_capacity=lambda:20,manager=None,_wait_local_prefix_valid=lambda s:True,_wait_required_staging_keys=lambda s:s.publication_candidate_keys,_prefix_mean_descriptor=lambda s,k:{'keys':k,'residency_versions':{}},_semantic_wait_nomination_candidate=lambda s,n:({'event_id':s.lease.hint.event_id,'eligible_resident_prefix_keys':s.publication_candidate_keys},1.),_attempt_wait_prefetch=lambda s,**kw:(attempts.append(s.lease.hint.event_id) or False),_wait_hint_exact_source_identity=lambda s:s.lease.hint.event_id)
 states=[NS(publication_candidate_keys=tuple(range(n)),lease=NS(hint=NS(event_id=name))) for n,name in [(12,'small'),(25,'large')]]
 return ns['_retry_prefix_mean_prefetches'],w,states,reads,attempts

def test_prefill_priority_prevents_all_descriptor_reads(case):
 fn,w,s,reads,a=case;w._admitted_prefill_in_progress=lambda:True
 w._prefetch_free_capacity=lambda:(_ for _ in ()).throw(AssertionError('unneededcapacityscan'))
 fn(w,s,1);assert not reads and not a

@pytest.mark.parametrize('capacity',[0,1])
def test_headroom_and_no_space_prevent_descriptor_reads(case,capacity):
 fn,w,s,reads,a=case;w._prefetch_free_capacity=lambda:capacity;fn(w,s,1);assert not reads and not a

def test_only_full_sources_that_can_fit_are_scored(case):
 fn,w,s,reads,a=case;fn(w,s,1);assert reads==[tuple(range(12))] and a==['small']

def test_later_capacity_opportunity_is_not_cached_away(case):
 fn,w,s,reads,a=case;fn(w,s,1);w._prefetch_free_capacity=lambda:40;fn(w,s,2)
 assert reads[-2:]==[tuple(range(12)),tuple(range(25))] and a[-2:]==['large','small']

def test_partial_mode_keeps_partial_candidate_eligibility(case):
 fn,w,s,reads,a=case;w._prefetch_require_full_prefix=False;fn(w,s,1)
 assert len(reads)==2 and a==['large','small']

@pytest.mark.parametrize('free',[0,1,6,13,18,26,38,60])
def test_old_and_new_selector_choose_same_fitting_full_sources(free):
    """Known-too-large sources must not affect feasible raw-dot-score order."""
    chosen=[]
    for label in ['fused-mean','opportunity-gates']:
        tree=ast.parse((H/'sources'/label/'semantic_offload/connector.py').read_text())
        node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_retry_prefix_mean_prefetches')
        ns={'PROFILE':NS(count=lambda *a:None),'record_wait_hint_event':lambda _:None,'_event_key':lambda k:k,'_ready_resident_prefix':lambda manager,keys:keys}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'selector','exec'),ns)
        remaining=[max(0,free-1)];selected=[]
        states=[NS(publication_candidate_keys=tuple(range(n)),lease=NS(hint=NS(event_id=name))) for n,name in [(5,'small'),(12,'medium'),(25,'large')]]
        def attempt(state,**kw):
            n=len(state.publication_candidate_keys)
            if n>remaining[0]:return False
            selected.append(state.lease.hint.event_id);remaining[0]-=n
            return False  # Suppress logging-only chosen detail; ownership modeled above.
        w=NS(_prefetch_require_full_prefix=True,_admitted_prefill_in_progress=lambda:False,gpu_block_pool=NS(num_gpu_blocks=100),_prefetch_budget_fraction=1.,_prefetch_reserved_blocks=0,_prefetch_headroom_blocks=1,_prefetch_bytes_per_key=1,_prefetch_free_capacity=lambda:free,manager=None,_wait_local_prefix_valid=lambda s:True,_wait_required_staging_keys=lambda s:s.publication_candidate_keys,_prefix_mean_descriptor=lambda s,k:{'keys':k,'residency_versions':{}},_semantic_wait_nomination_candidate=lambda s,n:({'event_id':s.lease.hint.event_id,'eligible_resident_prefix_keys':s.publication_candidate_keys},float(len(s.publication_candidate_keys))),_attempt_wait_prefetch=attempt,_wait_hint_exact_source_identity=lambda s:s.lease.hint.event_id)
        ns['_retry_prefix_mean_prefetches'](w,states,1);chosen.append(selected)
    assert chosen[0]==chosen[1]
