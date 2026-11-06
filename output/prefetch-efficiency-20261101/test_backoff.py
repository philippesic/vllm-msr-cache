import ast,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
H=Path(__file__).resolve().parent
s=ast.parse((H/'sources/reclaim-backoff/semantic_offload/connector.py').read_text())
names=['_free_wait_prefetch','_wait_complete_staging_available']
methods=[n for n in ast.walk(s) if isinstance(n,ast.FunctionDef) and n.name in names]
clock=NS(monotonic_ns=lambda:100)
ns={'_WaitPrefetchState':NS,'hashlib':hashlib,'time':clock,'record_wait_hint_event':lambda x:None,'_any_local_gpu_key':lambda p,k:False}
exec(compile(ast.Module(body=methods,type_ignores=[]),'<backoff>','exec'),ns)
free=ns['_free_wait_prefetch'];available=ns['_wait_complete_staging_available']
def fixture(keys=(b'a',),reason='demand_reclaim',blocks=(8,)):
 st=NS(job_id=None,lease=NS(hint=NS(event_id='e'),expire=lambda:None),bound_request_id=None,destination_released=False,gpu_block_ids=blocks,source=NS(keys=keys))
 freed=[]
 obj=NS(_release_wait_prefetch_protection=lambda s:None,_prefetched={},_wait_prefetch={'e':st},_wait_hint_request_bindings={},gpu_block_pool=NS(),_free_prefetch_blocks=lambda ids:freed.extend(ids),_prefetch_reserved_blocks=len(blocks),_prefetch_bytes_per_key=1,_wait_state_identity=lambda s:{})
 free(obj,st,reason);return obj,st,freed
def check(obj,keys=(b'a',)):return available(obj,keys,keys,keys,0,())
def test_reclaim_suppresses_exact_source_only():
 obj,st,freed=fixture();assert freed==[8] and obj._prefetch_reserved_blocks==0 and st.destination_released
 assert not check(obj);assert check(obj,(b'b',))
def test_expiry_reopens_same_source():
 obj,_,_=fixture();clock.monotonic_ns=lambda:5_000_000_100
 try:assert check(obj);assert not obj._reclaimed_source_backoff
 finally:clock.monotonic_ns=lambda:100
def test_other_retirement_and_no_release_do_not_penalize():
 for reason,blocks in [('retention_expired',(8,)),('demand_reclaim',())]:
  obj,_,_=fixture(reason=reason,blocks=blocks);assert check(obj);assert not getattr(obj,'_reclaimed_source_backoff',{})
def test_bounded_memory():
 obj,_,_=fixture();obj._reclaimed_source_backoff={i:10000 for i in range(256)}
 # Reuse the genuine retirement method with a new completed state.
 other,st,_=fixture((b'new',));st.destination_released=False;st.gpu_block_ids=(9,);obj._prefetch_reserved_blocks=1;obj._wait_prefetch={'e':st}
 free(obj,st,'demand_reclaim');assert len(obj._reclaimed_source_backoff)==256;assert 0 not in obj._reclaimed_source_backoff
