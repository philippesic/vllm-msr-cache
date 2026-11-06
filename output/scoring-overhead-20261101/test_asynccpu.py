import os, threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import pytest, torch
from tests.test_async_scoring import _worker, _key, _summary
from semantic_offload.async_scoring import AsyncScoringController


def controlled(w, budget=1_000_000, workers=2):
    w._scoring_schedule='continuous'
    w._async_scoring.shutdown()
    release=threading.Event()
    original=w._score_async_payload
    def scorer(payload):
        assert release.wait(5)
        return original(payload)
    w._async_scoring=AsyncScoringController(scorer,max_queue_items=2,
        max_snapshot_bytes=budget,workers=workers)
    return release


def test_cpu_payload_backend_frozen_and_matches_dense_scores():
    w=_worker('asynchronous');release=controlled(w)
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='cpu'):
            w._submit_async_scoring(['a'],torch.ones(1,1,4))
            w._submit_async_scoring(['b'],torch.ones(1,1,4)*2)
        pending=list(w._async_scoring._pending.values())
        assert all(job.snapshot.payload.math_backend=='cpu' for job in pending)
        release.set()
        a,b=[job.future.result(timeout=5).result for job in pending]
        assert a[0]['mean']['a']==[(_key(1),8.),(_key(2),-4.)]
        assert b[0]['mean']['b']==[(_key(1),16.),(_key(2),-8.)]
        assert w._async_cpu_bank is not None
        packed=pending[0].snapshot.payload.packed_fields[0][1]
        cached=w._async_cpu_bank[1]
        assert cached.device.type=='cpu' and cached.data_ptr()!=packed.data_ptr()
        assert not cached.requires_grad and cached.grad_fn is None
    finally:release.set();w._async_scoring.shutdown()


def test_concurrent_bank_copies_reuse_one_owned_cpu_allocation():
    w=_worker('asynchronous');release=controlled(w)
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='cpu'):
            w._submit_async_scoring(['a'],torch.ones(1,1,4))
        packed=next(iter(w._async_scoring._pending.values())).snapshot.payload.packed_fields[0][1]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(w._async_cpu_mean_bank,packed)
            second=pool.submit(w._async_cpu_mean_bank,packed)
            a,b=first.result(),second.result()
        assert a is b and a.data_ptr()!=packed.data_ptr()
    finally:release.set();w._async_scoring.shutdown()


def test_old_job_keeps_own_cpu_bank_when_latest_bank_replaces_cache():
    w=_worker('asynchronous');release=controlled(w)
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='cpu'):
            w._submit_async_scoring(['a'],torch.ones(1,1,4))
            first=next(iter(w._async_scoring._pending.values())).snapshot.payload
            old_cpu=w._async_cpu_mean_bank(first.packed_fields[0][1])
            w.durable_summaries[_key(1)]=[_summary(9.)]
            w._durable_summary_publications[_key(1)]=(0,8,6)
            w._submit_async_scoring(['b'],torch.ones(1,1,4))
            second=list(w._async_scoring._pending.values())[1].snapshot.payload
            new_cpu=w._async_cpu_mean_bank(second.packed_fields[0][1])
            assert new_cpu is w._async_cpu_bank[1] and new_cpu is not old_cpu
            assert old_cpu[0].eq(2).all() and new_cpu[0].eq(9).all()
            # An older job can copy for itself without replacing latest metadata.
            assert w._async_cpu_mean_bank(first.packed_fields[0][1]) is not new_cpu
            assert w._async_cpu_bank[1] is new_cpu
        release.set();w._drain_async_scores(wait=True)
        observations=w.pop_async_score_observations()
        assert all(key!=_key(1) for key,_ in observations[0]['scores']['mean']['a'])
        assert observations[1]['scores']['mean']['b'][0]==(_key(1),36.)
    finally:release.set();w._async_scoring.shutdown()


def test_extra_cpu_copies_and_workspace_are_charged_before_packing():
    gpu=_worker('asynchronous');release0=controlled(gpu)
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='gpu'):
            gpu._submit_async_scoring(['a'],torch.ones(1,1,4))
        gpu_charge=next(iter(gpu._async_scoring._pending.values())).snapshot.snapshot_bytes
    finally:release0.set();gpu._async_scoring.shutdown()
    cpu=_worker('asynchronous');release=controlled(cpu,budget=gpu_charge)
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='cpu'), \
             patch('torch.stack',side_effect=AssertionError('rejected CPU workspace allocated bank')):
            assert cpu._submit_async_scoring(['a'],torch.ones(1,1,4)) is None
        assert cpu._async_scoring.stats()['pending_snapshot_bytes']==0
        assert getattr(cpu,'_async_packed_bank',None) is None
    finally:release.set();cpu._async_scoring.shutdown()


def test_cpu_bank_drops_after_reset_drains_readers():
    w=_worker('asynchronous')
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='cpu'):
            w._submit_async_scoring(['a'],torch.ones(1,1,4))
        w._drain_async_scores(wait=True)
        assert w._async_cpu_bank is not None
        w.reset_semantic_state()
        assert w._async_cpu_bank is None and w._async_packed_bank is None
        assert w._async_scoring.stats()['pending_snapshot_bytes']==0
    finally:w._async_scoring.shutdown()


@pytest.mark.parametrize('aggregation',['mean','max'])
def test_only_mean_head_uses_cpu_and_other_aggregation_keeps_gpu_fallback(aggregation):
    w=_worker('asynchronous');release=controlled(w);w._head_aggregation=aggregation
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='cpu'):
            w._submit_async_scoring(['a'],torch.ones(1,1,4))
        payload=next(iter(w._async_scoring._pending.values())).snapshot.payload
        assert payload.math_backend==('cpu' if aggregation=='mean' else 'gpu')
    finally:release.set();w._async_scoring.shutdown()


def test_random_multihead_scores_match_unchanged_cpu_reference_bitwise():
    w=_worker('asynchronous');release=controlled(w,budget=16_000_000)
    generator=torch.Generator().manual_seed(901)
    queries=torch.randn(3,2,128,generator=generator)
    means=torch.randn(199,2,128,generator=generator)
    keys=tuple(_key(i) for i in range(199))
    w.durable_summaries={key:[type('Summary',(),{'mean':means[row,head]})()
        for head in range(2)] for row,key in enumerate(keys)}
    w._durable_summary_publications={key:(0,1,index) for index,key in enumerate(keys)}
    try:
        with patch.dict(os.environ,SEMANTIC_ASYNC_MATH='cpu'):
            w._submit_async_scoring(['a','b','c'],queries)
        pending=next(iter(w._async_scoring._pending.values()))
        reference=(queries[:,None]*means[None]).sum(-1).mean(-1)
        release.set();scores,counts=pending.future.result(timeout=5).result
        for row,request in enumerate(['a','b','c']):
            expected=sorted(zip(keys,reference[row].tolist()),key=lambda item:(-item[1],item[0]))
            assert scores['mean'][request]==expected
            assert counts['mean'][request]==2
    finally:release.set();w._async_scoring.shutdown()
