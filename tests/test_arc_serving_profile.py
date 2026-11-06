import json
import pytest
from benchmarks.research_checks.arc_serving_profile import EvictionProfile, snapshot, replay
from semantic_offload.manager import PreservationOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.base import ReqContext


def make_policy():
    p = ARCCachePolicy(32)
    for i in range(32):
        b = BlockStatus(i)
        b.ref_cnt = 1 if i < 4 else 0
        p.t1[i.to_bytes(32, 'little')] = b
    return p


def test_accounting_and_exact_replay(tmp_path):
    p, control = make_policy(), make_policy()
    profile = EvictionProfile(p, tmp_path)
    for n in [0, 1, 2, 4, 8, 32]:
        protected = {i.to_bytes(32, 'little') for i in range(8)}
        actual, expected = p.evict(n, protected), control.evict(n, protected)
        assert (None if actual is None else [k for k, b in actual]) == (None if expected is None else [k for k, b in expected])
        assert snapshot(p, n, protected) == snapshot(control, n, protected)
        profile.record_store(100, 50)
        profile.flush()
    rows = [json.loads(r) for r in (profile.directory / 'calls.jsonl').read_text().splitlines()]
    summary = json.loads((profile.directory / 'summary.json').read_text())
    assert len(rows) == summary['calls'] == 6
    assert summary['body_wall_ns'] == sum(r['body_wall_ns'] for r in rows)
    assert summary['body_cpu_ns'] == sum(r['body_cpu_ns'] for r in rows)
    assert summary['store_wall_ns'] == 600
    assert summary['observer_wall_ns'] >= 0
    for saved in profile.snapshots:
        assert replay(saved['state'], 3)['equivalent']
    for n in range(40):
        p.evict(2**n, set())
    assert len(profile.snapshots) <= 12


def test_exception_is_preserved_and_recorded(tmp_path):
    p = make_policy()
    def fail(n, protected):
        raise LookupError('expected')
    p.evict = fail
    profile = EvictionProfile(p, tmp_path)
    with pytest.raises(LookupError, match='expected'):
        p.evict(1, set())
    profile.flush()
    assert json.loads((profile.directory / 'calls.jsonl').read_text())['error'] == 'LookupError'


def test_manager_flushes_without_exit_hook(monkeypatch, tmp_path):
    monkeypatch.setenv('SEMANTIC_ARC_EVICTION_PROFILE_DIR', str(tmp_path))
    manager = PreservationOffloadingManager(num_blocks=8, cache_policy='arc')
    manager._policy.evict(0, set())
    manager.on_request_finished(ReqContext(req_id='test'))
    assert json.loads((manager._arc_eviction_profile.directory / 'summary.json').read_text())['calls'] == 1
