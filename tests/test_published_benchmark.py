"""Published-policy wiring: exact streams, causal metadata, and isolated pairs."""
import asyncio
from types import SimpleNamespace

import aiohttp
from aiohttp import web
import pytest

from benchmarks.serving_suite.published import request_fields, comparisons, command
from benchmarks.serving_suite.published_server import create_app
from benchmarks.serving_suite.runner import case_order
from benchmarks.serving_suite.workload import completion


def test_five_arm_order_and_terminal_metadata():
    plan = {'published_runtimes': True}
    assert case_order(plan, 0) == ['arc', 'wait', 'semantic', 'continuum', 'upstream']
    assert case_order(plan, 1) == ['wait', 'semantic', 'continuum', 'upstream', 'arc']
    assert request_fields('continuum', 's', 0, 4) == {'job_id': 's', 'is_last_step': False}
    assert request_fields('continuum', 's', 3, 4)['is_last_step']
    first = request_fields('upstream', 's', 0, 4)['kv_hints']['actions'][0]['payload']
    last = request_fields('upstream', 's', 3, 4)['kv_hints']['actions'][0]['payload']
    assert first['value'] == 10 and first['ttl_seconds'] == 60
    assert last['value'] is None and 'ttl_seconds' not in last
    assert last['revision'] > first['revision']


def test_pairing_does_not_mix_external_arms():
    cases = [dict(repeat=0, variant=v, evaluation={'status': 'inconclusive'})
             for v in case_order({'published_runtimes': True}, 0)]
    result = comparisons(cases)
    assert len(result) == 7
    assert all(c['eligible_pairs'] == 0 for c in result.values())
    with pytest.raises(ValueError, match='Duplicate'):
        comparisons(cases + [cases[0]])


@pytest.mark.parametrize('policy,retain', [('continuum', True), ('continuum', False), ('upstream', True), ('upstream', False)])
def test_real_http_bridge_forwards_engine_metadata_and_exact_tokens(policy,retain):
    class Engine:
        called = []
        stopped = False
        async def check_health(self): pass
        async def do_log_stats(self): pass
        async def abort(self, rid): raise AssertionError('Completed request aborted')
        def shutdown(self): self.stopped = True
        async def generate(self, prompt, params, rid, **kwargs):
            self.called.append((prompt, params, rid, kwargs))
            for ids, reason in [([31], None), ([31, 32], 'stop')]:
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, finish_reason=reason)],
                                      finished=reason is not None)

    async def scenario():
        engine = Engine()
        app = await create_app(engine, policy, SimpleNamespace, lambda value: ('typed', value), lambda: b'')
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        url = 'http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])
        payload = dict(request_id='one', prompt=[1, 2], max_tokens=2, priority=-7123,
                       **(request_fields(policy, 'session', 0, 4) if retain else {}))
        try:
            async with aiohttp.ClientSession() as client:
                row = await completion(client, url, payload, {}, lambda row: None)
                assert row['success'] and row['output_token_ids'] == [31, 32]
                assert row['finish_reason'] == 'stop'
                async with client.get(url+'/metrics') as response:
                    assert 'suite_completed_requests_total 1' in await response.text()
            call = engine.called[-1]
            if policy == 'continuum':
                assert call[1].extra_args == ({'job_id': 'session', 'is_last_step': False}
                                             if retain else {'job_id': 'one', 'is_last_step': True})
            else:
                if retain:
                    assert call[3]['kv_hints'] == ('typed', payload['kv_hints'])
                else:
                    assert 'kv_hints' not in call[3]
                assert call[3]['priority'] == -7123
        finally:
            await runner.cleanup()
        assert engine.stopped
    asyncio.run(scenario())


def test_external_command_requires_equal_cpu_capacity():
    plan = dict(published_runtime={'python': '/external/python', 'cpu_bytes': 100,
                    'kv_transfer_config': {'kv_connector': 'OffloadingConnector'}},
                published_variant='upstream', model='/model', max_model_len=8192,
                gpu_blocks=2560, concurrency=8, batch_tokens=8192,
                gpu_memory_utilization=.8, seed=1, cpu_bytes=200)
    with pytest.raises(ValueError, match='CPU budget'):
        command(plan, 8000)
    plan['published_runtime']['cpu_bytes'] = 200
    args, config = command(plan, 8000)
    assert '--enforce-eager' in args
    assert config['kv_connector_extra_config']['cpu_bytes_to_use'] == 200


def test_return_observer_preserves_policy_results(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    observer = Path(__file__).resolve().parents[1]/'benchmarks/serving_suite/published_observer'
    log = tmp_path/'events.jsonl'
    source = '''
from types import SimpleNamespace
def apply_g1_priority(action, request):
    return SimpleNamespace(status='applied', block_ids=(1, 2))
def take_blocks():
    return [3]
assert apply_g1_priority(SimpleNamespace(value=10), SimpleNamespace(request_id='sc|suite|test')).block_ids == (1, 2)
assert take_blocks() == [3]
'''
    script = f"exec(compile({source!r}, '/published/vllm/v1/core/kv_cache_priority.py', 'exec'))"
    subprocess.run([sys.executable, '-c', script], check=True,
                   env={**os.environ, 'PYTHONPATH': str(observer), 'SUITE_PUBLISHED_OBSERVATION': str(log)})
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(e['event'], e.get('affected_blocks', e.get('selected_blocks'))) for e in events] == [
        ('priority_application', 2), ('priority_selection', 1)]
    assert events[0]['priority'] == 10 and events[0]['request_id'] == 'sc|suite|test'


def test_four_way_has_no_baseline_hint_machinery_and_shared_local_config():
    import json
    from pathlib import Path
    from benchmarks.serving_suite.runner import connector, plan_for_variant
    config = json.loads((Path(__file__).resolve().parents[1]/
        'benchmarks/research_checks/fixtures/wait_retention_ungated.json').read_text())
    plan = dict(tier='agent', published_runtimes=True,
        published_variants=['arc', 'wait', 'continuum', 'upstream'],
        publish_hints=True, candidate_config=config, cpu_bytes=2818572288,
        compact_events=True)
    assert case_order(plan, 0) == ['arc', 'wait', 'continuum', 'upstream']
    a, c = plan_for_variant(plan, 'arc'), plan_for_variant(plan, 'wait')
    assert not a['publish_hints'] and c['publish_hints']
    left = connector(a, 'arc')['kv_connector_extra_config']
    right = connector(c, 'candidate')['kv_connector_extra_config']
    assert {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)} == {
        'prefetch_budget_fraction', 'wait_hint_cpu_retention_only'}


def test_effective_geometry_rejects_unequal_backing(tmp_path):
    import json
    from benchmarks.serving_suite.published import verify_geometry
    (tmp_path/'config.json').write_text(json.dumps(dict(hidden_size=3584,
        num_attention_heads=28, num_hidden_layers=28, num_key_value_heads=4)))
    observation = dict(event='native_worker_geometry', cpu_bytes=2818572288,
        gpu_bytes=2348810240, pinned=True, blocks_per_chunk=1, device_aliases=0)
    path = tmp_path/'runtime-observation.jsonl'
    path.write_text(json.dumps(observation)+'\n')
    plan = dict(model=str(tmp_path), gpu_blocks=2560, cpu_bytes=2818572288)
    assert verify_geometry(tmp_path, plan, 'wait')['cpu_chunk_tokens'] == 16
    observation['cpu_bytes'] += 1
    path.write_text(json.dumps(observation)+'\n')
    with pytest.raises(ValueError, match='differs from budget'):
        verify_geometry(tmp_path, plan, 'wait')


def test_geometry_observer_counts_worker_not_transfer_handler(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    observer = Path(__file__).resolve().parents[1]/'benchmarks/serving_suite/published_observer'
    log = tmp_path/'geometry.jsonl'
    source = """
class Tensor:
    def numel(self): return 100
    def element_size(self): return 1
    def is_pinned(self): return True
    def data_ptr(self): return 17
class Handler:
    def __init__(self, cpu_tensors, gpu_tensors, blocks_per_chunk): pass
class Worker:
    def __init__(self, kv_caches, blocks_per_chunk):
        cpu_tensors, gpu_tensors = [Tensor()], [Tensor()]
        Handler(cpu_tensors, gpu_tensors, blocks_per_chunk)
        Handler(cpu_tensors, gpu_tensors, blocks_per_chunk)
Worker(None, 1)
"""
    script = f"exec(compile({source!r}, '/published/vllm/v1/kv_offload/cpu/gpu_worker.py', 'exec'))"
    subprocess.run([sys.executable, '-c', script], check=True,
        env={**os.environ, 'PYTHONPATH': str(observer), 'SUITE_RUNTIME_OBSERVATION': str(log)})
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['cpu_bytes'] == 100 and rows[0]['pinned']


def test_decode_observer_requires_completed_decode_publication(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    observer = Path(__file__).resolve().parents[1]/'benchmarks/serving_suite/published_observer'
    log = tmp_path/'decode.jsonl'
    source = """
from types import SimpleNamespace as NS
def from_request_tracker(tracker, save_decode_cache):
    return NS(token_ids=list(range(2816)), save_spec=NS(can_save=True, skip_leading_tokens=2808))
tracker = NS(req_id='test', is_decode_phase=True, prompt_len=2808)
from_request_tracker(tracker, False)
from_request_tracker(tracker, True)
tracker.is_decode_phase = False
from_request_tracker(tracker, True)
"""
    script = f"exec(compile({source!r}, '/site/lmcache/integration/vllm/vllm_v1_adapter.py', 'exec'))"
    subprocess.run([sys.executable, '-c', script], check=True,
        env={**os.environ, 'PYTHONPATH': str(observer), 'SUITE_RUNTIME_OBSERVATION': str(log)})
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['decode_tokens_in_publication'] == 8
