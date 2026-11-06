import asyncio
import pytest
from benchmarks.serving_suite.workflow import execute_graph, validate_graph, validate_tool


def graph():
    return dict(schema='serving-workflow-v1', id='team', outputs=['synthesis'], nodes=[
        dict(id='synthesis', kind='request', prompt='Synthesize actual results', depends_on=['a','b','c']),
        *[dict(id=n, kind='request', prompt=n, depends_on=['coordinator']) for n in ('a','b','c')],
        dict(id='coordinator', kind='request', prompt='Dispatch', depends_on=[])])


def test_real_fanout_and_actual_dependency_propagation():
    async def scenario():
        active, maximum = 0, 0
        seen = {}
        async def request(node, deps):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            seen[node['id']] = deps
            active -= 1
            return {'success': True, 'text': 'generated-' + node['id']}
        async def tool(*args):
            raise AssertionError('No tools in this graph')
        result = await execute_graph(graph(), request, tool, request_concurrency=3, tool_concurrency=1, emit=lambda *a, **k: None)
        assert maximum == 3
        assert seen['synthesis']['a']['text'] == 'generated-a'
        assert result['synthesis']['text'] == 'generated-synthesis'
    asyncio.run(scenario())


def test_dependency_failure_cancels_owned_siblings():
    async def scenario():
        ended = asyncio.Event()
        async def request(node, deps):
            if node['id'] == 'a':
                await asyncio.sleep(0)
                return {'success': False}
            if node['id'] in ('b','c'):
                try:
                    await asyncio.Event().wait()
                finally:
                    ended.set()
            return {'success': True}
        with pytest.raises(RuntimeError, match='failed'):
            await execute_graph(graph(), request, request, request_concurrency=3, tool_concurrency=1, emit=lambda *a, **k: None)
        assert ended.is_set()
    asyncio.run(scenario())


def test_cycles_or_unknown_dependencies_rejected():
    value = graph()
    value['nodes'][-1]['depends_on'] = ['synthesis']
    with pytest.raises(ValueError, match='Cyclic'):
        validate_graph(value)
    value['nodes'][-1]['depends_on'] = ['missing']
    with pytest.raises(ValueError, match='Unknown'):
        validate_graph(value)


def test_model_shell_cannot_be_a_tool():
    spec = dict(adapter='python_module', argv=['python', '-c', 'print(1)'], cwd='.', valid_exit_codes=[0], timeout_s=1, output_limit_bytes=1000, checker='exit_code')
    with pytest.raises(ValueError, match='read-only'):
        validate_tool(spec)


@pytest.mark.parametrize('publish',[True,False])
def test_live_graph_adapter_streams_and_runs_real_allowlisted_tools(tmp_path,publish):
    import json
    import sys
    from aiohttp import web
    from benchmarks.serving_suite.campaign_workload import linear_graph, run_wave, diagnose
    class Tokenizer:
        all_special_ids = [66]
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs.get('enable_thinking') is False
            if kwargs.get('tokenize') is False:
                return messages[-2]['content']+'new turn'
            return list(range(64))
        def encode(self,text,**kwargs):
            return [66,67]
        def decode(self, tokens, **kwargs):
            return 'source-supported answer'
    async def scenario():
        sent, hints, rows = [], [], []
        async def generate(request):
            payload = await request.json()
            sent.append(payload)
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            data = dict(choices=[dict(index=0, text='answer', token_ids=[65, 66], finish_reason='stop')],
                        usage=dict(prompt_tokens=64, completion_tokens=2, total_tokens=66))
            await response.write(('data: ' + json.dumps(data) + '\n\n').encode())
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response
        async def hint(request):
            raw = await request.json()
            return web.json_response(dict(schema='semantic-known-prefix-wait-hint-ack-v1', event_id=raw['event_id'],
                status='no_action', reason='cpu_prefix_recency_refreshed', ready_cpu_prefix_keys=4,
                reserved_gpu_blocks=0, speculative_transfer_bytes=0))
        app = web.Application()
        app.router.add_post('/v1/completions', generate)
        app.router.add_post('/plugins/semantic-offload/wait-hints', hint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        evidence = tmp_path / 'evidence.txt'
        evidence.write_text('supported actual evidence\n')
        session = dict(id='actual', context='source', turns=['Plan', 'Report evidence'], repository=str(tmp_path),
                       publish_hints=publish, tools=[dict(kind='search', argv=['rg', 'supported', str(evidence)], valid_exit_codes=[0])])
        plan = dict(concurrency=2, client_concurrency=2, tool_concurrency=1, tool_timeout_s=5,
                    request_timeout_s=5, publish_hints=True, max_tokens=8, max_model_len=128,
                    seed=17, chat_template_kwargs={'enable_thinking': False}, expected_tools_per_wave=1, waves=1)
        session['graph'] = linear_graph(session, plan)
        try:
            await run_wave(f'http://127.0.0.1:{port}', plan, [session], Tokenizer(), 0, rows.append,
                           'candidate', {'model_revision': 'test'}, hints, tmp_path/'streams')
            assert len(rows) == 2 and all(r['success'] for r in rows)
            assert len(hints) == int(publish)
            if publish:
                assert sent[1]['kv_transfer_params']['observed_prefix_boundary_tokens'] == 64
            else:
                assert all(set(p['kv_transfer_params'])=={'external_request_id','benchmark_client_sequence'} for p in sent)
            assert next((tmp_path/'tools').iterdir()).read_text() == 'supported actual evidence\n'
            (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result = diagnose(tmp_path, dict(source_ids=['actual'], load_duration_s=1, hints=hints), plan)
            assert result['completed_units'] == 1 and result['tools'] == 1
            assert result['measurement_errors'] == []
        finally:
            await runner.cleanup()
    asyncio.run(scenario())


def test_continuation_preserves_noncanonical_observed_ids_and_eos():
    from benchmarks.serving_suite.campaign_workload import continuation_tokens
    class Tokenizer:
        all_special_ids = [99]
        def apply_chat_template(self,messages,**kwargs):
            return 'rewritten old template '+messages[-2]['content']+'<EOS>new user'
        def encode(self,text,**kwargs):
            assert text == '<EOS>new user'
            return [99,20,21]
    previous = dict(payload=dict(prompt=[1,2,3]),output_token_ids=[8,9,99])
    history=[dict(role='user',content='old'),dict(role='assistant',content='decoded differently'),dict(role='user',content='new user')]
    assert continuation_tokens(Tokenizer(),history,previous,{}) == [1,2,3,8,9,99,20,21]


def test_tool_declarations_preserve_prompts_across_host_path_rebinding():
    from benchmarks.serving_suite.campaign_workload import tool_declaration
    for prefix in ('/home/first/cache','/mnt/second/assets'):
        actual = dict(adapter='python_module',argv=[prefix+'/env/bin/python','-m',
            'benchmarks.research_checks.qualification_inputs','document','--file',
            prefix+'/sources/peps/peps/pep-0241.rst','--reference',prefix+'/corpora/document-references.json'])
        declaration = tool_declaration(actual)
        assert declaration == 'python -m benchmarks.research_checks.qualification_inputs document --file peps/pep-0241.rst --reference document-references.json'
        assert actual['argv'][0]==prefix+'/env/bin/python'


def test_chat_and_qa_do_not_receive_investigation_status_instruction():
    from benchmarks.serving_suite.campaign_workload import system_instruction
    ordinary = system_instruction(dict(turns=['Answer the user']))
    investigation = system_instruction(dict(tools=[dict(kind='test')]))
    assert 'test_status=PASS' not in ordinary
    assert 'do not add a test status' in ordinary
    assert 'After actual validation' in investigation
    assert 'without claiming a test outcome' in investigation
