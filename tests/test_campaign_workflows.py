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
