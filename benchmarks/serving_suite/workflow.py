"""Typed dependency graphs for fixed, real-tool workflows (never model shell).

Adapters supply request execution and tool execution. Graph scheduling, result
propagation and cancellation are independent of the serving backend.
"""
from __future__ import annotations

import asyncio
import math
import time

from .campaign_contracts import fields, identifier


def validate_graph(graph):
    fields(graph, ('schema', 'id', 'nodes', 'outputs'))
    if graph['schema'] != 'serving-workflow-v1':
        raise ValueError('Unknown workflow schema')
    identifier(graph['id'])
    if not isinstance(graph['nodes'], list) or not graph['nodes']:
        raise ValueError('Workflow nodes required')
    nodes = {}
    for node in graph['nodes']:
        fields(node, ('id', 'kind', 'depends_on'), ('prompt', 'tool', 'session', 'role', 'quality_reference', 'wait_s'))
        name = identifier(node['id'])
        if name in nodes or node['kind'] not in ('request', 'tool'):
            raise ValueError('Duplicate node or unsupported kind')
        dependencies = node['depends_on']
        if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
            raise ValueError('Dependencies must be a unique list')
        if node['kind'] == 'request':
            if not isinstance(node.get('prompt'), str) or not node['prompt'].strip() or 'tool' in node:
                raise ValueError('Request node requires prompt and no tool')
        else:
            if 'prompt' in node:
                raise ValueError('Tool node cannot contain model-written prompt command')
            validate_tool(node.get('tool'))
        delay = node.get('wait_s', 0)
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise ValueError('Declared waits must be finite nonnegative')
        nodes[name] = node
    if not isinstance(graph['outputs'], list) or not graph['outputs'] or len(graph['outputs']) != len(set(graph['outputs'])):
        raise ValueError('Workflow outputs must be unique declared node IDs')
    if set(graph['outputs']) - nodes.keys():
        raise ValueError('Missing output node')
    visited, visiting = set(), set()
    def visit(name):
        if name not in nodes:
            raise ValueError('Unknown dependency')
        if name in visiting:
            raise ValueError('Cyclic dependency graph')
        if name in visited:
            return
        visiting.add(name)
        for dependency in nodes[name]['depends_on']:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
    for name in nodes:
        visit(name)
    # Every node must contribute to a completed application, not orphaned load.
    contributing = set()
    def ancestors(name):
        contributing.add(name)
        for dependency in nodes[name]['depends_on']:
            if dependency not in contributing:
                ancestors(dependency)
    for name in graph['outputs']:
        ancestors(name)
    if contributing != set(nodes):
        raise ValueError('Orphaned workflow nodes')
    return graph


def validate_tool(tool):
    fields(tool, ('adapter', 'argv', 'cwd', 'valid_exit_codes', 'timeout_s', 'output_limit_bytes', 'checker'))
    if tool['adapter'] not in ('rg', 'python_module', 'pytest'):
        raise ValueError('Tool adapter is not allowlisted')
    argv = tool['argv']
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x or '\x00' in x for x in argv):
        raise ValueError('Typed argv required')
    if tool['adapter'] == 'rg' and argv[0] != 'rg':
        raise ValueError('rg adapter requires rg argv')
    if tool['adapter'] in ('python_module', 'pytest'):
        expected = 'pytest' if tool['adapter'] == 'pytest' else 'benchmarks.research_checks.qualification_inputs'
        if len(argv) < 3 or argv[1:3] != ['-m', expected]:
            raise ValueError('Only the declared read-only Python module is permitted')
        if any(x in argv for x in ('-c', '--override-ini', '-o', '--pyargs')):
            raise ValueError('Arbitrary Python/plugin settings are not allowed')
    if tool['checker'] not in ('exit_code', 'document_reference'):
        raise ValueError('Unknown tool result checker')
    for key in ('timeout_s', 'output_limit_bytes'):
        value = tool[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Tool timeout/output bound must be positive finite')
    if not isinstance(tool['valid_exit_codes'], list) or not tool['valid_exit_codes'] or any(type(x) is not int for x in tool['valid_exit_codes']):
        raise ValueError('Explicit valid exit codes required')
    if not isinstance(tool['cwd'], str) or not tool['cwd']:
        raise ValueError('Tool working directory required')
    return tool


async def execute_graph(graph, request, tool, *, request_concurrency, tool_concurrency, emit):
    """Await actual dependencies; a failure cancels and joins every owned task."""
    validate_graph(graph)
    if type(request_concurrency) is not int or request_concurrency < 1 or type(tool_concurrency) is not int or tool_concurrency < 1:
        raise ValueError('Positive request/tool worker limits required')
    limits = {'request': asyncio.Semaphore(request_concurrency), 'tool': asyncio.Semaphore(tool_concurrency)}
    tasks = {}
    async def run(node):
        dependencies = {name: await tasks[name] for name in node['depends_on']}
        if any(not result.get('success') for result in dependencies.values()):
            raise RuntimeError('Failed dependency cannot feed a downstream request')
        ready_ns = time.monotonic_ns()
        emit('node_ready', node_id=node['id'], ready_ns=ready_ns)
        async with limits[node['kind']]:
            started_ns = time.monotonic_ns()
            emit('node_start', node_id=node['id'], started_ns=started_ns,
                 queue_s=(started_ns-ready_ns)/1e9)
            result = await (request if node['kind'] == 'request' else tool)(node, dependencies)
        emit('node_end', node_id=node['id'], ended_ns=time.monotonic_ns(), success=result.get('success', False))
        if not result.get('success'):
            raise RuntimeError(f'Workflow node failed: {node["id"]}')
        return result
    # Scheduling starts only when all task handles exist, allowing arbitrary
    # declaration order without accidentally serializing fan-out children.
    for node in graph['nodes']:
        tasks[node['id']] = asyncio.create_task(run(node))
    try:
        await asyncio.gather(*tasks.values())
        return {name: tasks[name].result() for name in graph['outputs']}
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
