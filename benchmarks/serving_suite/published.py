"""Adapters for pinned published systems, not reimplementations of their policies."""
from __future__ import annotations

import json
import hashlib
import os
import tarfile
import tempfile
import shutil
import subprocess
from pathlib import Path

PINS = {
    'continuum': ('https://github.com/Hanchenli/vllm-continuum',
                  '316a58794a6ff86b216e579b74fd56ed0c5a911f'),
    'upstream': ('https://github.com/karen-sy/vllm',
                 '7966f717dd4eb09461dd97102db1352c924bbf50'),
}
VARIANTS = ('arc', 'wait', 'semantic', 'continuum', 'upstream')

def execution_identity(runtime):
    source = Path(runtime.get('execution_source', runtime['source']))
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source/'vllm').rglob('*'))
            if p.is_file() and (p.suffix in ('.py', '.so'))}

def execution_environment(runtime):
    env = dict(os.environ)
    if runtime.get('execution_source'):
        env['PYTHONPATH'] = runtime['execution_source']
    return env



def check_source(name, runtime):
    source = Path(runtime['source'])
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != PINS[name][1] or git('status', '--porcelain'):
        raise ValueError(f'{name}: published checkout changed')
    if runtime.get('execution_identity') != execution_identity(runtime):
        raise ValueError(f'{name}: executed published/compatibility source changed')


def freeze_sources(runtimes, output):
    for name, runtime in runtimes.items():
        check_source(name, runtime)
        with (Path(output)/f'{name}-source.tar.gz').open('xb') as stream:
            subprocess.run(['git', '-C', runtime['source'], 'archive', '--format=tar.gz',
                            PINS[name][1]], stdout=stream, check=True)
        # Git archives omit submodule support and local wheel assets. Freeze all
        # actually imported Python and record native content hashes explicitly.
        source = Path(runtime.get('execution_source', runtime['source']))
        with tarfile.open(Path(output)/f'{name}-executed-python.tar.gz', 'w:gz', dereference=True) as archive:
            for relative in runtime['execution_identity']:
                if relative.endswith('.py'):
                    archive.add(source/relative, arcname=relative, recursive=False)
        if runtime.get('compatibility_patch'):
            shutil.copyfile(runtime['compatibility_patch'], Path(output)/f'{name}-compatibility.patch')
        check_source(name, runtime)


def activation(directory, variant):
    directory = Path(directory)
    if variant == 'continuum':
        path = directory/'scheduler_timestamps'
        histories = json.loads(path.read_text()) if path.exists() else {}
        events = [e for job, history in histories.items() if str(job).startswith('w') for e in history]
        pins = sum('pinned_time' in e for e in events)
        return {'verified': pins > 0, 'pins': pins,
                'source': 'published scheduler recorder; excludes warmup'}
    path = directory/'published-policy.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    applications = sum(e['event'] == 'priority_application' and e['status'] == 'applied'
                       and e['affected_blocks'] > 0 and e.get('priority', 0) is not None
                       and e.get('priority', 0) > 0
                       and str(e.get('request_id', '')).startswith('sc|suite|') for e in events)
    selections = sum(e['event'] == 'priority_selection' and e['selected_blocks'] > 0 for e in events)
    return {'verified': applications > 0 and selections > 0,
            'applied_priority_updates': applications, 'priority_selections': selections,
            'source': 'read-only Python return observer; not counterfactual victim attribution'}


def diagnose_external(directory, case, plan, write=True):
    """Retain common workflow checks; distinguish unavailable native telemetry."""
    from .agent_report import diagnose
    from .contracts import write_json
    # Do not compare incompatible cross-version counters against our receipts.
    result = diagnose(directory, {**case, 'metrics': {}}, plan, write=False)
    unavailable = {'missing_or_duplicate_native_request_receipts',
                   'no_native_pressure_samples', 'native_endpoint_accounting_unavailable'}
    result['measurement_errors'] = [e for e in result['measurement_errors'] if e not in unavailable]
    result['native_accounting'] = 'unavailable: published runtime lacks our native receipts'
    result['policy_activation'] = activation(directory, case['variant'])
    if plan.get('published_retention_off'):
        if plan['published_variant'] != 'upstream':
            raise ValueError('A clean retention-only control is implemented only for upstream')
        rows = [json.loads(line) for line in (Path(directory)/'requests.jsonl').read_text().splitlines()]
        receipt = result['policy_activation']
        omitted = len(rows) == case['expected_requests'] and all(not r['payload'].get('kv_hints') for r in rows)
        receipt.update(verified=omitted and receipt['applied_priority_updates'] == 0,
            control='Native retention metadata omitted; native CPU ARC, priority scheduler and bridge unchanged',
            all_workflow_retention_metadata_omitted=omitted,
            expected_positive_priority_updates=0)
    result['opportunity'] = {'cache_competition_observed': None,
                             'scope': 'native cache competition unavailable'}
    result['cache'] = {'scope': 'native source/transfer accounting unavailable'}
    result['prefetch'] = {'scope': 'native prefetch accounting unavailable'}
    def count(name):
        lines = (Path(directory)/name).read_text().splitlines()
        values = [float(line.split()[1]) for line in lines
                  if line.startswith('suite_completed_requests_total ')]
        return values[0] if len(values) == 1 else None
    before, after = count('metrics-before.prom'), count('metrics-after.prom')
    delta = after-before if before is not None and after is not None else None
    result['consistency'] = {'bridge_completed_requests': {
        'observed': delta, 'expected': case['expected_requests'],
        'agrees': delta == case['expected_requests']}}
    if delta != case['expected_requests']:
        result['measurement_errors'].append('bridge_request_count_mismatch')
    result['hints'] = {'publications': 0, 'transport': 'request metadata, not local wait-hint RPC'}
    result['telemetry']['observation_overhead'] = (
        'Common client and resource sampling; native instrumentation differs by runtime. '
        'Published arms use a thin AsyncLLM HTTP bridge. This is a system comparison.')
    result['limitations'].append(
        'One pass, different runtime versions and instrumentation; exploratory system timings only.'
        if plan.get('repeats', 1) == 1 else
        'Repeated configured-system timings; different runtime versions, backends, instrumentation and live outputs prevent isolated algorithmic attribution.')
    if write:
        write_json(Path(directory)/'agent.json', result)
    return result


def load_runtimes(path):
    """Require explicit isolated interpreters and immutable clean source trees."""
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 'published-agent-runtimes-v1':
        raise ValueError('Expected published-agent-runtimes-v1')
    if set(data.get('runtimes', {})) != set(PINS):
        raise ValueError('Supply both continuum and upstream runtimes')
    for name, runtime in data['runtimes'].items():
        source = Path(runtime['source']).resolve()
        python = Path(runtime['python']).absolute()
        if not source.is_dir() or not python.is_file():
            raise ValueError(f'{name}: source checkout and installed Python are required')
        def git(*args):
            return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
        if git('rev-parse', 'HEAD') != PINS[name][1]:
            raise ValueError(f'{name}: source revision differs from published pin')
        if git('status', '--porcelain'):
            raise ValueError(f'{name}: published source has tracked modifications')
        if runtime.get('execution_source'):
            if name != 'upstream' or not runtime.get('compatibility_patch'):
                raise ValueError('Only explicit upstream compatibility patches are supported')
            execution = Path(runtime['execution_source']).resolve()
            patch = Path(runtime['compatibility_patch']).resolve()
            if hashlib.sha256(patch.read_bytes()).hexdigest() != runtime.get('compatibility_patch_sha256'):
                raise ValueError('Compatibility patch hash differs')
            original = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (source/'vllm').rglob('*.py')}
            actual = {str(p.relative_to(execution)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (execution/'vllm').rglob('*.py')}
            changed = 'vllm/v1/kv_offload/cpu/gpu_worker.py'
            if set(original) != set(actual) or any(original[k] != actual[k] for k in original if k != changed):
                raise ValueError('Compatibility source differs outside the declared module')
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary)/changed
                target.parent.mkdir(parents=True)
                shutil.copyfile(source/changed, target)
                subprocess.run(['patch', '-s', '-p1', '-d', temporary, '-i', str(patch)], check=True)
                if hashlib.sha256(target.read_bytes()).hexdigest() != actual[changed]:
                    raise ValueError('Executed module does not equal published source plus patch')
            runtime.update(execution_source=str(execution), compatibility_patch=str(patch))
        runtime['execution_identity'] = execution_identity(runtime)
        # An editable installation must really import this checkout, not an
        # unrelated wheel. Run away from our cwd to avoid local import shadowing.
        probe = subprocess.check_output([str(python), '-c',
            'import json,vllm,torch; print(json.dumps(dict(source=vllm.__file__,torch=torch.__version__)))'],
            cwd=source.parent, text=True, env=execution_environment(runtime))
        identity = json.loads(probe.splitlines()[-1])
        if not Path(identity['source']).resolve().is_relative_to(Path(runtime.get('execution_source', source))):
            raise ValueError(f'{name}: interpreter does not load the pinned checkout')
        runtime.update(source=str(source), python=str(python), repository=PINS[name][0],
                       revision=PINS[name][1], identity=identity)
        runtime['packages'] = subprocess.check_output(
            [str(python), '-m', 'pip', 'freeze'], text=True, cwd=source.parent).splitlines()
    return data['runtimes']


def request_fields(name, session, turn, turns, *, ttl=60.0):
    if name == 'continuum':
        return {'job_id': session, 'is_last_step': turn == turns-1}
    if name != 'upstream':
        return {}
    payload = dict(target='request_cached', location='local_g1',
                   claim_id='suite/'+session, revision=turn+1,
                   value=None if turn == turns-1 else 10)
    if payload['value'] is not None:
        payload['ttl_seconds'] = ttl
    return {'kv_hints': dict(protocol_version='0.1', message_id=f'{session}/{turn}',
        actions=[dict(action_id='retain', action_type='vllm.set_priority',
                      action_version='1.0', payload=payload)])}


def comparisons(cases):
    """Never let several candidate arms overwrite each other in pairing."""
    from .report import paired_comparison
    seen = set()
    for case in cases:
        key = case['repeat'], case['variant']
        if key in seen:
            raise ValueError('Duplicate variant/repetition')
        seen.add(key)
    result = {}
    for baseline in ('arc', 'wait'):
        for candidate in VARIANTS:
            if candidate == baseline or (baseline == 'wait' and candidate == 'arc'):
                continue
            selected = [{**c, 'arm': 'arc' if c['variant'] == baseline else 'candidate'}
                        for c in cases if c['variant'] in (baseline, candidate)]
            result[f'{candidate}_vs_{baseline}'] = paired_comparison(selected)
    return result


def command(plan, port):
    runtime = plan['published_runtime']
    bridge = Path(__file__).with_name('published_server.py').resolve()
    args = [runtime['python'], str(bridge), '--policy', plan['published_variant'],
            '--port', str(port), '--model', plan['model'], '--dtype', 'half',
            '--max-model-len', str(plan['max_model_len']), '--block-size', '16',
            '--num-gpu-blocks-override', str(plan['gpu_blocks']),
            '--max-num-seqs', str(plan['concurrency']), '--max-num-batched-tokens',
            str(plan['batch_tokens']), '--gpu-memory-utilization',
            str(plan['gpu_memory_utilization']), '--seed', str(plan['seed']),
            '--enable-prefix-caching', '--enforce-eager']
    if plan['published_variant'] == 'continuum':
        args += ['--scheduling-policy', 'continuum']
    else:
        args += ['--attention-backend', 'TRITON_ATTN']
        if plan.get('continuation_credit_ms', 0):
            args += ['--scheduling-policy', 'priority']
    # Config is explicit because the published preview uses LMCache whereas
    # upstream uses its native offloader. Never silently run without CPU backing.
    config = runtime.get('kv_transfer_config')
    if not isinstance(config, dict):
        raise ValueError('Published runtime requires an explicit kv_transfer_config')
    if runtime.get('cpu_bytes') != plan['cpu_bytes']:
        raise ValueError('Published runtime CPU budget must match the comparison')
    if plan['published_variant'] == 'upstream':
        if config.get('kv_connector') != 'OffloadingConnector':
            raise ValueError('Upstream arm requires native OffloadingConnector')
        config = json.loads(json.dumps(config))
        config.setdefault('kv_connector_extra_config', {})['cpu_bytes_to_use'] = plan['cpu_bytes']
    else:
        if config.get('kv_connector') != 'LMCacheConnectorV1':
            raise ValueError('Continuum preview requires its published LMCache integration')
        expected = plan['cpu_bytes'] / 2**30
        if float(runtime.get('env', {}).get('LMCACHE_MAX_LOCAL_CPU_SIZE', -1)) != expected:
            raise ValueError('Continuum LMCache CPU capacity differs from the common budget')
    args += ['--kv-transfer-config', json.dumps(config)]
    return args, config


def verify_geometry(directory, plan, variant):
    """Reject a changed effective backing budget before any measured requests."""
    path = Path(directory)/'runtime-observation.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    model = json.loads((Path(plan['model'])/'config.json').read_text())
    head_dim = model.get('head_dim', model['hidden_size']//model['num_attention_heads'])
    kv_heads = model.get('num_key_value_heads', model['num_attention_heads'])
    bytes_per_block = model['num_hidden_layers']*kv_heads*head_dim*2*2*16
    expected_gpu_bytes = plan['gpu_blocks']*bytes_per_block
    if variant == 'continuum':
        allocations = [r for r in rows if r['event'] == 'lmcache_cpu_geometry']
        if len(allocations) != 1 or allocations[0]['bytes'] != plan['cpu_bytes']:
            raise ValueError('Continuum actual CPU allocation differs from budget')
        if not allocations[0]['pinned']:
            raise ValueError('Continuum LMCache CPU backing is not pinned')
        # The published startup reports the effective GPU token capacity.
        if f"GPU KV cache size: {plan['gpu_blocks']*16:,} tokens" not in (Path(directory)/'server.log').read_text():
            raise ValueError('Continuum effective GPU capacity not established')
        return dict(cpu_bytes=allocations[0]['bytes'], cpu_pinned=True,
                    cpu_chunk_tokens=256, gpu_blocks=plan['gpu_blocks'],
                    gpu_block_tokens=16, gpu_kv_bytes=expected_gpu_bytes,
                    gpu_capacity_source='published effective GPU KV token-capacity startup record')
    allocations = [r for r in rows if r['event'] == 'native_worker_geometry']
    if len(allocations) != 1:
        raise ValueError('Missing/duplicate native CPU/GPU allocation observation')
    actual = allocations[0]
    if actual['cpu_bytes'] != plan['cpu_bytes'] or actual['gpu_bytes'] != expected_gpu_bytes:
        raise ValueError('Native actual CPU/GPU allocation differs from budget')
    if not actual['pinned'] or actual['blocks_per_chunk'] != 1:
        raise ValueError('Native pinned backing/chunk geometry differs')
    return dict(cpu_bytes=actual['cpu_bytes'], cpu_pinned=actual['pinned'],
                cpu_chunk_tokens=16, gpu_blocks=plan['gpu_blocks'],
                gpu_block_tokens=16, gpu_kv_bytes=actual['gpu_bytes'],
                device_aliases=actual['device_aliases'],
                gpu_capacity_source='actual worker tensor bytes')
