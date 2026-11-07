"""Maintained resumable campaign CLI layered on serving_suite.run_case.

Preparation is the only network phase. Compilation is GPU-free and preserves
unresolved requirements explicitly. No required cell is dropped by --only.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .campaign_contracts import (SCHEMA, PHASES, Ledger, atomic_json, cell_inputs, digest, fields,
                                 file_hash, load, local_path, seal_tree,
                                 read_protocol,write_protocol,validate_campaign, validate_host, verify_tree)

ROOT = Path(__file__).resolve().parents[2]


def catalogs(path):
    campaign = validate_campaign(load(path))
    base = Path(path).resolve().parent
    result = {}
    for name in ('models', 'workloads', 'systems'):
        record = load(base / campaign[name])
        fields(record, ('schema', name))
        if record['schema'] != f'serving-campaign-{name[:-1] if name != "systems" else "system"}-catalog-v1':
            raise ValueError(f'Unsupported {name} catalog')
        result[name] = record[name]
    for model, cells in campaign['matrix'].items():
        if model not in result['models']:
            raise ValueError('Model missing from catalog')
        for cell, systems in cells.items():
            if cell not in result['workloads'] or set(systems) - result['systems'].keys():
                raise ValueError('Matrix references unknown workload/system')
    policy = load(base / campaign['policy'])
    result['policy'] = policy
    return campaign, result


def compile_campaign(campaign_path, host_path, out):
    campaign, records = catalogs(campaign_path)
    host = validate_host(load(host_path))
    out = Path(out)
    if (out / 'campaign.json').exists():
        raise ValueError('Campaign already compiled; use status/resume or a new directory')
    out.mkdir(parents=True, exist_ok=True)
    assets_path = Path(host['asset_cache']) / (campaign['id'] + '-assets.json')
    assets = load(assets_path) if assets_path.exists() else {'models': {}}
    unresolved, cells, environments = [], [], {}
    system_identities = {}
    for system,runtime in host.get('resolved_published_runtimes',{}).items():
        from .assets import runtime_identity
        python = runtime['python']
        if python not in environments:
            environments[python] = runtime_identity(python)
        system_identities[system] = dict(environment_digest=digest(environments[python]),
                                         execution_digest=digest(runtime['execution_identity']))
    for model, workloads in campaign['matrix'].items():
        artifact = assets.get('models', {}).get(model, {})
        binding = host['models'].get(model) or artifact.get('path')
        profile = host.get('model_profiles', {}).get(model)
        python = (profile or {}).get('runtime_python',host['runtime_python'])
        if Path(python).is_file() and python not in environments:
            from .assets import runtime_identity
            environments[python] = runtime_identity(python)
        if artifact.get('status') != 'prepared' or not binding:
            unresolved.append(f'{model}: pinned model preparation required')
        if not profile:
            unresolved.append(f'{model}: measured resource profile required')
        for cell, systems in workloads.items():
            workload_binding = host.get('workload_bindings', {}).get(model + ':' + cell)
            if not workload_binding:
                unresolved.append(f'{model}:{cell}: frozen workload binding required')
            cells.append(dict(model=model, cell=cell, systems=systems,
                              artifact_identity=artifact.get('identity', {}).get('content_digest'),
                              profile=profile, workload_binding=workload_binding,
                              confirmation_binding=host.get('confirmation_bindings', {}).get(model + ':' + cell),
                              runtime_identity=digest(environments[python]) if python in environments else None,
                              system_identities={s:system_identities[s] for s in systems if s in system_identities}))
    if any('continuum' in c['systems'] for c in cells) and not host.get('published_runtimes'):
        unresolved.append('Pinned published runtime bindings required')
    portable = dict(schema=SCHEMA, specification=campaign, catalogs=records,
                    asset_identities={name: {k: v for k, v in record.items() if k != 'path'}
                                      for name, record in assets.get('models', {}).items()})
    portable['input_identities'] = [cell_inputs(c) for c in cells]
    atomic_json(out / 'campaign.json', portable)
    atomic_json(out / 'host.private.json', host)
    (out / 'host.private.json').chmod(0o600)
    resolved = dict(schema=SCHEMA, campaign_id=digest(portable), host_id=host['id'],
                    cells=cells, configurations_per_repeat=sum(len(c['systems']) for c in cells),
                    unresolved=unresolved, runnable=not unresolved,
                    cost=dict(status='requires measured pilots', confirmation_repeats=[6, 8, 10, 12],
                              overhead_and_operational='additional to matrix'))
    atomic_json(out / 'resolved.json', resolved)
    atomic_json(out / 'runtime-identities.private.json',environments)
    Ledger(out)
    return resolved


def status(out):
    out = Path(out)
    result = load(out / 'resolved.json')
    index = Ledger(out).index()
    counts = {}
    for row in index.values():
        counts[row['state']] = counts.get(row['state'], 0) + 1
    return dict(campaign_id=result['campaign_id'], runnable=result['runnable'],
                configurations_per_repeat=result['configurations_per_repeat'],
                unresolved=result['unresolved'], attempts=counts,
                protocol_frozen=(out / 'protocol.lock.json').is_file())


def source_snapshot():
    # Existing freezer captures Python imports; catalogs and templates are also
    # executed inputs and must be part of the campaign identity.
    from .runner import source_identity
    identity = source_identity()
    # The shared SSE parser has no dependency on historical coordinators. Tests
    # and unrelated research scripts are validation/provenance, not serving inputs.
    identity['files'] = {k:v for k,v in identity['files'].items() if not k.startswith('tests/') and
        (not k.startswith('benchmarks/') or k.startswith('benchmarks/serving_suite/') or k in ('benchmarks/csv_utils.py','benchmarks/cell_lifecycle.py'))}
    for name in ('qualification_inputs','wait_hint_runtime_probe'):
        relative = f'benchmarks/research_checks/{name}.py'
        identity['files'][relative] = file_hash(ROOT/relative)
    for directory in ('benchmarks/serving_suite/catalogs', 'benchmarks/serving_suite/campaigns',
                      'benchmarks/serving_suite/templates', 'benchmarks/serving_suite/runtime_requirements', 'patches'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                identity['files'][str(path.relative_to(ROOT))] = file_hash(path)
    for relative in ('requirements-runtime.txt','pyproject.toml','Makefile','vendor/vllm/LICENSE',
                     'evidence/upstream-pin-fix-20261104/mapped-host-pointer.patch',
                     'evidence/stage5-optimization-20261105/selected-candidate.json'):
        identity['files'][relative] = file_hash(ROOT/relative)
    identity['digest'] = digest(identity['files'])
    return identity


def order(systems, repeat, seed):
    """Williams balanced positions and predecessors for four systems."""
    systems = list(systems)
    random.Random(seed).shuffle(systems)
    if len(systems) == 4:
        rows = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
        return [systems[i] for i in rows[repeat % 4]]
    if len(systems) == 2:
        return systems if repeat % 2 == 0 else systems[::-1]
    raise ValueError('Only paired or four-arm blocks are supported')


def freeze(out):
    out = Path(out)
    if (out/'imported-protocol.json').exists():
        raise ValueError('Imported repetitions/order are frozen; use activate-import after local preflight')
    resolved = load(out / 'resolved.json')
    campaign = load(out / 'campaign.json')
    if resolved['unresolved']:
        raise ValueError('Cannot freeze unresolved bindings/capabilities')
    if (out / 'protocol.lock.json').exists():
        raise ValueError('Protocol already frozen; create a new campaign to revise it')
    from .campaign_gates import qualify
    capability = qualify(out)
    if not capability.get('all_required_correctness_passed'):
        raise ValueError('All required model/system correctness gates must pass')
    from .campaign_report import pilot_precision
    precision = pilot_precision(out)
    if precision.get('status') != 'complete':
        raise ValueError('Complete measured pilot precision/cost calculation required')
    selections = precision['repeats']
    schedule = []
    seed = campaign['specification']['seed']
    for cell in resolved['cells']:
        key = cell['model'] + ':' + cell['cell']
        binding = cell.get('confirmation_binding')
        if not binding or binding.get('partition') != 'confirmation':
            raise ValueError('Fresh confirmation bindings required before freeze')
        for kind in ('plan', 'corpus'):
            if digest(load(binding[kind])) != binding[kind + '_digest']:
                raise ValueError('Confirmation input changed before freeze')
        repeats = selections[key]
        if repeats not in (6, 8, 10, 12) or (len(cell['systems']) == 4 and repeats not in (8, 12)):
            raise ValueError('Invalid/balance-breaking repetition count')
        for repeat in range(repeats):
            schedule.append(dict(model=cell['model'], cell=cell['cell'], repeat=repeat,
                                 systems=order(cell['systems'], repeat, seed)))
    record = dict(schema='serving-campaign-protocol-v1', campaign_id=resolved['campaign_id'],
                  source=source_snapshot(), precision=precision, schedule=schedule,
                  capability_digest=digest(capability), cells_digest=digest(resolved['cells']))
    record['hardware_digest'] = digest(load(out/'hardware.private.json'))
    from .campaign_schedules import supplementary
    record['supplementary_schedules'] = supplementary(out,resolved)
    write_protocol(out,record)
    return dict(blocks=len(schedule), configurations=sum(len(b['systems']) for b in schedule))


def identity_for(out, phase, cell, repeat, system, source):
    resolved = load(Path(out) / 'resolved.json')
    hardware_path = Path(out)/'hardware.private.json'
    return dict(campaign_id=resolved['campaign_id'], hardware=resolved['host_id'],
                hardware_digest=digest(load(hardware_path)) if hardware_path.exists() else None,phase=phase,
                model=cell['model'], cell=cell['cell'], system=system, repeat=repeat,
                artifact_identity=cell['artifact_identity'], profile=cell['profile'],
                runtime_identity=cell.get('runtime_identity'),
                system_identity=cell.get('system_identities',{}).get(system),
                workload=digest(binding_for(cell, phase)), source=source['digest'])


def binding_for(cell, phase):
    if phase in ('confirm', 'overhead', 'operational'):
        binding = cell.get('confirmation_binding')
        if not binding:
            raise ValueError('Missing confirmation workload binding')
        return binding
    return cell['workload_binding']


def run_phase(out, phase, *, only=None, resume=False, dry_run=False):
    from harness.gpu_lock import gpu_reserved
    out = Path(out)
    resolved = load(out / 'resolved.json')
    if resolved['unresolved']:
        raise ValueError('Unresolved prerequisites: ' + '; '.join(resolved['unresolved']))
    source = source_snapshot()
    locked = read_protocol(out) if (out / 'protocol.lock.json').exists() else None
    if phase in ('confirm', 'overhead', 'operational'):
        if not locked or locked['source']['digest'] != source['digest']:
            raise ValueError('Missing locked protocol or executed source changed')
        if locked['cells_digest'] != digest(resolved['cells']):
            raise ValueError('Locked resource/workload bindings changed')
    if phase=='pilot':
        from .campaign_gates import qualify
        if not qualify(out)['all_required_correctness_passed']:
            raise ValueError('Complete native correctness, external activation and all development screens before pilots')
    if phase == 'confirm':
        schedule = locked['schedule']
    elif phase == 'preflight':
        schedule = [dict(model=c['model'],cell='B06',repeat=0,
                         systems=['wait','registry'] if c['model']=='qwen3_8b_awq' else ['wait'])
                    for c in resolved['cells'] if c['cell']=='B06']
    elif phase=='screen' and load(out/'campaign.json').get('diagnostic_only',{}).get('kind') in (
            'roomy-allocation-development','cache-envelope-development'):
        kind = load(out/'campaign.json')['diagnostic_only']['kind']
        expected = ['B07'] if kind=='roomy-allocation-development' else ['B06','B07']
        if locked or [c['cell'] for c in resolved['cells']]!=expected or any(c['systems']!=['arc'] for c in resolved['cells']):
            raise ValueError('Capacity diagnostic requires an unfrozen ARC capacity cell or matched pair')
        schedule = [dict(model=c['model'],cell=c['cell'],repeat=0,systems=['arc']) for c in resolved['cells']]
    else:
        count = 3 if phase == 'pilot' else 1
        spec = load(out / 'campaign.json')['specification']
        pilot_cells = [c for c in resolved['cells'] if
            (c['model'] == 'qwen3_8b_awq' and c['cell'] in ('B02','B06','B09','B11','B12')) or
            (c['model'] != 'qwen3_8b_awq' and c['cell'] == 'B06')]
        selected_cells = pilot_cells if phase == 'pilot' else resolved['cells']
        schedule = [dict(model=c['model'], cell=c['cell'], repeat=r,
                         systems=order(c['systems'], r, spec['seed']))
                    for c in selected_cells for r in range(count)]
    # Overhead and operational have distinct workload/behavior requirements and
    # must never silently execute the throughput matrix under another phase name.
    if phase in ('overhead', 'operational'):
        path = out / (phase + '-schedule.json')
        if not path.exists():
            raise ValueError(f'{phase}: explicit validated control/operational schedule required')
        schedule = load(path)['blocks']
        if schedule != locked['supplementary_schedules'][phase]:
            raise ValueError('Supplementary schedule changed after freeze')
    if only:
        schedule = [b for b in schedule if b['model'] + ':' + b['cell'] == only]
        if not schedule:
            raise ValueError('--only does not match a required cell')
    if dry_run:
        return {'phase': phase, 'blocks': schedule, 'partial_selection': bool(only)}

    @gpu_reserved
    def acquire():
        from .resources import available_gpu
        ledger = Ledger(out)
        with ledger.writer():
            if resume:
                ledger.recover()
            elif any(row['state'] in ('running', 'planned') for row in ledger.index().values()):
                raise ValueError('Interrupted campaign; use resume after checking owned resources')
            import os
            import psutil
            hardware = dict(gpu=available_gpu(),platform=platform.platform(),machine=platform.machine(),
                            logical_cpus=os.cpu_count(),ram_bytes=psutil.virtual_memory().total)
            hardware_path = out/'hardware.private.json'
            if hardware_path.exists() and load(hardware_path)!=hardware:
                raise ValueError('Hardware/driver profile changed; use a new hardware-bound campaign')
            if not hardware_path.exists():
                atomic_json(hardware_path,hardware)
                hardware_path.chmod(0o600)
            if locked and locked.get('hardware_digest')!=digest(hardware):
                raise ValueError('Actual hardware differs from the frozen acquisition profile')
            from .campaign_inputs import attest, unchanged
            local_host = load(out/'host.private.json')
            selected_cells = [c for c in resolved['cells'] if any(b['model']==c['model'] and b['cell']==c['cell'] for b in schedule)]
            input_stats = attest(out,selected_cells,local_host,phase)
            results = []
            for block in schedule:
                unchanged(input_stats)
                cell = next(c for c in resolved['cells'] if c['model'] == block['model'] and c['cell'] == block['cell'])
                block_cells = [cell]
                if phase=='preflight':
                    block_cells += [c for c in resolved['cells'] if c['model']==cell['model'] and c['cell']=='B08']
                block_stats = attest(out,block_cells,local_host,phase,full_models=False)
                if source_snapshot()['digest'] != source['digest']:
                    raise ValueError('Executed source changed; campaign stopped before next block')
                # A partially completed comparison block is rerun as a whole.
                identities = [identity_for(out, phase, cell, block['repeat'], system, source) for system in block['systems']]
                reused = [ledger.reusable(identity) for identity in identities]
                if all(reused) and len({r["details"].get("block_id") for r in reused}) == 1:
                    results.extend(reused)
                    continue
                block_id = digest(dict(block=block, phase=phase, source=source['digest'], previous=[r['attempt_id'] if r else None for r in reused]))
                for system, identity in zip(block['systems'], identities):
                    available_gpu()
                    attempt, directory = ledger.begin(identity, block_id=block_id)
                    atomic_json(directory/'verified-input-stats.json',block_stats)
                    row = ledger.index()[attempt]
                    from .source_snapshot import materialize
                    local_host = load(out / 'host.private.json')
                    snapshot = materialize(ROOT, Path(local_host['asset_cache']) / 'source-snapshots', source)
                    command = [cell.get('profile', {}).get('runtime_python', local_host['runtime_python']), '-m',
                               'benchmarks.serving_suite.campaign', '_worker', '--out', str(out.resolve()),
                               '--attempt', str(directory.resolve())]
                    print(f'{phase} {cell["model"]}:{cell["cell"]} repetition {block["repeat"] + 1} {system}', flush=True)
                    with (directory / 'worker.log').open('xb') as log:
                        try:
                            from .campaign_execution import run_worker
                            bound = load(binding_for(cell, phase)['plan'])
                            run_worker(command, cwd=snapshot, log=log,
                                       timeout=bound['startup_timeout_s'] + bound['case_timeout_s'] + 180)
                            unchanged(input_stats)
                            result = load(directory / 'result.json')
                            # Verify artifacts immediately, independent of interpretation.
                            from .verify import verify
                            verify(directory / 'measurement')
                            seal = seal_tree(directory)
                            state = 'verified' if result['status'] == 'passed' else 'failed'
                            event = ledger.event(row['case_id'], attempt, state, **row['details'], seal=seal,
                                                 measurement_status=result['status'])
                            results.append(event)
                        except BaseException as exc:
                            atomic_json(directory / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
                            seal = seal_tree(directory)
                            ledger.event(row['case_id'], attempt, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                         **row['details'], seal=seal, failure_class='runner_or_execution', error=str(exc))
                            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                                raise
            return {'attempts': len(results), 'phase': phase, 'partial_selection': bool(only)}
    return acquire()


def operational_workload(plan,corpus):
    full = plan.get('operational_corpus',corpus)
    if dict(full,sessions=full['sessions'][:len(corpus['sessions'])])!=corpus:
        raise ValueError('Operational cohort must preserve the complete pressure-task prefix')
    plan.update(sessions=len(full['sessions']),
        expected_requests_per_wave=sum(n['kind']=='request' for s in full['sessions'] for n in s['graph']['nodes']),
        expected_tools_per_wave=sum(n['kind']=='tool' for s in full['sessions'] for n in s['graph']['nodes']))
    return full


def worker(out, attempt):
    """One fresh child owns only the legacy leaf lock under its supervisor."""
    def terminate(signum, frame):
        raise KeyboardInterrupt('Campaign worker terminated; closing owned resources')
    signal.signal(signal.SIGTERM, terminate)
    from harness.gpu_lock import GPULock, LEGACY_LOCK
    from .runner import run_case, source_identity, freeze_source
    from .verify import seal, verify
    from .contracts import SCHEMA as SUITE_SCHEMA
    identity = load(Path(attempt) / 'identity.json')
    resolved = load(Path(out) / 'resolved.json')
    host = load(Path(out) / 'host.private.json')
    cell = next(c for c in resolved['cells'] if c['model'] == identity['model'] and c['cell'] == identity['cell'])
    binding = binding_for(cell, identity['phase'])
    if not isinstance(binding, dict) or not {'plan', 'corpus'} <= binding.keys():
        raise ValueError('Workload binding requires explicit plan and corpus')
    plan = load(binding['plan'])
    corpus = load(binding['corpus'])
    if digest(corpus) != binding.get('corpus_digest') or digest(plan) != binding.get('plan_digest'):
        raise ValueError('Frozen workload/plan binding identity changed')
    plan['campaign_phase'] = identity['phase']
    from .campaign_verify import transform
    original_project = host['runtime_bindings'].get('project_root')
    if original_project:
        plan = transform(plan, {original_project: str(ROOT)})
        corpus = transform(corpus, {original_project: str(ROOT)})
    from .runner import plan_for_variant
    system = identity['system']
    control = system if system in ('O0','O1','O2') else None
    base_system = ('arc' if control in ('O0','O1') else 'wait') if control else system
    if system=='registry':
        base_system = 'wait'
    requested_credit = plan.get('continuation_credit_ms', 0)
    plan = plan_for_variant(plan, base_system)
    plan['continuation_credit_ms'] = requested_credit
    if base_system == 'arc':
        plan['publish_hints'] = False
        # Match the candidate connector settings while using its actual off
        # switch. Native unloaded ARC is measured separately as O0.
        plan['qualification_conditions'] = ['arc', 'wait']
    if control:
        plan.update(publish_hints=False,load_plugin=control!='O0',native_arc_control=control=='O0',
                    qualification_trace=False,required_activity=[],control=control)
        if identity['model'] != 'qwen3_8b_awq':
            corpus['sessions'] = corpus['sessions'][:8]
            plan.update(sessions=8,expected_requests_per_wave=8)
    if identity['phase'] == 'operational':
        corpus = operational_workload(plan,corpus)
        # Keep one server alive across six anchor waves. Secondary sequences are
        # smaller, with the same isolated cancel/drain/recovery checks.
        plan.update(tier='operational',waves=6 if identity['model']=='qwen3_8b_awq' else 2,
                    wave_interval_s=0,operational_advice_probes=base_system=='wait')
    if identity['phase']=='preflight' and system=='registry':
        plan.update(registry_development_probe=True,qualification_trace=False,required_activity=[],sessions=4,
                    expected_requests_per_wave=16,expected_tools_per_wave=12)
        corpus['sessions'] = corpus['sessions'][:4]
        plan['candidate_config']['kv_connector_extra_config'].update(preservation_audit=False,
            observed_session_registry_size=2,observed_prefix_registry_size=2)
        for name in ('published_runtimes','published_variants'):
            plan.pop(name,None)
    if identity['phase'] == 'preflight' and base_system == 'wait' and system!='registry':
        document_binding = next(c for c in resolved['cells'] if c['model']==identity['model'] and c['cell']=='B08')['workload_binding']
        documents = load(document_binding['corpus'])['sessions'][:9]
        paths = [next(path for path in session['input_files'] if path.endswith('.rst')) for session in documents]
        kv = cell['profile']['kv_bytes_per_token']
        cpu = (512*2**20//(256*kv))*(256*kv)
        plan.update(campaign_adapter=False, sessions=1,turns=11,concurrency=1,client_concurrency=1,
            gpu_blocks=128,cpu_bytes=cpu,max_model_len=2048,batch_tokens=1024,max_tokens=24,
            runtime_probe=dict(documents=paths,prompt_tokens=1024,output_tokens=24,require_restore=True),
            required_activity=['store','restore','cpu_recency'])
        for name in ('published_runtimes','published_variants'):
            plan.pop(name,None)
        corpus = dict(schema='semantic-serving-corpus-v1',synthetic=False,
            provenance='Deterministic native restore/fresh-reference fixture over pinned development PEP documents; not performance evidence.',
            sessions=[dict(id='native-functional',context='',turns=['functional fixture']*11)])
    directory = Path(attempt) / 'measurement'
    directory.mkdir()
    with GPULock(LEGACY_LOCK):
        initial = source_snapshot()
        if initial['digest'] != identity['source']:
            raise ValueError('Worker source differs from planned identity')
        atomic_json(directory / 'plan.json', plan)
        atomic_json(directory / 'corpus.json', corpus)
        atomic_json(directory / 'source.json', initial)
        freeze_source(initial, directory)
        inputs = {path:expected for session in corpus['sessions'] for path,expected in session.get('input_files',{}).items()}
        for path in plan.get('runtime_probe',{}).get('documents',[]):
            inputs[path] = file_hash(path)
        if inputs:
            import tarfile
            with tarfile.open(directory/'external-inputs.tar.gz','w:gz') as archive:
                for index,(path,expected) in enumerate(sorted(inputs.items())):
                    if file_hash(path)!=expected:
                        raise ValueError('Executed tool/fixture input changed before freeze')
                    archive.add(path,arcname=f'inputs/{index:04d}/{Path(path).name}',recursive=False)
            atomic_json(directory/'external-input-identities.json',inputs)
        from .assets import runtime_identity
        environment = runtime_identity(sys.executable)
        if identity.get('runtime_identity') and digest(environment) != identity['runtime_identity']:
            raise ValueError('Installed environment/native libraries changed since compilation')
        atomic_json(directory/'environment.json',environment)
        if plan.get('published_runtime'):
            from .published import check_source
            check_source(base_system,plan['published_runtime'])
            external_environment = runtime_identity(plan['published_runtime']['python'])
            expected_system = identity.get('system_identity')
            if expected_system and (digest(external_environment)!=expected_system['environment_digest'] or
                digest(plan['published_runtime']['execution_identity'])!=expected_system['execution_digest']):
                raise ValueError('Published environment/native execution identity changed')
            atomic_json(directory/'external-environment.json',external_environment)
            from .published import freeze_sources
            freeze_sources({base_system:plan['published_runtime']},directory)
        case = run_case(plan, corpus['sessions'], 'arc' if base_system == 'arc' else 'candidate',
                        identity['repeat'], directory, variant=system)
        result = dict(schema=SUITE_SCHEMA, plan=plan, cases=[case], status=case['evaluation']['status'])
        if any(file_hash(path)!=expected for path,expected in inputs.items()):
            result['status'] = 'failed'
            result['input_error'] = 'Executed tool/fixture input changed during attempt'
        if digest(runtime_identity(sys.executable)) != digest(environment):
            result['status'] = 'failed'
            result['environment_error'] = 'Installed environment changed during attempt'
        if plan.get('published_runtime'):
            check_source(base_system,plan['published_runtime'])
            if runtime_identity(plan['published_runtime']['python'])!=external_environment:
                result['status'] = 'failed'
                result['external_environment_error'] = 'Published environment changed during attempt'
        if source_snapshot()['digest'] != initial['digest']:
            result['status'] = 'failed'
            result['source_error'] = 'Executed source changed during attempt'
        atomic_json(directory / 'result.json', result)
        seal(directory)
        verify(directory)
        atomic_json(Path(attempt) / 'result.json', dict(status=result['status'], case=case['name']))
    return {'status': result['status']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'compile'):
        p = sub.add_parser(name)
        p.add_argument('--campaign', type=Path, required=True)
        p.add_argument('--host', type=Path, required=True)
        if name == 'compile':
            p.add_argument('--out', type=Path, required=True)
        else:
            p.add_argument('--metadata-only', action='store_true')
    for name in ('status', 'freeze', 'activate-import', 'qualify', 'pilot-plan', 'quality', 'verify', 'report', 'export', 'run', 'resume', '_worker'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, required=True)
        if name in ('run', 'resume'):
            p.add_argument('--phase', choices=PHASES, required=True)
            p.add_argument('--only')
            p.add_argument('--dry-run', action='store_true')
        if name == 'export':
            p.add_argument('--destination', type=Path, required=True)
        if name == '_worker':
            p.add_argument('--attempt', type=Path, required=True)
        if name=='quality':
            p.add_argument('--ratings',type=Path)
    p = sub.add_parser('import')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--host', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            from .assets import prepare
            result = prepare(args.campaign, args.host, metadata_only=args.metadata_only)
            result = {k: {'status': v['status'], 'revision': v['revision']} for k, v in result['models'].items()}
        elif args.command == 'compile':
            result = compile_campaign(args.campaign, args.host, args.out)
        elif args.command == 'status':
            result = status(args.out)
        elif args.command == 'freeze':
            result = freeze(args.out)
        elif args.command == 'activate-import':
            from .campaign_verify import activate_import
            result = activate_import(args.out)
        elif args.command == 'qualify':
            from .campaign_gates import qualify
            result = qualify(args.out)
        elif args.command == 'pilot-plan':
            from .campaign_report import pilot_precision
            result = pilot_precision(args.out)
        elif args.command == 'quality':
            from .campaign_quality import prepare_review,finish_review
            result = finish_review(args.out,args.ratings) if args.ratings else prepare_review(args.out)
        elif args.command in ('run', 'resume'):
            result = run_phase(args.out, args.phase, only=args.only, resume=args.command == 'resume', dry_run=args.dry_run)
        elif args.command == '_worker':
            result = worker(args.out, args.attempt)
        elif args.command in ('verify', 'export', 'import'):
            from . import campaign_verify
            if args.command == 'verify':
                result = campaign_verify.verify_campaign(args.out)
            elif args.command == 'export':
                result = campaign_verify.export_campaign(args.out, args.destination)
            else:
                result = campaign_verify.import_campaign(args.bundle, args.host, args.out)
        else:
            from .campaign_report import report
            result = report(args.out)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
