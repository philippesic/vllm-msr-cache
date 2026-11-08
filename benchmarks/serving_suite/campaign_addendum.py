"""Fixed Stage 6 addendum over the existing campaign worker, ledger and locks.

Prepare is GPU-free. Run refuses an incomplete primary acquisition, preserves
whole comparison blocks and stops at persisted resource/attempt limits. It never
edits the primary campaign and never selects repetitions from observed timings.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from .campaign_contracts import (Ledger, atomic_json, cell_inputs, digest, file_hash,
    load, read_protocol, seal_tree, verify_tree, write_protocol)

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'qwen3_8b_awq'
LIMITS = dict(overall_s=21600, repository_s=7200, overhead_s=7200, advice_s=7200,
              repository_attempts=22, overhead_attempts=48, advice_attempts=15)


def schedule(seed):
    from .campaign import order
    blocks = [dict(id='repository-gate', section='repository', cases=[['R00', 'arc']], purpose_gate=True)]
    for repeat in range(4):
        blocks.append(dict(id=f'repository-{repeat}', section='repository', repeat=repeat,
            cases=[['R01', s] for s in order(['arc','wait','continuum','upstream'], repeat, seed)]))
        if repeat == 1:
            blocks.append(dict(id='repository-roomy', section='repository', cases=[['R02','arc']]))
    for repeat in range(24):
        arms = ['O0','O2'] if repeat % 2 == 0 else ['O2','O0']
        blocks.append(dict(id=f'overhead-{repeat}', section='overhead', repeat=repeat,
                           cases=[['Q01', s] for s in arms]))
    # Fixed first four rows of a six-order three-arm crossover. Position counts
    # differ by at most one; it is not a fully balanced six-block design.
    arms = [['H00','arc'], ['H01','wait'], ['H02','wait']]
    for repeat, indices in enumerate(((0,1,2),(1,2,0),(2,0,1),(2,1,0))):
        blocks.append(dict(id=f'advice-{repeat}', section='advice', repeat=repeat,
                           cases=[arms[i] for i in indices]))
    return blocks


def measured_overlap(root, cohorts):
    wanted = {s['id'] for c in cohorts for s in c['sessions']}
    hits, count = [], 0
    for path in Path(root).glob('*/attempts/*/*/measurement/corpus.json'):
        data = load(path)
        overlap = wanted & {s['id'] for s in data['sessions']}
        count += 1
        if overlap:
            hits.append(dict(path=str(path), source_ids=sorted(overlap)))
    return dict(measured_corpora_checked=count, overlap=hits)


def make_inputs(repo, qa, abandoned, repo_plan, qa_plan):
    """Task choice is source order, never measured activity or policy speed."""
    from .campaign_workload import linear_graph
    from .contracts import validate_corpus, validate_plan
    repo, qa = copy.deepcopy(repo), copy.deepcopy(qa)
    if len(repo['sessions']) != 16 or len(qa['sessions']) != 64 or len(abandoned['sessions']) < 4:
        raise ValueError('Expected reserved sixteen repository / sixty-four QA inputs')
    rp, qp = copy.deepcopy(repo_plan), copy.deepcopy(qa_plan)
    for p in (rp, qp):
        p.pop('operational_corpus', None)
        p.update(required_activity=[], case_timeout_s=900)
    rp.update(concurrency=16, tool_concurrency=4, workflow_concurrency=0)
    for s in repo['sessions']:
        s.update(publish_hints=True, traffic_class='ordinary')
        s['graph'] = linear_graph(s, rp)
    for s in qa['sessions']:
        s.update(publish_hints=False, traffic_class='unhinted')
        s['graph'] = linear_graph(s, qp)

    mixed = copy.deepcopy(repo)
    for index, s in enumerate(mixed['sessions']):
        s.update(advice_scenario='late' if index < 4 else 'ordinary',
                 traffic_class='late' if index < 4 else 'ordinary')
        if index < 4:
            s['wait_s'] = [45.0, 0, 0]
            s['graph'] = linear_graph(s, rp)
    for original in abandoned['sessions'][:4]:
        s = copy.deepcopy(original)
        s.update(id='abandoned-'+s['id'], turns=s['turns'][:1], tools=[],
                 publish_hints=True, traffic_class='abandoned', advice_scenario='abandoned')
        s['quality_reference'] = dict(s.get('quality_reference',{}), kind='intermediate',
            scope='Initial-only abandoned session; no final investigation or validation requested')
        s['graph'] = linear_graph(s, rp)
        mixed['sessions'].append(s)
    # Same offered traffic in every arm; eight interactive arrivals at t=0 and
    # eight at t=30 seconds, overlapping the deliberately late continuations.
    for index, original in enumerate(qa['sessions'][:16]):
        s = copy.deepcopy(original)
        s.update(id='mixed-'+s['id'], arrival_s=0.0 if index < 8 else 30.0,
                 traffic_class='unhinted')
        mixed['sessions'].append(s)
    mixed['provenance'] += (' Authored prediction-stress schedule: sixteen complete reserved workflows, '
        'first four returning at least 45s later; four development-source initial requests that never return; '
        'sixteen reserved QA requests, half arriving at 30s. Identical graphs/arrivals in all three arms.')

    def effective(plan, corpus):
        plan = copy.deepcopy(plan)
        plan.update(sessions=len(corpus['sessions']),
            expected_requests_per_wave=sum(n['kind']=='request' for s in corpus['sessions'] for n in s['graph']['nodes']),
            expected_tools_per_wave=sum(n['kind']=='tool' for s in corpus['sessions'] for n in s['graph']['nodes']))
        validate_plan(plan); validate_corpus(corpus)
        return plan, corpus

    roomy = copy.deepcopy(rp)
    roomy['cpu_bytes'] = 4831838208
    roomy['candidate_config']['kv_connector_extra_config']['cpu_bytes_to_use'] = roomy['cpu_bytes']
    for name, r in roomy.get('published_runtimes', {}).items():
        r['cpu_bytes'] = roomy['cpu_bytes']
        r['kv_transfer_config'].setdefault('kv_connector_extra_config', {})['cpu_bytes_to_use'] = roomy['cpu_bytes']
        if name == 'continuum':
            r['env']['LMCACHE_MAX_LOCAL_CPU_SIZE'] = str(roomy['cpu_bytes']/2**30)
    local = copy.deepcopy(rp)
    for name in ('published_variants','published_runtimes'):
        local.pop(name, None)
    return dict(R00=effective(rp, repo), R01=effective(rp, repo), R02=effective(roomy, repo), Q01=effective(qp, qa),
        H00=effective(local, mixed), H01=effective(dict(local, addendum_advice='accurate'), mixed),
        H02=effective(dict(local, addendum_advice='misleading'), mixed))


def prepare(parent, out):
    from .campaign import source_snapshot
    from .source_snapshot import materialize
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Prepare requires a new destination')
    primary = read_protocol(parent)
    host, resolved, campaign = (load(parent/name) for name in ('host.private.json','resolved.json','campaign.json'))
    cache = Path(host['asset_cache'])
    repo, qa = (load(cache/'corpora/repair'/(name+'.json')) for name in ('repository','qa'))
    overlap = measured_overlap(parent.parent, [repo, qa])
    if overlap['overlap']:
        raise ValueError('Reserved repair cohort already measured; declare a different design explicitly')
    cells = {c['cell']:c for c in resolved['cells'] if c['model']==MODEL}
    rp = load(cells['B06']['confirmation_binding']['plan'])
    qp = load(cells['B02']['confirmation_binding']['plan'])
    if rp['gpu_blocks'] != 1024 or rp['cpu_bytes'] != 2793406464:
        raise ValueError('Unexpected anchor resource envelope')
    inputs = make_inputs(repo, qa, load(cache/'corpora/development/repository.json'), rp, qp)
    new_cells = []
    for name, (plan, corpus) in inputs.items():
        cell = copy.deepcopy(cells['B02' if name=='Q01' else 'B06'])
        binding = dict(partition='repair-addendum', cohort_mode='fixed')
        for kind, value in (('plan',plan), ('corpus',corpus)):
            path = out/'inputs'/name/(kind+'.json')
            atomic_json(path, value)
            binding.update({kind:str(path), kind+'_digest':digest(value)})
        cell.update(cell=name, workload_binding=binding, confirmation_binding=None,
                    systems=['O0','O2'] if name=='Q01' else ['arc','wait','continuum','upstream'] if name=='R01' else ['arc'] if name in ('R00','R02','H00') else ['wait'])
        cell['profile']['repo_pressure_sessions'] = 16
        cell['profile'].update(cpu_bytes=plan['cpu_bytes'],gpu_blocks=plan['gpu_blocks'])
        new_cells.append(cell)
    source = source_snapshot()
    # All policy/native-serving implementation bytes remain the selected source.
    stable = {k:v for k,v in primary['source']['files'].items() if k.startswith(('semantic_offload/','vendor/vllm/vllm/'))}
    if any(source['files'].get(k)!=v for k,v in stable.items()):
        raise ValueError('Selected serving policy/runtime differs from primary source')
    snapshot = materialize(ROOT, cache/'source-snapshots', source)
    campaign.update(parent_campaign_id=resolved['campaign_id'],
        diagnostic_only=dict(kind='bounded-stage6-addendum', protocol='docs/STAGE6_ADDENDUM.md'),
        input_identities=[cell_inputs(c) for c in new_cells])
    atomic_json(out/'campaign.json', campaign)
    atomic_json(out/'resolved.json', dict(resolved, campaign_id=digest(campaign), cells=new_cells,
        configurations_per_repeat=None))
    atomic_json(out/'host.private.json', host); (out/'host.private.json').chmod(0o600)
    atomic_json(out/'hardware.private.json', load(parent/'hardware.private.json'))
    (out/'hardware.private.json').chmod(0o600)
    protocol = dict(schema='stage6-bounded-addendum-v1', parent=str(parent),
        campaign_id=digest(campaign),
        parent_protocol_sha256=file_hash(parent/'protocol.lock.json'),
        primary_source=primary['source']['digest'], source=source, snapshot=str(snapshot),
        source_policy_unchanged=True, held_out_check=overlap,
        schedule=schedule(campaign['specification']['seed']), limits=LIMITS,
        cells_digest=digest(new_cells), design_sha256=file_hash(ROOT/'docs/STAGE6_ADDENDUM.md'))
    write_protocol(out, protocol)
    Ledger(out)
    return dict(out=str(out), scheduled_configurations=78, maximum_attempts=85,
        protocol_sha256=file_hash(out/'protocol.lock.json'), source=source['digest'],
        held_out_check=overlap, limits=LIMITS, state='prepared_not_launched')


def primary_complete(parent):
    """Cheap whole-block check; full retained integrity audit remains separate."""
    from .campaign_validity import measurement_validity
    parent = Path(parent)
    lock = read_protocol(parent)
    groups = {}
    for e in Ledger(parent).index().values():
        i = e['details'].get('identity', {})
        if i.get('phase')!='confirm' or e['state'] not in ('failed','verified'):
            continue
        if i['source'] != lock['source']['digest']:
            raise ValueError('Primary source mismatch')
        if measurement_validity(parent/e['details']['path'], e)['valid']:
            groups.setdefault((i['model'],i['cell'],i['repeat'],e['details']['block_id']), set()).add(i['system'])
    return all(any(key[:3]==(b['model'],b['cell'],b['repeat']) and arms==set(b['systems'])
        for key, arms in groups.items()) for b in lock['schedule'])


def pressure(case):
    n = case.get('campaign', {}).get('native_cache') or {}
    return bool(n.get('complete') and n.get('returning_loaded_tokens',0)>0 and
                n.get('returning_requests_with_prior_cpu_eviction',0)>0)


def purpose_failure(block, measured):
    section = block['section']
    arc = next((c for _,s,c in measured if s=='arc'),None)
    if section in ('repository','advice') and arc is not None and block['id']!='repository-roomy' and not pressure(arc):
        return 'ARC lacks real returning CPU restores or eviction-affected returns'
    if section=='advice':
        bad = next(c for n,s,c in measured if n=='H02')
        touched = sum(h.get('response',{}).get('ready_cpu_prefix_keys',0) for h in bad['hints']
            if h.get('prediction_experiment',{}).get('false_positive'))
        if not touched:
            return 'No actual CPU recency touch for the valid no-return false positives'
    return None


def completed_attempt(out, event):
    if event['state']!='verified' or not event['details'].get('seal'):
        return None
    directory = Path(out)/event['details']['path']
    verify_tree(directory, event['details']['seal'])
    return load(directory/'measurement/result.json')['cases'][0]


def block_state(out, block, events):
    groups = {}
    for e in events.values():
        if e['details'].get('addendum_block') == block['id']:
            groups.setdefault(e['details']['block_id'], []).append(e)
    for bid, group in reversed(list(groups.items())):
        if len(group)==len(block['cases']) and all(completed_attempt(out,e) for e in group):
            actual = sorted((e['details']['identity']['cell'], e['details']['identity']['system']) for e in group)
            if actual == sorted(tuple(c) for c in block['cases']):
                return group, groups
    return None, groups


def run(out, *, resume=False):
    from harness.gpu_lock import GPULock, LEGACY_LOCK, gpu_reserved
    from .campaign import identity_for, source_snapshot
    from .campaign_execution import run_worker
    from .campaign_inputs import attest, unchanged
    from .resources import available_gpu
    out = Path(out).resolve()
    protocol = read_protocol(out)
    if not primary_complete(protocol['parent']):
        return dict(status='waiting_for_primary', gpu_launched=False)
    if source_snapshot()!=protocol['source']:
        raise ValueError('Use the attested addendum snapshot as the working directory')
    from benchmarks.research_checks.stage6_healthcheck import coordinators
    if coordinators(Path(protocol['parent'])):
        raise ValueError('Primary coordinator has not released ownership')
    cells = load(out/'resolved.json')['cells']
    if digest(cells)!=protocol['cells_digest']:
        raise ValueError('Addendum workload bindings changed')

    @gpu_reserved
    def acquire():
        # Probe the legacy lock without unlinking it or passing a descriptor.
        with GPULock(LEGACY_LOCK):
            available_gpu()
        ledger = Ledger(out)
        with ledger.writer():
            state_path = out/'execution.json'
            state = load(state_path) if state_path.exists() else dict(started_unix=time.time(), section_started={})
            if state.get('status') in ('complete','budget_exhausted'):
                return state
            if resume:
                ledger.recover(seal_pending=True)
            elif any(e['state'] in ('running','planned') for e in ledger.index().values()):
                raise ValueError('Interrupted addendum: inspect owned resources and use --resume')
            import psutil
            owner = psutil.Process()
            state.update(status='running', owner=dict(pid=owner.pid, create_time=owner.create_time()))
            atomic_json(state_path,state)
            stats = attest(out,cells,load(out/'host.private.json'),'screen')
            for block in protocol['schedule']:
                done, prior = block_state(out,block,ledger.index())
                if done:
                    measured = [(e['details']['identity']['cell'],e['details']['identity']['system'],
                                 completed_attempt(out,e)) for e in done]
                    failure = purpose_failure(block,measured)
                    if failure:
                        state.setdefault('purpose_failures',{})[block['section']] = dict(block=block['id'],reason=failure)
                        atomic_json(state_path,state)
                    continue
                section = block['section']
                if section in state.get('purpose_failures',{}) or section in state.get('closed_sections',{}):
                    continue
                if prior:
                    repair_path = out/'repairs'/(block['id']+'.json')
                    repair = load(repair_path) if repair_path.exists() else {}
                    if repair.get('prior_blocks')!=list(prior) or len(prior)>1 or section=='overhead':
                        raise RuntimeError('Diagnosed whole-block repair authorization required: '+block['id'])
                state['section_started'].setdefault(section,time.time())
                state.update(active_block=block['id'], active_attempt=None)
                atomic_json(state_path,state)
                used = sum(e['details'].get('section')==section for e in ledger.index().values())
                if used+len(block['cases']) > LIMITS[section+'_attempts']:
                    state.setdefault('closed_sections',{})[section] = 'section_attempt_cap'
                    atomic_json(state_path,state); continue
                bid = digest(dict(block=block, source=protocol['source']['digest'], previous=list(prior)))
                measured = []
                for cell_name, system in block['cases']:
                    now = time.time()
                    remaining = min(state['started_unix']+LIMITS['overall_s'],
                        state['section_started'][section]+LIMITS[section+'_s'])-now
                    if now >= state['started_unix']+LIMITS['overall_s']:
                        state.update(status='budget_exhausted',reason='persisted_wall_time_cap',section=section)
                        atomic_json(state_path,state); return state
                    if remaining <= 0:
                        state.setdefault('closed_sections',{})[section] = 'section_wall_time_cap'
                        atomic_json(state_path,state); break
                    unchanged(stats); available_gpu()
                    cell = next(c for c in cells if c['cell']==cell_name)
                    identity = identity_for(out,'screen',cell,block.get('repeat',0),system,protocol['source'])
                    identity['addendum_block'] = block['id']
                    aid, directory = ledger.begin(identity,block_id=bid,addendum_block=block['id'],section=section)
                    row = ledger.index()[aid]
                    state.update(active_attempt=str(directory), progress_unix=time.time())
                    atomic_json(state_path,state)
                    print(f"{block['id']} {cell_name} {system}",flush=True)
                    command = [sys.executable,'-m','benchmarks.serving_suite.campaign','_worker',
                               '--out',str(out),'--attempt',str(directory)]
                    plan = load(cell['workload_binding']['plan'])
                    try:
                        with (directory/'worker.log').open('xb') as log:
                            run_worker(command,cwd=protocol['snapshot'],log=log,
                                timeout=min(remaining,plan['startup_timeout_s']+plan['case_timeout_s']+180))
                        unchanged(stats)
                        from .verify import verify
                        verify(directory/'measurement')
                        result = load(directory/'measurement/result.json')
                        case = result['cases'][0]
                        passed = case['evaluation']['status']=='passed' and result['status']=='passed'
                        seal = seal_tree(directory)
                        ledger.event(row['case_id'],aid,'verified' if passed else 'failed',
                            **row['details'],seal=seal,measurement_status=result['status'])
                        if not passed:
                            raise RuntimeError('Incomplete/invalid measurement; preserve and diagnose '+str(directory))
                        measured.append((cell_name,system,case))
                    except BaseException as exc:
                        current = ledger.index()[aid]
                        if current['state']=='running':
                            atomic_json(directory/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
                            seal = seal_tree(directory)
                            ledger.event(row['case_id'],aid,'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                **row['details'],seal=seal,failure_class='runner_or_execution',error=str(exc))
                        state.update(status='attention',error=str(exc),progress_unix=time.time())
                        atomic_json(state_path,state)
                        raise
                # A section deadline may leave a partial block, always excluded.
                failure = purpose_failure(block,measured) if len(measured)==len(block['cases']) else None
                if failure:
                    state.setdefault('purpose_failures',{})[section] = dict(block=block['id'],reason=failure)
                state.update(progress_unix=time.time(),active_attempt=None)
                atomic_json(state_path,state)
            state.update(status='complete',completed_unix=time.time(),active_block=None,active_attempt=None,
                         interpretation='Acquisition ended; quality, comparisons and purpose qualifications remain required')
            atomic_json(state_path,state)
            return state
    return acquire()


def repair(out, block_id, kind, evidence, reason):
    out, evidence = Path(out).resolve(), Path(evidence).resolve()
    protocol = read_protocol(out)
    block = next(b for b in protocol['schedule'] if b['id']==block_id)
    if block['section']=='overhead' or block.get('purpose_gate') or len(block['cases']) not in (3,4):
        raise ValueError('Only one whole three/four-arm execution repair per section is allowed')
    if not evidence.is_file() or not evidence.is_relative_to(out) or not reason.strip():
        raise ValueError('Link preserved in-addendum diagnostic evidence and a concrete reason')
    existing = list((out/'repairs').glob('*.json'))
    if any(load(p)['section']==block['section'] for p in existing):
        raise ValueError('Section repair allowance already spent')
    done, prior = block_state(out,block,Ledger(out).index())
    if done or len(prior)!=1:
        raise ValueError('A single incomplete whole block is required; performance is not a repair trigger')
    record = dict(block=block_id,section=block['section'],prior_blocks=list(prior),kind=kind,
                  evidence=str(evidence),evidence_sha256=file_hash(evidence),reason=reason,authorized_unix=time.time())
    atomic_json(out/'repairs'/(block_id+'.json'),record)
    return record


def close_section(out, section, evidence, reason):
    """Retain an unresolved failed section and proceed without new repetitions."""
    out, evidence = Path(out).resolve(), Path(evidence).resolve()
    state = load(out/'execution.json')
    if state['status']!='attention' or not evidence.is_file() or not evidence.is_relative_to(out) or not reason.strip():
        raise ValueError('Stopped attention state and retained diagnostic evidence required')
    state.setdefault('closed_sections',{})[section] = dict(reason=reason,evidence=str(evidence),
        evidence_sha256=file_hash(evidence),closed_unix=time.time())
    atomic_json(out/'execution.json',state)
    return state


def inspect(out):
    """One bounded read: ownership, telemetry, timeout and useful raw progress."""
    import psutil
    from benchmarks.research_checks.stage6_healthcheck import gpu_snapshot, lock_owners
    from harness.gpu_lock import GPU_LOCK
    out = Path(out).resolve()
    protocol = read_protocol(out)
    if not (out/'execution.json').exists():
        return dict(status='prepared_not_launched',out=str(out))
    state = load(out/'execution.json'); issues=[]
    if state['status']!='running':
        return dict(state,out=str(out))
    owner=state['owner']
    try:
        p=psutil.Process(owner['pid'])
        if p.create_time()!=owner['create_time'] or p.uids().real!=os.getuid():
            raise psutil.NoSuchProcess(owner['pid'])
        owned={q.pid for q in [p]+p.children(recursive=True)}
        if p.pid not in lock_owners(GPU_LOCK):issues.append('missing_supervisor_reservation')
    except psutil.NoSuchProcess:
        owned=set(); issues.append('coordinator_stopped')
    try:
        gpu=gpu_snapshot()
        if set(gpu['processes'])-owned:issues.append('foreign_gpu_client')
    except Exception as exc:
        gpu=dict(error=str(exc));issues.append('missing_gpu_telemetry')
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    if memory.available<128*1024**2 and swap.free<128*1024**2:
        issues.append('severe_host_memory_exhaustion')
    active=Path(state['active_attempt']) if state.get('active_attempt') else None
    progress=state.get('progress_unix',state['started_unix'])
    if active:
        files=[*active.glob('measurement/*/requests.jsonl'),*active.glob('measurement/*/streams/*.sse')]
        progress=max([progress]+[p.stat().st_mtime for p in files])
        # Full startup allowance is intentional; acquisition receipts remain the
        # source of useful activity and worker supervision enforces tighter caps.
        if time.time()-progress>1020:issues.append('no_request_stream_or_attempt_progress')
    if time.time()>state['started_unix']+protocol['limits']['overall_s']+60:
        issues.append('overall_deadline_exceeded')
    return dict(state,status='attention' if issues else 'healthy',issues=issues,gpu=gpu,out=str(out),
                host=dict(available_memory_bytes=memory.available,swap_used_bytes=swap.used),
                attempts=len(Ledger(out).index()),progress_unix=progress)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('status');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('close-section');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--section',choices=('repository','overhead','advice'),required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--reason',required=True)
    p=sub.add_parser('repair');p.add_argument('--out',type=Path,required=True);p.add_argument('--block',required=True)
    p.add_argument('--kind',choices=('startup','transport','owned_cleanup','tool_execution'),required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--reason',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.parent,args.out)
    elif args.command=='run':result=run(args.out,resume=args.resume)
    elif args.command=='repair':result=repair(args.out,args.block,args.kind,args.evidence,args.reason)
    elif args.command=='close-section':result=close_section(args.out,args.section,args.evidence,args.reason)
    else:
        result=inspect(args.out)
        result['checked_unix']=time.time()
        atomic_json(args.out/'health/latest.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
