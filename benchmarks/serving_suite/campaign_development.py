"""Prepare a clearly separate persistent-operation development probe.

Use this before protocol freeze to exercise cancellation, advice perturbations,
drain and recovery through the same worker as the later operational sequence.
It cannot satisfy a parent's confirmation or supplementary completion gate.
"""
from pathlib import Path
import copy

from .campaign_contracts import Ledger,atomic_json,cell_inputs,digest,load


def prepare_roomy(parent,out,model,cpu_bytes,*,startup_timeout_s=180):
    """Probe a smaller roomy allocation using ARC and identical B07 tasks only."""
    parent,out = Path(parent).resolve(),Path(out).resolve()
    if out.exists() or (parent/'protocol.lock.json').exists():
        raise ValueError('Use a new development destination before freeze')
    if type(cpu_bytes) is not int or cpu_bytes<=0 or type(startup_timeout_s) is not int or startup_timeout_s<=0:
        raise ValueError('Probe bytes and startup bound must be positive integers')
    resolved = load(parent/'resolved.json')
    if resolved['unresolved']:
        raise ValueError('Resolve parent inputs before a capacity probe')
    cell = copy.deepcopy(next(c for c in resolved['cells'] if c['model']==model and c['cell']=='B07'))
    binding = cell['workload_binding']
    plan,corpus = load(binding['plan']),load(binding['corpus'])
    if digest(plan)!=binding['plan_digest'] or digest(corpus)!=binding['corpus_digest']:
        raise ValueError('Development inputs changed')
    if cpu_bytes>=plan['cpu_bytes']:
        raise ValueError('This probe only reduces the failed roomy allocation')
    plan.update(cpu_bytes=cpu_bytes,startup_timeout_s=startup_timeout_s)
    plan['candidate_config']['kv_connector_extra_config']['cpu_bytes_to_use'] = cpu_bytes
    cell['profile']['roomy_cpu_bytes'] = cpu_bytes
    out.mkdir(parents=True)
    for key,value in (('plan',plan),('corpus',corpus)):
        path = out/'inputs'/(key+'.json')
        atomic_json(path,value)
        binding[key],binding[key+'_digest'] = str(path),digest(value)
    cell.update(systems=['arc'],confirmation_binding=None)
    campaign = load(parent/'campaign.json')
    campaign.update(parent_campaign_id=resolved['campaign_id'],
        diagnostic_only=dict(kind='roomy-allocation-development',model=model,cpu_bytes=cpu_bytes,
            scope='ARC capacity/activity probe on exact parent B07 tasks; no policy speed selection or confirmation credit'),
        input_identities=[cell_inputs(cell)])
    atomic_json(out/'campaign.json',campaign)
    atomic_json(out/'resolved.json',dict(resolved,campaign_id=digest(campaign),cells=[cell],
        configurations_per_repeat=1,unresolved=[],runnable=True))
    atomic_json(out/'host.private.json',load(parent/'host.private.json'))
    (out/'host.private.json').chmod(0o600)
    Ledger(out)
    return dict(out=str(out),diagnostic_only=True,model=model,cpu_bytes=cpu_bytes,
        startup_timeout_s=startup_timeout_s,next='Run screen; compare ARC eviction activity and memory fit before compiling a replacement campaign')


def prepare_envelope(parent,out,model,*,sessions,roomy_cpu_bytes,startup_timeout_s=180):
    """Reduce the matched B06/B07 working set without selecting policy speed."""
    parent,out = Path(parent).resolve(),Path(out).resolve()
    if out.exists() or (parent/'protocol.lock.json').exists():
        raise ValueError('Use a new development destination before freeze')
    if type(sessions) is not int or sessions<=0 or type(roomy_cpu_bytes) is not int or roomy_cpu_bytes<=0:
        raise ValueError('Session count and capacity must be positive integers')
    resolved = load(parent/'resolved.json')
    if resolved['unresolved']:
        raise ValueError('Resolve parent inputs before an envelope probe')
    cells = []
    cohorts = []
    for name in ('B06','B07'):
        cell = copy.deepcopy(next(c for c in resolved['cells'] if c['model']==model and c['cell']==name))
        binding = cell['workload_binding']
        plan,corpus = load(binding['plan']),load(binding['corpus'])
        if digest(plan)!=binding['plan_digest'] or digest(corpus)!=binding['corpus_digest']:
            raise ValueError('Development inputs changed')
        if sessions>=len(corpus['sessions']):
            raise ValueError('Envelope fallback must reduce complete sessions')
        corpus['sessions'] = corpus['sessions'][:sessions]
        cohorts.append(digest(corpus))
        if name=='B07':
            if roomy_cpu_bytes>=plan['cpu_bytes']:
                raise ValueError('Envelope fallback must reduce the failed roomy capacity')
            plan['cpu_bytes'] = roomy_cpu_bytes
            plan['candidate_config']['kv_connector_extra_config']['cpu_bytes_to_use'] = roomy_cpu_bytes
        plan.update(sessions=sessions,startup_timeout_s=startup_timeout_s,
            expected_requests_per_wave=sum(n['kind']=='request' for s in corpus['sessions'] for n in s['graph']['nodes']),
            expected_tools_per_wave=sum(n['kind']=='tool' for s in corpus['sessions'] for n in s['graph']['nodes']))
        cell['profile'].update(repo_pressure_sessions=sessions,roomy_cpu_bytes=roomy_cpu_bytes)
        for key,value in (('plan',plan),('corpus',corpus)):
            path = out/'inputs'/name/(key+'.json')
            atomic_json(path,value)
            binding[key],binding[key+'_digest'] = str(path),digest(value)
        cell.update(systems=['arc'],confirmation_binding=None)
        cells.append(cell)
    if len(set(cohorts))!=1:
        raise ValueError('Capacity pair must retain identical complete tasks')
    if cells[0]['profile']['cpu_bytes']>=roomy_cpu_bytes:
        raise ValueError('Roomy capacity must exceed the base capacity')
    campaign = load(parent/'campaign.json')
    campaign.update(parent_campaign_id=resolved['campaign_id'],
        diagnostic_only=dict(kind='cache-envelope-development',model=model,sessions=sessions,
            scope='ARC pressure and contrast only; no policy speed selection or confirmation credit'),
        input_identities=[cell_inputs(c) for c in cells])
    atomic_json(out/'campaign.json',campaign)
    atomic_json(out/'resolved.json',dict(resolved,campaign_id=digest(campaign),cells=cells,
        configurations_per_repeat=2,unresolved=[],runnable=True))
    atomic_json(out/'host.private.json',load(parent/'host.private.json'))
    (out/'host.private.json').chmod(0o600)
    Ledger(out)
    return dict(out=str(out),diagnostic_only=True,sessions=sessions,roomy_cpu_bytes=roomy_cpu_bytes)


def prepare_operation(parent,out,model,*,waves=2):
    parent,out = Path(parent).resolve(),Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new diagnostic destination')
    if type(waves) is not int or not 2<=waves<=6:
        raise ValueError('Development operation requires two to six complete waves')
    if (parent/'protocol.lock.json').exists():
        raise ValueError('Develop before freeze; preserve frozen confirmation')
    resolved = load(parent/'resolved.json')
    if resolved['unresolved']:
        raise ValueError('Resolve parent inputs before a development operation')
    cell = copy.deepcopy(next(c for c in resolved['cells'] if c['model']==model and c['cell']=='B06'))
    binding = cell['workload_binding']
    plan,corpus = load(binding['plan']),load(binding['corpus'])
    if digest(plan)!=binding['plan_digest'] or digest(corpus)!=binding['corpus_digest']:
        raise ValueError('Development inputs changed')
    if 'operational_corpus' in plan:
        from .campaign import operational_workload
        corpus = operational_workload(plan,corpus)
    plan.update(tier='operational',waves=waves,wave_interval_s=0,operational_advice_probes=True)
    out.mkdir(parents=True)
    for key,value in (('plan',plan),('corpus',corpus)):
        path = out/'inputs'/(key+'.json')
        atomic_json(path,value)
        binding[key],binding[key+'_digest'] = str(path),digest(value)
    cell.update(systems=['arc','wait'],confirmation_binding=None)
    campaign = load(parent/'campaign.json')
    campaign.update(parent_campaign_id=resolved['campaign_id'],
        diagnostic_only=dict(kind='persistent-operation-development',model=model,waves=waves,
            scope='Development inputs; not parent screen, pilot or operational confirmation'),
        input_identities=[cell_inputs(cell)])
    atomic_json(out/'campaign.json',campaign)
    atomic_json(out/'resolved.json',dict(resolved,campaign_id=digest(campaign),cells=[cell],
        configurations_per_repeat=2,unresolved=[],runnable=True))
    host = load(parent/'host.private.json')
    atomic_json(out/'host.private.json',host)
    (out/'host.private.json').chmod(0o600)
    Ledger(out)
    return dict(out=str(out),diagnostic_only=True,model=model,waves=waves,
        next='Run campaign run --phase screen on this separate diagnostic; retain full operational confirmation later')


if __name__=='__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--waves',type=int,default=2)
    args = parser.parse_args()
    print(json.dumps(prepare_operation(args.parent,args.out,args.model,waves=args.waves),indent=2))
