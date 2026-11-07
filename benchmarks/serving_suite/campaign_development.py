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
