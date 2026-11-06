"""Derive capability gates from sealed native measurements, never authored wins."""
from pathlib import Path

from .campaign_contracts import Ledger, atomic_json, digest, load, local_path, verify_tree


def native_checks(out):
    out = Path(out)
    resolved = load(out/'resolved.json')
    selected = {}
    for event in Ledger(out).index().values():
        identity = event['details'].get('identity',{})
        if event['state']!='verified' or identity.get('phase')!='preflight' or identity.get('system')!='wait':
            continue
        directory = local_path(out,event['details']['path'])
        verify_tree(directory,event['details']['seal'])
        result = load(directory/'measurement/result.json')
        if not result['plan'].get('runtime_probe'):
            continue
        case = result['cases'][0]
        proof = load(directory/'measurement'/case['name']/'runtime-probe-0.json')
        expected = {'duplicate_is_idempotent','wrong_namespace_rejected','future_information_rejected',
                    'stale_advice_rejected','fresh_reference_recomputed_entire_prompt',
                    'return_really_restored_cpu_prefix','cached_equals_fresh_generation'}
        actual = {check['check']:check['passed'] for check in proof['checks']}
        if not expected<=actual.keys() or not all(actual.values()) or not proof['passed']:
            continue
        cell = next(c for c in resolved['cells'] if c['model']==identity['model'] and c['cell']=='B06')
        if identity.get('runtime_identity') != cell.get('runtime_identity'):
            continue
        if not case.get('effective_geometry',{}).get('cpu_pinned') or not case['cleanup']['verified']:
            continue
        previous = selected.get(identity['model'])
        if not previous or event['sequence']>previous['sequence']:
            selected[identity['model']] = dict(attempt=event['attempt_id'],path=event['details']['path'],
                seal=event['details']['seal'],sequence=event['sequence'],checks=actual,
                geometry=case['effective_geometry'])
    missing = sorted({c['model'] for c in resolved['cells']}-selected.keys())
    return dict(passed=not missing,missing_models=missing,models=selected)


def qualify(out):
    from .campaign_report import completed_blocks
    out = Path(out)
    resolved = load(out/'resolved.json')
    native = native_checks(out)
    screens = completed_blocks(out,'screen')
    missing = [c['model']+':'+c['cell'] for c in resolved['cells'] if (c['model'],c['cell'],0) not in screens]
    external = {}
    anchor = screens.get(('qwen3_8b_awq','B06',0))
    if anchor:
        for system in ('continuum','upstream'):
            data = anchor['systems'].get(system)
            if data:
                case = data['case']
                activation = case['campaign']['policy_activation']
                external[system] = dict(passed=bool(activation and activation['verified'] and
                    case['effective_geometry']['cpu_pinned'] and case['cleanup']['verified']),
                    attempt=data['event']['attempt_id'],activation=activation,geometry=case['effective_geometry'])
    passed = native['passed'] and not missing and all(external.get(s,{}).get('passed') for s in ('continuum','upstream'))
    result = dict(schema='serving-campaign-capability-v1',all_required_correctness_passed=passed,
                  native=native,external=external,missing_screen_cells=missing,
                  scope='Functional/activation/completeness gates, not performance or production qualification')
    atomic_json(out/'capability.json',result)
    return result
