"""Derive capability gates from sealed native measurements, never authored wins."""
from pathlib import Path

from .campaign_contracts import Ledger, atomic_json, digest, load, local_path, verify_tree


def serving_files(source):
    names = {'runner','published','workload','campaign_workload','workflow','repo_workload',
             'report','resources','contracts','advice_probe','campaign'}
    return {k:v for k,v in source['files'].items() if k.startswith(('semantic_offload/','vendor/vllm/',
        'harness/','tools/','benchmarks/serving_suite/published_observer/')) or
        k.startswith('benchmarks/serving_suite/') and Path(k).stem in names or
        k=='benchmarks/research_checks/wait_hint_runtime_probe.py'}


def current_serving(attempt, expected=None):
    from .campaign import source_snapshot
    return serving_files(load(attempt/'measurement/source.json'))==(expected if expected is not None else serving_files(source_snapshot()))


def native_checks(out, expected_source=None):
    out = Path(out)
    resolved = load(out/'resolved.json')
    selected = {}
    for event in Ledger(out).index().values():
        identity = event['details'].get('identity',{})
        if event['state']!='verified' or identity.get('phase')!='preflight' or identity.get('system')!='wait':
            continue
        directory = local_path(out,event['details']['path'])
        verify_tree(directory,event['details']['seal'])
        if not current_serving(directory,expected_source):
            continue
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
    from .campaign import source_snapshot
    expected_source = serving_files(source_snapshot())
    native = native_checks(out,expected_source)
    screens = completed_blocks(out,'screen')
    screens = {key:block for key,block in screens.items() if all(current_serving(
        local_path(out,data['event']['details']['path']),expected_source) for data in block['systems'].values())}
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
    registry = dict(passed=False)
    for event in Ledger(out).index().values():
        identity = event['details'].get('identity',{})
        if event['state']!='verified' or identity.get('phase')!='preflight' or identity.get('system')!='registry':
            continue
        directory = local_path(out,event['details']['path'])
        verify_tree(directory,event['details']['seal'])
        if current_serving(directory,expected_source):
            case = load(directory/'measurement/result.json')['cases'][0]
            registry = dict(case.get('campaign',{}).get('registry_probe',{}),attempt=event['attempt_id'])
    pressure = {}
    for model in sorted({c['model'] for c in resolved['cells']}):
        block = screens.get((model,'B06',0))
        if block:
            observed = block['systems']['arc']['case']['campaign'].get('native_cache',{})
            count = observed.get('returning_requests_with_prior_cpu_eviction')
            pressure[model] = dict(passed=count is not None and count>0,arc_native_cache=observed)
    roomy = dict(passed=False)
    base,large = screens.get(('qwen3_8b_awq','B06',0)),screens.get(('qwen3_8b_awq','B07',0))
    if base and large:
        small = base['systems']['arc']['case']['campaign'].get('native_cache',{}).get('returning_requests_with_prior_cpu_eviction')
        big = large['systems']['arc']['case']['campaign'].get('native_cache',{}).get('returning_requests_with_prior_cpu_eviction')
        roomy = dict(passed=small is not None and small>0 and big is not None and big<=small/2,
            base_returning_requests_with_prior_cpu_eviction=small,roomy_returning_requests_with_prior_cpu_eviction=big,
            rule='At least half fewer returning requests with prior CPU eviction in ARC; envelope selection never uses candidate speed')
    passed = (native['passed'] and registry.get('passed') and not missing and roomy['passed'] and
        all(pressure.get(m,{}).get('passed') for m in {c['model'] for c in resolved['cells']}) and
        all(external.get(s,{}).get('passed') for s in ('continuum','upstream')))
    result = dict(schema='serving-campaign-capability-v1',all_required_correctness_passed=passed,
                  native=native,external=external,registry_development_probe=registry,missing_screen_cells=missing,
                  pressure=pressure,roomy=roomy,
                  scope='Functional/activation/completeness gates, not performance or production qualification')
    atomic_json(out/'capability.json',result)
    return result
