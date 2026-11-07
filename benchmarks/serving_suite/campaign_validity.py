"""Separate complete measurements from absent CPU-restore activity.

Keep original sealed results and ledger states. B10 and the eight-session anchor
repository cohort can finish through GPU prefix reuse without CPU restoration.
This interpretation never qualifies a pressure or native-correctness fixture.
"""
from pathlib import Path
import json

from .campaign_contracts import load

VERSION = 'stage6-measurement-validity-v3'


def positive(value):
    return type(value) in (int,float) and value>0


def measurement_validity(directory,event,case=None):
    directory = Path(directory)
    identity = event['details'].get('identity',{})
    if case is None:
        path = directory/'measurement/result.json'
        if not path.is_file():
            return dict(version=VERSION,valid=False,classification='no_measurement')
        case = load(path)['cases'][0]
    evaluation = case.get('evaluation',{})
    result = dict(version=VERSION,valid=event['state']=='verified' and evaluation.get('status')=='passed',
        classification='original_measurement',raw_ledger_state=event['state'],raw_evaluation=evaluation.get('status'))
    restore = case.get('activity',{}).get('restore')
    if identity.get('system') in ('arc','wait') and type(restore) in (int,float):
        result['cpu_restore_activity'] = 'active' if positive(restore) else 'inactive' if restore==0 else 'invalid'
        native = case.get('campaign',{}).get('native_cache',{})
        result['cpu_pressure_evidence'] = ('observed' if positive(native.get('returning_requests_with_prior_cpu_eviction'))
            else 'inconclusive')
    if result['valid']:
        return result
    if (event['state']!='failed' or identity.get('phase') not in ('screen','pilot','confirm') or
        identity.get('cell') not in ('B06','B10','B12') or identity.get('system') not in ('arc','wait') or
        event['details'].get('measurement_status')!='inconclusive' or
        (directory/'failure.json').exists() or not (directory/'result.json').is_file() or
        load(directory/'result.json').get('status')!='inconclusive' or
        evaluation!=dict(status='inconclusive',errors=[],unavailable_activity=[],inactive_activity=['restore'])):
        return result
    measured = load(directory/'measurement/result.json')
    plan = measured['plan']
    corpus = load(directory/'measurement/corpus.json')
    native = case.get('campaign',{}).get('native_cache',{})
    activity = case.get('activity',{})
    summary = case.get('summary',{})
    expected = case.get('expected_requests',0)
    team = identity['cell']=='B10'
    sessions, requests, tools = (4,20,12) if team else (8,32,24)
    if (not team and (identity.get('model')!='qwen3_8b_awq' or
            identity.get('profile',{}).get('repo_pressure_sessions')!=8) or
        not plan.get('campaign_adapter') or plan.get('waves')!=1 or len(corpus['sessions'])!=sessions or
        summary.get('completed')!=expected or expected!=requests or summary.get('failed')!=0 or
        not case.get('cleanup',{}).get('verified') or not native.get('complete') or
        native.get('receipts')!=expected or not positive(native.get('returning_requests')) or
        any(native.get(k)!=0 for k in ('loaded_tokens','returning_loaded_tokens',
            'prior_cpu_evicted_key_observations','returning_requests_with_prior_cpu_eviction','preemptions')) or
        activity.get('restore')!=0 or not positive(activity.get('store')) or
        identity['system']=='wait' and not positive(activity.get('cpu_recency')) or
        not positive(case.get('metrics',{}).get('vllm:prefix_cache_hits_total')) or
        case['campaign'].get('tools')!=tools or
        sum(n['kind']=='tool' for s in corpus['sessions'] for n in s['graph']['nodes'])!=tools or
        case['campaign'].get('completed_units')!=sessions or case['campaign'].get('offered_units')!=sessions):
        return result
    rows = [json.loads(line) for line in (directory/'measurement'/case['name']/'requests.jsonl').read_text().splitlines()]
    expected_keys = {(0,s['id'],n['id']) for s in corpus['sessions'] for n in s['graph']['nodes'] if n['kind']=='request'}
    actual = [(r.get('wave'),r.get('source_id'),r.get('node_id')) for r in rows]
    if (len(actual)!=expected or len(set(actual))!=expected or set(actual)!=expected_keys or
        any(not r.get('success') or not r.get('terminal_complete') for r in rows)):
        return result
    result.update(valid=True,reinterpreted=True,
        classification='valid_gpu_resident_team_control' if team else 'complete_repository_without_cpu_restore',
        cpu_restore_activity='inactive',cpu_pressure_evidence='inconclusive',
        scope='Complete declared graph with GPU prefix hits, active treatment, zero CPU evictions/preemptions/restores. CPU-pressure and offload-benefit evidence are inconclusive; original sealed status remains unchanged.')
    return result
