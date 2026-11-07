"""Versioned measurement interpretation without rewriting sealed attempts.

The B10 team DAG can reuse GPU-resident prefixes without any CPU restoration.
Its original treatment-only restore requirement incorrectly rejected that valid
control. Keep the raw result and ledger state; apply the same bounded predicate
on collection and resume. This never qualifies a pressure or native fixture.
"""
from pathlib import Path
import json

from .campaign_contracts import load

VERSION = 'stage6-measurement-validity-v2'


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
    if result['valid']:
        return result
    if (event['state']!='failed' or identity.get('phase') not in ('screen','pilot','confirm') or
        identity.get('cell')!='B10' or identity.get('system') not in ('arc','wait') or
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
    if (not plan.get('campaign_adapter') or plan.get('waves')!=1 or len(corpus['sessions'])!=4 or
        summary.get('completed')!=expected or expected!=20 or summary.get('failed')!=0 or
        not case.get('cleanup',{}).get('verified') or not native.get('complete') or
        native.get('receipts')!=expected or not positive(native.get('returning_requests')) or
        any(native.get(k)!=0 for k in ('loaded_tokens','returning_loaded_tokens',
            'prior_cpu_evicted_key_observations','returning_requests_with_prior_cpu_eviction','preemptions')) or
        activity.get('restore')!=0 or not positive(activity.get('store')) or
        identity['system']=='wait' and not positive(activity.get('cpu_recency')) or
        not positive(case.get('metrics',{}).get('vllm:prefix_cache_hits_total')) or
        case['campaign'].get('tools')!=12 or
        sum(n['kind']=='tool' for s in corpus['sessions'] for n in s['graph']['nodes'])!=12 or
        case['campaign'].get('completed_units')!=4 or case['campaign'].get('offered_units')!=4):
        return result
    rows = [json.loads(line) for line in (directory/'measurement'/case['name']/'requests.jsonl').read_text().splitlines()]
    expected_keys = {(0,s['id'],n['id']) for s in corpus['sessions'] for n in s['graph']['nodes'] if n['kind']=='request'}
    actual = [(r.get('wave'),r.get('source_id'),r.get('node_id')) for r in rows]
    if (len(actual)!=expected or len(set(actual))!=expected or set(actual)!=expected_keys or
        any(not r.get('success') or not r.get('terminal_complete') for r in rows)):
        return result
    result.update(valid=True,reinterpreted=True,classification='valid_gpu_resident_team_control',
        scope='Complete team DAG with GPU prefix hits, active treatment, zero CPU evictions/preemptions/restores. No CPU offload benefit is implied; original sealed status remains unchanged.')
    return result
