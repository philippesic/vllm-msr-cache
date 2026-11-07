"""Resume, tamper and writer-contention checks using real filesystem artifacts."""
import pytest

from benchmarks.serving_suite.campaign_contracts import Ledger, atomic_json, digest, load, seal_tree


def finish(ledger, identity):
    attempt, directory = ledger.begin(identity)
    row = ledger.index()[attempt]
    atomic_json(directory / 'result.json', {'status': 'passed'})
    seal = seal_tree(directory)
    ledger.event(row['case_id'], attempt, 'verified', **row['details'], seal=seal)
    return attempt, directory


def test_resume_requires_exact_identity_and_seal(tmp_path):
    ledger = Ledger(tmp_path)
    identity = {'model': 'one', 'repeat': 1, 'source': 'a'}
    attempt, directory = finish(ledger, identity)
    assert ledger.reusable(identity)['attempt_id'] == attempt
    assert ledger.reusable({**identity, 'source': 'b'}) is None
    atomic_json(directory / 'result.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='Changed artifact'):
        ledger.reusable(identity)


@pytest.fixture
def resident_team(tmp_path):
    import json
    ledger = Ledger(tmp_path)
    identity = dict(model='fixture',cell='B10',system='wait',phase='screen',repeat=0,source='source',workload='workload')
    attempt,directory = ledger.begin(identity,block_id='matched-block')
    sessions = [dict(id=str(i),graph=dict(outputs=['synthesis'],nodes=[
        dict(id=n,kind='request') for n in ('coordinator','source','validation','limitations','synthesis')]+[
        dict(id='tool-'+str(j),kind='tool') for j in range(3)])) for i in range(4)]
    rows = [dict(wave=0,source_id=s['id'],node_id=n['id'],success=True,terminal_complete=True)
        for s in sessions for n in s['graph']['nodes'] if n['kind']=='request']
    case = dict(name='case',expected_requests=20,summary=dict(completed=20,failed=0),load_duration_s=10,
        evaluation=dict(status='inconclusive',errors=[],unavailable_activity=[],inactive_activity=['restore']),
        cleanup=dict(verified=True),activity=dict(store=100,restore=0,cpu_recency=40),
        metrics={'vllm:prefix_cache_hits_total':100},
        campaign=dict(completed_units=4,offered_units=4,completed_units_per_s=.4,tools=12,native_cache=dict(
            complete=True,receipts=20,returning_requests=4,loaded_tokens=0,returning_loaded_tokens=0,
            prior_cpu_evicted_key_observations=0,returning_requests_with_prior_cpu_eviction=0,preemptions=0)))
    atomic_json(directory/'result.json',dict(status='inconclusive'))
    atomic_json(directory/'measurement/result.json',dict(plan=dict(campaign_adapter=True,waves=1),cases=[case]))
    atomic_json(directory/'measurement/corpus.json',dict(sessions=sessions))
    (directory/'measurement/case').mkdir()
    (directory/'measurement/case/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    row=ledger.index()[attempt]
    event=ledger.event(row['case_id'],attempt,'failed',**row['details'],seal=seal_tree(directory),measurement_status='inconclusive')
    return ledger,identity,event,directory,case


def test_gpu_resident_team_reinterpretation_preserves_terminal_attempt(resident_team):
    from benchmarks.serving_suite.campaign_validity import measurement_validity
    from benchmarks.serving_suite.campaign_contracts import file_hash
    ledger,identity,event,directory,case=resident_team
    before=file_hash(directory/'measurement/result.json')
    validity=measurement_validity(directory,event)
    assert validity['valid'] and validity['reinterpreted']
    assert validity['classification']=='valid_gpu_resident_team_control'
    assert ledger.reusable(identity)['attempt_id']==event['attempt_id']
    assert ledger.index()[event['attempt_id']]['state']=='failed'
    assert file_hash(directory/'measurement/result.json')==before
    atomic_json(directory/'result.json',dict(status='passed'))
    with pytest.raises(ValueError,match='Changed artifact'):
        ledger.reusable(identity)


@pytest.mark.parametrize('problem',['missing_counter','eviction','preemption','no_gpu_hits','no_treatment',
    'incomplete_native','request_failure','partial_graph','wrong_cell','runner_failure'])
def test_resident_control_review_never_accepts_missing_or_failed_measurements(resident_team,problem):
    import copy
    from benchmarks.serving_suite.campaign_validity import measurement_validity
    ledger,identity,original,directory,case=resident_team
    event=copy.deepcopy(original)
    if problem=='missing_counter': case['activity']['restore']=None
    elif problem=='eviction': case['campaign']['native_cache']['returning_requests_with_prior_cpu_eviction']=1
    elif problem=='preemption': case['campaign']['native_cache']['preemptions']=1
    elif problem=='no_gpu_hits': case['metrics']['vllm:prefix_cache_hits_total']=0
    elif problem=='no_treatment': case['activity']['cpu_recency']=0
    elif problem=='incomplete_native': case['campaign']['native_cache']['complete']=False
    elif problem=='request_failure': case['summary']['failed']=1
    elif problem=='partial_graph':
        path=directory/'measurement/case/requests.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
    elif problem=='wrong_cell': event['details']['identity']['cell']='B06'
    elif problem=='runner_failure': (directory/'failure.json').write_text('{}')
    assert not measurement_validity(directory,event,case)['valid']


@pytest.fixture
def resident_repository(tmp_path):
    import json
    ledger=Ledger(tmp_path)
    identity=dict(model='qwen3_8b_awq',cell='B06',system='wait',phase='confirm',repeat=0,
        source='frozen',profile=dict(repo_pressure_sessions=8))
    attempt,directory=ledger.begin(identity,block_id='whole-block')
    sessions=[dict(id=str(i),graph=dict(nodes=[dict(id='r'+str(j),kind='request') for j in range(4)]+
        [dict(id='t'+str(j),kind='tool') for j in range(3)])) for i in range(8)]
    rows=[dict(wave=0,source_id=s['id'],node_id=n['id'],success=True,terminal_complete=True)
        for s in sessions for n in s['graph']['nodes'] if n['kind']=='request']
    case=dict(name='case',expected_requests=32,summary=dict(completed=32,failed=0),
        evaluation=dict(status='inconclusive',errors=[],unavailable_activity=[],inactive_activity=['restore']),
        cleanup=dict(verified=True),activity=dict(store=100,restore=0,cpu_recency=20),
        metrics={'vllm:prefix_cache_hits_total':100},campaign=dict(completed_units=8,offered_units=8,tools=24,
            native_cache=dict(complete=True,receipts=32,returning_requests=24,loaded_tokens=0,
                returning_loaded_tokens=0,prior_cpu_evicted_key_observations=0,
                returning_requests_with_prior_cpu_eviction=0,preemptions=0)))
    atomic_json(directory/'result.json',dict(status='inconclusive'))
    atomic_json(directory/'measurement/result.json',dict(plan=dict(campaign_adapter=True,waves=1),cases=[case]))
    atomic_json(directory/'measurement/corpus.json',dict(sessions=sessions))
    (directory/'measurement/case').mkdir()
    (directory/'measurement/case/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    row=ledger.index()[attempt]
    event=ledger.event(row['case_id'],attempt,'failed',**row['details'],seal=seal_tree(directory),measurement_status='inconclusive')
    return ledger,identity,event,directory,case


@pytest.mark.parametrize('cell',['B06','B12'])
def test_complete_repository_keeps_inactive_pressure_separate(resident_repository,cell):
    import copy
    from benchmarks.serving_suite.campaign_validity import measurement_validity
    from benchmarks.serving_suite.campaign_contracts import file_hash
    ledger,identity,event,directory,case=resident_repository
    event=copy.deepcopy(event)
    event['details']['identity']['cell']=cell
    before=file_hash(directory/'measurement/result.json')
    validity=measurement_validity(directory,event)
    assert validity['valid'] and validity['cpu_pressure_evidence']=='inconclusive'
    assert validity['classification']=='complete_repository_without_cpu_restore'
    assert validity['cpu_restore_activity']=='inactive'
    assert ledger.reusable(identity)['attempt_id']==event['attempt_id']
    assert ledger.index()[event['attempt_id']]['state']=='failed'
    assert file_hash(directory/'measurement/result.json')==before


@pytest.mark.parametrize('problem',['wrong_model','wrong_profile','wrong_cell','eviction','preemption',
    'no_gpu_hits','no_treatment','missing_counter','runner_error','duplicate_request','missing_tool'])
def test_repository_activity_interpretation_does_not_hide_failures(resident_repository,problem):
    from benchmarks.serving_suite.campaign_validity import measurement_validity
    ledger,identity,event,directory,case=resident_repository
    i=event['details']['identity']
    if problem=='wrong_model': i['model']='another-model'
    elif problem=='wrong_profile': i['profile']['repo_pressure_sessions']=16
    elif problem=='wrong_cell': i['cell']='B08'
    elif problem=='eviction': case['campaign']['native_cache']['prior_cpu_evicted_key_observations']=1
    elif problem=='preemption': case['campaign']['native_cache']['preemptions']=1
    elif problem=='no_gpu_hits': case['metrics']['vllm:prefix_cache_hits_total']=0
    elif problem=='no_treatment': case['activity']['cpu_recency']=0
    elif problem=='missing_counter': case['activity']['restore']=None
    elif problem=='runner_error': case['evaluation']['errors']=['request timeout']
    elif problem=='missing_tool': case['campaign']['tools']=23
    elif problem=='duplicate_request':
        p=directory/'measurement/case/requests.jsonl'; lines=p.read_text().splitlines()
        lines[-1]=lines[0]; p.write_text('\n'.join(lines)+'\n')
    assert not measurement_validity(directory,event,case)['valid']


def test_interrupted_case_is_new_whole_attempt(tmp_path):
    ledger = Ledger(tmp_path)
    identity = {'model': 'one', 'repeat': 1}
    old, directory = ledger.begin(identity)
    ledger.recover()
    assert ledger.index()[old]['state'] == 'interrupted'
    assert ledger.reusable(identity) is None
    new, fresh = ledger.begin(identity)
    assert old != new and directory != fresh
    with pytest.raises(ValueError, match='immutable'):
        ledger.event(digest(identity), old, 'running')


def test_index_is_rebuildable_and_event_chain_detects_mutation(tmp_path):
    ledger = Ledger(tmp_path)
    finish(ledger, {'cell': 'B06'})
    (tmp_path / 'index.json').write_text('corrupt index')
    assert len(ledger.index()) == 1
    path = tmp_path / 'ledger.jsonl'
    path.write_text(path.read_text().replace('B06', 'B09'))
    with pytest.raises(ValueError, match='chain'):
        ledger.index()


def test_torn_tail_retained_before_recovery(tmp_path):
    ledger = Ledger(tmp_path)
    finish(ledger, {'cell': 'B06'})
    with (tmp_path / 'ledger.jsonl').open('ab') as stream:
        stream.write(b'{"torn":')
    with pytest.raises(ValueError, match='Torn'):
        ledger.index()
    assert len(ledger.recover()) == 1
    assert next(tmp_path.glob('torn-ledger-*.bin')).read_bytes() == b'{"torn":'


def test_competing_writer_cannot_recover_live_attempt(tmp_path):
    first, second = Ledger(tmp_path), Ledger(tmp_path)
    with first.writer():
        with pytest.raises(RuntimeError, match='active writer'):
            with second.writer():
                second.recover()


def test_immutable_source_snapshot_detaches_later_edits(tmp_path):
    from benchmarks.serving_suite.source_snapshot import materialize
    from benchmarks.serving_suite.campaign_contracts import file_hash
    source = tmp_path/'source'
    source.mkdir()
    (source/'module.py').write_text('value = 1\n')
    files = {'module.py': file_hash(source/'module.py')}
    identity = {'files': files, 'digest': digest(files)}
    snapshot = materialize(source, tmp_path/'cache', identity)
    (source/'module.py').write_text('value = 2\n')
    assert (snapshot/'module.py').read_text() == 'value = 1\n'
    assert materialize(source, tmp_path/'cache', identity) == snapshot
    (snapshot/'module.py').write_text('value = 3\n')
    with pytest.raises(ValueError, match='snapshot changed'):
        materialize(source, tmp_path/'cache', identity)


def test_real_source_snapshot_runs_without_historical_campaigns(tmp_path):
    import json
    import subprocess
    import sys
    from benchmarks.serving_suite.campaign import ROOT,source_snapshot
    from benchmarks.serving_suite.source_snapshot import materialize
    identity = source_snapshot()
    snapshot = materialize(ROOT,tmp_path/'cache',identity)
    observed = json.loads(subprocess.check_output([sys.executable,'-c',
        'from benchmarks.serving_suite.campaign import source_snapshot; import json; print(json.dumps(source_snapshot()))'],cwd=snapshot))
    assert observed['digest']==identity['digest']
    assert not (snapshot/'benchmarks/eviction_prefetch_campaign.py').exists()
