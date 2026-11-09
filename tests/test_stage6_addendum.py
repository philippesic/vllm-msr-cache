"""CPU-only safety, lifecycle, fixed precision and whole-block recovery proofs."""
import asyncio
from contextlib import nullcontext

import pytest

from benchmarks.serving_suite import campaign_addendum as add
from benchmarks.serving_suite.addendum_advice import schedule
from benchmarks.serving_suite.addendum_report import class_metrics, estimate
from benchmarks.serving_suite.campaign_contracts import Ledger, atomic_json, digest, load, write_protocol


def test_locked_counts_orders_and_fixed_precision():
    blocks=add.schedule(17)
    assert sum(len(b['cases']) for b in blocks)==78
    for section, expected in [('repository',18),('overhead',48),('advice',12)]:
        assert sum(len(b['cases']) for b in blocks if b['section']==section)==expected
    assert sum(add.LIMITS[s+'_attempts'] for s in ('repository','overhead','advice'))==85
    repo=[b for b in blocks if len(b['cases'])==4]
    for position in range(4):
        assert len({b['cases'][position][1] for b in repo})==4
    assert estimate([.98]*23,24,margin=.97)['paired_95_interval'] is None
    full=estimate([.98]*24,24,margin=.97)
    assert full['bound_met'] and full['paired_95_interval']==pytest.approx([.98,.98])
    assert estimate([.94]*24,24,margin=.97)['bound_met'] is False


def test_fairness_latency_includes_client_queue_and_retains_tail_counts():
    rows=[dict(hint_group='unhinted',success=True,ttft_s=1,latency_s=2,client_queue_s=3,
               prompt_token_count=9,output_token_ids=[1,2])]
    c=class_metrics(rows)['unhinted']
    assert c['end_user_ttft_mean_s']==4 and c['end_user_e2e_mean_s']==5
    assert c['requests']==1 and c['end_user_e2e_p99_s']==5


def test_effective_pressure_and_prediction_graphs_keep_exact_traffic(tmp_path):
    from benchmarks.serving_suite.contracts import SCHEMA
    plan=dict(schema=SCHEMA,tier='screen',repeats=1,sessions=16,turns=4,waves=1,
        concurrency=8,tool_concurrency=4,max_tokens=256,gpu_blocks=1024,cpu_bytes=2793406464,
        max_model_len=8192,seed=17,startup_timeout_s=900,request_timeout_s=900,
        tool_timeout_s=180,case_timeout_s=3600,sample_interval_s=.25,
        gpu_memory_utilization=.85,required_activity=['restore'],
        candidate_config=dict(kv_connector='SemanticOffloadingConnector',
            kv_connector_extra_config=dict(cpu_bytes_to_use=2793406464)))
    repo=dict(schema='semantic-serving-corpus-v1',synthetic=False,provenance='Source fixture',
        sessions=[dict(id=f'task-{i}',repository=str(tmp_path),context='actual source',
            turns=['plan','search','validate','final'],tools=[dict(argv=['rg','source'],valid_exit_codes=[0])]*3)
            for i in range(16)])
    qa=dict(repo,sessions=[dict(id=f'qa-{i}',context='real QA reference',turns=['question']) for i in range(64)])
    inputs=add.make_inputs(repo,qa,repo,plan,dict(plan,max_tokens=128,concurrency=4))
    for name,expected in [('R01',(16,64,48)),('Q01',(64,64,0)),('H00',(36,84,48)),('H01',(36,84,48)),('H02',(36,84,48))]:
        p,c=inputs[name]
        assert (p['sessions'],p['expected_requests_per_wave'],p['expected_tools_per_wave'])==expected
        assert p['case_timeout_s']==900 and p['cpu_bytes']==2793406464
    assert inputs['H00'][1]==inputs['H01'][1]==inputs['H02'][1]
    assert [s['arrival_s'] for s in inputs['H00'][1]['sessions'][-16:]]==[0]*8+[30]*8
    assert inputs['H01'][0]['addendum_advice']=='accurate'
    assert inputs['H02'][0]['addendum_advice']=='misleading'
    assert inputs['R01'][0]['candidate_config']==plan['candidate_config']


@pytest.mark.parametrize('mode,expected',[('accurate',0),('misleading',1)])
def test_valid_no_return_prediction_is_abstained_or_published_then_cancelled(mode,expected):
    async def scenario():
        event=asyncio.Event(); calls=[]; events=[]
        async def announce():
            row=dict(hint=dict(event_id='observed-prefix'),response=dict(ready_cpu_prefix_keys=4))
            calls.append(row); return row
        task=await schedule(mode=mode,scenario='abandoned',index=0,later=[],event=event,
            announce=announce,emit=lambda *a,**k:events.append((a,k)))
        assert len(calls)==expected
        if task:
            assert calls[0]['prediction_experiment']['false_positive']
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            assert task.done()
    asyncio.run(scenario())


def test_accurate_late_prediction_does_not_block_tools_or_publish_after_first_token():
    async def scenario():
        event=asyncio.Event(); calls=[]
        async def announce():
            calls.append(1)
            return dict(hint=dict(event_id='x'),response={})
        task=await schedule(mode='accurate',scenario='late',index=0,later=[1],event=event,
            announce=announce,emit=lambda *a,**k:None,late_delay_s=.001)
        assert not calls # Returned to tool dependency immediately.
        event.set()
        await task
        assert not calls # A returning token cancels obsolete delayed advice.
    asyncio.run(scenario())


@pytest.fixture
def fake_campaign(tmp_path,monkeypatch):
    from benchmarks.serving_suite import campaign, campaign_inputs, campaign_execution, resources, verify
    from benchmarks.research_checks import stage6_healthcheck
    import harness.gpu_lock as locks
    out=tmp_path/'addendum';out.mkdir()
    source=dict(files={});source['digest']=digest(source['files'])
    cells=[]
    for name in ('R01','Q01'):
        plan=out/'inputs'/name/'plan.json';corpus=out/'inputs'/name/'corpus.json'
        atomic_json(plan,dict(startup_timeout_s=1,case_timeout_s=1))
        atomic_json(corpus,dict(sessions=[]))
        cells.append(dict(model=add.MODEL,cell=name,artifact_identity='fixed',profile={},
            workload_binding=dict(plan=str(plan),corpus=str(corpus)),confirmation_binding=None))
    atomic_json(out/'resolved.json',dict(campaign_id='test',host_id='test',cells=cells))
    atomic_json(out/'host.private.json',dict(runtime_python='/pinned/serving/python'))
    protocol=dict(source=source,parent=str(tmp_path/'primary'),snapshot=str(tmp_path),
        cells_digest=digest(cells),limits=add.LIMITS,schedule=[])
    def lock(blocks):
        protocol['schedule']=blocks
        protocol['cells_digest']=digest(load(out/'resolved.json')['cells'])
        write_protocol(out,protocol)
    monkeypatch.setattr(campaign,'source_snapshot',lambda:source)
    monkeypatch.setattr(add,'primary_complete',lambda p:True)
    monkeypatch.setattr(locks,'gpu_reserved',lambda f:f)
    monkeypatch.setattr(locks,'GPULock',lambda *a,**k:nullcontext())
    monkeypatch.setattr(resources,'available_gpu',lambda:None)
    monkeypatch.setattr(campaign_inputs,'attest',lambda *a,**k:{})
    monkeypatch.setattr(campaign_inputs,'unchanged',lambda x:None)
    monkeypatch.setattr(stage6_healthcheck,'coordinators',lambda p:[])
    monkeypatch.setattr(verify,'verify',lambda d:None)
    launched=[]
    def worker(command,**kwargs):
        from pathlib import Path
        assert command[0] == '/pinned/serving/python'
        d=Path(command[-1]); i=load(d/'identity.json');launched.append(i)
        case=dict(evaluation=dict(status='passed'),hints=[],campaign=dict(native_cache=dict(
            complete=True,returning_loaded_tokens=1,returning_requests_with_prior_cpu_eviction=1)))
        atomic_json(d/'measurement/result.json',dict(status='passed',cases=[case]))
    monkeypatch.setattr(campaign_execution,'run_worker',worker)
    return out,lock,launched,worker


def test_no_launch_before_primary_completion(fake_campaign,monkeypatch):
    out,lock,launched,_=fake_campaign
    lock([])
    monkeypatch.setattr(add,'primary_complete',lambda p:False)
    assert add.run(out)['status']=='waiting_for_primary'
    assert not launched and not (out/'execution.json').exists()


def test_failed_pressure_gate_preserves_control_and_does_not_expand(fake_campaign,monkeypatch):
    from benchmarks.serving_suite import campaign_execution
    out,lock,launched,worker=fake_campaign
    lock([dict(id='repository-gate',section='repository',cases=[['R01','arc']],purpose_gate=True),
          dict(id='repository-0',section='repository',cases=[['R01',s] for s in ('arc','wait','continuum','upstream')]),
          dict(id='overhead-0',section='overhead',cases=[['Q01','O0'],['Q01','O2']])])
    def inactive(command,**kwargs):
        from pathlib import Path
        worker(command,**kwargs)
        p=Path(command[-1])/'measurement/result.json';data=load(p)
        data['cases'][0]['campaign']['native_cache']['returning_loaded_tokens']=0
        atomic_json(p,data)
    monkeypatch.setattr(campaign_execution,'run_worker',inactive)
    result=add.run(out)
    assert result['status']=='complete' and 'repository' in result['purpose_failures']
    assert [i['system'] for i in launched]==['arc','O0','O2']
    assert len(Ledger(out).index())==3
    # A restart rechecks the sealed gate rather than bypassing it.
    assert add.run(out,resume=True)['purpose_failures']==result['purpose_failures']


def test_recovery_needs_diagnosis_and_replaces_whole_block(fake_campaign,monkeypatch):
    from benchmarks.serving_suite import campaign_execution
    out,lock,launched,worker=fake_campaign
    block=dict(id='repository-0',section='repository',cases=[['R01',s] for s in ('arc','wait','continuum','upstream')])
    lock([block])
    def fail_second(command,**kwargs):
        if len(launched)==1:
            raise RuntimeError('diagnosed transport fixture')
        worker(command,**kwargs)
    monkeypatch.setattr(campaign_execution,'run_worker',fail_second)
    with pytest.raises(RuntimeError,match='transport fixture'):
        add.run(out)
    with pytest.raises(RuntimeError,match='repair authorization'):
        add.run(out,resume=True)
    evidence=out/'diagnosis.txt';evidence.write_text('Retained test transport failure; inputs and policy unchanged.')
    add.repair(out,'repository-0','transport',evidence,'Transport failed before second measurement')
    monkeypatch.setattr(campaign_execution,'run_worker',worker)
    assert add.run(out,resume=True)['status']=='complete'
    events=Ledger(out).index()
    assert len(events)==6 # Original full/failed attempts remain alongside four replacements.
    done, groups=add.block_state(out,block,events)
    assert len(done)==4 and len(groups)==2
    assert len({e['details']['block_id'] for e in done})==1
    with pytest.raises(ValueError,match='already spent'):
        add.repair(out,'repository-0','transport',evidence,'Do not expand')


def test_overall_wall_budget_is_not_reset_by_resume(fake_campaign):
    out,lock,launched,_=fake_campaign
    lock([dict(id='repository-gate',section='repository',cases=[['R01','arc']],purpose_gate=True)])
    atomic_json(out/'execution.json',dict(status='attention',started_unix=1,section_started={}))
    result=add.run(out,resume=True)
    assert result['status']=='budget_exhausted' and result['started_unix']==1 and not launched


def test_health_catches_stopped_owner_and_missing_gpu_telemetry(fake_campaign,monkeypatch):
    import time
    import psutil
    from benchmarks.research_checks import stage6_healthcheck as health
    out,lock,_,_=fake_campaign
    lock([])
    atomic_json(out/'execution.json',dict(status='running',started_unix=time.time(),section_started={},
        owner=dict(pid=99999999,create_time=1),active_attempt=None))
    def unavailable():
        raise RuntimeError('fixture telemetry failure')
    monkeypatch.setattr(health,'gpu_snapshot',unavailable)
    result=add.inspect(out)
    assert result['status']=='attention'
    assert {'coordinator_stopped','missing_gpu_telemetry'}<=set(result['issues'])


def test_interrupted_capture_is_sealed_before_terminal_recovery(tmp_path):
    from benchmarks.serving_suite.campaign_contracts import verify_tree
    ledger=Ledger(tmp_path)
    aid,directory=ledger.begin(dict(model='fixture',phase='screen'))
    capture=directory/'measurement/case/requests.jsonl';capture.parent.mkdir(parents=True)
    capture.write_text('{"partial_capture":')
    ledger.recover(seal_pending=True)
    event=ledger.index()[aid]
    assert event['state']=='interrupted'
    assert verify_tree(directory,event['details']['seal'])>=3
    assert capture.read_text()=='{"partial_capture":'


def test_worker_uses_cell_runtime_override(fake_campaign, monkeypatch):
    from benchmarks.serving_suite import campaign_execution
    out, lock, _, worker = fake_campaign
    cells = load(out/'resolved.json')
    cells['cells'][0]['profile']['runtime_python'] = '/model/serving/python'
    atomic_json(out/'resolved.json', cells)
    lock([dict(id='repository-0',section='repository',cases=[['R01','arc']])])
    commands = []
    def capture(command, **kwargs):
        commands.append(command[0])
        worker(['/pinned/serving/python', *command[1:]], **kwargs)
    monkeypatch.setattr(campaign_execution, 'run_worker', capture)
    assert add.run(out)['status'] == 'complete'
    assert commands == ['/model/serving/python']
