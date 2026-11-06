"""Ordering and replay verification contracts for Stage 5 measurements."""
from collections import Counter
import pytest
from benchmarks.research_checks.stage5_optimization import balanced_order
from benchmarks.research_checks.wait_hint_qualification import check_replay_row


def test_two_arm_order_alternates():
    order=balanced_order(['arc','wait'],6,42)
    groups=[[name for r,name in order if r==repeat] for repeat in range(6)]
    assert all(groups[i]==groups[i-1][::-1] for i in range(1,6))


def test_three_arm_six_runs_balance_positions_and_precedence():
    order=balanced_order(['arc','wait','candidate'],6,42)
    groups=[[name for r,name in order if r==repeat] for repeat in range(6)]
    for position in range(3):assert Counter(g[position] for g in groups)==dict(arc=2,wait=2,candidate=2)
    for a,b in [('arc','wait'),('arc','candidate'),('wait','candidate')]:
        assert sum(g.index(a)<g.index(b) for g in groups)==3
    assert balanced_order(['arc','wait','candidate'],6,42)==order


def test_replay_operational_probe_does_not_weaken_work_identity():
    expected={('source',0):([1,2],[3,4])}
    row=dict(wave=0,source_id='source',turn=0,success=True,payload={'prompt':[1,2]},output_token_ids=[3,4])
    check_replay_row(row,expected)
    with pytest.raises(ValueError,match='identical'):
        check_replay_row({**row,'output_token_ids':[3,5]},expected)
    check_replay_row(dict(wave=-1,session_id='cancel',success=True,payload={'kv_transfer_params':{'suite_probe_id':'id'}}),expected)
    with pytest.raises(ValueError,match='Unexpected'):
        check_replay_row(dict(wave=-1,session_id='ordinary',success=True),expected)


@pytest.mark.parametrize('enabled',[False,True])
def test_abandoned_replay_hints_are_owned_and_do_not_add_requests(tmp_path,monkeypatch,enabled):
    import asyncio
    import json
    from benchmarks.serving_suite import replay_workload as replay
    class Client:
        def __init__(self,*args,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(replay.aiohttp,'ClientSession',Client)
    monkeypatch.setattr(replay.aiohttp,'TCPConnector',lambda **kw:None)
    calls=[]
    async def completion(client,url,payload,identity,writer,**kwargs):
        await asyncio.sleep(.03 if identity['source_id']=='slow' else 0)
        row=dict(**identity,payload=payload,output_token_ids=[2],success=True)
        writer(row);return row
    async def hint(client,url,previous,sid,owner,namespace,hints,**kwargs):
        row=dict(hint={'owner_id':owner,'session_id':sid},response={'ready_cpu_prefix_keys':1})
        hints.append(row);calls.append(owner);return row
    monkeypatch.setattr(replay,'completion',completion)
    monkeypatch.setattr(replay,'publish_hint',hint)
    sessions=[dict(id=name,abandoned_hint=name=='fast',replay=[dict(
        payload={'prompt':list(range(16)),'kv_transfer_params':{},'priority':123},tokens=[2])]) for name in ('fast','slow')]
    plan=dict(request_timeout_s=1,publish_hints=enabled,hint_refresh_s=.001)
    rows=[];hints=[]
    async def run():
        result=await replay.run_wave('unused',plan,sessions,None,0,rows.append,'candidate',{},hints,tmp_path/'streams')
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
        return result
    result=asyncio.run(run())
    assert len(result)==len(rows)==2
    assert all('priority' not in row['payload'] for row in rows)
    assert bool(calls)==enabled
    assert all(owner.endswith('sample-1') and 'abandon' not in owner for owner in calls)
    assert all(h.get('abandoned_advice_diagnostic') for h in hints)
    events=[json.loads(line) for line in (tmp_path/'workflow-0.jsonl').read_text().splitlines()]
    assert sum(e['event']=='session_end' for e in events)==2
    assert any(e['event']=='abandoned_hint' for e in events)==enabled


def test_priority_preflight_rejects_unsupported_condition_before_execution():
    from benchmarks.research_checks.stage5_optimization import preflight
    spec=dict(plan={'continuation_credit_ms':8000},conditions={'arc':{},'wait':{'continuation_credit_ms':16000}})
    with pytest.raises(ValueError,match='fixed priority credit'):
        preflight(spec)
    spec['conditions']['wait']['continuation_credit_ms']=0
    preflight(spec)


def test_coalescing_requires_latest_full_fresh_ack():
    from benchmarks.serving_suite.repo_workload import hint_refresh_delay
    receipt=dict(hint={'prefix_boundary_tokens':64},ended_ns=1_000_000_000,
        response={'reason':'cpu_prefix_recency_refreshed','ready_cpu_prefix_keys':4})
    assert hint_refresh_delay([receipt],1,1_250_000_000)==.75
    assert hint_refresh_delay([receipt],1,2_000_000_000)==0
    for bad in ({'error':'error'},{'cancelled_before_ack':True},{'response':None},
                {'response':{'reason':'nomination_deadline_elapsed','ready_cpu_prefix_keys':4}},
                {'response':{'reason':'cpu_prefix_recency_refreshed','ready_cpu_prefix_keys':3}}):
        assert hint_refresh_delay([receipt,{**receipt,**bad}],1,1_250_000_000)==0
    assert hint_refresh_delay([],1,1_250_000_000)==0


@pytest.mark.parametrize('cancel_tool',[False,True])
def test_overlapped_initial_hint_is_joined_and_owned(tmp_path,monkeypatch,cancel_tool):
    import asyncio
    import time
    from types import SimpleNamespace
    from benchmarks.serving_suite import repo_workload as live
    class Tokenizer:
        def apply_chat_template(self,*args,**kw):return list(range(32))
        def decode(self,*args,**kw):return 'response'
    async def scenario():
        began=asyncio.Event(); release=asyncio.Event(); ack=asyncio.Event(); cancelled=[]
        async def hint(client,url,previous,sid,owner,namespace,hints,**kw):
            row=dict(hint={'owner_id':owner,'event_id':'event','prefix_boundary_tokens':32},started_ns=time.monotonic_ns())
            began.set()
            try:
                await release.wait();ack.set()
                row['response']={'reason':'cpu_prefix_recency_refreshed','ready_cpu_prefix_keys':2}
            except asyncio.CancelledError:
                cancelled.append(True);row['cancelled_before_ack']=True;raise
            finally:
                row['ended_ns']=time.monotonic_ns();hints.append(row)
            return row
        async def tool(spec,cwd,timeout,artifact,on_running=None):
            await began.wait()
            proc=SimpleNamespace(returncode=None)
            refresh=asyncio.create_task(on_running(proc))
            await asyncio.sleep(0)
            proc.returncode=0;refresh.cancel();await asyncio.gather(refresh,return_exceptions=True)
            assert not cancelled and not ack.is_set()
            if cancel_tool:raise asyncio.CancelledError()
            release.set()
            return dict(started_ns=time.monotonic_ns(),ended_ns=time.monotonic_ns(),kind='test',exit_code=0,
                        success=True,output='actual',argv=['test'])
        async def completion(client,url,payload,identity,writer,**kw):
            if identity['turn']:assert ack.is_set()
            row=dict(**identity,payload=payload,output_token_ids=[5,6],success=True)
            writer(row);return row
        monkeypatch.setattr(live,'publish_hint',hint);monkeypatch.setattr(live,'execute_tool',tool)
        monkeypatch.setattr(live,'completion',completion)
        plan=dict(concurrency=1,turns=2,request_timeout_s=1,publish_hints=True,hint_timing='before-tool',
                  hint_overlap_initial=True,hint_refresh_s=1,max_model_len=128,max_tokens=8,seed=1)
        sessions=[dict(id='session',context='source',turns=['a','b'],tools=[{}],repository=str(tmp_path))]
        rows=[];hints=[]
        call=live.run_wave('http://unused',plan,sessions,Tokenizer(),0,rows.append,'candidate',{},hints,tmp_path/'streams')
        if cancel_tool:
            with pytest.raises(asyncio.CancelledError):await call
        else:await call
        assert len(rows)==(1 if cancel_tool else 2)
        assert bool(cancelled)==cancel_tool
        assert len(hints)==1
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    asyncio.run(scenario())


def test_cohort_completion_uses_designated_groups_and_excludes_abandoned_events(tmp_path):
    import json
    from benchmarks.research_checks.stage5_report import cohort_completion
    corpus={'sessions':[{'id':'a','publish_hints':True},{'id':'b','publish_hints':False}]}
    events=[]
    for name,end in [('a',3_000_000_000),('b',5_000_000_000)]:
        events.extend([dict(event='session_start',session_id='w0:'+name,source_id=name,ns=1_000_000_000),
                       dict(event='session_end',session_id='w0:'+name,source_id=name,ns=end)])
    events.append(dict(event='abandoned_hint',ns=8_000_000_000))
    path=tmp_path/'workflow-0.jsonl'
    path.write_text('\n'.join(map(json.dumps,events)))
    assert cohort_completion(tmp_path,corpus,dict(sessions=2,waves=1))=={
        'hinted':{'count':1,'mean_s':2},'unhinted':{'count':1,'mean_s':4}}
    path.write_text('\n'.join(map(json.dumps,events[:-2])))
    with pytest.raises(ValueError,match='Incomplete'):cohort_completion(tmp_path,corpus,dict(sessions=2,waves=1))


@pytest.mark.parametrize('window',[1500,3000,True,-1])
def test_hint_preflight_checks_actual_protocol(window):
    from benchmarks.research_checks.stage5_optimization import preflight
    with pytest.raises(ValueError,match='declared estimate'):
        preflight(dict(plan={'hint_declared_wait_ms':window},conditions={'wait':{}}))


def test_prompt_scope_does_not_publish_decode_tail_or_mutate_input():
    from benchmarks.serving_suite.repo_workload import known_prefix_tokens
    previous=dict(payload={'prompt':list(range(32))},output_token_ids=list(range(32,49)))
    assert known_prefix_tokens(previous)==list(range(48))
    assert known_prefix_tokens(previous,False)==list(range(32))
    assert previous['payload']['prompt']==list(range(32))
    assert previous['output_token_ids']==list(range(32,49))


def test_prompt_storage_preflight_requires_matching_publication_scope():
    from benchmarks.research_checks.stage5_optimization import preflight
    extra={'offload_prompt_only':True,'wait_hint_completed_transcript':True}
    plan={'publish_hints':True,'candidate_config':{'kv_connector_extra_config':extra}}
    spec=dict(plan=plan,conditions={'wait':{}})
    with pytest.raises(ValueError,match='cannot publish'):preflight(spec)
    extra['wait_hint_completed_transcript']=False
    with pytest.raises(ValueError,match='prompt-scoped'):preflight(spec)
    plan['hint_completed_transcript']=False
    preflight(spec)
