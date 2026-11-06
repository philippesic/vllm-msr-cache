"""CPU-only negative controls for operational harness changes; no inference."""
import ast
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import operational_waves_v2_nov3 as waves
import operational_state_v2_nov3 as state
import operational_abort_v2_nov3 as aborts
from operational_cancel_probe_v2_nov3 import probe_payload
from semantic_offload.request_priority import gateway_priority

H=Path(__file__).resolve().parent


def args(**kw):
    d=dict(name='cpu-case', waves=3, wave_seeds='14009,15013,16001', wave_settle_s=0,
           wave_release_interval_s=0, cancel_probe=False, registry_limit=4096,
           clients=1, turns=1)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.mark.parametrize('name,index', [('',0),('bad|name',0),('ok',-1),('ok',True),('ok',1.5)])
def test_identity_rejects_invalid(name,index):
    with pytest.raises(ValueError):waves.wave_identity(name,index)


def test_independent_module_namespace_and_shared_helper():
    a=waves.load_wave_module('cpu-case',0);b=waves.load_wave_module('cpu-case',1)
    assert a is not b and a.rid(0)!=b.rid(0)
    assert a.producer.__globals__['RUN']==a.RUN
    assert b.producer.__globals__['RUN']==b.RUN
    assert a.gateway_priority is gateway_priority is b.gateway_priority
    a.RUN='modified'
    assert 'modified' in a.rid(0) and 'modified' not in b.rid(0)


def test_public_helper_ast_equal_frozen_helper():
    old=ast.parse((H/'return_priority_nov3.py').read_text())
    new=ast.parse(Path(gateway_priority.__code__.co_filename).read_text())
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)


def test_missing_state_unavailable():
    out=state.snapshot()
    assert all(v is None for v in out['scheduler_lengths'].values())
    assert out['native_request_count'] is None
    assert out['session_registry_limit'] is None


def test_lengths_do_not_enumerate_values_or_mutate():
    class Poison:
        def __iter__(self):raise AssertionError('must not iterate')
    scheduler=SimpleNamespace(_observed_prefix_generations={'a':Poison()},_wait_hint_results={},_observed_session_limit=128)
    before=dict(vars(scheduler))
    out=state.snapshot(scheduler=scheduler)
    assert out['scheduler_lengths']['_observed_prefix_generations']==1
    assert out['scheduler_lengths']['_wait_hint_results']==0
    assert out['session_registry_limit']==128
    assert vars(scheduler)==before


def test_unexpected_collection_type_unavailable():
    class Poison:
        def __len__(self):raise AssertionError('must not call arbitrary length')
    out=state.lengths(SimpleNamespace(_wait_hint_results=Poison()),['_wait_hint_results'])
    assert out['_wait_hint_results'] is None


def test_capacity_classification_not_all_noactions():
    rows=[{'response':{'status':'no_action','reason':'session_registry_capacity'}},
          {'response':{'status':'no_action','reason':'gpu_prefix_full'}},
          {'response':{'status':'accepted'}},{'error':'transport'}]
    assert waves.classify_capacity(rows)=={'session_registry_capacity':1,'other_no_action':1,'accepted':1,'other':1}


@pytest.mark.parametrize('overlap',[False,True])
def test_wave_lifecycle_shared_clock_distinct_names_no_reset(monkeypatch,overlap):
    observed=[];exports=[]
    def loader(name,index):
        async def exercise(handle,p,a,r):
            observed.append((index,a.application_namespace,a.gateway_origin_ns,a.seed,p['namespace']))
            await asyncio.sleep(.001*(3-index))
            r.update(requests=[{'success':True,'natural_eos':True,'index':0}],hints=[],http_client_configuration={'force_close':True})
        return SimpleNamespace(exercise=exercise)
    monkeypatch.setattr(waves,'load_wave_module',loader)
    monkeypatch.setattr(waves,'control',lambda h,op:exports.append(op) or {'scalar':True})
    result={};namespace={'stable':'cache'}
    asyncio.run(waves.exercise(None,{'namespace':namespace},args(wave_release_interval_s=.001 if overlap else 0),result))
    assert len(result['requests'])==3 and len(result['waves'])==3
    assert len({x[1] for x in observed})==3
    assert len({x[2] for x in observed})==1
    assert {x[3] for x in observed}=={14009,15013,16001}
    assert all(x[4] is namespace for x in observed)
    assert exports==['export']*(1 if overlap else 6)
    if overlap:
        ordered=sorted(result['waves'],key=lambda x:x['wave_index'])
        assert [r['intended_wave_release_ns']-result['load_start_ns'] for r in ordered]==[0,1000000,2000000]
        assert all(r['actual_wave_start_ns']>=r['intended_wave_release_ns'] for r in ordered)


def test_overlap_failure_cancels_and_reaps_siblings(monkeypatch):
    closed=[]
    def loader(name,index):
        async def exercise(handle,p,a,r):
            try:
                if index==1:raise RuntimeError('injected')
                await asyncio.sleep(10)
            finally:closed.append(index)
        return SimpleNamespace(exercise=exercise)
    monkeypatch.setattr(waves,'load_wave_module',loader)
    monkeypatch.setattr(waves,'control',lambda h,op:pytest.fail('no export during failed overlap'))
    result={}
    with pytest.raises(RuntimeError,match='injected'):
        asyncio.run(waves.exercise(None,{},args(wave_release_interval_s=.001),result))
    assert 0 in closed and 1 in closed
    assert result['load_end_ns']>=result['load_start_ns']


def test_non_eos_wave_is_failure_not_success(monkeypatch):
    async def exercise(handle,p,a,r):r.update(requests=[{'success':True,'natural_eos':False}],hints=[])
    monkeypatch.setattr(waves,'load_wave_module',lambda *_:SimpleNamespace(exercise=exercise))
    monkeypatch.setattr(waves,'control',lambda *_:{})
    result={}
    with pytest.raises(RuntimeError,match='natural-EOS'):
        asyncio.run(waves.exercise(None,{},args(waves=1),result))
    assert result['waves'][0]['status']=='request_failures'
    assert result['requests'][0]['natural_eos'] is False


def test_abort_observer_preserves_inputs_result_and_only_actual_returns(monkeypatch):
    monkeypatch.setattr(aborts,'ABORT_COUNT',0);aborts.ABORTS.clear()
    aborted=SimpleNamespace(request_id='real',status=SimpleNamespace(name='FINISHED_ABORTED'))
    finished=SimpleNamespace(request_id='already',status=SimpleNamespace(name='FINISHED_STOPPED'))
    marker=[];out=[aborted,finished];seen=[]
    class Scheduler:
        def finish_requests(self,*a,**kw):seen.append((a,kw));return out
    aborts.install(Scheduler)
    result=Scheduler().finish_requests(marker,finished_status='status')
    assert result is out and seen==[((marker,),{'finished_status':'status'})]
    assert aborts.snapshot()['count']==1
    assert [x['request_id'] for x in aborts.snapshot()['records']]==['real']


def test_abort_observer_propagates_original_exception(monkeypatch):
    monkeypatch.setattr(aborts,'ABORT_COUNT',0);aborts.ABORTS.clear()
    class Scheduler:
        def finish_requests(self,*a,**kw):raise RuntimeError('native failure')
    aborts.install(Scheduler)
    with pytest.raises(RuntimeError,match='native failure'):Scheduler().finish_requests(None)
    assert aborts.snapshot()['count']==0


def test_abort_observer_bounded_and_no_already_finished_claim(monkeypatch):
    monkeypatch.setattr(aborts,'ABORT_COUNT',0);aborts.ABORTS.clear()
    class Scheduler:
        def finish_requests(self,n):return [] if n<0 else [SimpleNamespace(request_id=str(n),status=SimpleNamespace(name='FINISHED_ABORTED'))]
    aborts.install(Scheduler);scheduler=Scheduler()
    for i in range(100):scheduler.finish_requests(i)
    scheduler.finish_requests(-1)
    out=aborts.snapshot()
    assert out['count']==100 and len(out['records'])==32
    assert out['records'][0]['request_id']=='68'


def test_probe_new_identity_preserves_prompt_remaining_context_and_no_retry():
    source={'request_id':'old','prompt':[1,2,3], 'max_tokens':7677,'priority':0,
            'kv_transfer_params':{'session_id':'old','wait_hint_event_id':'old','observed_prefix_id':'old','observed_prefix_boundary_tokens':16}}
    before=copy.deepcopy(source)
    out,clock=probe_payload(source,'cpu-case',1,0)
    assert clock['priority_credit_ms']==8000 and clock['priority_turn']==0
    assert out['priority']==clock['priority_clock_ns']//1000000
    assert source==before
    assert out['prompt']==source['prompt'] and out['max_tokens']==7677
    assert out['request_id']!='old' and out['kv_transfer_params']['session_id']!='old'
    assert 'wait_hint_event_id' not in out['kv_transfer_params']


def test_operational_hook_is_common_and_no_residency_enumeration_added():
    hook=(H/'hooks-operational-v2/sitecustomize.py').read_text()
    assert 'install(m.Scheduler)' in hook
    assert 'operational_scalar_state' in hook and 'operational_worker_state' in hook
    assert 'operational_abort_transitions' in hook
    assert 'if LEAN:return' in hook

@pytest.mark.parametrize('early_terminal,backend_match,health_finish,qualified',[
    (False,True,'stop',True), (True,True,'stop',False),
    (False,False,'stop',False), (False,True,'length',False),
])
def test_cancel_probe_fake_streams_require_actual_abort_and_eos(monkeypatch,early_terminal,backend_match,health_finish,qualified):
    import json
    import operational_cancel_probe_v2_nov3 as probe
    posts=[];closes=[]
    class Content:
        def __init__(self,lines):self.lines=lines
        def __aiter__(self):
            async def gen():
                for line in self.lines:yield line
            return gen()
    class Response:
        status=200
        def __init__(self,body,first):
            choice={'token_ids':[10],'text':'text','finish_reason':('stop' if early_terminal else None) if first else health_finish}
            chunk={'id':'cmpl-'+body['request_id'],'choices':[choice]}
            self.content=Content([('data: '+json.dumps(chunk)+'\n').encode(),b'data: [DONE]\n'])
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        def close(self):closes.append(True)
    class Client:
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        def post(self,url,json,trace_request_ctx):
            posts.append((json,trace_request_ctx))
            return Response(json,len(posts)==1)
    def export(handle,op):
        assert op=='export'
        expected='cmpl-'+posts[0][0]['request_id']+'-0'
        return {'scheduler':{'operational_abort_transitions':{'count':1,'records':[{'request_id':'unrelated-internal-name','external_mapping_available':True,'external_request_id':posts[0][0]['request_id'] if backend_match else 'foreign','status':'FINISHED_ABORTED'}]}}}
    monkeypatch.setattr(probe,'control',export)
    body={'request_id':'original','prompt':[1,2,3],'max_tokens':7677,'priority':0,
          'kv_transfer_params':{'run_id':'old','cell_id':'old','session_id':'old'}}
    result={'waves':[{'requests':[{'turn':0,'payload':copy.deepcopy(body)} for _ in range(4)]}]}
    operation=probe.run(SimpleNamespace(base_url=lambda:'fake'),args(),result,SimpleNamespace(create_http_session=lambda r:Client()),0)
    if qualified:asyncio.run(operation)
    else:
        with pytest.raises(RuntimeError,match='backend abort'):asyncio.run(operation)
    outcome=result['cancellation_probe']
    assert len(posts)==5 # no retry/replacement, including negative controls
    assert outcome['qualified'] is qualified
    assert len(outcome['health_requests'])==4
    assert bool(closes) is (not early_terminal)
    assert posts[0][1]['kind']=='intentional_disconnect'
    assert all(p[1]['kind']=='health_generation' for p in posts[1:])
