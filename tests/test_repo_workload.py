"""Real subprocess ownership, causal hint window, and agent report contracts."""
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import psutil
import pytest
from aiohttp import web

from benchmarks.serving_suite.repo_workload import build_corpus, execute_tool, publish_hint
from benchmarks.serving_suite.agent_report import distribution, native_events, wait_hint_config
from benchmarks.serving_suite.runner import main, ROOT


def test_actual_repository_inputs_and_tools():
    corpus = build_corpus(ROOT)
    assert len(corpus['sessions']) == 8 and not corpus['synthetic']
    for s in corpus['sessions']:
        assert len(s['turns']) == 4 and len(s['tools']) == 3
        assert s['input_files'] and 'FILE ' in s['context']
        assert s['tools'][1]['argv'][-1].startswith('tests/')
        assert 'wait_s' not in s


def test_source_paths_must_exist_and_stay_in_repo(tmp_path):
    definition = {'schema':'repository-investigation-tasks-v1', 'provenance':'test',
                  'tasks':[{'source':'../escape','tests':'x','search':'x'}]}
    p = tmp_path/'tasks.json'
    p.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match='Invalid repository input'):
        build_corpus(tmp_path, p)


def test_real_subprocess_output_and_failure_are_preserved(tmp_path):
    for code, valid in ((0,True),(7,False)):
        row = asyncio.run(execute_tool(dict(kind='test',argv=[sys.executable,'-c',f'print("actual output");raise SystemExit({code})'],valid_exit_codes=[0]),tmp_path,5,tmp_path/f'{code}.txt'))
        assert row['success'] is valid and row['exit_code']==code
        assert row['output']=='actual output\n' and row['output_bytes']==14
        assert (tmp_path/f'{code}.txt').read_text()==row['output']
        assert not psutil.pid_exists(row['pid'])


def test_real_tool_timeout_reaps_process_group(tmp_path):
    row = asyncio.run(execute_tool(dict(kind='test',argv=[sys.executable,'-c','import time;time.sleep(60)'],valid_exit_codes=[0]),tmp_path,.05,tmp_path/'timeout.txt'))
    assert not row['success'] and 'TimeoutError' in row['error']
    assert not psutil.pid_exists(row['pid'])


def test_hint_is_sent_during_real_tool_and_has_no_future_input(tmp_path):
    async def scenario():
        received = []
        async def endpoint(request):
            raw = await request.json()
            received.append((time.monotonic_ns(),raw))
            return web.json_response(dict(schema='semantic-known-prefix-wait-hint-ack-v1',event_id=raw['event_id'],status='no_action',reason='test'))
        app = web.Application()
        app.router.add_post('/plugins/semantic-offload/wait-hints',endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner,'127.0.0.1',0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        hints = []
        try:
            async with aiohttp.ClientSession() as client:
                async def during(proc):
                    await publish_hint(client,f'http://127.0.0.1:{port}',dict(payload={'prompt':list(range(64))},output_token_ids=[65,66]),'session','owner',{},hints)
                row = await execute_tool(dict(kind='test',argv=[sys.executable,'-c','import time;time.sleep(.1);print("real")'],valid_exit_codes=[0]),tmp_path,3,tmp_path/'tool.txt',during)
            assert len(received)==1 and hints[0]['ended_ns'] < row['ended_ns']
            ns,raw = received[0]
            assert row['started_ns'] <= ns < row['ended_ns']
            assert raw['prefix_boundary_tokens']==64
            assert not {'actual_resume_ns','future_suffix','output'} & raw.keys()
        finally:
            await runner.cleanup()
    asyncio.run(scenario())


def test_agent_plan_defaults_and_equal_control(tmp_path):
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    out=tmp_path/'plan'
    assert main(['agent','--model',str(model),'--output',str(out),'--plan'])==0
    plan=json.loads((out/'plan.json').read_text())
    assert plan['gpu_blocks']==1024 and plan['max_model_len']==8192
    assert plan['publish_hints'] and plan['qualification_trace'] and plan['natural_eos']
    left,right=[wait_hint_config(plan,enabled=e)['kv_connector_extra_config'] for e in (False,True)]
    assert left['cpu_bytes_to_use']==right['cpu_bytes_to_use']
    assert left['preservation_audit'] and right['preservation_audit']
    assert not left['wait_hint_cpu_retention_only'] and right['wait_hint_cpu_retention_only']
    assert left['prefetch_budget_fraction']==0 < right['prefetch_budget_fraction']
    assert right['wait_hint_diagnostic_no_dispatch']


def test_three_way_plan_rotates_arms_and_requires_explicit_semantic_config(tmp_path):
    from benchmarks.serving_suite.runner import case_order, command
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    cfg=tmp_path/'semantic.json'
    config=wait_hint_config(dict(cpu_bytes=2**20,model=str(model)),True)
    config['kv_connector_extra_config'].update(instrumentation_only=False,capture_stride=8)
    cfg.write_text(json.dumps(config))
    out=tmp_path/'plan'
    assert main(['agent','--model',str(model),'--output',str(out),'--plan',
        '--semantic-config',str(cfg),'--client-concurrency','0','--continuation-credit-ms','8000'])==0
    plan=json.loads((out/'plan.json').read_text())
    assert [case_order(plan,i) for i in range(3)]==[['arc','wait','semantic'],['wait','semantic','arc'],['semantic','arc','wait']]
    assert '--scheduling-policy' in command(plan,'arc',1234)[0]
    assert plan['prompt_source_accounting_version']==3


def test_baseline_does_not_require_a_restore_that_only_advice_enables():
    from benchmarks.serving_suite.runner import required_activity_for_arm
    plan=dict(tier='agent',required_activity=['store','restore','cpu_recency'])
    assert required_activity_for_arm(plan,'arc')==['store']
    assert required_activity_for_arm(plan,'candidate')==['store','restore','cpu_recency']
    plan['tier']='screen'
    assert required_activity_for_arm(plan,'arc')==['store','restore']


def test_three_way_accepts_cpu_only_wait_control_and_rejects_semantic_control(tmp_path):
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    config=wait_hint_config(dict(cpu_bytes=2**20,model=str(model)),True)
    config['kv_connector_extra_config']['wait_hint_cpu_retention_gpu_gate']=False
    control=tmp_path/'control.json';control.write_text(json.dumps(config))
    semantic=tmp_path/'semantic.json'
    config['kv_connector_extra_config']['instrumentation_only']=False
    semantic.write_text(json.dumps(config))
    args=['agent','--model',str(model),'--plan','--candidate-config',str(control),
          '--semantic-config',str(semantic),'--output',str(tmp_path/'good')]
    assert main(args)==0
    plan=json.loads((tmp_path/'good'/'plan.json').read_text())
    assert not plan['candidate_config']['kv_connector_extra_config']['wait_hint_cpu_retention_gpu_gate']
    control.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='CPU-only wait-hint ARC'):
        main(args[:-1]+[str(tmp_path/'bad')])


def test_reports_preserve_missing_and_malformed():
    assert distribution([])['mean'] is None
    with pytest.raises(json.JSONDecodeError):
        native_events('marker {broken','marker ')


def test_server_visible_sessions_and_queue_hints_stop_at_first_token(tmp_path):
    from benchmarks.serving_suite.repo_workload import run_wave
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs): return list(range(80+len(messages)))
        def decode(self,tokens,**kwargs): return 'observed answer'
    async def scenario():
        initial=[]; all_initial=asyncio.Event(); sent=[]; hints=[]; rows=[]
        async def generate(request):
            raw=await request.json();sent.append(raw)
            if raw['request_id'].endswith('sample-0'):
                initial.append(raw)
                if len(initial)==2: all_initial.set()
                await asyncio.wait_for(all_initial.wait(),2)
            await asyncio.sleep(.06)
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"token_ids":[10],"text":"a"}]}\n\n')
            await asyncio.sleep(.06)
            await response.write(b'data: {"choices":[{"token_ids":[11],"text":"b","finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
            return response
        async def hint(request):
            raw=await request.json()
            return web.json_response(dict(schema='semantic-known-prefix-wait-hint-ack-v1',event_id=raw['event_id'],status='no_action',reason='test'))
        app=web.Application();app.router.add_post('/v1/completions',generate);app.router.add_post('/plugins/semantic-offload/wait-hints',hint)
        runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        sessions=[dict(id=str(i),context='source',repository=str(tmp_path),turns=['one','two'],
            tools=[dict(kind='search',argv=[sys.executable,'-c','print("live")'],valid_exit_codes=[0])]) for i in range(2)]
        plan=dict(concurrency=1,client_concurrency=0,tool_concurrency=1,request_timeout_s=5,max_model_len=512,max_tokens=20,
            seed=1,publish_hints=True,hint_refresh_s=.01,hint_timing='before-tool',hint_through_queue=True,
            continuation_credit_ms=8000,turns=2,waves=1)
        try:
            await run_wave(f'http://127.0.0.1:{port}',plan,sessions,Tokenizer(),0,rows.append,'candidate',{},hints,tmp_path/'streams')
        finally: await runner.cleanup()
        assert len(rows)==4 and all(r['success'] for r in rows)
        assert max(r['client_queue_s'] for r in rows)<.02
        assert all(p['priority']>=0 for p in initial)
        assert all(p['priority']<0 for p in sent if p not in initial)
        events=[json.loads(s) for s in (tmp_path/'workflow-0.jsonl').read_text().splitlines()]
        assert sum(e['event']=='pre_tool_hint' for e in events)==2
        tools=sorted([e for e in events if e['event']=='tool'],key=lambda e:e['started_ns'])
        assert len(tools)==2 and tools[1]['started_ns']>=tools[0]['ended_ns']
        assert all(e['queue_s']>=0 and e['offered_ns']<=e['started_ns'] for e in tools)
        windows=[e for e in events if e['event']=='queue_hint_window']
        assert len(windows)==2 and all(e['publications']>0 and e['first_token_observed'] for e in windows)
        for row in rows:
            if row['turn']:
                completed=[h for h in hints if h['hint']['owner_id']==row['request_id'] and not h.get('cancelled_before_ack')]
                assert completed and max(h['ended_ns'] for h in completed)<row['first_token_ns']+.01*1e9
    asyncio.run(scenario())


def test_canonical_request_ids_join_native_preservation_records():
    from semantic_offload.connector import _derive_external_request_id
    rid='sc|suite|case|w0-s3|sample-2'
    assert _derive_external_request_id('cmpl-'+rid+'-0-a1234567')==rid
    assert _derive_external_request_id('cmpl-agent-0-3-2-0-a1234567') is None


@pytest.mark.parametrize('policy', ['upstream', 'continuum'])
def test_mixed_published_sessions_omit_only_unhinted_retention_metadata(tmp_path, policy):
    from benchmarks.serving_suite.repo_workload import run_wave
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs): return list(range(80))
        def decode(self, tokens, **kwargs): return 'observed answer'
    async def scenario():
        sent=[]
        async def generate(request):
            sent.append(await request.json())
            return web.Response(text='data: {"choices":[{"token_ids":[10],"text":"a","finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', content_type='text/event-stream')
        app=web.Application();app.router.add_post('/v1/completions',generate)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        sessions=[dict(id=str(i),context='source',repository=str(tmp_path),turns=['one','two'],
            publish_hints=i==0,tools=[dict(kind='search',argv=[sys.executable,'-c','print("live")'],valid_exit_codes=[0])]) for i in range(2)]
        plan=dict(concurrency=2,client_concurrency=0,tool_concurrency=1,request_timeout_s=5,
            max_model_len=512,max_tokens=20,seed=1,publish_hints=False,turns=2,
            continuation_credit_ms=8000,published_variant=policy,published_runtime={})
        rows=[]
        try:
            await run_wave(f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}',plan,
                sessions,Tokenizer(),0,rows.append,'candidate',{},[],tmp_path/'streams')
        finally: await runner.cleanup()
        assert len(rows)==4 and all(row['success'] for row in rows)
        metadata='kv_hints' if policy=='upstream' else 'job_id'
        for row in rows:
            assert (metadata in row['payload']) == (row['hint_group']=='hinted')
            assert 'kv_transfer_params' not in row['payload']
            assert (row['payload']['priority']<0)==(row['turn']>0)
        assert len(sent)==4
    asyncio.run(scenario())


def test_agent_config_reaches_native_cpu_advice_branch():
    from semantic_offload.manager import PreservationOffloadingManager
    from tests.test_step_1_5_prefetch import (_StubBlockPool, _insert_resident,
        _make_connector_scheduler, _register_wait_prefix, _wait_hint, to_key)
    cfg=wait_hint_config(dict(cpu_bytes=2**20,model='local-model'),True)['kv_connector_extra_config']
    manager=PreservationOffloadingManager(num_blocks=8,cache_policy='arc')
    keys=[to_key(i) for i in range(4)]
    for i,key in enumerate(keys): _insert_resident(manager,key,i)
    scheduler=_make_connector_scheduler(manager,_StubBlockPool(20,free_blocks=5))
    for name in ('wait_hint_cpu_retention_only','wait_hint_cpu_retention_gpu_gate','prefetch_budget_fraction'):
        setattr(scheduler,'_'+name,cfg[name])
    prefix,boundary=_register_wait_prefix(scheduler,keys)
    result=scheduler.publish_wait_hint(_wait_hint(scheduler,prefix,boundary))
    assert result['reason']=='cpu_prefix_recency_refreshed'
    assert result['ready_cpu_prefix_keys']==4
    assert result['reserved_gpu_blocks']==result['speculative_transfer_bytes']==0


def test_semantic_cpu_advice_preserves_scores_evidence_and_references():
    from semantic_offload.manager import SemanticOffloadingManager
    from tests.test_step_1_5_prefetch import _insert_resident, to_key
    manager=SemanticOffloadingManager(num_blocks=4,chain_aware=True,session_aware=True)
    keys=[to_key(i) for i in range(3)]
    for i,key in enumerate(keys): _insert_resident(manager,key,i)
    manager.relevance_ema['mean']={key:float(i) for i,key in enumerate(keys)}
    policy=manager._policy
    before=dict(manager.relevance_ema['mean'])
    assert manager.refresh_wait_hint_cpu_recency(keys[:2])==2
    assert list(policy._lru.evictable_blocks)==[keys[2],keys[1],keys[0]]
    assert manager.relevance_ema['mean']==before
    assert not policy._chain_successor and not policy._last_touch_req_id
    assert all(policy.get(k).ref_cnt==0 for k in keys)
    policy.mark_non_evictable(keys[0]);policy.get(keys[0]).ref_cnt=1
    assert manager.refresh_wait_hint_cpu_recency([keys[0],to_key(9)])==0


def test_compact_native_receipt_preserves_metrics_without_claiming_full_proof():
    from semantic_offload.connector import _compact_benchmark_preservation
    raw=dict(authoritative=True,first_admission_prompt_source={'num_computed_tokens':3},
        server_timing={'ordering_valid':True},prior_evicted_keys=['key'],
        transfer_job_statuses=[{'large':'history'}])
    result=_compact_benchmark_preservation(raw)
    assert result['first_admission_prompt_source']==raw['first_admission_prompt_source']
    assert 'full_byte_and_lifetime_proof_not_retained' in result['evidence_scope']
    assert 'transfer_job_statuses' not in result and 'transfer_job_statuses' in raw


def test_complete_agent_workflow_executes_tools_and_preserves_generated_history(tmp_path):
    from benchmarks.serving_suite.repo_workload import run_wave
    from benchmarks.serving_suite.agent_report import diagnose

    class Tokenizer:
        histories=[]
        def apply_chat_template(self,messages,**kwargs):
            self.histories.append([dict(m) for m in messages])
            return list(range(80+len(messages)))
        def decode(self,tokens,**kwargs):
            return 'test_status=PASS; test_status=PASS'

    async def scenario():
        sent=[]
        async def generate(request):
            sent.append(await request.json())
            data={'choices':[{'token_ids':[10,11], 'text':'answer', 'finish_reason':'stop'}]}
            return web.Response(text='data: '+json.dumps(data)+'\n\ndata: [DONE]\n\n',content_type='text/event-stream')
        async def hint(request):
            raw=await request.json()
            return web.json_response(dict(schema='semantic-known-prefix-wait-hint-ack-v1',event_id=raw['event_id'],status='no_action',reason='test'))
        app=web.Application();app.router.add_post('/v1/completions',generate);app.router.add_post('/plugins/semantic-offload/wait-hints',hint)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        session=dict(id='task', context='actual source', repository=str(tmp_path), turns=['one','two','three','four'],
            tools=[dict(kind=k,argv=[sys.executable,'-c','print("actual subprocess result")'],valid_exit_codes=[0]) for k in ('search','test','evidence')])
        plan=dict(concurrency=1,request_timeout_s=5,max_model_len=512,max_tokens=20,natural_eos=False,
                  seed=1,publish_hints=True,hint_refresh_s=.02,turns=4,waves=1)
        tok=Tokenizer();rows=[];hints=[]
        def record(row):
            row['generated_text']=tok.decode(row['output_token_ids'])
            rows.append(row)
        try:
            await run_wave(f'http://127.0.0.1:{port}',plan,[session],tok,0,record,'candidate',{},hints,tmp_path/'streams')
        finally:
            await runner.cleanup()
        assert len(rows)==4 and all(r['success'] for r in rows)
        assert len(tok.histories[-1])==8
        assert 'actual subprocess result' in tok.histories[-1][-1]['content']
        assert tok.histories[-1][-2]['content']=='test_status=PASS; test_status=PASS'
        assert all(p['request_id'].startswith('sc|suite|case|') for p in sent)
        events=[json.loads(line) for line in (tmp_path/'workflow-0.jsonl').read_text().splitlines()]
        assert sum(e['event']=='tool' for e in events)==3
        assert sum(e['event']=='session_end' for e in events)==1
        assert hints and all('future_suffix' not in h['hint'] for h in hints)
        (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        native=[dict(external_request_id=r['request_id'], actual_prompt_tokens_locally_executed=10,
                     loaded_tokens=0, prior_evicted_keys=[], authoritative=True) for r in rows]
        (tmp_path/'server.log').write_text(''.join('SEMANTIC_PRESERVATION_EVENT '+json.dumps(e)+'\n' for e in native))
        (tmp_path/'resources.jsonl').write_text(json.dumps(dict(monotonic_s=0,gauges={}))+'\n')
        (tmp_path/'kv-qualification.jsonl').write_text(json.dumps(dict(event='step',running=1,waiting_count=0,reclaimable=10,total_blocks=20))+'\n')
        case=dict(source_ids=['task'],hints=hints,load_duration_s=1,metrics={
            'vllm:request_prefill_kv_computed_tokens_sum':40,
            'vllm:external_prefix_cache_hits_total':0,
            'vllm:prefix_cache_queries_total':sum(len(r['payload']['prompt']) for r in rows),
            'vllm:request_prefill_kv_computed_tokens_count':4})
        report=diagnose(tmp_path,case,plan)
        assert not report['measurement_errors']
        assert not report['opportunity']['cache_competition_observed']
        assert report['quality']['passed']==1 and report['quality']['strict_format_failures']==1
        case['metrics']['vllm:request_prefill_kv_computed_tokens_sum']=41
        assert 'native_endpoint_accounting_mismatch' in diagnose(tmp_path,case,plan)['measurement_errors']
        # Initial-source counters and cumulative execution differ after a
        # preemption. Compare the endpoint only to its actual source definition.
        case['metrics'].update({'vllm:request_prefill_kv_computed_tokens_sum':40,
            'vllm:num_preemptions_total':2,'vllm:kv_offload_load_bytes_total':123})
        plan.update(client_concurrency=0,prompt_source_accounting_version=3)
        for i,e in enumerate(native):
            e.update(actual_prompt_tokens_locally_executed=20,loaded_tokens=99,
                preemption_count=2 if i==0 else 0,
                first_admission_prompt_source=dict(num_prompt_tokens=len(rows[i]['payload']['prompt']),
                    num_computed_tokens=10,num_cached_tokens=len(rows[i]['payload']['prompt'])-10,
                    num_local_cached_tokens=len(rows[i]['payload']['prompt'])-10,num_external_cached_tokens=0))
        (tmp_path/'server.log').write_text(''.join('SEMANTIC_PRESERVATION_EVENT '+json.dumps(e)+'\n' for e in native))
        report=diagnose(tmp_path,case,plan)
        assert not report['measurement_errors']
        assert report['cache']['load_attribution_gap_bytes']==123
        assert report['cache']['locally_executed_prefill_tokens']==80
        assert report['consistency']['computed_prefill']['native']==40
        native[0]['first_admission_prompt_source']['num_computed_tokens']+=1
        (tmp_path/'server.log').write_text(''.join('SEMANTIC_PRESERVATION_EVENT '+json.dumps(e)+'\n' for e in native))
        assert 'native_endpoint_accounting_mismatch' in diagnose(tmp_path,case,plan)['measurement_errors']
    asyncio.run(scenario())
