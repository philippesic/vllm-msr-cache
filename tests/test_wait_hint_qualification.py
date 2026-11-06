"""Protect diagnostic replay and inactive-hint controls against false attribution."""
from types import SimpleNamespace
import pytest
import torch
from benchmarks.serving_suite.replay_logits import FixedTokenReplay
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from tests.test_wait_hint_cpu_recency import retention


def test_inactive_hints_preserve_arc_order_and_ownership(retention):
    scheduler,manager,pool,keys,hint=retention
    scheduler._benchmark_wait_hint_noop=True
    before=list(manager._policy.t1)
    result=scheduler.publish_wait_hint(hint)
    assert result['ready_cpu_prefix_keys']==0
    assert list(manager._policy.t1)==before
    assert not manager._policy.t2 and not scheduler._jobs
    assert result['reserved_gpu_blocks']==result['speculative_transfer_bytes']==0


def test_replay_tracks_output_positions_and_batch_moves():
    processor=FixedTokenReplay(None,None,False)
    output=[]
    params=SimpleNamespace(extra_args={'diagnostic_replay_tokens':[2,1]})
    processor.update_state(BatchUpdate(2,[],[(0,params,[8],output)],[]))
    assert processor.apply(torch.zeros(2,5))[0].argmax().item()==2
    output.append(2)
    processor.update_state(BatchUpdate(2,[],[],[(0,1,MoveDirectionality.UNIDIRECTIONAL)]))
    logits=processor.apply(torch.zeros(2,5))
    assert logits[1].argmax().item()==1
    assert torch.isfinite(logits[0]).all()
    processor.update_state(BatchUpdate(0,[1],[],[]))
    assert not processor.requests


@pytest.mark.parametrize('tokens',[[],[True],[-1],['3']])
def test_replay_rejects_invalid_sequences(tokens):
    with pytest.raises(ValueError):
        FixedTokenReplay.validate_params(SimpleNamespace(extra_args={'diagnostic_replay_tokens':tokens}))


def test_replay_cannot_generate_past_frozen_sequence():
    processor=FixedTokenReplay(None,None,False)
    processor.requests[0]=([1],[1])
    with pytest.raises(RuntimeError,match='beyond frozen'):
        processor.apply(torch.zeros(1,5))


def test_document_reference_rejects_a_known_wrong_status(tmp_path):
    import json
    import subprocess
    import sys
    from benchmarks.research_checks.qualification_inputs import parse_document
    doc=tmp_path/'pep.rst'; doc.write_text('PEP: 1\nTitle: Example\nStatus: Active\nType: Process\nCreated: 13-Jun-2000\n\nActual document body.\n')
    actual=parse_document(doc)
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps({'1':{k:actual[k] for k in ('PEP','Title','Status','Type','Created')}}))
    cmd=[sys.executable,'-m','benchmarks.research_checks.qualification_inputs','document','--file',str(doc),'--reference',str(reference),'--action','validate']
    assert subprocess.run(cmd,capture_output=True).returncode==0
    wrong=json.loads(reference.read_text());wrong['1']['Status']='Final'
    reference.write_text(json.dumps(wrong))
    rejected=subprocess.run(cmd,capture_output=True)
    assert rejected.returncode!=0 and b'differs from frozen reference' in rejected.stderr


def test_inactive_trace_reports_unchanged_queue(retention,capsys):
    import json
    scheduler,manager,pool,keys,hint=retention
    scheduler._benchmark_wait_hint_noop=True
    scheduler._benchmark_wait_hint_trace=True
    scheduler.publish_wait_hint(hint)
    record=json.loads(capsys.readouterr().out.split('WAIT_HINT_RETENTION_TRACE ')[1].splitlines()[0])
    assert record['before']==record['after'] and record['refreshed']==0


def test_lifecycle_link_requires_no_intervening_eviction(tmp_path):
    import json
    from benchmarks.research_checks.qualification_trace import analyze
    key=b'example';encoded=repr(key)
    trace=dict(event_id='hint',monotonic_ns=20,prefix_keys=[encoded],before={encoded:{'t1':0}},after={encoded:{'t1':4}})
    record=dict(external_request_id='return',key_lifecycle=dict(cpu_commits=[dict(key=key.hex(),committed_at_ns=10)],cpu_evictions=[]),
                transfer_job_statuses=[dict(direction='load',success=True,status='success',cancelled=False,is_prefetch=False,
                    job_id=1,keys=[key.hex()],started_at_ns=30,completed_at_ns=40)])
    def save():
        (tmp_path/'server.log').write_text('WAIT_HINT_RETENTION_TRACE '+json.dumps(trace)+'\nSEMANTIC_PRESERVATION_EVENT '+json.dumps(record)+'\n')
    save()
    assert analyze(tmp_path)['distinct_moved_survived_and_demand_loaded_keys']==1
    record['key_lifecycle']['cpu_evictions']=[dict(key=key.hex(),observed_at_ns=25)]
    save()
    assert analyze(tmp_path)['distinct_moved_survived_and_demand_loaded_keys']==0
    record['key_lifecycle']['cpu_evictions']=[]
    record['key_lifecycle']['cpu_commits'].append(dict(key=key.hex(),committed_at_ns=25))
    save()
    assert analyze(tmp_path)['distinct_moved_survived_and_demand_loaded_keys']==0


def test_upstream_off_control_requires_actual_metadata_omission(tmp_path,monkeypatch):
    import json
    from benchmarks.serving_suite.published import diagnose_external
    from benchmarks.serving_suite import agent_report
    monkeypatch.setattr(agent_report,'diagnose',lambda *a,**kw:
        dict(measurement_errors=[],telemetry={},limitations=[]))
    (tmp_path/'metrics-before.prom').write_text('suite_completed_requests_total 0\n')
    (tmp_path/'metrics-after.prom').write_text('suite_completed_requests_total 1\n')
    rows=tmp_path/'requests.jsonl'
    rows.write_text(json.dumps(dict(payload=dict(prompt=[1,2])))+'\n')
    case=dict(variant='upstream-off',expected_requests=1)
    plan=dict(published_retention_off=True,published_variant='upstream',repeats=3)
    assert diagnose_external(tmp_path,case,plan,write=False)['policy_activation']['verified']
    rows.write_text(json.dumps(dict(payload=dict(kv_hints={'actions':['retain']})))+'\n')
    assert not diagnose_external(tmp_path,case,plan,write=False)['policy_activation']['verified']


def test_mha_geometry_uses_attention_heads_when_no_separate_kv_heads(tmp_path):
    import json
    from benchmarks.serving_suite.published import verify_geometry
    (tmp_path/'config.json').write_text(json.dumps(dict(hidden_size=768,
        num_attention_heads=12,num_hidden_layers=12)))
    block_bytes=12*12*64*2*2*16
    (tmp_path/'runtime-observation.jsonl').write_text(json.dumps(dict(
        event='native_worker_geometry',cpu_bytes=298*block_bytes,
        gpu_bytes=149*block_bytes,pinned=True,blocks_per_chunk=1,device_aliases=0))+'\n')
    geometry=verify_geometry(tmp_path,dict(model=str(tmp_path),gpu_blocks=149,
        cpu_bytes=298*block_bytes),'wait')
    assert geometry['gpu_kv_bytes']==87883776



@pytest.mark.parametrize('mutation', ['http500', 'http400', 'schema', 'event', 'status', 'reason', 'ready', 'body'])
def test_protocol_rejection_cannot_be_fabricated_by_server_errors(mutation):
    from benchmarks.research_checks.wait_hint_runtime_probe import ACK_SCHEMA, valid_rejection
    response = dict(http_status=200, body=dict(schema=ACK_SCHEMA, event_id='event',
        status='rejected', reason='namespace_mismatch', ready_cpu_prefix_keys=0))
    assert valid_rejection(response, 'event', 'namespace_mismatch')
    if mutation.startswith('http'):
        response['http_status'] = int(mutation[4:])
    elif mutation == 'body':
        response['body'] = None
    else:
        key = {'event':'event_id', 'ready':'ready_cpu_prefix_keys'}.get(mutation, mutation)
        response['body'][key] = 1 if mutation == 'ready' else 'unrelated'
    assert not valid_rejection(response, 'event', 'namespace_mismatch')


def test_stale_advice_requires_its_specific_causal_rejection():
    from benchmarks.research_checks.wait_hint_runtime_probe import ACK_SCHEMA, valid_rejection
    response = dict(http_status=200, body=dict(schema=ACK_SCHEMA, event_id='event',
        status='rejected', reason='namespace_mismatch'))
    assert not valid_rejection(response, 'event', 'causal_clock_mismatch')
    response['body']['reason'] = 'causal_clock_mismatch'
    assert valid_rejection(response, 'event', 'causal_clock_mismatch')


def test_duplicate_requires_success_and_the_original_acknowledgement():
    from benchmarks.research_checks.wait_hint_runtime_probe import ACK_SCHEMA, valid_duplicate
    original = dict(schema=ACK_SCHEMA, event_id='event', status='no_action',
                    reason='cpu_prefix_recency_refreshed', ready_cpu_prefix_keys=4)
    response = dict(http_status=200, body={**original, 'idempotent_replay': True})
    assert valid_duplicate(response, 'event', original)
    response['http_status'] = 500
    assert not valid_duplicate(response, 'event', original)
    response['http_status'] = 200
    response['body']['ready_cpu_prefix_keys'] = 9
    assert not valid_duplicate(response, 'event', original)


def test_future_information_requires_the_pinned_validation_reason():
    from benchmarks.research_checks.wait_hint_runtime_probe import FUTURE_FIELD_DETAIL, valid_future_rejection
    response = dict(http_status=400, body=dict(detail=FUTURE_FIELD_DETAIL))
    assert valid_future_rejection(response)
    response['http_status'] = 500
    assert not valid_future_rejection(response)
    response.update(http_status=400, body=dict(detail='unrelated bad request'))
    assert not valid_future_rejection(response)


@pytest.mark.parametrize('mutation', [None, 'message', 'type', 'code', 'status'])
def test_future_validation_recognizes_only_the_exact_vllm_error(mutation):
    from benchmarks.research_checks.wait_hint_runtime_probe import FUTURE_FIELD_DETAIL, valid_future_rejection
    response = dict(http_status=400, body={'error': dict(message=FUTURE_FIELD_DETAIL,
        type='Bad Request', param=None, code=400)})
    if mutation == 'status':
        response['http_status'] = 500
    elif mutation:
        response['body']['error'][mutation] = 'unrelated'
    assert valid_future_rejection(response) is (mutation is None)


def test_probe_uses_actual_eager_directional_load_counter():
    from benchmarks.research_checks.wait_hint_runtime_probe import scoped_metric_deltas
    header = '# HELP vllm:kv_offload_load_bytes_total Total bytes loaded.\n'
    before = header + 'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 0\n'
    after = header + 'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 917504\n'
    values, sources = scoped_metric_deltas(before, after)
    name = 'vllm:kv_offload_load_bytes_total'
    assert values[name] == 917504
    assert sources[name]['before'] == sources[name]['after'] == 'eager_directional_compat_counter'


@pytest.mark.parametrize('before,after', [('', ''), (
    'vllm:kv_offload_load_bytes_total 100\n', 'vllm:kv_offload_load_bytes_total 40\n')])
def test_probe_never_fabricates_zero_from_missing_or_reset_load_counter(before, after):
    from benchmarks.research_checks.wait_hint_runtime_probe import scoped_metric_deltas
    values, _ = scoped_metric_deltas(before, after)
    assert values['vllm:kv_offload_load_bytes_total'] is None


def test_probe_prefers_modern_counter_without_double_counting_legacy():
    from benchmarks.research_checks.wait_hint_runtime_probe import scoped_metric_deltas
    before = 'vllm:kv_offload_load_bytes_total 4\nvllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 100\n'
    after = 'vllm:kv_offload_load_bytes_total 10\nvllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 200\n'
    values, _ = scoped_metric_deltas(before, after)
    assert values['vllm:kv_offload_load_bytes_total'] == 6
