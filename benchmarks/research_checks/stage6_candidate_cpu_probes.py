"""GPU-free behavioral probes for the selected wait-hint candidate's contracts."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / 'evidence/stage6-5080-20261109/candidate-audit-01'


def brief(result):
    return {k: result[k] for k in ('status', 'reason', 'ready_cpu_prefix_keys', 'idempotent_replay',
        'reserved_gpu_blocks', 'speculative_transfer_bytes') if k in result}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    target = DEST / 'cpu-probes.json'
    assert not target.exists()
    from semantic_offload.wait_hint import KnownPrefixWaitHint
    from tests.test_wait_hint import _hint
    from tests.test_wait_hint_cpu_recency import retention
    from tests.test_preservation_audit_mode import native, source
    from tests.test_step_1_5_prefetch import _insert_resident, _wait_hint
    from vllm.v1.request import RequestStatus
    from vllm.v1.core.sched.output import SchedulerOutput

    estimates = []
    for value in (0, 50, 250, 750, 1000, 2000, None):
        raw = _hint()
        raw['declared_wait_ms'] = value
        if value is None:
            raw['wait_provenance'] = None
        try:
            KnownPrefixWaitHint.from_mapping(raw)
            estimates.append(dict(estimate_ms=value, accepted=True))
        except ValueError as exc:
            estimates.append(dict(estimate_ms=value, accepted=False, reason=str(exc)))

    scheduler, manager = native(observed_session_registry_size=2)
    capacity = []
    for ordinal in range(3):
        name = f'finished-session-{ordinal}'
        request, prefix, _ = source(scheduler, name, ordinal=ordinal)
        scheduler._register_observed_prefix(request)
        result = scheduler.publish_wait_hint(_wait_hint(scheduler, prefix, 32,
            session_id=name, event_id=f'{ordinal + 1:064x}'))
        request.status = RequestStatus.FINISHED_STOPPED
        scheduler.request_finished(request)
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {request.request_id}
        scheduler.build_connector_meta(output)
        capacity.append(dict(session=ordinal, result=brief(result), active_requests_after_finish=len(scheduler._req_status),
                             lifetime_session_tombstones=len(scheduler._observed_prefix_generations)))
    assert capacity[-1]['result']['reason'] == 'session_registry_capacity'
    assert capacity[-1]['active_requests_after_finish'] == 0

    scheduler, manager = native(wait_hint_native_adoption=True)
    request, prefix, keys = source(scheduler)
    pool = scheduler.gpu_block_pool
    blocks = pool.get_new_blocks(1)
    pool.cache_full_blocks(request, blocks, 0, 1, 16, 0)
    pool.free_blocks(blocks)
    _insert_resident(manager, keys[1], 0)
    scheduler._register_observed_prefix(request)
    hint = _wait_hint(scheduler, prefix, 32, session_id='one')
    ordinary = scheduler.publish_wait_hint(hint)
    assert ordinary['ready_cpu_prefix_keys'] == 0
    scheduler._wait_hint_cpu_retention_gpu_aware = True
    bridge_hint = dict(hint, event_id='b' * 64)
    bridge = scheduler.publish_wait_hint(bridge_hint)
    assert bridge['ready_cpu_prefix_keys'] == 1
    prefix_gap = dict(selected_default=brief(ordinary), optional_bridge_not_selected=brief(bridge),
        gpu_prefix_keys=bridge['local_gpu_prefix_keys'], cpu_suffix_keys=bridge['additional_ready_cpu_suffix_keys'],
        scope='Real native CPU manager/block-index metadata, no GPU allocation. Demonstrates a capability limit, not its frequency in recorded serving.')

    scheduler, manager, pool, keys, hint = retention.__wrapped__()
    first = scheduler.publish_wait_hint(hint)
    before = list(manager._policy.t1)
    conflicting = copy.deepcopy(hint)
    conflicting['namespace']['model_revision'] = 'wrong-model'
    cached = scheduler.publish_wait_hint(conflicting)
    assert cached['idempotent_replay'] is True and cached['reason'] == first['reason']
    assert list(manager._policy.t1) == before
    fresh = scheduler.publish_wait_hint(dict(conflicting, event_id='c' * 64))
    assert fresh['reason'] == 'namespace_mismatch'
    replay = dict(original=brief(first), same_event_id_changed_namespace=brief(cached),
        new_event_id_changed_namespace=brief(fresh), queue_unchanged_on_replay=True,
        scope='A conflicting retry can receive an old success-like acknowledgement before namespace validation; no new wrong-namespace cache operation occurs. Not observed in campaign-generated unique IDs.')

    scheduler, manager, pool, keys, hint = retention.__wrapped__()
    original_order = list(manager._policy.t1)
    first = scheduler.publish_wait_hint(hint)
    changed_order = list(manager._policy.t1)
    assert original_order != changed_order
    later = first['expires_at_ns'] + 1
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=later):
        expired = scheduler.publish_wait_hint(dict(hint, event_id='d' * 64))
    assert expired['reason'] == 'nomination_deadline_elapsed'
    assert list(manager._policy.t1) == changed_order
    expiry = dict(ack=brief(first), late_fresh_event=brief(expired), expired_input_rejected=True,
        previous_recency_change_not_undone=True, no_retention_lease=scheduler._wait_prefetch == {},
        scope='CPU-only advice changes ordinary ARC recency. Expiry rejects late inputs; it is not rollback or a lease releasing earlier recency changes.')
    assert not scheduler._jobs and not scheduler._wait_prefetch_jobs
    result = dict(schema='stage6-candidate-cpu-structural-probes-v1', gpu_visible_devices='', gpu_arrays_allocated=False,
        tested_candidate_files=['semantic_offload/connector.py', 'semantic_offload/manager.py', 'semantic_offload/wait_hint.py'],
        estimate_contract=estimates, completed_session_capacity=capacity, mixed_gpu_cpu_prefix=prefix_gap,
        conflicting_idempotency_key=replay, expired_advice_semantics=expiry,
        candidate_implementation_changed=False,
        source_sha256={name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in
            ['semantic_offload/connector.py', 'semantic_offload/manager.py', 'semantic_offload/wait_hint.py']},
        scope='CPU-only structural demonstrations using native metadata and established unit fixtures; no new performance measurement, GPU inference, model allocation or benchmark reclassification.')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def api_input_probe():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from semantic_offload.wait_hint_endpoint import SemanticWaitHintEndpointPlugin
    class Engine:
        async def publish_kv_connector_wait_hint(self, raw):
            raise AssertionError('Malformed input must not reach the scheduler')
    target = DEST / 'cpu-api-input-probe-02.json'
    assert not target.exists()
    app = FastAPI()
    app.state.semantic_wait_hint_engine_client = Engine()
    SemanticWaitHintEndpointPlugin().attach_router(app)
    cases = []
    with TestClient(app, raise_server_exceptions=False) as client:
        for value in ([], None, 7, 'invalid', {}):
            # Pass explicit JSON bytes so null remains a JSON body.
            response = client.post('/plugins/semantic-offload/wait-hints',
                content=json.dumps(value), headers={'content-type': 'application/json'})
            cases.append(dict(json_type=type(value).__name__, http_status=response.status_code))
    assert [c['http_status'] for c in cases] == [500, 500, 500, 500, 400]
    result = dict(schema='stage6-candidate-api-input-cpu-probe-v1', gpu_visible_devices='',
        cases=cases, in_process_ASGI_only=True, scheduler_not_called=True,
        interpretation='Non-object JSON causes an uncaught AttributeError andHTTP500 instead of a client error. Empty object is correctly rejected with400. No model crash, cache mutation or campaign observation is claimed.',
        candidate_implementation_changed=False)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    api_input_probe() if '--api-only' in sys.argv else main()
