"""Prepare and review guarded evidence repairs; --apply only at quiet Stage3 end."""
import argparse, ast, difflib, hashlib, json, time
from pathlib import Path

b = Path(__file__).resolve().parent
root = b.parents[1]

ALLOCATION = r'''


def allocation_receipt(worker, *, num_cpu_blocks, kv_bytes_per_chunk,
                       requested_cpu_bytes, blocks_per_chunk, model_config):
    """Observe allocated backing without allocating, transferring or changing it."""
    from functools import partial

    dtype = getattr(model_config, "dtype", None)
    result = dict(schema="semantic-offload-allocation-v1", available=False,
        num_cpu_blocks=num_cpu_blocks, kv_bytes_per_chunk=kv_bytes_per_chunk,
        requested_cpu_bytes=requested_cpu_bytes, blocks_per_chunk=blocks_per_chunk,
        model_dtype=str(dtype) if dtype is not None else None,
        model_quantization=getattr(model_config, "quantization", None),
        cpu_buffers=[])
    try:
        store, load = worker._store_handler, worker._load_handler
        cpu, returning = store.dst_tensors, load.src_tensors
        if not isinstance(cpu, (list, tuple)) or not isinstance(returning, (list, tuple)):
            raise TypeError("Unsupported native offloading buffer containers")
        def selected_callable(handler):
            function = getattr(handler, "_swap_blocks_batch", None)
            while isinstance(function, partial):
                function = function.func
            if function is None:
                return None
            return (getattr(function, "__module__", type(function).__module__)
                    + "." + getattr(function, "__qualname__", type(function).__qualname__))
        for index, tensor in enumerate(cpu):
            other = returning[index] if index < len(returning) else None
            shared = (len(cpu) == len(returning) and other is not None
                and tensor.device == other.device and tensor.dtype == other.dtype
                and tuple(tensor.shape) == tuple(other.shape)
                and tensor.data_ptr() == other.data_ptr())
            result["cpu_buffers"].append(dict(device=str(tensor.device),
                dtype=str(tensor.dtype), shape=list(tensor.shape),
                nbytes=tensor.numel()*tensor.element_size(),
                pinned=bool(tensor.is_pinned()), load_uses_same_storage=bool(shared)))
        result.update(available=bool(cpu), selected_store_callable=selected_callable(store),
                      selected_load_callable=selected_callable(load))
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        result.update(available=False, error=f"{type(exc).__name__}: {exc}")
    return result
'''

ACK_HELPERS = r'''
ACK_SCHEMA = 'semantic-known-prefix-wait-hint-ack-v1'
FUTURE_FIELD_DETAIL = 'wait hint contains forbidden future or diagnostic fields'


def valid_ack(response, event_id):
    body = response.get('body')
    return (response.get('http_status') == 200 and isinstance(body, dict)
            and body.get('schema') == ACK_SCHEMA and body.get('event_id') == event_id)


def valid_rejection(response, event_id, reason):
    return (valid_ack(response, event_id) and response['body'].get('status') == 'rejected'
            and response['body'].get('reason') == reason
            and response['body'].get('ready_cpu_prefix_keys', 0) == 0)


def valid_duplicate(response, event_id, original):
    return (valid_ack(response, event_id) and isinstance(original, dict)
            and response['body'].get('idempotent_replay') is True
            and response['body'] == {**original, 'idempotent_replay': True})


def valid_future_rejection(response):
    body = response.get('body')
    return (response.get('http_status') == 400 and isinstance(body, dict)
            and body.get('detail') == FUTURE_FIELD_DETAIL)

'''

METRIC_TESTS = r'''


def test_allocation_observation_reports_actual_unpinned_backing():
    import torch
    from semantic_offload.benchmark_metrics import allocation_receipt
    tensor = torch.empty((4, 8), dtype=torch.int8)
    worker = SimpleNamespace(_store_handler=SimpleNamespace(dst_tensors=[tensor]),
                             _load_handler=SimpleNamespace(src_tensors=[tensor]))
    observed = allocation_receipt(worker, num_cpu_blocks=4, kv_bytes_per_chunk=8,
        requested_cpu_bytes=35, blocks_per_chunk=1,
        model_config=SimpleNamespace(dtype=torch.float16, quantization=None))
    assert observed['available'] and observed['cpu_buffers'][0]['nbytes'] == 32
    assert observed['cpu_buffers'][0]['pinned'] is False
    assert observed['cpu_buffers'][0]['load_uses_same_storage'] is True


def test_allocation_observation_detects_different_load_backing():
    import torch
    from semantic_offload.benchmark_metrics import allocation_receipt
    left, right = torch.empty((4, 8), dtype=torch.int8), torch.empty((4, 8), dtype=torch.int8)
    worker = SimpleNamespace(_store_handler=SimpleNamespace(dst_tensors=[left]),
                             _load_handler=SimpleNamespace(src_tensors=[right]))
    observed = allocation_receipt(worker, num_cpu_blocks=4, kv_bytes_per_chunk=8,
        requested_cpu_bytes=32, blocks_per_chunk=1, model_config=SimpleNamespace())
    assert observed['available']
    assert observed['cpu_buffers'][0]['load_uses_same_storage'] is False


def test_allocation_observation_missing_buffers_is_unavailable():
    from semantic_offload.benchmark_metrics import allocation_receipt
    observed = allocation_receipt(SimpleNamespace(), num_cpu_blocks=4,
        kv_bytes_per_chunk=8, requested_cpu_bytes=32, blocks_per_chunk=1,
        model_config=SimpleNamespace())
    assert observed['available'] is False and observed['error']
    assert observed['cpu_buffers'] == []
'''

PROBE_TESTS = r'''


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
'''


def prepared_files():
    updates = {}
    def change(name, substitutions=(), append=''):
        before = (root/name).read_text()
        after = before
        for old, new in substitutions:
            if after.count(old) != 1:
                raise ValueError('Expected unique unchanged source context: '+name+' '+old[:70])
            after = after.replace(old, new, 1)
        after += append
        ast.parse(after, filename=name)
        updates[name] = (before, after)
    change('semantic_offload/benchmark_metrics.py', append=ALLOCATION)
    change('semantic_offload/spec.py', [
        ('from typing_extensions import override', 'import json\n\nfrom typing_extensions import override'),
        ('        return SemanticOffloadingWorker(\n', '        worker = SemanticOffloadingWorker(\n'),
        ('            kv_svd_trace=self.extra_config.get("kv_svd_trace"),\n        )\n\n    @override\n    def get_manager',
         '''            kv_svd_trace=self.extra_config.get("kv_svd_trace"),
        )
        if self.extra_config.get("benchmark_telemetry") is True:
            from semantic_offload.benchmark_metrics import allocation_receipt
            observed = allocation_receipt(worker, num_cpu_blocks=self.num_blocks,
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
                requested_cpu_bytes=self.extra_config.get("cpu_bytes_to_use"),
                blocks_per_chunk=spec_blocks_per_chunk(self),
                model_config=self.vllm_config.model_config)
            print("SEMANTIC_OFFLOAD_ALLOCATION " + json.dumps(observed), flush=True)
        return worker

    @override
    def get_manager''')])
    change('benchmarks/serving_suite/runner.py', [
        ('else ["-cc.mode=0", "-cc.cudagraph_mode=FULL_DECODE_ONLY"]',
         'else ["-cc.mode=0", "-cc.cudagraph_mode=FULL_DECODE_ONLY", "--cudagraph-metrics"]')])
    change('benchmarks/research_checks/wait_hint_runtime_probe.py', [
        ('\n\nasync def run_wave(', '\n\n'+ACK_HELPERS+'\nasync def run_wave('),
        ("            raw=last_hint['hint']", "            if last_hint.get('error'):\n                raise RuntimeError('Probe final hint transport failed')\n            raw=last_hint['hint']"),
        ("passed=bool(duplicate['body'].get('idempotent_replay'))", "passed=valid_duplicate(duplicate,raw['event_id'],last_hint.get('response'))"),
        ("passed=wrong_namespace['body'].get('reason')!='cpu_prefix_recency_refreshed'\n                               and not wrong_namespace['body'].get('ready_cpu_prefix_keys',0)",
         "passed=valid_rejection(wrong_namespace,bad['event_id'],'namespace_mismatch')"),
        ("passed=future['http_status'] in (400,422)", "passed=valid_future_rejection(future)"),
        ("passed=stale['body'].get('reason')!='cpu_prefix_recency_refreshed'\n                               and not stale['body'].get('ready_cpu_prefix_keys',0)",
         "passed=valid_rejection(stale,bad['event_id'],'causal_clock_mismatch')"),
        ("schema='wait-hint-runtime-probe-v2'", "schema='wait-hint-runtime-probe-v3'")])
    change('tests/test_benchmark_metrics.py', append=METRIC_TESTS)
    change('tests/test_wait_hint_qualification.py', append=PROBE_TESTS)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    updates = prepared_files()
    identities = {name:dict(before_sha256=hashlib.sha256(old.encode()).hexdigest(),
        after_sha256=hashlib.sha256(new.encode()).hexdigest()) for name,(old,new) in updates.items()}
    receipt = b/'stage4-repairs-prepared.json'
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.apply:
        expected = json.loads(receipt.read_text())
        if expected['files'] != identities or expected['script_sha256'] != script_hash:
            raise RuntimeError('Prepared source/script changed: review before applying')
        applied = b/'stage4-repairs-applied.json'
        if applied.exists():
            raise FileExistsError('Preserve prior repair attempt')
        for name, (old,new) in updates.items():
            destination = root/name
            if destination.read_text() != old:
                raise RuntimeError('Concurrent source edit: '+name)
            backup = b/'stage4-repair-before'/name
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_text(old)
            destination.write_text(new)
        applied.write_text(json.dumps(dict(applied_unix=time.time(), files=identities,
            scope='Evidence-only startup observation, graph counters and strict protocol assertions; policy/allocation unchanged. Apply only after all Stage3 campaigns end.'),indent=2)+'\n')
        print('Applied',len(updates),'guarded evidence repair files; tests still required')
    else:
        if receipt.exists():
            raise FileExistsError('Preserve prior preparation')
        patch = ''.join(''.join(difflib.unified_diff(old.splitlines(keepends=True),
            new.splitlines(keepends=True),fromfile='before/'+name,tofile='after/'+name))
            for name,(old,new) in updates.items())
        (b/'stage4-evidence-repairs.patch').write_text(patch)
        receipt.write_text(json.dumps(dict(prepared_unix=time.time(),script_sha256=script_hash,
            files=identities, ast_parsed=True, applied=False,
            scope='Prepared patch only. No watched source changes and no tests or actual runtime measurements implied.'),indent=2)+'\n')
        print('Prepared/AST parsed',len(updates),'files; watched sources unchanged')


if __name__ == '__main__':
    main()
