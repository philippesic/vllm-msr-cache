"""Check the two actual-checkout verification runs; excluded from inference."""
import fcntl, gzip, hashlib, json
from pathlib import Path
from teacher_busy_frontier import check_pins
from work_inventory import compare

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
with open('/tmp/vllm-semantic-cache-gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((H / 'CPU-RECENCY-INTEGRATED.json').read_text())
    check_pins(plan)
    progress = json.loads((H / plan['progress_file']).read_text())
    assert progress['status'] == 'complete' and progress['active'] is None
    assert len(progress['completed']) == 2
    binding = json.loads((H / 'runs/cpu-recency-v3-batched-byte-qualified-c24-b2048-candidate-s1103/qualification-binding.json').read_text())
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'semantic_offload').glob('*.py')}
    assert pins == binding['source_pins']
    rows = []
    for run, completed in zip(plan['runs'], progress['completed']):
        assert completed['name'] == run['name']
        assert all(completed[k] == 72 for k in ['planned_requests', 'submitted_requests', 'completed_requests', 'completed_strict_proofs'])
        assert completed['fixed_work_integrity']['passed']
        raw = json.load(gzip.open(H / 'runs' / run['name'] / 'raw.json.gz'))
        summary = json.loads((H / 'runs' / run['name'] / 'summary.json').read_text())
        assert raw['status'] == 'complete' and raw['performance_eligible'] and summary['valid']
        assert raw['source_pins'] == pins and not raw['owned_descendants_remaining']
        for key in ['namespace', 'model_identity', 'runtime_pins', 'harness_pins', 'corpus_identity']:
            if key in binding:
                assert raw[key] == binding[key], key
        assert raw['geometry']['gpu_blocks'] == run['gpu_blocks']
        assert all(raw['geometry'][k] == v for k, v in binding['geometry'].items() if k not in ['gpu_blocks', 'gpu_kv_bytes'])
        if run['arm'] == 'candidate':
            assert raw['config'] == binding['config'] and raw['geometry'] == binding['geometry']
            responses = [h.get('response', {}) for h in raw['hints']]
            refreshed = [r for r in responses if r.get('reason') == 'cpu_prefix_recency_refreshed']
            assert refreshed and all(r['reserved_gpu_blocks'] == r['speculative_transfer_bytes'] == 0 for r in refreshed)
        else:
            assert not raw['config']['kv_connector_extra_config'].get('wait_hint_cpu_retention_only', False)
        assert not raw['speculative_transfer_completions']
        rows.append({'name': run['name'], 'arm': run['arm'], 'strict_requests': 72,
                     'source_config_geometry_matches': True,
                     'runtime_resolution': raw['runtime_resolution'],
                     'no_owned_descendants_remaining': True,
                     'no_speculative_gpu_transfers': True})
        del raw
    work = compare(plan['runs'][0]['name'], plan['runs'][1]['name'])
    assert work['requests'] == 72 and work['identical_input_token_ids'] and work['identical_output_token_ids']
    (H / 'integration/work-equivalence.json').write_text(json.dumps(work, indent=2) + '\n')
    result = {'passed': True, 'strict_requests': 144, 'exact_input_and_output_ids': True,
              'all_checkout_python_source_hashes_match_confirmed_and_byte_qualified_source': True,
              'source_file_count': len(pins), 'rows': rows,
              'import_provenance': 'Frozen integrated runner uses source=ROOT and child PYTHONPATH=hooks-teacher-replay:ROOT:ROOT; hook PathFinder resolves semantic_offload normally. CPU runtime verification asserts actual checkout module __file__. GPU logs record runtime geometry/configuration, not module __file__; do not infer direct GPU module-path telemetry.',
              'excluded_from_twelve_pair_inference': True}
    (H / 'integration/verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k not in ['rows', 'import_provenance']})
