"""Final pooled groups, correctness audit, and selected profile evidence."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import analyze as a

H = Path(__file__).resolve().parent
ROOT = H.parents[1]

def write(name, value):
    (H / name).write_text(json.dumps(value, indent=2) + '\n')

def raw(name):
    return json.loads(gzip.decompress((H/'runs'/name/'raw.json.gz').read_bytes()))

def output_compare(left, right):
    ls = {r['index']: r for r in left['requests']}
    rs = {r['index']: r for r in right['requests']}
    lk = {t['external_request_id']: t for t in left['terminals'] if t.get('external_request_id')}
    rk = {t['external_request_id']: t for t in right['terminals'] if t.get('external_request_id')}
    result = {'left': left['name'], 'right': right['name'], 'request_sets_equal': ls.keys() == rs.keys(), 'requests': []}
    for i in sorted(ls.keys() & rs.keys()):
        l, r = ls[i], rs[i]
        lo, ro = l.get('output_token_ids', []), r.get('output_token_ids', [])
        lt, rt = lk.get(l['payload']['request_id'], {}), rk.get(r['payload']['request_id'], {})
        key = lambda t: [(p['group_idx'], p['chunk_index'], p['key']) for p in t.get('ordered_key_positions', [])]
        diff = [n for n, (x, y) in enumerate(zip(lo, ro)) if x != y]
        result['requests'].append({'index': i, 'role': l['role'], 'prompt_equal': l['payload']['prompt'] == r['payload']['prompt'], 'output_equal': lo == ro, 'output_lengths': [len(lo), len(ro)], 'different_positions': diff, 'first_difference': diff[0] if diff else None, 'ordered_key_chains_equal': key(lt) == key(rt), 'native_gpu_hit_tokens': [lt.get('admitted_local_tokens'), rt.get('admitted_local_tokens')], 'external_reused_tokens': [lt.get('admitted_external_tokens'), rt.get('admitted_external_tokens')], 'actual_prompt_tokens_locally_executed': [lt.get('actual_prompt_tokens_locally_executed'), rt.get('actual_prompt_tokens_locally_executed')]})
    for field in ['prompt_equal', 'output_equal', 'ordered_key_chains_equal']:
        result[field + '_count'] = sum(r[field] for r in result['requests'])
    return result

def main():
    m = json.loads((H/'measurements.json').read_text())
    summaries = [s for s in m['summaries'] if s['phase'] in ['main', 'stress', 'diagnostic', 'repeatability']]
    results = {s['name']: a.analyze(H/'runs'/s['name']/'raw.json.gz') for s in summaries}
    pooled = []
    for phase in ['main', 'stress', 'diagnostic']:
        for workload in ['single', 'multi']:
            for policy in ['arc', 'semantic']:
                ss = [s for s in summaries if (s['phase'], s['workload'], s['policy']) == (phase, workload, policy)]
                rows = [r for s in ss for r in results[s['name']][2]]
                elapsed = sum(s['all']['elapsed_s'] for s in ss)
                total = lambda key: sum((s.get(key) or 0) for s in ss)
                rr = [s['resources'] for s in ss]
                resource_total = lambda key: sum((r.get(key) or 0) for r in rr)
                counts = collections.Counter()
                for s in ss: counts.update(s['manager_counts'])
                by_role = {role: a.aggregate([r for r in rows if r['role'].startswith('background') == (role == 'background')], elapsed) for role in ['agent', 'background']}
                sessions = [v['last_completion_offset_ms'] for s in ss for k, v in s['sessions'].items() if k.startswith('agent')]
                pooled.append({'phase': phase, 'workload': workload, 'policy': policy, 'runs': len(ss), 'all': a.aggregate(rows, elapsed), 'groups': by_role, 'agent_session_completion_ms': a.distribution(sessions), 'manager_counts': dict(counts), 'traffic': {k: a.safe_sum(ss,k) for k in ['demand_h2d_bytes', 'd2h_bytes', 'demand_h2d_worker_s', 'd2h_worker_s', 'prefetch_consumed_bytes', 'prefetch_unused_completed_bytes', 'prefetch_waste_bytes', 'splice_bytes']}, 'activation': {k: total(k) for k in ['worker_capture_calls', 'worker_capture_rows', 'staging_jobs_submitted', 'staging_jobs_ready', 'nonempty_nominations', 'scored_prefetch_candidates', 'constrained_prefetch_nominations', 'prefetch_consumed_blocks', 'preemptions', 'prefetch_allocation_failures']}, 'resources': {'sampled_cpu_s': resource_total('cpu_s_sampled'), 'sampled_energy_j': resource_total('device_energy_j_sampled'), 'sampled_energy_j_per_output_token': resource_total('device_energy_j_sampled') / a.aggregate(rows, elapsed)['output_tokens'], 'sampled_rss_gib_seconds': resource_total('rss_byte_seconds_sampled')/2**30, 'sampled_rss_peak_gib': max(r['rss_peak_bytes_sampled'] for r in rr)/2**30, 'sampled_gpu_utilization_pct_weighted': sum(r['device_gpu_utilization_mean_pct']*s['all']['elapsed_s'] for s,r in zip(ss,rr))/elapsed, 'sampled_device_memory_peak_gib': max(r['device_memory_used_peak_bytes_sampled'] for r in rr)/2**30, 'load_cuda_peak_allocated_mib': max(s['load_gpu_peak_allocated_bytes'] for s in ss)/2**20, 'load_cuda_peak_reserved_mib': max(s['load_gpu_peak_reserved_bytes'] for s in ss)/2**20, 'sampled_temperature_max_c': max(r['device_temperature_max_c'] for r in rr), 'sampled_sm_clock_mhz_weighted': sum(r['device_sm_clock_mean_mhz']*s['all']['elapsed_s'] for s,r in zip(ss,rr))/elapsed}})
    write('pooled-detail.json', pooled)
    repeat = raw('repeatability-single-2207-arc')
    repeat_results = [output_compare(raw('main-single-2207-arc'), repeat), output_compare(raw('main-single-2207-semantic'), repeat), output_compare(raw('main-single-2207-arc'), raw('main-single-2207-semantic'))]
    write('repeatability.json', repeat_results)
    profiles = []
    for s in summaries:
        if s['phase'] != 'diagnostic': continue
        r = results[s['name']][0]
        w = r['postload_export']['workers'][0]
        profiles.append({'name': s['name'], 'load_elapsed_s': s['all']['elapsed_s'], 'method_times': w['method_times'], 'capture': w['capture'], 'pending_profile_cuda': w['pending_profile_cuda'], 'component_exports': [json.loads(p.read_text()) for p in (H/'runs'/s['name']/'components').glob('*.json')], 'note': 'Inclusive nested times overlap. TP1 worker and scheduler can share a process. Sampled CUDA stream durations are not exclusive kernel time or whole-load sums.'})
    write('profile-detail.json', profiles)
    failed = [a.analyze(p) for p in sorted((H/'failed-attempts').glob('*/raw.json.gz'))]
    write('failed-attempt-summary.json', [r[1] for r in failed])
    frozen = json.loads((H/'frozen.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source_changes = [p for p,h in frozen['source_pins'].items() if sha(ROOT/p) != h]
    task_changes = [p for p,h in frozen['task_pins'].items() if sha(H/p) != h]
    prep_changes = [p for p,h in frozen['prepared_pins'].items() if sha(Path(p) if Path(p).is_absolute() else H/p) != h]
    checks = {'expected_successful_cells': len(summaries) == 21, 'source_pins_unchanged': not source_changes, 'measured_runtime_unchanged': not any(p != 'analyze.py' for p in task_changes), 'prepared_inputs_unchanged': not prep_changes}
    audit = []
    for s in summaries:
        r = results[s['name']][0]; rows = results[s['name']][2]
        expected = 18 if s['workload'] == 'single' else 45
        before = r['before_load_export']['workers'][0]
        after = r['postload_export']['workers'][0]
        rr = {'name': s['name'], 'status_complete': s['status'] == 'complete', 'request_count': len(rows) == expected, 'all_requests_successful': s['all']['failures'] == 0, 'fixed_output_budget': s['all']['output_tokens'] == (1536 if expected == 18 else 3264), 'terminals_authoritative': s['terminal_events_authoritative'] == expected, 'ordered_keys_complete': s['ordered_key_complete_requests'] == expected, 'physical_identity_complete': s['physical_identity_complete_requests'] == expected, 'physical_table_complete': s['block_table_complete_requests'] == expected, 'no_transfer_failures': s['transfer_failed_jobs'] == s['speculative_transfer_failed_jobs'] == 0, 'no_owned_descendants': s['owned_descendants_remaining'] == [], 'no_pending_cuda_or_async': after['pending_profile_cuda'] == after['pending_async_observations'] == 0, 'equal_geometry': r['geometry'] == json.loads((H/'prepared'/f"{s['workload']}-{s['seed']}.json").read_text())['geometry'], 'real_policy': s['policy_type'] == ('SemanticPolicy' if s['policy'] == 'semantic' else 'ARCCachePolicy'), 'symmetric_threshold': s['threshold'] == (0 if s['phase'] == 'stress' else 2), 'warmup_capture_calls': before['capture'].get('calls',0), 'load_capture_calls': after['capture'].get('calls',0) - before['capture'].get('calls',0)}
        audit.append(rr)
    checks['all_cell_checks_pass'] = all(v for r in audit for k,v in r.items() if isinstance(v,bool))
    validation = {'checks': checks, 'source_changed': source_changes, 'task_changed': task_changes, 'prepared_changed': prep_changes, 'cells': audit, 'total_successful_requests': sum(s['all']['completed'] for s in summaries), 'main_requests': sum(s['all']['completed'] for s in summaries if s['phase']=='main'), 'main_output_tokens': sum(s['all']['output_tokens'] for s in summaries if s['phase']=='main'), 'retained_failures': [{'path': 'failed-attempts/'+r[1]['name'], 'submitted_requests': r[1]['all']['requests'], 'successful': r[1]['all']['completed'], 'failed': r[1]['all']['failures'], 'status': r[1]['status'], 'error': r[1]['error']} for r in failed], 'correctness_exception': 'See repeatability.json for the primary single seed2207 output differences and the baseline repeat.'}
    write('VALIDATION.json', validation)
    print(json.dumps({'checks': checks, 'task_changed': task_changes, 'repeat_comparisons': [{k:v for k,v in r.items() if k!='requests'} for r in repeat_results]}, indent=2))
    assert all(checks.values()), validation

if __name__ == '__main__': main()
