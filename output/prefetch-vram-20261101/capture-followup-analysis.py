"""Idle-only assessment of explicitly excluded quality/component diagnostics."""
import collections
import importlib.util
import json
from pathlib import Path

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_confirmation', H / 'confirm_results.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
A = C.helper()


def load(name, role):
    path = H / 'runs' / name / 'raw.json.gz'
    data = A.analyze(path)
    case = {'role': role, 'seed': data[0]['seed']}
    return {'data': data, 'record': C.run_record(case, path, data, A)}


quality = {'performance_eligible': False, 'diagnostic_only': True, 'seeds': []}
controls = json.loads((H / 'confirmation-long-results.json').read_text())['runs']
for seed in (1103, 2207, 3301):
    item = load(f'quality-prefix-r1-competing_long_dram-{seed}-semantic', 'quality_diagnostic')
    directory = Path(item['record']['raw_evidence']['path']).parent
    workers = sorted(directory.glob('prefix-policy-quality-worker-*.json'))
    schedulers = sorted(directory.glob('prefix-policy-quality-scheduler-*.json'))
    assert len(workers) == len(schedulers) == 1
    worker = json.loads(workers[0].read_text())
    scheduler = json.loads(schedulers[0].read_text())
    comparisons = []
    for baseline in controls:
        if baseline['seed'] == seed and baseline['role'] in ('arc_full', 'arc_small'):
            comparisons.append(C.exact_comparison(load(baseline['name'], baseline['role']), item, A))
    incomparable = [r for r in scheduler['records'] if not r['fully_comparable']]
    quality['seeds'].append({
        'seed': seed, 'run': item['record'],
        'worker_evidence': C.pin(workers[0]), 'scheduler_evidence': C.pin(schedulers[0]),
        'worker_state': worker['state'], 'scheduler_quality': scheduler,
        'positive_worker_scheduler_transport': worker['state']['transport_batches'] > 0
            and scheduler['shadow_observations'] > 0
            and worker['state']['transport_composed_keys'] > 0
            and scheduler['shadow_applied_key_updates'] > 0,
        'incomparable_decisions': len(incomparable),
        'incomparable_decisions_all_unconstrained': all(r['shared_budget_constrained'] is False for r in incomparable),
        'incomparable_decisions_missing_raw_prefix_score': sum(
            any(c['raw_prefix_score'] is None for c in r['candidates']) for r in incomparable),
        'constrained_decisions': sum(r['shared_budget_constrained'] is True for r in scheduler['records']),
        'constrained_decisions_fully_comparable': sum(r['shared_budget_constrained'] is True and r['fully_comparable'] for r in scheduler['records']),
        'same_seed_arc_exact_comparisons': comparisons,
    })
states = quality['seeds']
quality['totals'] = {key: sum(s['scheduler_quality'][key] for s in states) for key in (
    'shadow_observations', 'shadow_applied_key_updates', 'actual_multi_candidate_staging_decisions',
    'fully_covered_comparable_decisions', 'top1_agreements', 'full_order_agreements')}
quality['totals']['top1_agreement_fraction'] = quality['totals']['top1_agreements'] / quality['totals']['fully_covered_comparable_decisions']
quality['totals']['all_errors_empty_and_valid'] = all(s['scheduler_quality']['valid'] and not s['scheduler_quality']['errors'] and not s['worker_state']['errors'] for s in states)
quality['totals']['all_positive_transport'] = all(s['positive_worker_scheduler_transport'] for s in states)
quality['totals']['all_same_seed_arc_exact_comparisons_passed'] = all(c['passed'] for s in states for c in s['same_seed_arc_exact_comparisons'])
quality['limitations'] = states[0]['scheduler_quality']['limitations']
(H / 'quality-prefix-assessment.json').write_text(json.dumps(quality, indent=2) + '\n')

profile = {'performance_eligible': False, 'diagnostic_only': True, 'runs': []}
items = {}
for policy in ('semantic', 'arc'):
    item = load(f'diagnostic-competing-1103-{policy}', 'component_diagnostic')
    items[policy] = item
    directory = Path(item['record']['raw_evidence']['path']).parent
    metrics = collections.defaultdict(lambda: {'count': 0, 'total_ns': 0, 'max_ns': 0})
    counters = collections.Counter()
    files = []
    pending = []
    for path in sorted((directory / 'components').glob('components-*.json')):
        data = json.loads(path.read_text())
        files.append(C.pin(path))
        pending.append(data['pending_cuda_samples'])
        counters.update(data['counters'])
        for name, values in data['metrics'].items():
            merged = metrics[name]
            merged['count'] += values['count']
            merged['total_ns'] += values['total_ns']
            merged['max_ns'] = max(merged['max_ns'], values['max_ns'])
    raw = item['data'][0]
    request_ids = {r['payload']['request_id'] for r in raw['requests']}
    jobs = {(j['job_id'], j['request_id']): j
            for t in raw['terminals'] if t.get('external_request_id') in request_ids
            for j in t.get('transfer_job_statuses', []) if j['direction'] == 'store'}
    sizes, runs, multiple = collections.Counter(), collections.Counter(), collections.Counter()
    for job in jobs.values():
        ids = job['source_block_ids']
        sizes[len(ids)] += 1
        run_count = sum(i == 0 or group != ids[i - 1][0] or block != ids[i - 1][1] + 1
                        for i, (group, block) in enumerate(ids))
        runs[run_count] += 1
        if len(ids) > 1:
            multiple['jobs'] += 1
            multiple['single_run_jobs'] += run_count == 1
            multiple['scattered_jobs'] += run_count > 1
    profile['runs'].append({'policy': policy, 'run': item['record'], 'profile_files': files,
        'pending_cuda_samples_by_file': pending, 'metrics': dict(metrics), 'counters': dict(counters),
        'load_only_store_layout': {'jobs': len(jobs), 'gpu_source_blocks': sum(len(j['source_block_ids']) for j in jobs.values()),
            'blocks_per_job_histogram': dict(sizes), 'contiguous_runs_per_job_histogram': dict(runs),
            'multi_block_jobs': dict(multiple),
            'provenance': 'Request-owned actual transfer source_block_ids in preserved order; warmup terminals excluded. Source groups qualify physical IDs. Tensor strides/shapes are not recorded by these profiles.'}})
profile['exact_comparison'] = C.exact_comparison(items['arc'], items['semantic'], A)
profile['limitations'] = [
    'All component totals are inclusive and overlap; they must not be summed into end-to-end cost.',
    'CUDA timings are stride16 completed samples only; totals are neither all-job GPU time nor directly extrapolated.',
    'Ordinary diagnostic phase raw performance_eligible field is true, but this explicit component plan excludes its timing from all performance confirmation.',
    'No per-job profile correlation connects a particular layout to a measured timing interval.',
]
(H / 'diagnostic-short-profile-results.json').write_text(json.dumps(profile, indent=2) + '\n')
print(json.dumps({'quality_totals': quality['totals'], 'profile_exact_passed': profile['exact_comparison']['passed'],
                  'profile_integrity': [r['run']['integrity']['passed'] for r in profile['runs']]}, indent=2))
