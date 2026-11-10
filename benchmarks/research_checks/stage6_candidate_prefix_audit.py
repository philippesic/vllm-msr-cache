"""Measure the CPU-prefix extent behind retained positive hint acknowledgements."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import statistics

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
DEST = REPO / 'evidence/stage6-5080-20261109/candidate-audit-01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = DEST / 'prefix-coverage.json'
    assert not target.exists()
    flow = json.loads((DEST / 'hint-flow.json').read_text())
    groups = defaultdict(list)
    cases = []
    for record in flow['cases']:
        if record['phase'] != 'confirm':
            continue
        folder = OUT / record['artifact_path']
        path = folder / 'measurement/result.json'
        assert sha(path) == record['result_sha256']
        case = json.loads(path.read_text())['cases'][0]
        fractions, ready_counts, full = [], [], 0
        for hint in case['hints']:
            response = hint.get('response', {})
            if response.get('reason') != 'cpu_prefix_recency_refreshed':
                continue
            expected = hint['hint']['prefix_boundary_tokens'] // 16
            assert expected > 0
            ready = response['ready_cpu_prefix_keys']
            assert 0 <= ready <= expected
            full += ready == expected
            fractions.append(ready / expected)
            ready_counts.append(ready)
        row = dict(attempt=record['attempt'], model=record['model'], cell=record['cell'], repeat=record['repeat'],
            selected_confirmation=record['selected_confirmation'], hint_events=record['hint_events'],
            positive_ready_events=record['positive_refresh_events'], full_cpu_prefix_ready_events=full,
            partial_cpu_prefix_ready_events=len(fractions) - full,
            cpu_prefix_ready_fraction_median=statistics.median(fractions) if fractions else None,
            cpu_prefix_ready_fraction_minimum=min(fractions) if fractions else None,
            ready_cpu_keys_histogram=dict(Counter(ready_counts)))
        cases.append(row)
        if record['selected_confirmation']:
            groups[(record['model'], record['cell'])].append((row, fractions))
    summary = []
    for (model, cell), records in groups.items():
        fractions = [f for _, values in records for f in values]
        events = sum(row['hint_events'] for row, _ in records)
        positive = sum(row['positive_ready_events'] for row, _ in records)
        full = sum(row['full_cpu_prefix_ready_events'] for row, _ in records)
        summary.append(dict(model=model, cell=cell, processes=len(records), hint_events=events,
            positive_ready_events=positive, full_cpu_prefix_ready_events=full,
            positive_fraction=positive / events if events else None,
            full_cpu_prefix_ready_fraction=full / events if events else None,
            median_extent_of_cpu_prefix=statistics.median(fractions) if fractions else None))
    illustrative = next(r for r in flow['cases'] if r['selected_confirmation'] and
                        (r['model'], r['cell']) == ('qwen3_8b_awq', 'B09'))
    folder = OUT / illustrative['artifact_path']
    c = json.loads((folder / 'measurement/result.json').read_text())['cases'][0]
    path = folder / 'measurement' / c['name'] / 'requests.jsonl'
    seal = json.loads((folder / 'seal.json').read_text())
    assert sha(path) == seal['files'][str(path.relative_to(folder))]
    prompts = [r['payload']['prompt'] for r in map(json.loads, path.read_text().splitlines())
               if not r['is_returning'] and not r['cancel_probe']]
    common = 0
    for i in range(min(map(len, prompts))):
        if len({p[i] for p in prompts}) != 1:
            break
        common += 1
    histogram = Counter(h.get('response', {}).get('ready_cpu_prefix_keys') for h in c['hints'])
    sample = dict(attempt=illustrative['attempt'], initial_prompts=len(prompts),
        common_leading_token_count=common, common_complete_16_token_blocks=common // 16,
        ready_key_count_histogram=dict(histogram), request_file_sha256=sha(path),
        scope='Illustrative retained case. Most common ready count is8 blocks;7 complete leading blocks have identical token content across all16 initial prompts. This diagnoses small-head refreshing, not unique useful retention.')
    result = dict(schema='stage6-candidate-cpu-prefix-extent-v1', groups=summary, cases=cases,
        illustrative_shared_head=sample, input_hint_flow_sha256=sha(DEST / 'hint-flow.json'),
        scope='CPU readiness at advice time, weighted by repeated hint events. Not full GPU+CPU cache coverage, later consumption, unique retained bytes or a new pass/fail rule. Full CPU residency is not required when GPU roots bridge to a CPU suffix.',
        benchmark_outcomes_unchanged=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([r for r in summary if r['hint_events']], indent=2))


if __name__ == '__main__':
    main()
