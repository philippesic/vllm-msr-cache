"""Extract all42 terminal confirmation comparisons, preserving campaign incompleteness."""
from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
BASE = REPO / 'output/stage6-5080-20261109'
EVIDENCE = REPO / 'evidence/stage6-5080-20261109'
MODEL = 'qwen3_8b_awq'


def main():
    assert Path.cwd() == SOURCE.resolve()
    assert sys.executable == '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
    assert sys.dont_write_bytecode and OUT.parent.is_mount()
    private = BASE / 'primary-confirmation-report-01'
    destination = EVIDENCE / 'primary-confirmation-results-01'
    assert not private.exists() and not destination.exists()
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [str(SOURCE)] + [p for p in sys.path if Path(p).resolve() != script_directory]
    from benchmarks.serving_suite import campaign_report as frozen
    from benchmarks.serving_suite.campaign_contracts import load, digest, atomic_json
    assert Path(frozen.__file__).resolve().is_relative_to(SOURCE.resolve())
    protocol = load(OUT / 'protocol.lock.json')
    frozen_sha = hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest()
    assert frozen_sha == protocol['source']['files']['benchmarks/serving_suite/campaign_report.py']
    inventory = load(EVIDENCE / 'confirmation-attempts-complete.json')
    assert inventory['execution_complete'] and inventory['source_digest'] == protocol['source']['digest']
    selected = frozen.completed_blocks(OUT, 'confirm')
    anchor = {k:v for k,v in selected.items() if k[0] == MODEL}
    assert len(anchor) == 129 and len(selected) == inventory['complete_matched_blocks'] == 207
    assert inventory['terminal_logical_configurations'] == 524
    live = OUT / 'analysis/current'

    def contents(path):
        return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in path.rglob('*') if p.is_file()} if path.exists() else {}

    before = contents(live)
    private.mkdir()
    original_open = Path.open

    def report_json(path, value):
        assert Path(path) == live / 'results.json'
        atomic_json(private / 'full-report.json', value)

    def report_open(path, *args, **kwargs):
        if path == live / 'results.csv':
            path = private / 'full-report.csv'
        return original_open(path, *args, **kwargs)

    original_blocks = frozen.completed_blocks

    def report_blocks(out, phase, **kwargs):
        if phase == 'confirm' and not kwargs:
            return selected
        return original_blocks(out, phase, **kwargs)

    with patch.object(frozen, 'completed_blocks', report_blocks), \
         patch.object(frozen, 'atomic_json', report_json), patch.object(Path, 'open', report_open):
        report = frozen.report(OUT)
    assert contents(live) == before, 'Live report directory must remain unchanged'
    assert not report['complete'] and not report['throughput_acquisition_complete']
    scoped = copy.deepcopy(report)
    scoped['comparisons'] = copy.deepcopy(report['comparisons'])
    assert len(scoped['comparisons']) == 42
    assert all(row['process_pairs'] >= 2 and 'estimate' in row for row in scoped['comparisons'])
    assert scoped['missing_blocks'] == report['missing_blocks']
    scoped['presentation_subset'] = dict(final_campaign=False,terminal_confirmation=True,
        full_planned_comparisons=len(report['comparisons']),displayed_comparisons=42,
        scope='All terminal confirmation comparisons; three original activity-invalid blocks stay excluded. Supplementary phases and final all-attempt-cap/campaign/portability audits remain pending. No aggregate across different workloads.')
    atomic_json(private / 'primary-confirmation-report.json', scoped)
    helper = REPO / 'benchmarks/research_checks/stage6_primary_findings.py'
    spec = importlib.util.spec_from_file_location('anchor_presentation', helper)
    presentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(presentation)
    data = presentation.summarize(scoped, retain_incomplete=True)
    data['presentation_subset'] = scoped['presentation_subset']
    data['scope'] = 'All42 terminal confirmation comparisons; supplementary phases and final audits pending. ' + data['scope']
    data['incompleteness'] = 'Terminal confirmation with three activity-invalid excluded blocks; final campaign audits pending. ' + data['incompleteness']
    prior = load(EVIDENCE / 'confirmation-three-cells.json')
    for old in prior['cells']:
        row = next(r for r in data['comparisons'] if r['cell'] == old['cell']
                   and r['numerator'] == 'wait' and r['denominator'] == 'arc')
        assert math.isclose(row['throughput_ratio'], old['throughput']['geometric_throughput_ratio'], rel_tol=1e-12)
        assert all(math.isclose(a, b, rel_tol=1e-12) for a,b in
                   zip(row['paired_95_interval'], old['throughput']['paired_95_interval'], strict=True))
        for role, system in (('numerator', 'wait'), ('denominator', 'arc')):
            summary = row[role + '_summary']
            assert math.isclose(summary['serving_seconds']['mean'], old['arms'][system]['mean_serving_seconds'], rel_tol=1e-12)
            for key, value in old['arms'][system]['latency'].items():
                assert math.isclose(summary['latency'][key]['mean_of_process_statistics'],
                                    value['mean_of_process_statistics'], rel_tol=1e-12)
    b06 = [r for r in data['comparisons'] if r['cell'] == 'B06' and r['model'] == MODEL]
    assert len(b06) == 5 and all(r['process_pairs'] == 7 and r['missing_repeats'] == [0]
                               and r['process_pairs_plus_reported_missing_blocks'] == 8 for r in b06)
    earlier = load(EVIDENCE / 'anchor-results-01/comparisons.json')
    def key(row):
        return tuple(row[k] for k in ('model','cell','numerator','denominator'))
    current = {key(row):row for row in data['comparisons']}
    assert len(current) == 42 and len(earlier['comparisons']) == 32
    for row in earlier['comparisons']:
        assert current[key(row)] == row, 'Every anchor estimate, interval, serving time, latency and diagnostic must reproduce exactly'
    for row in data['comparisons']:
        if row['model'] == 'ministral3_3b':
            expected = 4 if row['cell'] == 'B06' else 6
            assert row['process_pairs'] == expected
            assert row['missing_repeats'] == ([0,3] if row['cell'] == 'B06' else [])
        elif row['model'] == 'qwen3_4b_2507':
            assert row['process_pairs'] == 10 and row['missing_repeats'] == []
    assert len(data['missing_blocks']) == 3
    destination.mkdir()
    atomic_json(destination / 'comparisons.json', data)
    (destination / 'tables.md').write_text(presentation.markdown(data))
    encoded = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    (destination / 'comparisons.html').write_text(presentation.HTML.replace('__DATA__', encoded))
    atomic_json(destination / 'validation.json', dict(
        schema='stage6-5080-terminal-confirmation-findings-v1', final_campaign=False, terminal_confirmation=True,
        frozen_report_sha256=frozen_sha, presentation_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        full_private_report=str(private / 'full-report.json'), full_report_digest=digest(report),
        scoped_report_digest=digest(scoped), original_report_directory_unchanged=True,
        frozen_completed_blocks_snapshot=len(selected), anchor_completed_blocks=len(anchor),
        all_42_planned_comparisons_present=True, all_32_anchor_rows_reproduce_exactly=True, all_32_anchor_comparisons_present=True, all_missing_blocks_preserved=True,
        earlier_three_cell_ratios_intervals_serving_and_latency_reproduced=True,
        b06_seven_of_eight_all_comparisons=True,
        scope='Frozen report code and validated presentation functions; only JSON/CSV writes redirected. '
              'No acquisition, source, estimator, repetition, validity, quality or live-report change. '
              'All42 conditional estimates do not complete the campaign or establish quality equivalence; three original missing blocks are not imputed and supplementary phases/final audits remain separate.'))
    headlines = [dict(cell=r['cell'], comparison=r['numerator']+'/'+r['denominator'],
                     pairs=r['process_pairs'], ratio=r['throughput_ratio'], ci=r['paired_95_interval'])
                  for r in data['comparisons'] if r['cell'] in ('B06','B08','B09','B11','B12')
                  and r['numerator'] == 'wait']
    print(json.dumps(dict(displayed_comparisons=len(data['comparisons']),
                          findings_complete=data['findings_complete'], headlines=headlines), indent=2))


if __name__ == '__main__':
    main()
