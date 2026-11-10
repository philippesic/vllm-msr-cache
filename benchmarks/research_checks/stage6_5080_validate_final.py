"""Check final report, all42 presentation rows, latency and rendered projection."""
from pathlib import Path
import hashlib
import json
import math
import re
from benchmarks.serving_suite.campaign_contracts import digest, Ledger

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / 'evidence/stage6-5080-20261109'
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'


def main():
    root = EVIDENCE / 'final-results-01'
    def load(path):
        return json.loads(path.read_text())
    audit, data = load(EVIDENCE / 'completion-audit-01.json'), load(root / 'comparisons.json')
    prior = load(EVIDENCE / 'primary-confirmation-results-01/comparisons.json')
    report = load(OUT / 'analysis/current/results.json')
    assert digest(report) == audit['report_digest'] and digest(data) == audit['findings_digest']
    assert data['comparisons'] == prior['comparisons'] and len(data['comparisons']) == 42
    for directory, bindings in [(root, audit['findings_files']), (EVIDENCE, audit['audits'])]:
        for name, sha in bindings.items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == sha, name
    assert digest(Ledger(OUT).events()) == audit['stable_ledger_digest']
    assert len(Ledger(OUT).index()) == 641
    assert audit['completion_flags'] == dict(complete=False, throughput=False, quality_audit_complete=True,
                                           overhead=True, operational=True, recovery=True, portability=True)
    identities, latencies = set(), set()
    for row in data['comparisons']:
        key = tuple(row[k] for k in ('model', 'cell', 'numerator', 'denominator'))
        assert key not in identities
        identities.add(key)
        assert row['process_pairs'] == len(row['per_run_ratios'])
        assert all(math.isfinite(v) and v > 0 for v in [row['throughput_ratio'], *row['paired_95_interval']])
        assert row['paired_95_interval'][0] <= row['throughput_ratio'] <= row['paired_95_interval'][1]
        for role in ('numerator', 'denominator'):
            arm = row[role + '_summary']
            assert arm['failed_requests'] == 0 and arm['completed_requests'] > 0
            for stat in arm['latency'].values():
                assert math.isfinite(stat['mean_of_process_statistics']) and stat['mean_of_process_statistics'] >= 0
            latencies.add((row['model'], row['cell'], row[role]))
    text = (root / 'chart.html').read_text()
    chart = json.loads(text.split('const data=', 1)[1].split(';const $=', 1)[0])
    assert len(chart['comparisons']) == 42 and chart['missing_blocks'] == data['missing_blocks']
    projection = load(root / 'chart-validation.json')
    assert hashlib.sha256(text.encode()).hexdigest() == projection['chart_html_sha256']
    for original, shown in zip(data['comparisons'], chart['comparisons'], strict=True):
        for key in projection['presentation_fields']:
            assert original[key] == shown[key]
        for role in ('numerator_summary', 'denominator_summary'):
            assert original[role]['serving_seconds']['mean'] == shown[role]['serving_seconds']['mean']
            assert original[role]['length_caps'] == shown[role]['length_caps']
    assert (root / 'render.json').is_file()
    documents = [(p, p.read_text()) for p in (EVIDENCE / 'RESULTS.md', EVIDENCE / 'CONTINUE.md')]
    status = REPO / 'docs/STATUS.md'
    section = status.read_text().split('## Stage 6 RTX 5080 rebuild', 1)[1].split('\n## ', 1)[0]
    documents.append((status, section))
    for path, contents in documents:
        for link in re.findall(r'\]\(([^)]+)\)', contents):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent / link.split('#')[0]).exists(), (str(path), link)
    record = dict(schema='stage6-5080-final-presentation-independent-check-v1', all_42_rows_equal_terminal=True,
        all_final_report_and_audit_bindings_verified=True, all_latency_fields_finite=True,
        distinct_model_cell_system_latency_rows=len(latencies), all_chart_projection_fields_exact=True,
        chart_preview_and_render_receipt='render.json', missing_blocks_preserved=data['missing_blocks'],
        ledger_unchanged=True, all_641_attempts_terminal=all(e['state'] in ('verified', 'failed', 'interrupted')
                                                          for e in Ledger(OUT).index().values()),
        project_result_continuation_and_affected_status_links_resolve=True,
        scope='Final data/report/visual consistency; unrelated historical status links excluded. No acquisition or reinterpretation of the three genuine exclusions.')
    assert record['all_641_attempts_terminal']
    (root / 'final-validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
