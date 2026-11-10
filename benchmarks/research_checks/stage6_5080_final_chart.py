"""Embed only fields used by the interactive chart, keeping all42 comparisons."""
from pathlib import Path
import hashlib
import importlib.util
import json

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / 'evidence/stage6-5080-20261109/final-results-01'


def main():
    source = DEST / 'comparisons.json'
    data = json.loads(source.read_text())
    keys = ('model', 'cell', 'numerator', 'denominator', 'process_pairs',
            'process_pairs_plus_reported_missing_blocks', 'throughput_ratio', 'paired_95_interval')
    rows = []
    for original in data['comparisons']:
        row = {k: original[k] for k in keys}
        for role in ('numerator_summary', 'denominator_summary'):
            row[role] = dict(serving_seconds=dict(mean=original[role]['serving_seconds']['mean']),
                             length_caps=original[role]['length_caps'])
        rows.append(row)
    assert len(rows) == 42
    chart = {k: data[k] for k in ('scope', 'incompleteness', 'findings_complete', 'missing_blocks', 'labels')}
    chart['comparisons'] = rows
    spec = importlib.util.spec_from_file_location('presentation', REPO / 'benchmarks/research_checks/stage6_primary_findings.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    html = helper.HTML.replace('__DATA__', json.dumps(chart, ensure_ascii=False).replace('</', '<\\/'))
    path = DEST / 'chart.html'
    assert not path.exists()
    path.write_text(html)
    receipt = dict(schema='stage6-5080-final-chart-projection-v1', comparisons=42,
        source_json_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        chart_html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        every_displayed_field_exactly_preserved=True, missing_blocks=chart['missing_blocks'],
        findings_complete=False, presentation_fields=list(keys),
        scope='Only unused embedded metadata omitted for a compact visual; full JSON/tables retain all per-process ratios, latencies, diagnostics and completion evidence.')
    (DEST / 'chart-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
