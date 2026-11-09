"""Present a completed primary campaign without changing its frozen estimators."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import statistics

from benchmarks.serving_suite.campaign_contracts import digest, load
from benchmarks.serving_suite.campaign_report import paired_estimate


LABELS = {'arc': 'ARC', 'wait': 'Candidate', 'continuum': 'Continuum',
          'upstream': 'Upstream PR #59714'}
LATENCIES = ('end_user_ttft_mean_s', 'end_user_ttft_p95_s', 'e2e_mean_s', 'e2e_p95_s')
SCOPE = ('Completed offered graph units/s, conditional on the frozen cohort and host. '
         'Student t 95% CIs on paired whole-process log throughput ratios; no multiplicity '
         'adjustment. Requests within a process are not independent repetitions. '
         'Serving seconds exclude startup. Latencies are arithmetic means of process '
         'statistics, including process p95s, not pooled request percentiles. '
         'Live output lengths, quality, runtimes and interfaces can differ. '
         'Throughput is not correctness-adjusted useful work; quality equivalence '
         'and universal superiority are not established. Missing external native '
         'activity counters are unavailable, not zero.')


def arm_summary(points: list[dict]) -> dict:
    work = [p['request_work'] for p in points]
    result = dict(
        serving_seconds=dict(mean=statistics.mean(p['duration_s'] for p in points),
                             values=[p['duration_s'] for p in points]),
        startup_seconds=dict(mean=statistics.mean(p['startup_s'] for p in points),
                             values=[p['startup_s'] for p in points]),
        completed_requests=sum(p['completed'] for p in work),
        failed_requests=sum(p['failed'] for p in work),
        prompt_tokens=sum(p['prompt_tokens'] for p in work),
        output_tokens=sum(p['output_tokens'] for p in work),
        length_caps=sum(p['finish_reasons'].get('length', 0) for p in work),
        attempts=[p['attempt'] for p in points], latency={},
    )
    for field in LATENCIES:
        values = [p.get(field) for p in work]
        if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values):
            result['latency'][field] = dict(mean_of_process_statistics=statistics.mean(values), values=values)
        else:
            result['latency'][field] = dict(available=False, values=values)
    validity = [p['measurement_validity'] for p in points]
    result['cpu_pressure_evidence'] = dict(Counter(p.get('cpu_pressure_evidence', 'unavailable') for p in validity))
    result['cpu_restore_activity'] = dict(Counter(p.get('cpu_restore_activity', 'unavailable') for p in validity))
    result['measurement_classifications'] = dict(Counter(p['classification'] for p in validity))
    return result


def summarize(report: dict) -> dict:
    if not report.get('complete') or not report.get('throughput_acquisition_complete') or report.get('missing_blocks'):
        raise ValueError('Final findings require a completed primary report and all frozen blocks')
    rows, keys = [], set()
    for item in report['comparisons']:
        key = tuple(item[k] for k in ('model', 'cell', 'candidate', 'baseline'))
        if key in keys:
            raise ValueError('Duplicate primary comparison')
        keys.add(key)
        estimate = item['estimate']
        ratios = estimate['per_run_ratios']
        expected = paired_estimate(ratios)
        if (len(item['points']) != item['process_pairs'] or item['process_pairs'] != len(ratios)
                or item['process_pairs'] != estimate['process_pairs'] or len(ratios) < 2):
            raise ValueError('Comparison does not contain its whole matched process pairs')
        for a, b in zip([estimate['geometric_throughput_ratio'], *estimate['paired_95_interval']],
                        [expected['geometric_throughput_ratio'], *expected['paired_95_interval']], strict=True):
            if not math.isclose(a, b, rel_tol=1e-12):
                raise ValueError('Estimate differs from the frozen paired log-ratio estimator')
        for point, ratio in zip(item['points'], ratios, strict=True):
            numerator, denominator = point['candidate'], point['baseline']
            if (numerator['completed_units'] != denominator['completed_units']
                    or numerator['request_work']['failed'] or denominator['request_work']['failed']
                    or not math.isclose(numerator['completed_units_per_s'] / denominator['completed_units_per_s'],
                                        ratio, rel_tol=1e-12)):
                raise ValueError('Matched work or per-process throughput ratio disagrees')
        row = dict(model=item['model'], cell=item['cell'], numerator=item['candidate'],
                   denominator=item['baseline'], process_pairs=item['process_pairs'],
                   throughput_ratio=estimate['geometric_throughput_ratio'],
                   paired_95_interval=list(estimate['paired_95_interval']), per_run_ratios=list(ratios),
                   numerator_summary=arm_summary([p['candidate'] for p in item['points']]),
                   denominator_summary=arm_summary([p['baseline'] for p in item['points']]),
                   frozen_orientation=dict(candidate=item['candidate'], baseline=item['baseline']),
                   frozen_classification=item['status'],
                   frozen_traffic_classes=copy.deepcopy(item.get('traffic_classes', {})),
                   activity_evidence=copy.deepcopy(item.get('activity_evidence', {})))
        # Human-facing comparisons consistently put the native candidate above
        # external systems. Inverting a CI also reverses its bound order.
        if row['denominator'] == 'wait':
            row['numerator'], row['denominator'] = row['denominator'], row['numerator']
            row['numerator_summary'], row['denominator_summary'] = row['denominator_summary'], row['numerator_summary']
            row['throughput_ratio'] = 1 / row['throughput_ratio']
            row['paired_95_interval'] = [1 / row['paired_95_interval'][1], 1 / row['paired_95_interval'][0]]
            row['per_run_ratios'] = [1 / r for r in ratios]
            row['orientation_inverted'] = True
        else:
            row['orientation_inverted'] = False
        rows.append(row)
    if not rows:
        raise ValueError('Completed primary report has no comparisons')
    return dict(schema='stage6-primary-findings-v1', campaign_id=report['campaign_id'],
                report_digest=digest(report), report_complete=True, comparisons=rows,
                labels=LABELS, scope=SCOPE,
                frozen_detail_scope='Traffic-class estimates and classification retain frozen_orientation; displayed reciprocal throughput does not reclassify outcomes.')


def markdown(data: dict) -> str:
    lines = [SCOPE, '', '| Model / cell | Comparison | n | Numerator mean s | Denominator mean s | Throughput ratio [95% CI] |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for r in data['comparisons']:
        a, b = (r[k + '_summary']['serving_seconds']['mean'] for k in ('numerator', 'denominator'))
        lo, hi = r['paired_95_interval']
        lines.append(f"| {r['model']} / {r['cell']} | {LABELS[r['numerator']]}/{LABELS[r['denominator']]} | {r['process_pairs']} | {a:.3f} | {b:.3f} | {r['throughput_ratio']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines += ['', 'Latency values below are milliseconds. Each is the arithmetic mean of the corresponding per-process statistic; a mean of process p95s is not a pooled p95 or a population-tail guarantee.', '',
              '| Model / cell | Comparison arm | Mean TTFT ms | Mean process p95 TTFT ms | Mean completion ms | Mean process p95 completion ms |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    seen = set()
    for r in data['comparisons']:
        for arm in ('numerator', 'denominator'):
            key = (r['model'], r['cell'], r[arm])
            if key in seen:
                continue
            seen.add(key)
            values = [r[arm + '_summary']['latency'][k].get('mean_of_process_statistics') for k in LATENCIES]
            formatted = ['unavailable' if v is None else f'{v * 1000:.3f}' for v in values]
            lines.append(f"| {key[0]} / {key[1]} | {LABELS[key[2]]} | " + ' | '.join(formatted) + ' |')
    return '\n'.join(lines) + '\n'


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 6 primary comparisons</title>
<style>:root{color-scheme:light dark;--fg:#17202b;--bg:#fff;--line:#d7dde5;--accent:#1460b7} @media(prefers-color-scheme:dark){:root{--fg:#e8edf5;--bg:#161b23;--line:#424c5a;--accent:#7bb6ff}}body{margin:18px auto;padding:0 14px;max-width:1020px;font:14px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg)}h1{font-size:22px}select{font:inherit;padding:5px;margin:3px 14px 3px 5px;background:var(--bg);color:var(--fg);border:1px solid var(--line)}svg{width:100%;display:block}svg text{fill:var(--fg);font:12px system-ui}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:7px 5px;text-align:left}.scroll{overflow:auto}p{max-width:90ch}.scope{font-size:12px}.ci{stroke:var(--accent);stroke-width:3}.point{fill:var(--accent)}</style>
<h1>Stage 6 primary throughput comparisons</h1><p>Right of 1 is faster. Each point is a geometric mean of paired whole-process throughput ratios; bars show 95% confidence intervals. Serving time excludes startup. Candidate means the native wait-hint system.</p>
<label>Model<select id="model"></select></label><label>Comparison<select id="comparison"></select></label>
<svg id="plot" role="img" aria-label="Whole-process throughput ratios with confidence intervals"></svg>
<div class="scroll"><table><thead><tr><th>Cell</th><th>n</th><th>Ratio [95% CI]</th><th>Numerator mean s</th><th>Denominator mean s</th><th>Numerator / denominator caps</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="scope" id="scope"></p><p class="scope">Inspect the accompanying JSON and latency table for actual CPU-pressure scope, output lengths, latency figures and frozen traffic-class analyses. A workload name does not prove CPU pressure. External systems cover the configured anchor subset; this is not a full model/workload/system cross product.</p>
<script>const data=__DATA__;const $=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';const model=$('model'),comparison=$('comparison');$('scope').textContent=data.scope;
function option(target,value,label){target.add(new Option(label,value))}function pair(r){return r.numerator+'/'+r.denominator}function label(r){return data.labels[r.numerator]+' / '+data.labels[r.denominator]}
for(const v of [...new Set(data.comparisons.map(r=>r.model))])option(model,v,v);
function choose(){comparison.replaceChildren();const rows=data.comparisons.filter(r=>r.model===model.value);for(const v of [...new Set(rows.map(pair))])option(comparison,v,label(rows.find(r=>pair(r)===v)));comparison.value=rows.some(r=>pair(r)==='wait/arc')?'wait/arc':comparison.value;draw()}
function node(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;$('plot').append(e);return e}
function draw(){const rows=data.comparisons.filter(r=>r.model===model.value&&pair(r)===comparison.value);$('plot').replaceChildren();$('rows').replaceChildren();const height=70+rows.length*38; $('plot').setAttribute('viewBox',`0 0 800 ${height}`);const lo=Math.min(.95,...rows.map(r=>r.paired_95_interval[0]))*.98,hi=Math.max(1.05,...rows.map(r=>r.paired_95_interval[1]))*1.02,x=v=>100+(v-lo)/(hi-lo)*670;
for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;node('line',{x1:x(v),x2:x(v),y1:12,y2:height-35,stroke:'var(--line)'});node('text',{x:x(v),y:height-12,'text-anchor':'middle'},v.toFixed(3))}node('line',{x1:x(1),x2:x(1),y1:12,y2:height-35,stroke:'var(--fg)','stroke-dasharray':'4 4'});
rows.forEach((r,i)=>{const y=32+38*i;node('text',{x:8,y:y+4},r.cell);node('line',{x1:x(r.paired_95_interval[0]),x2:x(r.paired_95_interval[1]),y1:y,y2:y,class:'ci'});const point=node('circle',{cx:x(r.throughput_ratio),cy:y,r:4.5,class:'point'}),title=document.createElementNS(ns,'title');title.textContent=`${r.cell}: ${r.throughput_ratio.toFixed(4)} [${r.paired_95_interval.map(v=>v.toFixed(4)).join(', ')}]`;point.append(title);const tr=document.createElement('tr');for(const v of [r.cell,r.process_pairs,`${r.throughput_ratio.toFixed(4)} [${r.paired_95_interval.map(v=>v.toFixed(4)).join(', ')}]`,r.numerator_summary.serving_seconds.mean.toFixed(3),r.denominator_summary.serving_seconds.mean.toFixed(3),r.numerator_summary.length_caps+' / '+r.denominator_summary.length_caps]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('rows').append(tr)})}model.onchange=choose;comparison.onchange=draw;choose();</script></html>'''


def build(report_path: Path, destination: Path) -> dict:
    data = summarize(load(report_path))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'comparisons.json').write_text(json.dumps(data, indent=2) + '\n')
    (destination / 'tables.md').write_text(markdown(data))
    encoded = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    (destination / 'comparisons.html').write_text(HTML.replace('__DATA__', encoded))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = build(args.report, args.out)
    print(json.dumps(dict(comparisons=len(data['comparisons']), campaign_id=data['campaign_id'])))


if __name__ == '__main__':
    main()
