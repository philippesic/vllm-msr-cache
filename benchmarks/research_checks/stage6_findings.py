"""Build compact, explicitly qualified Stage 6 views from existing analysis only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def build(primary: Path, addendum: Path, destination: Path) -> dict:
    original = json.loads((primary / 'analysis/current/results.json').read_text())
    extra = json.loads((addendum / 'analysis/addendum-comparison.json').read_text())
    comparisons = []
    for item in original['comparisons']:
        estimate = item['estimate']
        row = {k: item[k] for k in ('model', 'cell', 'candidate', 'baseline', 'process_pairs')}
        row.update(throughput_ratio=estimate['geometric_throughput_ratio'],
                   paired_95_interval=estimate['paired_95_interval'],
                   per_run_ratios=estimate['per_run_ratios'])
        for arm in ('candidate', 'baseline'):
            points = [p[arm] for p in item['points']]
            seconds = [p['duration_s'] for p in points]
            row[arm + '_seconds'] = dict(mean=statistics.mean(seconds), median=statistics.median(seconds), values=seconds)
            row[arm + '_work'] = dict(
                requests=sum(p['request_work']['completed'] for p in points),
                failures=sum(p['request_work']['failed'] for p in points),
                prompt_tokens=sum(p['request_work']['prompt_tokens'] for p in points),
                output_tokens=sum(p['request_work']['output_tokens'] for p in points),
                caps=sum(p['request_work']['finish_reasons'].get('length', 0) for p in points),
                artifacts=[dict(attempt=p['attempt']) for p in points])
        assert len(item['points']) == row['process_pairs'] == len(row['per_run_ratios'])
        comparisons.append(row)
    pressure = defaultdict(lambda: dict(processes=0, pressure_processes=0, returning_loaded_tokens=0,
                                       eviction_affected_returns=0, returning_requests=0, caps=0))
    for case in extra['secondary_confirmation_pressure']['cases']:
        key = ':'.join(case[k] for k in ('model', 'cell', 'system'))
        group = pressure[key]
        native = case['native_cache']
        group['processes'] += 1
        group['pressure_processes'] += int(case['pressure_observed'])
        group['returning_loaded_tokens'] += native['returning_loaded_tokens']
        group['eviction_affected_returns'] += native['returning_requests_with_prior_cpu_eviction']
        group['returning_requests'] += native['returning_requests']
        group['caps'] += case['caps']
    result = dict(schema='stage6-compact-comparisons-v1', primary_comparisons=comparisons,
                  secondary_actual_pressure=dict(pressure), addendum_comparisons=extra['comparisons'],
                  addendum_missing_blocks=extra['missing_blocks'],
                  scope='Existing sealed whole-block measurements. Quality review/transfer completion separate. Paired whole-process log Student-t 95% CIs, no multiplicity adjustment. Throughput is completed offered graph units/s; not quality-adjusted useful output. Different live output lengths/runtimes matter. Anchor B06 lacks CPU pressure; repository addendum gate failed before serving, no repeated four-system pressure conclusion.')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'comparisons.json').write_text(json.dumps(result, indent=2) + '\n')
    data = json.dumps(result, ensure_ascii=False).replace('</', '<\\/')
    html = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:var(--font-sans);color:var(--foreground);margin:0}p{line-height:1.45;color:var(--muted-foreground)}select{font:inherit;color:var(--foreground);background:var(--secondary);border:1px solid var(--border);padding:5px}svg{width:100%;height:610px}text{fill:var(--foreground);font-size:12px}.axis{stroke:var(--border)}.line{stroke:var(--chart-1);stroke-width:2}.point{fill:var(--chart-1)}table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:7px 3px;text-align:left;border-bottom:1px solid var(--border)}.warn{color:var(--warning)}@media(max-width:500px){text{font-size:11px}table{font-size:11px}}
</style></head><body>
<p>Completed offered-work throughput ratio, selected system ÷ comparator. Right of 1 is faster. Paired whole-process 95% intervals; 6–12 independent matched repetitions per cell. Timing is qualified by output quality and generated-length differences.</p>
<label>Model <select id="model"></select></label> <label>System <select id="system"></select></label> <label>Comparator <select id="base"></select></label>
<svg id="plot" viewBox="0 0 700 610" role="img" aria-label="Throughput ratio and confidence intervals"></svg>
<table><thead><tr><th>Cell</th><th>System mean s</th><th>Comparator mean s</th><th>Ratio [95% CI]</th><th>n</th></tr></thead><tbody id="rows"></tbody></table>
<p class="warn">Anchor B06 is a GPU-reuse control with no CPU-pressure restoration. Repeated four-system repository pressure evidence is missing: the separate addendum gate failed before serving and was not replaced.</p>
<p>Fresh QA overhead: O2/O0 throughput 0.9667 [0.9539, 0.9797], 24 pairs. The 3% loss bound remains unresolved. Misleading/ARC unhinted mean TTFT: 1.1456 [1.1066, 1.1860], four pressure-qualified blocks. This is a measured interference cost. Tail percentiles are descriptive (16 unhinted requests/process).</p>
<p>Configured-system comparisons: native ARC shares the candidate runtime; Continuum and pinned upstream use different runtime/backend/chunk/hint interfaces at matched capacity. All 12 workloads run on anchor Qwen3-8B-AWQ; two secondary models cover a five-workload subset. External systems are an anchor five-workload subset. No full cross-product or universal superiority claim.</p>
<script>const data=DATA;const ns="http://www.w3.org/2000/svg";const $=id=>document.getElementById(id);const entries=data.primary_comparisons;const model=$("model"),base=$("base"),system=$("system");for(const v of [...new Set(entries.map(x=>x.model))])model.add(new Option(v,v));
function systems(){system.replaceChildren();for(const v of [...new Set(entries.filter(x=>x.model===model.value).map(x=>x.candidate))])system.add(new Option(v,v));bases()}
function bases(){base.replaceChildren();for(const v of [...new Set(entries.filter(x=>x.model===model.value&&x.candidate===system.value).map(x=>x.baseline))])base.add(new Option(v,v));draw()}
function el(tag,attrs,text){let e=document.createElementNS(ns,tag);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;$("plot").append(e)}
function draw(){const rows=entries.filter(x=>x.model===model.value&&x.candidate===system.value&&x.baseline===base.value);$("rows").replaceChildren();$("plot").replaceChildren();let lo=Math.min(.95,...rows.map(x=>x.paired_95_interval[0]))-.025,hi=Math.max(1.05,...rows.map(x=>x.paired_95_interval[1]))+.025;const x=v=>100+(v-lo)/(hi-lo)*560;for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;el("line",{x1:x(v),x2:x(v),y1:20,y2:570,class:"axis"});el("text",{x:x(v),y:595,"text-anchor":"middle"},v.toFixed(2))}el("line",{x1:x(1),x2:x(1),y1:20,y2:570,stroke:"var(--muted-foreground)","stroke-dasharray":"4 4"});rows.forEach((r,i)=>{let y=45+i*510/Math.max(1,rows.length-1);el("text",{x:12,y:y+4},r.cell+" (n="+r.process_pairs+")");el("line",{x1:x(r.paired_95_interval[0]),x2:x(r.paired_95_interval[1]),y1:y,y2:y,class:"line"});el("circle",{cx:x(r.throughput_ratio),cy:y,r:4,class:"point"});let tr=document.createElement("tr");[r.cell,r.candidate_seconds.mean.toFixed(3),r.baseline_seconds.mean.toFixed(3),r.throughput_ratio.toFixed(4)+" ["+r.paired_95_interval.map(v=>v.toFixed(4)).join(", ")+"]",r.process_pairs].forEach(v=>{let td=document.createElement("td");td.textContent=v;tr.append(td)});$("rows").append(tr)})}
model.onchange=systems;system.onchange=bases;base.onchange=draw;systems();</script></body></html>'''.replace('DATA', data)
    (destination / 'comparisons.html').write_text(html)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--addendum', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = build(a.primary, a.addendum, a.out)
    print(json.dumps(dict(primary_comparisons=len(r['primary_comparisons']), secondary_groups=len(r['secondary_actual_pressure']))))


if __name__ == '__main__':
    main()
