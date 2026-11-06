"""Standalone interactive chart from verified campaign comparison records."""
from pathlib import Path
import json

from .campaign_contracts import load

WORKLOADS = dict(B01='Chat pauses',B02='Short QA',B03='Long summaries',B04='Hot / scan / return',
    B05='Sequential repository',B06='Repository pressure',B07='Roomy repository',
    B08='Documents, full CPU',B09='Documents, half CPU',B10='Specialist teams',
    B11='Mixed traffic',B12='One tool worker')


def render(report_path,destination):
    report = load(report_path)
    points = []
    for row in report['comparisons']:
        if row['baseline']!='arc' or 'estimate' not in row:
            continue
        estimate = row['estimate']
        points.append(dict(model=row['model'],cell=row['cell'],system=row['candidate'],
            ratio=estimate['geometric_throughput_ratio'],interval=estimate['paired_95_interval'],
            n=estimate['process_pairs'],status=row['status']))
    if not points:
        raise ValueError('No measured comparisons to chart')
    data = json.dumps(dict(points=points,workloads=WORKLOADS,complete=report['complete'])).replace('</','<\\/')
    document = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font-family:var(--font-sans);color:var(--foreground)}
p{font-size:13px;line-height:1.5;color:var(--muted-foreground)}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px;font-size:13px;margin:12px 0}
select{font:inherit;color:var(--foreground);background:var(--secondary);border:1px solid var(--border);border-radius:var(--radius);padding:6px}
.key{display:inline-flex;align-items:center;gap:5px}.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
svg{width:100%;display:block;overflow:visible}svg text{fill:var(--foreground);font-family:var(--font-sans);font-size:11px}
.grid{stroke:var(--border)}.ref{stroke:var(--muted-foreground);stroke-dasharray:4 4}.band{fill:var(--secondary);opacity:.35}
.tick{fill:var(--muted-foreground)}#detail{min-height:38px;font-size:12px;line-height:1.5;color:var(--muted-foreground);margin:10px 0}
</style></head><body>
<p>Whole-process workflow throughput relative to ARC. Points show geometric ratios;
lines show paired 95% intervals. Values above 1 favor the named system. Each row is
conditional on its model, workload, resource profile and runtime. Intervals are not
adjusted for the number of comparisons; model profiles are not a controlled model-speed ranking.</p>
<div class="controls"><label>Model <select id="model"><option value="all">All selected models</option></select></label>
<span class="key"><i class="dot" style="background:var(--chart-1)"></i>Candidate</span>
<span class="key"><i class="dot" style="background:var(--chart-2)"></i>Continuum</span>
<span class="key"><i class="dot" style="background:var(--chart-3)"></i>Upstream + fix</span></div>
<svg id="chart" role="img" aria-label="Within-model throughput comparisons against ARC"></svg>
<div id="detail">Select or hover over a point for its estimate, interval and process-pair count.</div>
<p id="scope"></p><script>
const data=__DATA__,ns='http://www.w3.org/2000/svg',svg=document.getElementById('chart');
const labels={qwen3_8b_awq:'Qwen3 8B AWQ',ministral3_3b:'Ministral 3B',qwen3_4b_2507:'Qwen3 4B'},
colors={wait:'var(--chart-1)',continuum:'var(--chart-2)',upstream:'var(--chart-3)'},names={wait:'Candidate',continuum:'Continuum',upstream:'Upstream + fix'};
for(const model of [...new Set(data.points.map(p=>p.model))]){let o=document.createElement('option');o.value=model;o.textContent=labels[model]||model;document.getElementById('model').append(o)}
function el(tag,attrs,text){let e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs||{}))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e}
function draw(){svg.replaceChildren();let selected=document.getElementById('model').value,
pts=data.points.filter(p=>selected==='all'||p.model===selected),rows=[...new Set(pts.map(p=>p.model+':'+p.cell))],
left=205,right=690,top=30,rh=39,height=top+rows.length*rh+28;
svg.setAttribute('viewBox','0 0 720 '+height);svg.style.height=height+'px';
let lo=Math.min(.95,...pts.map(p=>p.interval?p.interval[0]:p.ratio)),hi=Math.max(1.05,...pts.map(p=>p.interval?p.interval[1]:p.ratio));
let pad=(hi-lo)*.08;lo=Math.max(0,lo-pad);hi+=pad;let x=v=>left+(v-lo)/(hi-lo)*(right-left);
for(let i=0;i<=4;i++){let v=lo+(hi-lo)*i/4;el('line',{x1:x(v),x2:x(v),y1:20,y2:height-28,class:'grid'});el('text',{x:x(v),y:height-8,'text-anchor':'middle',class:'tick'},v.toFixed(2)+'×')}
el('line',{x1:x(1),x2:x(1),y1:18,y2:height-28,class:'ref'});
rows.forEach((key,i)=>{let [m,c]=key.split(':'),y=top+i*rh;if(i%2===0)el('rect',{x:0,y:y-15,width:720,height:rh,class:'band'});
el('text',{x:0,y:y-2},(labels[m]||m)+' · '+c);el('text',{x:0,y:y+12,class:'tick'},data.workloads[c]);
let members=pts.filter(p=>p.model===m&&p.cell===c);members.forEach((p,j)=>{let py=y+(j-(members.length-1)/2)*9,color=colors[p.system];
if(p.interval)el('line',{x1:x(p.interval[0]),x2:x(p.interval[1]),y1:py,y2:py,stroke:color,'stroke-width':2});
let marker=el('circle',{cx:x(p.ratio),cy:py,r:4,fill:color,tabindex:0,role:'button','aria-label':names[p.system]+' '+c});
let message=(labels[m]||m)+' / '+c+' / '+names[p.system]+': '+p.ratio.toFixed(3)+'× ('+((p.ratio-1)*100).toFixed(1)+'% throughput), '+(p.interval?'95% interval '+p.interval.map(v=>v.toFixed(3)).join('–')+'×':'interval unavailable')+'; '+p.n+' process pairs; '+p.status+'.';
let title=document.createElementNS(ns,'title');title.textContent=message;marker.append(title);
for(const event of ['mouseenter','focus','click'])marker.addEventListener(event,()=>document.getElementById('detail').textContent=message)
})});}
document.getElementById('model').addEventListener('change',draw);
document.getElementById('scope').textContent=data.complete?'The campaign completion gate passed. RTX 5080 native execution remains a separate required acquisition.':'Campaign completion is pending. These retained measurements alone do not establish readiness for RTX 5080 confirmation.';
draw();</script></body></html>'''.replace('__DATA__',data)
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(document)
    return dict(html=str(destination),comparisons=len(points),campaign_complete=report['complete'])


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.report,args.out),indent=2))
