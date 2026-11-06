"""Present every retained wait/ARC comparison; no weighted aggregate or gate changes."""
import csv,html,json,math
from pathlib import Path
b=Path(__file__).resolve().parent
def role(name):
 if name.startswith('runtime-probe'):return 'functional diagnostic, not application timing','func'
 if 'replay' in name:return 'identical-token diagnostic, not live application timing','replay'
 if name in ('stage3-repository','stage3-repository-neutral-confirm','stage3-repository-cohost','stage3-documents','stage3-documents-cohost','stage3-documents-admission-diagnostic'):return 'development after input/admission investigation','dev'
 if name=='stage3-documents-pressure-control':return 'inactive-hint pressure diagnostic','diag'
 if 'reserve' in name:return 'fresh instruction-validation cohort; memory cells share tasks','fresh'
 if 'neutral-confirm' in name:return 'separate fixed neutral-overhead confirmation','confirm'
 if name=='stage3-soak':return 'finite repeated-task exposure, not independent repeated runs','exposure'
 if name.startswith('stage4-minimal'):return 'bounded integration-profile timing','profile'
 if name.startswith('stage4'):return 'selected compatibility-cell timing','compat'
 if name.startswith('stage3'):return 'original-workload operating regime','regime'
 if name.startswith('stage2'):return 'configured-system comparison','system'
 if name=='stage1-live':return 'live causal controls on original workload','live'
 return 'retained comparison; consult detailed evidence','other'
rows=[]
for path in sorted(b.glob('*-summary.json')):
 d=json.loads(path.read_text())
 for name,c in d.get('comparisons',{}).items():
  if name not in ('wait_vs_arc','full_vs_arc'):continue
  root=b/path.name.removesuffix('-summary.json')
  if not (root/'result.json').exists():continue
  r=json.loads((root/'result.json').read_text());counts=[x['summary']['completed'] for x in r['cases']]
  evidence_role,tag=role(root.name)
  rows.append(dict(campaign=root.name,evidence_role=evidence_role,display_tag=tag,comparison=name,status=r['status'],pairs=c['paired_runs'],time_reduction_pct=c['candidate_batch_time_reduction_pct'],ci_low_pct=(c['candidate_batch_time_reduction_95ci_pct'] or [None,None])[0],ci_high_pct=(c['candidate_batch_time_reduction_95ci_pct'] or [None,None])[1],all_pairs_eligible=c['all_pairs_eligible'],exact_prompt_output=c['identical_prompt_and_output_ids'],diagnostic_replay=d['diagnostic_replay'],total_responses=sum(counts),summary=path.name))
p=b/'all-wait-arc-comparisons.csv'
with p.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['campaign']);w.writeheader();w.writerows(rows)
(b/'all-wait-arc-comparisons.json').write_text(json.dumps(dict(rows=rows,scope='All retained derived wait/ARC comparisons, including diagnostic, development, neutral and adverse cells. No pooling or population weighting. Conditional whole-run95% intervals; answer quality and runtime limits require linked evidence.'),indent=2)+'\n')
estimated=[r for r in rows if r['time_reduction_pct'] is not None]
lo=min([-5]+[r['ci_low_pct'] if r['ci_low_pct'] is not None else r['time_reduction_pct'] for r in estimated]);hi=max([25]+[r['ci_high_pct'] if r['ci_high_pct'] is not None else r['time_reduction_pct'] for r in estimated]);lo=math.floor(lo/5)*5;hi=math.ceil(hi/5)*5
W=728;left=330;right=645;top=38;stride=30;H=top+len(rows)*stride+45
x=lambda v:left+(v-lo)/(hi-lo)*(right-left)
svg=[f'<svg role="img" aria-label="All wait-hint versus plain ARC comparisons with conditional95 percent intervals" viewBox="0 0 {W} {H}" style="width:100%;height:auto;font:11px var(--font-sans,sans-serif)">']
svg.append(f'<line x1="{x(0)}" x2="{x(0)}" y1="15" y2="{H-38}" stroke="var(--muted-foreground,#888)" stroke-dasharray="4 4"/>')
for i,r in enumerate(rows):
 y=top+i*stride;v=r['time_reduction_pct'];a=r['ci_low_pct'];z=r['ci_high_pct'];label=html.escape(r['campaign']+' ['+r['display_tag']+']'+(' ['+r['status']+']' if r['status']!='passed' else ''))
 svg.append(f'<text x="0" y="{y+4}" fill="var(--foreground,#333)">{label}</text>')
 if v is None:
  svg.append(f'<text x="{left}" y="{y+4}" fill="var(--muted-foreground,#888)">no eligible pair</text>')
  continue
 color='var(--chart-1,#69b3df)' if v>=0 else 'var(--destructive,#e97070)' 
 if a is not None:
  svg.append(f'<path d="M{x(a):.2f},{y}H{x(z):.2f} M{x(a):.2f},{y-4}V{y+4} M{x(z):.2f},{y-4}V{y+4}" fill="none" stroke="{color}" stroke-width="1.5"/>')
 svg.append(f'<circle cx="{x(v):.2f}" cy="{y}" r="3.5" fill="{color}"><title>{label}: {v:.2f}%; CI {a}, {z}; {r["pairs"]} whole-run pairs</title></circle><text x="660" y="{y+4}" fill="var(--foreground,#333)">{v:+.2f}%</text>')
for value in range(int(lo),int(hi)+1,5):svg.append(f'<text text-anchor="middle" x="{x(value):.2f}" y="{H-15}" fill="var(--muted-foreground,#999)">{value}%</text>')
svg.append('</svg>')
page='<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:var(--font-sans,sans-serif);color:var(--foreground,#222);margin:0}p{line-height:1.5;font-size:13px;color:var(--muted-foreground)}.chart{overflow:auto;max-height:1350px}svg{min-width:700px}a{color:var(--primary)}</style></head><body><p>Batch time saved by wait-hint versus plain ARC. Positive favors wait-hint. Bars show conditional95% intervals over whole-run pairs. Every retained comparison is shown; diagnostic and development runs remain labeled by campaign name.</p><div class="chart">'+''.join(svg)+'</div><p>No pooling across workloads. Measurement completion does not establish useful answer quality, production stability, or support for other runtimes/hardware. Consult the evidence index and CSV for scope and individual cases.</p></body></html>'
page=page.replace('.chart{overflow:auto;max-height:1350px}',f'.chart{{overflow:auto;height:{min(1350,H+20)}px}}')
page=page.replace('</div><p>No pooling', '</div><p>Labels: dev = development; fresh = fresh instruction-validation tasks (reused across predeclared memory cells); func/replay/diag = diagnostics; confirm = separate fixed confirmation; regime = original-workload operating condition; exposure = finite churn; compat/profile = selected integration timing. Diagnostic timings are not application speed claims.</p><p>No pooling')
(b/'all-wait-arc-comparisons.html').write_text(page)
print('Prepared',len(rows),'comparison rows and CSV/JSON/HTML')
