"""Record fixed owner-selected margins; never drive retries or qualification."""
import json,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 r=json.loads((b/name/'result.json').read_text());s=json.loads((b/(name+'-summary.json')).read_text())
 if r['status']=='running':raise RuntimeError('Decisions require a quiet completed attempt')
 cs=s['comparisons'];primary=cs.get('wait_vs_arc');secondary={}
 def assess(c):
  v=c['candidate_batch_time_reduction_pct'];ci=c['candidate_batch_time_reduction_95ci_pct'];eligible=c['all_pairs_eligible'] and r['status']=='passed'
  return dict(comparison=c,eligible=eligible,useful_two_pct_gain_established=bool(eligible and v is not None and v>=2 and ci and ci[0]>0),neutral_two_pct_overhead_bound_established=bool(eligible and ci and ci[0]>=-2))
 for condition in ('upstream','continuum'):
  c=cs.get(condition+'_vs_wait')
  if c:
   ratio=c['workflow_rate_ratio'];ci=c['workflow_rate_ratio_95ci'];secondary[condition]=dict(comparison=c,wait_time_overhead_pct=100*(ratio-1) if ratio else None,wait_time_overhead_95ci_pct=[100*(x-1) for x in ci] if ci else None,wait_within_three_pct_overhead_bound=bool(r['status']=='passed' and c['all_pairs_eligible'] and ci and ci[1]<=1.03),wait_resolved_faster=bool(r['status']=='passed' and c['all_pairs_eligible'] and ci and ci[1]<1))
 d=dict(recorded_unix=time.time(),campaign=name,measurement_status=r['status'],primary=assess(primary) if primary else None,secondary=secondary,scope='Arithmetic assessment of predeclared owner-selected timing margins using whole-run conditional intervals. Not a task-quality, causality, product or broad-runtime qualification decision. All ARC losses require separate retained diagnosis; no automatic extension or tuning.')
 (b/(name+'-performance-assessment.json')).write_text(json.dumps(d,indent=2)+'\n');print(name,json.dumps(d))
