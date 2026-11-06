"""Quiet-boundary verification and derived reports; never alters sealed runs."""
import json,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
sys.path.insert(0,str(b.parents[1]))
from benchmarks.research_checks.wait_hint_qualification import offline
from benchmarks.research_checks.qualification_summary import summarize
from benchmarks.research_checks.qualification_quality import score
from benchmarks.research_checks.qualification_shape import report as shape
for name in sys.argv[1:]:
 root=b/name;r=json.loads((root/'result.json').read_text())
 if r['status']=='running':raise RuntimeError('Campaign still running')
 checked=offline(root)
 reports=[('summary',summarize(root))]
 if r['plan'].get('runtime_probe'):
  waves=[dict(case=c['name'],file=str(p.relative_to(root)),**json.loads(p.read_text())) for c in r['cases'] for p in sorted((root/c['name']).glob('runtime-probe-*.json'))]
  expected_waves=len(json.loads((root/'spec.json').read_text())['order'])*r['plan']['waves']
  reports.append(('functional-checks',dict(expected_waves=expected_waves,observed_waves=len(waves),measurement_status=r['status'],scope='Functional diagnostic checks and scoped per-request native metric deltas; generic turn3 application-answer grading does not apply.',waves=waves,all_wave_checks_passed=r['status']=='passed' and len(waves)==expected_waves and all(w['passed'] for w in waves))))
 else:reports.extend([('quality',score(root)),('shape',shape(root))])
 for suffix,value in reports:
  (b/f'{name}-{suffix}.json').write_text(json.dumps(value,indent=2)+'\n')
 (b/f'{name}-verification.json').write_text(json.dumps(dict(checked_unix=time.time(),verification=checked),indent=2)+'\n')
 print(name,r['status'],len(r['cases']),json.dumps(summarize(root)['comparisons']),flush=True)
