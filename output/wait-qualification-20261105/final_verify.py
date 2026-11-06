"""Recheck every finished coordinator attempt; preserve sealed raw evidence."""
import json,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent;sys.path.insert(0,str(b.parents[1]))
from benchmarks.research_checks.wait_hint_qualification import offline
from benchmarks.research_checks.qualification_summary import summarize
from benchmarks.research_checks.qualification_quality import score
from benchmarks.research_checks.qualification_shape import report as shape
checked=[];errors=[]
for root in sorted(b.iterdir()):
 path=root/'result.json'
 if not root.is_dir() or not path.is_file():continue
 r=json.loads(path.read_text())
 if 'cases' not in r:continue
 try:
  assert r['status']!='running','Active campaign'
  proof=offline(root)
  reports=[('summary',summarize(root))]
  if r['plan'].get('runtime_probe'):
   waves=[dict(case=c['name'],file=str(p.relative_to(root)),**json.loads(p.read_text())) for c in r['cases'] for p in sorted((root/c['name']).glob('runtime-probe-*.json'))]
   expected=len(json.loads((root/'spec.json').read_text())['order'])*r['plan']['waves']
   reports.append(('functional-checks',dict(expected_waves=expected,observed_waves=len(waves),measurement_status=r['status'],scope='Functional diagnostic checks and native metric deltas, not application quality.',waves=waves,all_wave_checks_passed=r['status']=='passed' and len(waves)==expected and all(w['passed'] for w in waves))))
  else:reports.extend([('quality',score(root)),('shape',shape(root))])
  for suffix,value in reports:(b/(root.name+'-'+suffix+'.json')).write_text(json.dumps(value,indent=2)+'\n')
  (b/(root.name+'-verification.json')).write_text(json.dumps(dict(checked_unix=time.time(),verification=proof),indent=2)+'\n')
  checked.append(dict(campaign=root.name,measurement_status=r['status'],verification=proof));print(root.name,r['status'],'verified',flush=True)
 except Exception as exc:
  errors.append(dict(campaign=root.name,error=type(exc).__name__+': '+str(exc)));print(root.name,'VERIFY ERROR',str(exc),flush=True)
result=dict(passed=not errors,checked_unix=time.time(),verified_attempts=len(checked),attempts=checked,errors=errors,scope='Every finished immediate-child coordinator result, including failed/inconclusive attempts. Hash seals and current aggregate consistency; regenerated derived reports never alter sealed raw measurements. Example and isolation receipts, quality judgments and support decisions remain separate.')
(b/'all-retained-verification.json').write_text(json.dumps(result,indent=2)+'\n')
if errors:raise SystemExit(2)
