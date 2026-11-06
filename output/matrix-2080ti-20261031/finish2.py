import json,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
p=json.loads((H/'progress.json').read_text());p['failed_attempts']=['failed-attempts/diagnostic-multi-1103-arc','failed-attempts/diagnostic-multi-1103-semantic']
for name,cmd in [('diagnostic-multi-1103-semantic',[str(H/'diagnostic_retry.py')]),('repeatability-single-2207-arc',[str(H/'run.py'),'run','--workload','single','--seed','2207','--policy','arc','--phase','repeatability'])]:
 p.update(active=name,status='running');(H/'progress.json').write_text(json.dumps(p,indent=2)+'\n');print('START '+name,flush=True)
 r=subprocess.run([sys.executable,'-B',*cmd])
 if r.returncode:p.update(active=None,status='failed',failed=name);(H/'progress.json').write_text(json.dumps(p,indent=2)+'\n');raise SystemExit(r.returncode)
 p['completed'].append(name)
subprocess.run([sys.executable,'-B',str(H/'analyze.py')],check=True,stdout=(H/'final-analysis.log').open('w'))
p.update(active=None,status='complete',at_ns=time.monotonic_ns());p.pop('failed',None);p.pop('returncode',None);(H/'progress.json').write_text(json.dumps(p,indent=2)+'\n');print('MATRIX COMPLETE',flush=True)
