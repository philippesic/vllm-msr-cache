import json, subprocess, sys, time
from pathlib import Path
H=Path(__file__).resolve().parent
p=json.loads((H/'progress.json').read_text());completed=p['completed'];p['failed_attempt']='failed-attempts/diagnostic-multi-1103-arc'
for phase,w,seed,policy in [('diagnostic','multi',1103,'arc'),('diagnostic','multi',1103,'semantic'),('repeatability','single',2207,'arc')]:
 name=f'{phase}-{w}-{seed}-{policy}';p.update(active=name,status='running');(H/'progress.json').write_text(json.dumps(p,indent=2)+'\n');print('START '+name,flush=True)
 r=subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload',w,'--seed',str(seed),'--policy',policy,'--phase',phase])
 if r.returncode:p.update(active=None,status='failed',failed=name);(H/'progress.json').write_text(json.dumps(p,indent=2)+'\n');raise SystemExit(r.returncode)
 completed.append(name)
subprocess.run([sys.executable,'-B',str(H/'analyze.py')],check=True,stdout=(H/'final-analysis.log').open('w'))
p.update(active=None,status='complete',at_ns=time.monotonic_ns());(H/'progress.json').write_text(json.dumps(p,indent=2)+'\n');print('MATRIX COMPLETE',flush=True)
