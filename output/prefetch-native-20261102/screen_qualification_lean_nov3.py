"""Three finite correctness/configuration checks after frozen discovery exits."""
import datetime,json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
common=['--source-dir',str(H/'sources/nov3-hardened'),'--support-profile','--clients','8','--active','8','--seed','1103','--turns','2','--wait','bursty','--timeout','1800','--startup-timeout','600']
cases=[
 ('intrusive-hardened-gpu512','run_live_nov3.py',['--arm','candidate','--no-audit','--source-label','shared-lookup','--audit','--cohort','agent','--gpu-blocks','512','--cpu-blocks','4096']),
 ('lean-normalized-native-smoke','run_live_lean_nov3.py',['--arm','arc','--native-arc','--cohort','chat','--gpu-blocks','1024','--cpu-blocks','1024','--graphs']),
 ('lean-candidate-smoke','run_live_lean_nov3.py',['--arm','candidate','--no-audit','--source-label','shared-lookup','--cohort','chat','--gpu-blocks','1024','--cpu-blocks','1024','--graphs']),
]
progress=[]
for name,driver,args in cases:
 cmd=[sys.executable,str(H/driver),'--name','live-nov3-'+name,*common,*args]
 print('START',name,flush=True)
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 result=subprocess.run(cmd,cwd=str(H.parents[1]),check=False)
 progress.append({'name':name,'command':cmd,'exit_code':result.returncode,'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'intrusive correctness only' if '--audit' in args else 'lean executed configuration smoke, not statistical confirmation'})
 (H/'live-nov3-qualification-lean-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
 print('END',name,result.returncode,flush=True)
print('FINITE QUALIFICATION BLOCK COMPLETE',flush=True)
