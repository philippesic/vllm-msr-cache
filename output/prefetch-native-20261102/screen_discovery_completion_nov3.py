"""Finish skipped planned cases and one explicit retry; preserve original failure."""
import datetime,json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
common=['--source-dir',str(H/'sources/nov3-hardened'),'--support-profile','--clients','24','--cohort','chat','--seed','1103','--turns','3','--wait','bursty','--timeout','1800','--startup-timeout','600','--graphs']
cases=[
 ('support-fast-graph-a16-cpu1024-h1000',['--arm','candidate','--no-audit','--source-label','shared-lookup','--gpu-blocks','2048','--active','16','--cpu-blocks','1024','--hint-interval-ms','1000']),
 ('support-native-graph-matched2048-cpu1024',['--arm','arc','--native-arc','--gpu-blocks','2048','--active','16','--cpu-blocks','1024']),
 ('support-native-graph-gpu1024-cpu1024-retry1',['--arm','arc','--native-arc','--gpu-blocks','1024','--active','8','--cpu-blocks','1024']),
]
progress=[]
for name,args in cases:
 cmd=[sys.executable,str(H/'run_live_nov3.py'),'--name','live-nov3-'+name,*common,*args]
 print('START',name,flush=True);started=datetime.datetime.now(datetime.timezone.utc).isoformat();result=subprocess.run(cmd,check=False)
 progress.append({'name':name,'command':cmd,'exit_code':result.returncode,'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'one whole-workflow retry of unresolved client disconnect; original remains negative evidence' if name.endswith('retry1') else 'planned case skipped by overbroad operational graph failure heuristic'})
 (H/'live-nov3-discovery-completion-progress.json').write_text(json.dumps(progress,indent=2)+'\n');print('END',name,result.returncode,flush=True)
print('FINITE COMPLETION BLOCK COMPLETE',flush=True)
