import json,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
cases=[
 ('chat-arc-prechange', ['--arm','arc','--source-dir',str(H/'sources/nov3-prechange'),'--allocator-audit','1','--gpu-blocks','2112']),
 ('chat-fast-pressure-2048', ['--arm','candidate','--no-audit','--source-label','shared-lookup','--gpu-blocks','2048']),
 ('chat-native-pressure-2112', ['--arm','arc','--native-arc','--gpu-blocks','2112']),
 ('chat-fast-pressure-2048-b2048', ['--arm','candidate','--no-audit','--source-label','shared-lookup','--gpu-blocks','2048','--batch-tokens','2048']),
 ('chat-native-pressure-2112-b2048', ['--arm','arc','--native-arc','--gpu-blocks','2112','--batch-tokens','2048']),
]
results=[]
for name,options in cases:
 command=[sys.executable,str(H/'run_live_nov3.py'),'--name','live-nov3-'+name,'--source-dir',str(H/'sources/nov3-fast'),'--active','8','--clients','24','--cohort','chat','--seed','1103','--turns','3','--wait','bursty','--timeout','1800']+options
 print('START '+name,flush=True)
 code=subprocess.run(command).returncode
 results.append(dict(name=name,command=command,exit_code=code))
 (H/'live-nov3-screen-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 subprocess.run([sys.executable,str(H/'analyze_live_nov3.py')])
 if code:break
