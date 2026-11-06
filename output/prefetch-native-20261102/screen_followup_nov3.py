import json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
candidate=['--arm','candidate','--no-audit','--source-label','shared-lookup','--gpu-blocks','2048']
native=['--arm','arc','--native-arc','--gpu-blocks','2112']
cases=[
 ('support-fast-graph',candidate+['--graphs','--active','8']),
 ('support-native-graph',native+['--graphs','--active','8']),
 ('support-native-graph-a16',native+['--graphs','--active','16']),
 ('support-fast-graph-a16',candidate+['--graphs','--active','16']),
 ('support-fast-graph-a16-cpu1024',candidate+['--graphs','--active','16','--cpu-blocks','1024']),
 ('support-native-graph-a16-cpu1024',native+['--graphs','--active','16','--cpu-blocks','1024']),
 ('support-fast-eager-a16-cpu1024',candidate+['--active','16','--cpu-blocks','1024']),
 ('support-native-eager-a16-cpu1024',native+['--active','16','--cpu-blocks','1024']),
 ('support-fast-graph-gpu1024-cpu1024',candidate+['--graphs','--active','8','--cpu-blocks','1024','--gpu-blocks','1024']),
 ('support-native-graph-gpu1024-cpu1024',native+['--graphs','--active','8','--cpu-blocks','1024','--gpu-blocks','1024']),
 ('support-fast-graph-a16-cpu1024-h1000',candidate+['--graphs','--active','16','--cpu-blocks','1024','--hint-interval-ms','1000']),
 ('support-native-graph-matched2048-cpu1024',native+['--graphs','--active','16','--cpu-blocks','1024','--gpu-blocks','2048']),
]
results=[];graphs=True
for name,options in cases:
 if '--graphs' in options and not graphs:continue
 command=[sys.executable,str(H/'run_live_nov3.py'),'--name','live-nov3-'+name,'--source-dir',str(H/'sources/nov3-hardened'),'--support-profile','--clients','24','--cohort','chat','--seed','1103','--turns','3','--wait','bursty','--timeout','1800','--startup-timeout','600']+options
 print('START '+name,flush=True);code=subprocess.run(command).returncode
 results.append(dict(name=name,command=command,exit_code=code))
 (H/'live-nov3-followup-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 subprocess.run([sys.executable,str(H/'analyze_live_nov3.py')])
 if code and '--graphs' in options:graphs=False
