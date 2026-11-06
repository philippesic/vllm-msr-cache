"""Additional disclosed screen: original 0/2/4/8/12s private wait profile."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
s=(H/'short_tool_workload.py').read_text().replace('[10000,20000,30000,45000,60000]','[0,2000,4000,8000,12000]').replace('synthetic seeded private 10/20/30/45/60 seconds','synthetic seeded private 0/2/4/8/12 seconds')
(H/'short_cadence_workload.py').write_text(s)
s=(H/'run_short_tools.py').read_text().replace('from short_tool_workload import','from short_cadence_workload import').replace("'max_model_len':7680 if not a.qualification", "'max_model_len':6144 if not a.qualification")
(H/'run_short_cadence.py').write_text(s)
d=json.loads((H/'SHORT-TOOLS-PILOT.json').read_text());d['progress_file']='short-cadence-pilot-progress.json';runs=[]
for old in d['runs']:
 r=json.loads(json.dumps(old));arm=r['arm'];b=416 if arm=='candidate' else 480;name=f'short-cadence-b{b}-{arm}-s1103';r.update(name=name,gpu_blocks=b)
 r['command']=[x.replace(old['name'],name).replace('run_short_tools.py','run_short_cadence.py') for x in old['command']];r['command'][r['command'].index('--gpu-blocks')+1]=str(b);r['command'][r['command'].index('--timeout')+1]='300';runs.append(r)
d['runs']=runs
for f in [H/'short_cadence_workload.py',H/'run_short_cadence.py',H/'prepare_short_cadence.py']:d['pins'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
d['objective']='Exploratory additional screen, no confirmatory claim. Same frozen four fresh short-observation sources and6actual-output turns as SHORT-TOOLS-PILOT, but original0/2/4/8/12s private wait distribution and smaller common6144 context (all requests must fit, no extra truncation). ARC480 versus semantic416. More eviction opportunity with finite observed source prefixes; no future return timer to policy. Full24physicalproofs each; intrusive GPU bytes separately required for any winner.'
(H/'SHORT-CADENCE-PILOT.json').write_text(json.dumps(d,indent=2)+'\n');print('Prepared, not started.')
