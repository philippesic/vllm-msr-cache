"""Disclosed arrival sensitivity screen, same public prompts and actual histories."""
import ast,copy,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
old=(H/'cadence_workload.py').read_text()
s=old.replace('rng.choice([0,2000,4000,8000,12000])','rng.choice([10000,20000,30000,45000,60000])').replace('synthetic seeded private 0/2/4/8/12 seconds','synthetic seeded private 10/20/30/45/60 seconds')
assert s!=old
def producer(t):return ast.dump(next(n for n in ast.parse(t).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='producer'))
assert producer(old)==producer(s),'producer must not receive private timing or future text'
(H/'spaced_tool_workload.py').write_text(s)
s=(H/'run_cadence.py').read_text().replace('from cadence_workload import','from spaced_tool_workload import')
(H/'run_spaced.py').write_text(s)
base=json.loads((H/'LOWQUEUE.json').read_text());p=copy.deepcopy(base);p['runs']=[]
for blocks,order in [(520,['arc','candidate']),(640,['candidate','arc'])]:
 for arm in order:
  r=copy.deepcopy(next(r for r in base['runs'] if r['arm']==arm and r['gpu_blocks']==520))
  r.update(name=f'spaced-b{blocks}-{arm}-s1103',gpu_blocks=blocks,prefetch_fraction=.6)
  c=r['command'];c[2]=str(H/'run_spaced.py');c[c.index('--name')+1]=r['name'];c[c.index('--gpu-blocks')+1]=str(blocks);c[c.index('--timeout')+1]='360'
  p['runs'].append(r)
p['progress_file']='spaced-progress.json'
p['objective']='Arrival sensitivity: four public agent sessions, active4, three actual-output turns, EOS/96cap, private seeded10/20/30/45/60s simulated tool waits, unchanged immediate750ms causal estimate and1000ms refresh. Primary screen metric mean return TTFT; also report all-request mean/p95 and throughput. This is explicitly separate from original busy24client0-12s workload; no broad victory from a scoped win.'
p['pins']={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for d in ['sources/shared-lookup','sources/candidate','runtime','runtime-shared-lookup','hooks-defer','prepared'] for f in (H/d).rglob('*') if f.is_file() and f.suffix in ['.py','.json']}
for n in ['frontier.py','run_spaced.py','spaced_tool_workload.py','session_workload.py','runtime_support.py','campaign_metrics.py','readmission_observer.py','lookup_deferral_observer.py','byte_audit.py','model-selection.json']:
 p['pins'][n]=hashlib.sha256((H/n).read_bytes()).hexdigest()
(H/'SPACED.json').write_text(json.dumps(p,indent=2)+'\n')
print('Prepared spaced arrival screen; unchanged policy producer AST verified')
