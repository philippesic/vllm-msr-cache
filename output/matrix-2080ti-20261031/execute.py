"""Run finite serial matrix and preserve checkpoints; no background service."""
import json, subprocess, sys, time
from pathlib import Path
H=Path(__file__).resolve().parent
def write(name,value):(H/name).write_text(json.dumps(value,indent=2)+'\n')
frozen=json.loads((H/'frozen.json').read_text());completed=[]
def cell(phase,workload,seed,policy,threshold=2):
    name=f'{phase}-{workload}-{seed}-{policy}'
    write('progress.json',{'active':name,'completed':completed,'at_ns':time.monotonic_ns()})
    print('START '+name,flush=True)
    cmd=[sys.executable,'-B',str(H/'run.py'),'run','--workload',workload,'--seed',str(seed),'--policy',policy,'--phase',phase,'--threshold',str(threshold)]
    result=subprocess.run(cmd,check=False)
    if result.returncode:
        write('progress.json',{'active':None,'failed':name,'returncode':result.returncode,'completed':completed});raise SystemExit(result.returncode)
    completed.append(name)
for pair in frozen['main_order']:
    for policy in pair['policies']:cell('main',pair['workload'],pair['seed'],policy)
subprocess.run([sys.executable,'-B',str(H/'analyze.py')],check=True,stdout=(H/'main-analysis.log').open('w'))
measurements=json.loads((H/'measurements.json').read_text())
semantic=[s for s in measurements['summaries'] if s['phase']=='main' and s['policy']=='semantic']
stress=not any(s['manager_counts'].get('cpu_evicted_blocks',0)>0 for s in semantic) or not any((s.get('prefetch_consumed_blocks') or 0)>0 for s in semantic)
write('stress-decision.json',{'required':stress,'rule':frozen['conditional_admission_stress'],'main_semantic_cpu_evictions':sum(s['manager_counts'].get('cpu_evicted_blocks',0) for s in semantic),'main_semantic_consumed_prefetch_blocks':sum(s.get('prefetch_consumed_blocks') or 0 for s in semantic)})
if stress:
    for workload in ['single','multi']:
        for policy in ['arc','semantic']:cell('stress',workload,1103,policy,0)
for workload in ['single','multi']:
    for policy in ['arc','semantic']:cell('diagnostic',workload,1103,policy)
subprocess.run([sys.executable,'-B',str(H/'analyze.py')],check=True,stdout=(H/'final-analysis.log').open('w'))
write('progress.json',{'active':None,'status':'complete','completed':completed,'at_ns':time.monotonic_ns()})
print('MATRIX COMPLETE',flush=True)
