import copy,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
base=json.loads((H/'TAIL-FIRST.json').read_text())

def pin(plan,runner,workload):
 plan['pins']={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for d in ['sources','runtime','runtime-shared-lookup','runtime-partial-native','hooks-defer','prepared'] for f in (H/d).rglob('*') if f.is_file() and f.suffix in ['.py','.json']}
 for n in ['frontier.py',runner,workload,'runtime_support.py','campaign_metrics.py','readmission_observer.py','lookup_deferral_observer.py','byte_audit.py','model-selection.json']:
  plan['pins'][n]=hashlib.sha256((H/n).read_bytes()).hexdigest()

p=copy.deepcopy(base);p['runs']=[]
for arm,bind in [('candidate',False),('arc',False),('candidate',True)]:
 r=copy.deepcopy(next(r for r in base['runs'] if r['arm']==arm and r['gpu_blocks']==768))
 r['name']=f'partial-{("bind" if bind else "lookup")}-b768-{arm}-s1103'
 c=r['command'];c[2]=str(H/'run_partial.py');c[c.index('--name')+1]=r['name']
 if '--publish-at-bind' in c:c.remove('--publish-at-bind')
 if bind:c.append('--publish-at-bind')
 p['runs'].append(r)
p['progress_file']='partial-progress.json';p['objective']='Contiguous fenced shorter-prefix publication; strict full-source identity and existing proofs unchanged; separate allocation epoch witnesses credit only actual native-hit physical pages.'
pin(p,'run_partial.py','session_workload.py');(H/'PARTIAL.json').write_text(json.dumps(p,indent=2)+'\n')

s=(H/'session_workload.py').read_text();assert s.count("state['changed'].wait(),.2")==1
s=s.replace("state['changed'].wait(),.2","state['changed'].wait(),1.0").replace("'policy_refresh_ms':200","'policy_refresh_ms':1000")
(H/'cadence_workload.py').write_text(s)
s=(H/'run_frontier_defer.py').read_text().replace('from session_workload import','from cadence_workload import')
(H/'run_cadence.py').write_text(s)
p=copy.deepcopy(base);p['runs']=[]
for arm in ['candidate','arc']:
 r=copy.deepcopy(next(r for r in base['runs'] if r['arm']==arm and r['gpu_blocks']==2048));r['name']=f'cadence-b2048-{arm}-s1103';r['prefetch_fraction']=.6
 c=r['command'];c[2]=str(H/'run_cadence.py');c[c.index('--name')+1]=r['name'];c[c.index('--prefetch-fraction')+1]='.6'
 if '--publish-at-bind' in c:c.remove('--publish-at-bind')
 p['runs'].append(r)
p['progress_file']='cadence-progress.json';p['objective']='Reduce completed-source hint RPC refresh from 200ms to 1000ms, first hint still immediate; unchanged 750ms causal estimate, private waits excluded from producer; full-prefix semantic scorer and default60% cap retained.'
pin(p,'run_cadence.py','cadence_workload.py');(H/'CADENCE.json').write_text(json.dumps(p,indent=2)+'\n')
print('Prepared two serial follow-up screens')
