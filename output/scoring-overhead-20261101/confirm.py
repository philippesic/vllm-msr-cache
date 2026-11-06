import os,subprocess,sys,json,time
from pathlib import Path
H=Path(__file__).resolve().parent
import hashlib
protocol=json.loads((H/'confirmation-protocol.json').read_text())
completed=[]
for wi,workload in enumerate(['single','multi']):
 for i,seed in enumerate([1103,2207,3301]):
  order=['arc','candidate','original'] if i%2==0 else ['original','candidate','arc']
  for arm in order:
   policy='arc' if arm=='arc' else 'semantic'
   env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources'/('original' if arm=='original' else 'multitoken')),SEMANTIC_CAPTURE_BACKEND='attention' if arm=='candidate' else 'dispatch',SEMANTIC_CAPTURE_STEPS='multi_token' if arm=='candidate' else 'all',EXPERIMENT_STRIDE='1')
   source_arm='original' if arm=='original' else 'multitoken'
   for filename,pin in protocol['source_pins'][source_arm].items():
    assert hashlib.sha256((H/'sources'/source_arm/'semantic_offload'/filename).read_bytes()).hexdigest()==pin
   phase='confirm-'+arm
   active=f'{phase}-{workload}-{seed}-{policy}'
   (H/'confirmation-progress.json').write_text(json.dumps({'active':active,'completed':completed},indent=2))
   print('START',active,flush=True)
   subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload',workload,'--seed',str(seed),'--policy',policy,'--phase',phase,'--threshold','0'],env=env,check=True)
   completed.append(active)
(H/'confirmation-progress.json').write_text(json.dumps({'active':None,'completed':completed,'status':'complete'},indent=2))
