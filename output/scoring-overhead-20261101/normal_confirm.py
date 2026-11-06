import os,subprocess,sys,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
protocol=json.loads((H/'ordinary-confirmation-protocol.json').read_text())
for seed,order in zip([1103,2207,3301],[['arc','candidate','original'],['original','candidate','arc'],['arc','candidate','original']]):
 for arm in order:
  source='original' if arm=='original' else 'multitoken'
  for name,pin in protocol['source_pins'][source].items():assert hashlib.sha256((H/'sources'/source/'semantic_offload'/name).read_bytes()).hexdigest()==pin
  for name,pin in protocol['harness_pins'].items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==pin
  env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources'/source),SEMANTIC_CAPTURE_BACKEND='attention' if arm=='candidate' else 'dispatch',SEMANTIC_CAPTURE_STEPS='multi_token' if arm=='candidate' else 'all',EXPERIMENT_STRIDE='1',EXPERIMENT_SCORING='synchronous')
  print('START',seed,arm,flush=True)
  subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload','multi','--seed',str(seed),'--policy','arc' if arm=='arc' else 'semantic','--phase','normalconfirm-'+arm,'--threshold','2'],env=env,check=True)
