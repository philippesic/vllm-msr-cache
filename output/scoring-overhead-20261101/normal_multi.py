import os,subprocess,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent
for workload in ['multi']:
 for arm in (sys.argv[1:] or ['candidate','arc','original']):
  env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources'/('original' if arm=='original' else 'multitoken')),SEMANTIC_CAPTURE_BACKEND='attention' if arm=='candidate' else 'dispatch',SEMANTIC_CAPTURE_STEPS='multi_token' if arm=='candidate' else 'all',EXPERIMENT_STRIDE='1')
  subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload',workload,'--seed','1103','--policy','arc' if arm=='arc' else 'semantic','--phase','normal-'+arm,'--threshold','2'],env=env,check=True)
