import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for workload in ['multi','single']:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/multitoken'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='multi_token')
 subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload',workload,'--policy','semantic','--phase','multitoken','--threshold','0'],env=env,check=True)
