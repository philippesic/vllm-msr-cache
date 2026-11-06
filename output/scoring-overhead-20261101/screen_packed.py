import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for phase,steps in [('packed','all'),('packedmulti','multi_token')]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/asyncpacked'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS=steps,EXPERIMENT_SCORING='asynchronous')
 subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy','semantic','--phase',phase,'--threshold','0'],env=env,check=True)
