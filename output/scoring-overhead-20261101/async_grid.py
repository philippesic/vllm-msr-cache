import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for stride in [1,4,8]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/asyncpacked'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='all',EXPERIMENT_SCORING='asynchronous',EXPERIMENT_STRIDE=str(stride),EXPERIMENT_ASYNC_WORKERS='1')
 subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy','semantic','--phase','async1-stride'+str(stride),'--threshold','0'],env=env,check=True)
