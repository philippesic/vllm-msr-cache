import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for stride in [1,4]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/asynccpu_pinned'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='all',SEMANTIC_ASYNC_MATH='cpu',EXPERIMENT_SCORING='asynchronous',EXPERIMENT_STRIDE=str(stride),EXPERIMENT_ASYNC_WORKERS='1')
 subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy','semantic','--phase','pinned-stride'+str(stride),'--threshold','0'],env=env,check=True)
