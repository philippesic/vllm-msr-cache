import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/asynccpu_pinned'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='multi_token',SEMANTIC_ASYNC_MATH='cpu',EXPERIMENT_SCORING='asynchronous',EXPERIMENT_STRIDE='1',EXPERIMENT_ASYNC_WORKERS='1',EXPERIMENT_OMP_THREADS='1')
subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy','semantic','--phase','cpu-multitoken-thread-one','--threshold','0'],env=env,check=True)
