import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for phase,quality in [('asynccpu','0'),('cpu-quality','1')]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/asynccpu'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='all',SEMANTIC_ASYNC_MATH='cpu',EXPERIMENT_SCORING='asynchronous',EXPERIMENT_STRIDE='1',EXPERIMENT_ASYNC_WORKERS='1',MATRIX_CPU_QUALITY=quality)
 subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy','semantic','--phase',phase,'--threshold','0'],env=env,check=True)
