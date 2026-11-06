import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for name,threads,policy in [('cpu-thread-default',None,'semantic'),('cpu-thread-one','1','semantic'),('arc-thread-one','1','arc')]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/asynccpu_pinned'),SEMANTIC_CAPTURE_BACKEND='attention' if policy=='semantic' else 'dispatch',SEMANTIC_CAPTURE_STEPS='all',SEMANTIC_ASYNC_MATH='cpu',EXPERIMENT_SCORING='asynchronous' if policy=='semantic' else 'synchronous',EXPERIMENT_STRIDE='1',EXPERIMENT_ASYNC_WORKERS='1')
 if threads:env['EXPERIMENT_OMP_THREADS']=threads
 subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy',policy,'--phase',name,'--threshold','0'],env=env,check=True)
