import os,subprocess,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent
cases=[('multitoken-thread-one','multitoken','multi_token','synchronous','gpu','1'),('cpu-thread-one-stride4','asynccpu_pinned','all','asynchronous','cpu','4'),('cpu-thread-one-stride16','asynccpu_pinned','all','asynchronous','cpu','16')]
for name,source,steps,scoring,math,stride in cases:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources'/source),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS=steps,SEMANTIC_ASYNC_MATH=math,EXPERIMENT_SCORING=scoring,EXPERIMENT_STRIDE=stride,EXPERIMENT_ASYNC_WORKERS='1',EXPERIMENT_OMP_THREADS='1')
 subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy','semantic','--phase',name,'--threshold','0'],env=env,check=True)
