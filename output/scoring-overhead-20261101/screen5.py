import os,subprocess,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent
for name,policy,backend,scoring,stride in [('compact','semantic','attention','synchronous',1),('compact8','semantic','attention','synchronous',8)]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/compact'),SEMANTIC_CAPTURE_BACKEND=backend,EXPERIMENT_SCORING=scoring,EXPERIMENT_STRIDE=str(stride))
 cmd=[sys.executable,'-B',str(H/'run.py'),'run','--workload','multi','--policy',policy,'--phase',name,'--threshold','0']
 print('START',name,flush=True)
 subprocess.run(cmd,env=env,check=True)
