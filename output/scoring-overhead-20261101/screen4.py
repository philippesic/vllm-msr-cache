import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for name,source,stride in [('batched32','batched',32),('batchedgemm','batchedgemm',1)]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources'/source),SEMANTIC_CAPTURE_BACKEND='attention',EXPERIMENT_STRIDE=str(stride))
 subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload','multi','--policy','semantic','--phase',name,'--threshold','0'],env=env,check=True)
