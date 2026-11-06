import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
base=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/batched'),SEMANTIC_CAPTURE_BACKEND='attention',EXPERIMENT_STRIDE='1',PYTHONPATH=str(H/'hooks')+os.pathsep+str(H.parents[1]),MATRIX_RUN_DIR=str(H/'cuda-check'))
subprocess.run([sys.executable,'-B',str(H/'cuda_check.py')],env=base,check=True)
for policy,stride,quality in [('semantic',1,'1'),('arc',1,'0')]:
 env=dict(base,EXPERIMENT_STRIDE=str(stride),MATRIX_QUALITY=quality)
 subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload','multi','--policy',policy,'--phase','diagnostic','--threshold','0'],env=env,check=True)
