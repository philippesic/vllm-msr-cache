import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/multitoken'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='all',MATRIX_QUALITY='1')
subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload','multi','--policy','semantic','--phase','quality-final','--threshold','0'],env=env,check=True)
