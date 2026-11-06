import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for name,steps,delivery,scoring in [('deferred','all','after_execute','synchronous'),('deferredmulti','multi_token','after_execute','synchronous'),('asyncmulti','multi_token','inline','asynchronous')]:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/deferred'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS=steps,SEMANTIC_CAPTURE_DELIVERY=delivery,EXPERIMENT_SCORING=scoring)
 subprocess.run([sys.executable,'-B',str(H/'run_future.py'),'run','--workload','multi','--policy','semantic','--phase',name,'--threshold','0'],env=env,check=True)
