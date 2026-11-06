import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
cases=[('arc-full','qualify','arc',640,False),('arc-small','qualify','arc',320,False),('splice-small','qualify','semantic',320,False),('native-small','qualify','semantic',320,True),('causal-native-small','causal','semantic',320,True),('causal-arc-full','causal','arc',640,False),('causal-arc-small','causal','arc',320,False)]
for name,workload,policy,blocks,adoption in cases:
 env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/full'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='multi_token',EXPERIMENT_GPU_BLOCKS=str(blocks),EXPERIMENT_ADOPTION=str(int(adoption)),EXPERIMENT_COALESCE='1',EXPERIMENT_FULL_ONLY='1')
 print('START',name,flush=True)
 subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload',workload,'--seed','1103','--policy',policy,'--phase',name,'--threshold','0'],env=env,check=True)
