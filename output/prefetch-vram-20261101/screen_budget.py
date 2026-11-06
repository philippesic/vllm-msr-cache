import os,subprocess,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent
cases=[('competing-fast-mean208','opportunity','required',8,.6),('competing-cont8-both208','continuous','all',8,.99)]
(H/'budget-screen-protocol.json').write_text(json.dumps({'cases':cases,'seed':1103,'blocks':208,'source':'sources/budget','motivation':'separate futile nomination overhead from withholding one full prefix; continuous scoring must produce real numerical rankings'},indent=2)+'\n')
for phase,schedule,fields,stride,fraction in cases:
    env=dict(os.environ,EXPERIMENT_SOURCE=str(H/'sources/budget'),SEMANTIC_CAPTURE_BACKEND='attention',SEMANTIC_CAPTURE_STEPS='all',SEMANTIC_SUMMARY_FIELDS=fields,EXPERIMENT_STRIDE=str(stride),EXPERIMENT_GPU_BLOCKS='208',EXPERIMENT_SELECTOR='semantic',EXPERIMENT_COALESCING_MIN='2',EXPERIMENT_COALESCING_MS='50',EXPERIMENT_OVERLAP='1',EXPERIMENT_PREFETCH_FRACTION=str(fraction),EXPERIMENT_SCHEDULE=schedule)
    print('START',phase,flush=True)
    subprocess.run([sys.executable,'-B',str(H/'run.py'),'run','--workload','competing','--seed','1103','--policy','semantic','--phase',phase,'--threshold','0'],env=env,check=True)
