"""Exploratory causal ablations after the initial frozen confirmation."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
p=json.loads((H/'CPU-SIMPLE-CONFIRM.json').read_text());base=p['runs'][0];p['runs']=[];p.pop('pairs');p['progress_file']='cpu-advice-screen-progress.json'
variants=[('recency-v2','run_cpu_wait_recency_v2_busy.py','cpu-wait-recency-v2'),('bounded','run_cpu_wait_bounded_busy.py','cpu-wait-bounded'),('bounded-cold','run_cpu_wait_bounded_cold_busy.py','cpu-wait-bounded-cold')]
for seed in (2207,1103,3301):
 for label,runner,source in variants:
  if seed==3301 and label!='recency-v2':continue
  r=json.loads(json.dumps(base));r.update(name=f'cpu-advice-{label}-c24-b2048-candidate-s{seed}-pilot',seed=seed);c=r['command'];c[2]=str(H/runner)
  for option,value in [('--name',r['name']),('--seed',str(seed))]:c[c.index(option)+1]=value
  p['runs'].append(r)
  for f in (H/'sources'/source).rglob('*.py'):p['pins'][str(f.relative_to(H))]=hashlib.sha256(f.read_bytes()).hexdigest()
  p['pins'][runner]=hashlib.sha256((H/runner).read_bytes()).hexdigest()
for name in ['prepare_cpu_advice_screen.py','build_cpu_wait_recency.py','build_cpu_wait_bounded.py','build_cpu_wait_cold.py','test_cpu_wait_recency_v2.py','test_cpu_wait_bounded.py','test_cpu_wait_cold.py']:
 p['pins'][name]=hashlib.sha256((H/name).read_bytes()).hexdigest()
p['primary']='Exploratory only. Compare fixed exact72-workload with fresh ARC controls from CPU-SIMPLE-CONFIRM. Every72physicalproof and actualrecordedtokenID mandatory. No pilot enters future confirmation. This is a screen, not proof from reused controls.'
p['hypothesis']='Separate over-crediting predicted demand frequency (refreshonlyARCqueueMRU), unboundedper-sourcepredictedCPUhitpressure (cap128pages), and CPU/GPUduplicateretention (skipactualGPUresidentheadbeforecappedCPUretention). Allparameters policy choices fixedbeforethese GPU outcomes; no privatefutureinformation.'
p['objective']='Continue toward proven throughput/VRAM or TTFT/VRAM improvement; run only after completing and assessing frozen initial confirmation. Defaultroot code unchanged.'
(H/'CPU-ADVICE-SCREEN.json').write_text(json.dumps(p,indent=2)+'\n');print('Prepared seven exploratory CPU-advice runs; not started.')
