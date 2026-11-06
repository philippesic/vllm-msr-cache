"""Descriptive nearby memory frontier; separate from frozen primary inference."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
p=json.loads((H/'CPU-SIMPLE-CONFIRM.json').read_text());base=p['runs'][0];p['runs']=[];p.pop('pairs');p['progress_file']='cpu-simple-curve-progress.json'
for arm,b in [('arc',2032),('arc',2049),('candidate',1984),('arc',1984)]:
 r=json.loads(json.dumps(base));r.update(name=f'cpu-simple-curve-b{b}-{arm}-s1103',arm=arm,gpu_blocks=b,seed=1103)
 c=r['command']
 for option,value in [('--name',r['name']),('--arm',arm),('--source-label','shared-lookup' if arm=='candidate' else 'candidate'),('--gpu-blocks',str(b)),('--seed','1103')]:c[c.index(option)+1]=value
 p['runs'].append(r)
p['primary']='Descriptive curve points only; do not include in confirmation statistics or call single-pair gains proven. Actual allocated/reserved/NVML, throughput and TTFT; every exact72 strict proof mandatory.'
p['hypothesis']='Check ARC around CUDA allocator boundaries: nominal block counts do not define actual VRAM. A cheaper or equal-memory ARC point can invalidate a claimed frontier improvement. Include second lower-memory budget for both policies.'
p['pins']['prepare_cpu_simple_curve.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(H/'CPU-SIMPLE-CURVE.json').write_text(json.dumps(p,indent=2)+'\n');print('Prepared four descriptive nearby memory points; not run.')
