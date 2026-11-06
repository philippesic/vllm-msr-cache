"""Check measured-memory ARC frontier before spending fresh confirmation runs."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
p=json.loads((H/'CPU-ADVICE-SCREEN.json').read_text());base=p['runs'][0];p['runs']=[];p['progress_file']='cpu-recency-v2-curve-progress.json'
for arm,blocks in [('arc',1792),('candidate',1792),('arc',2032),('arc',2049),('candidate',2112),('arc',2112)]:
 r=json.loads(json.dumps(base));r.update(name=f'cpu-recency-v2-curve-b{blocks}-{arm}-s1103',arm=arm,gpu_blocks=blocks,seed=1103)
 c=r['command']
 for option,value in [('--name',r['name']),('--arm',arm),('--source-label','shared-lookup' if arm=='candidate' else 'candidate'),('--gpu-blocks',str(blocks)),('--seed','1103')]:c[c.index(option)+1]=value
 p['runs'].append(r)
p['primary']='Descriptive fixed-work curves, no standalone statistical claim. All72 requests strict/exact. Measure actual load CUDA allocated/reserved and sampled NVML; nominal KV-block count alone never establishes VRAM superiority. All points use recency-v2 samepolicy or unchanged ARC, common7680 context/4096CPUblocks/8active/24sessions.'
p['hypothesis']='Read the ARC frontier around allocator boundaries2032/2048/2049 before newformalconfirmation. Three candidate budgets1792/2048/2112 and matching ARC points give descriptive throughput/TTFT vs actualservingVRAM curves;2048pilot alreadycomplete andexcludednewstatisticalconfirmation.'
p['pins']['prepare_cpu_recency_curve.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(H/'CPU-RECENCY-V2-CURVE.json').write_text(json.dumps(p,indent=2)+'\n');print('Prepared six descriptive actual-memory curve runs; not started during bytequalification.')
