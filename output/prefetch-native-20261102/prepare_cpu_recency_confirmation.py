"""Freeze a larger fresh confirmation after screening; no optional stopping."""
from pathlib import Path
import argparse,hashlib,json
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--gpu-blocks',type=int,default=2048);ap.add_argument('--arc-blocks',type=int,default=2048);a=ap.parse_args()
p=json.loads((H/'CPU-ADVICE-SCREEN.json').read_text());base=p['runs'][0];p['runs']=[];p['pairs']=[];p['progress_file']='cpu-recency-v2-confirm-progress.json'
seeds=(1103,2207,3301)
for rep,order in enumerate([(1103,2207,3301),(2207,3301,1103),(3301,1103,2207),(3301,2207,1103)],1):
 for seed in order:
  pair={'seed':seed,'repetition':rep}
  arms=('candidate','arc') if (seeds.index(seed)+rep)%2 else ('arc','candidate')
  for arm in arms:
   blocks=a.gpu_blocks if arm=='candidate' else a.arc_blocks
   r=json.loads(json.dumps(base));r.update(name=f'cpu-recency-v2-confirm-b{blocks}-{arm}-s{seed}-r{rep}',arm=arm,gpu_blocks=blocks,seed=seed)
   for option,value in [('--name',r['name']),('--arm',arm),('--source-label','shared-lookup' if arm=='candidate' else 'candidate'),('--gpu-blocks',str(blocks)),('--seed',str(seed))]:r['command'][r['command'].index(option)+1]=value
   p['runs'].append(r);pair[arm]=r['name']
  p['pairs'].append(pair)
p['primary']={'metric':'paired log(candidate complete-workflow output throughput / ARC throughput), private waits included','criterion':'All twelve fresh pairs complete/strict/exact; paired Student t 95% lower bound >0 AND exact one-sided paired sign-flip p<=.05; actual load CUDA allocated/reserved plus sampled devicewide NVML no higher for candidate in every pair. No positive early stopping. Separate actual-byte qualification, ARC frontier check and integrated-source verification remain required.','sample_size':'3 fixed public trace partitions x4fresh server repetitions =12independent workflowpairs;24servers/1728requests. Chosen before these outcomes to narrow the heterogeneous six-pair interval. Allscreening/priorconfirmations excluded.','scope':'Qwen2.5-7B-Instruct-AWQ FP16 on RTX2080Ti WSL; recorded-output serving-compute replay with normal forwards,24sessions/8active/3turns, private0/2/4/8/12s waits, publicconstant750ms hint estimate, CPU4096blocks/common7680 context. Conditional fixed-trace serving-load memory/rate inference, not unconstrained generation quality or other hardware.','secondary':'All and returning-request TTFT mean/p95, CPU work, demand H2D/D2H, evictions, and zero speculative GPU bytes; report latency regressions even when throughput passes.'}
p['hypothesis']='Exact-source causal pending hints refresh ready CPU ARC recency within their existing queues. Do not count advice as actual-hit frequency, change ghost/target/ownership, or stage GPU data. Recency-v2 selected from discoveryscreen before anyfreshconfirmation outcomes.'
p['objective']='Proven throughput/VRAM improvement; continue research if criterion fails. Canonicalroot remains unchanged until result qualified.'
for name in ['prepare_cpu_recency_confirmation.py','analyze_cpu_recency_confirmation.py']:
 p['pins'][name]=hashlib.sha256((H/name).read_bytes()).hexdigest()
(H/'CPU-RECENCY-V2-CONFIRM.json').write_text(json.dumps(p,indent=2)+'\n');print('Frozen24fresh runs/12pairs; not yet started.')
