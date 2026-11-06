"""Freeze confirmation before observing any confirmation run."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
p=json.loads((H/'CPU-SIMPLE-PILOT.json').read_text())
base=p['runs'][0];p['runs']=[];p['pairs']=[];p['progress_file']='cpu-simple-confirm-progress.json'
for rep in (1,2):
 for i,seed in enumerate((1103,2207,3301)):
  pair={'seed':seed,'repetition':rep}
  arms=('candidate','arc') if (i+rep)%2 else ('arc','candidate')
  for arm in arms:
   r=json.loads(json.dumps(base));r.update(name=f'cpu-simple-confirm-b2048-{arm}-s{seed}-r{rep}',arm=arm,seed=seed)
   c=r['command']
   for option,value in [('--name',r['name']),('--arm',arm),('--source-label','shared-lookup' if arm=='candidate' else 'candidate'),('--seed',str(seed))]:c[c.index(option)+1]=value
   p['runs'].append(r);pair[arm]=r['name']
  p['pairs'].append(pair)
p['primary']={'metric':'paired log(candidate throughput / ARC throughput), complete workflow rate including private waits','criterion':'six fresh workflow pairs; paired Student t 95% lower bound > 0 AND exact one-sided paired sign-flip p <= .05; candidate actual CUDA allocated/reserved and sampled devicewide VRAM no greater in every pair; all exact72 input/output token IDs and strict proofs, actual transfer byte qualification separately required','sample_size':'3 fixed public trace partitions x 2 fresh server repetitions; requests are not independent samples; no positive early stopping; pilots excluded','scope':'Recorded-output serving-compute replay on Qwen2.5-7B-Instruct-AWQ/RTX2080Ti, 24 sessions/8 active/3 turns/4096 CPU blocks/common7680 context; inference conditional on these fixed traces, not production generalization','secondary':'All/return mean and p95 first-token latency, actual memory and copy volume reported regardless of primary outcome. No latency win inferred from throughput.'}
p['hypothesis']='Minimal default-off CPU-only wait retention on canonical source: valid causal pending-prefix hints refresh normal CPU ARC recency; GPU speculative transfers disabled. Contiguous ready prefix variant selected before confirmation. No private future wait or future request tokens enter policy.'
p['objective']='Confirm or reject throughput/VRAM improvement; no claim at preparation or negative milestone. Continue authorized investigation if this candidate fails.'
p['pins']['prepare_cpu_simple_confirmation.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
p['pins']['analyze_cpu_simple_confirmation.py']=hashlib.sha256((H/'analyze_cpu_simple_confirmation.py').read_bytes()).hexdigest()
(H/'CPU-SIMPLE-CONFIRM.json').write_text(json.dumps(p,indent=2)+'\n')
print('Frozen twelve fresh runs; no confirmation results observed.')
