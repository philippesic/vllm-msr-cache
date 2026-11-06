"""Descriptive actual-memory curves; all pilots excluded from confirmation."""
import argparse,csv,fcntl,json
from pathlib import Path
from analyze_cpu_advice_screen import certified
from teacher_busy_frontier import check_pins
H=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('plans',nargs='+');a=ap.parse_args()
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  rows=[];canonical={}
  for name in a.plans:
   p=json.loads((H/name).read_text());check_pins(p)
   progress=json.loads((H/p['progress_file']).read_text())
   assert progress['status']=='complete' and len(progress['completed'])==len(p['runs'])
   for run in p['runs']:
    seed=run['seed'];pin=p['pins'][f'prepared/fixed-agent-{seed}.json']
    assert seed not in canonical or canonical[seed]==pin
    canonical[seed]=pin
    point=certified(run['name'],seed)
    rows.append(dict(seed=seed,arm=run['arm'],gpu_blocks=run['gpu_blocks'],**point,
      output_tokens_s_per_device_GiB=point['throughput_tokens_s']/(point['nvml_devicewide_peak_bytes']/2**30)))
  assert len({r['name'] for r in rows})==len(rows)
  out={'scope':'Discovery curves from fresh finite servers, not independent statistical confirmation. Device VRAM is sampled NVML load peak; CUDA allocated and reserved peaks are also reported. Exact actual input/output token equality independently certified against identical hash-pinned canonical traces with wrong controls. Recorded-output serving-compute replay, normal model forwards; not unconstrained generation quality. CPU-only hint-guided ARC recency, no speculative GPU transfers; prefetch useful fraction undefined.','plans':a.plans,'rows':rows}
  (H/'cpu-recency-v3-curves.json').write_text(json.dumps(out,indent=2)+'\n')
  fields=['name','seed','arm','gpu_blocks','throughput_tokens_s','nvml_devicewide_peak_bytes','cuda_reserved_peak_bytes','cuda_allocated_peak_bytes','all_mean_ttft_ms','return_mean_ttft_ms','output_tokens_s_per_device_GiB','cpu_evicted_blocks','demand_h2d_bytes','d2h_bytes','copied_bytes','used_bytes','strict_proofs']
  with (H/'cpu-recency-v3-curves.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
  print(json.dumps([{k:r[k] for k in ['seed','arm','gpu_blocks','throughput_tokens_s','nvml_devicewide_peak_bytes','all_mean_ttft_ms']} for r in rows],indent=2))
if __name__=='__main__':main()
