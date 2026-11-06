"""Small-summary screen, canonical-token equality certified by per-run GPU judge."""
import fcntl,hashlib,json,math,statistics
from pathlib import Path
from analyze_busy_points import point
from teacher_busy_frontier import check_pins
H=Path(__file__).resolve().parent

def certified(name,seed):
 d=json.loads((H/'runs'/name/'frontier-disposition.json').read_text());i=d['fixed_work_integrity']
 assert d['outcome']=='complete' and d['complete_load_eligible']
 assert d['planned_requests']==d['submitted_requests']==d['completed_requests']==d['completed_strict_proofs']==72
 assert i['passed'] and i['exact_recorded_output_ids'] and i['invalid_control_rejected']
 return point(name)

def main():
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=json.loads((H/'CPU-ADVICE-SCREEN.json').read_text());controls=json.loads((H/'CPU-SIMPLE-CONFIRM.json').read_text());check_pins(plan);check_pins(controls)
  progress=json.loads((H/plan['progress_file']).read_text());assert progress['status']=='complete' and len(progress['completed'])==7
  rows=[]
  for run in plan['runs']:
   seed=run['seed'];rel=f'prepared/fixed-agent-{seed}.json';assert plan['pins'][rel]==controls['pins'][rel]==hashlib.sha256((H/rel).read_bytes()).hexdigest()
   c=certified(run['name'],seed);arcs=[certified(f'cpu-simple-confirm-b2048-arc-s{seed}-r{rep}',seed) for rep in (1,2)]
   ratio=math.exp(statistics.mean(math.log(c['throughput_tokens_s']/a['throughput_tokens_s']) for a in arcs))
   mem=all(c[k]<=a[k] for a in arcs for k in ('cuda_reserved_peak_bytes','cuda_allocated_peak_bytes','nvml_devicewide_peak_bytes'))
   rows.append({'name':run['name'],'seed':seed,'candidate':c,'arc_controls':arcs,'descriptive_throughput_gain_fraction':ratio-1,'memory_no_greater_all_controls':mem,'all_mean_ttft_change_ms':c['all_mean_ttft_ms']-statistics.mean(a['all_mean_ttft_ms'] for a in arcs),'return_mean_ttft_change_ms':c['return_mean_ttft_ms']-statistics.mean(a['return_mean_ttft_ms'] for a in arcs)})
  result={'scope':'Exploratory reused-control screen, not statistical proof. Exact actual tokens certified independently per arm against the same hash-pinned canonical traces by teacher_busy_judge, including wrong controls. Private wait timers/future prompts remain outside producer.','all_rows_memory_no_greater':all(r['memory_no_greater_all_controls'] for r in rows),'rows':rows}
  (H/'cpu-advice-screen-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'scope':result['scope'],'rows':[{k:r[k] for k in ('name','descriptive_throughput_gain_fraction','memory_no_greater_all_controls','all_mean_ttft_change_ms')} for r in rows]},indent=2))
if __name__=='__main__':main()
