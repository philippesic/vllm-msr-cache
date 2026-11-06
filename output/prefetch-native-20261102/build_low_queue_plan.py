from pathlib import Path
import json,copy,hashlib
H=Path(__file__).resolve().parent;p=json.loads((H/'BIND.json').read_text());base=p['runs'];runs=[]
for budget in [768,1024]:
 for arm in ['arc','candidate']:
  r=copy.deepcopy(base[0] if arm=='arc' else base[1]);r['gpu_blocks']=budget;r['name']=f'lowqueue-b{budget}-{arm}-s1103';c=r['command'];c[c.index('--name')+1]=r['name'];c[c.index('--gpu-blocks')+1]=str(budget);c[c.index('--clients')+1]='4';c[c.index('--active')+1]='4'
  if '--publish-at-bind' in c:c.remove('--publish-at-bind')
  c[c.index('--prefetch-fraction')+1]='.3';r['prefetch_fraction']=.3;runs.append(r)
p['runs']=runs;p['progress_file']='lowqueue-progress.json';p['objective']='Additional real-public-content workload: four agent sessions and four active slots, same3turns96EOS/0..12s waits, 768/1024 pools. Judge end-to-end and returning-request TTFT, bytes and strict proofs. This broadens workload scope; never relabels prior24session failure. Investigator choice, not a new user constraint.'
p['pins']={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*') if f.is_file() and f.suffix in ['.py','.json'] and any(n in f.parts for n in ['sources','runtime','runtime-shared-lookup','hooks-defer','prepared'])}
for n in ['frontier.py','run_frontier_defer.py','runtime_support.py','session_workload.py','campaign_metrics.py','readmission_observer.py','lookup_deferral_observer.py','corpus.py','byte_audit.py','model-selection.json']:p['pins'][n]=hashlib.sha256((H/n).read_bytes()).hexdigest()
(H/'LOWQUEUE.json').write_text(json.dumps(p,indent=2)+'\n')
