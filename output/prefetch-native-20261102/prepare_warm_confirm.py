"""Freeze a six-run confirmation, before observing any of its outcomes."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
old=(H/'run_idle_ready.py').read_text()
s=old.replace("'max_model_len':8192 if not a.qualification", "'max_model_len':7680 if not a.qualification")
s=s.replace("body={'model':'matrix-model','prompt':'An ordinary warm-up sentence.','max_tokens':8,'temperature':0}","body={'model':'matrix-model','prompt':('An ordinary neutral sentence used to warm the serving engine. ' * 100),'max_tokens':32,'temperature':0}\n   result['warmup_configuration']=body")
assert s!=old;(H/'run_warm_confirm.py').write_text(s)
d=json.loads((H/'IDLE-READY.json').read_text());base=d['runs'][0];d['runs']=[];d['progress_file']='warm-confirm-progress.json'
for seed,arms in [(1103,['candidate','arc']),(2207,['arc','candidate']),(3301,['candidate','arc'])]:
 for arm in arms:
  r=json.loads(json.dumps(base));b=512 if arm=='candidate' else 576
  r.update(name=f'warm-confirm-b{b}-{arm}-s{seed}-r1',arm=arm,gpu_blocks=b,seed=seed,prefetch_fraction=.95)
  c=list(base['command']);c=[x.replace('run_idle_ready.py','run_warm_confirm.py').replace(base['name'],r['name']) for x in c]
  for option,value in [('--arm',arm),('--source-label','shared-lookup' if arm=='candidate' else 'candidate'),('--gpu-blocks',str(b)),('--seed',str(seed)),('--prefetch-fraction','.95')]:c[c.index(option)+1]=value
  r['command']=c;d['runs'].append(r)
for f in [H/'run_warm_confirm.py',H/'prepare_warm_confirm.py']:
 d['pins'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
d['objective']='Predefined six-run confirmation across disjoint1103/2207/3301 publiccase partitions, balanced pair order. Candidate512 (fused, fixed global60s TTL, .95reclaimable stagingcap, event-driven readyidle<=50ms) versus stronger ARC576 with same native runtime and common max_model_len7680. Identical neutral1100-token/32decode warmup in both arms before stats reset; no testcontent inwarmup. Primary paired whole-workflow mean returnTTFT delta; also allrequest mean/p95, throughput, total CUDAreserved/NVML and complete copied/usefulbytes. Do not treat requests as independent replicates. Need physical proof12/12 each, work length/token inventories, bytequalification. Exploratory tuning decisions precede this plan; no claim of broadbusy24improvement. Freeze allruns; no positive earlystopping.'
(H/'WARM-CONFIRM.json').write_text(json.dumps(d,indent=2)+'\n');print('Prepared six-run frozen confirmation.')
