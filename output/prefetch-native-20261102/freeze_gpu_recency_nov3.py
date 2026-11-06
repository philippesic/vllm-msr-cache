from pathlib import Path
import json,hashlib,datetime,xml.etree.ElementTree as ET
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=ET.parse(H/'live-nov3-gpu-recency-broad-tests.xml').getroot()[0];assert int(t.attrib['tests'])==625 and int(t.attrib['failures'])==int(t.attrib['errors'])==0
base=json.loads((H/'gpu-coverage-frozen-plan-nov3.json').read_text());pbase=json.loads((H/'promotion-discovery-frozen-plan-nov3.json').read_text())
m={'schema':'native-free-GPU-prefix-recency-discovery-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exactly9 invocations on spent7703/8803/9901, balanced3-arm Latinorder. Native versus CPUadvice versus CPU+freeGPUevictionadvice. Same geometry/h1000/naturalEOS/commonHTTP. All outcomes retained, no replacements/outputcap/inferentialCI/productionclaim. Operational candidate queuewalk/lookup costs remain in its timing; no dummy control work.','python':base['python'],'runner':str((H/'run_live_gpu_recency_nov3.py').relative_to(R)),'pairs':[],'execution_order':[],'file_pins':{},'stop_conditions':base['stop_conditions']}
arms=['native','cpu','gpu']
for i,seed in enumerate([7703,8803,9901]):
 order=arms[i:]+arms[:i];group={'pair':i,'partition':seed,'order':order,'runs':{}}
 for arm in order:
  prior=next(x for x in pbase['execution_order'] if x['partition']==seed and x['arm']==('native' if arm=='native' else 'periodic'));args=prior['args'].copy();name=f'live-nov3-gpu-recency-s{seed}-{arm}';args[args.index('--name')+1]=name
  if arm=='gpu':args[args.index('--source-dir')+1]=str((H/'sources/nov3-gpu-recency').relative_to(R));args+=['--gpu-recency']
  group['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':i,'partition':seed,'arm':arm,'kind':'native_free_GPU_eviction_advice_discovery','args':args})
 m['pairs'].append(group)
paths={R/p for p in base['file_pins']};paths.update((H/'sources/nov3-gpu-recency/semantic_offload').glob('*.py'))
for n in ['prepare_gpu_recency_nov3.py','run_live_gpu_recency_nov3.py','test_gpu_recency_nov3.py','validate_gpu_recency_nov3.py','validate_gpu_recency_broad_nov3.py','live-nov3-gpu-recency-broad-tests.xml',Path(__file__).name]:paths.add(H/n)
for p in sorted(paths):
 assert p.is_file(),p
 m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['CPU_qualification']={'passed':625,'seconds':float(t.attrib['time']),'frozen_GPU_source_files':len(list((H/'sources/nov3-gpu-recency/semantic_offload').glob('*.py'))),'controls':'Queue/hash/ref/free-capacity invariants; noallocation/touch; allselectedvalidationatomicfailure; active/duplicate/recycled/unreachablemiddle controls; causalguard/replay/boundedstate/qualifiedlayout; existingownership/completedtranscript/prefetch/residency/summaries/scoring suites under common pinned BlockPool overlay.'}
m['limits']=['No GPU byte qualification of this new eviction policy yet; required before performance confirmation or production endorsement.','Snapshot coverage and refresh operation counts are not demand hits, unique saved bytes, or proof of benefit.','Eviction preference necessarily ages other free cached prefixes; fairness/tail effects must be measured.']
p=H/'gpu-recency-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'runs':len(m['execution_order']),'pins':len(m['file_pins']),'tests':625}))
