from pathlib import Path
import json,hashlib,datetime,xml.etree.ElementTree as ET
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=ET.parse(H/'live-nov3-gpu-coverage-tests.xml').getroot()[0];assert int(t.attrib['tests'])==146 and int(t.attrib['failures'])==int(t.attrib['errors'])==0
base=json.loads((H/'promotion-discovery-frozen-plan-nov3.json').read_text());m={'schema':'all-valid-hint-GPU-prefix-coverage-diagnostic-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exactly3 observation-only invocations on spent7703/8803/9901. Not performance eligible. Same prior CPU recency policy, no bridge or GPU queue mutation. Snapshot coverage cannot prove future survival, demand hits, avoided bytes or latency benefit. No replacements or output cap.','python':base['python'],'runner':str((H/'run_live_gpu_coverage_nov3.py').relative_to(R)),'execution_order':[],'file_pins':{},'stop_conditions':base['stop_conditions']}
for i,seed in enumerate([7703,8803,9901]):
 prior=next(x for x in base['execution_order'] if x['partition']==seed and x['arm']=='periodic');args=prior['args'].copy();name=f'live-nov3-gpu-coverage-s{seed}'
 args[args.index('--name')+1]=name;args[args.index('--source-dir')+1]=str((H/'sources/nov3-gpu-coverage').relative_to(R));args+=['--observe-gpu-prefix']
 m['execution_order'].append({'name':name,'partition':seed,'arm':'candidate','kind':'observation_only_all_valid_hints','args':args})
paths={R/p for p in base['file_pins']};paths.update((H/'sources/nov3-gpu-coverage/semantic_offload').glob('*.py'))
for n in ['run_live_gpu_coverage_nov3.py','prepare_gpu_coverage_nov3.py','test_gpu_coverage_nov3.py','validate_gpu_coverage_nov3.py','live-nov3-gpu-coverage-tests.xml',Path(__file__).name]:paths.add(H/n)
for p in sorted(paths):
 assert p.is_file(),p
 m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['CPU_controls']={'passed':146,'mutation':'Native queue order, references, hashes and links exactly unchanged in valid/invalid/recycled/foreign-state controls; causal gates precede observation. Full CPU-ready case still observed.','performance_eligible':False}
p=H/'gpu-coverage-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'runs':len(m['execution_order']),'pins':len(m['file_pins'])}))
