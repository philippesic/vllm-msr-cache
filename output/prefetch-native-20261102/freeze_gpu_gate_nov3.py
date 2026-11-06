from pathlib import Path
import json,hashlib,datetime,xml.etree.ElementTree as ET
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=ET.parse(H/'live-nov3-gpu-gate-tests.xml').getroot()[0];assert int(t.attrib['tests'])==606 and int(t.attrib['failures'])==int(t.attrib['errors'])==0
base=json.loads((H/'gpu-recency-frozen-plan-nov3.json').read_text());m={k:base[k] for k in ['python','stop_conditions']};m.update(schema='GPU-state-gated-CPU-retention-discovery-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Exactly9 invocations on spent7703/8803/9901, balanced3-arm Latinorder. Native versus alwaysCPUadvice versus GPU-state-gatedCPUadvice. Same1536GPU/2048CPU/a8/h1000/publicinputs/naturalEOS/commonHTTP. No GPU mutation. All outcomes retained; no replacements/outputcap/inferentialCI/productionclaim. Candidate exactindexlookup costs stay measured; no dummy control work.',runner=str((H/'run_live_gpu_gate_nov3.py').relative_to(R)),pairs=[],execution_order=[],file_pins={})
arms=['native','cpu','gate']
for i,seed in enumerate([7703,8803,9901]):
 order=arms[i:]+arms[:i];group={'pair':i,'partition':seed,'order':order,'runs':{}}
 for arm in order:
  prior=next(x for x in base['execution_order'] if x['partition']==seed and x['arm']==('native' if arm=='native' else 'cpu'));args=prior['args'].copy();name=f'live-nov3-gpu-gate-s{seed}-{arm}';args[args.index('--name')+1]=name
  if arm=='gate':args[args.index('--source-dir')+1]=str((H/'sources/nov3-gpu-gate').relative_to(R));args+=['--gpu-gated-cpu-retention']
  group['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':i,'partition':seed,'arm':arm,'kind':'GPU_state_gated_CPU_advice_discovery','args':args})
 m['pairs'].append(group)
paths={R/p for p in base['file_pins']};paths.update((H/'sources/nov3-gpu-gate/semantic_offload').glob('*.py'))
for n in ['prepare_gpu_gate_nov3.py','run_live_gpu_gate_nov3.py','test_gpu_gate_nov3.py','validate_gpu_gate_nov3.py','live-nov3-gpu-gate-tests.xml',Path(__file__).name]:paths.add(H/n)
for p in sorted(paths):
 assert p.is_file(),p
 m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['policy_rules']={'bridge_flag':False,'observation_flag':False,'GPU_gate':'Skip CPUrefresh only when native exact contiguousGPUcoverage available ANDfullsourcecovered. No managercall then. Partial/miss usesoriginalreadyCPUroots. Unavailablefallsback originalCPUadvice andGPUcountNone. NeverresurrectevictedCPUkeys.','telemetry':'ready_cpu_prefix_keys is actualrefreshedcount; cpu_retention_ready_prefix_keys is before-actionreadyCPUpositioncount; skipcounts are repeatedoperations, not savedbytes/demandhits.'};m['CPU_qualification']={'passed':606,'seconds':float(t.attrib['time']),'scope':'586existing+20newCPUcontrols underpinnedBlockPool overlay, including nativeauditofffull→partial/read-onlyGPUstate, unavailablefallback, evictioncannotresurrect, actualreturn0versusreadypositive, causal/replay/layout/defaultoff.'};m['limits']=['Full GPU coverage may disappear before the next hint after CPUdata alreadyevicted; this is an evaluatedrisk, not a safety guarantee.','Newpolicy bytequalification/heldoutconfirmation/endurance required before productionendorsement.']
p=H/'gpu-gate-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'runs':len(m['execution_order']),'pins':len(m['file_pins']),'tests':606}))
