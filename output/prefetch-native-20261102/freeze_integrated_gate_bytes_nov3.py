from pathlib import Path
import json,hashlib,datetime,shutil,xml.etree.ElementTree as ET
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=ET.parse(H/'live-nov3-integrated-gpu-gate-tests.xml').getroot()[0];assert int(t.attrib['tests'])==607 and int(t.attrib['failures'])==int(t.attrib['errors'])==0
D=H/'sources/nov3-gpu-gate-integrated/semantic_offload';D.mkdir(parents=True,exist_ok=False)
for p in (R/'semantic_offload').glob('*.py'):shutil.copyfile(p,D/p.name);assert sha(p)==sha(D/p.name)
base=json.loads((H/'gpu-gate-frozen-plan-nov3.json').read_text());m={'schema':'integrated-GPU-gated-CPU-both-arm-byte-qualification-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exactly2 intrusive audit invocations on spent9901. No performance claim; compare mapped native D2H/H2D page bytes, run one-byte comparator controls, reconcile native aggregate byte/job counters. Same selected geometry/source/runtime as prospective fresh confirmation.','python':base['python'],'runner':str((H/'run_live_integrated_gpu_gate_nov3.py').relative_to(R)),'pairs':[],'execution_order':[],'file_pins':{},'stop_conditions':base['stop_conditions']}
group={'pair':0,'partition':9901,'order':['native','candidate'],'runs':{}}
for arm in group['order']:
 prior=next(x for x in base['execution_order'] if x['partition']==9901 and x['arm']==('native' if arm=='native' else 'cpu'));args=prior['args'].copy();name=f'live-nov3-integrated-gate-byte-{arm}';args[args.index('--name')+1]=name
 if '--no-audit' in args:args.remove('--no-audit')
 if arm=='candidate':args[args.index('--source-dir')+1]=str(D.parent.relative_to(R));args+=['--gpu-gated-cpu-retention']
 args+=['--audit'];group['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':0,'partition':9901,'arm':arm,'kind':'intrusive_both_arm_byte_qualification','args':args})
m['pairs'].append(group)
paths={R/p for p in base['file_pins']};paths.update(D.glob('*.py'));paths.update((R/'semantic_offload').glob('*.py'))
for p in [R/'tests/test_gpu_gated_cpu_retention.py',R/'README.md',H/'run_live_integrated_gpu_gate_nov3.py',H/'validate_integrated_gpu_gate_nov3.py',H/'live-nov3-integrated-gpu-gate-tests.xml',Path(__file__)]:paths.add(p)
for p in sorted(paths):
 assert p.is_file(),p
 m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['integrated_source']={'path':str(D.parent.relative_to(R)),'files':len(list(D.glob('*.py'))),'exact_ROOT_copy':True,'API':'wait_hint_cpu_retention_gpu_gate defaultFalse; strictly qualifiedlayout; bridge+gate rejected; no observation-only/GPUmutation modules integrated.','CPU_tests':607,'CPU_test_seconds':float(t.attrib['time'])};m['required_qualification']=['All144requests naturalEOS with successfulHTTP/nohiddenretries.','Atleastone actualD2HandH2Dauditedjobperarm;every28component bytecomparison andone-byte negativecontrolpass.','Reconcile all auditedmapped-pagebytes andjobcounts to nativefinal andload-deltaPrometheuscounts; do notclaim independent perjobTransfer.num_bytes observation.','Candidatevalid nonreplayedGPUgatingACKs exercised withactualfullGPUskips andpartial/missfallback; noGPUmutation.','Allsource/runtime/model/harnesspins held; noownedprocessesleft; logscompressedlosslessly.']
p=H/'integrated-gate-bytes-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'runs':2,'pins':len(m['file_pins']),'sourcefiles':m['integrated_source']['files'],'tests':607}))
