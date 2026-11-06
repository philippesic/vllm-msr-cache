from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent;R=H.parents[1];E=R/'evidence/live-optimization-2080ti-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((H/'return-priority-frozen-plan-nov3.json').read_text())
for p,h in b['file_pins'].items():assert sha(R/p)==h,p
m={k:b[k] for k in ['python','runner','stop_conditions','measurement_contract']};m.update(schema='exact8s-priority-both-arm-byte-qualification-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Exactly2intrusiveinvocations onspent7703, each24sessions3turns72naturalEOS. Exactnative8s andGPUgate8s configurations selectedafterdiscovery, notperformance/quality/endurance. PreservationauditFalse keptcandidate; independentREAL_KV_BYTE_AUDITTruebotharms. Allmappedpagecomponents+onebytecomparatornegativecontrols/nativeaggregatefinalandloadreconciliationrequired.',pairs=[{'pair':0,'partition':7703,'order':['native','candidate'],'runs':{}}],execution_order=[],file_pins={})
for arm,prior_arm in [('native','native8000'),('candidate','gate8000')]:
 prior=next(x for x in b['execution_order'] if x['partition']==7703 and x['arm']==prior_arm);a=prior['args'].copy();name='live-nov3-return-priority-byte-'+arm;a[a.index('--name')+1]=name;a+=['--audit'];assert '--no-audit' in a if arm=='candidate' else '--no-audit' not in a
 assert not (H/'runs'/name).exists()
 m['pairs'][0]['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':0,'partition':7703,'arm':arm,'priority_credit_ms':8000,'gate_enabled':arm=='candidate','kind':'intrusive_priority_path_byte_qualification','args':a})
paths={R/p for p in b['file_pins']}
for p in [Path(__file__),E/'return-priority-root-verification.json',H/'root_verify_return_priority_nov3.py',H/'verify_integrated_gate_bytes_astra_nov3.py',H/'verify_selected_bytes_astra_nov3.py',H/'gpu_gate_receipt_astra_nov3.py',H/'http_confirmation_receipt_astra_nov3.py']:paths.add(p)
for p in sorted(paths):m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['required_receipts']=['All407priorpins+newqualificationpinsbeforeeach/end;exact2namesabsentbeforelaunch;alloutcomesretained.','Actualprioritypayloadformula/commonnativepriorityschedulerverified;byteauditingTrue/performanceeligibleFalsebotharms.','CandidateboundedpreservationFalse/fullGPUskippositiveCPUreadyandpartialGPUactualCPUtouchobserved.','MappedD2H/H2D28components×32768bytes/page/everyrowactualequalityandonebytecomparatorexecution;nativefinalandloadcount/bytesreconciled. NoindependentperjobTransfer.num_bytesfieldclaim.','Both72actualnaturalEOSremainingcontextonly,noHTTPerrors/retries/cleanupfault;allownedprocessesreapedandGPU/lockidle.','No freshmodelgenerationinthisphase; anycorpusacquisitionmustretainexclusion/prospectivepathselectionproof.']
p=H/'return-priority-bytes-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'runs':2,'pins':len(m['file_pins'])}))
