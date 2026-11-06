from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent;R=H.parents[1];E=R/'evidence/live-optimization-2080ti-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((E/'gpu-gate-confirmation-frozen.json').read_text());q=json.loads((H/'live-nov3-gate-once-cpu-qualification.json').read_text());assert len(q['cases'])==8 and all(x['passed'] for x in q['cases'])
for p,h in b['file_pins'].items():assert sha(R/p)==h,p
m={k:b[k] for k in ['python','stop_conditions','measurement_contract']};m.update(schema='GPU-gated-first-actual-CPU-touch-discovery-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Exactly18invocations: native/periodicGPUgate/oncepositiveGPUgate across ALL6spent24session traces. Six permutations balance position/order globally; one sample per arm per trace, discoveryonly, no inferentialCI/replacement/favorable stopping/profile change/productionclaim.',runner=str((H/'run_live_gate_once_nov3.py').relative_to(R)),pairs=[],execution_order=[],file_pins={})
orders=[['native','periodic','once_positive'],['periodic','once_positive','native'],['once_positive','native','periodic'],['once_positive','periodic','native'],['native','once_positive','periodic'],['periodic','native','once_positive']]
for i,(seed,order) in enumerate(zip([7703,8803,9901,11003,12007,13001],orders)):
 g={'pair':i,'partition':seed,'order':order,'runs':{}}
 for arm in order:
  prior=next(x for x in b['execution_order'] if x['arm']==('native' if arm=='native' else 'candidate'));a=prior['args'].copy();name=f'live-nov3-gate-once-s{seed}-{arm}';a[a.index('--name')+1]=name;a[a.index('--seed')+1]=str(seed)
  corpus=H/('fresh-agent-nov3' if seed<10000 else 'fresh-agent-gate-nov3')/'session-corpus.json';a[a.index('--corpus-path')+1]=str(corpus.relative_to(R))
  if arm=='once_positive':a+=['--stop-after-first-cpu-touch']
  assert not (H/'runs'/name).exists() and '--audit' not in a
  g['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':i,'partition':seed,'arm':arm,'kind':'spent_input_GPU_gate_first_actual_positive_touch_discovery','args':a})
 m['pairs'].append(g)
paths={R/p for p in b['file_pins']}
for n in ['prepare_gate_once_nov3.py','run_live_gate_once_nov3.py','qualify_gate_once_nov3.py','live-nov3-gate-once-cpu-qualification.json','live_once_service_workload_nov3.py','fresh-agent-nov3/session-corpus.json',Path(__file__).name]:paths.add(H/n)
for p in sorted(paths):m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['measurement_contract']=dict(m['measurement_contract']);m['measurement_contract']['client']='force_closeTrue/noPOSTretry. Oncepositiveproducer alone stops after serverACK actualready_cpu_prefix_keys>0 with correctstatus/reason; fullGPUskipACK0 neverstops. Producer seesACK/completedprefix/pendingflag only, notfuturewait or message. No new persistentserverstate. This reducesHTTPandactualCPUadvice; those costs remain measured.'
m['policy_risk']='Stopping afterfirstpartialCPUtouch may miss later readiness growth, further GPUeviction or CPUreprotection. This is an evaluated risk, not a cache-survival guarantee.'
m['analysis_plan']={'comparison':'Everyvariant versus same-blocknative ANDperiodicGate perseed; all18 outcomes, no traces discarded. Wholeworkflow/tokenrate/tokenGiB/mean&p95TTFT/E2E/startup&servingGPU/hostprocessRSS/swap/CPUcacheconfiguredbytes/preemptions/nativecopydeltas reported. No inferentialCI fromsinglepairpertrace.','mechanism':'FreshvalidACKs/fullskips/actualCPUrefreshes; oncepositive everyproducerstop mustbindpositivepartialGPUtouch and no laterhintfor sameproducer. Reconstruct actualgeneratedprefixcontiguity. Sameinputsandprivatewaitsacrossarms, no artificialmatch of subsequentoutputs.','quality':'Knownsource-readingandformatdefectsremain; preserveallgeneratedoutputs. Reviewfirst4slots/turn0and2 ofeacharm,plusallformatviolations fornewoncearm; noqualityequivalenceclaim.','eligibility':'All18×72 requestsnaturalEOSwithHTTP/cleanup/sourcepins; no favorable subset. A robustnewcandidatewouldneed separatefreshconfirmation/endurance beforeproduction.'}
p=H/'gate-once-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'invocations':len(m['execution_order']),'pins':len(m['file_pins']),'CPU_cases':len(q['cases'])}))
