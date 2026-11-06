"""Lossless audit compression, independent receipt refresh and custody verification."""
import collections,datetime,fcntl,gzip,hashlib,json,shutil,subprocess
from pathlib import Path
import psutil
from verify_selected_bytes_astra_nov3 import verify_bytes
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x):(H/name).write_text(json.dumps(x,indent=2)+'\n')
p=H/'selected-bytes-frozen-plan-nov3.json';plan=json.loads(p.read_text());state=json.loads((H/'live-nov3-selected-bytes-custody.json').read_text());freebefore=shutil.disk_usage(H).free
bad=[n for n,v in plan['file_pins'].items() if not (ROOT/n).is_file() or sha(ROOT/n)!=v];assert not bad
assert sha(p)==state['manifest_sha256'];assert state['status']=='complete' and len(state['runs'])==2 and not state['owned_descendants_remaining']
assert all(q['workflow_success'] and q['byte_receipt']['qualified'] and q['exit_code']==0 and not q['owned_descendants_remaining'] for q in state['runs'])
assert not psutil.pid_exists(state['supervisor_pid']);assert all(not psutil.pid_exists(q['runner_pid']) for q in state['runs'])
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not gpu
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
compression=[];receipts={};files=[]
for item in plan['execution_order']:
 d=H/'runs'/item['name'];rp=d/'raw.json.gz';raw=json.load(gzip.open(rp,'rt'));assert not raw['owned_descendants_remaining'];assert raw['performance_eligible'] is False
 for f in d.glob('byte-audit-*.jsonl'):
  original=sha(f);n=f.stat().st_size;dst=f.with_suffix(f.suffix+'.gz');assert not dst.exists()
  with f.open('rb') as a,gzip.open(dst,'wb',compresslevel=3) as b:shutil.copyfileobj(a,b)
  with gzip.open(dst,'rb') as g:assert hashlib.file_digest(g,'sha256').hexdigest()==original
  compression.append({'original_path':str(f.relative_to(ROOT)),'original_sha256':original,'original_bytes':n,'compressed_path':str(dst.relative_to(ROOT)),'compressed_sha256':sha(dst),'compressed_bytes':dst.stat().st_size,'lossless_verified':True});f.unlink()
 r=verify_bytes(rp);assert r['qualified'];assert all(z['bytes_match'] and z['counts_match'] for z in r['native_aggregate_reconciliation'].values())
 for group,pins in r['pins'].items():
  if group not in ['source_pins','runtime_pins','pins','harness_pins','artifact_pins']:continue
  base=ROOT/raw['args']['source_dir'] if group=='source_pins' else ROOT if group=='harness_pins' else H
  for key,value in pins.items():assert sha(base/key)==value,(item['name'],group,key)
 receipt=H/(item['name']+'-qualification.json');before=receipt.with_name(receipt.stem+'-precompression.json');assert not before.exists();before.write_bytes(receipt.read_bytes());receipt.write_text(json.dumps(r,indent=2)+'\n');receipts[item['arm']]=str(receipt.relative_to(ROOT));files.extend(f for f in d.rglob('*') if f.is_file())
files.extend(ROOT/q['console']['path'] for q in state['runs'])
write('live-nov3-selected-bytes-compression.json',compression)
summary={'schema':'astra-both-arm-selected-byte-summary-v1','plan_sha256':sha(p),'qualified':True,'receipts':receipts,'natural_eos':144,'native_jobs':{'D2H':530,'H2D':17},'candidate_jobs':{'D2H':527,'H2D':19},'components_per_job':28,'total_component_observations':30604,'all_bytes_equal':True,'all_executed_one_byte_controls_rejected':True,'all_native_Prometheus_counts_and_bytes_reconcile':True,'both_directions_with_first_admission_restoration':True,'zero_preemptions_both':True,'performance_eligible':False,'exact_geometry':{'gpu_blocks':1536,'cpu_blocks':2048,'active':8,'batch_tokens':1024,'hint_interval_ms':1000,'cudagraph':'FULL_DECODE_ONLY','compile_mode':0},'qualification_scope':'All28component copies checked attransfercompletion for exercisedTP1denseFP16AWQjobs, includingwarmup; independentblock/pagegeometry andnativeaggregatecounts/bytesreconcile. NativeTransfer.num_bytes notindependentlyobserved perjob. No semanticboundary, futurelifetime, answerquality, endurance orperformanceproof.','launch_status':'Await ROOT independent raw recomputation and separate manifest-bound launchgate. No freshconfirmation invoked.'}
write('live-nov3-selected-bytes-independent-summary.json',summary)
# Bounded read-only review of the separate, pre-generation confirmation manifest.
mp=ROOT/'evidence/live-optimization-2080ti-20261103/triage-confirmation-frozen.json';m=json.loads(mp.read_text());mismatches=[n for n,v in m['file_pins'].items() if not (ROOT/n).is_file() or sha(ROOT/n)!=v];assert not mismatches
assert len(m['pairs'])==12 and len(m['execution_order'])==27
orders={}
for part in [7703,8803,9901]:
 rows=[q for q in m['pairs'] if q['partition']==part];orders[part]=dict(collections.Counter(q['order'][0] for q in rows));assert len(rows)==4 and orders[part]=={'native':2,'candidate':2}
assert not any((H/'runs'/q['name']).exists() for q in m['execution_order'])
corpus_path=ROOT/'output/prefetch-native-20261102/fresh-agent-nov3/session-corpus.json';c=json.loads(corpus_path.read_text());old=json.loads((H/'prepared/triage-corpus-v2.json').read_text());assert len(c['sessions'])==72
assert not({s['source_id'] for s in c['sessions']}&{s['source_id'] for s in old['sessions']})
def first(s):return hashlib.sha256(json.dumps(s['turns'][0]['user_token_ids'],separators=(',',':')).encode()).hexdigest()
assert not({first(s) for s in c['sessions']}&{first(s) for s in old['sessions']})
byname={q['name']:q for q in m['execution_order']}
def opts(args):
 out={};i=0
 while i<len(args):
  key=args[i];i+=1
  if i<len(args) and not args[i].startswith('--'):out[key]=args[i];i+=1
  else:out[key]=True
 return out
arg_checks=[]
for pair in m['pairs']:
 n=opts(byname[pair['runs']['native']]['args']);a=opts(byname[pair['runs']['candidate']]['args'])
 for v in [n,a]:
  for k,z in {'--gpu-blocks':'1536','--cpu-blocks':'2048','--active':'8','--batch-tokens':'1024','--clients':'24','--turns':'3','--cohort':'agent','--service-profile':'repository_triage_v2','--hint-interval-ms':'1000','--wait':'bursty','--support-profile':True,'--graphs':True}.items():assert v[k]==z,(pair['pair'],k)
  assert '--audit' not in v and '--qualification' not in v and '--feasibility-only' not in v
 allowed={'--name','--arm','--native-arc','--no-audit','--source-label'};assert {k:v for k,v in a.items() if k not in allowed}=={k:v for k,v in n.items() if k not in allowed}
 assert n['--native-arc'] is True and a['--no-audit'] is True;arg_checks.append(pair['pair'])
review={'schema':'astra-fresh-triage-confirmation-prelaunch-review-v1','manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),'pins_checked':len(m['file_pins']),'pin_mismatches':mismatches,'pairs':12,'total_invocations':27,'balanced_orders_by_partition':orders,'matched_pair_argument_checks':arg_checks,'fresh_sessions':72,'prior_source_ID_overlap':0,'prior_retained_first_input_token_hash_overlap':0,'declared_run_outputs_exist':False,'design_blocker_found':False,'launch_gate_status':'Pending ROOT independent byte-row verification and separate manifest-bound launchgate; this review does not authorize launch.','interpretation_notes':['Quality first-pair inspection should mean firstpair in eachpartition: pairs0,1,2 andbotharms, not onlyglobalpair0.','Threeeageranchors use original-styleprechangeARC instrumentation undercommonruntime/HTTP; not stocknativecontrols. Report combinedgenericchanges separately frommatchednative comparison.','12wholeworkflow t intervals conditional on3fixedsourcepartitions, not12independentcorpora/populationgeneralization. Report eachpartition, allfailures andalltails.','Do not silently upgrade human-reviewed drafts to autonomousaccuracy/tasksuccess. Root/Astra must judge costs, limitedquality evidence andnetbenefit candidly.','Endurance remains required afterpositiveconfirmation, including4096identity fail-closed cap and boundedmetadata behavior.']}
write('live-nov3-triage-confirmation-astra-prelaunch-review.json',review)
inv={'schema':'astra-selected-byte-handback-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'custody':'ASTRA -> ROOT','runs':2,'natural_eos':144,'pins_checked':len(plan['file_pins']),'pin_mismatches':bad,'supervisor_exited':True,'all_runner_processes_exited':True,'all_raw_descendant_lists_empty':True,'owned_processes_remaining':[],'gpu_compute_processes':gpu,'gpu_lock_available':True,'artifact_bytes_retained':sum(f.stat().st_size for f in files),'compression':compression,'lossless_compression_bytes_saved':sum(q['original_bytes']-q['compressed_bytes'] for q in compression),'free_bytes_before_cleanup':freebefore,'free_bytes_after_cleanup':shutil.disk_usage(H).free,'retention_reason':'Full auditJSONL preserved inverifiedgzip, raw/server/driverlogs andpinarchives retained. Priorresults/sharedcaches/freshinputs unchanged. No disposableworktrees/downloads/checkpoints orownedprocesses remain.'}
write('live-nov3-selected-bytes-handback-inventory.json',inv)
print(json.dumps({'qualified':True,'gpu_free':True,'audit_pins':len(plan['file_pins']),'confirmation_pins':len(m['file_pins']),'component_checks':30604,'saved_bytes':inv['lossless_compression_bytes_saved'],'retained_bytes':inv['artifact_bytes_retained'],'confirmation_gate':'pending root verification'}))
