"""Preserve the stopped first operational block, including formal failures."""
import hashlib,json,shutil,subprocess,fcntl
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
a=json.loads((H/'live-nov3-operational-independent-analysis.json').read_text())
rows={r['arm']:r for r in a['cases'].values() if r['attempted']}
summary={}
for arm,r in rows.items():
    summary[arm]={'normal':r['normal'],'resources':r['resources'],'process_memory':r['process_memory'],
                  'metadata_ranges':r['metadata']['observed_ranges'],'case_status':r['status'],'error':r['error'],
                  'health_EOS':sum(q.get('natural_eos') is True for q in r['cancellation_probe']['health_requests']),
                  'abort_observer':r['cancellation_probe']['abort_observer'],'formal_abort_qualified':False,
                  'store_accepted':r['scheduler_counts'].get('store_accepted_keys'),'store_completed':r['scheduler_counts'].get('store_completed_keys'),
                  'explicit_store_failed':r['scheduler_counts'].get('store_failed_keys'),'final_Prometheus_available':False}
out={'schema':'astra-operational-A-final-assessment-v1','production_endorsement':False,'goal_achieved':False,
 'block_status':'Stopped after2of9cases atREADMEpin drift; seven cases never launched.',
 'case_status':'Both A cases formally FAILED frozen abort identity qualification despite1728normalEOS+4healthEOS each. Failures remain unchanged.',
 'observed_actual_aborts':'One FINISHED_ABORTED per arm; internal randomized suffix defeated frozen exact matcher. Posthoc mapping is distinct from a qualification pass; randomizer source was not pre-pinned.',
 'bounded_state_judgment':'No post-cap growth/leak observed in the finite measured window. Candidate prefix1024/ACK4096/residency2048 caps hold, session registry576 below4096. Observed pending/audit histories empty at boundaries. This is not indefinite stability or production4096-session saturation proof.',
 'memory_judgment':'Candidate engine initial-to-finalRSS+16.44MiB while filling metadata; last-quarter range0.0508MiB and slope+0.0122MiB/min. Native corresponding range0.0586MiB. Absolute cross-processRSS differences are descriptive, not uniquely attributed savings. Driver retains benchmark evidence.',
 'descriptive_comparison':a['descriptive_comparisons'][0],
 'fairness_judgment':'Initial p95TTFT differs+2.46ms, zero-wait and positive-wait return distributions improve descriptively. One persistent engine perarm and repeated inputs give no newCI; fresh confirmation remains unchanged.',
 'missing_evidence':['Active4,longtool,accelerated128registrycap and finite overlap were not executed.','Final Prometheus metrics were skipped after probe failure; totalnativeH2D/D2H/preemptions/firstadmission attribution unavailable.','Exact abort qualification failed; no quality/task-success or human-productivity equivalence.'],
 'pin_drift':'ROOT identified a concurrent local commit/merge changing README only. Preserve mergedREADME. Supervisor stopped and did not resume. Original manifest and failures remain historical evidence.',
 'next_recommendation':'Newnames/freeze for remaining sevenB–E cases, unchanged serving configs, two new direct-mapped cancellation probes inC. PreserveA failures, add exact publichelperclock receipts and best-effort failuremetrics capture. No GPU untilROOTreview/freeze.',
 'cases':summary,'quality_receipt':'live-nov3-operational-quality-review.json','cleanup_receipt':'live-nov3-operational-custody.json'}
(H/'live-nov3-operational-a-independent-assessment.json').write_text(json.dumps(out,indent=2)+'\n')
# Bound cleanup to disposable bytecode generated for this task, never source/pins.
free_before=shutil.disk_usage(H).free;removed=[]
for base,stem in [(H/'__pycache__',x) for x in ['operational_state_nov3','operational_waves_nov3','live_operational_workload_nov3','run_live_operational_nov3']]+[(H/'hooks-operational/__pycache__','sitecustomize')]:
    for p in base.glob(stem+'.cpython-*.pyc') if base.exists() else []:
        if p.is_symlink():continue
        removed.append({'path':str(p),'bytes':p.stat().st_size});p.unlink()
files=[]
for name in ['live-nov3-operational-a-native8','live-nov3-operational-a-candidate']:
    files.extend(p for p in (H/'runs'/name).rglob('*') if p.is_file())
files.extend(p for p in H.glob('live-nov3-operational-*') if p.is_file() and 'inventory' not in p.name)
files=sorted(set(files));inventory=[]
for p in files:inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'retention':'Useful failed/partial operational evidence, qualification and recovery artifacts; preserve.'})
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert not gpu
state=json.loads((H/'live-nov3-operational-custody.json').read_text())
report={'schema':'astra-operational-A-handback-inventory-v1','GPU_custody':'ROOT','GPU_compute_apps':[],'lock_available':True,
        'supervisor_exit':1,'owned_descendants_remaining':state['cleanup']['remaining_owned'],
        'retained':inventory,'retained_bytes':sum(x['bytes'] for x in inventory),'removed_disposable_bytecode':removed,
        'free_bytes_before':free_before,'free_bytes_after':shutil.disk_usage(H).free,
        'scope':'WSL filesystem only; no WindowsVHDX reclaim claim. Shared caches, committedREADME, all frozen inputs and useful earlier evidence preserved.'}
(H/'live-nov3-operational-a-handback-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
recovery=H/'live-nov3-operational-astra-recovery-state.json'
r=json.loads(recovery.read_text());r.update(GPU_custody='ROOT',exec_session=None,current='Block stopped afterA;READMEpin drift. Twoformalabortfailures retained.',remaining='Prepare NEWv2remaining7case harness/controls underH; noGPU beforeROOTfreeze.',assessment='live-nov3-operational-a-independent-assessment.json',inventory='live-nov3-operational-a-handback-inventory.json');recovery.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'retained_bytes':report['retained_bytes'],'removed_bytecode':sum(x['bytes'] for x in removed),'GPUidle':True,'custody':'ROOT'}))
