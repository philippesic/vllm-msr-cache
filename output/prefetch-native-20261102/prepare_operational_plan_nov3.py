"""Review plan only. Execution requires independent ROOT review and a new freeze."""
import hashlib,json,sysconfig
from pathlib import Path
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
original=json.loads((ROOT/'evidence/live-optimization-2080ti-20261103/priority-confirmation-frozen.json').read_text())

def set_arg(args,key,value):
    if key in args:args[args.index(key)+1]=str(value)
    else:args.extend([key,str(value)])

cases=[]
# Alternating matched order is declared before any operational generation.
for phase,active,wait,waves,overlap,arms in [
    ('A',8,'bursty',24,0,('native8','candidate')),
    ('B',4,'bursty',3,0,('candidate','native8')),
    ('C',8,'longtool',3,0,('native8','candidate')),
    ('D',8,'bursty',8,0,('candidate',)),
    ('E',8,'bursty',3,20,('candidate','native8')),
]:
    for arm in arms:
        args=list(next(x['args'] for x in original['execution_order'] if x['arm']==arm))
        name='live-nov3-operational-'+phase.lower()+'-'+arm
        for key,value in [('--name',name),('--active',active),('--wait',wait),('--waves',waves),
                          ('--wave-seeds','14009,15013,16001'),('--wave-settle-s',2),
                          ('--wave-release-interval-s',overlap),('--registry-limit',128 if phase=='D' else 4096),
                          ('--timeout',7200)]:
            set_arg(args,key,value)
        if arm=='candidate':set_arg(args,'--source-dir',str((H/'sources/nov3-priority-integrated').relative_to(ROOT)))
        if phase=='A':args.append('--cancel-probe')
        cases.append({'phase':phase,'arm':arm,'name':name,'args':args,'normal_natural_eos_planned':waves*72,
                      'deliberate_abort_planned':int(phase=='A'),'health_natural_eos_planned':4 if phase=='A' else 0,
                      'operational_qualification_only':True,'formal_performance_eligible':False})
files=[H/x for x in ['prepare_operational_harness_nov3.py','prepare_operational_plan_nov3.py',
    'run_live_operational_nov3.py','live_operational_workload_nov3.py','operational_waves_nov3.py',
    'operational_state_nov3.py','operational_abort_nov3.py','operational_cancel_probe_nov3.py',
    'test_operational_harness_nov3.py','hooks-operational/sitecustomize.py',
    'live-nov3-operational-harness-cpu-tests.xml','live-nov3-priority-integration.json',
    'live-nov3-priority-integrated-tests.xml']]
files += list((H/'sources/nov3-priority-integrated/semantic_offload').glob('*.py'))
files += [ROOT/'semantic_offload/request_priority.py',Path(sysconfig.get_paths()['purelib'])/'vllm/entrypoints/openai/completion/serving.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan={
 'schema':'astra-operational-review-plan-v1','launch_authorized':False,'freeze_pending_root_review':True,
 'python':original['python'],'runner':str(H/'run_live_operational_nov3.py'), 'execution_order':cases,
 'planned_server_cases':len(cases),'planned_waves':74,'normal_natural_eos_planned':5328,
 'health_natural_eos_planned':8,'total_natural_eos_planned':5336,'deliberate_aborts_planned':2,
 'planning_provenance':'Finite coverage choices agreed by ROOT/ASTRA; not user numeric limits or indefinite stability proof.',
 'scope':{
  'A':'24 persistent serial regenerated waves per arm, 576 distinct application sessions. Default 4096 session cap NOT reached. Prefix/ACK maps may saturate; actual counts determine coverage. Then one actual abort plus four natural-EOS health checks.',
  'B':'Three persistent waves per arm at active4/bursty; reduced scheduler concurrency, same24 clients.',
  'C':'Three persistent waves per arm at active8/longtool; existing exact 4zero/4two/6thirty/5fortyfive/5sixty seconds per private wait.',
  'D':'Candidate-only correctness case: registry128, eight24-session waves. Tests fail-closed new advice with inference continuing; not a production4096 saturation proof.',
  'E':'Three overlapping finite burst waves per arm released at0/20/40 seconds, independent module globals and application identities, one priority epoch. Not indefinite open-arrival stability.',
 },
 'measurement_contract':{
  'intrusive_byte_audit':False,'preservation_audit_candidate':False,'source_critical_22_unchanged':True,
  'one_gateway_priority_origin_per_server':True,'model_admission_namespace_unchanged':True,
  'unique_application_run_cell_session_per_wave':True,'actual_histories_regenerated':True,
  'unbounded_natural_eos_remaining_context_only':True,'no_between_wave_cache_or_counter_reset':True,
  'initial_preload_counter_and_cuda_peak_reset_only':True,'no_residency_walks':True,
  'metadata_export':'Explicit O(1) lengths of fixed known collections; missing unavailable. Worker/scheduler export synchronizes only between serial waves or after all overlap waves.',
  'resource_sampling':'Existing continuous250ms per-process RSS/CPU and NVML/hostswap; driver retained evidence memory separate from server memory.',
  'native_policy':'Same normalizednativeARC, priority8s, lean runtime; native source oldhardened22; candidate integrated23 includes identical publichelper.',
  'new_hook_overhead':'Common bounded abort observer and scalar exports. These operational cases do not replace the733-pin performance confirmation.',
  'cancellation':'Close exactlyone stream afterfirsttoken beforeterminal; exactnative returnedFINISHED_ABORTED proof required. Losing EOS race is NOT exercised/fail, no retry. Four naturalEOS health requests retained regardless.',
 },
 'acceptance_checks':[
  'All scheduled normal and health requests have exact HTTP200, complete naturalEOS, only remaining-context max_tokens, actual generated history, exact public priority formula/shared epoch.',
  'No connection reuse, automatic/application POST retry, unexpected transport exception, lost wave identities or cross-wave hint binding. Deliberate disconnect separate; intentional hint cancellation separately matched.',
  'Each source/runtime/harness/model/corpus/helper pin holds before every launch and after end; all outcomes and raw failures retained without replacement.',
  'Default maps stay within4096session/1024observedprefix/4096ACK limits; session count is actual, not inferred. Accelerated128 case reaches limit and has real session_registry_capacity ACKs with successful subsequent generation.',
  'Preservation lifetime audit histories remain empty; pending/native requests and transfer metadata return to observed quiescent values. Missing counters unavailable, no zero substitution. Per-process server RSS and CUDA/NVML growth reported, no invented numeric plateau threshold.',
  'Actual fullGPUskip and partialCPUrefresh behavior remains causally valid; native transfer count/bytes load deltas and preemptions retained. Firstadmission attribution remains distinct from physical work.',
  'Report initial/zero-wait/positive-wait fairness, perwave output/workflowrates, mean/p95/maxTTFT andE2E. Single persistent case/arm/config is operational evidence, no fresh CI or quality/productivity equivalence.',
  'Abort must have exact backend request-ID transition once; healthchecks must allEOS, and settled metadata captured. Shutdown/reap all owned processes, compress logs, verifyGPUidle/lockfree.',
 ],
 'stop_conditions':['Sourcepin drift or resource/correctness safety blocker stops before next case.',
                    'A failedwave stops its servercase, preserves all partial outcomes and reaps concurrent wave tasks.',
                    'Three consecutive failedservercases stop block; no retries/replacements/favorable stop.',
                    '7200second per-case watchdog is an explicit engineering safety timeout, not a claimed user time budget.'],
 'existing_frozen_inputs':original['file_pins'],
 'new_review_file_pins':{str(p.resolve()):sha(p) for p in files},
 'unresolved_before_freeze':['ROOT independent implementation/control review; finalsource/runtime/allnewfilepins freeze; actualGPUidle/lock/memory preflight.'],
 'production_endorsement':False,
}
assert len(cases)==9 and sum(c['normal_natural_eos_planned'] for c in cases)==5328
(H/'operational-review-plan-nov3.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'cases':9,'waves':74,'normal_EOS':5328,'health_EOS':8,'abort_probes':2,'launch_authorized':False}))
