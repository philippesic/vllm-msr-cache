"""All-outcome B/C-only assessment after documentation pin stop."""
import collections,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((H/'live-nov3-operational-v2-independent-analysis.json').read_text())
c=json.loads((H/'live-nov3-operational-v2-custody.json').read_text())
cases=[r for r in a['cases'].values() if r['attempted']];comparisons=[]
for pair in a['descriptive_comparisons']:
 if not pair['both_attempted']:continue
 ca=a['cases'][pair['candidate']];na=a['cases'][pair['control']]
 fairness={category:{metric:{stat:ca['normal']['categories'][category][metric][stat]-na['normal']['categories'][category][metric][stat] for stat in ('mean','p95','max')} for metric in ('TTFT_ms','E2E_ms')} for category in ('initial','return_zero_wait','return_positive_wait')}
 comparisons.append({**pair,'fairness_differences_ms':fairness,'serving_NVML_difference_MiB':ca['resources']['serving_nvml_peak_mib']-na['resources']['serving_nvml_peak_mib'],'startup_NVML_difference_MiB':ca['resources']['startup_nvml_peak_mib']-na['resources']['startup_nvml_peak_mib']})
summaries=[]
for r in cases:
 engine=max(r['process_memory']['per_process'].values(),key=lambda v:v['peak_RSS_MiB'])
 summary={'case':r['name'],'normal':r['normal'],'resources':r['resources'],'host':r['host_resources'],'thermal':r['thermal'],'engine_process_RSS':engine,
  'metadata_observed_ranges':r['metadata']['observed_ranges'],'metadata_missing_or_issue':r['metadata']['issues'],'first_admission':r['first_admission'],'native_transfers':r['total_native_transfers'],
  'stores':{k:r['scheduler_counts'].get(k) for k in ('store_accepted_keys','store_completed_keys','store_failed_keys','cpu_evicted_blocks')},
  'abort_receipt':r['basic_receipt']['abort_receipt'],'capacity_no_action_ACKs':r['basic_receipt']['capacity_no_action_ACKs']}
 summaries.append(summary)
out={'schema':'astra-operational-v2-bc-assessment-v1','analysis_sha256':sha(H/'live-nov3-operational-v2-independent-analysis.json'),'quality_receipt_sha256':sha(H/'live-nov3-operational-v2-quality-review.json'),
 'block_status':c['status'],'attempted_cases':4,'qualified_cases':sum(r['basic_receipt']['qualified'] for r in cases),'normal_EOS':sum(r['normal']['natural_eos'] for r in cases),'health_EOS':8,'direct_exact_backend_aborts':2,'all_return_prefixes_contiguous':576,
 'pin_result':{'checked':813,'matching':812,'mismatches':c['pin_checks'][-1]['mismatches'],'historical_failure_not_relabelled':True},
 'normal_rate_denominator':'min normal-wave load_start_ns through max normal-wave load_end_ns; includes serial inter-wave settling/exports, excludes startup and postnormal abort/health. Not mean/geomean of wave ratios. All operational rates descriptive, no CIs.',
 'comparisons':comparisons,'case_summaries':summaries,
 'judgment':{'production_endorsed':False,'fresh_performance_evidence':'Previously frozen fresh12workflowtriples remain positive for throughput/workflow/meanTTFT versus native8, with uncertain pooledp95 and noVRAMsavings. No operational result changes those conditional intervals.',
 'measured_progress':'Active4 and long private waits completed all requests; direct aborted Request external-ID mapping, subsequent healthEOS, causal priority/hints and quiescent pending state qualified in both arms. No preemptions. Metadata bounded at observed sizes, not capacity fallback.',
 'adverse_findings':'C maxTTFT+67.8ms and E2Ep95+71.0ms; B initialp95TTFT+96.2ms and initialmax+534.6ms despite favorable aggregate means. Higher candidate summedRSS partly reflects different residency/swap; no host-memory-saving claim. Source-reading/format defects remain in drafts.',
 'remaining_concrete_blocker':'Accelerated128-session-cap inference fallback and finite overlapping-wave isolation/fairness remain unexecuted because README pin drift stopped before D/E. Need new reviewed three-case continuation, not repeat B/C.',
 'limits':['Human-reviewed status drafts, not autonomous issue resolution or executed tools.','Continuation priority intentionally sacrifices initial-request queue latency versus allzero reference; no accepted customerSLA established.','Production4096 session-registry limit was not reached; finite128cap exercise would only qualify same path at acceleratedlimit.','Short B/C RSS ranges cannot establish asymptotic stability; previous24-wave A postcap evidence remains separate with historical failed abortmatcher.','Sampled hostRAM/swap system-wide, processRSS maysharepages; actual lockedbytes unavailable.','First-admission token attribution is not physical executedprefill time; C transfer/admission totals include abort+health.','Single persistentengine perarm/profile on three spent inputsets is operational evidence, not new randomized performanceconfirmation.']},
 'cleanup':{'owned_remaining':c['cleanup']['remaining_owned'],'final_idle':c['final_idle'],'custody_returned_to':'ROOT','no_follow_on_GPU_authorized':True}}
(H/'live-nov3-operational-v2-independent-assessment.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'qualified_cases':out['qualified_cases'],'normal_EOS':out['normal_EOS'],'health_EOS':8,'aborts':2,'production_endorsed':False}))
