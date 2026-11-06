"""Final independent discovery assessment; does not modify frozen experiment inputs."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent

def read(n): return json.loads((H/n).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=read('live-nov3-gate-once-independent-analysis.json');s=read('live-nov3-gate-once-independent-summary.json');q=read('live-nov3-gate-once-quality-review.json')
 assert q['sample_receipt_sha256']==sha(H/'live-nov3-gate-once-quality-samples.json')
 assert len(q['rows'])==q['reviewed_observations']==145
 assert a['all18eligible'] and s['natural_eos']==1296 and s['http']['all_qualified']
 comparisons=[c for c in s['descriptive_comparisons_no_CI'] if c['candidate']=='once_positive']
 totals={}
 for arm in ['periodic','once_positive']:
  rows=[r for r in a['rows'].values() if r['arm']==arm]
  assert all(r['gate_once_receipt']['qualified'] for r in rows)
  totals[arm]={k:sum(r['gate_once_receipt']['per_hint'][k] for r in rows) for k in ['observations','GPU_available','GPU_unavailable','full_GPU_skips','CPU_actual_refresh_operations']}
  totals[arm]['actual_positive_policy_stops']=sum(len(r['gate_once_receipt']['actual_positive_stop_receipts']) for r in rows)
 report={'schema':'astra-gated-once-final-independent-assessment-v1','production_endorsement':False,'advance_to_endurance':False,
 'judgment':'Once-positive gating is not an advance: lower workflow rate than periodic gating on5/6 traces and native on4/6; no meaningful VRAM reduction. Continue optimization, not production qualification.',
 'scope':'18 retained workflows, one per arm per six spent traces; no inferential confidence interval or independent held-out claim.',
 'completion':{'invocations':18,'natural_EOS':1296,'HTTP':s['http'],'all396pins_held_at_cleanup':True,'preemptions':0,'postselection_or_replacements':False},
 'comparisons':comparisons,'activation_totals':totals,
 'mechanism':'Actual CPU key touches fall from29716 to8548, yet13001 has identical first-admission GPU/external/local attribution and aggregate H2D/D2H under periodic and once. Removing repeated late touches does not remove that trace\'s cache harm; aggregate counters do not identify exact eviction victims or prove first-touch causation.11003/12007 lose much of periodic restoration benefit.',
 'resources':s['resources'],'transfer_and_first_admission_receipt':str(H/'live-nov3-gate-once-independent-summary.json'),
 'quality':{'receipt':str(H/'live-nov3-gate-once-quality-review.json'),'observations':145,'distinct_cases':57,'exact_prior_evidence_and_text_review_reuses':46,'mechanical_all_outputs':q['mechanical_all_outputs'],'assessment':q['assessment'],'examples':['Django10914: requested default0o644 misreported as existing behavior.','Sphinx9281: unlabelled Exit0/None upgraded to successful documentation build.','Once7703 Astropy7671 turn2 emits fenced function definition contrary to no-code contract.']},
 'limitations':['Quality defects occur across arms; no accuracy equivalence, task-success or measured human productivity claim. Caching is not expected to repair model reasoning.','Raw driver pins still name periodic workload in once runs; explicit archived workload hash against the frozen selected module, args and executed producer receipts establish actual selection.','First-admission prompt attribution is not physical GPU work/time; total transfers are distinct.','Configured host KV cache1792MiB perarm; actual pinned/locked bytes unavailable. Summed RSS may double-count shared mappings; swap is host-wide. Missing store-failure counter unavailable, not zero.','All396 frozen input pins held; useful compressed raw/log/source evidence retained.'],
 'next_direction':{'recommendation':'Prefer a bounded causal-return priority discovery through the existing API before more cadence variants or bounded native-prefetch expansion. Keep matched native priority controls to separate generic scheduling from incremental cache policy.',
 'priority_review':['No cache-hit estimate or future wait/output oracle; actual released turn>0 and one shared gateway clock suffice.','Common --scheduling-policy priority also changes memory-pressure preemption victim selection in installed Scheduler.schedule. It is not waiting-only. Count preemptions and recomputation, and avoid claiming exact FCFS equivalence of all-zero mode under memory pressure.','Fixed credit bounds submission-time overtaking up to integer-ms quantization, not wall-clock service/starvation. Measure initial, zero-wait and positive-wait tails and max latency, not only pooled TTFT.','Credits0/2000/8000 crossed with native/gate and balanced six-position order are reasonable prospective discovery choices, not user thresholds. Any selected credit needs fresh confirmation;36 one-shot runs are not36 independent corpora.','Measure actual order changes if authoritative existing telemetry allows; client payload priority alone proves submission but not scheduling opportunity. Require server command and accepted HTTP priority receipt.'],
 'alternatives':['T1-only recency is cheap and causally distinct, preserving native frequent-T2 order, but earlier promotion had sparse T1 opportunities. Establish actual eligibility before another large grid.','True native-prefix prefetch has higher bounded-state, ownership, native-adoption and copy qualification cost. Existing busy replay negatives are not natural-EOS proof but lower its priority.','Neither discovery failures nor cheap-policy coverage establish exhaustion of the user objective.']},
 'custody':'ASTRA -> ROOT; no further GPU invocation authorized by this completed block; owned processes already reaped in handback inventory.'}
 p=H/'live-nov3-gate-once-independent-assessment.json';p.write_text(json.dumps(report,indent=2)+'\n')
 a['assessment']=report['judgment'];a['final_assessment_receipt']=str(p);(H/'live-nov3-gate-once-independent-analysis.json').write_text(json.dumps(a,indent=2)+'\n')
 invp=H/'live-nov3-gate-once-handback-inventory.json';inv=read(invp.name)
 existing={x['path'] for x in inv['retained_files']}; extra=[]
 candidates=list(H.glob('live-nov3-gate-once-*'))+list(H.glob('*gate_once*astra_nov3.py'))+[H/'gate-once-frozen-astra-copy.json']
 for f in sorted(set(candidates)):
  if not f.is_file() or f==invp:continue
  relative=str(f.relative_to(H.parents[1]))
  if relative not in existing:extra.append({'path':relative,'bytes':f.stat().st_size})
 inv['analysis_artifacts']=extra;inv['analysis_artifact_bytes']=sum(x['bytes'] for x in extra);inv['total_retained_run_and_analysis_bytes']=inv['retained_bytes']+inv['analysis_artifact_bytes'];inv['retention_reason']='Compressed raw results/logs, frozen invocation receipts, source-grounded quality and reproducible negative assessment are useful evidence. Shared model/source caches untouched; no disposable owned process/worktree remains.'
 invp.write_text(json.dumps(inv,indent=2)+'\n');print(json.dumps({'assessment':str(p),'activation':totals,'retained_bytes_excluding_inventory':inv['total_retained_run_and_analysis_bytes']}))
if __name__=='__main__':main()
