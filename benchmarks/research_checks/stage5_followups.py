"""Declare evidence-driven follow-ups before measurements; never alter prior runs."""
import argparse
import copy
import hashlib
from pathlib import Path
from benchmarks.research_checks.stage5_optimization import prepare, balanced_order, preflight
from benchmarks.serving_suite.contracts import write_json, digest


def add(spec,name,*,extra=None,**plan):
    condition=copy.deepcopy(spec['conditions']['wait'])
    condition.update(plan)
    condition['candidate_config']['kv_connector_extra_config'].update(extra or {})
    spec['conditions'][name]=condition


def write(root,label,spec,question,seed):
    preflight(spec)
    names=list(spec['conditions'])
    spec['order']=balanced_order(names,spec['plan']['repeats'],seed)
    spec['stage5'].update(question=question,variants=names,seed=seed)
    path=Path(__file__).resolve()
    spec['input_files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root/(label+'-spec.json'),spec)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    spec=prepare('mixed',['arc','wait','boost50','cadence5','pressure75'],1,700)
    for i,s in enumerate(spec['corpus']['sessions']):s['abandoned_hint']=i%4==0
    spec['corpus']['provenance']+=' Authored false-positive advice stress: every fourth session continues causal advice for its final completed prefix until all real sessions finish; no further request is offered. The policy receives no abandonment marker.'
    spec['plan'].update(corpus_sha256=digest(spec['corpus']),corpus_provenance=spec['corpus']['provenance'])
    spec['stage5']['workload']='abandoned-mixed'
    for budget in (512,1024,2048):add(spec,f'wait-budget{budget}',extra={'wait_hint_arc_advice':{'session_budget':budget,'budget_scope':'wait'}})
    add(spec,'prefix75',extra={'wait_hint_arc_advice':{'fraction':.75}})
    write(a.output,'abandoned',spec,'Mixed hinted/unhinted traffic with explicit abandoned-future advice; distinguish per-wait from exhausted whole-session quotas.',700)
    spec=prepare('live-repository',['arc','wait'],1,701)
    add(spec,'during-tool',hint_timing='during-tool')
    add(spec,'coalesced-client',hint_coalesce_initial=True)
    add(spec,'joined-overlap',hint_overlap_initial=True)
    write(a.output,'live-publication',spec,'Real tools: overlap publication with tool execution, join the initial RPC before continuation, and remove immediate duplicate pre-tool/on-start publications. Live outputs and task outcomes retained.',701)
    spec=prepare('repository',['arc','wait','pressure75'],3,702)
    write(a.output,'pressure-repeat',spec,'Three fixed repeated development comparisons of apparent pressure-gate screen improvement; no pooling with selecting screen.',702)
    spec=prepare('repository',['arc','wait','window50','window250','grace750','grace2250','tool-only'],1,703)
    add(spec,'cadence05',hint_refresh_s=.5)
    write(a.output,'timing',spec,'Strict supported declarations plus opt-in late-delivery grace for CPU recency only. Grace extends neither GPU leases nor client estimates. Half-second refresh fills the cadence grid; no future-time oracle.',703)
    spec=prepare('repository',['arc','wait','audit-off'],1,704,{'tier':'operational','repository_workload':True})
    write(a.output,'audit',spec,'Matched fixed-token operational replay; audit-off removes native request histories, so do not claim cache-work attribution from missing receipts. Abort/recovery independently checked.',704)
    spec=prepare('uneven',['arc','wait','boost50','cadence2','cooldown5','pressure75'],1,705)
    write(a.output,'uneven',spec,'Authored uneven tool waits test cadence, cooldown, partial boost and pressure gating beyond one delay distribution.',705)
    spec=prepare('live-documents',['arc','wait','cadence2','pressure75','boost50'],1,706)
    write(a.output,'documents',spec,'Independent tight document tools and natural outputs; test promising low-cost and displacement controls with actual task observations.',706)
    spec=prepare('tight',['arc','wait','cadence2','cadence5'],1,707)
    write(a.output,'tight-cadence',spec,'Tight-cache no-op screen suggested publication overhead but timing was noisy. Test reduced publication cadence under the same pressure.',707)
    spec=prepare('tight',['arc','wait'],1,708)
    add(spec,'arc-prompt-only',publish_hints=False,extra={'offload_prompt_only':True,
        'wait_hint_completed_transcript':False,'wait_hint_cpu_retention_only':False,
        'prefetch_budget_fraction':0,'wait_hint_cpu_retention_gpu_gate':False})
    add(spec,'wait-prompt-scope',hint_completed_transcript=False,extra={'wait_hint_completed_transcript':False})
    add(spec,'wait-prompt-only',hint_completed_transcript=False,
        extra={'offload_prompt_only':True,'wait_hint_completed_transcript':False})
    write(a.output,'prompt-storage',spec,'Matched storage ablation: exclude newly decoded tails. Full-store prompt-scope control separates the necessarily smaller published prefix from storage itself. Common offloading change, not retention-only gain.',708)
    spec=prepare('tight',['arc','wait','linear-wait'],1,709)
    add(spec,'small-batch',batch_tokens=512)
    add(spec,'fcfs',continuation_credit_ms=0)
    add(spec,'combined',batch_tokens=512,continuation_credit_ms=0,extra={'arc_linear_eviction':True})
    write(a.output,'combination',spec,'One contemporaneous combination screen: single-pass eviction, smaller prefill batching and FCFS each had small favorable wait-arm points. Test interaction without pooling earlier selecting screens.',709)
    spec=prepare('tight',['arc','wait'],1,710)
    write(a.output,'default-reference',spec,'Fresh default 8-second-credit reference after the 16-second unsupported-setting failure; already tested supported maximum, not a new tuning point.',710)
    spec=prepare('repository',['arc','wait','gpu-bridge'],1,711,{'cpu_bytes':1409286144})
    write(a.output,'bridge',spec,'Retarget inactive GPU bridge with CPU smaller than GPU; require additional suffix-key activation before attributing any timing difference.',711)
    spec=prepare('repository',['arc','wait'],1,712,{'cpu_bytes':3758096384})
    write(a.output,'capacity',spec,'Same offered work and admission with more CPU cache; unequal-resource headroom reference only, never a selectable policy gain.',712)

if __name__=='__main__':main()
