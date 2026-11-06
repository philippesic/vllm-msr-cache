"""Offline finite campaign consolidation; no serving-policy inputs."""
import csv,gzip,hashlib,json
from pathlib import Path
import campaign_metrics as metrics
H=Path(__file__).resolve().parent

def main():
 old=json.loads((H/'CONFIRMATION_RESULTS.json').read_text());new=json.loads((H/'AGENT_READMISSION_RESULTS.json').read_text());plan=json.loads((H/'DIAGNOSTIC_PLAN.json').read_text());diags=[];pairs=[]
 for r in plan['runs']:
  d=H/'runs'/r['name'];s=json.loads((d/'summary.json').read_text());diags.append({'name':r['name'],'summary':s,'summary_path':str((d/'summary.json').relative_to(H))})
 for l,r,label in [('diagnostic-agent-abandonedwait-arc','diagnostic-agent-abandonedwait-candidate','abandoned_wait'),('diagnostic-openloop-chat-50pct-arc','diagnostic-openloop-chat-50pct-candidate','finite_openloop_50pct'),('diagnostic-openloop-chat-85pct-arc','diagnostic-openloop-chat-85pct-candidate','finite_openloop_85pct')]:
  a=json.load(gzip.open(H/'runs'/l/'raw.json.gz'));b=json.load(gzip.open(H/'runs'/r/'raw.json.gz'));c=metrics.compare(a,b);p=H/'readmission-comparisons'/(label+'.json');p.write_text(json.dumps(c,indent=2,allow_nan=False)+'\n');pairs.append({'left':l,'right':r,'label':label,'comparison':c,'path':str(p.relative_to(H))})
 q=H/'runs/diagnostic-agent-quality-shadow-originalEMA';quality={'scheduler':[{'path':str(p.relative_to(H)),'data':json.loads(p.read_text())} for p in q.glob('prefix-policy-quality-scheduler-*.json')],'worker':[{'path':str(p.relative_to(H)),'data':json.loads(p.read_text())} for p in q.glob('prefix-policy-quality-worker-*.json')],'transport_selfcheck':json.loads((H/'quality-transport-validation.json').read_text()),'performance_eligible':False,'limitation':'shadow originalEMA agreement on actual candidate trajectory is not task answer quality or original-retention counterfactual'}
 rows=[dict(r,observer_revision='old frozen observer') for r in old['runs'] if r['cohort']=='chat']+[dict(r,observer_revision='V2 readmission observer-only revision') for r in new['runs']]
 output={'schema':'quantized-public-final-results-v1','accepted_core_scope':'old valid chat12 + new agent12, distinct observer revisions; no pooling old agent runs','core_runs':rows,'core_aggregates':[a for a in old['aggregates'] if a['cohort']=='chat']+new['aggregates'],'core_paired_comparisons':[a for a in old['paired_comparisons'] if a['cohort']=='chat']+new['paired_comparisons'],'old_failed_matrix':{'path':'CONFIRMATION_RESULTS.json','requests':1296,'strict_passed':1295,'failed_cell':'confirm-agent-a8-c24-candidate-s1103','not_accepted':True,'orchestration_failure':'schema validity erroneously used for proof gate; subsequent old runs preserved; no posthoc repair'},'diagnostics':diags,'diagnostic_comparisons':pairs,'quality':quality,'model':json.loads((H/'model-selection.json').read_text()),'plan_paths':['CONFIRMATION_PLAN.json','AGENT_READMISSION_PLAN.json','DIAGNOSTIC_PLAN.json'],'claims':{'exact_work_99pct_throughput_goal_established':False,'production_concurrency_capacity_established':False,'40pct_total_VRAM_reduction_established':False,'actual_scoring_and_native_prefetch_exercised':True,'agent_cross_arm_exact_histories':False,'fixed_GPU_KV_pool_reduction_bytes':939524096,'same_reduction_available_to_small_ARC':True}}
 (H/'final-results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
 with (H/'final-results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'core_requests':sum(r['requests'] for r in rows),'strict_proofs':sum(r['strict_proofs_passed'] for r in rows),'diagnostic_runs':len(diags),'diagnostic_requests':sum(d['summary']['all']['requests'] for d in diags),'quality_scheduler_files':len(quality['scheduler'])}))
if __name__=='__main__':main()
