"""Grounded fixed samples; exact prior text/evidence note reuse is explicit."""
import hashlib,json,collections
from pathlib import Path
H=Path(__file__).resolve().parent
NOTES={('sympy__sympy-15976',2):'Latest grep lists MathML tests/implementations without demonstrating browser rendering. Response correctly limits that observation, but blanket missing-evidence language overlooks the initial reporter example and screenshot links; those images were not fetched/rendered by this service. No new executed rendering test.',('sympy__sympy-15599',2):'Material source-state error: initial issue combines proposed diff and PR-style success claims, but latest mod.py still has old if mod_l branch without was/changed logic. Output upgrades report to successful current implementation despite contradictory source, then says verification is missing. Test assertion exists in initial proposed patch but no executed test is shown.',('scikit-learn__scikit-learn-9288',2):'Faithful separation of initial reported inertia difference and latest grep locating KMeans at line763. Cause and intended behavior remain unverified; no service execution claimed.'}
def key(x):return(x['source_id'],x['turn'],x['generated_text'],json.dumps(x['source_evidence_turns'],sort_keys=True))
def main():
 p=H/'live-nov3-return-priority-quality-samples.json';q=json.loads(p.read_text());old={}
 for pref in ['gate-once','gpu-gate-confirmation','triage-confirmation']:
  sp=H/f'live-nov3-{pref}-quality-samples.json';rp=H/f'live-nov3-{pref}-quality-review.json';ss=json.loads(sp.read_text());rr=json.loads(rp.read_text());lookup={(x['run'],x['index']):x for x in rr['rows']}
  for run in ss['runs']:
   for x in run['samples']:
    if (run['name'],x['index']) in lookup:
     note=lookup[(run['name'],x['index'])];old[key(x)]={'receipt':str(rp),'receipt_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'run':run['name'],'index':x['index'],'finding':note.get('finding',note.get('note'))}
 rows=[];unique=set()
 for run in q['runs']:
  for x in run['samples']:
   prior=old.get(key(x));note=prior['finding'] if prior else NOTES[(x['source_id'],x['turn'])];unique.add(key(x))
   rows.append({'run':run['name'],'arm':run['arm'],'partition':run['partition'],'source_id':x['source_id'],'turn':x['turn'],'index':x['index'],'text_sha256':hashlib.sha256(x['generated_text'].encode()).hexdigest(),'source_evidence_sha256':hashlib.sha256(json.dumps(x['source_evidence_turns'],sort_keys=True).encode()).hexdigest(),'finding':note,'exact_prior_review_reuse':prior,'format_flags':x['format_flags'],'self_claimed_service_execution_or_edit_observed':False,'runaway_repetition_observed':False})
 a=json.loads((H/'live-nov3-return-priority-independent-analysis.json').read_text());formats={}
 for arm in a['arm_outcomes']:
  rr=[r for r in a['rows'].values() if r['arm']==arm];formats[arm]={k:sum(r['service_format'][k] for r in rr) for k in ['responses','starts_status','single_blank_line_block','with_code_fence']};formats[arm]['completed_action_phrase_flags']=sum(len(r['service_format']['completed_action_phrase_indices']) for r in rr)
 out={'schema':'astra-return-priority-source-grounded-quality-v1','sample_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reviewed_observations':len(rows),'distinct_cases':len(unique),'exact_prior_review_reuses':sum(bool(x['exact_prior_review_reuse']) for x in rows),'rows':rows,'mechanical_all_outputs':formats,'assessment':'No selected runaway or self-claimed service action; shared material source-reading defects persist. New-credit outputs had no mechanical formatting flags requiring extra samples. This is not accuracy equivalence, autonomous task success, semantic KV validation or measured human productivity. Human review remains necessary.','limits':['All18distinct selected texts inspected; exact source/text matches reuse prior grounded reviews with provenance. New variants grounded against retained initial/latest evidence.','Sample is fixed slot0turn0and2 perarm/trace,72observations, not a random accuracy study.','Recorded tools were not executed; quality defects are not attributed to caching and no outputs/runs excluded.']}
 (H/'live-nov3-return-priority-quality-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
