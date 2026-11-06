"""Single-workflow geometry discovery, using the independent raw extractor."""
import hashlib,json,math
from pathlib import Path
from analyze_confirmation_astra_nov3 import read,paired
H=Path(__file__).resolve().parent

def main():
 p=H/'pressure-grid-frozen-plan-nov3.json';plan=json.loads(p.read_text());rows={item['name']:read(item) for item in plan['execution_order']};comparisons=[]
 for pair in plan['pairs']:
  a=rows[pair['runs']['candidate']];b=rows[pair['runs']['native']];row={**pair,'both_eligible':a['eligible'] and b['eligible'],'scope':'One spent-partition workflow per arm; descriptive discovery only, no confidence interval or production endorsement.'}
  if a['attempted'] and b['attempted']:row.update(first_prompts_equal=a['first_prompt_digests']==b['first_prompt_digests'],private_waits_equal=a['private_wait_ms']==b['private_wait_ms'])
  if row['both_eligible']:
   d=paired(a,b);row['differences']=d;row['rate_changes_percent']={k:100*math.expm1(v) for k,v in d.items() if k.endswith('log_ratio')}
  comparisons.append(row)
 report={'schema':'astra-pressure-grid-discovery-analysis-v1','plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'extractor_sha256':hashlib.sha256((H/'analyze_confirmation_astra_nov3.py').read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'comparisons':comparisons,'limitations':['All inputs are spent discovery data.','Natural outputs can vary; report workflow rate and token rate separately.','Missing counters are unavailable, never assumed zero.','Initial-prefill attribution excludes repeated preemption work and is not physical GPU time.','Matched capacities isolate comparison within each geometry; cross-geometry resource gains are generic.','No quality or task-success claim from EOS/byte equality.']}
 (H/'live-nov3-pressure-grid-independent-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'comparisons':comparisons,'outcomes':{n:{k:x.get(k) for k in ['status','eligible','requests','natural_eos','load_s','tokens','tokens_s','first_prefill_attribution','transfers']} for n,x in rows.items()}},indent=2))
if __name__=='__main__':main()
