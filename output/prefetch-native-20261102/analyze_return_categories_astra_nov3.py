"""Retrospective wait categories only; no policy access to private future waits."""
import collections,gzip,hashlib,json
from pathlib import Path
from analyze_confirmation_astra_nov3 import dist
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(items):
 return {'requests':len(items),'natural_eos':sum(x['natural_eos'] for x in items),'generated_tokens':sum(x['output_tokens'] for x in items),'logical_prompt_tokens':sum(x['prompt_tokens'] for x in items),'TTFT_ms':dist([x['TTFT_ms'] for x in items if x['TTFT_ms'] is not None]),'E2E_ms':dist([x['E2E_ms'] for x in items if x['E2E_ms'] is not None]),'return_contiguous':dict(collections.Counter(str(x['return_contiguous']) for x in items)),'private_wait_ms_counts':dict(collections.Counter(str(x['private_wait_ms']) for x in items))}
def main():
 reports={}
 for block in ['gpu-recency','gpu-gate']:
  m=json.loads((H/(block+'-frozen-plan-nov3.json')).read_text());rows={}
  for item in m['execution_order']:
   p=H/'runs'/item['name']/'raw.json.gz';r=json.load(gzip.open(p,'rt'));byindex={q['index']:q for q in r['requests']};cont=r.get('prefix_contiguity');cont_by={x['index']:x for x in cont} if isinstance(cont,list) else {};issues=[];qs=[]
   if cont is None:issues.append('missing_prefix_contiguity_records')
   elif len(cont_by)!=len(cont):issues.append('duplicate_contiguity_records')
   for q in r['requests']:
    turn=q['turn'];slot=q['index']//4;assert q['index']==slot*4+turn
    wait=None;category='initial';same=None
    if turn:
     key=f'{slot}:{turn-1}';wait=r['workload']['private_wait_ms'].get(key);category='return_zero_wait' if wait==0 else 'return_positive_wait' if isinstance(wait,(int,float)) and wait>0 else 'return_wait_unavailable'
     previous=byindex[q['index']-1];assert previous['session_id']==q['session_id']
     source=previous['payload']['prompt']+previous['output_token_ids'][:-1];source=source[:len(source)//16*16];same=q['payload']['prompt'][:len(source)]==source;record=cont_by.get(q['index'])
     if record is None or record.get('contiguous') is not same or record.get('previous_computed_fullblock_tokens')!=len(source):issues.append('reported_recomputed_contiguity_mismatch:'+str(q['index']))
    qs.append({'index':q['index'],'session_id':q['session_id'],'turn':turn,'category':category,'private_wait_ms':wait,'return_contiguous':same,'natural_eos':q.get('natural_eos',False),'output_tokens':len(q.get('output_token_ids',[])),'prompt_tokens':len(q['payload']['prompt']),'TTFT_ms':(q['first_token_ns']-q['request_start_ns'])/1e6 if 'first_token_ns' in q else None,'E2E_ms':(q['request_end_ns']-q['request_start_ns'])/1e6 if 'request_end_ns' in q else None})
   rows[item['name']]={'arm':item['arm'],'partition':item['partition'],'raw_sha256':sha(p),'status':r.get('status'),'issues':issues,'return_prefix_contiguity_raw':dict(collections.Counter(str(x.get('contiguous')) for x in cont)) if cont is not None else None,'return_prefix_contiguity_recomputed':dict(collections.Counter(str(x['return_contiguous']) for x in qs if x['turn']>0)),'categories':{c:summarize([x for x in qs if x['category']==c]) for c in ['initial','return_zero_wait','return_positive_wait','return_wait_unavailable']},'requests':qs}
  diffs=[]
  for pair in m['pairs']:
   candidate='gpu' if block=='gpu-recency' else 'gate'
   for control in ['native','cpu']:
    aa=rows[pair['runs'][candidate]];bb=rows[pair['runs'][control]]
    for category in ['initial','return_zero_wait','return_positive_wait']:
     x=aa['categories'][category];y=bb['categories'][category];assert x['requests']==y['requests']
     diffs.append({'partition':pair['partition'],'candidate':candidate,'control':control,'category':category,'requests_per_arm':x['requests'],'mean_TTFT_difference_ms':x['TTFT_ms']['mean']-y['TTFT_ms']['mean'],'p95_TTFT_difference_ms':x['TTFT_ms']['p95']-y['TTFT_ms']['p95'],'mean_E2E_difference_ms':x['E2E_ms']['mean']-y['E2E_ms']['mean'],'p95_E2E_difference_ms':x['E2E_ms']['p95']-y['E2E_ms']['p95']})
  reports[block]={'manifest_sha256':sha(H/(block+'-frozen-plan-nov3.json')),'rows':rows,'category_differences':diffs}
 out={'schema':'astra-retrospective-return-categories-v1','analyzer_sha256':sha(Path(__file__)),'scope':'Descriptive request distributions only. Private wait categories used retrospectively, never as policy inputs. No request-level inference, subgroup selection, quality validation or production claim. Small zero-wait groups make p95 unstable.','blocks':reports}
 (H/'live-nov3-return-category-independent-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({b:{'contiguity':{n:r['return_prefix_contiguity_recomputed'] for n,r in report['rows'].items()},'issues':sum(len(r['issues']) for r in report['rows'].values()),'candidate_vs_native': [x for x in report['category_differences'] if x['control']=='native']} for b,report in reports.items()},indent=2))
if __name__=='__main__':main()
