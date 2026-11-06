"""Actual complete workflow work comparison, never fed back to prefetch policy."""
import argparse,gzip,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent

def inventory(name):
 path=H/'runs'/name/'raw.json.gz'
 with gzip.open(path,'rt') as f:raw=json.load(f)
 rows={}
 for r in raw['requests']:
  identity=(r['session_id'],r['turn'])
  if identity in rows:raise ValueError('duplicate workflow identity')
  rows[identity]=(r['payload']['prompt'],r['output_token_ids'])
 assert raw['status']=='complete' and len(rows)==raw['planned_requests']
 return rows

def differences(a,b):
 return sum(x!=y for x,y in zip(a,b))+abs(len(a)-len(b))

def compare(left,right):
 a,b=inventory(left),inventory(right)
 assert set(a)==set(b),'workflows must complete the same sessions/turns'
 rows=[{'identity':list(k),'left_input_tokens':len(a[k][0]),'right_input_tokens':len(b[k][0]),'input_token_differences':differences(a[k][0],b[k][0]),'left_output_tokens':len(a[k][1]),'right_output_tokens':len(b[k][1]),'output_token_differences':differences(a[k][1],b[k][1])} for k in sorted(a)]
 return {'left':left,'right':right,'requests':len(rows),'identical_input_token_ids':all(r['input_token_differences']==0 for r in rows),'identical_output_token_ids':all(r['output_token_differences']==0 for r in rows),'identical_per_request_decode_counts':all(r['left_output_tokens']==r['right_output_tokens'] for r in rows),'totals':{f'{arm}_{kind}_tokens':sum(r[f'{arm}_{kind}_tokens'] for r in rows) for arm in ['left','right'] for kind in ['input','output']},'rows':rows,'limit':'Exact IDs/counts prove workload equivalence when true; not a task quality score or a substitute for physical KV/byte qualification.'}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('left');ap.add_argument('right');ap.add_argument('--output',required=True);args=ap.parse_args()
 result=compare(args.left,args.right);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
