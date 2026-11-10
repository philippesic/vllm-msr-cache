"""Manual review17 context viewer and judgment retention; never assigns ratings automatically.

Import compact_view/save from mutable REPO with the pinned interpreter and bytecode
disabled. Read each answer and every unmatched/full referenced field before save.
retain_bindings refuses incomplete manual coverage. Frozen finish remains separate.
"""
from pathlib import Path
import json
from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json
repo=Path('/home/philip/vllm-semantic-cache-ext');e=repo/'evidence/stage6-5080-20261109';b=repo/'output/stage6-5080-20261109/quality-confirm-checkpoint-17'
packet=load(b/'blind-packet.json');n=load(b/'novel-contexts.json');fields=load(e/'confirm-review-index-16.json')['signature_fields']
prior_novel=[]
for no in ('02','03','04','05','06','07','08','09','10','11','12','13','14','15','16'):
 for j,x in enumerate(load(b.parent/f'quality-confirm-checkpoint-{no}/novel-contexts.json')):prior_novel.append((no,j,x))
cap_packet=load(repo/'output/stage6-5080-20261109/caps-all-checkpoint-07/blind-packet.json');assert digest(cap_packet['items'])==load(e/'caps-all-review-index-07.json')['packet_digest']
for j,x in enumerate(cap_packet['items']):prior_novel.append(('cap07',j,x))
nonanswer=[k for k in fields if k!='answer'];seen={};deps={}
for no,j,x in prior_novel:
 for key in nonanswer:seen.setdefault((key,digest(x[key])),dict(snapshot=no,index=j,field=key))
 for d,t in enumerate(x['actual_dependencies']):deps.setdefault(digest(t),dict(snapshot=no,index=j,dependency=d))
bindings=[]
for i,x in enumerate(n):
 row=dict(index=i,review_id=x['review_id'],components={},dependency_components=[])
 for key in nonanswer:
  sig=digest(x[key]);old=seen.get((key,sig));row['components'][key]=dict(digest=sig,prior_explicitly_read=old)
  seen.setdefault((key,sig),dict(snapshot='17',index=i,field=key))
  if key=='actual_dependencies'and old is None:
   for d,t in enumerate(x[key]):
    ds=digest(t);olddep=deps.get(ds);row['dependency_components'].append(dict(dependency=d,digest=ds,prior_explicitly_read=olddep));deps.setdefault(ds,dict(snapshot='17',index=i,dependency=d))
 bindings.append(row)
def compact_view(i):
 x=n[i];bind=bindings[i]
 print('INDEX',i,'ID',x['review_id'],'ROLE',x['response_role'],'WORDS',len(x['answer'].split()))
 for key in nonanswer:
  old=bind['components'][key]['prior_explicitly_read']
  if old and old['snapshot']!='17':print(key.upper(),'EXACT prior explicitly read',old);continue
  if old and old['snapshot']=='17':print(key.upper(),'EARLIER17 equality; repeated here in full',x[key]);continue
  if key=='actual_dependencies':
   print('NEW COMPLETE DEPENDENCY CHAIN')
   for part in bind['dependency_components']:
    d=part['dependency'];old=part['prior_explicitly_read']
    print('DEPENDENCY',d,'EXACT prior explicitly read'if old and old['snapshot']!='17'else'NEW OR REPEATED17 IN FULL',old if old and old['snapshot']!='17'else x[key][d])
  else:print(key.upper(),x[key])
 print('ANSWER\n'+x['answer'])
def save(part,rows):
 rs=[]
 for i,s,f,u,claims,note in rows:
  assert 0<=i<len(n)and all(type(v)is bool for v in(s,f,u))and isinstance(claims,list)and note
  x=n[i];rs.append(dict(index=i,review_id=x['review_id'],signature=digest({k:x[k]for k in fields}),source_supported=s,instruction_followed=f,useful=u,unsupported_claims=claims,whitespace_words=len(x['answer'].split()),notes=note+' Supplied source explicitly read earlier or in full at its first unmatched component in this snapshot; exact bindings checked separately. All new reference/instruction/dependency/tool/answer components explicitly read and judged. No automatic novel judgment; objective/timing unchanged.'))
 target=e/f'confirm-investigation-judgments-17-part-{part:02}.json';assert not target.exists()
 atomic_json(target,dict(schema='stage6-5080-explicit-novel-context-judgments-v1',packet_digest=digest(packet['items']),signature_fields=fields,reviewer='Codex explicit blinded source/test/instruction/dependency/tool/answer review',raw_packet=str((b/'blind-packet.json').resolve()),ratings=rs));print('SAVED',target.name,len(rs))
def retain_bindings():
 reviewed={}
 for path in sorted(e.glob('confirm-investigation-judgments-17-part-*.json')):
  part=load(path);assert part['packet_digest']==digest(packet['items'])and part['signature_fields']==fields
  for row in part['ratings']:
   i=row['index'];assert i not in reviewed and row['review_id']==n[i]['review_id']and row['signature']==digest({k:n[i][k]for k in fields})
   reviewed[i]=row
 assert set(reviewed)==set(range(len(n))),'Finish every explicit novel-context judgment first'
 target=e/'confirm-review-components-17.json';assert not target.exists()
 atomic_json(target,dict(schema='stage6-5080-explicit-review-component-bindings-v1',packet_digest=digest(packet['items']),signature_fields=fields,contexts=bindings,scope='Exact full fields and individual dependencies bind prior explicitly read confirm02 through confirm16, completed cap07 or earlier confirm17 contexts. Every unmatched component and answer explicitly read; no novel rating automatic. Source opening/full boundaries retained.'))
