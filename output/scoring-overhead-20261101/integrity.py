import gzip,json
from pathlib import Path
H=Path(__file__).parent
raws={p.parent.name:json.loads(gzip.decompress(p.read_bytes())) for p in (H/'runs').glob('*/raw.json.gz')}
checks=[]
for n,r in raws.items():
 terms=[t for t in r.get('terminals',[]) if t.get('external_request_id')]
 checks.append({'run':n,'requests':len(r['requests']),'all_success':all(x['success'] for x in r['requests']),'authoritative':sum(t.get('authoritative') is True for t in terms),'exact_key_coverage':sum(t.get('ordered_key_positions_complete') is True for t in terms),'physical_identity':sum(t.get('physical_block_identity_complete') is True for t in terms),'owned_remaining':r['owned_descendants_remaining']})
pairs=[]
for n,a in raws.items():
 if a['policy']!='semantic':continue
 for m,b in raws.items():
  if b['policy']!='arc' or a['workload']!=b['workload'] or a['seed']!=b['seed'] or a['threshold']!=b['threshold']:continue
  ar=sorted(a['requests'],key=lambda x:x['index']);br=sorted(b['requests'],key=lambda x:x['index'])
  pairs.append({'a':n,'b':m,'n':len(ar),'same_prompts':sum(x['payload']['prompt']==y['payload']['prompt'] for x,y in zip(ar,br)),'same_outputs':sum(x.get('output_token_ids')==y.get('output_token_ids') for x,y in zip(ar,br))})
(H/'integrity.json').write_text(json.dumps({'runs':checks,'paired_outputs':pairs},indent=2))
print(json.dumps({'runs':checks,'paired_outputs':pairs},indent=2))
