"""Verify author adjudication identities without certifying factual judgments."""
import hashlib,json,time
from pathlib import Path
b=Path(__file__).resolve().parent;items=json.loads((b/'manual-quality-adjudications.json').read_text());seen=set();checked=[]
for d in items:
 key=(d['campaign'],d['case'],d['source_id']);assert key not in seen,key;seen.add(key)
 root=b/key[0];r=json.loads((root/'result.json').read_text());case=next(c for c in r['cases'] if c['name']==key[1]);q=next(x for x in case['agent']['quality']['details'] if x['source_id']==key[2]);assert q.get('status_unparseable') is True,key
 row=next(x for x in map(json.loads,(root/key[1]/'requests.jsonl').read_text().splitlines()) if x['source_id']==key[2] and x.get('turn')==3);sha=hashlib.sha256(row['generated_text'].encode()).hexdigest()
 if 'text_sha256' in d:assert sha==d['text_sha256'],key
 checked.append(dict(campaign=key[0],case=key[1],source_id=key[2],actual_text_sha256=sha,original_record_had_hash='text_sha256' in d,manual_correct=d.get('manual_correct')))
r=dict(passed=True,checked_unix=time.time(),records=len(checked),details=checked,scope='Unique adjudication keys, exact retained response identities and original frozen unparseable-status outcome only. Does not certify author factual/status judgments or silently change original scores.')
(b/'adjudication-integrity.json').write_text(json.dumps(r,indent=2)+'\n');print('Verified',len(checked),'manual adjudication identities')
