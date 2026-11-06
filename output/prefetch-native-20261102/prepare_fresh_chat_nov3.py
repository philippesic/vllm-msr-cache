"""Additional bounded public input windows, selected without seeing model outputs."""
from pathlib import Path
import json,hashlib,urllib.request,urllib.parse,re,datetime
H=Path(__file__).resolve().parent;D=H/'fresh-chat-nov3';D.mkdir(exist_ok=True)
REV='7d6490e462285cf85d91eabea0f9a954fbddcd1f';OFFSETS=[20100,30100,50100,150000,250000,350000,500000,700000]
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
plan={'revision':REV,'offsets':OFFSETS,'rows_per_window':100,'max_response_bytes':3000000,'selection':'Same English multi-turn filters and token truncation as prior corpus. Exclude all prior source IDs and duplicate first-user text. SHA256(session ID) order, first72 suitable; cyclic disjoint partitions4409/5501/6607. No model responses inspected; no selection by model performance.','source_policy':'Discard unused personal/network fields and historical assistant text; regex-redact email/phone as prior corpus. This does not assert complete PII removal.','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=D/'plan.json'
if p.exists():assert {k:v for k,v in json.loads(p.read_text()).items() if k!='created_utc'}=={k:v for k,v in plan.items() if k!='created_utc'}
else:write(p,plan)
for offset in OFFSETS:
 dest=D/f'normalized-{offset}.json'
 if dest.exists():continue
 params={'dataset':'allenai/WildChat-1M','config':'default','split':'train','offset':offset,'length':100,'revision':REV};url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(params)
 try:
  with urllib.request.urlopen(url,timeout=60) as response:
   body=response.read(3000001);header=response.headers.get('x-revision')
  assert len(body)<=3000000,'Response exceeds predeclared bound'
  assert header==REV,('Dataset revision drift',header)
  rows=[]
  for item in json.loads(body)['rows']:
   r=item['row'];users=[]
   for i,m in enumerate(r['conversation']):
    if m['role']!='user':continue
    text=m.get('content','');text=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b','[EMAIL REDACTED]',text);text=re.sub(r'(?<!\w)(?:\+\d[\d ()-]{10,}\d)(?!\w)','[PHONE REDACTED]',text)
    users.append({'source_message_index':i,'user_text':text.strip()})
   rows.append({'source_id':r['conversation_hash'],'row_idx':item['row_idx'],'language':r['language'],'turn':r['turn'],'toxic':r.get('toxic'),'moderation_flagged':any(bool(m.get('flagged')) for m in (r.get('openai_moderation') or []) if isinstance(m,dict)),'users':users[:4]})
  write(dest,{'offset':offset,'url':url,'revision':header,'response_sha256':sha(body),'response_bytes':len(body),'rows':rows})
  print(json.dumps({'offset':offset,'status':'normalized','rows':len(rows)}),flush=True)
 except Exception as e:
  write(D/'fetch-blocker.json',{'offset':offset,'error':repr(e),'partial_windows_retained':True});raise
