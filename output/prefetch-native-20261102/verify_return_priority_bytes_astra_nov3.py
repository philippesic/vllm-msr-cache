"""Selected8s nativepriority byte qualification, independent operational receipt."""
import gzip,json
from pathlib import Path
from verify_integrated_gate_bytes_astra_nov3 import verify_bytes as verify_base
from return_priority_receipt_astra_nov3 import verify_priority

def verify_bytes(path):
 r=json.load(gzip.open(path,'rt'));out=verify_base(path);candidate=r['args']['arm']=='candidate';receipt=verify_priority(r,{'gate_enabled':candidate,'priority_credit_ms':8000});out['priority_receipt']=receipt;out['schema']='astra-selected-return-priority-byte-qualification-v1'
 if not receipt['qualified']:out['issues'].append('priority_receipt_failed')
 if r.get('performance_eligible') is not False or r['args'].get('audit') is not True:out['issues'].append('intrusive_mode_not_explicit')
 out['qualified']=not out['issues'];return out
