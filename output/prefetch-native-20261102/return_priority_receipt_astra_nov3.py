"""Independent arithmetic and archive checks for causal return priority discovery."""
import hashlib
from pathlib import Path
from gate_once_receipt_astra_nov3 import verify_gpu_gate
H=Path(__file__).resolve().parent

def verify_priority(raw,item):
 gate=item['gate_enabled']; credit=item['priority_credit_ms'];out=verify_gpu_gate(raw,'gate' if gate else 'native');issues=out['issues']
 args=raw.get('args',{});cfg=raw.get('config',{}).get('kv_connector_extra_config',{})
 if args.get('return_priority_ms')!=credit:issues.append('credit_argument_mismatch')
 if gate and (cfg.get('preservation_audit') is not False or args.get('no_audit') is not True):issues.append('bounded_candidate_mode_mismatch')
 command=raw.get('server_command',[])
 if isinstance(command,str):import shlex;command=shlex.split(command)
 if '--scheduling-policy' not in command or command[command.index('--scheduling-policy')+1]!='priority':issues.append('native_priority_scheduler_not_executed')
 archives=[]
 for label,source in [('workload.py','live_return_priority_workload_nov3.py'),('return_priority_helper_astra.py','return_priority_nov3.py')]:
  p=H/source;q=H/'runs'/raw['name']/label;expected=hashlib.sha256(p.read_bytes()).hexdigest();actual=hashlib.sha256(q.read_bytes()).hexdigest() if q.exists() else None
  archives.append({'source':str(p),'archive':str(q),'source_hash':expected,'archive_hash':actual,'raw_pin':raw.get('pins',{}).get(source)})
  if actual!=expected or raw.get('pins',{}).get(source)!=expected:issues.append('priority_archive_or_raw_pin_mismatch:'+source)
 checked=[]
 for r in raw.get('requests',[]):
  b=r.get('payload',{}); t=r.get('turn');submitted=r.get('priority_clock_ns');origin=r.get('priority_origin_ns')
  if not all(type(v) is int for v in [t,submitted,origin]) or submitted<origin or origin!=raw.get('load_start_ns') or submitted!=r.get('request_start_ns') or r.get('priority_credit_ms')!=credit:issues.append('priority_clock_contract:'+str(r.get('index')));continue
  expected=0 if credit==0 else (submitted-origin)//1000000-(credit if t>0 else 0)
  if b.get('priority')!=expected:issues.append('priority_formula:'+str(r['index']))
  if b.get('max_tokens')!=raw['geometry']['max_model_len']-len(b.get('prompt',[])) or b.get('ignore_eos',False) is not False or b.get('stop') or b.get('stop_token_ids'):issues.append('generation_contract:'+str(r['index']))
  checked.append({'index':r['index'],'turn':t,'submitted_ns':submitted,'origin_ns':origin,'expected_priority':expected,'actual_priority':b.get('priority')})
 if any(h.get('producer_policy_stop') for h in raw.get('hints',[])):issues.append('unexpected_once_producer_stop')
 out.update(qualified=out['cadence']['qualified'] and not issues,priority_credit_ms=credit,priority_rows=checked,archive_receipts=archives,priority_scope='Shared native priority mode also affects preemption victims; no hard service guarantee or cache-hit classification. Helper copy is supervisor-owned, original raw pins bind executed module.')
 return out
