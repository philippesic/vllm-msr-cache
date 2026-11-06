from pathlib import Path
H=Path(__file__).resolve().parent
helper='''"""One gateway clock; fixed credit for released continuation requests."""
def gateway_priority(submitted_ns, origin_ns, turn, credit_ms):
    for name, value in (("submitted_ns", submitted_ns), ("origin_ns", origin_ns), ("turn", turn), ("credit_ms", credit_ms)):
        if type(value) is not int:
            raise TypeError(name + " must be an integer")
    if submitted_ns < origin_ns or origin_ns < 0 or turn < 0 or credit_ms not in (0, 2000, 8000):
        raise ValueError("Invalid gateway clock, turn, or fixed priority credit")
    if credit_ms == 0:
        return 0
    return (submitted_ns - origin_ns) // 1_000_000 - (credit_ms if turn > 0 else 0)
'''
p=H/'return_priority_nov3.py';assert not p.exists();p.write_text(helper)
s=(H/'live_service_workload_nov3.py').read_text().replace('import asyncio,hashlib,json,random,time','import asyncio,hashlib,json,random,time\nfrom return_priority_nov3 import gateway_priority',1)
needle="   row={'index':index,'role':a.cohort+('-return' if turn else '-initial')"
assert s.count(needle)==1
s=s.replace(needle,"   submitted_ns=time.monotonic_ns()\n   body['priority']=gateway_priority(submitted_ns,result['load_start_ns'],turn,a.return_priority_ms)\n"+needle,1)
s=s.replace("'request_start_ns':time.monotonic_ns(),'payload':body", "'request_start_ns':submitted_ns,'priority_clock_ns':submitted_ns,'priority_origin_ns':result['load_start_ns'],'priority_credit_ms':a.return_priority_ms,'payload':body",1)
needle="'policy_refresh_ms':a.hint_interval_ms,"
assert s.count(needle)==1;s=s.replace(needle,needle+"'request_priority_policy':'native_priority_FIFO_allzero' if a.return_priority_ms==0 else 'gateway_submission_ms_minus_fixed_return_credit','return_priority_credit_ms':a.return_priority_ms,'priority_clock':'single_gateway_host_monotonic_relative_to_load_origin','priority_fairness':'A later return submitted more than credit_ms after an initial cannot overtake that initial by priority; no wall-clock service bound under overload.',")
p=H/'live_return_priority_workload_nov3.py';assert not p.exists();p.write_text(s)
s=(H/'run_live_integrated_gpu_gate_nov3.py').read_text().replace('live_service_workload_nov3','live_return_priority_workload_nov3')
s=s.replace("'source_pins':{str(f.relative_to(source))", "'source_pins':{str(f.relative_to(source))",1)
needle="H/'live_return_priority_workload_nov3.py',H/'hooks-live-lean/sitecustomize.py'";assert s.count(needle)==1;s=s.replace(needle,"H/'live_return_priority_workload_nov3.py',H/'return_priority_nov3.py',H/'hooks-live-lean/sitecustomize.py'",1)
needle="['--served-model-name','matrix-model'";assert s.count(needle)==1;s=s.replace(needle,"['--scheduling-policy','priority','--served-model-name','matrix-model'",1)
s=s.replace("ap=argparse.ArgumentParser();", "ap=argparse.ArgumentParser();ap.add_argument('--return-priority-ms',type=int,choices=[0,2000,8000],default=0);",1)
p=H/'run_live_return_priority_nov3.py';assert not p.exists();p.write_text(s)
print('Separate helper, workload and runner created; existing sources/runtimes untouched.')
