"""Best-effort post-failure diagnostics, preserving original failure status/error."""
import time
import urllib.request
from runtime_support import control

def capture(handle,result):
    receipt={'started_ns':time.monotonic_ns(),'operations':{},'scope':'After workload task cleanup only; no reset, no new generation, original failure unchanged.'}
    result['failure_final_capture']=receipt
    if 'postload_export' not in result:
        try:
            result['postload_export']=control(handle,'export')
            receipt['operations']['export']={'success':True}
        except Exception as exc:receipt['operations']['export']={'success':False,'error':repr(exc)}
    if 'metrics_final' not in result:
        try:
            with urllib.request.urlopen(handle.metrics_url(),timeout=30) as response:result['metrics_final']=response.read().decode()
            receipt['operations']['metrics']={'success':True}
        except Exception as exc:receipt['operations']['metrics']={'success':False,'error':repr(exc)}
    receipt['completed_ns']=time.monotonic_ns()
