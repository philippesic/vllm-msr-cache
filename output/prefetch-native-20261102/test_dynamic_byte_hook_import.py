"""The connector is imported before gpu_worker adds the audit module path."""
import os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent

def probe(directory):
 return subprocess.run([sys.executable,'-B','-c','import byte_audit_dynamic; assert callable(byte_audit_dynamic.patch)'],cwd='/tmp',env=dict(os.environ,PYTHONPATH=str(H/directory)),capture_output=True,text=True)

def test_fixed_hook_can_resolve_audit_module_before_gpu_worker_import():
 p=probe('hooks-dynamic-byte-v2');assert p.returncode==0,p.stderr

def test_original_import_order_failure_is_detected_by_control():
 p=probe('hooks-dynamic-byte');assert p.returncode!=0 and "No module named 'byte_audit_dynamic'" in p.stderr
