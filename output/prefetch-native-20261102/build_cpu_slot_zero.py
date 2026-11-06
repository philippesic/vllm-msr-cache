"""CPU slot zero is data; GPU block zero remains the null sentinel."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
for src,label in [('candidate','arc-cpu-slot0'),('cpu-wait-recency-v2','cpu-wait-recency-v3')]:
 root=H/'sources'/label;shutil.copytree(H/'sources'/src,root,dirs_exist_ok=True)
 f=root/'semantic_offload/connector.py';s=f.read_text();a='from vllm.v1.outputs import KVConnectorOutput\n';assert s.count(a)==1
 s=s.replace(a,'from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec\n'+a)
 a='    if not group_sizes:\n        return [(0, block_id) for block_id in block_ids if block_id]\n';assert s.count(a)==1
 s=s.replace(a,'''    if isinstance(spec, CPULoadStoreSpec):
        # CPU slot zero contains ordinary KV. Only GPU block zero is null.
        return [(0, block_id) for block_id in block_ids]
'''+a);ast.parse(s);f.write_text(s)
 (root/'pins.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')},indent=2)+'\n')
s=(H/'run_cpu_wait_recency_v2_busy.py').read_text().replace("'cpu-wait-recency-v2' if a.arm=='candidate' else a.source_label","'cpu-wait-recency-v3' if a.arm=='candidate' else 'arc-cpu-slot0'");assert s!=(H/'run_cpu_wait_recency_v2_busy.py').read_text();(H/'run_cpu_wait_recency_v3_busy.py').write_text(s)
s=(H/'run_cpu_wait_recency_v2_audit_v2.py').read_text().replace("'cpu-wait-recency-v2' if a.arm=='candidate' else a.source_label","'cpu-wait-recency-v3' if a.arm=='candidate' else 'arc-cpu-slot0'");(H/'run_cpu_wait_recency_v3_audit.py').write_text(s)
print('Prepared identicalCPU-slot0 recording fix in botharms; no transfer or GPU-null logic changed.')
