"""Retain CPU backing only beyond the exact resident GPU prefix."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
for src,label,runner in [('cpu-wait-simple','cpu-wait-cold','run_cpu_wait_simple_busy.py'),('cpu-wait-recency','cpu-wait-cold-recency','run_cpu_wait_recency_busy.py')]:
 source=H/'sources'/label;shutil.copytree(H/'sources'/src,source,dirs_exist_ok=True)
 p=source/'semantic_offload/connector.py';s=p.read_text();a='            ready = _ready_resident_prefix(self.manager, source.keys)\n';assert s.count(a)==1
 s=s.replace(a,'''            local_count, _, coverage = _local_gpu_prefix(self.gpu_block_pool, source.keys)
            ready = (_ready_resident_prefix(self.manager, source.keys[local_count:])
                     if coverage else [])
''');ast.parse(s);p.write_text(s)
 (source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
 s=(H/runner).read_text().replace("'"+src+"' if a.arm=='candidate'","'"+label+"' if a.arm=='candidate'");(H/('run_'+label.replace('-','_')+'_busy.py')).write_text(s)
print('Prepared CPU-only cold-prefix retention and recency-only ablation; not measured.')
