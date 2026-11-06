"""Limit predicted CPU hits per source so advice cannot flood ARC's T2."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
for src,label,runner in [('cpu-wait-simple','cpu-wait-bounded','run_cpu_wait_simple_busy.py'),('cpu-wait-cold','cpu-wait-bounded-cold','run_cpu_wait_cold_busy.py')]:
 source=H/'sources'/label;shutil.copytree(H/'sources'/src,source,dirs_exist_ok=True)
 p=source/'semantic_offload/connector.py';s=p.read_text();a='        self._wait_hint_cpu_retention_only = retain_only\n';assert s.count(a)==1
 s=s.replace(a,a+'''        cpu_hint_limit = extra_config.get("wait_hint_cpu_retention_max_blocks", 128)
        if type(cpu_hint_limit) is not int or cpu_hint_limit < 1:
            raise ValueError("wait_hint_cpu_retention_max_blocks must be a positive integer")
        self._wait_hint_cpu_retention_max_blocks = cpu_hint_limit
''')
 a='            if ready:\n                self.manager.touch(ready, ReqContext(\n';assert s.count(a)==1
 s=s.replace(a,'''            ready = ready[:getattr(self, "_wait_hint_cpu_retention_max_blocks", 128)]
'''+a);ast.parse(s);p.write_text(s)
 (source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
 s=(H/runner).read_text().replace("'"+src+"' if a.arm=='candidate'","'"+label+"' if a.arm=='candidate'");(H/('run_'+label.replace('-','_')+'_busy.py')).write_text(s)
print('Prepared bounded128-page CPU retention, whole prefix and GPU-cold prefix; not measured.')
