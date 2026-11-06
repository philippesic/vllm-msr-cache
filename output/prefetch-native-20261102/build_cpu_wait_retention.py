"""Keep exact ready CPU prefixes recent while a validated wait is pending."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/cpu-wait-retention';shutil.copytree(H/'sources/decode-headroom',source,dirs_exist_ok=True)
f=source/'semantic_offload/connector.py';s=f.read_text()
a='        self._prefetch_reserve_decode_budget = reserve_decode\n';assert s.count(a)==1
s=s.replace(a,a+'''        cpu_retain = extra_config.get("wait_hint_retain_cpu_prefix", False)
        if not isinstance(cpu_retain, bool):
            raise TypeError("wait_hint_retain_cpu_prefix must be a JSON boolean")
        self._wait_hint_retain_cpu_prefix = cpu_retain
''')
a='        if getattr(self, "_wait_hint_coalesce_exact_source", False):\n';assert s.count(a)==1
s=s.replace(a,'''        if getattr(self, "_wait_hint_retain_cpu_prefix", False):
            # Source identity, namespace, generation, causal clocks and the
            # public nomination deadline were validated above. A valid hint
            # supplies evidence of possible reuse even while all GPU pages
            # are still cached or no staging reservation can fit.
            ready = _ready_resident_prefix(self.manager, source.keys)
            if ready:
                self.manager.touch(ready, ReqContext(
                    req_id=f"wait-retain:{hint.event_id}",
                    kv_transfer_params={"run_id": hint.run_id, "cell_id": hint.cell_id,
                                        "session_id": hint.session_id, "generation": hint.generation}))
                record_wait_hint_event({"event": "hint_cpu_retention", "event_id": hint.event_id,
                    "at_ns": now_ns, "ready_prefix_keys": len(ready),
                    "source_key_count": len(source.keys), "reservation_bytes": 0})
'''+a)
ast.parse(s);f.write_text(s);(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
s=(H/'run_decode_headroom_busy.py').read_text().replace("'decode-headroom' if a.arm=='candidate'","'cpu-wait-retention' if a.arm=='candidate'").replace('prefetch_reserve_decode_budget=True)','prefetch_reserve_decode_budget=True,wait_hint_retain_cpu_prefix=True)');ast.parse(s);(H/'run_cpu_wait_retention_busy.py').write_text(s)
print('Prepared causal CPU wait-retention + guarded evictable GPU prefetch. Not measured.')
