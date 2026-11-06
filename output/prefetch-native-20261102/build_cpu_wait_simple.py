"""Minimal retention-only action on the unchanged canonical source snapshot."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/cpu-wait-simple';shutil.copytree(H/'sources/candidate',source,dirs_exist_ok=True)
f=source/'semantic_offload/connector.py';old=f.read_text();s=old
a='        self._soft_retention = soft_retention\n';assert s.count(a)==1
s=s.replace(a,a+'''        retain_only = extra_config.get("wait_hint_cpu_retention_only", False)
        if not isinstance(retain_only, bool):
            raise TypeError("wait_hint_cpu_retention_only must be a JSON boolean")
        if retain_only and self._prefetch_trigger != "known_prefix_wait_hint":
            raise ValueError("CPU retention-only requires known-prefix wait hints")
        self._wait_hint_cpu_retention_only = retain_only
''')
a='        if getattr(self, "_wait_hint_coalesce_exact_source", False):\n';assert s.count(a)==1
s=s.replace(a,'''        if getattr(self, "_wait_hint_cpu_retention_only", False):
            # Exact source/generation/namespace and causal clocks were checked
            # above. Recency is refreshed only for the ready CPU prefix; no
            # lease, GPU reservation, transfer, or future request is created.
            ready = _ready_resident_prefix(self.manager, source.keys)
            if ready:
                self.manager.touch(ready, ReqContext(
                    req_id=f"wait-retain:{hint.event_id}",
                    kv_transfer_params={"run_id": hint.run_id, "cell_id": hint.cell_id,
                                        "session_id": hint.session_id, "generation": hint.generation}))
            return self._store_wait_hint_result(
                hint, status="no_action", reason="cpu_prefix_retained",
                now_ns=now_ns, nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns, ready_cpu_prefix_keys=len(ready),
                reserved_gpu_blocks=0, speculative_transfer_bytes=0,
            )
'''+a)
ast.parse(s);f.write_text(s);(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
(H/'cpu-wait-simple-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/semantic_offload/connector.py',tofile='b/semantic_offload/connector.py')))
s=(H/'run_evictable_busy.py').read_text().replace("'evictable-warm-v2' if a.arm=='candidate'","'cpu-wait-simple' if a.arm=='candidate'")
a=" if a.arm=='candidate':cfg['kv_connector_extra_config'].update(wait_hint_ttl_ms=60000,wait_hint_cold_run_publication=True,wait_hint_evictable_warm=True)";assert s.count(a)==1
s=s.replace(a," if a.arm=='candidate':cfg['kv_connector_extra_config'].update(wait_hint_cpu_retention_only=True,shared_prefetch_scorer=False,prefix_mean_prefetch_scoring=False,scoring_schedule='continuous',prefetch_selector='request_order',wait_hint_diagnostic_no_dispatch=True)")
ast.parse(s);(H/'run_cpu_wait_simple_busy.py').write_text(s)
print('Prepared minimal CPU-only wait action on canonical snapshot; not measured.')
