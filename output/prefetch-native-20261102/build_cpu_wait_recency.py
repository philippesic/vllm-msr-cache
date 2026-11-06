"""Test advice that changes recency without pretending it was a demand hit."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/cpu-wait-recency';shutil.copytree(H/'sources/cpu-wait-simple',source,dirs_exist_ok=True)
p=source/'semantic_offload/manager.py';s=p.read_text()
a='    def __init__(\n        self,\n        *args,\n';assert s.count(a)==1
method='''    def refresh_wait_hint_cpu_recency(self, keys: Collection[OffloadKey]) -> int:
        """Refresh ready ARC residents without promoting speculative accesses.

        Hints are advice, not actual cache hits: preserve queue membership,
        ghost history, the adaptive target, ownership and eviction eligibility.
        Return zero for unsupported policies rather than applying hit semantics.
        """
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
        policy = self._policy
        if not isinstance(policy, ARCCachePolicy):
            return 0
        refreshed = 0
        for key in reversed(list(keys)):
            block = policy.get(key)
            if block is None or not block.is_ready:
                continue
            queue = policy.t1 if key in policy.t1 else policy.t2
            queue.move_to_end(key)
            refreshed += 1
        return refreshed

'''
s=s.replace(a,method+a);ast.parse(s);p.write_text(s)
p=source/'semantic_offload/connector.py';s=p.read_text()
a='''            if ready:
                self.manager.touch(ready, ReqContext(
                    req_id=f"wait-retain:{hint.event_id}",
                    kv_transfer_params={"run_id": hint.run_id, "cell_id": hint.cell_id,
                                        "session_id": hint.session_id, "generation": hint.generation}))
''';assert s.count(a)==1
s=s.replace(a,'''            ready_count = self.manager.refresh_wait_hint_cpu_recency(ready)
''').replace('ready_cpu_prefix_keys=len(ready),','ready_cpu_prefix_keys=ready_count,').replace('reason="cpu_prefix_retained"','reason="cpu_prefix_recency_refreshed"')
ast.parse(s);p.write_text(s)
(source/'pins.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')},indent=2)+'\n')
s=(H/'run_cpu_wait_simple_busy.py').read_text().replace("'cpu-wait-simple' if a.arm=='candidate'","'cpu-wait-recency' if a.arm=='candidate'");(H/'run_cpu_wait_recency_busy.py').write_text(s)
print('Prepared CPU advice without demand-frequency promotion; not measured.')
