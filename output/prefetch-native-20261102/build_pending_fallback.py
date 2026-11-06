"""Actual demand never waits for an optional nomination with no started DMA."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/demand-fallback';shutil.copytree(H/'sources/demand-capacity',source)
f=source/'semantic_offload/connector.py';old=f.read_text();s=old
needle='''        if (
            state.status == "pending"
            and not state.coalescing_complete
            and self._wait_hint_coalescing_enabled()
        ):'''
assert s.count(needle)==1
start=s.index(needle);end=s.index('        state.bound_request_id = request.request_id',start)
s=s[:start]+'''        # Identity and generation checks above remain mandatory. No transfer
        # has started for a pending nomination, so actual demand can take the
        # ordinary exact load path immediately instead of waiting out the
        # speculative nomination deadline. In-flight copies retain all fences.
        if (state.status == "pending" and state.job_id is None
                and not state.gpu_block_ids and state.pending_contributors == 0):
            self._free_wait_prefetch(state, "demand_arrived_before_dispatch")
            record_wait_hint_event({
                "event": "resume_binding_rejected", "event_id": event_id,
                "request_id": request.request_id,
                "reason": "undispatched_nomination_demand_fallback",
                "observed_at_ns": now_ns,
            })
            return
'''+s[end:]
ast.parse(s);f.write_text(s);(H/'pending-fallback-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='demand-capacity/connector.py',tofile='demand-fallback/connector.py')))
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
s=(H/'run_demand_capacity.py').read_text().replace("'demand-capacity' if a.arm=='candidate'","'demand-fallback' if a.arm=='candidate'");(H/'run_demand_fallback.py').write_text(s)
print('Prepared pending-nomination demand fallback; in-flight/ready handling unchanged.')
