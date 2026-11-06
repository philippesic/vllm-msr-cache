"""Follow-up candidate only; never modifies the running frozen confirmation."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/idle-pending';runtime=H/'runtime-idle-pending';hooks=H/'hooks-idle-pending'
shutil.copytree(H/'sources/idle-ready-v2',source);shutil.copytree(H/'runtime-idle-ready',runtime);shutil.copytree(H/'hooks-defer',hooks)
f=source/'semantic_offload/connector.py';s=f.read_text();a=s.index('    def ready_staging_idle_timeout(');b=s.index('    def has_pending_push_work(',a)
s=s[:a]+'''    def ready_staging_idle_timeout(self) -> float | None:
        """Wait for input or the next deadline when all transfer work is done.

        Pending nominations can retry on the next input or within50ms; live DMA,
        cleanup and request work always keep ordinary completion polling.
        """
        import os
        if (os.environ.get("SEMANTIC_READY_IDLE_WAIT") != "1"
                or super().has_pending_push_work()
                or getattr(self, "_wait_prefetch_jobs", {})
                or getattr(self, "_pending_splice_releases", {})
                or getattr(self, "_bounded_prefill_cuda_timing", False)):
            return None
        states = list(getattr(self, "_wait_prefetch", {}).values())
        if not states or any(s.status not in {"ready", "pending"}
                             or s.bound_request_id is not None for s in states):
            return None
        now = time.monotonic_ns()
        deadlines = [s.lease.nomination_deadline_ns if s.status == "pending"
                     else s.lease.expires_at_ns for s in states]
        if any(s.lease.state in {"cancelled", "expired"} for s in states):
            return 0.0
        return min(0.05, max(0.0, (min(deadlines) - now) / 1_000_000_000))

'''+s[b:];ast.parse(s);f.write_text(s)
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source/'semantic_offload').glob('*.py')},indent=2)+'\n')
f=runtime/'vllm/v1/engine/core.py';s=f.read_text();s=s.replace('''            if timeout is not None and timeout > 0:
                try:
                    self._handle_client_request(*self.input_queue.get(timeout=timeout))
                except queue.Empty:
                    pass''','''            if timeout is not None and timeout > 0:
                self.semantic_staging_idle_wait_calls = getattr(
                    self, "semantic_staging_idle_wait_calls", 0) + 1
                try:
                    self._handle_client_request(*self.input_queue.get(timeout=timeout))
                    self.semantic_staging_idle_input_wakes = getattr(
                        self, "semantic_staging_idle_input_wakes", 0) + 1
                except queue.Empty:
                    self.semantic_staging_idle_timer_wakes = getattr(
                        self, "semantic_staging_idle_timer_wakes", 0) + 1''');ast.parse(s);f.write_text(s)
f=hooks/'sitecustomize.py';s=f.read_text();needle="            return state()\n        m.EngineCore.matrix_export=export";assert s.count(needle)==1
s=s.replace(needle,"""            value=state()
            value['idle_loop']={key:getattr(self,key,0) for key in ['semantic_staging_idle_wait_calls','semantic_staging_idle_input_wakes','semantic_staging_idle_timer_wakes']}
            return value
        m.EngineCore.matrix_export=export""");f.write_text(s)
s=(H/'run_warm_confirm_v2.py').read_text().replace("'idle-ready-v2' if a.arm=='candidate'", "'idle-pending' if a.arm=='candidate'").replace("H/'runtime-idle-ready'", "H/'runtime-idle-pending'").replace("else 'hooks-defer'", "else 'hooks-idle-pending'");(H/'run_idle_pending.py').write_text(s)
t=(H/'test_idle_ready_v2.py').read_text().replace('sources/idle-ready-v2/','sources/idle-pending/').replace('runtime-idle-ready/','runtime-idle-pending/').replace("['pending','in_flight'", "['in_flight'").replace("lease=types.SimpleNamespace(expires_at_ns=61_000_000_000)", "lease=types.SimpleNamespace(expires_at_ns=61_000_000_000,nomination_deadline_ns=1_750_000_000,state='ready')")
t+='''\n\ndef test_pending_nomination_bounds_idle_wake(ready):
 ready._wait_prefetch['e'].status='pending'
 assert ready.ready_staging_idle_timeout()==.05
 ready._wait_prefetch['e'].lease.nomination_deadline_ns=1_003_000_000
 assert ready.ready_staging_idle_timeout()==.003
 ready._wait_prefetch['e'].lease.nomination_deadline_ns=999_000_000
 assert ready.ready_staging_idle_timeout()==0

@pytest.mark.parametrize('state',['cancelled','expired'])
def test_terminal_lease_needs_immediate_cleanup(ready,state):
 ready._wait_prefetch['e'].lease.state=state
 assert ready.ready_staging_idle_timeout()==0
''';(H/'test_idle_pending.py').write_text(t)
d=json.loads((H/'WARM-PILOT-V2.json').read_text());d['progress_file']='idle-pending-progress.json';r=d['runs'][0];old=r['name'];r['name']='idle-pending-b512-candidate-s1103';r['command']=[x.replace(old,r['name']).replace('run_warm_confirm_v2.py','run_idle_pending.py') for x in r['command']]
for f in [H/'run_idle_pending.py',H/'test_idle_pending.py',*(source).rglob('*.py'),source/'pins.json',*(runtime).rglob('*.py'),hooks/'sitecustomize.py']:d['pins'][str(f.relative_to(H))]=hashlib.sha256(f.read_bytes()).hexdigest()
d['objective']='Exploratory idle polling follow-up, separatefromfrozen12runconfirmation: allow idlependingnominations too, timerbounded bynextnominationdeadline<=50ms; allnativepushjobs/copyfences/cleanup/live requestsstillordinarypoll. Passivecorewait/wakecountsproveactivation. Samefused/60s/.95config andneutralwarmup asconfirmation. No broadthroughputclaim or performanceevidence untilserving runs.';(H/'IDLE-PENDING.json').write_text(json.dumps(d,indent=2)+'\n');print('Prepared idlependingfollow-up without changing activeconfirmation.')
