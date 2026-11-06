"""Private event-driven idle staging experiment; no inference path changes."""
import ast,difflib,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent
source=H/'sources/idle-ready';runtime=H/'runtime-idle-ready'
shutil.copytree(H/'sources/fused-mean',source);shutil.copytree(H/'runtime-shared-lookup',runtime)
p=source/'semantic_offload/connector.py';old=p.read_text();needle='    def has_pending_push_work(self) -> bool:\n';assert old.count(needle)==1
method='''    def ready_staging_idle_timeout(self) -> float | None:
        """Bound the next idle wake by TTL; incoming queue messages wake sooner.

        Called only by an idle native core. Pending DMA, cleanup, nominations,
        request ownership and diagnostic timing all require ordinary polling.
        """
        if (os.environ.get("SEMANTIC_READY_IDLE_WAIT") != "1"
                or super().has_pending_push_work()
                or getattr(self, "_wait_prefetch_jobs", {})
                or getattr(self, "_pending_splice_releases", {})
                or getattr(self, "_bounded_prefill_cuda_timing", False)):
            return None
        states = list(getattr(self, "_wait_prefetch", {}).values())
        if not states or any(s.status != "ready" or s.bound_request_id is not None
                             for s in states):
            return None
        now = time.monotonic_ns()
        remaining = min(s.lease.expires_at_ns for s in states) - now
        return min(0.05, max(0.0, remaining / 1_000_000_000))

'''
s=old.replace(needle,method+needle);ast.parse(s);p.write_text(s)
(H/'idle-ready-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='fused-mean/connector.py',tofile='idle-ready/connector.py')))
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source/'semantic_offload').glob('*.py')},indent=2)+'\n')
p=runtime/'vllm/v1/engine/core.py';old=p.read_text();needle='''        waited = False
        while not self.has_work() and self.is_running():''';assert old.count(needle)==1
s=old.replace(needle,'''        # Ready speculative pages need expiry timers, not empty GPU steps.
        # Queue.get wakes immediately for real inputs, cancellation or shutdown.
        # Keep ordinary polling for all transfer/cleanup and live-request work.
        if (self.process_input_queue_block and not self.engines_running
                and not self.batch_queue and self.scheduler.requests == {}
                and self.is_running()):
            connector = self.scheduler.get_kv_connector()
            inner = getattr(connector, "connector_scheduler", None)
            idle_timeout = getattr(inner, "ready_staging_idle_timeout", None)
            timeout = idle_timeout() if idle_timeout is not None else None
            if timeout is not None and timeout > 0:
                try:
                    self._handle_client_request(*self.input_queue.get(timeout=timeout))
                except queue.Empty:
                    pass

        waited = False
        while not self.has_work() and self.is_running():''');ast.parse(s);p.write_text(s)
(H/'idle-ready-runtime.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='shared-lookup/core.py',tofile='idle-ready/core.py')))
old=(H/'run_fused_long_lease.py').read_text();s=old.replace("'fused-mean' if a.arm=='candidate'", "'idle-ready' if a.arm=='candidate'")
s=s.replace(" runtime_root=H/(('runtime-'+a.source_label) if a.source_label.startswith('shared-') else 'runtime')", " runtime_root=H/'runtime-idle-ready'")
s=s.replace(" env['SEMANTIC_FUSED_MEAN_SUMMARY']='1' if a.arm=='candidate' else '0'", " env['SEMANTIC_FUSED_MEAN_SUMMARY']='1' if a.arm=='candidate' else '0'\n env['SEMANTIC_READY_IDLE_WAIT']='1' if a.arm=='candidate' else '0'")
(H/'run_idle_ready.py').write_text(s);print('Private idle candidate built')
