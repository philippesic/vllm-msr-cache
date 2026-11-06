"""Port existing full-prefix capacity/priority shortcuts to prefix-mean ranking."""
import ast,difflib,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent
source=H/'sources/opportunity-gates';shutil.copytree(H/'sources/fused-mean',source)
f=source/'semantic_offload/connector.py';old=f.read_text();needle='''    def _retry_prefix_mean_prefetches(self, pending, now_ns):
        ranked, descriptors = [], {}
        for state in pending:
''';assert old.count(needle)==1
s=old.replace(needle,'''    def _retry_prefix_mean_prefetches(self, pending, now_ns):
        # As in the per-key selector, optional copies cannot act during
        # admitted prefill or without capacity. Avoid building per-key CPU
        # descriptors and GPU score requests until a full candidate can fit.
        full = getattr(self, '_prefetch_require_full_prefix', False)
        if full and self._admitted_prefill_in_progress():
            PROFILE.count('scheduler.prefix_mean_blocked_prefill')
            return
        pool = self.gpu_block_pool
        available = min(
            max(0, int(pool.num_gpu_blocks * self._prefetch_budget_fraction)
                - self._prefetch_reserved_blocks),
            max(0, self._prefetch_free_capacity() - self._prefetch_headroom_blocks),
        ) if pool is not None else 0
        if available <= 0:
            PROFILE.count('scheduler.prefix_mean_blocked_capacity')
            return
        ranked, descriptors = [], {}
        for state in pending:
            if full and len(state.publication_candidate_keys) > available:
                PROFILE.count('scheduler.prefix_mean_full_source_cannot_fit')
                continue
''');ast.parse(s);f.write_text(s)
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source/'semantic_offload').glob('*.py')},indent=2)+'\n')
(H/'opportunity-gates.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='fused-mean/connector.py',tofile='opportunity-gates/connector.py')))
print('Prepared full-prefix opportunity shortcut; source and actual submit checks unchanged')
