"""Causal wait-hint cache recency refresh; no reservation or speculative DMA."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/residency-refresh'
shutil.copytree(H/'sources/demand-fallback',source,dirs_exist_ok=True)
f=source/'semantic_offload/connector.py';old=f.read_text();s=old
needle='        self._wait_hint_native_adoption = native_adoption\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''        residency = extra_config.get("wait_hint_residency_only", False)
        if not isinstance(residency, bool):
            raise TypeError("wait_hint_residency_only must be a JSON boolean")
        self._wait_hint_residency_only = residency
''')
needle='        if getattr(self, "_wait_hint_coalesce_exact_source", False):\n'
assert s.count(needle)==1
s=s.replace(needle,'''        if getattr(self, "_wait_hint_residency_only", False):
            refreshed = self._refresh_wait_hint_gpu_residency(source)
            return self._store_wait_hint_result(
                hint, status="no_action", reason="gpu_residency_refreshed",
                now_ns=now_ns, nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns, refreshed_gpu_blocks=refreshed,
                reserved_gpu_blocks=0, speculative_transfer_bytes=0,
            )
'''+needle)
needle="    @profiled('scheduler.prefetch_submit')\n    def _wait_staging_candidates(self, source_keys, local_count):\n"
assert s.count(needle)==1
s=s.replace(needle,'''    def _refresh_wait_hint_gpu_residency(self, source):
        # Called only after exact namespace/generation/source/causal-clock and
        # nomination-window validation. All keys are from a completed request.
        # Native touch/free changes recency only: refs, hashes, free capacity,
        # allocation epochs and KV bytes are unchanged at return. Positive-ref
        # pages already have demand owners and are left entirely alone.
        pool = self.gpu_block_pool
        if (len(self.config.kv_group_configs) != 1
                or config_blocks_per_chunk(self.config) != 1
                or not getattr(pool, "enable_caching", False)
                or not callable(getattr(pool, "get_cached_block", None))):
            return 0
        blocks = []
        seen = set()
        for key in source.keys:
            if get_offload_group_idx(key) != 0:
                return 0
            found = pool.get_cached_block(get_offload_block_hash(key), [0])
            if not found:
                continue
            block = found[0]
            if (block.is_null or block.block_hash != key or block.ref_cnt < 0
                    or pool.blocks[block.block_id] is not block):
                return 0
            if block.ref_cnt == 0 and block.block_id not in seen:
                seen.add(block.block_id)
                blocks.append(block)
        if not blocks:
            return 0
        pool.touch(blocks)
        pool.free_blocks(reversed(blocks))
        return len(blocks)

'''+needle)
ast.parse(s);f.write_text(s)
(H/'residency-refresh-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='demand-fallback/connector.py',tofile='residency-refresh/connector.py')))
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
for mode,base in [('teacher','run_teacher_replay.py'),('natural','run_teacher_replay.py')]:
 s=(H/base).read_text()
 old_source="'demand-fallback' if a.arm=='candidate'"
 assert old_source in s
 s=s.replace(old_source,"'residency-refresh' if a.arm=='candidate'")
 needle=" if a.arm=='candidate':cfg['kv_connector_extra_config'].update("
 at=s.index(needle);end=s.index('\n',at)
 s=s[:end]+"\n if a.arm=='candidate':cfg['kv_connector_extra_config']['wait_hint_residency_only']=True"+s[end:]
 if mode=='natural':
  s=s.replace('teacher_workload','short_cadence_workload').replace('hooks-teacher-replay','hooks-idle-drain').replace("'recorded_token_replay_not_unconstrained_generation'","'ordinary_generation'")
 ast.parse(s);(H/f'run_residency_{mode}.py').write_text(s)
print('Prepared residency-only candidate; no measurement yet.')
