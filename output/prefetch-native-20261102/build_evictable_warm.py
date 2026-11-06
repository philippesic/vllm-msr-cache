"""Publish fenced observed KV into the ordinary evictable GPU hash cache."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/evictable-warm';runtime=H/'runtime-evictable-warm'
shutil.copytree(H/'sources/demand-fallback',source,dirs_exist_ok=True)
shutil.copytree(H/'runtime-cold-run',runtime,dirs_exist_ok=True)
def once(s,a,b):
 assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)
f=runtime/'vllm/v1/core/block_pool.py';old=f.read_text()
s=once(old,'    def publish_completed_prefix(\n','''    def publish_completed_observed_keys(
        self, keys, blocks, *, start_index, block_size, completion_verified,
    ) -> bool:
        """Cache exact hashes from an observed, fenced dense single-group copy.

        No future Request or tokens are synthesized. Parent-chain hashes and
        offsets come from the caller's validated completed-source key chain.
        Events that require original request tokens are deliberately unsupported.
        Validate every page before making any new hash visible.
        """
        if (completion_verified is not True or not self.enable_caching
                or self.enable_kv_cache_events or block_size != self.hash_block_size
                or type(start_index) is not int or start_index < 0
                or not keys or len(keys) != len(blocks)
                or len(set(keys)) != len(keys)):
            return False
        ids = [b.block_id for b in blocks]
        if len(set(ids)) != len(ids):
            return False
        for key, block in zip(keys, blocks):
            if (not isinstance(key, bytes) or len(key) <= 4 or get_group_id(key) != 0
                    or not 0 < block.block_id < self.num_gpu_blocks
                    or self.blocks[block.block_id] is not block or block.is_null
                    or block.ref_cnt != 1 or block.block_hash is not None
                    or block.block_id in self.cached_block_hashes_by_block
                    or self.get_cached_block(get_block_hash(key), [0]) is not None):
                return False
        try:
            for i, (key, block) in enumerate(zip(keys, blocks)):
                self._insert_block_hash(key, block,
                    num_tokens=(start_index + i + 1) * block_size)
        except Exception:
            self.evict_blocks(set(ids))
            raise
        self.free_blocks(reversed(blocks))
        return True

    def publish_completed_prefix(
''')
ast.parse(s);f.write_text(s);(H/'evictable-warm-runtime.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='cold-run/block_pool.py',tofile='evictable-warm/block_pool.py')))
f=source/'semantic_offload/connector.py';old=f.read_text();s=old
s=once(s,'    destination_released: bool = False\n','    destination_released: bool = False\n    warmed_allocation_epochs: tuple[int, ...] = ()\n')
needle='        self._wait_hint_cold_run_publication = cold_run\n'
s=once(s,needle,needle+'''        warm = extra_config.get("wait_hint_evictable_warm", False)
        if not isinstance(warm, bool):
            raise TypeError("wait_hint_evictable_warm must be a JSON boolean")
        if warm and not cold_run:
            raise ValueError("evictable warm requires cold-run exact native publication")
        self._wait_hint_evictable_warm = warm
''')
needle='    def _wait_local_prefix_valid(self, state):\n'
s=once(s,needle,needle+'''        if getattr(state, "warmed_allocation_epochs", ()):
            return self._warmed_wait_prefix_valid(state)
''')
needle="    @profiled('scheduler.native_publish')\n"
s=once(s,needle,'''    def _warmed_wait_prefix_valid(self, state):
        epochs = getattr(state, "warmed_allocation_epochs", ())
        if (not state.destination_released or len(epochs) != len(state.gpu_block_ids)
                or not epochs):
            return False
        pool = self.gpu_block_pool
        for key, block_id, epoch in zip(state.keys, state.gpu_block_ids, epochs):
            found = pool.get_cached_block(get_offload_block_hash(key), [0])
            if (not found or found[0].block_id != block_id
                    or found[0].block_hash != key or found[0].ref_cnt < 0
                    or pool.semantic_allocation_epochs[block_id] != epoch):
                return False
        count = state.publication_local_gpu_prefix_count
        return all(pool.get_cached_block(get_offload_block_hash(key), [0])
                   for key in state.source.keys[:count])

    def _warm_completed_wait_prefetch(self, state):
        if (not getattr(self, "_wait_hint_evictable_warm", False)
                or state.bound_request_id is not None or state.status != "ready"
                or not state.source_released or state.pending_contributors != 0
                or state.destination_released):
            return False
        record = state.transfer_record or {}
        now = time.monotonic_ns()
        hint = state.lease.hint
        if (record.get("success") is not True or record.get("cancelled") is not False
                or record.get("contributors") != 1 or record.get("expected_contributors") != 1
                or type(record.get("completed_at_ns")) is not int
                or not 0 < record["completed_at_ns"] <= now
                or self._observed_prefix_generations.get((hint.run_id, hint.cell_id, hint.session_id)) != hint.generation
                or now > state.lease.expires_at_ns):
            return False
        start = state.publication_local_gpu_prefix_count
        if (not state.keys or tuple(state.keys) != tuple(state.source.keys[start:start + len(state.keys)])
                or len(state.keys) != len(state.gpu_block_ids)
                or len(self.config.kv_group_configs) != 1
                or config_blocks_per_chunk(self.config) != 1):
            return False
        pool = self.gpu_block_pool
        publish = getattr(pool, "publish_completed_observed_keys", None)
        if not callable(publish):
            return False
        blocks = [pool.blocks[i] for i in state.gpu_block_ids]
        if not publish(list(state.keys), blocks, start_index=start,
                       block_size=group_config_tokens_per_block(self.config.kv_group_configs[0]),
                       completion_verified=True):
            return False
        state.warmed_allocation_epochs = tuple(pool.semantic_allocation_epochs[i] for i in state.gpu_block_ids)
        state.destination_released = True
        self._prefetch_reserved_blocks -= len(blocks)
        self._release_wait_prefetch_protection(state)
        record_wait_hint_event({"event": "staging_cache_warmed", "event_id": hint.event_id,
            "observed_at_ns": now, "block_ids": list(state.gpu_block_ids),
            "allocation_epochs": list(state.warmed_allocation_epochs),
            "released_reservation_blocks": len(blocks), "ownership": "native_prefix_cache_evictable"})
        return True

'''+needle)
s=once(s,'        ids = list(state.publication_local_gpu_block_ids) + suffix_ids if shared else suffix_ids\n','''        local_ids = list(state.publication_local_gpu_block_ids)
        warmed = bool(getattr(state, "warmed_allocation_epochs", ()))
        if warmed and shared:
            local_ids = [self.gpu_block_pool.get_cached_block(get_offload_block_hash(key), [0])[0].block_id
                         for key in expected_keys[:local_count]]
        ids = local_ids + suffix_ids if shared else suffix_ids
''')
s=once(s,'        if shared:\n            published = publish(\n','''        if warmed:
            published = (not request.skip_reading_prefix_cache
                         and self._warmed_wait_prefix_valid(state))
        elif shared:
            published = publish(
''')
s=once(s,'        state.destination_released = True\n        self._prefetch_reserved_blocks -= len(suffix_ids)\n','''        if not state.destination_released:
            state.destination_released = True
            self._prefetch_reserved_blocks -= len(suffix_ids)
''')
s=once(s,'            state.status = "ready"\n            state.lease.state =','            state.status = "ready"\n            self._warm_completed_wait_prefetch(state)\n            state.lease.state =')
s=once(s,'                and state.status == "ready"\n            ),\n','                and state.status == "ready"\n                and not state.destination_released\n            ),\n')
ast.parse(s);f.write_text(s);(H/'evictable-warm-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='demand-fallback/connector.py',tofile='evictable-warm/connector.py')))
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
s=(H/'run_teacher_replay.py').read_text().replace("'demand-fallback' if a.arm=='candidate'","'evictable-warm' if a.arm=='candidate'").replace("H/'runtime-cold-run'","H/'runtime-evictable-warm'")
s=once(s," if a.arm=='candidate':cfg['kv_connector_extra_config'].update(wait_hint_ttl_ms=60000,wait_hint_cold_run_publication=True)"," if a.arm=='candidate':cfg['kv_connector_extra_config'].update(wait_hint_ttl_ms=60000,wait_hint_cold_run_publication=True,wait_hint_evictable_warm=True)")
ast.parse(s);(H/'run_evictable_teacher.py').write_text(s)
print('Prepared default-off evictable cache publication; not GPU qualified.')
