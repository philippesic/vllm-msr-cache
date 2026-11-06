"""Private candidate: publish a fenced contiguous subprefix, with reuse witnesses."""
import ast,difflib,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent

def replace_once(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)

runtime=H/'runtime-partial-native'
shutil.copytree(H/'runtime-prefix-order',runtime)
p=runtime/'vllm/v1/core/block_pool.py';old=p.read_text();s=old
s=replace_once(s,'        self.enable_caching = enable_caching','        self.semantic_allocation_epochs = [0] * num_gpu_blocks\n        self.enable_caching = enable_caching')
s=replace_once(s,'        # In order to only iterate the list once, we duplicated code a bit','        for block in ret:\n            self.semantic_allocation_epochs[block.block_id] += 1\n\n        # In order to only iterate the list once, we duplicated code a bit')
s=replace_once(s,'        block_hashes: list[BlockHashWithGroupId] = []\n        if block.block_hash is not None:', '        self.semantic_allocation_epochs[block.block_id] += 1\n        block_hashes: list[BlockHashWithGroupId] = []\n        if block.block_hash is not None:')
s=replace_once(s,'        for block in self.blocks:\n            block.reset_hash()', '        for block in self.blocks:\n            self.semantic_allocation_epochs[block.block_id] += 1\n            block.reset_hash()')
ast.parse(s);p.write_text(s)
(H/'partial-runtime.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='tail-first/block_pool.py',tofile='partial/block_pool.py')))

source=H/'sources/partial-native';shutil.copytree(H/'sources/shared-lookup',source)
p=source/'semantic_offload/connector.py';old=p.read_text();s=old
s=replace_once(s,'        count = boundary // block_size\n        if (\n            getattr(group, "group_idx", -1)', '''        hinted_count = boundary // block_size
        partial_native = bool(getattr(self, "_wait_hint_partial_native_publication", False))
        count = local_count + len(state.keys) if partial_native else hinted_count
        if (
            count > hinted_count
            or (not partial_native and count != hinted_count)
            or getattr(group, "group_idx", -1)''')
s=replace_once(s,'            or boundary != count * block_size','            or boundary != hinted_count * block_size')
s=replace_once(s,'        expected_keys = [make_offload_key(h, 0) for h in request.block_hashes[:count]]\n        if (\n            expected_keys != list(state.source.keys)\n            or set(expected_keys[local_count:]) != set(state.keys)\n        ):\n            return False', '''        full_expected_keys = [make_offload_key(h, 0) for h in request.block_hashes[:hinted_count]]
        expected_keys = full_expected_keys[:count]
        if (
            len(expected_keys) != count
            or full_expected_keys != list(state.source.keys)
            or expected_keys[local_count:] != list(state.keys)
        ):
            return False''')
s=replace_once(s,'            "prefetched_block_count": len(suffix_ids), "local_prefix_count": local_count,','''            "prefetched_block_count": len(suffix_ids), "local_prefix_count": local_count,
            "allocation_epochs": [self.gpu_block_pool.semantic_allocation_epochs[i] for i in ids],''')
s=replace_once(s,'        adopted = (\n            num_external_tokens == 0', '''        req_status = self._req_status.get(request.request_id)
        native_tokens = getattr(req_status, "num_locally_computed_tokens", 0)
        block_size = group_config_tokens_per_block(self.config.kv_group_configs[0])
        native_count = max(0, native_tokens // block_size) if type(native_tokens) is int else 0
        epochs = getattr(self.gpu_block_pool, "semantic_allocation_epochs", ())
        expected_epochs = published.get("allocation_epochs", ())
        confirmed = []
        for i in range(min(len(ids), len(prefix_blocks), native_count)):
            block = prefix_blocks[i]
            if (block.block_id != ids[i] or block.block_hash != published["keys"][i]
                    or block.ref_cnt <= 0 or i >= len(expected_epochs)
                    or not 0 <= block.block_id < len(epochs)
                    or epochs[block.block_id] != expected_epochs[i]):
                break
            if i >= published.get("local_prefix_count", 0):
                confirmed.append(block.block_id)
        witness_full = (native_count >= len(ids)
                        and len(confirmed) == published.get("prefetched_block_count", len(ids)))
        adopted = (
            witness_full and num_external_tokens == 0''')
s=replace_once(s,'        usable_ns = time.monotonic_ns()\n        state = getattr(self, "_preservation_requests", {}).get(request.request_id)', '''        usable_ns = time.monotonic_ns()
        record_wait_hint_event({
            "schema": "semantic-known-prefix-wait-hint-event-v1",
            "event": "native_prefetch_consumption_witness",
            "event_id": published["event_id"], "request_id": request.request_id,
            "observed_at_ns": usable_ns, "num_external_tokens": num_external_tokens,
            "native_hit_tokens": native_tokens, "block_size_tokens": block_size,
            "local_prefix_count": published.get("local_prefix_count", 0),
            "published_block_ids": ids, "actual_block_ids": actual,
            "published_allocation_epochs": list(expected_epochs),
            "current_allocation_epochs": [epochs[i] for i in ids],
            "confirmed_prefetched_block_ids": confirmed,
            "confirmed_prefetched_blocks": len(confirmed),
            "evidence": "native_hit_positions_same_physical_hash_positive_owner_unchanged_allocation_epoch",
        })
        state = getattr(self, "_preservation_requests", {}).get(request.request_id)''')
s=replace_once(s,'        self._published_wait_prefixes: dict[str, dict[str, object]] = {}','''        self._published_wait_prefixes: dict[str, dict[str, object]] = {}
        self._wait_hint_partial_native_publication = bool(
            getattr(spec, "extra_config", {}).get("wait_hint_partial_native_publication", False))''')
s=s.replace('getattr(self, "_prefetch_require_full_prefix", False)\n            and self._admitted_prefill_in_progress()','(getattr(self, "_prefetch_require_full_prefix", False)\n             or getattr(self, "_wait_hint_partial_native_publication", False))\n            and self._admitted_prefill_in_progress()')
s=s.replace("getattr(self, '_prefetch_require_full_prefix', False) and self._admitted_prefill_in_progress()","(getattr(self, '_prefetch_require_full_prefix', False) or getattr(self, '_wait_hint_partial_native_publication', False)) and self._admitted_prefill_in_progress()")
ast.parse(s);p.write_text(s)
(H/'partial-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='shared-lookup/connector.py',tofile='partial/connector.py')))
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source/'semantic_offload').glob('*.py')},indent=2)+'\n')

old=(H/'run_tailfirst.py').read_text()
s=old.replace("runtime_root=H/'runtime-prefix-order'","runtime_root=H/'runtime-partial-native'")
s=s.replace("source=H/'sources'/('original' if a.arm=='original' else a.source_label)","source=H/'sources'/('partial-native' if a.arm=='candidate' else a.source_label)")
s=s.replace(" cfg=config('arc' if a.arm=='arc' else 'semantic',p,0)"," cfg=config('arc' if a.arm=='arc' else 'semantic',p,0)\n if a.arm=='candidate':cfg['kv_connector_extra_config'].update(prefetch_require_full_prefix=False,wait_hint_partial_native_publication=True)")
(H/'run_partial.py').write_text(s)
print('Built isolated partial-prefix candidate; canonical checkout unchanged')
