"""Fill the first cold run, preserving exact cached islands after that run."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
H=Path(__file__).resolve().parent
def once(s,a,b):
 assert s.count(a)==1,(a[:90],s.count(a));return s.replace(a,b)
runtime=H/'runtime-cold-run';shutil.copytree(H/'runtime-idle-drain',runtime)
f=runtime/'vllm/v1/core/block_pool.py';s=f.read_text()
s=once(s,'        self.enable_caching = enable_caching','        self.semantic_allocation_epochs = [0] * num_gpu_blocks\n        self.enable_caching = enable_caching')
s=once(s,'        # In order to only iterate the list once, we duplicated code a bit','        for block in ret:\n            self.semantic_allocation_epochs[block.block_id] += 1\n\n        # In order to only iterate the list once, we duplicated code a bit')
s=once(s,'        block_hashes: list[BlockHashWithGroupId] = []\n        if block.block_hash is not None:', '        self.semantic_allocation_epochs[block.block_id] += 1\n        block_hashes: list[BlockHashWithGroupId] = []\n        if block.block_hash is not None:')
s=once(s,'        for block in self.blocks:\n            block.reset_hash()', '        for block in self.blocks:\n            self.semantic_allocation_epochs[block.block_id] += 1\n            block.reset_hash()');ast.parse(s);f.write_text(s)
source=H/'sources/cold-run';shutil.copytree(H/'sources/short-fixed',source)
f=source/'semantic_offload/connector.py';old=f.read_text();s=old
s=once(s,'        self._published_wait_prefixes: dict[str, dict[str, object]] = {}','''        cold_run = extra_config.get("wait_hint_cold_run_publication", False)
        if not isinstance(cold_run, bool):
            raise TypeError("wait_hint_cold_run_publication must be a JSON boolean")
        if cold_run and not (self._wait_hint_shared_local_prefix
                             and self._wait_hint_publish_at_lookup):
            raise ValueError("cold-run publication requires shared exact native lookup publication")
        self._wait_hint_cold_run_publication = cold_run
        self._published_wait_prefixes: dict[str, dict[str, object]] = {}''')
s=once(s,'        candidate_keys = source.keys[local_gpu_prefix_count:]','        candidate_keys = self._wait_staging_candidates(source.keys, local_gpu_prefix_count)')
s=once(s,'    def _wait_required_staging_keys(self, state):','''    def _wait_staging_candidates(self, source_keys, local_count):
        keys = tuple(source_keys[local_count:])
        if not getattr(self, "_wait_hint_cold_run_publication", False):
            return keys
        lookup = getattr(self.gpu_block_pool, "get_cached_block", None)
        if not callable(lookup):
            return ()
        cold = []
        for key in keys:
            if lookup(get_offload_block_hash(key), [get_offload_group_idx(key)]):
                break
            cold.append(key)
        return tuple(cold)

    def _wait_required_staging_keys(self, state):''')
s=once(s,'            return state.source.keys[state.publication_local_gpu_prefix_count:]','            return self._wait_staging_candidates(state.source.keys, state.publication_local_gpu_prefix_count)')
s=once(s,'                and tuple(state.publication_candidate_keys) == state.source.keys[count:]\n                and _any_local_gpu_key(self.gpu_block_pool, state.source.keys[count:]) is False)', '                and tuple(state.publication_candidate_keys) == self._wait_staging_candidates(state.source.keys, count)\n                and _any_local_gpu_key(self.gpu_block_pool, state.publication_candidate_keys) is False)')
s=once(s,'                and tuple(candidate_keys) == source_keys[local_count:]','                and tuple(candidate_keys) == self._wait_staging_candidates(source_keys, local_count)')
s=once(s,'        if (getattr(self, "_wait_hint_shared_local_prefix", False)\n                and state.publication_local_gpu_prefix_count > 0):','        if (getattr(self, "_wait_hint_cold_run_publication", False)\n                or (getattr(self, "_wait_hint_shared_local_prefix", False)\n                    and state.publication_local_gpu_prefix_count > 0)):')
s=once(s,'        count = boundary // block_size\n        if (\n            getattr(group, "group_idx", -1)', '''        hinted_count = boundary // block_size
        count = (local_count + len(state.keys)
                 if getattr(self, "_wait_hint_cold_run_publication", False)
                 else hinted_count)
        if (
            not 0 < count <= hinted_count
            or getattr(group, "group_idx", -1)''')
s=once(s,'            or boundary != count * block_size','            or boundary != hinted_count * block_size')
s=once(s,'        expected_keys = [make_offload_key(h, 0) for h in request.block_hashes[:count]]\n        if (\n            expected_keys != list(state.source.keys)\n            or set(expected_keys[local_count:]) != set(state.keys)\n        ):\n            return False','''        full_expected_keys = [make_offload_key(h, 0) for h in request.block_hashes[:hinted_count]]
        expected_keys = full_expected_keys[:count]
        if (len(expected_keys) != count
                or full_expected_keys != list(state.source.keys)
                or expected_keys[local_count:] != list(state.keys)):
            return False''')
s=once(s,'            "prefetched_block_count": len(suffix_ids), "local_prefix_count": local_count,','''            "prefetched_block_count": len(suffix_ids), "local_prefix_count": local_count,
            "allocation_epochs": [self.gpu_block_pool.semantic_allocation_epochs[i] for i in ids],''')
s=once(s,'        adopted = (\n            num_external_tokens == 0\n            and actual == ids','''        # An ordinary demand load can fill a later hole after the newly
        # published native prefix. Credit only positions covered by the actual
        # native hit, with unchanged physical allocation epochs and key hashes.
        status = self._req_status.get(request.request_id)
        native_tokens = getattr(status, "num_locally_computed_tokens", None)
        block_size = group_config_tokens_per_block(self.config.kv_group_configs[0])
        epochs = self.gpu_block_pool.semantic_allocation_epochs
        expected_epochs = published.get("allocation_epochs", ())
        adopted = (
            type(native_tokens) is int and native_tokens >= len(ids) * block_size
            and len(expected_epochs) == len(ids)
            and all(epochs[i] == e for i, e in zip(ids, expected_epochs))
            and actual == ids''')
s=once(s,'            "observed_at_ns": usable_ns, "gpu_splice_bytes": 0,','''            "observed_at_ns": usable_ns, "gpu_splice_bytes": 0,
            "native_hit_tokens": native_tokens, "num_external_tokens": num_external_tokens,
            "published_allocation_epochs": list(expected_epochs),
            "current_allocation_epochs": [epochs[i] for i in ids],
            "adoption_evidence": "native_hit_positions_same_physical_hash_positive_owner_unchanged_allocation_epoch",''')
ast.parse(s);f.write_text(s)
(H/'cold-run-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='short-fixed/connector.py',tofile='cold-run/connector.py')))
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
s=(H/'run_short_fixed.py').read_text().replace("'short-fixed' if a.arm=='candidate'","'cold-run' if a.arm=='candidate'").replace("H/'runtime-idle-drain'","H/'runtime-cold-run'").replace("if a.arm=='candidate':cfg['kv_connector_extra_config']['wait_hint_ttl_ms']=60000","if a.arm=='candidate':cfg['kv_connector_extra_config'].update(wait_hint_ttl_ms=60000,wait_hint_cold_run_publication=True)")
(H/'run_cold_fixed.py').write_text(s)
print('Prepared first-cold-run source; no GPU qualification or performance claim.')
