"""One-shot private snapshot construction; never edits canonical checkout."""
from pathlib import Path
import shutil,json,hashlib
H=Path(__file__).resolve().parent
source=H/'sources/shared-prefix';source.mkdir(exist_ok=False)
shutil.copytree(H/'sources/candidate/semantic_offload',source/'semantic_offload')
shutil.copytree(H/'runtime',H/'runtime-shared-prefix')
p=source/'semantic_offload/connector.py';s=p.read_text()
def rep(old,new):
 global s
 assert s.count(old)==1,(old[:100],s.count(old));s=s.replace(old,new)
rep('for option in ("wait_hint_coalesce_exact_source", "prefetch_require_full_prefix"):', 'for option in ("wait_hint_coalesce_exact_source", "prefetch_require_full_prefix",\n                       "wait_hint_shared_local_prefix"):')
marker='    def _attempt_wait_prefetch(\n'
helper='''    def _wait_required_staging_keys(self, state):
        if getattr(self, "_wait_hint_shared_local_prefix", False):
            return state.source.keys[state.publication_local_gpu_prefix_count:]
        return state.source.keys

    def _wait_local_prefix_valid(self, state):
        """No borrowed pin: detect allocator/hash changes before stage or bind."""
        if not getattr(self, "_wait_hint_shared_local_prefix", False):
            return True
        count, ids, available = _local_gpu_prefix(self.gpu_block_pool, state.source.keys)
        return (available and count == state.publication_local_gpu_prefix_count
                and tuple(ids) == state.publication_local_gpu_block_ids
                and tuple(state.publication_candidate_keys) == state.source.keys[count:]
                and _any_local_gpu_key(self.gpu_block_pool, state.source.keys[count:]) is False)

    def _wait_complete_staging_available(self, source_keys, candidate_keys,
                                         resident_keys, local_count, local_ids):
        if not getattr(self, "_wait_hint_shared_local_prefix", False):
            return (bool(source_keys) and tuple(candidate_keys) == source_keys
                    and tuple(resident_keys) == source_keys
                    and _any_local_gpu_key(self.gpu_block_pool, source_keys) is False)
        count, ids, available = _local_gpu_prefix(self.gpu_block_pool, source_keys)
        return (bool(candidate_keys) and available and count == local_count
                and tuple(ids) == tuple(local_ids) and len(ids) == local_count
                and tuple(candidate_keys) == source_keys[local_count:]
                and tuple(resident_keys) == tuple(candidate_keys)
                and _any_local_gpu_key(self.gpu_block_pool, candidate_keys) is False)

'''
rep(marker,helper+marker)
old='''            not source.keys
            or tuple(candidate_keys) != source.keys
            or tuple(resident_prefix) != source.keys
            or _any_local_gpu_key(self.gpu_block_pool, source.keys) is not False'''
rep(old,'''            not self._wait_complete_staging_available(
                source.keys, candidate_keys, resident_prefix,
                local_gpu_prefix_count, local_gpu_block_ids)''')
rep('''            if available_budget < len(source.keys):''','''            if available_budget < len(candidate_keys):''')
old='''            not state.source.keys
            or tuple(state.publication_candidate_keys) != state.source.keys
            or tuple(eligible_resident_prefix) != state.source.keys
            or allocation_budget < len(state.source.keys)
            or _any_local_gpu_key(self.gpu_block_pool, state.source.keys) is not False'''
rep(old,'''            not self._wait_complete_staging_available(
                state.source.keys, state.publication_candidate_keys,
                eligible_resident_prefix, state.publication_local_gpu_prefix_count,
                state.publication_local_gpu_block_ids)
            or allocation_budget < len(self._wait_required_staging_keys(state))''')
rep('''                and tuple(selected) != state.source.keys):''','''                and tuple(selected) != self._wait_required_staging_keys(state)):''')
rep('''            src_spec = self.manager.prepare_load(resident, state.req_context)''','''            if not self._wait_local_prefix_valid(state):
                self.gpu_block_pool.free_blocks(reserved)
                state.selection_evidence["status"] = "local_prefix_changed_during_reservation"
                return False
            src_spec = self.manager.prepare_load(resident, state.req_context)''')
rep('''                list(gpu_block_ids), group_sizes=[count], block_indices=[0]''','''                list(gpu_block_ids), group_sizes=[count],
                block_indices=[state.publication_local_gpu_prefix_count
                               if getattr(self, "_wait_hint_shared_local_prefix", False) else 0]''')
rep('''        return bool(keys) and descriptor == expected and all(''','''        return bool(keys) and self._wait_local_prefix_valid(state) and descriptor == expected and all(''')
rep('''            if not keys or (getattr(self, '_prefetch_require_full_prefix', False) and keys != state.source.keys):''','''            if (not keys or not self._wait_local_prefix_valid(state)
                    or (getattr(self, '_prefetch_require_full_prefix', False)
                        and keys != self._wait_required_staging_keys(state))):''')
# Shared-local-prefix states never silently use legacy splice fallback on rejection.
rep('''        if self._try_publish_wait_prefix(state):
            return
        binding_id =''','''        if self._try_publish_wait_prefix(state):
            return
        if (getattr(self, "_wait_hint_shared_local_prefix", False)
                and state.publication_local_gpu_prefix_count > 0):
            self._free_wait_prefetch(state, "shared_prefix_native_publication_rejected")
            return
        binding_id =''')
rep('''        publish = getattr(self.gpu_block_pool, "publish_completed_prefix", None)''','''        shared = (getattr(self, "_wait_hint_shared_local_prefix", False)
                  and state.publication_local_gpu_prefix_count > 0)
        publish = getattr(self.gpu_block_pool,
                          "publish_completed_suffix" if shared else "publish_completed_prefix", None)
        local_count = state.publication_local_gpu_prefix_count if shared else 0''')
rep('''            or len(state.keys) != count
            or len(state.gpu_block_ids) != count
            or len(set(state.keys)) != count''','''            or len(state.keys) != count - local_count
            or len(state.gpu_block_ids) != count - local_count
            or len(set(state.keys)) != count - local_count
            or (shared and not self._wait_local_prefix_valid(state))''')
rep('''            or set(expected_keys) != set(state.keys)''','''            or set(expected_keys[local_count:]) != set(state.keys)''')
rep('''        ids = [by_key[key] for key in expected_keys]''','''        suffix_ids = [by_key[key] for key in expected_keys[local_count:]]
        ids = list(state.publication_local_gpu_block_ids) + suffix_ids if shared else suffix_ids''')
rep('''        if not publish(
            request, blocks, block_size=block_size, kv_cache_group_id=0,
            completion_verified=True,
        ):
            return False''','''        if shared:
            published = publish(
                request, blocks[:local_count], blocks[local_count:],
                block_size=block_size, kv_cache_group_id=0, completion_verified=True)
        else:
            published = publish(request, blocks, block_size=block_size,
                                kv_cache_group_id=0, completion_verified=True)
        if not published:
            return False''')
rep('''        self._prefetch_reserved_blocks -= count
        state.status = "published"''','''        self._prefetch_reserved_blocks -= len(suffix_ids)
        state.status = "published"''')
rep('''            "published_at_ns": published_ns, "keys": expected_keys,
        }''','''            "published_at_ns": published_ns, "keys": expected_keys,
            "prefetched_block_count": len(suffix_ids), "local_prefix_count": local_count,
        }''')
rep('''            "ownership": "native_prefix_cache_evictable", "gpu_splice_bytes": 0,
        })''','''            "ownership": "native_prefix_cache_evictable", "gpu_splice_bytes": 0,
            "local_prefix_count": local_count, "prefetched_suffix_block_ids": suffix_ids,
            "prefetched_suffix_start_index": local_count,
        })''')
rep('''                "actual_block_ids": actual,
            }
            if adopted:
                state["prefetch_consumed_keys"] += len(ids)
                state["prefetch_consumed_blocks"] += len(ids)''','''                "actual_block_ids": actual,
                "local_prefix_count": published.get("local_prefix_count", 0),
                "prefetched_suffix_block_ids": ids[published.get("local_prefix_count", 0):],
                "prefetched_suffix_block_count": published.get("prefetched_block_count", len(ids)),
            }
            if adopted:
                consumed = published.get("prefetched_block_count", len(ids))
                state["prefetch_consumed_keys"] += consumed
                state["prefetch_consumed_blocks"] += consumed''')
rep('''        if (
            state.status == "pending"
            and not state.coalescing_complete''','''        if (getattr(self, "_wait_hint_shared_local_prefix", False)
                and state.publication_local_gpu_prefix_count > 0
                and not self._wait_local_prefix_valid(state)):
            self._retire_wait_prefetch(state, "local_prefix_changed_before_bind", now_ns=now_ns)
            record_wait_hint_event({"event": "resume_binding_rejected", "event_id": event_id,
                                    "request_id": request.request_id,
                                    "reason": "local_prefix_changed_before_bind",
                                    "observed_at_ns": now_ns})
            return
        if (
            state.status == "pending"
            and not state.coalescing_complete''')
p.write_text(s)
# Native offset-aware publication is private; original full-prefix helper untouched.
p=H/'runtime-shared-prefix/vllm/v1/core/block_pool.py';s=p.read_text();marker='    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:\n';assert s.count(marker)==1
helper='''    def publish_completed_suffix(
        self, request: Request, local_blocks: list[KVCacheBlock],
        suffix_blocks: list[KVCacheBlock], *, block_size: int,
        kv_cache_group_id: int, completion_verified: bool,
    ) -> bool:
        """Atomically validate a cached prefix and publish only owned suffix KV.

        Scheduler-thread operation: prefix references/hashes are never mutated.
        The caller still owns all suffix reservations on any rejected operation.
        """
        local_count = len(local_blocks)
        blocks = local_blocks + suffix_blocks
        if (completion_verified is not True or not self.enable_caching
                or request.skip_reading_prefix_cache or kv_cache_group_id != 0
                or block_size <= 0 or block_size != self.hash_block_size
                or not local_blocks or not suffix_blocks
                or request.num_tokens <= len(blocks) * block_size):
            return False
        hashes = list(request.block_hashes[:len(blocks)])
        if (len(hashes) != len(blocks) or len(set(hashes)) != len(hashes)
                or any(not isinstance(h, bytes) or not h for h in hashes)):
            return False
        ids = [b.block_id for b in blocks]
        if len(set(ids)) != len(ids):
            return False
        for i, (block, block_hash) in enumerate(zip(blocks, hashes)):
            if (not 0 < block.block_id < self.num_gpu_blocks
                    or self.blocks[block.block_id] is not block or block.is_null):
                return False
            indexed = self.get_cached_block(block_hash, [kv_cache_group_id])
            if i < local_count:
                if (block.ref_cnt < 0 or indexed is None or indexed[0] is not block
                        or block.block_hash != make_block_hash_with_group_id(block_hash, kv_cache_group_id)
                        or block.block_hash_num_tokens != (i + 1) * block_size):
                    return False
            elif (block.ref_cnt != 1 or block.block_hash is not None
                    or block.block_id in self.cached_block_hashes_by_block
                    or indexed is not None):
                return False
        try:
            self.cache_full_blocks(request, blocks, local_count, len(blocks),
                                   block_size, kv_cache_group_id)
        except Exception:
            self.evict_blocks(set(ids[local_count:]))
            raise
        self.free_blocks(suffix_blocks)
        return True

'''
s=s.replace(marker,helper+marker);p.write_text(s)
files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'semantic_offload').glob('*.py')};(source/'pins.json').write_text(json.dumps({'source':'shared-prefix-private','files':files},indent=2)+'\n')
print('Private source/runtime snapshot created')
