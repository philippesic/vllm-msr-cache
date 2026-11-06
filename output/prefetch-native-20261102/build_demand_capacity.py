"""Count cached free-list pages that native admission must acquire too."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/demand-capacity';shutil.copytree(H/'sources/cold-run',source)
f=source/'semantic_offload/connector.py';old=f.read_text();s=old
needle='    def _reclaim_wait_prefetch_for_demand(\n';assert s.count(needle)==1
s=s.replace(needle,'''    def _native_prefix_free_block_ids(self, request, num_computed_tokens):
        # Native allocation acquires both the uncached remainder and every
        # cached prefix page whose refcount is zero. Those pages are included
        # in get_num_free_blocks(), so they cannot also back the remainder.
        # Inspect only the qualified dense, aligned single-group layout.
        groups = self.config.kv_group_configs
        pool = self.gpu_block_pool
        if (len(groups) != 1 or config_blocks_per_chunk(self.config) != 1
                or not getattr(pool, "enable_caching", False)):
            return set()
        size = group_config_tokens_per_block(groups[0])
        if type(num_computed_tokens) is not int or num_computed_tokens < 0 or num_computed_tokens % size:
            return set()
        lookup = getattr(pool, "get_cached_block", None)
        if not callable(lookup):
            return set()
        free_ids = set()
        for block_hash in request.block_hashes[:num_computed_tokens // size]:
            found = lookup(block_hash, [groups[0].group_idx])
            if found:
                block = found[0]
                if block.ref_cnt == 0 and not block.is_null:
                    free_ids.add(block.block_id)
        return free_ids

'''+needle)
needle='        demand_blocks = cdiv(remaining_tokens, tokens_per_block)\n        free_blocks = self.gpu_block_pool.get_num_free_blocks()'
assert s.count(needle)==1;s=s.replace(needle,'''        cached_free_ids = self._native_prefix_free_block_ids(request, num_computed_tokens)
        demand_blocks = cdiv(remaining_tokens, tokens_per_block) + len(cached_free_ids)
        free_blocks = self.gpu_block_pool.get_num_free_blocks()''')
needle='                "demand_blocks": demand_blocks,\n';assert s.count(needle)==1;s=s.replace(needle,needle+'                "native_cached_free_prefix_blocks": len(cached_free_ids),\n')
ast.parse(s);f.write_text(s);(H/'demand-capacity-source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='cold-run/connector.py',tofile='demand-capacity/connector.py')))
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
s=(H/'run_cold_fixed.py').read_text().replace("'cold-run' if a.arm=='candidate'","'demand-capacity' if a.arm=='candidate'");(H/'run_demand_capacity.py').write_text(s)
print('Prepared native cached-prefix capacity correction; no measurement yet.')
