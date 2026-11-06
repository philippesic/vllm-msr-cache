from pathlib import Path
import shutil,json,hashlib
H=Path(__file__).resolve().parent
src=H/'sources/shared-lookup';dst=H/'sources/demand-headroom';shutil.copytree(src,dst)
f=dst/'semantic_offload/connector.py';s=f.read_text();old='''        free = pool.get_num_free_blocks()
        if not getattr(self, "_prefetch_preserve_gpu_prefix_cache", False):
''';new='''        free = pool.get_num_free_blocks()
        # Reserve capacity for the largest currently known waiting prompt.
        # Only live request progress is consulted; no future arrival/return
        # time is available. Recompute every retry; never change ownership.
        demand = 0
        tokens_per_block = group_config_tokens_per_block(self.config.kv_group_configs[0])
        for req_status in getattr(self, "_req_status", {}).values():
            request = getattr(req_status, "req", None)
            status = getattr(getattr(request, "status", None), "name", None)
            if status in {"WAITING", "PREEMPTED", "WAITING_FOR_REMOTE_KVS"}:
                remaining = max(0, int(request.num_prompt_tokens) - int(request.num_computed_tokens))
                demand = max(demand, cdiv(remaining, tokens_per_block))
        free = max(0, free - demand)
        if not getattr(self, "_prefetch_preserve_gpu_prefix_cache", False):
'''
assert s.count(old)==1;s=s.replace(old,new);f.write_text(s)
pins=json.loads((dst/'pins.json').read_text());print(type(pins),list(pins)[:4])
files=pins.get('files',pins)
for name in files:
 p=dst/name if '/' in name else dst/'semantic_offload'/name
 if p.is_file():files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(dst/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
f=H/'run_pressure.py';s=(H/'run_frontier_defer.py').read_text().replace("source=H/'sources'/('original' if a.arm=='original' else a.source_label)","source=H/'sources'/('original' if a.arm=='original' else ('demand-headroom' if a.arm=='candidate' else a.source_label))")
f.write_text(s)
