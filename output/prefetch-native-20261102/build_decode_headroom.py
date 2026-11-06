"""Reserve remaining declared decode growth before optional GPU allocation."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
source=H/'sources/decode-headroom';shutil.copytree(H/'sources/evictable-warm-v2',source,dirs_exist_ok=True)
f=source/'semantic_offload/connector.py';s=f.read_text()
a='        self._wait_hint_evictable_warm = warm\n';assert s.count(a)==1
s=s.replace(a,a+'''        reserve_decode = extra_config.get("prefetch_reserve_decode_budget", False)
        if not isinstance(reserve_decode, bool):
            raise TypeError("prefetch_reserve_decode_budget must be a JSON boolean")
        self._prefetch_reserve_decode_budget = reserve_decode
''')
a='    def _prefetch_free_capacity(self) -> int:\n';assert s.count(a)==1
s=s.replace(a,'''    def _prefetch_decode_headroom_blocks(self):
        if not getattr(self, "_prefetch_reserve_decode_budget", False):
            return 0
        size = group_config_tokens_per_block(self.config.kv_group_configs[0])
        needed = 0
        seen = set()
        for status in self._req_status.values():
            req = getattr(status, "req", None)
            if req is None or getattr(getattr(req, "status", None), "name", None) not in {"RUNNING", "WAITING_FOR_REMOTE_KVS"}:
                continue
            if req.request_id in seen:
                continue
            seen.add(req.request_id)
            prompt = getattr(req, "num_prompt_tokens", None)
            computed = getattr(req, "num_computed_tokens", None)
            maximum = getattr(getattr(req, "sampling_params", None), "max_tokens", None)
            if (type(prompt) is not int or prompt < 0 or type(computed) is not int or computed < 0
                    or type(maximum) is not int or maximum <= 0):
                # Unknown declared demand cannot safely be borrowed by an
                # optional copy. This affects admission only, never serving.
                return self.gpu_block_pool.num_gpu_blocks
            # Last sampled token need not be computed. Existing ownership is
            # lower-bounded by the computed-token pages; over-reserving one
            # already allocated lookahead page is intentionally conservative.
            needed += max(0, cdiv(prompt + maximum - 1, size) - cdiv(computed, size))
        return needed

'''+a)
a='        free = pool.get_num_free_blocks()\n';assert s.count(a)==1
s=s.replace(a,'        free = max(0, pool.get_num_free_blocks() - self._prefetch_decode_headroom_blocks())\n')
ast.parse(s);f.write_text(s);(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')},indent=2)+'\n')
for mode,base in [('light','run_evictable_teacher.py'),('busy','run_evictable_busy.py')]:
 s=(H/base).read_text().replace("'evictable-warm' if a.arm=='candidate'","'decode-headroom' if a.arm=='candidate'").replace("'evictable-warm-v2' if a.arm=='candidate'","'decode-headroom' if a.arm=='candidate'")
 s=s.replace('wait_hint_evictable_warm=True)','wait_hint_evictable_warm=True,prefetch_reserve_decode_budget=True)')
 ast.parse(s);(H/f'run_decode_headroom_{mode}.py').write_text(s)
print('Prepared decode-budget headroom; no performance measurement yet.')
