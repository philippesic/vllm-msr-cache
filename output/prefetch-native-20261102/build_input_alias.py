"""Optional exact completed-input source, for response-independent serving replay."""
import ast,difflib,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent
source=H/'sources/full-input-alias';shutil.copytree(H/'sources/fused-mean',source)
f=source/'semantic_offload/connector.py';old=f.read_text();needle='''        if getattr(self, "_wait_hint_completed_transcript", False):
            self._register_completed_transcript(request, record)''';assert old.count(needle)==1
s=old.replace(needle,'''        if getattr(self, "_wait_hint_completed_transcript", False):
            import os
            if os.environ.get("SEMANTIC_REGISTER_FULL_INPUT_PREFIX") == "1":
                self._register_completed_input(request, record)
            self._register_completed_transcript(request, record)''')
needle='    def _register_completed_transcript(\n';assert s.count(needle)==1
method='''    def _register_completed_input(self, request: Request, declared: _ObservedPrefix) -> None:
        """Register this request's executed input, independently of its output.

        A return request may bind a shorter prior source. Saving only that
        declared source leaves its newly completed full input unaddressable.
        This alias contains no generated or future token, and is added only
        after the declared source passes identity/generation validation.
        """
        if getattr(getattr(request, "status", None), "name", None) not in {
            "FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED",
        }:
            return
        req_status = self._req_status.get(request.request_id)
        computed = getattr(request, "num_computed_tokens", None)
        if req_status is None or type(computed) is not int or computed < request.num_prompt_tokens:
            return
        group = self.config.kv_group_configs[0]
        chunk_tokens = group_config_tokens_per_chunk(group)
        boundary = request.num_prompt_tokens // chunk_tokens * chunk_tokens
        if (boundary <= 0 or boundary > len(request.prompt_token_ids)
                or boundary > self._calc_num_offloadable_tokens(req_status, computed)):
            return
        req_status.update_offload_keys()
        count = boundary // chunk_tokens
        keys = tuple(req_status.group_states[0].offload_keys[:count])
        expected = tuple(make_offload_key(h, 0) for h in request.block_hashes[:count])
        if len(keys) != count or keys != expected or len(set(keys)) != count:
            return
        record = _ObservedPrefix(
            run_id=declared.run_id, cell_id=declared.cell_id,
            session_id=declared.session_id, generation=declared.generation,
            observed_prefix_id=_observed_prefix_digest(
                list(request.prompt_token_ids[:boundary]), self._wait_hint_namespace),
            boundary_tokens=boundary, keys=keys, observed_ns=time.monotonic_ns(),
            source_request_id=request.request_id,
        )
        self._save_observed_prefix(record, provenance="completed_full_input_prefix")

'''
s=s.replace(needle,method+needle);ast.parse(s);f.write_text(s)
(source/'pins.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source/'semantic_offload').glob('*.py')},indent=2)+'\n')
(H/'full-input-alias.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='fused-mean/connector.py',tofile='full-input-alias/connector.py')))
print('Built opt-in completed-input source alias; existing candidate untouched')
