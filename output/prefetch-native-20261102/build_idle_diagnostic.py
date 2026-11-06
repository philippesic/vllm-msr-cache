"""Passive sampled guard diagnosis; explicitly excluded from performance claims."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
runtime=H/'runtime-idle-diagnostic';hooks=H/'hooks-idle-diagnostic'
shutil.copytree(H/'runtime-idle-pending',runtime);shutil.copytree(H/'hooks-idle-pending',hooks)
f=runtime/'vllm/v1/engine/core.py';s=f.read_text();needle='''        # Ready speculative pages need expiry timers, not empty GPU steps.''';assert s.count(needle)==1
s=s.replace(needle,'''        if os.environ.get("SEMANTIC_IDLE_DIAGNOSTIC") == "1":
            sequence = getattr(self, "semantic_idle_diagnostic_sequence", 0) + 1
            self.semantic_idle_diagnostic_sequence = sequence
            if sequence % 64 == 1:
                inner = getattr(self.scheduler.get_kv_connector(), "connector_scheduler", None)
                states = list(getattr(inner, "_wait_prefetch", {}).values())
                if states:
                    state_counts = {}
                    for state in states:
                        state_counts[state.status] = state_counts.get(state.status, 0) + 1
                    request_counts = {}
                    for request in self.scheduler.requests.values():
                        name = getattr(getattr(request, "status", None), "name", "unknown")
                        request_counts[name] = request_counts.get(name, 0) + 1
                    snapshot = {
                        "input_blocking": self.process_input_queue_block,
                        "engines_running": self.engines_running,
                        "batch_queue_len": len(self.batch_queue or []),
                        "native_requests": request_counts,
                        "wait_states": state_counts,
                        "native_jobs": len(getattr(inner, "_jobs", {})),
                        "manager_pending": inner.manager.has_pending_work(),
                        "wait_jobs": len(getattr(inner, "_wait_prefetch_jobs", {})),
                        "splice_releases": len(getattr(inner, "_pending_splice_releases", {})),
                        "idle_timeout": inner.ready_staging_idle_timeout(),
                    }
                    rows = getattr(self, "semantic_idle_diagnostic", {})
                    key = repr(snapshot)
                    if key in rows:
                        rows[key]["samples"] += 1
                    elif len(rows) < 64:
                        rows[key] = {"snapshot": snapshot, "samples": 1}
                    self.semantic_idle_diagnostic = rows

'''+needle);ast.parse(s);f.write_text(s)
f=hooks/'sitecustomize.py';s=f.read_text();needle="            value['idle_loop']=";idx=s.index(needle);s=s[:idx]+"            value['idle_loop_diagnostic']=getattr(self,'semantic_idle_diagnostic',{})\n"+s[idx:];f.write_text(s)
s=(H/'run_idle_pending.py').read_text().replace("H/'runtime-idle-pending'", "H/'runtime-idle-diagnostic'").replace("else 'hooks-idle-pending'", "else 'hooks-idle-diagnostic'")
s=s.replace(" env['SEMANTIC_READY_IDLE_WAIT']='1' if a.arm=='candidate' else '0'", " env['SEMANTIC_READY_IDLE_WAIT']='1' if a.arm=='candidate' else '0'\n env['SEMANTIC_IDLE_DIAGNOSTIC']='1'")
s=s.replace("'performance_eligible':not (a.audit or a.quality or a.boundary_diagnostic or a.feasibility_only or a.observe_native_queries)", "'performance_eligible':False,'idle_diagnostic_only':True");(H/'run_idle_diagnostic.py').write_text(s)
d=json.loads((H/'IDLE-PENDING.json').read_text());d['progress_file']='idle-diagnostic-progress.json';r=d['runs'][0];old=r['name'];r['name']='idle-diagnostic-b512-candidate-s1103';r['command']=[x.replace(old,r['name']).replace('run_idle_pending.py','run_idle_diagnostic.py') for x in r['command']];r['command'][r['command'].index('--turns')+1]='2'
for f in [H/'run_idle_diagnostic.py',*(runtime).rglob('*.py'),hooks/'sitecustomize.py']:d['pins'][str(f.relative_to(H))]=hashlib.sha256(f.read_bytes()).hexdigest()
d['objective']='Passiveguarddiagnostic every64inputloops, bounded64snapshots. Record actualnativebatchqueuelen/requeststatuses, nativejob/managerpending, ready/pendingstates andhelperreturn; no policy changes. Sourcehelperidentical to previously21testedidle-pending. 4clients2turns publictrace,8strictproofs, no performanceeligibility. Locatewhyactualwaitcountremainszero beforechangingcorequeuebehavior.';(H/'IDLE-DIAGNOSTIC.json').write_text(json.dumps(d,indent=2)+'\n');print('Prepared bounded passive idle-guard diagnosis.')
