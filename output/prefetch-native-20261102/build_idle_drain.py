"""Drain existing empty futures before using the tested bounded queue wait."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent
runtime=H/'runtime-idle-drain';hooks=H/'hooks-idle-drain'
shutil.copytree(H/'runtime-idle-pending',runtime)
shutil.copytree(H/'hooks-idle-pending',hooks)
f=runtime/'vllm/v1/engine/core.py';s=f.read_text()
start=s.index('    def step_with_batch_queue(');end=s.index('    def _process_aborts_queue',start)
part=s[start:end];needle='        if self.scheduler.has_requests():\n';assert part.count(needle)==1
part=part.replace(needle,'''        # A connector may keep the engine alive with only ready/pending leases.
        # Do not replenish an empty GPU batch ahead of its existing future:
        # drain that future through the ordinary completion/update path first.
        # Live requests, nonempty batches and transfer/cleanup work retain the
        # native path. No futures are dropped and no completion fence is skipped.
        drain_idle = False
        if (batch_queue and self.scheduler.requests == {}
                and not getattr(self, "engines_running", True)
                and all(output.total_num_scheduled_tokens == 0
                        for _, output, _ in batch_queue)):
            connector = self.scheduler.get_kv_connector()
            inner = getattr(connector, "connector_scheduler", None)
            idle_timeout = getattr(inner, "ready_staging_idle_timeout", None)
            timeout = idle_timeout() if idle_timeout is not None else None
            drain_idle = timeout is not None and timeout > 0
        if drain_idle:
            self.semantic_staging_idle_drain_calls = getattr(
                self, "semantic_staging_idle_drain_calls", 0) + 1
        if not drain_idle and self.scheduler.has_requests():
''')
s=s[:start]+part+s[end:];ast.parse(s);f.write_text(s)
f=hooks/'sitecustomize.py';s=f.read_text().replace("'semantic_staging_idle_timer_wakes']", "'semantic_staging_idle_timer_wakes','semantic_staging_idle_drain_calls']");f.write_text(s)
s=(H/'run_idle_pending.py').read_text().replace("H/'runtime-idle-pending'","H/'runtime-idle-drain'").replace("else 'hooks-idle-pending'", "else 'hooks-idle-drain'")
(H/'run_idle_drain.py').write_text(s)
# Same real prompts/histories; six turns with smaller distinct observations.
s=(H/'spaced_tool_workload.py').read_text().replace("prepared/session-corpus.json", "prepared/short-tool-corpus.json").replace('index=slot*4+turn','index=slot*16+turn')
s=s.replace("'recorded_tools_executed':False", "'recorded_tools_executed':False,'length_stratum':'first five distinct recorded observations <=256 tokens including disclosure; source cases absent from original corpus'")
(H/'short_tool_workload.py').write_text(s)
s=(H/'run_idle_drain.py').read_text().replace('from spaced_tool_workload import','from short_tool_workload import').replace('prepared/session-corpus.json','prepared/short-tool-corpus.json')
(H/'run_short_tools.py').write_text(s)
d=json.loads((H/'IDLE-PENDING.json').read_text());base=d['runs'][0];d['runs']=[];d['progress_file']='short-tools-pilot-progress.json'
for arm,budget in [('arc',576),('candidate',512)]:
 r=json.loads(json.dumps(base));old=r['name'];name=f'short-tools-b{budget}-{arm}-s1103';r.update(name=name,arm=arm,gpu_blocks=budget)
 r['command']=[x.replace(old,name).replace('run_idle_pending.py','run_short_tools.py') for x in r['command']]
 for option,value in [('--arm',arm),('--source-label','candidate' if arm=='arc' else 'shared-lookup'),('--gpu-blocks',str(budget)),('--turns','6'),('--timeout','600')]:r['command'][r['command'].index(option)+1]=value
 d['runs'].append(r)
files=[H/'build_idle_drain.py',H/'run_idle_drain.py',H/'run_short_tools.py',H/'short_tool_workload.py',H/'prepare_short_tools.py',H/'prepared/short-tool-corpus.json',*runtime.rglob('*.py'),hooks/'sitecustomize.py']
for f in files:d['pins'][str(f.relative_to(H))]=hashlib.sha256(f.read_bytes()).hexdigest()
d['objective']='Exploratory pilot, no confirmatory claim. Four fresh disjoint public source cases, six actual-output turns, shorter distinct external recorded observations <=256 tokens, private simulated10-60s waits. ARC576 versus semantic512, same common7680 context and neutral warmup, candidate fused summary/fixed60sTTL/.95cap/guarded idle batch drain. All24 strict proofs per arm; actual memory and work inventory, no timer oracle. Do not generalize length stratum to production population or busy24.'
(H/'SHORT-TOOLS-PILOT.json').write_text(json.dumps(d,indent=2)+'\n')
print('Prepared guarded drain and disclosed short-observation pilot.')
