"""Require complete retained functional/operational receipts; no timing win claim."""
import json,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 root=b/name;r=json.loads((root/'result.json').read_text());checks=json.loads((b/(name+'-functional-checks.json')).read_text());allocation=json.loads((b/(name+'-allocation-receipts.json')).read_text());probe=r['plan']['runtime_probe']
 graph_path=b/(name+'-graph-receipts.json');graph=json.loads(graph_path.read_text()) if graph_path.exists() else None
 operational=[dict(case=c['name'],passed=c.get('operational',{}).get('passed'),checks=c.get('operational'),cleanup=c['cleanup']) for c in r['cases']]
 passed=(r['status']=='passed' and checks['all_wave_checks_passed'] and allocation['all_actual_pinned_paths_established'] and all(c['passed'] and c['cleanup']['verified'] for c in operational) and (r['plan']['execution']!='graphs' or graph and graph['all_timed_graph_paths_established']))
 d=dict(recorded_unix=time.time(),campaign=name,selected_functional_cell_passed=bool(passed),expected_waves=checks['expected_waves'],observed_waves=checks['observed_waves'],requires_actual_cpu_restore=probe.get('require_restore',True),requires_abandoned_advice_eviction=bool(probe.get('require_no_cache_after_abandon')),all_wave_checks_passed=checks['all_wave_checks_passed'],actual_pinned_shared_backing=allocation['all_actual_pinned_paths_established'],actual_timed_graph_path=(graph['all_timed_graph_paths_established'] if graph else None),operational=operational,scope='Observed native restore/fresh-computation, exact greedy generation, protocol and bounded operational checks. Not direct KV byte comparison, useful-answer evaluation, broad model support or production reliability. Raw diagnostic time includes additional candidate protocol RPC checks and is not a matched application-performance comparison; all point losses remain in pair-alarms.')
 (b/(name+'-functional-decision.json')).write_text(json.dumps(d,indent=2)+'\n');print(name,'complete functional receipt pass:',bool(passed),'waves',checks['observed_waves'])
 if not passed:raise RuntimeError('Functional receipt incomplete/failed; diagnose before advancing')
