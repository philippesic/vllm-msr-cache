"""Inventory required execution cells, without converting completion to qualification."""
import json,time
from pathlib import Path
b=Path(__file__).resolve().parent
names=['stage1-live','stage1-replay-fresh-sockets','stage1-trace','stage2-published','stage2-resource','stage2-upstream-control','stage2-secondary-confirm','stage3-repository','stage3-repository-neutral-confirm','stage3-repository-cohost','stage3-repository-reserve','stage3-documents','stage3-documents-cohost','stage3-documents-pressure-control','stage3-documents-admission-diagnostic','stage3-documents-quality-calibration','stage3-documents-reserve','stage3-documents-reserve-full','stage3-ample','stage3-ample-neutral-confirm','stage3-cpu-pressure','stage3-concurrency','stage3-serialized-tools','stage3-mixed','stage3-soak','runtime-probe','runtime-probe-repaired','runtime-probe-qwen15','runtime-probe-opt125','runtime-probe-opt125-complete','runtime-probe-minimal','runtime-probe-abandon','runtime-probe-abandon-declared','runtime-probe-abandon-effective','runtime-probe-long','runtime-probe-qwen15-long','runtime-probe-graphs','stage4-graphs','stage4-qwen15-performance','stage4-qwen15-cohost','stage4-minimal-performance','example-client','isolated-example-client']
rows=[]
for name in names:
 p=b/name/'result.json'
 if not p.exists():rows.append(dict(campaign=name,execution='pending'));continue
 d=json.loads(p.read_text());state=d.get('status',('passed' if d.get('passed') else 'failed'))
 rows.append(dict(campaign=name,execution=state,cases=len(d.get('cases',[])),error=d.get('error'),attempt_finished=state!='running'))
report=dict(recorded_unix=time.time(),cells=rows,missing_or_running=[r['campaign'] for r in rows if r['execution'] in ('pending','running')],scope='Required execution inventory, including diagnosed development cells; completed measurements are not qualification passes. Additional failed attempts remain in the full catalog. Runtime interface, installation, source audits, archive verification and cleanup require their separate receipts.')
(b/'completion-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print('Required cells:',len(rows),'pending/running:',len(report['missing_or_running']));print('\n'.join(report['missing_or_running']))
