"""Generate finite fresh confirmation plans; never overwrite a frozen plan."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
base=json.loads((H/'prefix-capacity-screen.json').read_text())
arc={'EXPERIMENT_GENERATION_OFFLOAD':'0','EXPERIMENT_COMPLETED_TRANSCRIPT':'0','EXPERIMENT_PREFIX_SCORE':'0','SEMANTIC_SUMMARY_FIELDS':'all','SEMANTIC_CAPTURE_STEPS':'all'}
original={**arc,'EXPERIMENT_ORIGINAL':'1','EXPERIMENT_SCHEDULE':'continuous','EXPERIMENT_STRIDE':'1','SEMANTIC_CAPTURE_BACKEND':'dispatch'}
orders=[('arc_full','candidate','arc_small'),('arc_small','candidate','arc_full'),('candidate','arc_full','arc_small'),('arc_full','arc_small','candidate'),('candidate','arc_small','arc_full'),('arc_small','arc_full','candidate')]
for label,workload,small,large,cpu,repetitions in [('long','competing_long_dram',1568,2560,8192,2),('short','competing',400,640,2048,1)]:
    env={**base['environment'],'EXPERIMENT_GPU_BLOCKS':str(small),'EXPERIMENT_CPU_BLOCKS':str(cpu),'EXPERIMENT_LEAD_MS':'0'}
    cases=[];number=0
    for rep in range(repetitions):
        for seed in [1103,2207,3301]:
            pair=f'{label}-{seed}-r{rep}'
            for role in orders[number%len(orders)]:
                c={'phase':f'confirm-{label}-r{rep}-{role}','policy':'semantic' if role=='candidate' else 'arc','workload':workload,'seed':seed,'role':role,'pair_id':pair}
                if role!='candidate':c['environment']={**arc,'EXPERIMENT_GPU_BLOCKS':str(large if role=='arc_full' else small)}
                cases.append(c)
            if rep==0:cases.append({'phase':f'confirm-{label}-original','policy':'semantic','workload':workload,'seed':seed,'role':'original','source':'starting','environment':original})
            number+=1
    p={'source':'prefix-mean','workload':workload,'environment':env,'cases':cases,'design':'Fresh servers; alternating order; paired large ARC and equal-cache ARC; task-original semantic references by seed. No extra post-peer idle. Frozen prepared token arrays, identical model and CPU cache budget per workload. Source/harness/runtime/config pins in every raw run.', 'criterion':'All primary capacity pairs and equal-budget pairs at least99% ARC throughput; strict outputs/keys/physical proofs, actual numerical scoring and native consumption. Report all/return latency and quality without silently assuming a materiality threshold.', 'excludes':'All screens and quality/component diagnostics; no statistical equivalence claim.'}
    out=H/f'confirmation-{label}.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(p,indent=2)+'\n')
    print(out,len(cases))
