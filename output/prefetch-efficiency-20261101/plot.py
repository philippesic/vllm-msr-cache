"""Measured scatter only: actual GPU allocator bytes include model/runtime."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
rows=json.loads((H/'RESULTS.json').read_text());old=json.loads((H.parent/'vram-frontier-20261102/SCREEN_RESULTS.json').read_text())
print(type(old),list(old)[:5])
if isinstance(old,dict): old=old.get('rows',old.get('points',[]))
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for arm,color in [('arc','gray'),('candidate','orange')]:
 rr=[r for r in old if r['arm']==arm and r.get('complete_load_eligible')]
 for ax,key in zip(axes[:2],['TPS','TTFT_p95_ms']):
  ax.scatter([r['CUDA_allocated_peak_bytes']/2**30 for r in rr],[r[key]/(1000 if 'ms' in key else 1) for r in rr],color=color,alpha=.45,label='prior '+arm)
for variant,marker,color in [('ARC','x','black'),('cap','o','blue'),('headroom','*','purple'),('backoff','D','green')]:
 rr=[r for r in rows if r['variant']==variant and r.get('complete_load_eligible')]
 if not rr:continue
 for ax,key in zip(axes[:2],['TPS','TTFT_p95_ms']):
  ax.scatter([r['CUDA_allocated_peak_bytes']/2**30 for r in rr],[r[key]/(1000 if 'ms' in key else 1) for r in rr],marker=marker,c=color,label='new '+variant)
 for r in rr:
  if r['useful_pct'] is not None:axes[2].scatter(r['prefetch_H2D_bytes']/1e9,r['useful_pct'],marker=marker,c=color)
  if r['variant']=='cap': axes[0].annotate(f"{r['prefetch_fraction']:.0%}", (r['CUDA_allocated_peak_bytes']/2**30,r['TPS']),fontsize=7)
for ax in axes[:2]:ax.set_xlabel('Actual peak CUDA allocation (GiB)');ax.grid(alpha=.25);ax.legend(fontsize=7)
axes[0].set_ylabel('Output tokens/s (higher is better)');axes[1].set_ylabel('p95 first-token latency (s; lower is better)')
axes[2].set_xlabel('Completed speculative transfers (GB)');axes[2].set_ylabel('Bytes used by intended request (%)');axes[2].grid(alpha=.25)
fig.suptitle('Prefetch efficiency screen — actual-output workflow; individual-run points, no fitted curve',fontsize=11);fig.tight_layout()
fig.savefig(H/'efficiency-frontier.png',dpi=170);fig.savefig(H/'efficiency-frontier.svg')
