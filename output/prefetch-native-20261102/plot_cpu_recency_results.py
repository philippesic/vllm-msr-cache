"""Publication-style standalone figures from retained small measured summaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent

def main():
 curve=json.loads((H/'cpu-recency-v3-curves.json').read_text())
 result=json.loads((H/'cpu-recency-v3-confirm-analysis.json').read_text())
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 colors={'arc':'#2864ad','candidate':'#a3459b'};labels={'arc':'ARC','candidate':'Wait-hint CPU recency'}
 for arm in ['arc','candidate']:
  rows=sorted((r for r in curve['rows'] if r['seed']==1103 and r['arm']==arm and r['gpu_blocks'] in [1792,2048,2112]),key=lambda r:r['gpu_blocks'])
  for ax,key,div in [(axes[0,0],'throughput_tokens_s',1),(axes[0,1],'all_mean_ttft_ms',1000)]:
   xs=[r['nvml_devicewide_peak_bytes']/2**30 for r in rows];ys=[r[key]/div for r in rows]
   ax.plot(xs,ys,'o-' if arm=='arc' else 's-',color=colors[arm],label=labels[arm],lw=1.6,ms=6)
   for x,y,r in zip(xs,ys,rows):ax.annotate(str(r['gpu_blocks']),(x,y),xytext=(4,6 if arm=='candidate' else -14),textcoords='offset points',fontsize=8,color=colors[arm])
 controls=[r for r in curve['rows'] if r['seed']==1103 and r['arm']=='arc' and r['gpu_blocks'] not in [1792,2048,2112]]
 for ax,key,div in [(axes[0,0],'throughput_tokens_s',1),(axes[0,1],'all_mean_ttft_ms',1000)]:
  ax.scatter([r['nvml_devicewide_peak_bytes']/2**30 for r in controls],[r[key]/div for r in controls],marker='D',facecolors='none',edgecolors=colors['arc'],label='Nearby ARC controls',s=35)
 for ax in axes[0]:
  ax.set_xlabel('Serving-load device VRAM peak (GiB)');ax.grid(alpha=.18);ax.legend(fontsize=8)
 axes[0,0].set(title='Discovery curve: first trace (one run per point)',ylabel='Output tokens/s — higher is better')
 axes[0,1].set(title='First-token latency on that same trace',ylabel='Mean first-token latency (s) — lower is better')
 ax=axes[1,0];stats=[result['statistics'],result['throughput_per_VRAM_secondary']['nvml_devicewide_peak_bytes']]
 for y,s in enumerate(stats):
  mean=100*s['geometric_mean_gain_fraction'];lo=100*s['lower95_gain_fraction'];hi=100*s['upper95_gain_fraction']
  ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color='#367b50')
 ax.set_yticks([0,1],['Throughput','Throughput / device GiB']);ax.axvline(0,color='#777',lw=1);ax.grid(axis='x',alpha=.18)
 ax.set(title='Fresh confirmation: 12 pairs, three traces',xlabel='Candidate improvement (%) with paired 95% interval');ax.invert_yaxis()
 ax=axes[1,1]
 metrics=[('all_ttft_saving_ms','All: mean'),('all_p95_ttft_saving_ms','All: p95'),('return_ttft_saving_ms','Returning: mean'),('return_p95_ttft_saving_ms','Returning: p95')]
 for y,(key,label) in enumerate(metrics):
  s=result[key];ax.errorbar(s['mean'],y,xerr=[[s['mean']-s['lower95']],[s['upper95']-s['mean']]],fmt='o',capsize=5,color='#9b652e')
 ax.set_yticks(range(len(metrics)),[m[1] for m in metrics]);ax.axvline(0,color='#777',lw=1);ax.grid(axis='x',alpha=.18);ax.invert_yaxis()
 ax.set(title='Fresh confirmation: latency trade-offs',xlabel='ARC minus candidate TTFT (ms); positive favors candidate')
 fig.suptitle('Causal CPU retention versus ARC\nQwen2.5-7B-Instruct AWQ / FP16 · RTX 2080 Ti / WSL',fontsize=14)
 fig.supxlabel('Recorded-output serving-compute replay with normal model forwards; 24 sessions, 8 active, 3 turns.\nSame 3.5 GiB CPU cache. Confirmation: candidate 2,048 versus ARC 2,112 GPU blocks. No speculative GPU copies.',fontsize=9)
 for ext in ['svg','png']:fig.savefig(H/f'cpu-recency-v3-results.{ext}',dpi=180)
 plt.close(fig)
 print('Saved standalone PNG/SVG curves and fresh paired intervals; plots do not substitute for byte/integration qualification.')
if __name__=='__main__':main()
