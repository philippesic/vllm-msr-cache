"""Standalone plots of observed points, never fits or confidence intervals."""
import argparse,json,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='SCREEN_RESULTS.json');ap.add_argument('--reference',choices=['predeclared','highest-observed'],default='predeclared');a=ap.parse_args();j=json.loads((H/a.input).read_text());rows=j['rows'];complete=[r for r in rows if r['complete_load_eligible'] and r['CUDA_allocated_peak_bytes'] is not None];colors={'arc':'#2563a6','candidate':'#dc7d20'}
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,grid=plt.subplots(2,2,figsize=(12,6),height_ratios=[12,1.2],layout='constrained');axs=grid[0];strips=grid[1]
 tpskey='TPS_fraction_highARC' if a.reference=='predeclared' else 'TPS_fraction_highest_observedARC';latkey='TTFT_p95_ratio_highARC' if a.reference=='predeclared' else 'TTFT_p95_ratio_highest_observedARC';anchorlabel='predeclared largest-budget ARC' if a.reference=='predeclared' else 'highest-observed ARC (posthoc)'
 stem='vram-frontier' if a.reference=='predeclared' else 'vram-frontier-highestARC-sensitivity'
 for arm in ['arc','candidate']:
  for phase in ['screen','confirmation']:
   q=[r for r in complete if r['arm']==arm and r['phase']==phase];x=[r['CUDA_allocated_peak_bytes']/2**30 for r in q]
   if not q:continue
   for active in [True,False]:
    subset=[(r,xx) for r,xx in zip(q,x) if (arm=='arc' or (r['numerical_calls']>0 and r['native_adoptions']>0))==active]
    if not subset:continue
    label=(f'ARC: {phase}' if arm=='arc' else f'candidate: {phase}, '+('score + adopt' if active else 'no adoption'))
    options=dict(color=colors[arm],marker='o' if phase=='screen' else 'D',s=50,facecolors=colors[arm] if active and phase=='screen' else 'none',label=label)
    axs[0].scatter([xx for r,xx in subset],[r.get(tpskey,float('nan')) for r,xx in subset],**options);axs[1].scatter([xx for r,xx in subset],[r.get(latkey,float('nan')) for r,xx in subset],**options)
   if phase=='screen':
    for r,xx in zip(q,x):axs[0].annotate(str(r['gpu_blocks']),(xx,r.get(tpskey,0)),xytext=(3,7 if arm=='arc' else -11),textcoords='offset points',fontsize=7)
 for y in [.99,.97,.95]:axs[0].axhline(y,color='gray',lw=.6,ls='--');axs[0].text(.99,y,f'{y:.0%}',transform=axs[0].get_yaxis_transform(),ha='right',va='bottom',fontsize=8)
 for y in [1.05,1.10]:axs[1].axhline(y,color='gray',lw=.6,ls='--');axs[1].text(.99,y,f'{y:.0%}',transform=axs[1].get_yaxis_transform(),ha='right',va='bottom',fontsize=8)
 invalid=[r for r in rows if r['outcome']=='strict_proof_invalid']
 for ax,strip in zip(axs,strips):
  strip.set_xlim(ax.get_xlim());strip.set_ylim(0,1);strip.set_yticks([]);strip.set_xticks([])
  for spine in strip.spines.values():spine.set_visible(False)
  strip.set_ylabel('Invalid\nproof',rotation=0,labelpad=20,fontsize=8,color='#ae2835')
  for index,r in enumerate(invalid):
   xx=r['CUDA_allocated_peak_bytes']/2**30;yy=.2+.55*index
   strip.scatter([xx],[yy],marker='x',s=55,color='#ae2835')
   strip.annotate(f"{r['arm']} b{r['gpu_blocks']}",(xx,yy),xytext=(6,0),textcoords='offset points',va='center',fontsize=7,color='#ae2835')
  strip.set_xlabel('Measured peak CUDA allocation (GiB), model + KV + runtime');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axs[0].set_ylabel('Output throughput / '+anchorlabel);axs[1].set_ylabel('p95 first-token latency / '+anchorlabel)
 fig.suptitle('Observed VRAM frontier — public agent24 / maxactive8 / AWQ4bit7B'+('\nPosthoc highest-observed ARC sensitivity' if a.reference!='predeclared' else ''))
 cap=[r for r in rows if r['outcome'].endswith('time_capped')];invalid=[r for r in rows if r['outcome']=='strict_proof_invalid'];caption='Labels: configured16-token KV blocks. One screening partition; no fit or confidence interval. Generated histories can differ.'
 if cap:caption+=' Censored budgets excluded from complete-load normalization: '+','.join(f"{r['arm']}b{r['gpu_blocks']}" for r in cap)
 if invalid:caption+=' Proof-invalid points excluded: '+','.join(f"{r['arm']}b{r['gpu_blocks']}" for r in invalid)
 fig.text(.5,-.025,caption,ha='center',fontsize=8,wrap=True)
 for suffix in ['png','svg']:fig.savefig(H/(stem+'.'+suffix),dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for arm in ['arc','candidate']:
  q=sorted([r for r in rows if r['phase']=='screen' and r['arm']==arm],key=lambda r:r['gpu_blocks']);x=[r['GPU_KV_MiB'] for r in q]
  for ax,key,label in [(axs[0,0],'TTFT_p95_ms','p95 first-token latency (seconds)'),(axs[0,1],'CPU_s','Server CPU seconds per finite load'),(axs[1,0],'preemptions','Native preemptions')]:
   valid=[(r,xx) for r,xx in zip(q,x) if r.get(key) is not None]
   for completed in [True,False]:
    data=[(r,xx) for r,xx in valid if r['complete_load_eligible']==completed]
    if data:ax.scatter([xx for r,xx in data],[r[key]/1000 if key=='TTFT_p95_ms' else r[key] for r,xx in data],color=colors[arm],marker='o' if completed else 'x',s=45,label=arm+(' complete' if completed else ' unaccepted'))
   ax.set_ylabel(label);ax.grid(alpha=.2)
  if arm=='candidate':
   for accepted in [True,False]:
    good=[(r,xx) for r,xx in zip(q,x) if r.get('prefetch_H2D_bytes') is not None and r['complete_load_eligible']==accepted]
    axs[1,1].scatter([xx for r,xx in good],[r['prefetch_H2D_bytes']/2**20 for r,xx in good],color=colors[arm],marker='o' if accepted else 'x',label='completed speculative H2D'+('' if accepted else ' (invalid run)'))
    good=[(r,xx) for r,xx in zip(q,x) if r.get('named_consumed_prefetch_bytes') is not None and r['complete_load_eligible']==accepted]
    axs[1,1].scatter([xx for r,xx in good],[r['named_consumed_prefetch_bytes']/2**20 for r,xx in good],color='#27824d',marker='D' if accepted else 'x',label='consumed by named demand'+('' if accepted else ' (invalid run)'))
 axs[0,0].axhline(15,color='gray',ls='--',lw=.8,label='prior15s agent SLO');axs[1,1].set_ylabel('Observed speculative transfer volume (MiB)')
 for ax in axs.flat:ax.set_xlabel('Configured GPU KV pool (MiB); weights/runtime additional');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Operational costs — crosses are unaccepted points; see proof/cap flags')
 for suffix in ['png','svg']:fig.savefig(H/('vram-operational-costs.'+suffix),dpi=180,bbox_inches='tight')
 plt.close(fig)
 (H/('PLOT_RECEIPT.json' if a.reference=='predeclared' else 'PLOT_SENSITIVITY_RECEIPT.json')).write_text(json.dumps({'tool':'matplotlib','version':matplotlib.__version__,'input':a.input,'reference':a.reference,'artifact_files':[stem+'.png',stem+'.svg','vram-operational-costs.png','vram-operational-costs.svg'],'fitted_model':False,'confidence_interval':False,'censored_points_normalized':False},indent=2)+'\n')
 print('four standalone plot artifacts written')
if __name__=='__main__':main()
