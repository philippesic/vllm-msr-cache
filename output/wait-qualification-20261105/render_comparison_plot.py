"""Export the complete retained comparison table with standard Matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path(__file__).resolve().parent
rows=json.loads((b/'all-wait-arc-comparisons.json').read_text())['rows']
fig,ax=plt.subplots(figsize=(12,max(5,len(rows)*.32+1.8)))
for i,r in enumerate(rows):
 v=r['time_reduction_pct'];lo=r['ci_low_pct'];hi=r['ci_high_pct']
 if v is None:
  ax.text(.02,i,'no eligible pair',transform=ax.get_yaxis_transform(),color='gray',fontsize=8,va='center')
  continue
 c='#1769a1' if v>=0 else '#bb4b18'
 if lo is not None:ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=c,markersize=4,capsize=3,lw=1)
 else:ax.plot(v,i,'o',color=c,markersize=4)
 ax.annotate(f'{v:+.2f}%',(v,i),xytext=(5,5),textcoords='offset points',fontsize=7)
labels=[r['campaign']+' ['+r['display_tag']+']'+(' ['+r['status']+']' if r['status']!='passed' else '') for r in rows]
ax.set_yticks(range(len(rows)),labels,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8,ls='--');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
ax.set_xlabel('Batch time saved versus plain ARC (%) — positive favors wait-hint')
ax.set_title('All retained wait-hint / plain ARC comparisons',loc='left',fontsize=12)
fig.text(.01,.006,'Conditional 95% whole-run intervals; no pooling. dev=development; fresh=fresh validation tasks reused across memory cells.\nfunc/replay/diag=diagnostics (not application speed); confirm=separate confirmation; regime=original workload; exposure=finite churn.\ncompat/profile=selected integration timing. No general useful-answer, production-reliability or other-environment claim.',fontsize=8)
fig.tight_layout(rect=[0,.055,1,1])
for suffix in ('png','pdf'):fig.savefig(b/f'all-wait-arc-comparisons.{suffix}',dpi=200,bbox_inches='tight')
print('Exported PNG and PDF for',len(rows),'rows')
