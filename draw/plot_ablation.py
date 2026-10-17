#!/usr/bin/env python3
"""Plot the three RQ2 figures from recomputed metrics (no checkpoints required)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
 'selection':['full','random','static','searched_static','combination','prompt','combination_mlp','state_mlp','mean_mlp'],
 'components':['full','wo_bpr','wo_ce','wo_error_emb','wo_pos_emb','wo_code','wo_verus'],
 'length':['len1','len2','len3','len4','full'],
}
LABELS=dict(full='VeruSteer',random='Random',static='Learned static priority',searched_static='Searched static priority',
 combination='Combination rule',prompt='Prompt-based selector',combination_mlp='Combination MLP',state_mlp='State MLP',
 mean_mlp='Mean-pooled MLP',wo_bpr='w/o BPR loss',wo_ce='w/o CE loss',wo_error_emb='w/o error-type emb.',
 wo_pos_emb='w/o positional emb.',wo_code='w/o program emb.',wo_verus='w/o feedback emb.',
 len1='L = 1',len2='L = 2',len3='L = 3',len4='L = 4')

def main():
    with (ROOT/'outputs/metrics/spaces.csv').open() as f:
        rows={(r['variant'],r['space']):r for r in csv.DictReader(f)
              if r['pipeline']=='verusage' and r['objective']=='verification' and r['configuration']=='Total'}
    out=ROOT/'outputs/figures'; out.mkdir(parents=True,exist_ok=True)
    for name,variants in GROUPS.items():
        fig,axes=plt.subplots(1,3,figsize=(12,max(3,len(variants)*.48)),sharey=True)
        for i,space in enumerate(('Retention','Exploration')):
            for y,variant in enumerate(variants):
                r=rows[variant,space]; value=float(r['success_pct'])
                axes[i].barh(y,value,color='#335d8d' if variant=='full' else '#b4c5da')
                axes[i].text(2,y,f"{float(r['successes']):g}/{r['eligible']} ({value:.1f}%)",va='center',fontsize=9)
            axes[i].set(title=space,xlim=(0,105),xlabel='Success rate (%)')
        for y,variant in enumerate(variants):
            value=sum(float(rows[variant,s]['success_pct']) for s in ('Retention','Exploration'))-100
            axes[2].barh(y,value,color='#335d8d' if variant=='full' else '#b4c5da')
            axes[2].text(value+1,y,f'{value:.1f} pts',va='center',fontsize=9)
        axes[2].set(title='Net gain',xlim=(0,85),xlabel='Exploration minus retention drop')
        axes[0].set_yticks(range(len(variants)),[LABELS[v] for v in variants]); axes[0].invert_yaxis()
        for ax in axes:
            ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); fig.savefig(out/f'ablation_{name}.pdf'); plt.close(fig)
    print('Wrote RQ2 figures to',out)

if __name__=='__main__': main()
