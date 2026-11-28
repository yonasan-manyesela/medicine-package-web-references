#!/usr/bin/env python3
"""Figure 7: product identity across visual ranks (labels of 2026-11-28).

All same-medicine images are included, irrespective of packaging-difference flag.
Three of four mutually exclusive identity categories are shown; cannot determine is
omitted from plot but remains in the accompanying results table.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE/'data'/'retrieval_qc_2026-11-28_246_images.csv')
assert len(df) == 246
ranks = [1,2,3]
n = df.groupby('rank').size().reindex(ranks).to_numpy()
assert n.tolist() == [88,83,75]

series = [
    ('Same medicine', 'same', '#168575', 'o'),
    ('Variant', 'variant', '#407BA8', '^'),
    ('Different product', 'different', '#B85E2D', 's'),
]
expected = {'same':[54,41,23], 'variant':[17,16,20], 'different':[6,11,21]}
percentages = {}
for _, key, _, _ in series:
    counts = [int(((df['rank']==r)&(df['revised_primary']==key)).sum()) for r in ranks]
    assert counts == expected[key], (key, counts)
    percentages[key] = 100*np.array(counts)/n
print('Verified rank denominators:',dict(zip(ranks,n)))
print('Verified counts:',expected)
print('Verified percentages:',{key: [round(float(v),1) for v in p] for key,p in percentages.items()})
assert pd.crosstab(df['rank'],df['revised_primary']).sum().sum()==246

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
fig, ax = plt.subplots(figsize=(10.6, 4.7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
x=np.arange(3)
for label, key, color, marker in series:
    ys=percentages[key]
    ax.plot(x,ys, color=color, linewidth=2.8, marker=marker, markersize=9.5,
            label=label, zorder=3)
    for i,yy in enumerate(ys):
        # Keep neighboring rank-3 labels visually separated.
        dx, dy = 0, 10
        if key=='different' and i==1: dy=-19
        if key=='variant' and i==2: dx, dy = -58, -15
        if key=='different' and i==2: dx, dy = 25, -18
        ax.annotate(f'{yy:.1f}%', (x[i], yy), xytext=(dx,dy),
                    textcoords='offset points', ha='center',
                    va='bottom' if dy>0 else 'top', color=color,
                    fontsize=12.5, fontweight='medium')
ax.set_xlim(-.18,2.31)
ax.set_ylim(0,70)
ax.set_xticks(x, [f'Rank {r}\n(n = {int(n[i])})' for i,r in enumerate(ranks)], fontsize=12.5)
ax.set_yticks(np.arange(0,71,10), [f'{v}%' for v in range(0,71,10)], fontsize=11)
ax.set_ylabel('Share of matched images (%)', fontsize=12.5, labelpad=12)
ax.grid(axis='y',color='#E1E6EB',linewidth=0.9,zorder=0)
ax.set_axisbelow(True)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('bottom','left'): ax.spines[s].set_color('#BAC5CD')
ax.tick_params(axis='both',which='both',length=0,pad=9)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=3,fontsize=11.5,
          frameon=False,columnspacing=1.8,handlelength=2.2,handletextpad=.6)
fig.subplots_adjust(left=.105,right=.975,bottom=.17,top=.85)
for ext in ('png','pdf','svg'):
    f=BASE/f'figure7_product_identity_rank_20261128.{ext}'
    fig.savefig(f,dpi=250,facecolor='white')
    print('Saved',f,f.stat().st_size)
plt.close(fig)
