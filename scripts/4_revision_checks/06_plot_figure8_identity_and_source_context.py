"""Balanced-spacing publication figure. No legend or panel divider. Consistent text size.
Reproduces 21-Nov-2026 verified results from included data/ inputs.
"""
from pathlib import Path
import csv
from collections import Counter, defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
DATA = OUT / 'data'


def load(filename):
    with (DATA / filename).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

images = load('retrieval_qc_2026-11-28_246_images.csv')
source = load('identity_context_images.csv')
consistency = load('panel_c_reconciled_2026-11-28.csv')
source_by_rank = {(r['query_id'], r['rank']): r for r in source}
per_query = defaultdict(list)
for r in images:
    s = source_by_rank[(r['query_id'], r['rank'])]
    per_query[r['query_id']].append({**r, 'page_context': s['context']})


def n_queries(check):
    return sum(any(check(r) for r in rows) for rows in per_query.values())

retrieval = [
    ('At least one visually matched image', len(per_query)),
    ('Same medicine among top three', n_queries(lambda r: r['revised_primary'] == 'same')),
    ('Same medicine; no packaging difference observed', n_queries(lambda r: r['revised_primary'] == 'same' and r['revised_M'] == '0')),
    ('Same medicine; no packaging difference observed,\nwith a product-information page',
     n_queries(lambda r: r['revised_primary'] == 'same' and r['revised_M'] == '0' and r['page_context'] == 'product_information')),
]
assert [v for _, v in retrieval] == [88,64,48,35]

group_map = {'variant':'Variant', 'different':'Different product', 'cannot_determine':'Cannot determine'}
readable = Counter(); product_info = Counter()
for r in images:
    context = source_by_rank[(r['query_id'], r['rank'])]['context']
    if context in ('not saved', 'no readable text'):
        continue
    group = f"Same, M={r['revised_M']}" if r['revised_primary'] == 'same' else group_map[r['revised_primary']]
    readable[group] += 1
    if context == 'product_information':
        product_info[group] += 1
order_b = ['Same, M=0', 'Variant', 'Same, M=1', 'Different product', 'Cannot determine']
assert [(product_info[g],readable[g]) for g in order_b] == [(48,52), (35,37), (34,40), (12,31), (31,35)]
labels_b = {
    'Same, M=0':'Same medicine; no packaging difference observed',
    'Variant':'Variant',
    'Same, M=1':'Same medicine; packaging differs',
    'Different product':'Different product',
    'Cannot determine':'Cannot determine',
}

consistency_counts = Counter((r['new_group'],r['outcome']) for r in consistency)
order_c = ['Same, M=0','Variant','Same, M=1','Cannot determine']
actual = {g:tuple(consistency_counts[(g,out)] for out in ('Contradiction','All three agree','Incomplete evidence')) for g in order_c}

FONT_SIZE = 15
BAR_HEIGHT = 34
LABEL_TO_BAR = 16
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': FONT_SIZE,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.unicode_minus': False,
})
W, H = 1448, 890
fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor='white')
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

LIGHT_BLUE = '#9dbbcf'
FOCUS = '#104b78'
TRACK = '#e6e8ea'
BROWN = '#c56f3e'
TEAL = '#2d8c7b'
INCOMPLETE = '#99b8cd'
TEXT = '#121212'
MUTED = '#5e6e77'

def txt(x, y, s, *, heading=False, color=TEXT, ha='left', va='center'):
    return ax.text(x, y, s, fontsize=FONT_SIZE,
                   fontweight='bold' if heading else 'normal', color=color,
                   ha=ha, va=va, zorder=6, linespacing=1.05)


def trackbar(x, y, w, fraction, *, focus=False):
    ax.add_patch(Rectangle((x, y), w, BAR_HEIGHT, facecolor=TRACK, edgecolor='none', zorder=1))
    ax.add_patch(Rectangle((x, y), w*fraction, BAR_HEIGHT,
                           facecolor=FOCUS if focus else LIGHT_BLUE,
                           edgecolor='none', zorder=2))

# Panel a

txt(20, 44, '(a) Web image retrieval and candidate yield', heading=True)
for i, (lab, value) in enumerate(retrieval):
    y = 88 + i*100
    # split final long label slightly higher but keep bar close
    txt(24, y, lab)
    line_count = lab.count('\n') + 1
    bar_y = y + LABEL_TO_BAR + (line_count-1)*14
    trackbar(24, bar_y, 502, value/100, focus=(i == 3))
    txt(548, bar_y + BAR_HEIGHT/2, f'{value}/100')

# Panel b

txt(716, 44, '(b) Product-information pages by image identity', heading=True)
for i, g in enumerate(order_b):
    y = 88 + i*74
    txt(720, y, labels_b[g], color=TEXT)
    bar_y = y + LABEL_TO_BAR
    n, d = product_info[g], readable[g]
    trackbar(720, bar_y, 480, n/d, focus=(g == 'Variant'))
    txt(1222, bar_y + BAR_HEIGHT/2, f'{n}/{d} ({n/d:.0%})')

# Panel c

txt(20, 500, '(c) Source-page attribute consistency', heading=True)
labels_c = {
    'Same, M=0':'Same medicine; no packaging difference observed',
    'Variant':'Variant',
    'Same, M=1':'Same medicine; packaging differs',
    'Cannot determine':'Cannot determine',
}
for i,g in enumerate(order_c):
    n = sum(actual[g])
    y = 545 + i*88
    txt(24, y, f'{labels_c[g]} (n={n})')
    bar_y = y + LABEL_TO_BAR
    left, width = 24, 1400
    for j, (num, col) in enumerate(zip(actual[g], (BROWN, TEAL, INCOMPLETE))):
        seg = width*num/n
        if num:
            ax.add_patch(Rectangle((left,bar_y), seg, BAR_HEIGHT, facecolor=col, edgecolor='white', linewidth=0.7, zorder=3))
            label = f'{num}/{n} ({num/n:.0%})' if seg >= 175 else str(num)
            fg = '#15232d' if j == 2 else 'white'
            txt(left+seg/2, bar_y+BAR_HEIGHT/2, label, color=fg, ha='center')
        left += seg

for ext in ('png','pdf','svg'):
    fig.savefig(OUT/f'fig_product_identity_balanced_20261128.{ext}', dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.0)

fig.canvas.draw(); renderer = fig.canvas.get_renderer(); bad=[]
for t in ax.texts:
    bbox = t.get_window_extent(renderer=renderer)
    xright = bbox.x1 / fig.canvas.get_width_height()[0] * W
    if xright > W-12:
        bad.append((t.get_text(), round(xright)))
if bad:
    print('Right edge warnings:', bad)
else:
    print('All text within canvas')
print('done')
plt.close(fig)
