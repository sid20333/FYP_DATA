import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def classify(path):
    if '/refine/cmaes' in path: return 'refine CMA-ES'
    if '/refine/pso' in path: return 'refine PSO'
    if '/latent/sobol' in path: return 'latent Sobol (D=10)'
    if '/latent/cmaes' in path: return 'latent CMA-ES (D=10)'
    if '/latent/pso' in path: return 'latent PSO (D=10)'
    if '/lhs/' in path: return 'LHS (16D)'
    if '/sobol/' in path: return '16D Sobol'
    if '/ga/' in path: return 'v2 GA'
    if '/pso/' in path: return 'v2 PSO'
    if '/cmaes/' in path: return 'v2 CMA-ES'
    return 'other'

groups = defaultdict(lambda: ([], []))
all_rows = []

with open('all_designs.csv', 'r') as f:
    for r in csv.reader(f):
        if len(r) < 3:
            continue
        try:
            cl, cd = float(r[0]), float(r[1])
        except ValueError:
            continue
        
        src = classify(r[2])
        groups[src][0].append(cl)
        groups[src][1].append(cd)
        all_rows.append((cl, cd, src, r[2]))

all_rows.sort(key=lambda r: -(r[0] - 4 * r[1]))
top10 = all_rows[:10]

def pareto(rows):
    pts = sorted(rows, key=lambda r: -r[0])
    front = []
    min_cd = np.inf
    for r in pts:
        if r[1] < min_cd:
            front.append(r)
            min_cd = r[1]
    return sorted(front, key=lambda r: r[1])

front = pareto(all_rows)

colors = {
    'LHS (16D Sobol)': '#999999',
    '16D Sobol': '#777777',
    'v2 GA': '#cc7777',
    'v2 PSO': '#cc4444',
    'v2 CMA-ES': '#aa2222',
    'refine CMA-ES': '#7777cc',
    'refine PSO': '#3333aa',
    'latent Sobol (D=10)': '#77cc77',
    'latent CMA-ES (D=10)': '#33aa33',
    'latent PSO (D=10)': '#117711',
    'other': '#000000',
}

order = [
    'LHS (16D Sobol)', '16D Sobol', 'v2 GA', 'v2 PSO', 'v2 CMA-ES',
    'refine CMA-ES', 'refine PSO', 'latent Sobol (D=10)', 
    'latent CMA-ES (D=10)', 'latent PSO (D=10)', 'other'
]

fig, ax = plt.subplots(figsize=(10, 7))

for src in order:
    if src not in groups or not groups[src][0]:
        continue
    cl = np.array(groups[src][0])
    cd = np.array(groups[src][1])
    ax.scatter(cd, cl, s=8, c=colors.get(src, '#000000'), alpha=0.55, 
               label=f'{src} (n={len(cl)})', edgecolor='none')

pf_cd = [r[1] for r in front]
pf_cl = [r[0] for r in front]
ax.plot(pf_cd, pf_cl, '-', color='black', lw=1.3, alpha=0.65, label=f'Pareto front (n={len(front)})')

t_cd = [r[1] for r in top10]
t_cl = [r[0] for r in top10]
ax.scatter(t_cd, t_cl, s=110, facecolors='none', edgecolors='gold', lw=2.0, label='Top-10 by CL−4·CD')

best_cl, best_cd = top10[0][0], top10[0][1]
ax.annotate(f'best: CL={best_cl:.3f}\nCD={best_cd:.4f}\nscore={best_cl - 4 * best_cd:.4f}',
            xy=(best_cd, best_cl), xytext=(best_cd + 0.02, best_cl - 0.3),
            fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))

score_lines = [2.5, 2.8, 3.0, 3.05]
max_cd = max(r[1] for r in all_rows) * 1.05
cd_axis = np.linspace(0, max_cd, 50)

for s in score_lines:
    ax.plot(cd_axis, s + 4 * cd_axis, ':', color='grey', alpha=0.5, lw=0.7)
    ax.text(cd_axis[-1], s + 4 * cd_axis[-1], f' score={s}', fontsize=7, color='grey', va='center')

ax.set_xlabel('CD (drag coefficient) — lower is better')
ax.set_ylabel('CL (downforce coefficient) — higher is better')
ax.set_title(f'F1 wing campaign — CL vs CD ({len(all_rows)} designs total)')
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('football_plot.pdf')
plt.savefig('football_plot.png', dpi=140)

print('saved → football_plot.pdf / .png\n')
print('Top-10:')
for i, (cl, cd, src, p) in enumerate(top10, 1):
    print(f'  {i:2d}. score={cl - 4 * cd:.4f}  CL={cl:.4f}  CD={cd:.4f}  ({src})')







