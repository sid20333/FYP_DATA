import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import geometry_check as geom

plt.rcParams.update({
    'font.family': 'serif', 
    'mathtext.fontset': 'cm',
    'font.size': 36, 
    'axes.labelsize': 36, 
    'legend.fontsize': 34,
    'xtick.labelsize': 30, 
    'ytick.labelsize': 24, 
    'axes.linewidth': 0.8,
})

SHORT = {
    'main_m': 'main $m$', 'main_p': 'main $p$', 'main_t': 'main $t$', 'main_aoa': r'main $\alpha$',
    'flap1_m': 'flap1 $m$', 'flap1_p': 'flap1 $p$', 'flap1_t': 'flap1 $t$', 'flap1_aoa': r'flap1 $\alpha$',
    'flap2_m': 'flap2 $m$', 'flap2_p': 'flap2 $p$', 'flap2_t': 'flap2 $t$', 'flap2_aoa': r'flap2 $\alpha$',
    'gap1': 'gap1', 'gap2': 'gap2', 'overlap1': 'overlap1', 'overlap2': 'overlap2',
}

PARAMS = geom.PARAM_NAMES
LO = geom.BOUNDS[:, 0]
HI = geom.BOUNDS[:, 1]
ND = len(PARAMS)

Xl = np.loadtxt('lhs_X.csv', delimiter=',')
Xs = np.loadtxt('sobol_X.csv', delimiter=',')
X_all = np.vstack([Xl, Xs])
src = np.array(['LHS'] * len(Xl) + ['Sobol'] * len(Xs))

print(f'Pool: {len(X_all)} = {len(Xl)} LHS + {len(Xs)} Sobol')

df = pd.read_csv('all_designs.csv', header=None, names=['C_L', 'C_D', 'path'])
df['d_idx'] = df['path'].str.extract(r'/d(\d+)/').astype(float)

S = np.full(len(X_all), -np.inf)
CL = np.full(len(X_all), np.nan)
CD = np.full(len(X_all), np.nan)

for pool_name, offset in [('lhs', 0), ('sobol', 200)]:
    sub = df[df['path'].str.contains(f'/campaign_runs/{pool_name}/')].copy()
    for _, row in sub.iterrows():
        idx = int(row['d_idx']) - 1 + offset
        if 0 <= idx < 400:
            CL[idx] = row['C_L']
            CD[idx] = row['C_D']
            S[idx] = row['C_L'] - 4.0 * row['C_D']

valid = S > -np.inf
Xv = X_all[valid]
Sv = S[valid]
CLv = CL[valid]
CDv = CD[valid]
sv = src[valid]

print(f'Valid (have CFD): {len(Xv)} ({(sv == "LHS").sum()} LHS + {(sv == "Sobol").sum()} Sobol)')

Xn = (Xv - LO) / (HI - LO)
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xn)
labs = km.labels_

basins = []
for c in range(5):
    mask = labs == c
    idx = np.where(mask)[0]
    bi = idx[np.argmax(Sv[idx])]
    nL = int((sv[mask] == 'LHS').sum())
    nS = int((sv[mask] == 'Sobol').sum())
    
    basins.append({
        'cluster_raw': c, 
        'n': int(mask.sum()), 
        'nL': nL, 
        'nS': nS,
        'X_best': Xv[bi],
        'CL': float(CLv[bi]), 
        'CD': float(CDv[bi]),
        'S': float(Sv[bi]), 
        'src': sv[bi]
    })

basins.sort(key=lambda b: -b['S'])

print('\n=== Basin breakdown (k=5, sorted by best score) ===')
print(f"{'basin':>5} | {'size':>4} | {'LHS':>4} | {'Sobol':>5} | {'best S':>7} | {'CL':>5} | {'CD':>6} | {'from':>5}")
print('-' * 65)

for r, b in enumerate(basins):
    print(f"  r{r:<3} | {b['n']:>4} | {b['nL']:>4} | {b['nS']:>5} | {b['S']:>7.4f} | {b['CL']:>5.3f} | {b['CD']:>6.4f} | {b['src']:>5}")

theta = np.linspace(0, 2 * np.pi, ND, endpoint=False).tolist()
theta_c = theta + [theta[0]]

fig = plt.figure(figsize=(22, 14))
ax = fig.add_subplot(111, polar=True)
ax.set_position([0.02, 0.05, 0.54, 0.90])
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_facecolor('#fbfbfb')

colors = ['#3b6db5', '#c0504d', '#5a9d5a', '#8064a2', '#d99c4a']
markers = ['o', 's', '^', 'D', 'v']

for r, b in enumerate(basins):
    Xn_b = (b['X_best'] - LO) / (HI - LO)
    vals = list(Xn_b) + [Xn_b[0]]
    ax.plot(theta_c, vals, '-', marker=markers[r], color=colors[r],
            lw=2.6, ms=10, mec='white', mew=1.2,
            label=rf"r{r}  ($f(\mathbf{{X}})={b['S']:.3f},\ n={b['n']}$)")
    ax.fill(theta_c, vals, color=colors[r], alpha=0.12)

def _fmt(v):
