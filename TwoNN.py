import os
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, pearsonr, spearmanr
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import torch
import torch.nn as nn

ROOT = '/Users/shrey/Coursework/FYP'

hist = np.load(f'{ROOT}/convergence_histories.npz', allow_pickle=True)
finals = {opt: np.array([hist[f'{opt}_v2_r{r}_best'][-1] for r in range(5)])
          for opt in ('ga', 'pso', 'cmaes')}

stat, p = friedmanchisquare(finals['ga'], finals['pso'], finals['cmaes'])

pairs = [('pso', 'cmaes'), ('pso', 'ga'), ('cmaes', 'ga')]
wilc = {}
for a, b in pairs:
    res = wilcoxon(finals[a], finals[b], zero_method='wilcox', alternative='two-sided')
    wilc[(a, b)] = res

with open(f'{ROOT}/extra_friedman_wilcoxon.txt', 'w') as f:
    f.write("Final-score per restart (paired by basin r0..r4):\n")
    for opt, arr in finals.items():
        f.write(f"  {opt:>6}: {arr.tolist()}\n")
    f.write(f"\nFriedman (k=3 optimisers, n=5 restarts):\n  chi² = {stat:.4f}\n  p    = {p:.4f}\n")
    f.write("\nPairwise Wilcoxon (two-sided):\n")
    for (a, b), res in wilc.items():
        f.write(f"  {a:>6} vs {b:<6}: W = {res.statistic:.3f}, p = {res.pvalue:.4f}\n")

def load_X(fn):
    d = np.load(f'{ROOT}/{fn}', allow_pickle=True)
    return d['X'], d['ok'] if 'ok' in d.files else np.ones(len(d['X']), bool)

pools = ['merged_d10_pool.npz', 'recovered_latent_v2_d10.npz',
         'recovered_latent_v2_d8.npz', 'recovered_latent_sobol.npz']
X_list = []
for fn in pools:
    X, ok = load_X(fn)
    X_list.append(X[ok])

X_all = np.vstack(X_list)
X_all = np.unique(np.round(X_all, 10), axis=0)

BOUNDS = np.array([
    [0.04, 0.09], [0.20, 0.50], [0.12, 0.16], [-5.0,  5.0],
    [0.04, 0.09], [0.30, 0.60], [0.10, 0.14], [10.0, 27.5],
    [0.04, 0.09], [0.30, 0.60], [0.10, 0.13], [15.0, 45.0],
    [0.005, 0.030], [0.003, 0.020], [0.005, 0.040], [0.003, 0.025],
])
X_norm = (X_all - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
X_norm = np.clip(X_norm, 0, 1)

def twonn(X, fraction=0.9):
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=3)
    r1, r2 = dists[:, 1], dists[:, 2]
    mask = (r1 > 0)
    mu = r2[mask] / r1[mask]
    mu_sorted = np.sort(mu)
    n_use = int(fraction * len(mu_sorted))
    mu_use = mu_sorted[:n_use]
    F = np.arange(1, n_use + 1) / len(mu_sorted)
    y = np.log(mu_use)
    x = -np.log(1 - F + 1e-12)
    slope, _ = np.polyfit(x, y, 1)
    return 1.0 / slope, mu

d_id, mu = twonn(X_norm)

ds = []
rng = np.random.default_rng(0)
for _ in range(40):
    idx = rng.choice(len(X_norm), size=min(400, len(X_norm)), replace=False)
    di, _ = twonn(X_norm[idx])
    ds.append(di)
ds = np.array(ds)

class AE(nn.Module):
    def __init__(self, in_dim=16, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, 512),    nn.LayerNorm(512),  nn.SiLU(),
            nn.Linear(512, 256),     nn.LayerNorm(256),  nn.SiLU(),
            nn.Linear(256, 128),     nn.LayerNorm(128),  nn.SiLU(),
            nn.Linear(128, 64),      nn.LayerNorm(64),   nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.LayerNorm(64),   nn.SiLU(),
            nn.Linear(64, 128),         nn.LayerNorm(128),  nn.SiLU(),
            nn.Linear(128, 256),        nn.LayerNorm(256),  nn.SiLU(),
            nn.Linear(256, 512),        nn.LayerNorm(512),  nn.SiLU(),
            nn.Linear(512, 1024),       nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, in_dim),    nn.Sigmoid(),
        )
    def encode(self, x):
        return self.encoder(x)

ckpt = torch.load(f'{ROOT}/ae_d10_ckpt.pt', map_location='cpu', weights_only=False)
ae = AE(16, ckpt['latent_dim'])
ae.load_state_dict(ckpt['state_dict'])
ae.eval()

LO = ckpt['LO'].numpy() if hasattr(ckpt['LO'], 'numpy') else np.asarray(ckpt['LO'])
HI = ckpt['HI'].numpy() if hasattr(ckpt['HI'], 'numpy') else np.asarray(ckpt['HI'])
X_for_ae = (X_all - LO) / (HI - LO + 1e-12)
with torch.no_grad():
    Z = ae.encode(torch.tensor(X_for_ae, dtype=torch.float32)).numpy()

def standardise(A):
    A = np.asarray(A)
    s = A.std(axis=0, keepdims=True)
    s[s < 1e-12] = 1.0
    return (A - A.mean(axis=0, keepdims=True)) / s

N_SUB = min(800, len(X_norm))
rng = np.random.default_rng(0)
idx = rng.choice(len(X_norm), size=N_SUB, replace=False)

dX = pdist(standardise(X_norm[idx]))
dZ = pdist(standardise(Z[idx]))

pr = pearsonr(dX, dZ).statistic
sr = spearmanr(dX, dZ).statistic

BATCH = 8   

curves = {}
for opt in ('cmaes', 'pso', 'ga'):
    traces = []
    for r in range(5):
        bs = hist[f'{opt}_v2_r{r}_best']
        if len(bs) < 2: 
            continue
        total = bs[-1] - bs[0]
        if abs(total) < 1e-12: 
            continue
        frac = (bs - bs[0]) / total
        n_axis = np.arange(1, len(bs) + 1) * BATCH
        traces.append((n_axis, frac))
    curves[opt] = traces

grid = np.arange(BATCH, 100 * BATCH + 1, BATCH)  
for opt in ('cmaes', 'pso', 'ga'):
    interps = []
    for n_axis, frac in curves[opt]:
        f = np.interp(grid, n_axis, frac, left=0, right=frac[-1])
        interps.append(f)
    M = np.array(interps)

for opt in ('cmaes', 'pso', 'ga'):
    n50, n80, n95 = [], [], []
    for n_axis, frac in curves[opt]:
        for thr, lst in [(0.5, n50), (0.8, n80), (0.95, n95)]:
            idx = np.searchsorted(frac, thr)
            if idx < len(n_axis):
                lst.append(n_axis[idx])
            else:
                lst.append(n_axis[-1])
