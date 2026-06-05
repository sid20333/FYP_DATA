import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist

DIMS = [4, 8, 12, 16]
N = 1024
N_SEEDS = 20

def maximin(X):
    return pdist(X).min()

def lhs(d, seed):
    return qmc.LatinHypercube(d=d, seed=seed).random(N)

def sobol(d):
    return qmc.Sobol(d=d, scramble=False).random(N)

def rand(d, seed):
    return np.random.default_rng(seed).random((N, d))

lhs_mean, lhs_std = [], []
sob_det = []
rnd_mean, rnd_std = [], []

for d in DIMS:
    lhs_vals = [maximin(lhs(d, s)) for s in range(N_SEEDS)]
    rnd_vals = [maximin(rand(d, s)) for s in range(N_SEEDS)]
    
    lhs_mean.append(np.mean(lhs_vals))
    lhs_std.append(np.std(lhs_vals))
    
    rnd_mean.append(np.mean(rnd_vals))
    rnd_std.append(np.std(rnd_vals))
    
    sob_det.append(maximin(sobol(d)))
