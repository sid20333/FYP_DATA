import numpy as np
from sklearn.decomposition import PCA
import geometry_check as geom

X = np.load('training_X_valid_500k.npy').astype(np.float32)
LO = geom.BOUNDS[:, 0].astype(np.float32)
HI = geom.BOUNDS[:, 1].astype(np.float32)

Xn = (X - LO) / (HI - LO)

rng = np.random.default_rng(0)
idx = rng.permutation(len(Xn))
n_tr = int(0.9 * len(Xn))

Xtr = Xn[idx[:n_tr]]
Xval = Xn[idx[n_tr:]]

baseline_var = float(Xval.var(axis=0).mean())

DIMS_PCA = [2, 4, 6, 8, 10, 12, 14, 16]
pca_r2 = {}

for d in DIMS_PCA:
    pca = PCA(n_components=d).fit(Xtr)
    Z = pca.transform(Xval)
    Xrec = pca.inverse_transform(Z)
    mse = float(((Xval - Xrec) ** 2).mean())
    r2 = 1.0 - mse / baseline_var
    pca_r2[d] = r2

ae_r2 = {
    8:  1 - 0.00413 / baseline_var,
    10: 1 - 0.00244 / baseline_var,
    12: 1 - 0.00107 / baseline_var,
}

ae_r2_aggressive = {
    8: 1 - 0.00374 / baseline_var
}

np.savez(
    'pca_results.npz', 
    dims=DIMS_PCA, 
    r2=[pca_r2[d] for d in DIMS_PCA],
    baseline_var=baseline_var
)
