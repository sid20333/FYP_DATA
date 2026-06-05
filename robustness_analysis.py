import json
import numpy as np
import cma
import run_optimisers_v2 as R
import geometry_check as geom

B = np.array(geom.BOUNDS, float)
RNG = B[:, 1] - B[:, 0]
EPS = 0.02 * RNG

with open('recovered/best_X.json') as f:
    bx = json.load(f)

designs = {
    'PSO_3.0666': np.array(bx['pso_best_current']['X'], float),
    'CMA_3.0661': np.array(bx['cmaes_best_current']['X'], float)
}

fd = np.load('fd_sensitivity_results.npz', allow_pickle=True)

seeds = {}
for key, x0 in designs.items():
    dfwd = fd[f'{key}_dfwd']
    dbwd = fd[f'{key}_dbwd']
    lin = float(np.sum(np.minimum(np.minimum(dfwd, dbwd), 0.0)))
    dseed = np.where(dfwd <= dbwd, EPS, -EPS)
    seeds[key] = np.clip(x0 + dseed, B[:, 0], B[:, 1])

results = {}
for name, x0 in designs.items():
    lo = np.clip(x0 - EPS, B[:, 0], B[:, 1])
    hi = np.clip(x0 + EPS, B[:, 0], B[:, 1])
    
    cl0, cd0 = R.batch_eval(x0[None, :], f'wc_{name}/nom', slurm_name='wc.slurm')
    s0 = R.score(cl0, cd0)[0]
    
    es = cma.CMAEvolutionStrategy(
        seeds[name].tolist(), 
        float(np.mean(EPS) / 2),
        {
            'bounds': [lo.tolist(), hi.tolist()], 
            'popsize': 10, 
            'maxfevals': 80, 
            'verbose': -9
        }
    )
    
    worst = np.inf
    wx = None
    g = 0
    
    while not es.stop():
        sols = np.array(es.ask())
        cl, cd = R.batch_eval(sols, f'wc_{name}/g{g}', slurm_name='wc.slurm')
        s = R.score(cl, cd)
        valid = s > R.PENALTY
        
        es.tell(sols.tolist(), [float(v) if vd else 100.0 for v, vd in zip(s, valid)])
        
        if valid.any() and s[valid].min() < worst:
            worst = float(s[valid].min())
            wx = sols[valid][s[valid].argmin()]
            
        g += 1
        
    dd = (wx - x0) / RNG * 100 if wx is not None else None
    results[name] = dict(s0=s0, worst=worst, wx=wx, x0=x0, pct=dd)

save_dict = {}
for k in results:
    for m in ['s0', 'worst', 'x0', 'pct']:
        if results[k][m] is not None:
            save_dict[f'{k}_{m}'] = results[k][m]
    if results[k].get('wx') is not None:
        save_dict[f'{k}_wx'] = results[k]['wx']

np.savez('worst_case_results.npz', **save_dict)
