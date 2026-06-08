import json, numpy as np
import run_optimisers_v2 as R
import geometry_check as geom

B = np.array(geom.BOUNDS, float)
NAMES = geom.PARAM_NAMES
DELTA = 0.02 * (B[:, 1] - B[:, 0])

bx = json.load(open('recovered/best_X.json'))
designs = {'PSO_3.0666': np.array(bx['pso_best_current']['X'], float),
           'CMA_3.0661': np.array(bx['cmaes_best_current']['X'], float)}

out = {}
for name, x0 in designs.items():
    rows = [x0.copy()]
    for i in range(16):
        xf = x0.copy(); xf[i] = np.clip(x0[i] + DELTA[i], B[i, 0], B[i, 1]); rows.append(xf)
        xb = x0.copy(); xb[i] = np.clip(x0[i] - DELTA[i], B[i, 0], B[i, 1]); rows.append(xb)
    
    X = np.array(rows)
    cl, cd = R.batch_eval(X, f'fd_{name}', partition='short', slurm_name='fd.slurm')
    s = R.score(cl, cd)
    s0 = s[0]
    
    dfwd = s[1::2] - s0
    dbwd = s[2::2] - s0
    grad = (s[1::2] - s[2::2]) / (2 * DELTA)
    
    out[name] = dict(S=s, s0=s0, cl=cl, cd=cd, dfwd=dfwd, dbwd=dbwd, grad=grad)
    
    print(f'\n=== {name}  (nominal re-eval S0={s0:.4f}) ===', flush=True)
    order = np.argsort(-np.maximum(np.abs(dfwd), np.abs(dbwd)))
    print(f'{"param":>10} {"+2%":>8} {"-2%":>8} {"asym":>9}')
    
    for i in order:
        asym = 'FWD-worse' if dfwd[i] < dbwd[i] - 0.005 else ('BWD-worse' if dbwd[i] < dfwd[i] - 0.005 else '~sym')
        print(f'{NAMES[i]:>10} {dfwd[i]:>+8.4f} {dbwd[i]:>+8.4f} {asym:>9}')

np.savez('fd_sensitivity_results.npz',
         names=np.array(NAMES), delta=DELTA,
         **{f'{k}_{m}': out[k][m] for k in out for m in ['S','dfwd','dbwd','grad','cl','cd']},
         **{f'{k}_s0': out[k]['s0'] for k in out})
         
print('\nsaved -> fd_sensitivity_results.npz')
