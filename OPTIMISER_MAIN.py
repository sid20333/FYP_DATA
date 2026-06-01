import os, time, subprocess, tarfile, argparse
import shutil
import numpy as np
import geometry_check as geom

HOST = 'ss4922@spitfire.ae.ic.ac.uk'
ROOT = '/home/ss4922/campaign_runs'
mac = '/home/ss4922/final_macro_multi.java'
podkey = 'UJnDsrE8fyyD0pS+EK0Jxw'
W_CD = 4.0
LO, HI = geom.BOUNDS[:, 0], geom.BOUNDS[:, 1]
ND = geom.N_PARAMS

def dump_csv(c, p):
    np.savetxt(p, np.column_stack((c[:, 0], c[:, 1], np.zeros(len(c)))), fmt='%.6f,%.6f,%.6f,')

def run_ssh(cmd, tries=12):
    if cmd[0] in ('ssh', 'scp'):
        cmd = [cmd[0], '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '-i', os.path.expanduser('~/.ssh/id_ed25519')] + cmd[1:]
    for k in range(tries):
        try:
            return subprocess.run(cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(min(60, 5 * (1.5 ** k)))

def eval_batch(X, tag):
    n = len(X)
    valid = [geom.is_valid_geometry(x)[0] for x in X]
    b = f'opt_runs/{tag}'
    if os.path.exists(b): shutil.rmtree(b)
    os.makedirs(b, exist_ok=True)
    ids = []
    for i, x in enumerate(X, 1):
        if not valid[i - 1]: continue
        os.makedirs(f'{b}/d{i}', exist_ok=True)
        m, f1, f2 = geom.build_wing_geometry(geom.params_to_dict(x))
        dump_csv(m, f'{b}/d{i}/main_spline.csv')
        dump_csv(f1, f'{b}/d{i}/flap1_spline.csv')
        dump_csv(f2, f'{b}/d{i}/flap2_spline.csv')
        ids.append(i)
        
    if not ids: return np.full(n, -10.0), np.full(n, 5.0)

    tname = f'{tag.replace("/", "_")}.tar'
    with tarfile.open(tname, 'w') as t: t.add(b, arcname=tag)
    
    rpath = f'{ROOT}/{tag}'
    run_ssh(['ssh', HOST, f'rm -rf {rpath} && mkdir -p {rpath}'])
    run_ssh(['scp', tname, f'{HOST}:{ROOT}/'])
    run_ssh(['ssh', HOST, f'cd {ROOT} && tar xf {tname} && rm {tname}'])

    slrm_script = f"""#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:13:00
source /etc/profile.d/modules.sh 2>/dev/null
module load star-ccm+/20.04.007-R8
unset CDLMD_LICENSE_FILE
export LM_LICENSE_FILE=""
cd $DDIR
timeout -k 30 600 starccm+ -np 8 -new -batch {mac} -podkey {podkey} -power
if ! grep -qE '^(final|capped),' $DDIR/force_report.csv 2>/dev/null; then
  vals=$(awk '$1 ~ /^[0-9]+$/ && NF>=8 && $(NF-1)+0>0.5 && $(NF-1)+0<5 && $NF+0>0.03 && $NF+0<0.4 {{cl=$(NF-1); cd=$NF}} END{{if(cl!="") print cl, cd}}' $DDIR/out.log 2>/dev/null)
  if [ -n "$vals" ]; then
    printf 'Iteration,C_L,C_D\\ncapped,%s\\n' "$(echo $vals | tr " " ,)" > $DDIR/force_report.csv
  fi
fi
rm -f $DDIR/*.sim $DDIR/*.sim~ 2>/dev/null
touch $DDIR/DONE
"""
    open('one.slurm', 'w').write(slrm_script)
    run_ssh(['scp', 'one.slurm', f'{HOST}:{rpath}/one.slurm'])
    
    id_str = ' '.join(str(i) for i in ids)
    run_ssh(['ssh', HOST, f'cd {rpath} && for i in {id_str}; do sbatch --job-name={tag.replace("/", "_")}_d$i --output=d$i/out.log --export=ALL,DDIR=$PWD/d$i one.slurm >/dev/null; done'])

    # polling loop added
    while True:
        try:
            d = run_ssh(['ssh', HOST, f'(ls {rpath}/d*/DONE 2>/dev/null | wc -l) || echo 0']).stdout.strip()
            if int(d or 0) >= len(ids): break
        except: pass
        time.sleep(20)

    c_l = np.full(n, -10.0)
    c_d = np.full(n, 5.0)

    try:
        out = run_ssh(['ssh', HOST, f'for i in {id_str}; do if [ -f {rpath}/d$i/force_report.csv ]; then echo "DESIGN$i:"; tail -1 {rpath}/d$i/force_report.csv; fi; done']).stdout
    except: out = ''
    
    curr = None
    for l in out.splitlines():
        l = l.strip()
        if l.startswith('DESIGN'):
            curr = int(l[6:-1])
        elif curr is not None and ('final,' in l or 'capped,' in l):
            pts = l.split(',')
            c_l[curr - 1] = float(pts[1])
            c_d[curr - 1] = float(pts[2])
            curr = None
    return c_l, c_d

def get_score(cl, cd):
    return cl - W_CD * cd

def load_seeds(phase):
    x = np.loadtxt(f'{phase}_X.csv', delimiter=',')
    out = subprocess.run(['ssh', HOST, f'for i in $(seq 1 {len(x)}); do echo -n "$i,"; tail -1 {ROOT}/{phase}/d$i/force_report.csv 2>/dev/null; echo; done'], capture_output=True, text=True).stdout
    c_l, c_d = np.full(len(x), -10.0), np.full(len(x), 5.0)
    for l in out.splitlines():
        p = l.split(',')
        if len(p) >= 4 and p[1] == 'final':
            idx = int(p[0]) - 1
            c_l[idx], c_d[idx] = float(p[2]), float(p[3])
    s = get_score(c_l, c_d)
    o = np.argsort(-s)
    return x[o], s[o]

def rand_valid(n):
    o = []
    while len(o) < n:
        r = LO + np.random.rand(ND) * (HI - LO)
        if geom.is_valid_geometry(r)[0]: o.append(r)
    return np.array(o)

def is_stuck(hist):
    if len(hist) < 8: return False
    return all(b <= hist[-8] + 1e-6 for b in hist[-7:])

def do_ga(tag, pool_p, pop=16):
    px, ps = pool_p[0].copy(), pool_p[1].copy()
    print(f'starting ga on {tag}...')
    hist = [ps.max()]
    for g in range(1, 13):
        def trn(n):
            c = []
            for _ in range(n):
                ix = np.random.choice(len(px), size=min(4, len(px)), replace=False)
                c.append(ix[np.argmax(ps[ix])])
            return px[c]
        kds = []
        for a, b in zip(trn(pop-2), trn(pop-2)):
            al = np.random.rand(ND)
            c = al * a + (1 - al) * b + np.random.normal(0, 0.10, ND) * (HI - LO)
            kds.append(np.clip(c, LO, HI))
        xgen = np.array(kds + rand_valid(2).tolist())
        cl, cd = eval_batch(xgen, f'{tag}/g{g}')
        sg = get_score(cl, cd)
        px = np.vstack([px, xgen])
        ps = np.concatenate([ps, sg])
        hist.append(ps.max())
        print(f'{tag} g{g} max: {sg.max():.3f} global: {ps.max():.3f}')
        if is_stuck(hist):
            print('stuck, breaking ga')
            break
    return ps.max()

def do_pso(tag, seed_x, pop=8):
    x = seed_x[:pop].copy()
    v = np.zeros_like(x)
    cl, cd = eval_batch(x, f'{tag}/g0')
    s = get_score(cl, cd)
    px, ps, gx = x.copy(), s.copy(), x[np.argmax(s)].copy()
    hist = [ps.max()]
    print(f'pso {tag} init max {ps.max():.3f}')
    for g in range(1, 13):
        r1, r2 = np.random.rand(pop, ND), np.random.rand(pop, ND)
        v = 0.7 * v + 1.5 * r1 * (px - x) + 1.5 * r2 * (gx - x)
        x = np.clip(x + v, LO, HI)
        cl, cd = eval_batch(x, f'{tag}/g{g}')
        s = get_score(cl, cd)
        imp = s > ps
        px[imp], ps[imp] = x[imp], s[imp]
        gx = px[np.argmax(ps)].copy()
        hist.append(ps.max())
        print(f'{tag} g{g} pso max {ps.max():.3f}')
        if is_stuck(hist):
            print('stuck, stopping pso')
            break
    return ps.max()

def do_cma(tag, x0_real):
    import cma
    x0 = np.clip((x0_real - LO) / (HI - LO), 0, 1)
    es = cma.CMAEvolutionStrategy(x0, 0.3, {'bounds': [0, 1], 'popsize': 12, 'maxfevals': 144, 'verbose': -9})
    hist = []
    g = 0
    while not es.stop() and g < 12:
        sols = es.ask()
        x = LO + np.array(sols) * (HI - LO)
        cl, cd = eval_batch(x, f'{tag}/g{g}')
        es.tell(sols, [-float(v) for v in get_score(cl, cd)])
        hist.append(-es.result.fbest)
        print(f'{tag} g{g} cma max {-es.result.fbest:.3f}')
        if is_stuck(hist): break
        g += 1
    return -es.result.fbest

def get_clusters(lx, ls, sx, ss):
    from sklearn.cluster import KMeans
    ax = np.vstack([lx, sx])
    as_ = np.concatenate([ls, ss])
    val = as_ > -5.0
    xv, sv = ax[val], as_[val]
    xn = (xv - LO) / (HI - LO)
    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(xn)
    clsts = []
    for c in range(5):
        i = np.where(km.labels_ == c)[0]
        o = i[np.argsort(-sv[i])]
        clsts.append((xv[o], sv[o]))
    clsts.sort(key=lambda c: -c[1][0])
    return clsts

def run_drivers(lx, ls, sx, ss, opts):
    c = get_clusters(lx, ls, sx, ss)
    for r in range(5):
        xc, sc = c[r]
        if 'ga' in opts:
            print(f'=== ga r{r} ===')
            do_ga(f'ga/r{r}', (xc, sc))
        if 'pso' in opts:
            print(f'=== pso r{r} ===')
            sds = xc[:8]
            while len(sds) < 8: sds = np.vstack([sds, xc[:8 - len(sds)]])
            do_pso(f'pso/r{r}', sds)
        if 'cmaes' in opts:
            print(f'=== cmaes r{r} ===')
            do_cma(f'cmaes/r{r}', xc[0])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--opt', default='ga,pso,cmaes')
    a = p.parse_args()
    lx, ls = load_seeds('lhs')
    sx, ss = load_seeds('sobol')
    print(f'seeds parsed. lhs max={ls[0]:.3f}, sobol max={ss[0]:.3f}')
    run_drivers(lx, ls, sx, ss, a.opt.split(','))
