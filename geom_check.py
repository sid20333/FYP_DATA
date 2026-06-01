
import numpy as np
from matplotlib.path import Path


CHORD_MAIN  = 0.5
CHORD_FLAP1 = 0.3
CHORD_FLAP2 = 0.2

PARAMS = {
    'main_m':     {'low': 0.04,  'high': 0.09},
    'main_p':     {'low': 0.20,  'high': 0.50},
    'main_t':     {'low': 0.12,  'high': 0.16},
    'main_aoa':   {'low': 0.0,   'high': 5.0},
    'flap1_m':    {'low': 0.04,  'high': 0.09},
    'flap1_p':    {'low': 0.30,  'high': 0.60},
    'flap1_t':    {'low': 0.10,  'high': 0.14},
    'flap1_aoa':  {'low': 10.0,  'high': 27.5},
    'flap2_m':    {'low': 0.04,  'high': 0.09},
    'flap2_p':    {'low': 0.30,  'high': 0.60},
    'flap2_t':    {'low': 0.10,  'high': 0.13},
    'flap2_aoa':  {'low': 27.5,  'high': 45.0},


    'gap1':       {'low': 0.0087, 'high': 0.0261},
    'gap2':       {'low': 0.0087, 'high': 0.0261},


    'overlap1':   {'low': -0.015, 'high': 0.0261},
    'overlap2':   {'low': -0.015, 'high': 0.0261},
}
PARAM_NAMES = list(PARAMS.keys())
N_PARAMS    = len(PARAM_NAMES)
BOUNDS      = np.array([[PARAMS[p]['low'], PARAMS[p]['high']] for p in PARAM_NAMES])

FIA_MAX_HEIGHT = 0.4174


def params_to_dict(x):
    return {name: float(val) for name, val in zip(PARAM_NAMES, x)}


def naca4_airfoil(m, p, t, chord=1.0, n_points=100):
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    if m < 1e-6 or p < 0.01:
        yc = np.zeros_like(x); theta = np.zeros_like(x)
    else:
        yc = np.where(x <= p, (m/p**2)*(2*p*x - x**2),
                      (m/(1-p)**2)*((1-2*p) + 2*p*x - x**2))
        dyc = np.where(x <= p, (2*m/p**2)*(p-x), (2*m/(1-p)**2)*(p-x))
        theta = np.arctan(dyc)
    xu = x - yt*np.sin(theta); yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta); yl = yc - yt*np.cos(theta)
    x_all = np.concatenate([xu[::-1], xl[1:]]) * chord
    y_all = np.concatenate([yu[::-1], yl[1:]]) * chord
    return np.column_stack([x_all, -y_all])


def rotate_profile(coords, angle_deg):
    le = coords[np.argmin(coords[:, 0])].copy()
    a = np.radians(angle_deg); c, s = np.cos(a), np.sin(a)
    sh = coords - le
    return np.column_stack([sh[:, 0]*c - sh[:, 1]*s, sh[:, 0]*s + sh[:, 1]*c]) + le


def trailing_edge(coords):
    return coords[np.argmax(coords[:, 0])].copy()


def position_element(coords, upstream, overlap, gap):
    """Place a downstream element relative to the whole upstream element.
      overlap : horizontal stagger of the downstream LE behind the upstream TE
                (LE_x = upstream_TE_x - overlap; +overlap = tucked upstream).
      gap     : vertical SLOT CLEARANCE = lowermost point of this (downstream)
                element minus uppermost point of the upstream element. gap >= 0
                guarantees no intersection (clean cascade slot)."""
    out = coords.copy()
    le_x = out[np.argmin(out[:, 0]), 0]
    te_x_up = upstream[np.argmax(upstream[:, 0]), 0]
    out[:, 0] += (te_x_up - overlap) - le_x
    out[:, 1] += (upstream[:, 1].max() + gap) - out[:, 1].min()
    return out


def build_wing_geometry(p):
    main = rotate_profile(naca4_airfoil(p['main_m'], p['main_p'], p['main_t'], CHORD_MAIN), p['main_aoa'])
    f1 = rotate_profile(naca4_airfoil(p['flap1_m'], p['flap1_p'], p['flap1_t'], CHORD_FLAP1), p['main_aoa'] + p['flap1_aoa'])
    f1 = position_element(f1, main, p['overlap1'], p['gap1'])
    f2 = rotate_profile(naca4_airfoil(p['flap2_m'], p['flap2_p'], p['flap2_t'], CHORD_FLAP2), p['main_aoa'] + p['flap2_aoa'])
    f2 = position_element(f2, f1, p['overlap2'], p['gap2'])
    return main, f1, f2


def elements_overlap(c1, c2):
    return Path(c1).contains_points(c2).any() or Path(c2).contains_points(c1).any()


def is_valid_geometry(x, check_height=True):
    """Returns (valid: bool, reason: str). Builds wing from param vector x."""
    p = params_to_dict(x)
    main, f1, f2 = build_wing_geometry(p)
    if elements_overlap(main, f1): return False, 'main-flap1 intersect'
    if elements_overlap(f1, f2):   return False, 'flap1-flap2 intersect'
    if elements_overlap(main, f2): return False, 'main-flap2 intersect'
    if check_height:
        ys = np.concatenate([main[:, 1], f1[:, 1], f2[:, 1]])
        h = ys.max() - ys.min()
        if h > FIA_MAX_HEIGHT:
            return False, f'height {h*1000:.0f}mm > FIA {FIA_MAX_HEIGHT*1000:.0f}mm'
    return True, 'ok'

#declaration: test written by claude
if __name__ == '__main__':
    from scipy.stats import qmc
    N = 20000
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    X = lo + qmc.LatinHypercube(d=N_PARAMS, seed=1).random(N) * (hi - lo)

    reasons = {}
    heights = []
    valid = 0
    for x in X:
        ok, why = is_valid_geometry(x)
        if ok:
            valid += 1
        reasons[why] = reasons.get(why, 0) + 1
        p = params_to_dict(x)
        m, f1, f2 = build_wing_geometry(p)
        ys = np.concatenate([m[:, 1], f1[:, 1], f2[:, 1]])
        heights.append((ys.max() - ys.min()) * 1000)

    print(f"Feasibility over {N} LHS samples (total chord 1.0 m):")
    print(f"  VALID: {valid}/{N} = {100*valid/N:.1f}%")
    print("  breakdown:")
    for why, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {why:28s}: {c:6d}  ({100*c/N:.1f}%)")
    heights = np.array(heights)
    print(f"  profile height [mm]: min {heights.min():.0f}, median {np.median(heights):.0f}, "
          f"max {heights.max():.0f}  (FIA {FIA_MAX_HEIGHT*1000:.0f})")


    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valids = [x for x in X if is_valid_geometry(x)[0]]
    rng2 = np.random.default_rng(7)
    pick = [valids[i] for i in rng2.choice(len(valids), size=6, replace=False)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, x in zip(axes.ravel(), pick):
        m, f1, f2 = build_wing_geometry(params_to_dict(x))
        for c, col, lab in [(m, '#1f77b4', 'main'), (f1, '#ff7f0e', 'flap1'), (f2, '#2ca02c', 'flap2')]:
            ax.fill(c[:, 0], c[:, 1], alpha=0.35, color=col)
            ax.plot(c[:, 0], c[:, 1], color=col, lw=1.2)
        ax.set_aspect('equal'); ax.grid(alpha=0.25)
    fig.suptitle('Sample valid geometries (new gap/overlap convention, 1.0 m chord)')
    plt.tight_layout()
    plt.savefig('/Users/shrey/Coursework/FYP/geometry_check_samples.png', dpi=140)
    print('  saved geometry_check_samples.png')
