"""Probe limb / torso / head landmarks from the tiger mesh cross-sections."""
import sys

import numpy as np

from analyze_mesh import load


def report(P, label, mask_fn, y0, y1, step=0.02, side=1):
    print(f"\n== {label} (side={side})")
    for y in np.arange(y0, y1, step):
        m = (P[:, 1] >= y) & (P[:, 1] < y + step) & mask_fn(P)
        if m.sum() < 8:
            continue
        q = P[m]
        w = q[:, 0]
        print(f"  y={y:6.2f} n={m.sum():6d} x[{w.min():6.3f},{w.max():6.3f}] ctr_x={w.mean():6.3f} "
              f"z[{q[:,2].min():6.3f},{q[:,2].max():6.3f}] ctr_z={q[:,2].mean():6.3f}")


if __name__ == "__main__":
    P, N, I, g = load(sys.argv[1])
    print(f"verts {len(P)}")
    report(P, "right leg", lambda P: P[:, 0] > 0.15, -0.95, 0.15)
    report(P, "left leg", lambda P: P[:, 0] < -0.15, -0.95, 0.15)
    report(P, "right arm", lambda P: P[:, 0] > 0.30, -0.30, 0.65)
    report(P, "head+neck", lambda P: P[:, 1] > 0.50, 0.50, 0.96, step=0.02)
    report(P, "torso core", lambda P: (np.abs(P[:, 0]) < 0.32), -0.20, 0.65, step=0.04)

    # ear tips: topmost vertices with |x| > 0.04
    top = P[:, 1] > 0.80
    q = P[top]
    for s in (-1, 1):
        m = q[:, 0] * s > 0.04
        if m.sum():
            e = q[m]
            k = np.argmax(e[:, 1])
            print(f"\near {'L' if s < 0 else 'R'}: tip={np.round(e[k], 3)} n={m.sum()} "
                  f"y_max={e[:,1].max():.3f} x_range[{e[:,0].min():.3f},{e[:,0].max():.3f}]")

    # snout: max z
    k = np.argmax(P[:, 2])
    print(f"\nsnout/front-most vertex: {np.round(P[k], 3)}  (z_max={P[:,2].max():.3f})")
    k = np.argmin(P[:, 2])
    print(f"back-most vertex:       {np.round(P[k], 3)}  (z_min={P[:,2].min():.3f})")
    # tail? vertices with |x|>0.3 and y>0.3 (behind shoulders)
    m = (P[:, 1] > 0.20) & (np.abs(P[:, 0]) > 0.30)
    print(f"\nregions with |x|>0.30 & y>0.20: n={m.sum()} x[{P[m,0].min():.3f},{P[m,0].max():.3f}] "
          f"y[{P[m,1].min():.3f},{P[m,1].max():.3f}] z[{P[m,2].min():.3f},{P[m,2].max():.3f}]")
