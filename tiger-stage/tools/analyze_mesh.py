"""Anatomical analysis of the tiger mesh: cross-sections and landmarks."""
import sys

import numpy as np

from glblib import read_glb


def load(path):
    g = read_glb(path)
    prim = g.js["meshes"][0]["primitives"][0]
    P = np.asarray(g.accessor(prim["attributes"]["POSITION"]), np.float64)
    N = np.asarray(g.accessor(prim["attributes"]["NORMAL"]), np.float64)
    I = g.indices(prim).reshape(-1, 3)
    return P, N, I, g


def cross_sections(P, step=0.05, pct=(2, 98)):
    lo, hi = P.min(0), P.max(0)
    print(f"bbox  x[{lo[0]:.3f},{hi[0]:.3f}]  y[{lo[1]:.3f},{hi[1]:.3f}]  z[{lo[2]:.3f},{hi[2]:.3f}]")
    print(f"size  {hi[0]-lo[0]:.3f} x {hi[1]-lo[1]:.3f} x {hi[2]-lo[2]:.3f}")
    ys = np.arange(lo[1], hi[1] + step, step)
    print(f"\n{'y':>7} {'n':>7} {'x_lo':>7} {'x_hi':>7} {'z_lo':>7} {'z_hi':>7}   clusters(x)")
    for y in ys:
        m = (P[:, 1] >= y) & (P[:, 1] < y + step)
        if m.sum() < 5:
            continue
        q = P[m]
        xs = np.sort(q[:, 0])
        xlo, xhi = np.percentile(xs, pct[0]), np.percentile(xs, pct[1])
        # cluster x values into groups separated by > 0.06
        groups = [[xs[0]]]
        for v in xs[1:]:
            if v - groups[-1][-1] > 0.06:
                groups.append([v])
            else:
                groups[-1].append(v)
        cl = " ".join(f"[{g[0]:.2f}..{g[-1]:.2f}]({len(g)})" for g in groups if len(g) > 8)
        print(f"{y:7.2f} {m.sum():7d} {xlo:7.3f} {xhi:7.3f} {q[:,2].min():7.3f} {q[:,2].max():7.3f}   {cl}")


def z_sections(P, step=0.05):
    lo, hi = P.min(0), P.max(0)
    print(f"\n{'z':>7} {'n':>7} {'x_lo':>7} {'x_hi':>7} {'y_lo':>7} {'y_hi':>7}")
    for z in np.arange(lo[2], hi[2] + step, step):
        m = (P[:, 2] >= z) & (P[:, 2] < z + step)
        if m.sum() < 5:
            continue
        q = P[m]
        print(f"{z:7.2f} {m.sum():7d} {q[:,0].min():7.3f} {q[:,0].max():7.3f} {q[:,1].min():7.3f} {q[:,1].max():7.3f}")


if __name__ == "__main__":
    P, N, I, g = load(sys.argv[1])
    print(f"verts {len(P)} tris {len(I)}")
    cross_sections(P, float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
    z_sections(P)
