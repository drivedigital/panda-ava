"""Software renderer for skinned meshes: ortho/perspective, skeleton overlay."""
import math

import numpy as np
from PIL import Image

from meshutil import decimate_cluster


def look_at(eye, target, up=(0, 1, 0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= max(np.linalg.norm(f), 1e-9)
    s = np.cross(f, up)
    s /= max(np.linalg.norm(s), 1e-9)
    u = np.cross(s, f)
    M = np.eye(4)
    M[:3, :3] = np.array([s, u, -f])
    M[:3, 3] = -M[:3, :3] @ np.asarray(eye, float)
    return M


class Renderer:
    def __init__(self, P, N, T, I, albedo=None, decimate=None):
        if decimate:
            P, N, T, I = decimate_cluster(P, N, T, I, decimate)
        self.P, self.N, self.T, self.I = P.astype(np.float64), N.astype(np.float64), T.astype(np.float64), I.astype(np.int64)
        if albedo is None:
            self.albedo = np.tile(np.array([0.62, 0.64, 0.68], np.float32), (len(P), 1))
        else:
            self.albedo = albedo[P.shape[0]] if albedo.shape[0] != P.shape[0] else albedo
        self.vcol = None

    def render(self, azim=35, elev=8, W=460, H=620, center=(0, 0.0, 0), dist=3.0,
               fov=None, ortho_scale=None, bg=(0.13, 0.14, 0.17), bones=None,
               bone_color=(1.0, 0.85, 0.25), lines=None, points=None, alpha_bg=None):
        a, e = math.radians(azim), math.radians(elev)
        eye = np.array(center) + dist * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        V = look_at(eye, center)
        aspect = W / H
        if fov is None and ortho_scale is None:
            sx = (self.P @ V[:3, :3].T + V[:3, 3])[:, 0]
            sy = (self.P @ V[:3, :3].T + V[:3, 3])[:, 1]
            ortho_scale = max(sx.max() - sx.min(), (sy.max() - sy.min()) * aspect) * 0.58
        Vp = np.eye(4)
        if fov is None:
            Vp[0, 0] = 1 / (ortho_scale * aspect)
            Vp[1, 1] = 1 / ortho_scale
        else:
            Vp[0, 0] = 1 / math.tan(math.radians(fov) / 2) / aspect
            Vp[1, 1] = 1 / math.tan(math.radians(fov) / 2)
        Vp[2, 2] = 1
        Vp[2, 3] = -1

        img = np.ones((H, W, 3), np.float32) * np.array(bg, np.float32)
        zbuf = np.full((H, W), 1e9, np.float32)
        lights = np.array([[0.45, 0.75, 0.55], [-0.6, 0.2, -0.5], [0.1, -0.6, -0.8]])
        lights /= np.linalg.norm(lights, axis=1, keepdims=True)

        X = self.P @ V[:3, :3].T + V[:3, 3]
        C = X @ Vp[:3, :3].T + Vp[:3, 3]
        sx = C[:, 0] * W / 2 + W / 2
        sy = H / 2 - C[:, 1] * H / 2
        sz = C[:, 2]
        Vn = self.N @ V[:3, :3].T
        col = self.albedo[:len(self.P)] if self.vcol is None else self.vcol[:len(self.P)]

        tri = self.I
        tix = np.stack([sx[tri[:, 0]], sx[tri[:, 1]], sx[tri[:, 2]]])
        tiy = np.stack([sy[tri[:, 0]], sy[tri[:, 1]], sy[tri[:, 2]]])
        tiz = np.stack([sz[tri[:, 0]], sz[tri[:, 1]], sz[tri[:, 2]]])
        x0 = np.clip(np.floor(tix.min(0)).astype(int), 0, W - 1)
        x1 = np.clip(np.ceil(tix.max(0)).astype(int), 0, W - 1)
        y0 = np.clip(np.floor(tiy.min(0)).astype(int), 0, H - 1)
        y1 = np.clip(np.ceil(tiy.max(0)).astype(int), 0, H - 1)
        vis = (x1 > x0) & (y1 > y0) & (tiz.max(0) < -1e-4)
        e1x, e1y = sx[tri[:, 1]] - sx[tri[:, 0]], sy[tri[:, 1]] - sy[tri[:, 0]]
        e2x, e2y = sx[tri[:, 2]] - sx[tri[:, 0]], sy[tri[:, 2]] - sy[tri[:, 0]]
        det = e1x * e2y - e1y * e2x
        ids = np.nonzero(vis & (np.abs(det) > 1e-9))[0]
        order = ids[np.argsort(-tiz[1, ids])]  # painter's order, near last
        for i in order:
            a_, b_, c_ = tri[i]
            px = np.arange(x0[i], x1[i] + 1) + 0.5
            py = np.arange(y0[i], y1[i] + 1) + 0.5
            gx, gy = np.meshgrid(px, py)
            l1 = ((gx - sx[a_]) * e2y[i] - (gy - sy[a_]) * e2x[i]) / det[i]
            l2 = (e1x[i] * (gy - sy[a_]) - e1y[i] * (gx - sx[a_])) / det[i]
            l3 = 1 - l1 - l2
            ins = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not ins.any():
                continue
            zz = l1 * tiz[0, i] + l2 * tiz[1, i] + l3 * tiz[2, i]
            sl = (slice(y0[i], y1[i] + 1), slice(x0[i], x1[i] + 1))
            m = ins & (zz < zbuf[sl])
            if not m.any():
                continue
            nw = l1[..., None] * Vn[a_] + l2[..., None] * Vn[b_] + l3[..., None] * Vn[c_]
            nw /= np.maximum(np.linalg.norm(nw, axis=-1, keepdims=True), 1e-9)
            diff = np.maximum(0, nw @ lights.T) @ np.array([0.42, 0.24, 0.16]) + 0.16
            diff = diff * 0.75 + 0.25 + 0.25 * np.clip(-nw[..., 2], 0, 1)
            cc = col[a_][None, None, :] * diff[..., None]
            zbuf[sl][m] = zz[m]
            img[sl][m] = np.clip(cc, 0, 1)[m]

        if bones is not None:
            for (j0, j1) in bones:
                self._line(img, zbuf, V, Vp, W, H, bones_pts=j0, bones_pts2=j1, color=bone_color)
        if lines:
            for (p0, p1, c) in lines:
                self._line(img, zbuf, V, Vp, W, H, p0, p1, c)
        if points:
            for (p, c, r) in points:
                self._dot(img, zbuf, V, Vp, W, H, p, c, r)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def _proj(self, p, V, Vp, W, H):
        c = np.asarray(p, float) @ V[:3, :3].T + V[:3, 3]
        c = c @ Vp[:3, :3].T + Vp[:3, 3]
        return c[0] * W / 2 + W / 2, H / 2 - c[1] * H / 2, c[2]

    def _line(self, img, zbuf, V, Vp, W, H, p0, p1, color):
        x0, y0, z0 = self._proj(p0, V, Vp, W, H)
        x1, y1, z1 = self._proj(p1, V, Vp, W, H)
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        t = np.linspace(0, 1, max(n, 2))
        xs = np.clip((x0 + (x1 - x0) * t).astype(int), 0, W - 1)
        ys = np.clip((y0 + (y1 - y0) * t).astype(int), 0, H - 1)
        zs = z0 + (z1 - z0) * t
        ok = zs < zbuf[ys, xs]
        img[ys[ok], xs[ok]] = np.asarray(color, np.float32)

    def _dot(self, img, zbuf, V, Vp, W, H, p, color, r=3):
        x, y, z = self._proj(p, V, Vp, W, H)
        if not (0 <= x < W and 0 <= y < H):
            return
        rr = r
        y0, y1 = max(int(y) - rr, 0), min(int(y) + rr + 1, H)
        x0, x1 = max(int(x) - rr, 0), min(int(x) + rr + 1, W)
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = ((yy - y) ** 2 + (xx - x) ** 2) <= rr * rr
        if m.any():
            img[yy[m], xx[m]] = np.asarray(color, np.float32)


def sheet(tiles, cols=4):
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    return np.concatenate(rows, axis=0)


def save(img, path):
    Image.fromarray(img).save(path)
