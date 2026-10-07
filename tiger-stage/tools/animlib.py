"""FK sampler for animation-only GLBs (numpy). Reads node hierarchy + TRS tracks,
produces world-space rotations/positions per node per frame on a uniform time grid."""
import numpy as np

from glblib import read_glb


# ---------------- quaternion helpers (x, y, z, w) ----------------
def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def qconj(q):
    out = np.array(q, copy=True)
    out[..., :3] *= -1
    return out


def qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qrot(q, v):
    """rotate vec3 by quat"""
    u = q[..., :3]
    w = q[..., 3:4]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def nlerp(q0, q1, t):
    d = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    return qnorm(q0 * (1 - t) + q1 * t)


def shortest_arc(a, b):
    """quat rotating unit vec a to unit vec b"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -0.999999:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + d])
    return q / np.linalg.norm(q)


# ---------------- GLB animation model ----------------
class AnimGlb:
    def __init__(self, path, fps=30):
        g = read_glb(path)
        self.gltf = g
        js = g.js
        self.nodes = js["nodes"]
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.name2idx = {n.get("name"): i for i, n in enumerate(self.nodes)}
        self.rest_t = {}
        self.rest_r = {}
        self.rest_s = {}
        for i, n in enumerate(self.nodes):
            self.rest_t[i] = np.array(n.get("translation", [0, 0, 0]), float)
            self.rest_r[i] = np.array(n.get("rotation", [0, 0, 0, 1]), float)
            self.rest_s[i] = np.array(n.get("scale", [1, 1, 1]), float)
        anims = js.get("animations", [])
        self.anim = anims[0] if anims else None
        self.fps = fps
        self._load_tracks()

    def _load_tracks(self):
        self.tracks = {}  # node -> {"t":(times,vals), "r":..., "s":...}
        t_max = 0.0
        t_min = 1e9
        if self.anim:
            for ch in self.anim["channels"]:
                smp = self.anim["samplers"][ch["sampler"]]
                times = np.asarray(self.gltf.accessor(smp["input"]), float).reshape(-1)
                vals = np.asarray(self.gltf.accessor(smp["output"]), float)
                node = ch["target"]["node"]
                path = ch["target"]["path"][0]  # t / r / s
                self.tracks.setdefault(node, {})[path] = (times, vals)
                t_max = max(t_max, times[-1])
                t_min = min(t_min, times[0])
        self.t_min = 0.0 if not self.tracks else t_min
        self.t_max = 1.0 if not self.tracks else t_max

    def duration(self):
        return self.t_max - self.t_min

    def _sample_track(self, times, vals, t, is_quat):
        i = np.searchsorted(times, t, side="right") - 1
        i = np.clip(i, 0, len(times) - 2) if len(times) > 1 else 0
        if len(times) == 1:
            return vals[0].astype(float)
        t0, t1 = times[i], times[i + 1]
        f = 0.0 if t1 <= t0 else np.clip((t - t0) / (t1 - t0), 0, 1)
        v0, v1 = vals[i].astype(float), vals[i + 1].astype(float)
        if is_quat:
            return nlerp(v0, v1, f)
        return v0 * (1 - f) + v1 * f

    def local_trs(self, node, t):
        tr = self.tracks.get(node, {})
        if "t" in tr:
            lt = self._sample_track(*tr["t"], t, False)
        else:
            lt = self.rest_t[node]
        if "r" in tr:
            lr = self._sample_track(*tr["r"], t, True)
        else:
            lr = self.rest_r[node]
        if "s" in tr:
            ls = self._sample_track(*tr["s"], t, False)
        else:
            ls = self.rest_s[node]
        return lt, lr, ls

    def world_at(self, t):
        """returns (world_quat[node], world_pos[node]) for all nodes at time t."""
        order = []
        seen = set()
        def visit(i):
            if i in seen:
                return
            if i in self.parent:
                visit(self.parent[i])
            seen.add(i)
            order.append(i)
        for i in range(len(self.nodes)):
            visit(i)
        wq, wp, ws = {}, {}, {}
        for i in order:
            lt, lr, ls = self.local_trs(i, t)
            p = self.parent.get(i)
            if p is None:
                wq[i] = qnorm(lr)
                wp[i] = np.array(lt, float)
                ws[i] = np.array(ls, float)
            else:
                wq[i] = qnorm(qmul(wq[p], lr))
                wp[i] = wp[p] + qrot(wq[p], lt * ws[p])
                ws[i] = ws[p] * ls
        return wq, wp

    def rest_world(self):
        """world quat/pos at rest (static TRS, ignoring animation)."""
        order = []
        seen = set()
        def visit(i):
            if i in seen:
                return
            if i in self.parent:
                visit(self.parent[i])
            seen.add(i)
            order.append(i)
        for i in range(len(self.nodes)):
            visit(i)
        wq, wp = {}, {}
        for i in order:
            p = self.parent.get(i)
            if p is None:
                wq[i] = qnorm(self.rest_r[i])
                wp[i] = np.array(self.rest_t[i], float)
            else:
                wq[i] = qnorm(qmul(wq[p], self.rest_r[i]))
                wp[i] = wp[p] + qrot(wq[p], self.rest_t[i])
        return wq, wp
