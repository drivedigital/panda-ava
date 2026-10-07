"""Mesh utilities: welding, vertex-cluster decimation, skinning, evaluation."""
import numpy as np


def weld(P, N, T, I, tol=1e-5):
    """Merge vertices sharing position+normal+uv (keeps attribute seams intact)."""
    key = np.round(np.hstack([P, N, T]) / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first)
    remap = np.empty(len(first), np.int64)
    remap[order] = np.arange(len(first))
    return P[first], N[first], T[first], remap[inverse], _remap(I, remap[inverse])


def _remap(I, inv):
    tri = inv[I]
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    return tri[keep]


def decimate_cluster(P, N, T, I, cell):
    """Vertex-clustering decimation: collapses vertices onto a grid."""
    key = np.floor(P / cell).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    flat = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    # representative = vertex closest to the cluster centroid
    cent = np.zeros((len(uniq), 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=P[:, k]) / counts
    rep = np.zeros(len(uniq), np.int64)
    d2 = ((P - cent[inverse]) ** 2).sum(1)
    order = np.lexsort((d2, inverse))
    firsts = order[np.concatenate([[True], inverse[order][1:] != inverse[order][:-1]])]
    rep[inverse[firsts]] = firsts
    Pt, Nt, Tt = P[rep], N[rep], T[rep]
    return Pt, Nt, Tt, _remap(I, inverse)


def skin(P, weights, mats):
    """Linear blend skinning. weights: (n,4) indices, (n,4) weights; mats: (4,4)"""
    out = np.zeros_like(P)
    for k in range(4):
        idx = weights[:, k].astype(np.int64)
        w = weights[:, k + 4][:, None]
        out += w * (mats[idx, :3, :3] @ P[:, :, None]).squeeze(-1) + w * mats[idx, :3, 3]
    return out


def build_weights(spread):
    """spread: (n,4) bone indices in slot order. Reorders so slots are sorted desc by weight."""
    idx, w = spread
    order = np.argsort(-w, axis=1)
    return np.hstack([np.take_along_axis(idx, order, 1), np.take_along_axis(w, order, 1)])


def bone_chain_matrices(joints, parents, local_trs):
    """Forward kinematics -> world matrices and local matrices for a joint list."""
    import numpy as np
    n = len(joints)
    world = [None] * n
    local = np.tile(np.eye(4), (n, 1, 1))
    by_parent = {}
    for i, p in enumerate(parents):
        by_parent.setdefault(p, []).append(i)
    order = []
    stack = list(by_parent.get(None, []))
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(by_parent.get(i, []))
    for i in order:
        T = np.eye(4)
        T[:3, :3] = local_trs[i]["R"]
        T[:3, 3] = local_trs[i]["t"]
        p = parents[i]
        local[i] = T
        world[i] = T if p is None else world[p] @ T
    return np.array(world), local


def normalise(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)
