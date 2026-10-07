"""Verify the generated rig: rest-pose fidelity, skeleton overlay, weight maps."""
import json
import sys

import numpy as np
from PIL import Image

from glblib import read_glb
from meshutil import decimate_cluster, skin
from render import Renderer, save, sheet


def load_rig(path="../out/tiger_rig.glb"):
    g = read_glb(path)
    prim = g.js["meshes"][0]["primitives"][0]
    P = np.asarray(g.accessor(prim["attributes"]["POSITION"]), np.float64)
    N = np.asarray(g.accessor(prim["attributes"]["NORMAL"]), np.float64)
    T = np.asarray(g.accessor(prim["attributes"]["TEXCOORD_0"]), np.float64)
    I = g.indices(prim).reshape(-1, 3).astype(np.int64)
    J = np.asarray(g.accessor(prim["attributes"]["JOINTS_0"]), np.int64)
    W = np.asarray(g.accessor(prim["attributes"]["WEIGHTS_0"]), np.float64)
    ibm = np.asarray(g.accessor(g.js["skins"][0]["inverseBindMatrices"]), np.float64).reshape(-1, 4, 4)
    joints = g.js["skins"][0]["joints"]
    names = [g.js["nodes"][j]["name"] for j in joints]
    idx_of = {n: i for i, n in enumerate(names)}
    return P, N, T, I, J, W, ibm, names, idx_of


def node_local(g, joints):
    loc = []
    for j in joints:
        n = g.js["nodes"][j]
        loc.append(n)
    return loc


def rest_local(g, joints):
    """World-space rest matrices of the joint nodes (traverse the node hierarchy)."""
    def q2m(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    parent = {}
    for i, n in enumerate(g.js["nodes"]):
        for c in n.get("children", []):
            parent[c] = i
    local = {}
    for j in joints:
        n = g.js["nodes"][j]
        M = np.eye(4)
        M[:3, :3] = q2m(n.get("rotation", [0, 0, 0, 1]))
        M[:3, 3] = n.get("translation", [0, 0, 0])
        local[j] = M
    world = {}
    for j in joints:
        chain = []
        k = j
        while k is not None and k not in world:
            chain.append(k)
            k = parent.get(k)
        M = world.get(k, np.eye(4)) if k is not None else np.eye(4)
        for node in reversed(chain):
            M = M @ local[node]
            world[node] = M
    return np.array([world[j] for j in joints])


def skin_with(P, J, W, mats, ibm, scale=1.0):
    """LBS: p' = sum_j w_j * (mats_j * ibm_j * p)."""
    out = np.zeros_like(P)
    n = len(P)
    for k in range(4):
        jj = J[:, k]
        ww = W[:, k]
        M = mats[jj] @ ibm[jj]
        out += ww[:, None] * (M[:, :3, :3] @ P[:, :, None] + M[:, :3, 3:4]).squeeze(-1)
    return out


def main():
    P, N, T, I, J, W, ibm, names, idx_of = load_rig()
    g = read_glb("../out/tiger_rig.glb")
    mats = rest_local(g, g.js["skins"][0]["joints"])
    skinned = skin_with(P, J, W, mats, ibm)

    # weld to compare like-for-like
    print("rest-pose displacement stats (skinned vs bind mesh):")
    d = np.linalg.norm(skinned - P, axis=1)
    for q in (50, 90, 99, 99.9, 100):
        print(f"  p{q}: {np.percentile(d, q):.5f}")
    print(f"  mean: {d.mean():.5f}  >1mm: {(d > 0.001).mean()*100:.2f}%  >5mm: {(d > 0.005).mean()*100:.3f}%")

    ref = Renderer(P, N, T, I, decimate=0.005)
    got = Renderer(skinned, N, T, I, decimate=0.005)
    tiles = []
    for a, e in [(0, 3), (90, 3)]:
        tiles.append(ref.render(azim=a, elev=e, W=300, H=460))
        tiles.append(got.render(azim=a, elev=e, W=300, H=460))
    save(sheet(tiles, 4), "../out/qa_restpose.png")

    # skeleton overlay
    meta = json.load(open("../out/skeleton.json"))
    bone_lines, points = [], []
    for b in meta["bones"]:
        if b["tail"] is None:
            continue
        bone_lines.append((b["head"], b["tail"], (0.95, 0.85, 0.25)))
        points.append((b["head"], (1.0, 0.3, 0.3), 3))
    tiles = [got.render(azim=a, elev=e, W=320, H=480, lines=bone_lines, points=points)
             for a, e in [(0, 3), (90, 3), (35, 15)]]
    save(sheet(tiles, 3), "../out/qa_skeleton.png")

    # weight heat maps for tricky bones
    def heatmap(bone):
        b = idx_of[bone]
        w = np.where(J == b, W, 0).sum(1)
        r = Renderer(P, N, T, I, decimate=0.006)
        r.vcol = np.stack([np.clip(w * 2.2, 0, 1), np.clip(1 - abs(w - 0.5) * 2, 0, 1), np.clip(1 - w * 2.2, 0, 1)], 1)
        return [r.render(azim=a, elev=e, W=260, H=380) for a, e in [(0, 3), (90, 3)]]

    tiles = []
    for bn in ("LeftUpLeg", "LeftLeg", "LeftArm", "LeftForeArm", "LeftHand", "Spine2"):
        tiles += heatmap(bn)
    save(sheet(tiles, 6), "../out/qa_weights.png")
    print("wrote QA images")


if __name__ == "__main__":
    main()