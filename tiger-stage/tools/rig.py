"""Build a humanoid skeleton for the tiger mesh, auto-skin it, export GLB + skeleton JSON.

Skeleton uses Mixamo-compatible bone names so the FBX martial-arts clips can be
retargeted 1:1 (see retarget.py).  Extra `spring_*` chains carry secondary motion
(ears, apron panels, sleeve cuffs) for the runtime spring-bone solver.
"""
import io
import json
import sys

import numpy as np
import scipy.sparse as sp
from PIL import Image

from glblib import Gltf, read_glb

SRC = "../../Characters/tiger_Meshy_AI_Crimson_Tiger_Guardia_1002022828_texture.glb"

# ---------------------------------------------------------------------------
# Skeleton definition.  Values are measured from the mesh (see probe.py output):
# ground y=-0.952, crown y=+0.951, character left = +X (faces +Z).
# ---------------------------------------------------------------------------
def build_bone_table():
    B = []

    def add(name, parent, head, tail=None, spring=None):
        B.append({"name": name, "parent": parent, "head": list(map(float, head)),
                  "tail": None if tail is None else list(map(float, tail)), "spring": spring})
        return B[-1]

    def mirror(pts):
        return [[-x, y, z] for x, y, z in pts]

    spine = [("Hips", None, (0.0, -0.035, 0.0), (0.0, 0.10, -0.005)),
             ("Spine", "Hips", (0.0, 0.10, -0.005), (0.0, 0.225, -0.005)),
             ("Spine1", "Spine", (0.0, 0.225, -0.005), (0.0, 0.35, -0.005)),
             ("Spine2", "Spine1", (0.0, 0.35, -0.005), (0.0, 0.45, -0.015)),
             ("Neck", "Spine2", (0.0, 0.45, -0.015), (0.0, 0.565, -0.010)),
             ("Neck1", "Neck", (0.0, 0.565, -0.010), (0.0, 0.68, 0.005)),
             ("Head", "Neck1", (0.0, 0.68, 0.005), (0.0, 0.93, -0.010)),
             ("HeadEnd", "Head", (0.0, 0.93, -0.010), (0.0, 0.945, -0.010)),
             ("Jaw", "Head", (0.0, 0.715, 0.055), (0.0, 0.665, 0.115))]
    for n, p, h, t in spine:
        add(n, p, h, t)

    # arms (left = +X), fingers modelled from the clenched-fist geometry
    def arm(side):
        s = 1 if side == "Left" else -1
        X = lambda v: (s * v[0], v[1], v[2])
        add(f"{side}Shoulder", "Spine2", X((0.085, 0.44, -0.01)), X((0.20, 0.465, -0.012)))
        add(f"{side}Arm", f"{side}Shoulder", X((0.34, 0.465, -0.015)), X((0.475, 0.27, -0.05)))
        add(f"{side}ForeArm", f"{side}Arm", X((0.475, 0.27, -0.05)), X((0.605, 0.055, -0.06)))
        add(f"{side}Hand", f"{side}ForeArm", X((0.605, 0.055, -0.06)), X((0.695, 0.005, -0.03)))
        fingers = {
            # finger: (mcp, pip, dip, tip)
            "Thumb":  ((0.660, -0.010, 0.050), (0.700, 0.008, 0.044), (0.718, 0.020, 0.020)),
            "Index":  ((0.680, 0.010, 0.016), (0.712, 0.014, 0.015), (0.734, -0.024, 0.013)),
            "Middle": ((0.678, 0.006, -0.022), (0.710, 0.010, -0.024), (0.732, -0.028, -0.026)),
            "Ring":   ((0.675, 0.002, -0.060), (0.707, 0.005, -0.063), (0.729, -0.032, -0.066)),
            "Pinky":  ((0.670, -0.003, -0.096), (0.701, 0.000, -0.100), (0.721, -0.036, -0.104)),
        }
        for fname, pts in fingers.items():
            prev = f"{side}Hand"
            for i, p in enumerate(pts, start=1):
                nm = f"{side}Hand{fname}{i}"
                add(nm, prev, X(p), None)
                prev = nm
            # terminal node so the chain has a tip
            tipx = [pts[2][0] - 0.012, pts[2][1] - 0.030, pts[2][2]]
            add(f"{prev}End", prev, X(tipx), None)

    arm("Left")
    arm("Right")

    def leg(side):
        s = 1 if side == "Left" else -1
        X = lambda v: (s * v[0], v[1], v[2])
        add(f"{side}UpLeg", "Hips", X((0.145, -0.070, 0.0)), X((0.205, -0.44, -0.02)))
        add(f"{side}Leg", f"{side}UpLeg", X((0.205, -0.44, -0.02)), X((0.305, -0.82, -0.035)))
        add(f"{side}Foot", f"{side}Leg", X((0.305, -0.82, -0.035)), X((0.325, -0.885, 0.115)))
        add(f"{side}ToeBase", f"{side}Foot", X((0.325, -0.885, 0.115)), X((0.330, -0.905, 0.180)))
        add(f"{side}ToeEnd", f"{side}ToeBase", X((0.330, -0.905, 0.180)), None)

    leg("Left")
    leg("Right")

    # ---- spring chains (secondary motion) -------------------------------
    for side in ("Left", "Right"):
        s = 1 if side == "Left" else -1
        X = lambda v: (s * v[0], v[1], v[2])
        add(f"spring_{side}Ear_1", "Head", X((0.070, 0.880, -0.012)), X((0.095, 0.915, -0.008)),
            {"chain": f"{side}Ear", "stiffness": 0.62, "damping": 0.16, "drag": 0.12})
        add(f"spring_{side}Ear_2", f"spring_{side}Ear_1", X((0.095, 0.915, -0.008)),
            X((0.113, 0.951, -0.004)), {"chain": f"{side}Ear"})
        add(f"spring_{side}Ear_3", f"spring_{side}Ear_2", X((0.113, 0.951, -0.004)),
            X((0.121, 0.966, -0.002)), {"chain": f"{side}Ear"})
    # centre-front apron / tassel panel
    add("spring_ApronF_1", "Hips", (0.0, -0.10, 0.190), (0.0, -0.28, 0.205),
        {"chain": "ApronFront", "stiffness": 0.48, "damping": 0.22, "drag": 0.2})
    add("spring_ApronF_2", "spring_ApronF_1", (0.0, -0.28, 0.205), (0.0, -0.46, 0.195),
        {"chain": "ApronFront"})
    add("spring_ApronF_3", "spring_ApronF_2", (0.0, -0.46, 0.195), (0.0, -0.62, 0.190),
        {"chain": "ApronFront"})
    # back skirt panel
    add("spring_ApronB_1", "Hips", (0.0, -0.06, -0.190), (0.0, -0.24, -0.235),
        {"chain": "ApronBack", "stiffness": 0.44, "damping": 0.24, "drag": 0.22})
    add("spring_ApronB_2", "spring_ApronB_1", (0.0, -0.24, -0.235), (0.0, -0.42, -0.245),
        {"chain": "ApronBack"})
    # side skirt panels
    for side in ("Left", "Right"):
        s = 1 if side == "Left" else -1
        X = lambda v: (s * v[0], v[1], v[2])
        add(f"spring_Apron{side[0]}_1", "Hips", X((0.155, -0.12, 0.150)), X((0.195, -0.28, 0.165)),
            {"chain": f"Apron{side}", "stiffness": 0.42, "damping": 0.24, "drag": 0.24})
        add(f"spring_Apron{side[0]}_2", f"spring_Apron{side[0]}_1", X((0.195, -0.28, 0.165)),
            X((0.205, -0.44, 0.150)), {"chain": f"Apron{side}"})
    # sleeve cuffs
    for side in ("Left", "Right"):
        s = 1 if side == "Left" else -1
        X = lambda v: (s * v[0], v[1], v[2])
        add(f"spring_Cuff{side[0]}_1", f"{side}ForeArm", X((0.560, 0.110, -0.065)),
            X((0.605, 0.055, -0.060)), {"chain": f"Cuff{side}", "stiffness": 0.55, "damping": 0.20, "drag": 0.15})
        add(f"spring_Cuff{side[0]}_2", f"spring_Cuff{side[0]}_1", X((0.605, 0.055, -0.060)),
            X((0.638, 0.030, -0.058)), {"chain": f"Cuff{side}"})
    return B


# ---------------------------------------------------------------------------
def bone_frames(bones):
    """Rest world matrices (bone local +Y points along head->tail) + segment list."""
    by = {b["name"]: i for i, b in enumerate(bones)}
    heads = np.array([b["head"] for b in bones], np.float64)
    tails = []
    for i, b in enumerate(bones):
        t = b["tail"]
        if t is None:
            # extend along parent direction so the segment is non-degenerate
            p = by.get(b["parent"])
            if p is not None:
                d = heads[i] - heads[p]
                n = np.linalg.norm(d)
                d = d / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
                t = heads[i] + d * 0.03
            else:
                t = heads[i] + np.array([0.0, 0.03, 0.0])
        tails.append(t)
    tails = np.array(tails, np.float64)
    dirs = tails - heads
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    R = np.tile(np.eye(3), (len(bones), 1, 1))
    for i in range(len(bones)):
        y = dirs[i]
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y, up)) > 0.97:
            up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, y)
        x /= max(np.linalg.norm(x), 1e-9)
        z = np.cross(x, y)
        R[i] = np.stack([x, y, z], axis=1)
    # Absolute bone transform: world[i] = T(head) @ R   (bone origin at its head)
    world = np.tile(np.eye(4), (len(bones), 1, 1))
    world[:, :3, :3] = R
    world[:, :3, 3] = heads
    # glTF local transforms must be relative to the parent bone
    local = world.copy()
    for i, b in enumerate(bones):
        p = by.get(b["parent"])
        if p is not None:
            local[i] = np.linalg.inv(world[p]) @ world[i]
    return local, world, heads, tails


def compute_weights(P, I, heads, tails, bone_ids, k=8):
    """Distance-to-bone-segment weights, then Laplacian smoothing over the mesh."""
    a = heads[bone_ids]
    b = tails[bone_ids]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    L = np.linalg.norm(ab, axis=1)
    sigma = np.clip(L * 0.85, 0.020, 0.10)
    best_idx = np.zeros((len(P), k), np.int64)
    best_w = np.zeros((len(P), k), np.float64)
    step = 20000
    for s in range(0, len(P), step):
        p = P[s:s + step, None, :]
        ap = p - a[None]
        t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(p - proj, axis=-1)
        w = np.exp(-((d / sigma[None]) ** 2))
        idx = np.argpartition(-w, k - 1, axis=1)[:, :k]
        wv = np.take_along_axis(w, idx, 1)
        order = np.argsort(-wv, axis=1)
        best_idx[s:s + step] = np.take_along_axis(idx, order, 1)
        best_w[s:s + step] = np.take_along_axis(wv, order, 1)
    best_w /= np.maximum(best_w.sum(1, keepdims=True), 1e-9)
    # Laplacian smoothing (weights diffuse over the surface like a heat solve)
    e0 = np.concatenate([I[:, 0], I[:, 1], I[:, 2]])
    e1 = np.concatenate([I[:, 1], I[:, 2], I[:, 0]])
    n = len(P)
    A = sp.coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(n, n)).tocsr()
    A = A + A.T
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    An = sp.diags(1.0 / deg) @ A
    W = best_w.copy()
    for _ in range(6):
        W = 0.42 * W + 0.58 * (An @ W)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    top = np.argsort(-W, axis=1)[:, :4]
    out_i = np.take_along_axis(best_idx, top, 1)
    out_w = np.take_along_axis(W, top, 1)
    out_w = np.maximum(out_w, 1e-6)
    out_w /= out_w.sum(1, keepdims=True)
    return out_i.astype(np.uint16), out_w.astype(np.float32)


def build_adjacency(I, n):
    e = np.concatenate([I[:, 0], I[:, 1], I[:, 2]])
    f = np.concatenate([I[:, 1], I[:, 2], I[:, 0]])
    rows = np.concatenate([e, f])
    cols = np.concatenate([f, e])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def main():
    bones = build_bone_table()
    local, world, heads, tails = bone_frames(bones)
    names = [b["name"] for b in bones]
    by = {b["name"]: i for i, b in enumerate(bones)}
    print(f"{len(bones)} bones")

    g = read_glb(SRC)
    prim = g.js["meshes"][0]["primitives"][0]
    P = np.asarray(g.accessor(prim["attributes"]["POSITION"]), np.float64)
    N = np.asarray(g.accessor(prim["attributes"]["NORMAL"]), np.float64)
    T = np.asarray(g.accessor(prim["attributes"]["TEXCOORD_0"]), np.float64)
    I = g.indices(prim).reshape(-1, 3).astype(np.int64)
    print(f"mesh: {len(P)} verts {len(I)} tris")

    # bones that deform the mesh (everything except the 'End' tip helpers)
    deform = [i for i, b in enumerate(bones) if not b["name"].endswith("End")]
    JI, JW = compute_weights(P, I, heads, tails, np.array(deform), k=8)
    remap = np.full(len(bones), 0, np.int64)
    for k, i in enumerate(deform):
        remap[i] = k
    JI = remap[JI].astype(np.uint16)
    print("weight stats: mean influences %.2f" % (JI.size and (JW > 0.01).sum(1).mean()))

    # ---- build the output glb ----
    out = Gltf({"asset": {"version": "2.0", "generator": "tiger-rig"}, "scene": 0}, b"")
    out.js["scenes"] = [{"nodes": []}]
    out.js["buffers"] = [{"byteLength": 0}]

    mat = {
        "name": "tiger_guardian",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1],
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 1},
        },
        "normalTexture": {"index": 2, "scale": 1.0},
        "doubleSided": True,
    }
    out.js["materials"] = [mat]
    specs = [(0, 2048, 88), (1, 1024, 82), (2, 2048, 88)]
    for idx, size, q in specs:
        img = g.js["images"][idx]
        bv = g.js["bufferViews"][img["bufferView"]]
        blob = bytes(g.bin[bv.get("byteOffset", 0):bv["byteOffset"] + bv["byteLength"]])
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        if max(im.size) > size:
            sc = size / max(im.size)
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        out.add_image(buf.getvalue(), "image/jpeg")
        print(f"  texture {idx}: {im.size} {len(buf.getvalue())/1e6:.2f}MB")

    a_pos = out.add_accessor(P.astype(np.float32), 5126, 34962, True)
    a_nrm = out.add_accessor(N.astype(np.float32), 5126, 34962)
    a_uv = out.add_accessor(T.astype(np.float32), 5126, 34962)
    a_j = out.add_accessor(JI, 5123, 34962)
    a_w = out.add_accessor(JW, 5126, 34962)
    a_idx = out.add_accessor(I.astype(np.uint32).reshape(-1), 5125, 34963)
    out.js["meshes"] = [{"name": "tiger_mesh", "primitives": [{
        "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv,
                       "JOINTS_0": a_j, "WEIGHTS_0": a_w},
        "indices": a_idx, "material": 0, "mode": 4}]}]

    node_of = {}
    for i, b in enumerate(bones):
        R = local[i]
        quat = mat_to_quat(R[:3, :3])
        out.js["nodes"].append({"name": b["name"], "translation": [float(x) for x in R[:3, 3]],
                                "rotation": [float(x) for x in quat]})
        node_of[i] = len(out.js["nodes"]) - 1
    for i, b in enumerate(bones):
        p = by.get(b["parent"])
        if p is not None:
            out.js["nodes"][node_of[p]].setdefault("children", []).append(node_of[i])
    mesh_node = len(out.js["nodes"])
    out.js["nodes"].append({"name": "TigerMesh", "mesh": 0, "skin": 0})
    out.js["scenes"][0]["nodes"] = [mesh_node, node_of[0]]

    ibm = np.linalg.inv(world)
    flat = ibm.reshape(len(bones), 16)
    a_ibm = out.add_accessor(flat.astype(np.float32), 5126)
    out.js["skins"] = [{"name": "tiger_skeleton", "inverseBindMatrices": a_ibm,
                        "joints": [node_of[i] for i in range(len(bones))], "skeleton": node_of[0]}]
    out.js["buffers"][0]["byteLength"] = len(out.bin)
    out.save("../out/tiger_rig.glb")
    print("wrote ../out/tiger_rig.glb")

    # ---- skeleton metadata for the runtime spring-bone solver ----
    chains = {}
    for b in bones:
        if b.get("spring"):
            c = chains.setdefault(b["spring"]["chain"], {"joints": [], **b["spring"]})
            c["joints"].append(b["name"])
    meta = {
        "bones": [{"name": b["name"], "parent": b["parent"], "head": b["head"],
                   "tail": b["tail"]} for b in bones],
        "restWorld": world.tolist(),
        "chains": list(chains.values()),
        "scale": 1.0,
    }
    json.dump(meta, open("../out/skeleton.json", "w"), indent=1)
    print("chains:", {k: len(v["joints"]) for k, v in chains.items()})


def mat_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                         (m[1, 0] - m[0, 1]) * s, 0.25 / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


if __name__ == "__main__":
    main()