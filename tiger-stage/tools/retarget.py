"""Retarget FBX-derived animation GLBs onto the tiger_detached_skirt skeleton.

World-space rotation-delta retargeting with per-bone rest-direction calibration
(T-pose alignment), then bakes local-space tracks into the tiger GLB.

Usage: python3 retarget.py <src_anim_dir> <target_glb> <out_glb> <manifest.json>
"""
import json
import os
import sys

import numpy as np

from animlib import AnimGlb, qconj, qmul, qnorm, shortest_arc
from glblib import read_glb

# target bone -> source bone
BONE_MAP = {
    "pelvis": "Hips",
    "spine_01": "Spine1", "spine_02": "Spine2", "spine_03": "Chest",
    "neck_01": "Neck1", "head": "Head",
    "clavicle_l": "LeftShoulder", "upperarm_l": "LeftArm", "lowerarm_l": "LeftForeArm", "hand_l": "LeftHand",
    "index_01_l": "LeftHandIndex1", "index_02_l": "LeftHandIndex2", "index_03_l": "LeftHandIndex3",
    "clavicle_r": "RightShoulder", "upperarm_r": "RightArm", "lowerarm_r": "RightForeArm", "hand_r": "RightHand",
    "index_01_r": "RightHandIndex1", "index_02_r": "RightHandIndex2", "index_03_r": "RightHandIndex3",
    "thigh_l": "LeftLeg", "calf_l": "LeftShin", "foot_l": "LeftFoot", "ball_l": "LeftToeBase",
    "thigh_r": "RightLeg", "calf_r": "RightShin", "foot_r": "RightFoot", "ball_r": "RightToeBase",
}
# direction-defining child (target, source); None -> identity calibration
DIR_CHILD = {
    "spine_01": ("spine_02", "Spine2"), "spine_02": ("spine_03", "Chest"),
    "spine_03": ("neck_01", "Neck1"), "neck_01": ("head", "Head"),
    "clavicle_l": ("upperarm_l", "LeftArm"), "upperarm_l": ("lowerarm_l", "LeftForeArm"),
    "lowerarm_l": ("hand_l", "LeftHand"), "hand_l": ("index_01_l", "LeftHandIndex1"),
    "index_01_l": ("index_02_l", "LeftHandIndex2"), "index_02_l": ("index_03_l", "LeftHandIndex3"),
    "clavicle_r": ("upperarm_r", "RightArm"), "upperarm_r": ("lowerarm_r", "RightForeArm"),
    "lowerarm_r": ("hand_r", "RightHand"), "hand_r": ("index_01_r", "RightHandIndex1"),
    "index_01_r": ("index_02_r", "RightHandIndex2"), "index_02_r": ("index_03_r", "RightHandIndex3"),
    "thigh_l": ("calf_l", "LeftShin"), "calf_l": ("foot_l", "LeftFoot"), "foot_l": ("ball_l", "LeftToeBase"),
    "ball_l": (None, "LeftToeEnd"),
    "thigh_r": ("calf_r", "RightShin"), "calf_r": ("foot_r", "RightFoot"), "foot_r": ("ball_r", "RightToeBase"),
    "ball_r": (None, "RightToeEnd"),
}

CATEGORIES = {
    "tkd": "Taekwondo", "muay_thai": "Muay Thai", "sword": "Sword", "staff": "Staff",
    "spear": "Spear", "wrestling": "Wrestling", "judo": "Judo",
}
FPS = 30


def display_name(base):
    for prefix in sorted(CATEGORIES, key=len, reverse=True):
        if base.startswith(prefix + "_"):
            rest = base[len(prefix) + 1:].replace("_", " ").title()
            return CATEGORIES[prefix], f"{CATEGORIES[prefix]} · {rest}"
    return "Other", base.replace("_", " ").title()


def hierarchy_order(names, parent_of):
    order, seen = [], set()
    def visit(n):
        if n in seen:
            return
        p = parent_of.get(n)
        if p is not None and p in names:
            visit(p)
        seen.add(n)
        order.append(n)
    for n in names:
        visit(n)
    return order


def main():
    src_dir, target_path, out_path, manifest_path = sys.argv[1:5]

    tgt = AnimGlb(target_path)
    t_rest_q, t_rest_p = tgt.rest_world()
    t_name = tgt.name2idx  # target bone name -> node idx
    t_parent_name = {}
    for i, n in enumerate(tgt.nodes):
        for c in n.get("children", []):
            t_parent_name[tgt.nodes[c].get("name")] = n.get("name")

    order = hierarchy_order([b for b in BONE_MAP if b in t_name], t_parent_name)

    manifest = {"animations": [], "bones": order}
    g = tgt.gltf
    js = g.js

    # fix image mime types (some entries miss it -> three.js would fail)
    for img in js.get("images", []):
        if "bufferView" in img and "mimeType" not in img:
            img["mimeType"] = "image/jpeg"

    src_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".glb"))
    for si, src_file in enumerate(src_files):
        base = os.path.splitext(src_file)[0]
        src = AnimGlb(os.path.join(src_dir, src_file))
        s_rest_q, s_rest_p = src.rest_world()
        s_name = src.name2idx

        missing = [BONE_MAP[b] for b in order if BONE_MAP[b] not in s_name]
        if missing:
            print(f"SKIP {base}: missing source bones {missing}")
            continue

        hips = s_name["Hips"]
        k = t_rest_p[t_name["pelvis"]][1] / s_rest_p[hips][1]

        # ---- calibration pass: cal_q[bone] = world rotation of aligned rest ----
        cal_q, cal_p, cal_s = {}, {}, {}
        cal_world = {}
        for b in tgt.name2idx:  # full target hierarchy top-down
            parent = t_parent_name.get(b)
            i = t_name[b]
            rq, rp = tgt.rest_r[i], tgt.rest_t[i]
            if parent is None:
                cal_q[b] = qnorm(rq); cal_p[b] = np.array(rp, float)
            else:
                cal_q[b] = qnorm(qmul(cal_q[parent], rq))
                cal_p[b] = cal_p[parent] + _qrot(cal_q[parent], rp * cal_s.get(parent, np.ones(3)))
            cal_s[b] = (cal_s.get(parent, np.ones(3)) * tgt.rest_s[i])
            if b in BONE_MAP and b in DIR_CHILD and DIR_CHILD[b][0] is not None:
                pass  # computed below after chain position known
        # second pass with direction alignment in hierarchy order
        for b in order:
            ct, cs = DIR_CHILD.get(b, (None, None))
            if ct is None:
                continue
            d_t = cal_p[ct] - cal_p[b]
            if np.linalg.norm(d_t) < 1e-8:
                continue
            s_rest_pos_b = s_rest_p[s_name[BONE_MAP[b]]]
            s_rest_pos_c = s_rest_p[s_name[cs]]
            d_s = s_rest_pos_c - s_rest_pos_b
            if np.linalg.norm(d_s) < 1e-8:
                continue
            C = shortest_arc(d_t, d_s)
            # apply C to this bone and re-propagate to descendants
            cal_q[b] = qnorm(qmul(C, cal_q[b]))
            # recompute positions/rotations of descendants of b
            for d in tgt.name2idx:
                if d == b or not _is_descendant(d, b, t_parent_name):
                    continue
                parent = t_parent_name[d]
                i = t_name[d]
                cal_q[d] = qnorm(qmul(cal_q[parent], tgt.rest_r[i]))
                cal_p[d] = cal_p[parent] + _qrot(cal_q[parent], tgt.rest_t[i] * cal_s.get(parent, np.ones(3)))

        # ---- bake frames ----
        dur = src.duration()
        n_frames = max(2, int(round(dur * FPS)) + 1)
        times = (np.arange(n_frames) / FPS).astype(np.float32)
        rot_tracks = {b: np.zeros((n_frames, 4), np.float32) for b in order}
        pelvis_t = np.zeros((n_frames, 3), np.float32)

        # static parent (world) of pelvis for localizing translation
        pelvis_parent = t_parent_name.get("pelvis")
        pp_q = t_rest_q[t_name[pelvis_parent]] if pelvis_parent else np.array([0., 0., 0., 1.])
        pp_p = t_rest_p[t_name[pelvis_parent]] if pelvis_parent else np.zeros(3)
        # pelvis parent scale (walk up rest scales)
        pp_s = np.ones(3)
        n = pelvis_parent
        chain = []
        while n is not None:
            chain.append(n)
            n = t_parent_name.get(n)
        for n in reversed(chain):
            pp_s = pp_s * tgt.rest_s[tgt.name2idx[n]]

        for f in range(n_frames):
            t = src.t_min + f / FPS
            sw_q, sw_p = src.world_at(t)
            wq = {}
            for b in order:
                s = s_name[BONE_MAP[b]]
                D = qmul(sw_q[s], qconj(s_rest_q[s]))
                wq[b] = qnorm(qmul(D, cal_q[b]))
            for b in order:
                parent = t_parent_name.get(b)
                if parent in wq:
                    pq = wq[parent]
                elif parent is not None:
                    pq = t_rest_q[t_name[parent]]
                else:
                    pq = np.array([0., 0., 0., 1.])
                rot_tracks[b][f] = qmul(qconj(pq), wq[b]).astype(np.float32)
            # pelvis translation (world delta scaled, then localized)
            pw = t_rest_p[t_name["pelvis"]] + (sw_p[hips] - s_rest_p[hips]) * k
            loc = _qrot(qconj(pp_q), pw - pp_p) / pp_s
            pelvis_t[f] = loc.astype(np.float32)

        # quaternion continuity fix (avoid sign flips)
        for b in order:
            tr = rot_tracks[b]
            for f in range(1, n_frames):
                if np.dot(tr[f], tr[f - 1]) < 0:
                    tr[f] = -tr[f]

        # ---- write animation into glb ----
        t_acc = g.add_accessor(times.reshape(-1, 1), 5126, minmax=True)
        chans = []
        for b in order:
            v_acc = g.add_accessor(rot_tracks[b], 5126)
            chans.append((t_name[b], "rotation", t_acc, v_acc))
        v_acc = g.add_accessor(pelvis_t, 5126, minmax=True)
        chans.append((t_name["pelvis"], "translation", t_acc, v_acc))

        samplers = []
        channels = []
        for (node_i, path, in_a, out_a) in chans:
            samplers.append({"input": in_a, "output": out_a, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": int(node_i), "path": path}})
        cat, disp = display_name(base)
        js["animations"].append({"name": disp, "samplers": samplers, "channels": channels})
        manifest["animations"].append({"id": base, "name": disp, "category": cat,
                                       "duration": round(dur, 3)})
        print(f"OK {base:36s} -> {disp:34s} {dur:5.2f}s {n_frames}f")

    # ---- T-pose calibration reference animation (1 pose, loopable) ----
    # uses last source's calibrations is wrong; recompute against a neutral source:
    # skip if no sources
    if src_files:
        src0 = AnimGlb(os.path.join(src_dir, src_files[0]))
        s_rest_q, s_rest_p = src0.rest_world()
        # recompute calibration for src0 (same skeleton across files, but be exact)
        cal_q, cal_p, cal_s = {}, {}, {}
        for b in tgt.name2idx:
            parent = t_parent_name.get(b)
            i = t_name[b]
            if parent is None:
                cal_q[b] = qnorm(tgt.rest_r[i]); cal_p[b] = np.array(tgt.rest_t[i], float)
            else:
                cal_q[b] = qnorm(qmul(cal_q[parent], tgt.rest_r[i]))
                cal_p[b] = cal_p[parent] + _qrot(cal_q[parent], tgt.rest_t[i] * cal_s.get(parent, np.ones(3)))
            cal_s[b] = cal_s.get(parent, np.ones(3)) * tgt.rest_s[i]
        for b in order:
            ct, cs = DIR_CHILD.get(b, (None, None))
            if ct is None:
                continue
            d_t = cal_p[ct] - cal_p[b]
            s_b = src0.name2idx.get(BONE_MAP[b]); s_c = src0.name2idx.get(cs)
            if s_b is None or s_c is None:
                continue
            d_s = s_rest_p[s_c] - s_rest_p[s_b]
            if np.linalg.norm(d_t) < 1e-8 or np.linalg.norm(d_s) < 1e-8:
                continue
            C = shortest_arc(d_t, d_s)
            cal_q[b] = qnorm(qmul(C, cal_q[b]))
            for d in tgt.name2idx:
                if d == b or not _is_descendant(d, b, t_parent_name):
                    continue
                parent = t_parent_name[d]
                i = t_name[d]
                cal_q[d] = qnorm(qmul(cal_q[parent], tgt.rest_r[i]))
                cal_p[d] = cal_p[parent] + _qrot(cal_q[parent], tgt.rest_t[i] * cal_s.get(parent, np.ones(3)))
        n_frames = 2
        times = np.array([0.0, 1.0], np.float32)
        t_acc = g.add_accessor(times.reshape(-1, 1), 5126, minmax=True)
        samplers = []
        channels = []
        for b in order:
            parent = t_parent_name.get(b)
            pq = cal_q[parent] if parent in cal_q else np.array([0., 0., 0., 1.])
            L = qmul(qconj(pq), cal_q[b]).astype(np.float32)
            vals = np.stack([L, L])
            v_acc = g.add_accessor(vals, 5126)
            samplers.append({"input": t_acc, "output": v_acc, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": int(t_name[b]), "path": "rotation"}})
        js["animations"].append({"name": "Reference · T-Pose", "samplers": samplers, "channels": channels})
        manifest["animations"].append({"id": "_tpose", "name": "Reference · T-Pose",
                                       "category": "Reference", "duration": 1.0})

    g.save(out_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"WROTE {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB), {len(manifest['animations'])} anims")


def _qrot(q, v):
    u = np.array(q[:3]); w = q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _is_descendant(d, b, parent_of):
    n = parent_of.get(d)
    while n is not None:
        if n == b:
            return True
        n = parent_of.get(n)
    return False


if __name__ == "__main__":
    main()
