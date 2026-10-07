"""Validate retargeted animations: FK the output GLB and compare bone world
directions against the source animations, frame by frame.

Usage: python3 validate.py <out_glb> <src_anim_dir> [anim_id ...]
"""
import os
import sys

import numpy as np

from animlib import AnimGlb
from retarget import BONE_MAP, DIR_CHILD

PAIRS = [  # (target bone, target child, source bone, source child)
    ("upperarm_l", "lowerarm_l", "LeftArm", "LeftForeArm"),
    ("lowerarm_l", "hand_l", "LeftForeArm", "LeftHand"),
    ("upperarm_r", "lowerarm_r", "RightArm", "RightForeArm"),
    ("lowerarm_r", "hand_r", "RightForeArm", "RightHand"),
    ("thigh_l", "calf_l", "LeftLeg", "LeftShin"),
    ("calf_l", "foot_l", "LeftShin", "LeftFoot"),
    ("thigh_r", "calf_r", "RightLeg", "RightShin"),
    ("calf_r", "foot_r", "RightShin", "RightFoot"),
    ("spine_01", "spine_02", "Spine1", "Spine2"),
    ("spine_03", "neck_01", "Chest", "Neck1"),
    ("neck_01", "head", "Neck1", "Head"),
    ("foot_l", "ball_l", "LeftFoot", "LeftToeBase"),
    ("foot_r", "ball_r", "RightFoot", "RightToeBase"),
]


def ang_err(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))


def main():
    out_path, src_dir = sys.argv[1], sys.argv[2]
    ids = sys.argv[3:] or [f[:-4] for f in sorted(os.listdir(src_dir)) if f.endswith(".glb")]
    tgt = AnimGlb(out_path)
    js = tgt.gltf.js
    anims = {a["name"]: a for a in js["animations"]}

    from retarget import display_name
    overall_worst = 0.0
    for aid in ids:
        src = AnimGlb(os.path.join(src_dir, aid + ".glb"))
        cat, disp = display_name(aid)
        if disp not in anims:
            print("MISSING", aid)
            continue
        tgt.anim = anims[disp]
        tgt._load_tracks()
        n = min(int(round(src.duration() * 30)) + 1, int(round(tgt.duration() * 30)) + 1)
        worst, tot, cnt = 0.0, 0.0, 0
        worst_desc = ""
        for f in range(0, n, 3):
            t_s = src.t_min + f / 30
            t_t = tgt.t_min + f / 30
            swq, swp = src.world_at(t_s)
            twq, twp = tgt.world_at(t_s if t_s <= tgt.t_max else tgt.t_max)
            # recompute at target's own clock: animations start at same offset
            twq, twp = tgt.world_at(tgt.t_min + f / 30)
            for tb, tc, sb, sc in PAIRS:
                si, sj = src.name2idx.get(sb), src.name2idx.get(sc)
                ti, tj = tgt.name2idx.get(tb), tgt.name2idx.get(tc)
                if None in (si, sj, ti, tj):
                    continue
                e = ang_err(swp[sj] - swp[si], twp[tj] - twp[ti])
                tot += e; cnt += 1
                if e > worst:
                    worst = e
                    worst_desc = f"{tb} f{f}"
        mean = tot / max(cnt, 1)
        overall_worst = max(overall_worst, worst)
        flag = "OK " if worst < 12 else "BAD"
        print(f"{flag} {aid:34s} mean {mean:5.2f}°  worst {worst:5.2f}° ({worst_desc})")
    print(f"overall worst: {overall_worst:.2f}°")


if __name__ == "__main__":
    main()
