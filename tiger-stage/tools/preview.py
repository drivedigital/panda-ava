"""Tiny software GLB previewer: z-buffered Lambert render to PNG (no GPU needed)."""
import math
import os
import sys

import numpy as np
from PIL import Image

from glblib import read_glb


def look_at(eye, target, up=(0, 1, 0)):
    f = np.array(target, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([s, u, -f])


def render(gltf, azim, elev, W=440, H=520, fov=32.0, bg=(0.16, 0.17, 0.20)):
    js = gltf.js
    eye_r = 3.2
    az, el = math.radians(azim), math.radians(elev)
    eye = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)]) * eye_r
    target = np.array([0.0, 0.85, 0.0])
    V = look_at(eye, target)
    M = np.eye(4)
    M[:3, :3] = V
    M[:3, 3] = -V @ eye
    V = M

    def proj(p):
        c = V[:3, :3] @ p.T + V[:3, 3:4]
        z = -c[2]
        f = 1.0 / math.tan(math.radians(fov) / 2)
        return np.stack([(c[0] * f / np.maximum(z, 1e-4)) * W / 2 + W / 2,
                         (1 - c[1] * f / np.maximum(z, 1e-4)) * H / 2, z])

    img = np.ones((H, W, 3), np.float32) * np.array(bg, np.float32)
    zbuf = np.full((H, W), 1e9, np.float32)
    lights = [np.array([0.5, 0.8, 0.6]), np.array([-0.7, 0.25, -0.4]), np.array([0.0, 0.6, -1.0])]

    textures = {}
    for mat in js.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        tex = pbr.get("baseColorTexture")
        bc = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1]), np.float32)
        if tex is not None:
            img_i = js["images"][tex["index"]]
            data = bytes(gltf.bin[js["bufferViews"][img_i["bufferView"]].get("byteOffset", 0):
                                  js["bufferViews"][img_i["bufferView"]]["byteOffset"] +
                                  js["bufferViews"][img_i["bufferView"]]["byteLength"]])
            tex_i = Image.open(__import__("io").BytesIO(data)).convert("RGB")
            textures[mat.get("name", str(id(mat)))] = np.asarray(tex_i, np.float32) / 255.0
        else:
            textures[mat.get("name", str(id(mat)))] = bc[:3]

    for node in js["nodes"]:
        if "mesh" not in node:
            continue
        mesh = js["meshes"][node["mesh"]]
        for prim in mesh["primitives"]:
            P = np.asarray(gltf.accessor(prim["attributes"]["POSITION"]), np.float64)
            N = np.asarray(gltf.accessor(prim["attributes"]["NORMAL"]), np.float64)
            I = gltf.indices(prim).reshape(-1, 3)
            albedo = textures.get(js["materials"][prim["material"]].get("name", "") if "material" in prim else "", np.ones(3, np.float32))
            uv = None
            if "TEXCOORD_0" in prim["attributes"]:
                T = np.asarray(gltf.accessor(prim["attributes"]["TEXCOORD_0"]), np.float64)
                use_tex = isinstance(albedo, np.ndarray) and albedo.ndim == 3
                uv = T if use_tex else None
            Vn = (V[:3, :3] @ N.T).T
            S = proj(P)
            sx, sy, sz = S[0], S[1], S[2]
            tri_x = np.stack([sx[I[:, 0]], sx[I[:, 1]], sx[I[:, 2]]])
            tri_y = np.stack([sy[I[:, 0]], sy[I[:, 1]], sy[I[:, 2]]])
            x0 = np.clip(np.floor(tri_x.min(axis=0)), 0, W - 1).astype(int)
            x1 = np.clip(np.ceil(tri_x.max(axis=0)), 0, W - 1).astype(int)
            y0 = np.clip(np.floor(tri_y.min(axis=0)), 0, H - 1).astype(int)
            y1 = np.clip(np.ceil(tri_y.max(axis=0)), 0, H - 1).astype(int)
            keep = (x1 > x0) & (y1 > y0) & (sz[I[:, 0]] > 0)
            for i in np.nonzero(keep)[0]:
                a, b, c = I[i]
                pa, pb, pc = P[a], P[b], P[c]
                ar = np.cross(pb - pa, pc - pa)
                if ar @ ar < 1e-18:
                    continue
                area2 = np.linalg.norm(ar)
                nrm = ar / area2
                lo_x, hi_x, lo_y, hi_y = x0[i], x1[i], y0[i], y1[i]
                ys, xs = np.mgrid[lo_y:hi_y + 1, lo_x:hi_x + 1]
                px = xs + 0.5
                py = ys + 0.5
                # barycentric in screen space
                e1 = (sx[b] - sx[a], sy[b] - sy[a])
                e2 = (sx[c] - sx[a], sy[c] - sy[a])
                det = e1[0] * e2[1] - e1[1] * e2[0]
                if abs(det) < 1e-12:
                    continue
                l1 = ((px - sx[a]) * e2[1] - (py - sy[a]) * e2[0]) / det
                l2 = (e1[0] * (py - sy[a]) - e1[1] * (px - sx[a])) / det
                l3 = 1 - l1 - l2
                inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
                if not inside.any():
                    continue
                zz = l1 * sz[a] + l2 * sz[b] + l3 * sz[c]
                sub = zbuf[lo_y:hi_y + 1, lo_x:hi_x + 1]
                m = inside & (zz < sub)
                if not m.any():
                    continue
                wn = (l1[..., None] * Vn[a] + l2[..., None] * Vn[b] + l3[..., None] * Vn[c])
                nrm2 = np.linalg.norm(wn, axis=-1, keepdims=True)
                wn = wn / np.maximum(nrm2, 1e-9)
                diff = np.zeros_like(wn)
                for L in lights:
                    Ln = V[:3, :3] @ L / np.linalg.norm(L)
                    diff += np.maximum(0.0, (wn * Ln).sum(-1, keepdims=True)) * 0.36
                diff += 0.10
                col = np.array(albedo, np.float32)
                if uv is not None:
                    tu = l1 * uv[a] + l2 * uv[b] + l3 * uv[c]
                    ih, iw = albedo.shape[:2]
                    tx = np.clip((tu[..., 0] % 1.0 * iw).astype(int), 0, iw - 1)
                    ty = np.clip(((1 - tu[..., 1] % 1.0) * ih).astype(int), 0, ih - 1)
                    col = albedo[ty, tx]
                sub[m] = zz[m]
                tgt = img[lo_y:hi_y + 1, lo_x:hi_x + 1]
                tgt[m] = np.clip(col * diff, 0, 1)[m]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    src, outdir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    g = read_glb(src)
    views = [("front", 0, 4), ("side", 90, 4), ("threeq", 35, 12), ("top", 30, 55)]
    tiles = []
    for name, az, el in views:
        tiles.append(render(g, az, el))
        Image.fromarray(tiles[-1]).save(os.path.join(outdir, f"{tag}_{name}.png"))
    sheet = np.concatenate(tiles, axis=1)
    Image.fromarray(sheet).save(os.path.join(outdir, f"{tag}_sheet.png"))
    print("wrote", os.path.join(outdir, f"{tag}_sheet.png"))


if __name__ == "__main__":
    main()
