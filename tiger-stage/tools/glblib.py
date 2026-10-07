"""Minimal glTF/GLB reader-writer used by the tiger rigging pipeline."""
import json
import struct

import numpy as np

COMP = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
NBYTES = {np.dtype(np.int8).itemsize: 1, np.dtype(np.uint8).itemsize: 1,
          np.dtype(np.int16).itemsize: 2, np.dtype(np.uint16).itemsize: 2,
          np.dtype(np.int32).itemsize: 4, np.dtype(np.uint32).itemsize: 4,
          np.dtype(np.float32).itemsize: 4}


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a GLB")
    off, js, binary = 12, None, None
    while off < length:
        clen, ctype = struct.unpack("<II", data[off:off + 8])
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binary = chunk
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    return Gltf(js, binary)


class Gltf:
    def __init__(self, js, binary):
        self.js = js
        self.bin = bytearray(binary or b"")
        self.js.setdefault("accessors", [])
        self.js.setdefault("bufferViews", [])
        self.js.setdefault("meshes", [])
        self.js.setdefault("nodes", [])
        self.js.setdefault("materials", [])

    # ---------------- reading ----------------
    def accessor(self, idx):
        a = self.js["accessors"][idx]
        bv = self.js["bufferViews"][a["bufferView"]]
        n = NCOMP[a["type"]]
        dt = np.dtype(COMP[a["componentType"]])
        base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or n * dt.itemsize
        if stride == n * dt.itemsize:
            arr = np.frombuffer(self.bin, dtype=dt, count=a["count"] * n, offset=base)
            arr = arr.reshape(a["count"], n)
        else:
            raw = np.frombuffer(self.bin, dtype=np.uint8, count=(a["count"] - 1) * stride + n * dt.itemsize,
                                offset=base).reshape(a["count"], stride)
            arr = np.ascontiguousarray(raw[:, :n * dt.itemsize]).view(dt).reshape(a["count"], n)
        if a["componentType"] in (5120, 5121, 5122, 5123, 5125):
            arr = arr.astype(np.int64)
        return arr

    def indices(self, prim):
        return np.asarray(self.accessor(prim["indices"])).reshape(-1)

    # ---------------- writing ----------------
    def add_view(self, arr, target=None):
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        arr = np.ascontiguousarray(arr)
        off = len(self.bin)
        self.bin += arr.tobytes()
        pad = (4 - len(self.bin) % 4) % 4
        self.bin += b"\x00" * pad
        self.js["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": int(arr.nbytes)})
        if target:
            self.js["bufferViews"][-1]["target"] = target
        return len(self.js["bufferViews"]) - 1

    def add_accessor(self, arr, atype, target=None, minmax=False):
        view = self.add_view(arr, target)
        acc = {"bufferView": view, "componentType": atype, "count": int(arr.shape[0]), "type": atype2type(arr.shape[1:])}
        if minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.js["accessors"].append(acc)
        return len(self.js["accessors"]) - 1

    def add_image(self, blob, mime):
        view = self.add_view(np.frombuffer(blob, dtype=np.uint8))
        self.js.setdefault("images", []).append({"bufferView": view, "mimeType": mime})
        return len(self.js["images"]) - 1

    def save(self, path):
        js = json.dumps(self.js, separators=(",", ":"))
        jsb = js.encode("utf-8")
        jsb += b" " * ((4 - len(jsb) % 4) % 4)
        total = 12 + 8 + len(jsb) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(jsb), 0x4E4F534A))
            f.write(jsb)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942))
            f.write(bytes(self.bin))


def atype2type(shape):
    if shape == ():
        return "SCALAR"
    return {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}[shape[0]]
