"""Binary FBX 7.x reader: node tree, rest-pose transforms, animation curves.

Follows the same record semantics as three.js FBXLoader: `endOffset` is an
absolute file offset marking where the record's own payload ends; child
records follow immediately and are read until the cursor passes endOffset.
A record whose endOffset is zero is a NULL-record (13 zero bytes) and is
skipped 4 bytes at a time.
"""
import struct
import zlib


class FbxNode:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props):
        self.name = name
        self.props = props
        self.children = []

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def findall(self, name):
        return [c for c in self.children if c.name == name]

    def val(self, default=None):
        return self.props[0] if self.props else default

    def __repr__(self):
        return f"<{self.name} {self.props[:2]} kids={len(self.children)}>"


def _read_prop(buf, off):
    t = buf[off:off + 1].decode("latin1")
    off += 1
    if t == "Y":
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if t == "C":
        return bool(buf[off]), off + 1
    if t == "I":
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if t == "F":
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if t == "D":
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if t == "L":
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if t in ("S", "R"):
        n = struct.unpack_from("<I", buf, off)[0]
        raw = buf[off + 4:off + 4 + n]
        return (raw.decode("utf-8", "replace") if t == "S" else raw), off + 4 + n
    if t in ("f", "d", "l", "i", "b"):
        n = struct.unpack_from("<I", buf, off)[0]
        enc = struct.unpack_from("<I", buf, off + 4)[0]
        off += 8
        sz = {"f": 4, "d": 8, "l": 8, "i": 4, "b": 1}[t]
        if enc == 0:
            raw = buf[off:off + n * sz]
            off += n * sz
        else:
            d = zlib.decompressobj()
            rest = buf[off:]
            raw = d.decompress(rest) + d.flush()
            off += len(rest) - len(d.unused_data)
        fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "B"}[t]
        return list(struct.unpack_from("<" + fmt * (len(raw) // sz), raw, 0)), off
    raise ValueError(f"unsupported property type {t!r} at {off}")


def parse_fbx(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:20] != b"Kaydara FBX Binary  ":
        raise ValueError(f"{path}: not a binary FBX")
    ver = struct.unpack_from("<I", buf, 23)[0]
    wide = ver >= 7500
    order = "<QQQ" if wide else "<III"
    esz = 8 if wide else 4

    def read_record(off, depth=0):
        if off + esz * 3 + 1 > len(buf):
            return None, off + 1
        raw_end = struct.unpack_from(order, buf, off)[0]
        end = raw_end if raw_end >= off else off + raw_end
        nprops = struct.unpack_from(order, buf, off + esz)[0]
        nl = buf[off + esz * 3]
        name = buf[off + esz * 3 + 1:off + esz * 3 + 1 + nl].decode("latin1")
        if raw_end == 0:
            return None, off + 4
        node = FbxNode(name, [])
        p = off + esz * 3 + 1 + nl
        try:
            for k in range(nprops):
                v, p = _read_prop(buf, p)
                node.props.append(v)
        except (ValueError, struct.error, zlib.error) as exc:
            if DEBUG:
                print(f"  !! {name}@{off} prop {k}/{nprops} off={p}: {exc}")
                print("     bytes:", buf[off:off+64].hex(' '))
            return None, off + 4
        guard = 0
        while p < end and guard < 200000:
            guard += 1
            child, p2 = read_record(p, depth + 1)
            if p2 <= p:
                break
            p = p2
            if child is not None:
                node.children.append(child)
        return node, p

    root = FbxNode("__root__", [])
    off = 27
    while off < len(buf) - 160 - 16:
        node, noff = read_record(off)
        if noff <= off:
            break
        off = noff
        if node is not None:
            root.children.append(node)
    return root, ver


def _prop_map(node):
    """Map 'Lcl Translation' etc. to values, from Properties70 or P children."""
    out = {}
    p70 = node.find("Properties70")
    if p70 is not None:
        for p in p70.findall("P"):
            out[p.val()] = (p.props[4] if len(p.props) > 4 else None)
    for child in node.children:
        if child.name in ("Translation", "Rotation", "Scaling", "PreRotation", "Lcl Translation",
                          "Lcl Rotation", "Lcl Scaling", "Lcl PreRotation"):
            out[child.name] = child.val()
    for child in node.children:
        if child.name in ("T", "R", "S") and len(child.props) == 1 and isinstance(child.val(), list):
            out[{"T": "Translation", "R": "Rotation", "S": "Scaling"}[child.name]] = child.val()
    return out


def read_skeleton(root):
    """Return (nodes, parent) where nodes = {name: {t,r,s}} in the FBX rest pose."""
    objs = root.find("Objects")
    models, conn = {}, {}
    if objs is None:
        return {}, {}
    for m in objs.findall("Model"):
        props = m.props
        if len(props) < 2:
            continue
        mid = props[0]
        info = {"id": mid, "name": props[1], "type": props[2] if len(props) > 2 else "Model"}
        info.update(_prop_map(m))
        models[mid] = info
    for c in objs.findall("Connection"):
        if len(c.props) >= 3 and c.props[0] == "OO":
            a, b = c.props[1], c.props[2]
            if a in models and b in models:
                models[b]["parent"] = models[a]
    by_name = {i["name"]: i for i in models.values()}
    return by_name, models


def read_curves(root):
    """{stack: {model_name: {"Lcl Translation"|"Lcl Rotation": (times, values)}}}"""
    objs = root.find("Objects")
    if objs is None:
        return {}
    conn = {}
    for c in objs.findall("Connection"):
        if len(c.props) >= 3 and c.props[0] == "OO":
            conn[c.props[1]] = c.props[2]
    stacks = {}
    for st in objs.findall("AnimationStack"):
        stacks[st.val()] = [conn.get(k.val()) for k in st.findall("AnimationStackNode")]
    layer_stack = []
    for lay in objs.findall("AnimationLayer"):
        for s in lay.findall("AnimationStackNode"):
            name = conn.get(s.val())
            if name:
                layer_stack.append(name)
    if not layer_stack:
        layer_stack = [s.val() for s in objs.findall("AnimationStack")]
    out = {}
    for ac in objs.findall("AnimationCurve"):
        name = ac.props[0].split("|")[0] if isinstance(ac.props[0], str) else None
        dprop = ac.props[1].split("|")[1] if len(ac.props) > 1 and isinstance(ac.props[1], str) else None
        if not name or not dprop:
            continue
        times, vals = [], []
        for c in ac.findall("Packed"):
            if "KeyTime" in c.props[0]:
                times = c.val()
            elif "KeyValueFloat" in c.props[0]:
                vals = c.val()
        if times and vals:
            for st in layer_stack:
                out.setdefault(st, {}).setdefault(name, {})[dprop] = (times, vals)
    return out
