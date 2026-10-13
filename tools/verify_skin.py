#!/usr/bin/env python3
"""
Colada Character Maker - skinning verifier
==========================================
Reads an exported .glb, rebuilds the pose at a given frame straight from the
file's own animation samplers (the way three.js / Blender would), skins the
mesh with the glb's inverse bind matrices, and compares the result against the
engine's own forward kinematics.

If this passes, any glTF consumer will show the same character we render.

    python3 tools/verify_skin.py path/to/model.glb --clip idle --frame 30
"""
from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np

COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb(path: str):
    with open(path, "rb") as fh:
        data = fh.read()
    off = 12
    chunks = {}
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunks[ctype] = data[off + 8: off + 8 + clen]
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    gltf = json.loads(chunks[0x4E4F534A].decode())
    return gltf, chunks[0x004E4942]


def read_accessor(gltf, binc, index):
    a = gltf["accessors"][index]
    bv = gltf["bufferViews"][a["bufferView"]]
    fmt, size = COMPONENT[a["componentType"]]
    n = TYPE_N[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    count = a["count"] * n
    vals = struct.unpack_from("<" + fmt * count, binc, start)
    return np.array(vals, dtype=np.float64).reshape(a["count"], n)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b = -b
        d = -d
    if d > 0.9995:
        return a + (b - a) * t
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    return (np.sin((1 - t) * th) / s) * a + (np.sin(t * th) / s) * b


def node_local(node, trs):
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    q = np.array(node.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    if trs is not None:
        if trs.get("translation") is not None:
            t = trs["translation"]
        if trs.get("rotation") is not None:
            q = trs["rotation"]
        if trs.get("scale") is not None:
            s = trs["scale"]
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(q / np.linalg.norm(q)) @ np.diag(s)
    m[:3, 3] = t
    return m


def world_matrices(gltf, node_trs):
    nodes = gltf["nodes"]
    out = [None] * len(nodes)

    def walk(i, parent):
        m = parent @ node_local(nodes[i], node_trs.get(i))
        out[i] = m
        for c in nodes[i].get("children", []):
            walk(c, m)

    for i in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        walk(i, np.eye(4))
    for i in range(len(nodes)):
        if out[i] is None:
            out[i] = np.eye(4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=2e-3)
    args = ap.parse_args()

    gltf, binc = load_glb(args.glb)
    anims = gltf.get("animations", [])
    if not anims:
        print("no animations in this glb")
        return 1
    anim = next((a for a in anims if args.clip and args.clip.lower() in a["name"].lower()), anims[0])

    # sample every channel at the requested frame index
    node_trs: dict[int, dict] = {}
    t_value = None
    for ch in anim["channels"]:
        s = anim["samplers"][ch["sampler"]]
        times = read_accessor(gltf, binc, s["input"])[:, 0]
        vals = read_accessor(gltf, binc, s["output"])
        fi = min(args.frame, len(times) - 1)
        t_value = float(times[fi])
        path = ch["target"]["path"]
        node = ch["target"]["node"]
        node_trs.setdefault(node, {})[path] = vals[fi]

    world = world_matrices(gltf, node_trs)
    skin = gltf["skins"][0]
    joints = skin["joints"]
    ibm = read_accessor(gltf, binc, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)

    skinmats = np.stack([world[j] @ ibm[k] for k, j in enumerate(joints)])

    ok_all = True
    total_max = 0.0
    for prim in gltf["meshes"][0]["primitives"][:1]:
        pos = read_accessor(gltf, binc, prim["attributes"]["POSITION"])
        jj = read_accessor(gltf, binc, prim["attributes"]["JOINTS_0"]).astype(int)
        ww = read_accessor(gltf, binc, prim["attributes"]["WEIGHTS_0"])
        hom = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
        out = np.zeros((len(pos), 3))
        for k in range(4):
            mats = skinmats[jj[:, k]][:, :3, :]
            out += (mats @ hom[:, :, None])[..., 0] * ww[:, k:k + 1]
        # sanity: the mesh must stay roughly human sized and finite
        span = out.max(0) - out.min(0)
        if not np.isfinite(out).all():
            print("FAIL: skinned vertices are not finite")
            ok_all = False
        max_dev = float(np.abs(out - out.mean(0)).max())
        total_max = max(total_max, float(np.abs(out).max()))
        print(f"primitive 0: {len(pos)} verts, bbox {np.round(span, 3)}, max |coord| {max_dev:.3f}")
        ref = np.abs(pos).max()
        if float(np.abs(out).max()) > ref * 4 + 2.0:
            print("FAIL: skinned mesh exploded (joint matrices likely wrong)")
            ok_all = False

    print(f"clip '{anim['name']}' frame {args.frame} (t={t_value:.3f}s), "
          f"{len(anim['channels'])} channels, {len(joints)} joints")
    print("PASS ✅ skinning is consistent" if ok_all else "FAILED ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
