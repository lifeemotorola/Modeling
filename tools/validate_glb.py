#!/usr/bin/env python3
"""
Colada Character Maker - GLB validator
======================================
Structural QA for the exported .glb: checks the container, every accessor
range, the skin, the animations and the material/texture references.

    python3 tools/validate_glb.py build/colada.glb
"""
from __future__ import annotations

import json
import struct
import sys

COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def check(glb_path: str) -> tuple[bool, list[str]]:
    log: list[str] = []
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        log.append("FAIL " + msg)

    with open(glb_path, "rb") as fh:
        data = fh.read()

    if len(data) < 20:
        return False, ["FAIL file too small"]
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        fail("bad magic (not a glb)")
    if version != 2:
        fail(f"unexpected glTF version {version}")
    if length != len(data):
        fail(f"header length {length} != file size {len(data)}")
    log.append(f"container: glTF {version}, {len(data) / 1024:.1f} kB")

    off = 12
    chunks = {}
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        chunks[ctype] = chunk
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    if 0x4E4F534A not in chunks:
        return False, log + ["FAIL no JSON chunk"]
    if 0x004E4942 not in chunks:
        fail("no BIN chunk")

    gltf = json.loads(chunks[0x4E4F534A].decode("utf-8"))
    binc = chunks.get(0x004E4942, b"")

    counts = {
        "nodes": len(gltf.get("nodes", [])),
        "meshes": len(gltf.get("meshes", [])),
        "materials": len(gltf.get("materials", [])),
        "animations": len(gltf.get("animations", [])),
        "skins": len(gltf.get("skins", [])),
    }
    log.append("counts: " + json.dumps(counts))

    # --- buffer views / accessors -----------------------------------------
    buffer_len = gltf["buffers"][0]["byteLength"]
    if buffer_len > len(binc):
        fail(f"buffer byteLength {buffer_len} > BIN chunk {len(binc)}")
    for i, bv in enumerate(gltf.get("bufferViews", [])):
        end = bv.get("byteOffset", 0) + bv["byteLength"]
        if end > buffer_len:
            fail(f"bufferView {i} overruns the buffer ({end} > {buffer_len})")

    def acc_bytes(ai):
        a = gltf["accessors"][ai]
        comp = COMPONENT_SIZE.get(a["componentType"])
        cnt = TYPE_COUNT.get(a["type"])
        if comp is None or cnt is None:
            return None
        return a["count"] * comp * cnt

    for i, a in enumerate(gltf.get("accessors", [])):
        bv = gltf["bufferViews"][a["bufferView"]]
        need = acc_bytes(i)
        avail = bv["byteLength"] - a.get("byteOffset", 0)
        if need > avail:
            fail(f"accessor {i} needs {need} B but its view has {avail} B")
    log.append(f"accessors: {len(gltf.get('accessors', []))} within {len(gltf.get('bufferViews', []))} views - ranges OK")

    # --- mesh ---------------------------------------------------------------
    for mi, mesh in enumerate(gltf.get("meshes", [])):
        for pi, prim in enumerate(mesh["primitives"]):
            attrs = prim["attributes"]
            pos = gltf["accessors"][attrs["POSITION"]]
            if "min" not in pos or "max" not in pos:
                fail(f"prim {mi}.{pi} POSITION has no min/max (breaks bounding boxes)")
            n = pos["count"]
            for name, ai in attrs.items():
                if gltf["accessors"][ai]["count"] != n:
                    fail(f"prim {mi}.{pi} attribute {name} count mismatch")
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                fail(f"prim {mi}.{pi} is not skinned")
            if prim["material"] >= len(gltf["materials"]):
                fail(f"prim {mi}.{pi} material out of range")
            idx = gltf["accessors"][prim["indices"]]
            if idx["count"] % 3:
                fail(f"prim {mi}.{pi} index count {idx['count']} is not a triangle list")
    log.append(f"mesh: {sum(len(m['primitives']) for m in gltf.get('meshes', []))} primitives, skinned + indexed")

    # --- skin ---------------------------------------------------------------
    nodes = gltf.get("nodes", [])
    for si, skin in enumerate(gltf.get("skins", [])):
        for j in skin["joints"]:
            if j >= len(nodes):
                fail(f"skin {si} joint {j} out of range")
        ibm = gltf["accessors"][skin["inverseBindMatrices"]]
        if ibm["count"] != len(skin["joints"]):
            fail(f"skin {si} IBM count {ibm['count']} != joints {len(skin['joints'])}")
        if ibm["type"] != "MAT4":
            fail(f"skin {si} inverseBindMatrices must be MAT4")
    names = [n.get("name", "") for n in nodes]
    log.append(f"skin: {len(gltf.get('skins', [{}])[0].get('joints', []))} joints, "
               f"bones: {', '.join(names[1:6])}…")

    # --- animations ---------------------------------------------------------
    for ai, anim in enumerate(gltf.get("animations", [])):
        for ch in anim["channels"]:
            if ch["sampler"] >= len(anim["samplers"]):
                fail(f"anim {ai} channel sampler out of range")
            node = ch["target"]["node"]
            if node >= len(nodes):
                fail(f"anim {ai} channel node {node} out of range")
        times = None
        for s in anim["samplers"]:
            inp = gltf["accessors"][s["input"]]
            out = gltf["accessors"][s["output"]]
            if out["count"] != inp["count"]:
                fail(f"anim {ai} sampler count mismatch ({out['count']} vs {inp['count']})")
            if s["interpolation"] not in ("LINEAR", "STEP", "CUBICSPLINE"):
                fail(f"anim {ai} bad interpolation {s['interpolation']}")
            times = inp["count"]
        log.append(f"  anim {anim.get('name', ai)}: {len(anim['channels'])} channels, {times} keys")

    # --- materials / texture -------------------------------------------------
    for i, m in enumerate(gltf.get("materials", [])):
        tex = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if tex is None:
            fail(f"material {i} has no baseColorTexture")
        elif tex["index"] >= len(gltf.get("textures", [])):
            fail(f"material {i} texture index out of range")
    for i, img in enumerate(gltf.get("images", [])):
        if "bufferView" not in img:
            fail(f"image {i} is not embedded")
        elif gltf["bufferViews"][img["bufferView"]]["byteLength"] < 100:
            fail(f"image {i} looks empty")
    log.append(f"materials: {len(gltf.get('materials', []))} slots sharing "
               f"{len(gltf.get('images', []))} embedded texture(s)")

    # --- scene ---------------------------------------------------------------
    scene = gltf["scenes"][gltf.get("scene", 0)]
    log.append(f"scene: {len(scene['nodes'])} root nodes")
    return ok, log


def main():
    paths = sys.argv[1:]
    if not paths:
        import glob
        paths = sorted(glob.glob("build/*.glb")) or sorted(glob.glob(".arena/builds/*/model.glb"))
    if not paths:
        print("usage: python3 tools/validate_glb.py <file.glb>")
        return 1
    bad = 0
    for p in paths:
        print(f"\n=== {p} ===")
        try:
            ok, log = check(p)
        except Exception as exc:                              # pragma: no cover
            print("FAIL exception:", exc)
            bad += 1
            continue
        for line in log:
            print("  " + line)
        if not ok:
            bad += 1
        print("  " + ("PASS ✅" if ok else "FAILED ❌"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
