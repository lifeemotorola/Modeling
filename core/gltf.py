"""
Colada Character Maker - glTF 2.0 / GLB exporter
================================================
Writes a single .glb containing the skinned mesh, the armature and every
animation clip.  Imports straight into Blender (File > Import > glTF 2.0),
three.js, Godot, Unity, Unreal...
"""
from __future__ import annotations

import base64
import io
import json
import struct
import numpy as np

from . import atlas
from .animations import _continuity
from .skeleton import Rig

# component types
FLOAT = 5126
UINT = 5125
USHORT = 5123
UBYTE = 5121


class GLBBuilder:
    def __init__(self):
        self.bin = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def _align(self, n=4):
        while len(self.bin) % n:
            self.bin.append(0)

    def add_view(self, data: bytes, target: int | None = None) -> int:
        self._align(4)
        off = len(self.bin)
        self.bin += data
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def add_accessor(self, arr: np.ndarray, type_: str, component: int, target: int | None = None,
                     minmax: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        view = self.add_view(arr.tobytes(), target)
        acc = {
            "bufferView": view,
            "componentType": component,
            "count": int(arr.shape[0]),
            "type": type_,
        }
        if minmax and arr.size:
            acc["min"] = [float(v) for v in np.min(arr, axis=0)]
            acc["max"] = [float(v) for v in np.max(arr, axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def export_glb(spec, metrics, rig: Rig, mesh, texture_png: bytes, clips: list[dict],
               generator: str = "Colada Character Maker") -> bytes:
    b = GLBBuilder()

    # ---------------------------------------------------------------- mesh
    verts = np.asarray(mesh.verts, dtype=np.float32)
    norms = np.asarray(mesh.normals, dtype=np.float32)
    uvs = np.asarray(mesh.uvs, dtype=np.float32).copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]                                # glTF 2.0 UV origin is top-left
    joints = np.asarray(mesh.joints, dtype=np.uint16)          # (N,4) -> skin.joints index
    weights = np.asarray(mesh.weights, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    flat_faces = faces.reshape(-1)

    acc_pos = b.add_accessor(verts, "VEC3", FLOAT, target=34962, minmax=True)
    acc_nrm = b.add_accessor(norms, "VEC3", FLOAT, target=34962)
    acc_uv = b.add_accessor(uvs, "VEC2", FLOAT, target=34962)
    acc_jnt = b.add_accessor(joints, "VEC4", USHORT, target=34962)
    acc_wgt = b.add_accessor(weights, "VEC4", FLOAT, target=34962)

    primitives = []
    for material, ranges in mesh.index_ranges.items():
        idx = np.concatenate([flat_faces[s:s + c] for s, c in ranges]).astype(np.uint32)
        if idx.size == 0:
            continue
        acc_idx = b.add_accessor(idx, "SCALAR", UINT, target=34963)
        primitives.append({
            "attributes": {"POSITION": acc_pos, "NORMAL": acc_nrm, "TEXCOORD_0": acc_uv,
                           "JOINTS_0": acc_jnt, "WEIGHTS_0": acc_wgt},
            "indices": acc_idx,
            "material": atlas.MATERIALS.index(material) if material in atlas.MATERIALS else 0,
        })

    # -------------------------------------------------------------- texture
    view_img = b.add_view(texture_png)
    images = [{"bufferView": view_img, "mimeType": "image/png", "name": "colada_atlas"}]
    textures = [{"source": 0, "sampler": 0}]
    samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]

    double_sided = {"hair", "glasses", "extras", "extras_skin", "hat"}
    materials = []
    for name in atlas.MATERIALS:
        materials.append({
            "name": f"colada_{name}",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0 if name != "glasses" else 0.35,
                "roughnessFactor": 0.62 if name in ("head", "skin", "eyes", "brows", "mouth", "hat", "top", "bottom") else 0.78,
            },
            "doubleSided": name in double_sided,
        })

    # --------------------------------------------------------------- nodes
    # node 0 = armature root, nodes 1..N = bones, last node = skinned mesh
    from .animations import _mat_to_quat
    T, R = rig.local_transforms()
    nodes: list[dict] = [{"name": "Colada_Armature", "children": []}]
    node_of_bone: dict[str, int] = {}
    for i, bone in enumerate(rig.bones):
        nodes.append({
            "name": bone.name,
            "translation": [float(v) for v in T[i]],
            "rotation": [float(v) for v in _mat_to_quat(R[i])],
            "children": [],
        })
        node_of_bone[bone.name] = i + 1
    for bone in rig.bones:
        if bone.parent is not None:
            nodes[node_of_bone[bone.parent]]["children"].append(node_of_bone[bone.name])
    nodes[0]["children"] = [node_of_bone["Root"]]
    nodes.append({"name": "Colada_Mesh", "mesh": 0, "skin": 0})
    mesh_node_index = len(nodes) - 1

    # ---------------------------------------------------------------- skin
    deform_names = [bn.name for bn in rig.deform_bones]
    ibm = rig.inverse_bind().astype(np.float32).transpose(0, 2, 1)   # glTF matrices are column major
    acc_ibm = b.add_accessor(ibm.reshape(len(ibm), 16), "MAT4", FLOAT)
    skin = {
        "name": "Colada_Rig",
        "joints": [node_of_bone[n] for n in deform_names],
        "inverseBindMatrices": acc_ibm,
        "skeleton": node_of_bone["Root"],
    }

    # ---------------------------------------------------------- animations
    animations = []
    for clip in clips:
        times = np.asarray(clip["times"], dtype=np.float32)
        quats = _continuity(np.asarray(clip["quats"], dtype=np.float64))
        trans = np.asarray(clip["trans"], dtype=np.float64)
        acc_t = b.add_accessor(times.reshape(-1, 1), "SCALAR", FLOAT, minmax=True)
        samplers_ = []
        channels = []
        for bi, bone in enumerate(rig.bones):
            node_idx = node_of_bone[bone.name]
            # rotation
            acc_q = b.add_accessor(quats[:, bi, :].astype(np.float32), "VEC4", FLOAT)
            samplers_.append({"input": acc_t, "output": acc_q, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers_) - 1, "target": {"node": node_idx, "path": "rotation"}})
            # translation (only where it actually moves, keeps files small)
            tr = trans[:, bi, :]
            if np.max(np.abs(tr - tr[0])) > 1e-6:
                acc_v = b.add_accessor(tr.astype(np.float32), "VEC3", FLOAT)
                samplers_.append({"input": acc_t, "output": acc_v, "interpolation": "LINEAR"})
                channels.append({"sampler": len(samplers_) - 1, "target": {"node": node_idx, "path": "translation"}})
        animations.append({
            "name": f"Colada_{clip['name'].capitalize()}",
            "samplers": samplers_,
            "channels": channels,
        })

    # ---------------------------------------------------------------- json
    extras = {
        "generator": generator,
        "spec": {k: v for k, v in spec.items()},
        "metrics": {
            "height_m": round(metrics.H, 4),
            "head_top_y": round(metrics.head_top_y, 4),
            "deform_bones": len(rig.deform_bones),
            "vertices": int(verts.shape[0]),
            "triangles": int(faces.shape[0]),
        },
    }
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"name": "Colada", "nodes": [0, mesh_node_index]}],
        "nodes": nodes,
        "meshes": [{"name": "ColadaCharacter", "primitives": primitives}],
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": samplers,
        "skins": [skin],
        "animations": animations,
        "bufferViews": b.views,
        "accessors": b.accessors,
        "buffers": [{"byteLength": len(b.bin)}],
        "extras": extras,
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    bin_bytes = bytes(b.bin)
    while len(bin_bytes) % 4:
        bin_bytes += b"\x00"
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = io.BytesIO()
    out.write(struct.pack("<III", 0x46546C67, 2, total))
    out.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    out.write(json_bytes)
    out.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
    out.write(bin_bytes)
    return out.getvalue()
