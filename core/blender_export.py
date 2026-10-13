"""
Colada Character Maker - Blender export
=======================================
Generates a SINGLE self contained Python file that rebuilds the character
natively inside Blender:

  * real mesh (vertex groups, per-material slots, baked texture atlas)
  * real armature with correct bone rolls + deform flags
  * real actions per animation clip (keyed pose bones, pushed to NLA)
  * optional IK targets, studio lighting rig and camera

The pipeline bakes geometry, rig, animation and the texture into the script
(zlib + base64) so it needs nothing but Blender itself.
"""
from __future__ import annotations

import base64
import io
import json
import struct
import time
import zlib

import numpy as np

from . import atlas


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------

def _pack_mesh(mesh) -> str:
    verts = np.asarray(mesh.verts, dtype="<f4")
    norms = np.asarray(mesh.normals, dtype="<f4")
    uvs = np.asarray(mesh.uvs, dtype="<f4")
    faces = np.asarray(mesh.faces, dtype="<u4")
    joints = np.asarray(mesh.joints, dtype="<u2")
    weights = np.asarray(mesh.weights, dtype="<f4")
    mat_index = np.zeros(len(faces), dtype=np.uint8)
    for material, ranges in mesh.index_ranges.items():
        mi = atlas.MATERIALS.index(material)
        for start, count in ranges:
            mat_index[start // 3: (start + count) // 3] = mi
    blob = b"".join([
        struct.pack("<I", len(verts)), verts.tobytes(), norms.tobytes(), uvs.tobytes(),
        struct.pack("<I", len(faces)), faces.tobytes(), joints.tobytes(), weights.tobytes(),
        mat_index.tobytes(),
    ])
    return base64.b64encode(zlib.compress(blob, 9)).decode()


def _pack_rig(rig) -> str:
    from .animations import _mat_to_quat
    T, R = rig.local_transforms()
    rows = []
    for i, b in enumerate(rig.bones):
        rows.append({
            "name": b.name,
            "parent": b.parent,
            "head": [round(float(v), 6) for v in b.head],
            "tail": [round(float(v), 6) for v in b.tail],
            "z": [round(float(v), 6) for v in rig.rest_world[i][:3, 2]],
            "deform": bool(b.deform),
            "group": b.group,
            "loc": [round(float(v), 6) for v in T[i]],
            "quat": [round(float(v), 6) for v in _mat_to_quat(R[i])],
        })
    payload = {"bones": rows, "height": round(float(rig.metrics.H), 4),
               "head_y": round(float(rig.metrics.head_top_y), 4)}
    return base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()


def _pack_anim(clips: list[dict]) -> str:
    from .animations import _continuity
    out = []
    for clip in clips:
        q = _continuity(np.asarray(clip["quats"], dtype=np.float64))
        t = np.asarray(clip["trans"], dtype=np.float64)
        out.append({
            "name": clip["name"],
            "duration": float(clip["duration"]),
            "loop": bool(clip["loop"]),
            "fps": int(clip["fps"]),
            "times": [round(float(x), 5) for x in clip["times"]],
            "quats": [[[round(float(v), 6) for v in bone] for bone in frame] for frame in q],
            "trans": [[[round(float(v), 6) for v in bone] for bone in frame] for frame in t],
        })
    return base64.b64encode(zlib.compress(json.dumps(out).encode(), 6)).decode()


# ---------------------------------------------------------------------------
# the generated script (token templated - braces are safe)
# ---------------------------------------------------------------------------

TEMPLATE = r'''"""
Colada Character Maker - native Blender build script
====================================================
Generated @@DATE@@   |   fingerprint @@FINGERPRINT@@

HOW TO USE
----------
1. Open Blender 3.x / 4.x
2. Scripting workspace > Open > this file  (or paste it in the Text Editor)
3. Press "Run Script"  (Alt+P)

WHAT YOU GET
------------
  Colada_Mesh     skinned mesh, one material slot per body part, baked texture
  Colada_Rig      armature (deform bones + muted IK targets for hands/feet)
  Actions         one per clip (@@CLIPNAMES@@) pushed onto NLA tracks
  Studio          three area lights + tracked camera, ready to render

Character: @@HEIGHT@@ m tall, @@VERTS@@ verts, @@TRIS@@ tris, @@BONES@@ bones,
@@MATS@@ materials, @@NCLIPS@@ animation clips, @@TEXSIZE@@ texture atlas.
All data is baked into this file - no addon, no external assets.
"""
import base64
import json
import math
import os
import struct
import sys
import tempfile
import time
import zlib

try:
    import bpy
    from mathutils import Vector, Quaternion
except ImportError:
    print("Run this script inside Blender: Scripting > Open > Run Script.")
    sys.exit(1)

import numpy as np

# ---------------------------------------------------------------------------
# baked data
# ---------------------------------------------------------------------------
SPEC = json.loads("""@@SPEC_JSON@@""")
MATERIALS = json.loads("""@@MATERIALS_JSON@@""")
MESH_B64 = """@@MESH_B64@@"""
RIG_B64 = """@@RIG_B64@@"""
ANIM_B64 = """@@ANIM_B64@@"""
TEX_B64 = """@@TEX_B64@@"""

BLEND_FPS = @@FPS@@
MAKE_IK = @@MAKE_IK@@
MAKE_STUDIO = True
SHADE_SMOOTH = True


# ---------------------------------------------------------------------------
# unpacking
# ---------------------------------------------------------------------------
def unpack_mesh():
    raw = zlib.decompress(base64.b64decode(MESH_B64))
    off = 0
    (n_verts,) = struct.unpack_from("<I", raw, off)
    off += 4
    verts = np.frombuffer(raw, "<f4", n_verts * 3, off).reshape(-1, 3).copy(); off += n_verts * 12
    norms = np.frombuffer(raw, "<f4", n_verts * 3, off).reshape(-1, 3).copy(); off += n_verts * 12
    uvs = np.frombuffer(raw, "<f4", n_verts * 2, off).reshape(-1, 2).copy(); off += n_verts * 8
    (n_faces,) = struct.unpack_from("<I", raw, off); off += 4
    faces = np.frombuffer(raw, "<u4", n_faces * 3, off).reshape(-1, 3).copy(); off += n_faces * 12
    joints = np.frombuffer(raw, "<u2", n_verts * 4, off).reshape(-1, 4).copy(); off += n_verts * 8
    weights = np.frombuffer(raw, "<f4", n_verts * 4, off).reshape(-1, 4).copy(); off += n_verts * 16
    mat_index = np.frombuffer(raw, "u1", n_faces, off).astype(np.int32).copy()
    return verts, norms, uvs, faces, joints, weights, mat_index


def unpack_rig():
    return json.loads(zlib.decompress(base64.b64decode(RIG_B64)))


def unpack_anim():
    return json.loads(zlib.decompress(base64.b64decode(ANIM_B64)))


def texture_path():
    path = os.path.join(tempfile.gettempdir(), "colada_atlas_@@FINGERPRINT@@.png")
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(TEX_B64))
    return path


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.materials,
                 bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for item in list(coll):
            try:
                coll.remove(item)
            except Exception:
                pass


def make_materials(image):
    mats = []
    for name in MATERIALS:
        m = bpy.data.materials.new("colada_" + name)
        m.use_nodes = True
        nt = m.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = "Linear"
        tex.location = (-420, 220)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.35 if name == "glasses" else (
                0.55 if name in ("head", "skin") else 0.75)
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.4 if name == "glasses" else 0.0
        m.use_backface_culling = name in ("hair", "glasses", "hat", "extras", "extras_skin")
        mats.append(m)
    return mats


def build_mesh(verts, norms, uvs, faces, joints, weights, mat_index, bones, mats):
    me = bpy.data.meshes.new("Colada_Mesh")
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()

    uv_layer = me.uv_layers.new(name="UVMap")
    loops = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loops)
    uv_layer.data.foreach_set("uv", uvs[loops].astype(np.float32).reshape(-1))

    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", mat_index)
    if SHADE_SMOOTH:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))

    obj = bpy.data.objects.new("Colada_Mesh", me)
    bpy.context.collection.objects.link(obj)

    try:
        me.normals_split_custom_set([tuple(n) for n in norms[loops]])
    except Exception as exc:
        print("  (custom normals skipped:", exc, ")")

    groups = {}
    for b in bones:
        if b["deform"]:
            groups[b["name"]] = obj.vertex_groups.new(name=b["name"])
    index = {b["name"]: i for i, b in enumerate(bones)}
    for vi in range(len(verts)):
        for k in range(4):
            w = float(weights[vi][k])
            if w <= 1e-5:
                continue
            name = bones[int(joints[vi][k])]["name"]
            if name in groups:
                groups[name].add([vi], w, "REPLACE")
    return obj


def build_armature(bones):
    arm = bpy.data.armatures.new("Colada_Rig")
    obj = bpy.data.objects.new("Colada_Rig", arm)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    created = {}
    for b in bones:
        eb = arm.edit_bones.new(b["name"])
        eb.head = Vector(b["head"])
        eb.tail = Vector(b["tail"])
        eb.use_deform = bool(b["deform"])
        created[b["name"]] = eb
    for b in bones:
        if b["parent"]:
            created[b["name"]].parent = created[b["parent"]]
        created[b["name"]].align_roll(Vector(b["z"]))
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return obj


def add_ik(rig):
    """Muted IK constraints + targets so you can switch to IK control."""
    for bone_name in ("Hand.L", "Hand.R", "Foot.L", "Foot.R"):
        pb = rig.pose.bones.get(bone_name)
        if not pb:
            continue
        empty = bpy.data.objects.new("IK_" + bone_name, None)
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = 0.04
        empty.location = rig.matrix_world @ pb.head
        bpy.context.collection.objects.link(empty)
        con = pb.constraints.new("IK")
        con.target = empty
        con.chain_count = 2
        con.influence = 0.0      # enable this to hand control to the empty


def _fill_fcurve(act, path, index, times, values, group, fps):
    try:
        fc = act.fcurves.new(path, index=index, action_group=group)
    except TypeError:                       # Blender 4.4+ slotted actions
        fc = act.fcurves.new(path, index=index, action_group=group)
    n = len(times)
    fc.keyframe_points.add(n)
    co = np.empty(n * 2, dtype=np.float32)
    co[0::2] = (np.asarray(times, dtype=np.float32) * fps) + 1.0
    co[1::2] = np.asarray(values, dtype=np.float32)
    fc.keyframe_points.foreach_set("co", co)
    fc.update()


def apply_animations(rig, bones, clips):
    rig.animation_data_create()
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:
        pass
    made = []
    for clip in clips:
        act = bpy.data.actions.new("Colada_" + clip["name"].capitalize())
        times = clip["times"]
        for bi, b in enumerate(bones):
            name = b["name"]
            qpath = 'pose.bones["%s"].rotation_quaternion' % name
            for ch in range(4):
                _fill_fcurve(act, qpath, ch, times,
                             [clip["quats"][fi][bi][ch] for fi in range(len(times))], name, clip["fps"])
            lpath = 'pose.bones["%s"].location' % name
            moves = any(abs(clip["trans"][fi][bi][c]) > 1e-9
                        for fi in range(len(times)) for c in range(3))
            if moves or name in ("Root", "Hips"):
                for ch in range(3):
                    _fill_fcurve(act, lpath, ch, times,
                                 [clip["trans"][fi][bi][ch] for fi in range(len(times))], name, clip["fps"])
        act.use_fake_user = True
        made.append((clip, act))
    # push to NLA, keep Idle active for instant playback
    rig.animation_data.action = None
    active = None
    for clip, act in made:
        track = rig.animation_data.nla_tracks.new()
        track.name = clip["name"].capitalize()
        strip = track.strips.new(clip["name"].capitalize(), 1, act)
        strip.action_frame_start = 1
        strip.action_frame_end = max(2, 1 + int(round(clip["duration"] * clip["fps"])))
        track.mute = True
        if clip["name"] == "idle":
            active = act
            track.mute = False
    if active is None and made:
        active = made[0][1]
        rig.animation_data.nla_tracks[0].mute = False
    if active is not None:
        try:
            rig.animation_data.action = active
        except Exception:
            pass
    return made


def studio(rig, height):
    for name, loc, power, size in (("Key", (2.4, -2.6, 2.3), 900.0, 1.0),
                                   ("Fill", (-2.8, -1.7, 1.5), 260.0, 1.5),
                                   ("Rim", (0.7, 3.2, 2.5), 700.0, 0.9)):
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = power
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        con = lo.constraints.new("TRACK_TO")
        con.target = rig
        bpy.context.collection.objects.link(lo)
    cam_d = bpy.data.cameras.new("Camera")
    cam_d.lens = 62
    cam = bpy.data.objects.new("Camera", cam_d)
    cam.location = (2.0, -3.4, height * 0.60)
    con = cam.constraints.new("TRACK_TO")
    con.target = rig
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.05, 0.07, 0.10, 1.0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    clear_scene()
    bpy.context.scene.render.fps = BLEND_FPS

    verts, norms, uvs, faces, joints, weights, mat_index = unpack_mesh()
    rig_data = unpack_rig()
    bones = rig_data["bones"]
    clips = unpack_anim()

    image = bpy.data.images.load(texture_path())
    image.colorspace_settings.name = "sRGB"
    mats = make_materials(image)

    obj = build_mesh(verts, norms, uvs, faces, joints, weights, mat_index, bones, mats)
    rig = build_armature(bones)

    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    obj.parent = rig

    apply_animations(rig, bones, clips)
    if MAKE_IK:
        add_ik(rig)
    if MAKE_STUDIO:
        studio(rig, rig_data["height"])

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 120
    bpy.context.view_layer.update()
    print("Colada character built in %.2fs: %d verts, %d bones, %d actions"
          % (time.time() - t0, len(obj.data.vertices), len(bones), len(bpy.data.actions)))
    print("Tip: select Colada_Rig, open the Dope Sheet > Action Editor to scrub clips,")
    print("     or unmute a track in the NLA editor to blend animations.")


if __name__ == "__main__":
    main()
'''


def blender_script(spec: dict, want_clips: list[str] | None = None, fps: int = 30,
                   make_ik: bool | None = None, fingerprint: str = "") -> str:
    """Generate the standalone Blender script for a spec."""
    from .animations import CLIPS, DEFAULT_CLIP_ORDER
    from .builder import build_character

    names = [c for c in (want_clips or spec.get("clips") or DEFAULT_CLIP_ORDER) if c in CLIPS] or ["idle"]
    b = build_character(spec, want_clips=names, fps=fps)
    if make_ik is None:
        make_ik = bool(spec.get("ik_setup", True))
    fp = fingerprint or f"{(abs(hash(json.dumps(spec, sort_keys=True))) % (16 ** 8)):08x}"

    buf = io.BytesIO()
    b.texture.save(buf, format="PNG")
    subs = {
        "@@DATE@@": time.strftime("%Y-%m-%d %H:%M"),
        "@@FINGERPRINT@@": fp,
        "@@HEIGHT@@": f"{b.metrics.H:.2f}",
        "@@VERTS@@": str(b.stats["vertices"]),
        "@@TRIS@@": str(b.stats["triangles"]),
        "@@BONES@@": str(b.stats["bones"]),
        "@@MATS@@": str(b.stats["materials"]),
        "@@NCLIPS@@": str(len(names)),
        "@@TEXSIZE@@": str(b.spec["texture_size"]),
        "@@CLIPNAMES@@": ", ".join(c.capitalize() for c in names),
        "@@SPEC_JSON@@": json.dumps(spec, sort_keys=True),
        "@@MATERIALS_JSON@@": json.dumps(atlas.MATERIALS),
        "@@MESH_B64@@": _pack_mesh(b.mesh),
        "@@RIG_B64@@": _pack_rig(b.rig),
        "@@ANIM_B64@@": _pack_anim(b.clips),
        "@@TEX_B64@@": base64.b64encode(buf.getvalue()).decode(),
        "@@FPS@@": str(fps),
        "@@MAKE_IK@@": "True" if make_ik else "False",
    }
    out = TEMPLATE
    for k, v in subs.items():
        out = out.replace(k, v)
    return out
