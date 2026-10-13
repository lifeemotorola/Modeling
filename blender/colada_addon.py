# -*- coding: utf-8 -*-
"""
Colada Character Maker - Blender addon
======================================
A real Blender integration for characters built with the Colada maker.

Install
-------
Edit > Preferences > Add-ons > Install… pick this file, then enable
"Colada Character Maker".  The panel lives in the 3D viewport sidebar (N).

What it does
------------
* Import a Colada export (.zip bundle or .glb) and set everything up:
  NLA tracks per clip, IK targets for hands/feet, studio lighting + camera.
* Run a generated ``*_blender.py`` build script (native armature, vertex
  groups, per-part materials, keyed actions).
* Browser of every Colada clip with instant playback in the viewport.
* One click glTF re-export, action renaming, and rig tidying helpers.

bl_info version 1.2.0  -  works on Blender 3.3 LTS through 4.x
"""
bl_info = {
    "name": "Colada Character Maker",
    "author": "Colada Character Maker",
    "version": (1, 2, 0),
    "blender": (3, 3, 0),
    "location": "View3D > Sidebar (N) > Colada",
    "description": "Import, rig-check and animate Colada characters (glb / native build scripts)",
    "category": "Character",
}

import os
import zipfile
import tempfile

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

PREFIX = "Colada_"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def find_rig(context):
    """Prefer the selected armature, else any armature named like ours."""
    for ob in context.selected_objects:
        if ob.type == "ARMATURE":
            return ob
    for ob in context.scene.objects:
        if ob.type == "ARMATURE" and any(pb.name in ("Root", "Hips") for pb in ob.pose.bones):
            return ob
    return next((ob for ob in context.scene.objects if ob.type == "ARMATURE"), None)


def colada_actions():
    return [a for a in bpy.data.actions if a.name.startswith(PREFIX) or a.name.lower() in
            ("idle", "walk", "run", "jump", "wave", "dance", "kick", "cheer", "zombie", "tpose")]


def clip_items(self, context):
    items = []
    for act in colada_actions():
        items.append((act.name, act.name.replace(PREFIX, ""), ""))
    if not items:
        items = [("NONE", "no clips", "import a Colada character first")]
    return items


def push_actions_to_nla(rig, active=None):
    """Give every Colada action its own muted NLA track and activate one."""
    rig.animation_data_create()
    existing = {t.name for t in rig.animation_data.nla_tracks}
    made = 0
    for act in colada_actions():
        if act.name in existing:
            continue
        track = rig.animation_data.nla_tracks.new()
        track.name = act.name
        start = int(act.frame_range[0]) if act.frame_range else 1
        strip = track.strips.new(act.name, start, act)
        strip.name = act.name
        track.mute = True
        made += 1
    target = active or ("Colada_Idle" if "Colada_Idle" in bpy.data.actions else
                        (colada_actions()[0].name if colada_actions() else None))
    if target:
        try:
            rig.animation_data.action = bpy.data.actions[target]
            act = bpy.data.actions[target]
            sc = bpy.context.scene
            sc.frame_start = int(act.frame_range[0])
            sc.frame_end = int(act.frame_range[1])
            if act.frame_range[1] - act.frame_range[0] > 2:
                for t in rig.animation_data.nla_tracks:
                    if t.name == target:
                        t.mute = True
        except Exception:
            pass
    return made


def add_ik_targets(rig):
    made = []
    for bone_name in ("Hand.L", "Hand.R", "Foot.L", "Foot.R"):
        pb = rig.pose.bones.get(bone_name)
        if not pb or any(c.type == "IK" for c in pb.constraints):
            continue
        empty = bpy.data.objects.new("IK_" + bone_name, None)
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = 0.045
        empty.location = rig.matrix_world @ pb.head
        bpy.context.collection.objects.link(empty)
        con = pb.constraints.new("IK")
        con.target = empty
        con.chain_count = 2
        con.influence = 0.0
        made.append(empty)
    return made


def add_studio(rig, height=1.7):
    for name, loc, power, size in (("Colada_Key", (2.4, -2.6, 2.3), 900.0, 1.0),
                                   ("Colada_Fill", (-2.8, -1.7, 1.5), 260.0, 1.5),
                                   ("Colada_Rim", (0.7, 3.2, 2.5), 700.0, 0.9)):
        if name in bpy.data.objects:
            continue
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = power
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        con = lo.constraints.new("TRACK_TO")
        con.target = rig
        bpy.context.collection.objects.link(lo)
    if "Colada_Camera" not in bpy.data.objects:
        cam_d = bpy.data.cameras.new("Colada_Camera")
        cam_d.lens = 62
        cam = bpy.data.objects.new("Colada_Camera", cam_d)
        cam.location = (2.0, -3.4, height * 0.60)
        con = cam.constraints.new("TRACK_TO")
        con.target = rig
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [ob for ob in bpy.data.objects if ob not in before]


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class COLADA_OT_import_bundle(Operator):
    """Import a Colada .zip bundle or .glb and wire up clips, IK and lights"""
    bl_idname = "colada.import_bundle"
    bl_label = "Import Colada character"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    use_ik: BoolProperty(name="Add IK targets", default=True)
    use_studio: BoolProperty(name="Add studio lights + camera", default=True)
    use_nla: BoolProperty(name="One NLA track per clip", default=True)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        path = self.filepath
        if not path or not os.path.exists(path):
            self.report({"ERROR"}, "pick a .zip or .glb file")
            return {"CANCELLED"}
        workdir = tempfile.mkdtemp(prefix="colada_")
        spec_path = None
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                z.extractall(workdir)
            glbs = [os.path.join(r, f) for r, _, fs in os.walk(workdir) for f in fs if f.endswith(".glb")]
            specs = [os.path.join(r, f) for r, _, fs in os.walk(workdir) for f in fs
                     if f.endswith(".json") and "spec" in f]
            if not glbs:
                self.report({"ERROR"}, "no .glb inside the bundle")
                return {"CANCELLED"}
            path = glbs[0]
            spec_path = specs[0] if specs else None
        objs = import_glb(path)
        rig = find_rig(context) or next((o for o in objs if o.type == "ARMATURE"), None)
        if rig is None:
            self.report({"WARNING"}, "imported the mesh but found no armature")

        if rig is not None:
            if self.use_nla:
                made = push_actions_to_nla(rig)
                self.report({"INFO"}, f"{made} clips set up as NLA tracks")
            if self.use_ik:
                add_ik_targets(rig)
            if self.use_studio:
                z_max = max((ob.dimensions.z for ob in objs), default=1.7)
                add_studio(rig, z_max or 1.7)
            bpy.context.view_layer.objects.active = rig
            rig.select_set(True)
        if spec_path:
            try:
                import json
                with open(spec_path) as fh:
                    context.scene["colada_spec"] = json.dumps(json.load(fh))
            except Exception:
                pass
        self.report({"INFO"}, "Colada character ready")
        return {"FINISHED"}


class COLADA_OT_run_script(Operator):
    """Run a generated build_character_in_blender.py script"""
    bl_idname = "colada.run_script"
    bl_label = "Run Colada build script"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({"ERROR"}, "pick a *_blender.py file")
            return {"CANCELLED"}
        code = open(self.filepath, encoding="utf-8").read()
        ns = {"__name__": "__colada__", "__file__": self.filepath}
        try:
            exec(compile(code, self.filepath, "exec"), ns)
            build = ns.get("main")
            if build:
                build()
        except Exception as exc:                                   # pragma: no cover
            self.report({"ERROR"}, f"build failed: {exc}")
            return {"CANCELLED"}
        if not context.scene.get("colada_spec"):
            import re
            m = re.search(r'SPEC = json\.loads\("""(.*?)"""\)', code, re.S)
            if m:
                context.scene["colada_spec"] = m.group(1)
        self.report({"INFO"}, "character built")
        return {"FINISHED"}


class COLADA_OT_play_clip(Operator):
    """Play the chosen clip on the active rig and set the frame range"""
    bl_idname = "colada.play_clip"
    bl_label = "Play clip"

    def execute(self, context):
        rig = find_rig(context)
        act = bpy.data.actions.get(context.scene.colada.clip)
        if rig is None or act is None:
            self.report({"WARNING"}, "no rig or clip selected")
            return {"CANCELLED"}
        rig.animation_data_create()
        rig.animation_data.action = act
        rng = act.frame_range
        context.scene.frame_start = int(rng[0])
        context.scene.frame_end = int(rng[1])
        context.scene.frame_set(int(rng[0]))
        bpy.ops.screen.animation_play()
        return {"FINISHED"}


class COLADA_OT_setup_rig(Operator):
    """Add IK targets and/or the studio light rig to the selected armature"""
    bl_idname = "colada.setup_rig"
    bl_label = "Setup rig helpers"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        rig = find_rig(context)
        if rig is None:
            self.report({"WARNING"}, "select an armature first")
            return {"CANCELLED"}
        props = context.scene.colada
        if props.add_ik:
            add_ik_targets(rig)
        if props.add_studio:
            add_studio(rig, rig.dimensions.z or 1.7)
        if props.make_nla:
            push_actions_to_nla(rig)
        self.report({"INFO"}, "rig helpers added")
        return {"FINISHED"}


class COLADA_OT_export_glb(Operator):
    """Export the Colada character (mesh + rig + actions) back to .glb"""
    bl_idname = "colada.export_glb"
    bl_label = "Export Colada .glb"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        self.filepath = "colada_character.glb"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        rig = find_rig(context)
        bpy.ops.object.select_all(action="DESELECT")
        for ob in context.scene.objects:
            if ob.type in ("MESH", "ARMATURE") and (rig is None or ob.parent == rig or ob == rig):
                ob.select_set(True)
        bpy.ops.export_scene.gltf(filepath=self.filepath, export_format="GLB",
                                  use_selection=True, export_animations=True,
                                  export_apply=False)
        self.report({"INFO"}, f"exported {os.path.basename(self.filepath)}")
        return {"FINISHED"}


class COLADA_OT_rename_actions(Operator):
    """Make sure every action carries the Colada_ prefix (glTF round trips)"""
    bl_idname = "colada.rename_actions"
    bl_label = "Normalise action names"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = 0
        for act in bpy.data.actions:
            if not act.name.startswith(PREFIX) and act.name.lower() in (
                    "idle", "walk", "run", "jump", "wave", "dance", "kick", "cheer", "zombie", "tpose"):
                act.name = PREFIX + act.name.capitalize()
                n += 1
        self.report({"INFO"}, f"{n} actions renamed")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------

class ColadaProps(PropertyGroup):
    clip: EnumProperty(name="Clip", items=clip_items)
    add_ik: BoolProperty(name="IK targets", default=True)
    add_studio: BoolProperty(name="Studio lights", default=True)
    make_nla: BoolProperty(name="NLA per clip", default=True)


class COLADA_PT_panel(Panel):
    bl_label = "Colada Character Maker"
    bl_idname = "COLADA_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Colada"

    def draw(self, context):
        layout = self.layout
        props = context.scene.colada
        col = layout.column(align=True)
        col.operator("colada.import_bundle", icon="IMPORT")
        col.operator("colada.run_script", icon="SCRIPT")

        box = layout.box()
        box.label(text="Playback", icon="PLAY")
        box.prop(props, "clip", text="")
        box.operator("colada.play_clip", icon="PLAY")

        box = layout.box()
        box.label(text="Rig helpers", icon="ARMATURE_DATA")
        box.prop(props, "add_ik")
        box.prop(props, "add_studio")
        box.prop(props, "make_nla")
        box.operator("colada.setup_rig", icon="CONSTRAINT")

        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        box.operator("colada.export_glb", icon="MESH_DATA")
        box.operator("colada.rename_actions", icon="SORTALPHA")

        rig = find_rig(context)
        if rig:
            box = layout.box()
            box.label(text=f"Rig: {rig.name}", icon="OUTLINER_OB_ARMATURE")
            box.label(text=f"{len(rig.pose.bones)} bones, {len(colada_actions())} clips")


classes = (
    ColadaProps,
    COLADA_OT_import_bundle,
    COLADA_OT_run_script,
    COLADA_OT_play_clip,
    COLADA_OT_setup_rig,
    COLADA_OT_export_glb,
    COLADA_OT_rename_actions,
    COLADA_PT_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.colada = PointerProperty(type=ColadaProps)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    if hasattr(bpy.types.Scene, "colada"):
        del bpy.types.Scene.colada


if __name__ == "__main__":
    register()
