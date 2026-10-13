"""
Colada Character Maker - build pipeline
=======================================
spec -> metrics -> rig -> mesh -> texture -> glb (+ previews)
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import spec as spec_mod
from .animations import CLIPS, DEFAULT_CLIP_ORDER, Sampler
from .gltf import export_glb
from .mesh import build_mesh, face_layout
from .skeleton import Rig, build_metrics
from .textures import paint


@dataclass
class Build:
    spec: dict
    metrics: object
    rig: Rig
    mesh: object
    texture: Image.Image
    glb: bytes
    clips: list
    stats: dict
    seconds: float


def build_character(raw_spec: dict, want_clips: list[str] | None = None, fps: int = 30) -> Build:
    t0 = time.time()
    sp = spec_mod.clamp_spec(raw_spec)
    m = build_metrics(sp)
    rig = Rig(sp, m)
    mesh = build_mesh(sp, m, rig)
    layout = face_layout(m, sp)
    size = int(sp["texture_size"])
    tex = paint(sp, m, layout, size)
    buf = io.BytesIO()
    tex.save(buf, format="PNG", optimize=False)
    png = buf.getvalue()

    names = [c for c in (want_clips or sp.get("clips") or DEFAULT_CLIP_ORDER) if c in CLIPS]
    if not names:
        names = ["idle"]
    sampler = Sampler(rig)
    clips = [sampler.sample(n, fps=fps) for n in names]

    glb = export_glb(sp, m, rig, mesh, png, clips)
    stats = {
        "vertices": int(mesh.verts.shape[0]),
        "triangles": int(mesh.faces.shape[0]),
        "bones": len(rig.bones),
        "deform_bones": len(rig.deform_bones),
        "materials": len(set(mesh.index_ranges.keys())),
        "texture": f"{size}x{size}",
        "glb_kb": round(len(glb) / 1024.0, 1),
        "clips": names,
        "tri_count": int(mesh.faces.shape[0]),
    }
    return Build(sp, m, rig, mesh, tex, glb, clips, stats, time.time() - t0)


def quick_preview(spec: dict, size=(340, 500), yaw=25) -> Image.Image:
    """Single-image preview without exporting (used by the API's thumbnails)."""
    from .preview import render
    sp = spec_mod.clamp_spec(spec)
    m = build_metrics(sp)
    rig = Rig(sp, m)
    mesh = build_mesh(sp, m, rig)
    tex = paint(sp, m, face_layout(m, sp), int(sp["texture_size"]))
    return render(mesh, tex, size=size, yaw=yaw)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import os

    from . import spec as S

    ap = argparse.ArgumentParser(description="Build a Colada character")
    ap.add_argument("--preset", default="colada")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="./build")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    spc = S.random_spec(args.seed) if args.random else S.preset_spec(args.preset)
    b = build_character(spc)
    os.makedirs(args.out, exist_ok=True)
    name = f"colada_{args.preset}{'_' + str(args.seed) if args.seed is not None else ''}"
    glb_path = os.path.join(args.out, name + ".glb")
    with open(glb_path, "wb") as fh:
        fh.write(b.glb)
    b.texture.save(os.path.join(args.out, name + "_texture.png"))
    print(f"{glb_path}  ({b.stats['glb_kb']} kB, {b.stats['vertices']} verts, "
          f"{b.stats['bones']} bones, {len(b.clips)} clips, {b.seconds:.2f}s)")
    if not args.no_preview:
        from .preview import contact_sheet
        sheet = contact_sheet(b.mesh, b.texture, size=(300, 440))
        sheet.save(os.path.join(args.out, name + "_preview.png"))
        print(f"{args.out}/{name}_preview.png")
