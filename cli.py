#!/usr/bin/env python3
"""
Colada Character Maker - command line studio
===========================================
Build characters without the web UI.

    python3 cli.py --list-presets
    python3 cli.py --preset colada --out build
    python3 cli.py --random --seed 42 --preview
    python3 cli.py --spec my_char.json --blender --zip
    python3 cli.py --all-presets --gallery
    python3 cli.py --validate build/model.glb

Outputs (all optional): <name>.glb, <name>_texture.png, <name>_spec.json,
<name>_blender.py, <name>_preview.png, <name>_bundle.zip
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import spec as spec_mod                       # noqa: E402
from core.animations import DEFAULT_CLIP_ORDER          # noqa: E402
from core.builder import build_character                # noqa: E402
from core.blender_export import blender_script          # noqa: E402


def banner():
    print(r"""
   ___      _           _        ___  _           _         _
  / __|___ | |__ _  __| |__ _   / __|| |_  ___  _| |_ ___ _| |
 | (__/ _ \| / _` |/ _` / _` | | (__ | ' \/ _ \/ _` / -_) _` |
  \___\___/|_\__,_|\__,_\__,_|  \___||_||_\___/\__,_\___\__,_|   character maker
""")


def parse_args():
    ap = argparse.ArgumentParser(description="Colada Character Maker (CLI)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="colada", help="preset name (see --list-presets)")
    ap.add_argument("--random", action="store_true", help="randomise the whole character")
    ap.add_argument("--seed", type=int, default=None, help="seed for --random (reproducible)")
    ap.add_argument("--spec", default=None, help="JSON spec file (or '-' for stdin)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a spec field, e.g. --set hair_style=spiky --set height=1.9")
    ap.add_argument("--clips", default=None, help="comma separated clip list (default: all)")
    ap.add_argument("--out", default="build", help="output directory")
    ap.add_argument("--name", default=None, help="output base name")
    ap.add_argument("--texture-size", type=int, choices=[512, 1024, 2048], default=None)
    ap.add_argument("--fps", type=int, default=30, help="animation sample rate")
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--preview", action="store_true", help="render a contact sheet (CPU)")
    ap.add_argument("--blender", action="store_true", help="write a native Blender build script")
    ap.add_argument("--zip", action="store_true", help="write a one click export bundle")
    ap.add_argument("--validate", default=None, metavar="FILE.glb", help="run the glb validator")
    ap.add_argument("--verify", default=None, metavar="FILE.glb", help="run the skinning verifier")
    ap.add_argument("--library", action="store_true", help="also save into ./library")
    ap.add_argument("--all-presets", action="store_true", help="build every preset")
    ap.add_argument("--gallery", action="store_true", help="contact sheet of all presets")
    ap.add_argument("--list-presets", action="store_true")
    return ap.parse_args()


def jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def build_one(spec: dict, args, name: str, quiet: bool = False) -> dict:
    clips = [c.strip() for c in args.clips.split(",")] if args.clips else None
    t0 = time.time()
    b = build_character(spec, want_clips=clips, fps=args.fps)
    os.makedirs(args.out, exist_ok=True)
    made = []
    if not args.no_glb:
        p = os.path.join(args.out, f"{name}.glb")
        with open(p, "wb") as fh:
            fh.write(b.glb)
        made.append(p)
    if not args.no_texture:
        p = os.path.join(args.out, f"{name}_texture.png")
        b.texture.save(p)
        made.append(p)
    p = os.path.join(args.out, f"{name}_spec.json")
    with open(p, "w") as fh:
        json.dump(jsonable(b.spec), fh, indent=2)
    made.append(p)
    if args.blender:
        p = os.path.join(args.out, f"{name}_blender.py")
        with open(p, "w") as fh:
            fh.write(blender_script(b.spec, want_clips=clips, fps=args.fps))
        made.append(p)
    if args.preview:
        from core.preview import contact_sheet
        p = os.path.join(args.out, f"{name}_preview.png")
        contact_sheet(b.mesh, b.texture, size=(320, 470)).save(p)
        made.append(p)
    if args.zip:
        p = os.path.join(args.out, f"{name}_bundle.zip")
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{name}.glb", b.glb)
            buf = io.BytesIO()
            b.texture.save(buf, format="PNG")
            z.writestr(f"{name}_texture.png", buf.getvalue())
            z.writestr("character_spec.json", json.dumps(jsonable(b.spec), indent=2))
            z.writestr("build_character_in_blender.py",
                       blender_script(b.spec, want_clips=clips, fps=args.fps))
        made.append(p)
    if not quiet:
        print(f"  {name}: {b.stats['vertices']} verts, {b.stats['triangles']} tris, "
              f"{b.stats['bones']} bones, {len(b.clips)} clips, {b.stats['glb_kb']} kB glb "
              f"({time.time() - t0:.2f}s)")
        for m in made:
            print("    " + m)
    return {"stats": b.stats, "files": made, "build": b}


def load_spec(args) -> dict:
    if args.spec:
        raw = sys.stdin.read() if args.spec == "-" else open(args.spec).read()
        spec = json.loads(raw)
    elif args.random:
        spec = spec_mod.random_spec(seed=args.seed)
    else:
        spec = spec_mod.preset_spec(args.preset)
    for item in args.set:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        meta = spec_mod.ALL_FIELDS.get(k.strip())
        if not meta:
            print(f"  ! unknown field '{k.strip()}' (skipped)")
            continue
        try:
            if meta["type"] == "bool":
                v = v.strip().lower() in ("1", "true", "yes", "on")
            elif meta["type"] in ("range",):
                v = float(v) if "." in v else int(v)
            elif meta["type"] == "checkboxes":
                v = [x.strip() for x in v.split(",")]
            else:
                v = v.strip()
        except ValueError:
            pass
        spec[k.strip()] = v
    if args.texture_size:
        spec["texture_size"] = str(args.texture_size)
    return spec_mod.clamp_spec(spec)


def main():
    args = parse_args()

    if args.list_presets:
        print("\nPresets:")
        for key, p in spec_mod.PRESETS.items():
            print(f"  {key:10s} {p['icon']}  {p['label']}")
        print("\nClips:", ", ".join(DEFAULT_CLIP_ORDER))
        print("Fields:", ", ".join(sorted(spec_mod.ALL_FIELDS))[:400], "…")
        return 0

    if args.validate:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
        from tools.validate_glb import check
        ok, log = check(args.validate)
        print(f"=== {args.validate} ===")
        for line in log:
            print("  " + line)
        print("  " + ("PASS ✅" if ok else "FAILED ❌"))
        return 0 if ok else 1

    if args.verify:
        os.system(f"{sys.executable} tools/verify_skin.py {args.verify}")
        return 0

    banner()

    if args.all_presets:
        total = time.time()
        print(f"building {len(spec_mod.PRESETS)} presets into {args.out}/")
        shots = []
        for key in spec_mod.PRESETS:
            spec = spec_mod.preset_spec(key)
            if args.texture_size:
                spec["texture_size"] = str(args.texture_size)
            res = build_one(spec, args, key)
            if args.gallery:
                shots.append((key, res["build"]))
        if args.gallery and shots:
            from PIL import Image
            from core.preview import render
            ims = []
            for key, b in shots:
                im = render(b.mesh, b.texture, size=(300, 440), yaw=24)
                ims.append(im)
            cols = 5
            rows = (len(ims) + cols - 1) // cols
            W, H = ims[0].size
            sheet = Image.new("RGB", (W * cols, H * rows), (14, 17, 24))
            for i, im in enumerate(ims):
                sheet.paste(im, ((i % cols) * W, (i // cols) * H))
            p = os.path.join(args.out, "gallery.png")
            sheet.save(p)
            print(f"  gallery: {p}")
        print(f"done in {time.time() - total:.1f}s")
        return 0

    spec = load_spec(args)
    name = args.name or (f"colada_{args.preset}" if not args.random else
                         f"colada_random{'_' + str(args.seed) if args.seed is not None else ''}")
    res = build_one(spec, args, name)

    if args.library:
        lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")
        d = os.path.join(lib, time.strftime("char_%Y%m%d_%H%M%S"))
        os.makedirs(d, exist_ok=True)
        b = res["build"]
        with open(os.path.join(d, "model.glb"), "wb") as fh:
            fh.write(b.glb)
        b.texture.save(os.path.join(d, "texture.png"))
        with open(os.path.join(d, "character.json"), "w") as fh:
            json.dump({"id": os.path.basename(d), "label": name, "spec": jsonable(b.spec),
                       "stats": jsonable(b.stats), "created": time.strftime("%Y-%m-%d %H:%M"),
                       "files": {"glb": "model.glb", "texture": "texture.png"}}, fh, indent=2)
        print(f"    saved to library/{os.path.basename(d)}")

    print("\nBlender: Scripting > Open > *_blender.py > Run Script, "
          "or File > Import > glTF 2.0 > *.glb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
