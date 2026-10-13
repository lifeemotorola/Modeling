#!/usr/bin/env python3
"""
Colada Character Maker - web server
===================================
Zero dependency (standard library only) HTTP server that serves the studio UI
and runs the Python character engine.

    python3 web/server.py --port 8000

Routes
------
GET  /                       studio UI
GET  /static/...             js / css / vendored three.js
GET  /api/schema             schema + presets + clip list for the UI
POST /api/random             {seed?, base?} -> a random spec
POST /api/build              {spec, clips?, preview?} -> glb/texture/exports
GET  /api/library            saved characters
POST /api/library            save the current spec (+glb) to the library
DEL  /api/library/<id>       delete a saved character
GET  /build/<id>/<file>      generated assets (glb, texture, blender script, zip)
GET  /api/preview            ?spec=<b64>&size=&yaw= -> PNG (server render)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import spec as spec_mod                       # noqa: E402
from core.animations import CLIPS, DEFAULT_CLIP_ORDER   # noqa: E402
from core.builder import build_character                # noqa: E402
from core.blender_export import blender_script          # noqa: E402
from core import atlas                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "static")   # the front-end lives here
BUILDS = os.path.join(ROOT, ".arena", "builds")   # excluded from snapshots
LIBRARY = os.path.join(ROOT, "library")
os.makedirs(BUILDS, exist_ok=True)
os.makedirs(LIBRARY, exist_ok=True)

BUILD_LOCK = threading.Lock()
CACHE_LIMIT = 24
_cache: dict[str, dict] = {}
_order: list[str] = []


# ---------------------------------------------------------------------------
# build helpers
# ---------------------------------------------------------------------------

def spec_key(sp: dict, clips: list[str]) -> str:
    blob = json.dumps([sp, clips], sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def do_build(spec: dict, clips: list[str] | None = None, want_preview: bool = True,
             want_thumbs: bool = False) -> dict:
    """Build (or reuse) a character and return the manifest the UI needs."""
    sp = spec_mod.clamp_spec(spec)
    names = [c for c in (clips or sp.get("clips") or DEFAULT_CLIP_ORDER) if c in CLIPS]
    key = spec_key(sp, names)
    with BUILD_LOCK:
        if key in _cache:
            _order.remove(key)
            _order.append(key)
            return _cache[key]
        b = build_character(sp, want_clips=names)
        out_dir = os.path.join(BUILDS, key)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "model.glb"), "wb") as fh:
            fh.write(b.glb)
        b.texture.save(os.path.join(out_dir, "texture.png"))
        with open(os.path.join(out_dir, "spec.json"), "w") as fh:
            json.dump(sp, fh, indent=2)
        with open(os.path.join(out_dir, "build_character_in_blender.py"), "w") as fh:
            fh.write(blender_script(sp, want_clips=names))
        # animation thumbnails (cheap, server side, also handy as a contact sheet)
        thumbs = {}
        if want_preview:
            try:
                from core.preview import render
                render(b.mesh, b.texture, size=(340, 500), yaw=24).save(os.path.join(out_dir, "portrait.png"))
                if want_thumbs:
                    from core.animations import CLIPS as _C, Pose  # noqa
                    from core.preview import pose_verts
                    for name in names[:10]:
                        pose = Pose()
                        _C[name]["fn"](0.42 if not _C[name]["loop"] else 0.25, pose, _C[name]["duration"])
                        v = pose_verts(b.mesh, b.rig, pose)
                        im = render(b.mesh, b.texture, size=(200, 300), yaw=26, verts_override=v)
                        fn = f"thumb_{name}.png"
                        im.save(os.path.join(out_dir, fn))
                        thumbs[name] = f"/build/{key}/{fn}"
            except Exception:                                   # pragma: no cover
                traceback.print_exc()

        # one-click export bundle
        meta = {
            "id": key,
            "spec": sp,
            "stats": b.stats,
            "clips": [{"name": c["name"], "duration": round(c["duration"], 3), "loop": c["loop"],
                       "label": CLIPS[c["name"]]["label"]} for c in b.clips],
            "files": {
                "glb": f"/build/{key}/model.glb",
                "texture": f"/build/{key}/texture.png",
                "spec": f"/build/{key}/spec.json",
                "blender": f"/build/{key}/build_character_in_blender.py",
                "zip": f"/build/{key}/colada_{key}.zip",
                "portrait": f"/build/{key}/portrait.png",
            },
            "thumbs": thumbs,
            "seconds": round(b.seconds, 2),
        }
        zip_path = os.path.join(out_dir, f"colada_{key}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(out_dir, "model.glb"), "model.glb")
            z.write(os.path.join(out_dir, "texture.png"), "texture.png")
            z.write(os.path.join(out_dir, "spec.json"), "character_spec.json")
            z.write(os.path.join(out_dir, "build_character_in_blender.py"), "build_character_in_blender.py")
            z.writestr("README.txt", EXPORT_README)
        _cache[key] = meta
        _order.append(key)
        while len(_order) > CACHE_LIMIT:
            old = _order.pop(0)
            _cache.pop(old, None)
    return meta


EXPORT_README = """Colada Character Maker export
=============================

model.glb                 - skinned mesh + rig + every animation clip.
                            Drag into Blender (File > Import > glTF 2.0), or
                            load it in three.js / Godot / Unity / Unreal.
texture.png               - the baked 1x1 texture atlas (all materials).
character_spec.json       - every parameter used to generate this character.
build_character_in_blender.py
                          - regenerates the whole character natively inside
                            Blender (real armature, vertex groups, materials,
                            animations) instead of importing the glb.

Blender quick start
-------------------
1. File > Import > glTF 2.0 > model.glb
   (the character arrives rigged and animated; press Space to play)
2. To rebuild it natively instead:
   Scripting > Open > build_character_in_blender.py > Run Script
"""


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------

def library_list() -> list[dict]:
    out = []
    for name in sorted(os.listdir(LIBRARY), reverse=True):
        p = os.path.join(LIBRARY, name)
        meta = os.path.join(p, "character.json")
        if os.path.isdir(p) and os.path.exists(meta):
            try:
                with open(meta) as fh:
                    data = json.load(fh)
                data["files"] = {k: f"/library/{name}/{v}" for k, v in data.get("files", {}).items()}
                out.append(data)
            except Exception:
                continue
    return out


def library_save(spec: dict, meta: dict | None = None) -> dict:
    sp = spec_mod.clamp_spec(spec)
    name = time.strftime("char_%Y%m%d_%H%M%S")
    d = os.path.join(LIBRARY, name)
    os.makedirs(d, exist_ok=True)
    b = build_character(sp, want_clips=list(CLIPS.keys()))
    with open(os.path.join(d, "model.glb"), "wb") as fh:
        fh.write(b.glb)
    b.texture.save(os.path.join(d, "texture.png"))
    with open(os.path.join(d, "build_character_in_blender.py"), "w") as fh:
        fh.write(blender_script(sp))
    from core.preview import render
    render(b.mesh, b.texture, size=(300, 440), yaw=22).save(os.path.join(d, "portrait.png"))
    data = {
        "id": name,
        "label": (meta or {}).get("label") or name,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "spec": sp,
        "stats": b.stats,
        "files": {"glb": "model.glb", "texture": "texture.png", "portrait": "portrait.png",
                  "blender": "build_character_in_blender.py"},
    }
    with open(os.path.join(d, "character.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    return data


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".glb": "model/gltf-binary",
    ".zip": "application/zip",
    ".py": "text/x-python; charset=utf-8",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "ColadaMaker/1.0"

    def log_message(self, fmt, *args):        # quieter logs
        if os.environ.get("COLADA_VERBOSE"):
            super().log_message(fmt, *args)

    # -- plumbing -------------------------------------------------------
    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=60")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=_default).encode(), "application/json")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _file(self, path: str, ctype=None):
        if not os.path.isfile(path):
            return self._error(404, "not found")
        with open(path, "rb") as fh:
            data = fh.read()
        self._send(200, data, ctype or CTYPES.get(os.path.splitext(path)[1], "application/octet-stream"))

    # -- routes ---------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        q = parse_qs(u.query)
        try:
            if p in ("/", "/index.html"):
                return self._file(os.path.join(STATIC, "index.html"))
            if p.startswith("/static/"):
                rel = os.path.normpath(p[len("/static/"):]).lstrip("./")
                return self._file(os.path.join(STATIC, rel))
            if p.startswith("/build/"):
                return self._build_file(p)
            if p.startswith("/library/"):
                return self._library_file(p)
            if p == "/api/schema":
                return self._json({
                    "schema": spec_mod.SCHEMA,
                    "presets": {k: {"label": v["label"], "icon": v["icon"]} for k, v in spec_mod.PRESETS.items()},
                    "materials": atlas.MATERIALS,
                    "clips": [{"name": k, "label": CLIPS[k]["label"], "duration": CLIPS[k]["duration"],
                               "loop": CLIPS[k]["loop"]} for k in DEFAULT_CLIP_ORDER],
                    "defaults": spec_mod.defaults(),
                })
            if p == "/api/library":
                return self._json({"items": library_list()})
            if p == "/api/preview":
                return self._preview(q)
            return self._error(404, "not found")
        except Exception as exc:                              # pragma: no cover
            traceback.print_exc()
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        p = urlparse(self.path).path
        body = self._body()
        try:
            if p == "/api/random":
                seed = body.get("seed")
                sp = spec_mod.random_spec(seed=seed if seed is not None else None,
                                          base=body.get("base"))
                return self._json({"spec": sp})
            if p == "/api/preset":
                name = body.get("name", "colada")
                return self._json({"spec": spec_mod.preset_spec(name)})
            if p == "/api/build":
                spec = body.get("spec") or {}
                clips = body.get("clips")
                t0 = time.time()
                meta = do_build(spec, clips, want_preview=bool(body.get("preview", False)),
                                want_thumbs=bool(body.get("thumbs")))
                meta = dict(meta)
                meta["request_seconds"] = round(time.time() - t0, 2)
                return self._json(meta)
            if p == "/api/library":
                spec = body.get("spec") or {}
                return self._json(library_save(spec, body.get("meta")))
            if p == "/api/preview":
                spec = body.get("spec") or {}
                from core.builder import quick_preview
                import io as _io
                im = quick_preview(spec, size=(int(body.get("w", 320)), int(body.get("h", 470))),
                                   yaw=float(body.get("yaw", 25)))
                buf = _io.BytesIO()
                im.save(buf, format="PNG")
                return self._send(200, buf.getvalue(), "image/png")
            return self._error(404, "not found")
        except Exception as exc:                              # pragma: no cover
            traceback.print_exc()
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def do_DELETE(self):
        p = urlparse(self.path).path
        m = re.match(r"^/api/library/([\w\-\.]+)$", p)
        if not m:
            return self._error(404, "not found")
        d = os.path.join(LIBRARY, m.group(1))
        if os.path.isdir(d) and d.startswith(LIBRARY):
            import shutil
            shutil.rmtree(d)
            return self._json({"ok": True})
        return self._error(404, "not found")

    # -- helpers --------------------------------------------------------
    def _build_file(self, p: str):
        m = re.match(r"^/build/([0-9a-f]{6,32})/([\w\-.]+)$", p)
        if not m:
            return self._error(404, "not found")
        key, name = m.groups()
        path = os.path.join(BUILDS, key, name)
        if not os.path.abspath(path).startswith(BUILDS):
            return self._error(403, "no")
        if name == "model.glb":
            return self._file(path, "model/gltf-binary")
        return self._file(path)

    def _library_file(self, p: str):
        m = re.match(r"^/library/([\w\-\.]+)/([\w\-.]+)$", p)
        if not m:
            return self._error(404, "not found")
        name, fname = m.groups()
        path = os.path.join(LIBRARY, name, fname)
        if not os.path.abspath(path).startswith(LIBRARY):
            return self._error(403, "no")
        return self._file(path)

    def _preview(self, q):
        raw = (q.get("spec") or [""])[0]
        try:
            spec = json.loads(base64.b64decode(raw + "==").decode())
        except Exception:
            spec = {}
        from core.builder import quick_preview
        size = (int((q.get("w") or ["300"])[0]), int((q.get("h") or ["440"])[0]))
        im = quick_preview(spec, size=size, yaw=float((q.get("yaw") or ["25"])[0]))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        self._send(200, buf.getvalue(), "image/png")


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description="Colada Character Maker studio server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"\n  🍍 Colada Character Maker")
    print(f"     studio   http://localhost:{args.port}/")
    print(f"     engine   python3 cli.py --help")
    print(f"     ctrl-c to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
